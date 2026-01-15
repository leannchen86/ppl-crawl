"""
Permutation Test: Shuffle labels randomly and retrain to detect optimization artifacts.

Why:
  - If correlations (e.g., weight-norm vs pred-freq) persist under random labels,
    they are artifacts of the optimization process, not real signal
  - If performance drops to random baseline → confirms model is learning real patterns

Design:
  - Use same 30-name benchmark setup
  - Randomly permute all labels (destroy true face-name associations)
  - Train linear probe with same hyperparameters
  - Compare: accuracy, prediction skew, weight-norm correlations

Outputs:
  - results/permutation_test/
      - summary.json
      - predictions.npy
      - true_labels.npy (permuted)
      - precision_recall_metrics.csv
      - weight_norms.json

Usage:
  python permutation_test.py --tag permuted_labels
"""

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_clip import create_model_and_transforms

from index_utils import ImageSource, resolve_good_images


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedMultiNameDataset(Dataset):
    """Dataset with optional label permutation."""

    def __init__(
        self,
        index_dir: str,
        names: List[str],
        transform,
        split: str,
        train_ratio: float,
        seed: int,
        max_per_name: int,
        per_name: int = 0,
        strict_per_name: bool = False,
        image_source: ImageSource = "chips",
        permute_labels: bool = False,
        permutation_seed: int = 9999,
    ):
        assert split in ("train", "val")
        self.transform = transform
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.samples: List[Tuple[str, int]] = []
        self.permute_labels = permute_labels

        rng = random.Random(seed)
        per_name_candidates: Dict[str, List[str]] = {}
        for name in names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                continue
            with open(index_path) as f:
                data = json.load(f)
            good_images = list(resolve_good_images(data, image_source=image_source))
            rng.shuffle(good_images)
            split_idx = int(len(good_images) * train_ratio)
            selected = good_images[:split_idx] if split == "train" else good_images[split_idx:]

            if per_name and per_name > 0:
                per_name_candidates[name] = selected
                continue

            if max_per_name and len(selected) > max_per_name:
                selected = selected[:max_per_name]
            for p in selected:
                if os.path.exists(p):
                    self.samples.append((p, self.name_to_idx[name]))

        if per_name and per_name > 0:
            filtered_candidates: Dict[str, List[str]] = {}
            available_counts: Dict[str, int] = {}
            for name, cand in per_name_candidates.items():
                exist = [p for p in cand if os.path.exists(p)]
                filtered_candidates[name] = exist
                available_counts[name] = len(exist)

            min_avail = min(available_counts.values()) if available_counts else 0
            effective = min(per_name, min_avail) if min_avail > 0 else 0

            if effective <= 0:
                msg = f"[{split}] No samples available after filtering existing paths."
                if strict_per_name:
                    raise ValueError(msg)
                print("WARNING:", msg)
            elif effective < per_name:
                msg = (
                    f"[{split}] Requested per_name={per_name} but at least one class only has "
                    f"{min_avail} existing samples after split+filter; using per_name={effective}."
                )
                if strict_per_name:
                    raise ValueError(msg)
                print("WARNING:", msg)

            for name in names:
                exist = filtered_candidates.get(name, [])
                take = exist[:effective]
                for p in take:
                    self.samples.append((p, self.name_to_idx[name]))

        rng.shuffle(self.samples)

        # Permute labels if requested
        if permute_labels:
            perm_rng = random.Random(permutation_seed)
            labels = [label for _, label in self.samples]
            perm_rng.shuffle(labels)
            self.samples = [(path, labels[i]) for i, (path, _) in enumerate(self.samples)]
            print(f"  [{split}] Labels permuted with seed {permutation_seed}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, y = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y


@torch.no_grad()
def extract_embeddings(model, loader, device) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    embs = []
    ys = []
    use_cuda = (device.type == "cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else torch.autocast(device_type="cpu", enabled=False)
    with ctx:
        for images, labels in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device, non_blocking=True)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu())
            ys.append(labels.cpu())
    return torch.cat(embs, dim=0), torch.cat(ys, dim=0)


def train_linear_head(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    head = nn.Linear(train_X.shape[1], num_classes).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    loss_fn = nn.CrossEntropyLoss()

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    best = {"acc": -1.0, "epoch": 0, "state": None}
    for ep in range(1, epochs + 1):
        head.train()
        opt.zero_grad()
        logits = head(train_X)
        loss = loss_fn(logits, train_y)
        loss.backward()
        opt.step()

        head.eval()
        with torch.no_grad():
            val_logits = head(val_X)
            val_preds = val_logits.argmax(dim=1)
            acc = (val_preds == val_y).float().mean().item()
        if acc > best["acc"]:
            best = {"acc": acc, "epoch": ep, "state": {k: v.detach().cpu() for k, v in head.state_dict().items()}}
        if ep in (1, 10, 20, 30, 40, 50) or ep == epochs:
            print(f"  epoch {ep:3d}: val_acc={acc:.4f}")

    # Restore best and extract diagnostics
    head.load_state_dict(best["state"])
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        val_logits = head(val_X)
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_preds = val_logits.argmax(dim=1).cpu().numpy()

    # Weight norms and biases
    weight_norms = torch.norm(head.weight, p=2, dim=1).detach().cpu().numpy()
    biases = head.bias.detach().cpu().numpy()

    return best["acc"], best["epoch"], val_preds, val_probs, weight_norms, biases


def per_class_prf(preds: np.ndarray, true: np.ndarray, names: List[str]):
    rows = []
    for c, name in enumerate(names):
        tp = int(((preds == c) & (true == c)).sum())
        fp = int(((preds == c) & (true != c)).sum())
        fn = int(((preds != c) & (true == c)).sum())
        support = int((true == c).sum())
        predicted_count = int((preds == c).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        rows.append(
            {
                "name": name,
                "recall": recall,
                "precision": precision,
                "f1_score": f1,
                "support": support,
                "predicted_count": predicted_count,
                "pred_bias": (predicted_count / support) if support else 0.0,
            }
        )
    rows.sort(key=lambda r: r["f1_score"], reverse=True)
    return rows


@dataclass
class Summary:
    tag: str
    test_type: str
    num_classes: int
    val_size: int
    overall_accuracy: float
    random_baseline: float
    pred_count_cv: float
    weight_norm_correlation: float
    bias_correlation: float
    best_epoch: int
    permutation_seed: int


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index-dir",
        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001",
    )
    p.add_argument("--names-json", default="/home/leann/face-detection/results/scale_up_results/names.json")
    p.add_argument("--tag", default="permuted_labels")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--permutation-seed", type=int, default=9999)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-per-name", type=int, default=500)
    p.add_argument(
        "--per-name",
        type=int,
        default=0,
        help="If >0, enforce a uniform per-class sample count per split (after train/val split). "
             "If some class has fewer images, the per-class count is reduced to the minimum across classes "
             "(unless --strict-per-name).",
    )
    p.add_argument(
        "--strict-per-name",
        action="store_true",
        help="If set with --per-name, error out instead of reducing per-class counts when data is insufficient.",
    )
    p.add_argument(
        "--image-source",
        choices=["chips", "original"],
        default="chips",
        help="Choose which images to use from the index files: "
             "'chips' uses index['good']; 'original' uses index['meta'][chip].src_path.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with open(args.names_json) as f:
        names = json.load(f)
    names = [n.strip().lower() for n in names]

    out_dir = Path("/home/leann/face-detection/results/permutation_test") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("PERMUTATION TEST: SHUFFLED LABELS")
    print("=" * 70)
    print(f"Tag: {args.tag}")
    print(f"Classes: {len(names)}")
    print(f"Random baseline: {1/len(names):.4f}")
    print(f"Permutation seed: {args.permutation_seed}")
    print(f"Output dir: {out_dir}")
    print()
    print("Purpose: If accuracy stays high or correlations persist → artifact")
    print("         If accuracy drops to random → model was learning real signal")
    print()

    index_dir = Path(args.index_dir).expanduser()
    if not index_dir.is_dir():
        raise SystemExit(f"--index-dir is not a directory: {index_dir}")

    # Load CLIP
    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Datasets with PERMUTED labels
    train_ds = BalancedMultiNameDataset(
        index_dir=str(index_dir),
        names=names,
        transform=preprocess,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_per_name=args.max_per_name,
        per_name=args.per_name,
        strict_per_name=args.strict_per_name,
        image_source=args.image_source,  # type: ignore[arg-type]
        permute_labels=True,
        permutation_seed=args.permutation_seed,
    )
    val_ds = BalancedMultiNameDataset(
        index_dir=str(index_dir),
        names=names,
        transform=preprocess,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_per_name=args.max_per_name,
        per_name=args.per_name,
        strict_per_name=args.strict_per_name,
        image_source=args.image_source,  # type: ignore[arg-type]
        permute_labels=True,
        permutation_seed=args.permutation_seed,
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Save the exact split membership (with PERMUTED labels) for audit.
    def _write_split_jsonl(ds: BalancedMultiNameDataset, out_path: Path) -> None:
        with open(out_path, "w", encoding="utf-8") as f:
            for img_path, y in ds.samples:
                f.write(json.dumps({"path": img_path, "y": int(y), "name": ds.names[int(y)]}) + "\n")

    _write_split_jsonl(train_ds, out_dir / "train_split.jsonl")
    _write_split_jsonl(val_ds, out_dir / "val_split.jsonl")
    (out_dir / "split_config.json").write_text(
        json.dumps(
            {
                "seed": int(args.seed),
                "train_ratio": float(args.train_ratio),
                "max_per_name": int(args.max_per_name),
                "per_name": int(args.per_name),
                "strict_per_name": bool(args.strict_per_name),
                "index_dir": str(index_dir),
                "image_source": args.image_source,
                "num_classes": int(len(names)),
                "permutation_seed": int(args.permutation_seed),
            },
            indent=2,
        )
    )

    train_X, train_y = extract_embeddings(model, train_loader, device)
    val_X, val_y = extract_embeddings(model, val_loader, device)

    print()
    print(f"Training linear head on PERMUTED labels ({len(names)} classes) ...")
    best_acc, best_epoch, preds, probs, weight_norms, biases = train_linear_head(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        num_classes=len(names),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    true = val_y.numpy()
    overall_acc = float((preds == true).mean())

    pred_counts = np.bincount(preds, minlength=len(names))
    pred_cv = float(pred_counts.std() / pred_counts.mean())

    # Correlations (like baseline analysis)
    from scipy.stats import pearsonr
    weight_norm_corr = float(pearsonr(weight_norms, pred_counts)[0])
    bias_corr = float(pearsonr(biases, pred_counts)[0])

    # Save artifacts
    (out_dir / "names.json").write_text(json.dumps(names))
    np.save(out_dir / "predictions.npy", preds)
    np.save(out_dir / "true_labels.npy", true)

    weight_norm_dict = {name: float(norm) for name, norm in zip(names, weight_norms)}
    bias_dict = {name: float(bias) for name, bias in zip(names, biases)}
    pred_count_dict = {name: int(count) for name, count in zip(names, pred_counts)}

    diagnostics = {
        "weight_norms": weight_norm_dict,
        "biases": bias_dict,
        "prediction_counts": pred_count_dict,
        "correlations": {
            "weight_norm_vs_pred_freq": weight_norm_corr,
            "bias_vs_pred_freq": bias_corr,
        },
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))

    rows = per_class_prf(preds, true, names)
    with open(out_dir / "precision_recall_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=["name", "recall", "precision", "f1_score", "support", "predicted_count", "pred_bias"],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = Summary(
        tag=args.tag,
        test_type="permutation",
        num_classes=len(names),
        val_size=int(len(true)),
        overall_accuracy=overall_acc,
        random_baseline=float(1 / len(names)),
        pred_count_cv=pred_cv,
        weight_norm_correlation=weight_norm_corr,
        bias_correlation=bias_corr,
        best_epoch=int(best_epoch),
        permutation_seed=args.permutation_seed,
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))

    print()
    print("=" * 70)
    print("PERMUTATION TEST RESULTS")
    print("=" * 70)
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"Random baseline:  {1/len(names):.4f} ({100/len(names):.1f}%)")
    print(f"Best epoch:       {best_epoch}")
    print()
    print("Prediction distribution:")
    print(f"  Pred CV: {pred_cv:.3f}  (baseline=0.400)")
    print()
    print("Correlations (check if they persist under random labels):")
    print(f"  Weight-norm vs pred-freq:  {weight_norm_corr:+.3f}  (baseline=+0.609)")
    print(f"  Bias vs pred-freq:         {bias_corr:+.3f}  (baseline=+0.040)")
    print()
    print("INTERPRETATION:")
    if overall_acc > 1.5 / len(names):
        print("  ⚠️  Accuracy significantly above random → possible data leakage or artifact!")
    else:
        print("  ✓  Accuracy near random → model was learning real signal (not artifact)")
    if abs(weight_norm_corr) > 0.3:
        print("  ⚠️  Weight-norm correlation persists → optimization artifact!")
    else:
        print("  ✓  Weight-norm correlation dropped → correlation was due to real signal")

    print()
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()
