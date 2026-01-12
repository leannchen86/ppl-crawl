"""
Ablation experiment: remove one or more names (classes) and re-run the balanced
linear-probe benchmark.

Why:
  - To test whether “dominance” (over-prediction) is specific to a name like
    William, or if the classifier tends to concentrate mass on whichever class
    has the strongest separability / confounds.

Design choices (for controlled comparison):
  - Reuse the exact prior 30-name list from results/scale_up_results/names.json
    unless --names is provided.
  - Balanced sampling: cap at --max-per-name per split.
  - Frozen CLIP embeddings (ViT-B-32 openai) + linear classifier head.

Outputs:
  - results/ablations/<tag>/
      - names.json
      - predictions.npy
      - true_labels.npy
      - precision_recall_metrics.csv
      - summary.json

Usage examples:
  # Remove William from the prior 30-name benchmark
  python ablation_remove_names.py --exclude william --tag no_william

  # Remove multiple names
  python ablation_remove_names.py --exclude william nick lisa --tag no_top3
"""

import argparse
import csv
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from open_clip import create_model_and_transforms


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class BalancedMultiNameDataset(Dataset):
    def __init__(
        self,
        index_dir: str,
        names: List[str],
        transform,
        split: str,
        train_ratio: float,
        seed: int,
        max_per_name: int,
    ):
        assert split in ("train", "val")
        self.transform = transform
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.samples: List[Tuple[str, int]] = []
        self.stats = {
            "split": split,
            "index_dir": str(index_dir),
            "names_total": len(names),
            "index_files_found": 0,
            "index_files_missing": 0,
            "selected_paths_total": 0,
            "selected_paths_exist": 0,
        }

        rng = random.Random(seed)
        for name in names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                self.stats["index_files_missing"] += 1
                continue
            self.stats["index_files_found"] += 1
            with open(index_path) as f:
                data = json.load(f)
            good_images = list(data.get("good", []))
            rng.shuffle(good_images)
            split_idx = int(len(good_images) * train_ratio)
            selected = good_images[:split_idx] if split == "train" else good_images[split_idx:]
            if max_per_name and len(selected) > max_per_name:
                selected = selected[:max_per_name]
            for p in selected:
                self.stats["selected_paths_total"] += 1
                if os.path.exists(p):
                    self.stats["selected_paths_exist"] += 1
                    self.samples.append((p, self.name_to_idx[name]))

        rng.shuffle(self.samples)

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
    # embeddings are float32 on CPU; we use autocast on GPU for speed
    use_cuda = (device.type == "cuda")
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else torch.autocast(device_type="cpu", enabled=False)
    with ctx:
        for images, labels in tqdm(loader, desc="Extracting embeddings"):
            images = images.to(device, non_blocking=True)
            feats = model.encode_image(images)
            feats = feats / feats.norm(dim=-1, keepdim=True)
            embs.append(feats.cpu())
            ys.append(labels.cpu())
    if not embs or not ys:
        ds_len = None
        try:
            ds_len = len(loader.dataset)  # type: ignore[attr-defined]
        except Exception:
            pass
        raise ValueError(
            "No embeddings were extracted because the dataloader yielded 0 batches "
            f"(dataset_len={ds_len}). This usually means your dataset is empty; "
            "check --index-dir and that image paths in the index files exist on disk."
        )
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
    # simple, stable training (mini-batch not needed for 15k)
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

    # restore best
    head.load_state_dict(best["state"])
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        val_logits = head(val_X)
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
    return best["acc"], best["epoch"], val_preds, val_probs


def per_class_prf(preds: np.ndarray, true: np.ndarray, names: List[str]):
    n = len(names)
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
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "pred_bias": (predicted_count / support) if support else 0.0,
            }
        )
    rows.sort(key=lambda r: r["f1_score"], reverse=True)
    return rows


@dataclass
class Summary:
    tag: str
    num_classes: int
    val_size: int
    overall_accuracy: float
    random_baseline: float
    pred_count_min: int
    pred_count_max: int
    pred_count_cv: float
    best_epoch: int
    excluded: List[str]


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index-dir",
        default="/home/leann/face-detection/data/index_files",
        help="Directory containing per-name index files like index_<name>.json",
    )
    p.add_argument("--names-json", default="/home/leann/face-detection/results/scale_up_results/names.json")
    p.add_argument("--names", nargs="*", default=None, help="Override names list explicitly")
    p.add_argument("--exclude", nargs="*", default=["william"], help="Names to exclude")
    p.add_argument("--tag", default="no_william")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-per-name", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load names
    if args.names is not None and len(args.names) > 0:
        names = [n.strip().lower() for n in args.names]
    else:
        with open(args.names_json) as f:
            names = json.load(f)
        names = [n.strip().lower() for n in names]

    exclude = set(n.strip().lower() for n in (args.exclude or []))
    kept = [n for n in names if n not in exclude]
    if len(kept) == 0:
        raise SystemExit(f"After applying --exclude, no classes remain (exclude={sorted(exclude)}).")

    out_dir = Path("/home/leann/face-detection/results/ablations") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("ABLATION: REMOVE NAMES AND RE-RUN LINEAR PROBE")
    print("=" * 70)
    print(f"Tag: {args.tag}")
    print(f"Excluded: {sorted(exclude)}")
    print(f"Kept classes: {len(kept)} (was {len(names)})")
    print(f"Random baseline: {1/len(kept):.4f}")
    print(f"Output dir: {out_dir}")
    print()

    # Validate index dir early so we don't silently build empty datasets.
    index_dir = Path(args.index_dir).expanduser()
    if not index_dir.is_dir():
        suggestions = []
        repo_default = Path(__file__).resolve().parent / "data" / "index_files"
        if repo_default.is_dir():
            suggestions.append(str(repo_default))
        hardcoded_default = Path("/home/leann/face-detection/data/index_files")
        if hardcoded_default.is_dir() and str(hardcoded_default) not in suggestions:
            suggestions.append(str(hardcoded_default))
        msg = f"--index-dir is not a directory: {index_dir}"
        if suggestions:
            msg += "\nDid you mean one of:\n  - " + "\n  - ".join(suggestions)
        raise SystemExit(msg)

    # Load CLIP
    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    for param in model.parameters():
        param.requires_grad = False

    # Datasets
    train_ds = BalancedMultiNameDataset(
        index_dir=str(index_dir),
        names=kept,
        transform=preprocess,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_per_name=args.max_per_name,
    )
    val_ds = BalancedMultiNameDataset(
        index_dir=str(index_dir),
        names=kept,
        transform=preprocess,
        split="val",
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_per_name=args.max_per_name,
    )
    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")
    if len(train_ds) == 0 or len(val_ds) == 0:
        print()
        print("ERROR: Empty dataset after reading index files.")
        print(f"Index dir: {index_dir}")
        print(f"Index files found (train split): {train_ds.stats['index_files_found']}/{train_ds.stats['names_total']}")
        print(
            "Selected image paths that exist on disk (train split): "
            f"{train_ds.stats['selected_paths_exist']}/{train_ds.stats['selected_paths_total']}"
        )
        if train_ds.stats["index_files_found"] == 0:
            print("Cause: no index_<name>.json files were found for the requested names.")
            print("Fix: point --index-dir at the folder containing index_<name>.json files.")
            print("Example: /home/leann/face-detection/data/index_files")
        elif train_ds.stats["selected_paths_total"] > 0 and train_ds.stats["selected_paths_exist"] == 0:
            print("Cause: index files were found, but none of the selected image paths exist on disk.")
            print("Fix: check that /home/leann/ppl-images is present and paths in index files are still valid.")
        raise SystemExit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    train_X, train_y = extract_embeddings(model, train_loader, device)
    val_X, val_y = extract_embeddings(model, val_loader, device)

    print()
    print(f"Training linear head ({len(kept)} classes) ...")
    best_acc, best_epoch, preds, probs = train_linear_head(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        num_classes=len(kept),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )

    true = val_y.numpy()
    overall_acc = float((preds == true).mean())

    pred_counts = np.bincount(preds, minlength=len(kept))
    pred_cv = float(pred_counts.std() / pred_counts.mean())

    # Save artifacts
    (out_dir / "names.json").write_text(json.dumps(kept))
    np.save(out_dir / "predictions.npy", preds)
    np.save(out_dir / "true_labels.npy", true)

    rows = per_class_prf(preds, true, kept)
    with open(out_dir / "precision_recall_metrics.csv", "w", newline="") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "recall",
                "precision",
                "f1_score",
                "support",
                "predicted_count",
                "true_positives",
                "false_positives",
                "false_negatives",
                "pred_bias",
            ],
        )
        w.writeheader()
        for r in rows:
            w.writerow(r)

    summary = Summary(
        tag=args.tag,
        num_classes=len(kept),
        val_size=int(len(true)),
        overall_accuracy=overall_acc,
        random_baseline=float(1 / len(kept)),
        pred_count_min=int(pred_counts.min()),
        pred_count_max=int(pred_counts.max()),
        pred_count_cv=pred_cv,
        best_epoch=int(best_epoch),
        excluded=sorted(exclude),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))

    print()
    print("=" * 70)
    print("ABLATION RESULT SUMMARY")
    print("=" * 70)
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"Random baseline:  {1/len(kept):.4f} ({100/len(kept):.1f}%)")
    print(f"Best epoch:       {best_epoch}")
    print(f"Pred count range: [{pred_counts.min()}, {pred_counts.max()}]")
    print(f"Pred count CV:    {pred_cv:.3f}  (higher=more skew)")
    print()
    print("Top-5 by F1 (who becomes dominant?):")
    for r in rows[:5]:
        print(
            f"  {r['name']:<10s} "
            f"F1={r['f1_score']*100:5.1f}%  "
            f"P={r['precision']*100:5.1f}%  "
            f"R={r['recall']*100:5.1f}%  "
            f"pred/actual={r['pred_bias']:.2f}x"
        )


if __name__ == "__main__":
    main()


