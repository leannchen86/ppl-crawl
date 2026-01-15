"""
Cosine Classifier Test: Use normalized weight vectors to eliminate weight-norm-driven skew.

Why:
  - The baseline linear probe has weight-norm vs prediction-frequency correlation of 0.609
  - This suggests prediction skew may be an artifact of weight vector magnitudes
  - Cosine classifier forces all weight vectors to unit norm, eliminating this artifact

Design:
  - Reuse exact 30-name benchmark from results/scale_up_results/names.json
  - Balanced sampling: 500 per name per split
  - Frozen CLIP embeddings (ViT-B-32 openai)
  - **Cosine classifier**: logits = (W/||W||) @ (x/||x||) * scale + bias

Outputs:
  - results/cosine_classifier/
      - names.json
      - predictions.npy
      - true_labels.npy
      - precision_recall_metrics.csv
      - summary.json
      - weight_norms.json  (should all be ~1.0)

Usage:
  python cosine_classifier_test.py --tag cosine_baseline
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
import torch.nn.functional as F
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


class CosineClassifier(nn.Module):
    """Cosine similarity classifier with normalized weight vectors."""

    def __init__(self, in_features: int, num_classes: int, scale: float = 10.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, in_features))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        self.scale = scale  # Temperature scaling for cosine similarity

        # Initialize with normalized weights
        nn.init.xavier_normal_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Normalize input features
        x_norm = F.normalize(x, p=2, dim=1)

        # Normalize weight vectors
        w_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity: (W/||W||) @ (x/||x||)
        logits = F.linear(x_norm, w_norm, self.bias) * self.scale

        return logits

    def get_weight_norms(self) -> torch.Tensor:
        """Get L2 norms of weight vectors (for debugging)."""
        return torch.norm(self.weight, p=2, dim=1)


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
        per_name: int = 0,
        strict_per_name: bool = False,
        image_source: ImageSource = "chips",
    ):
        assert split in ("train", "val")
        self.transform = transform
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.samples: List[Tuple[str, int]] = []
        # For audit/reproducibility: exact filepaths selected for this split per class
        self.selected_paths_by_name: Dict[str, List[str]] = {n: [] for n in names}
        self.stats = {
            "split": split,
            "index_dir": str(index_dir),
            "names_total": len(names),
            "index_files_found": 0,
            "index_files_missing": 0,
            "selected_paths_total": 0,
            "selected_paths_exist": 0,
            "max_per_name": int(max_per_name),
            "per_name_requested": int(per_name) if per_name else None,
            "strict_per_name": bool(strict_per_name),
        }

        rng = random.Random(seed)
        per_name_candidates: Dict[str, List[str]] = {}
        for name in names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                self.stats["index_files_missing"] += 1
                continue
            self.stats["index_files_found"] += 1
            with open(index_path) as f:
                data = json.load(f)
            good_images = list(resolve_good_images(data, image_source=image_source))
            rng.shuffle(good_images)
            split_idx = int(len(good_images) * train_ratio)
            selected = good_images[:split_idx] if split == "train" else good_images[split_idx:]

            # If per_name is enabled, defer sampling until we know the minimum available per class.
            if per_name and per_name > 0:
                per_name_candidates[name] = selected
                continue

            if max_per_name and len(selected) > max_per_name:
                selected = selected[:max_per_name]
            for p in selected:
                self.stats["selected_paths_total"] += 1
                if os.path.exists(p):
                    self.stats["selected_paths_exist"] += 1
                    self.samples.append((p, self.name_to_idx[name]))
                    self.selected_paths_by_name[name].append(p)

        if per_name and per_name > 0:
            missing = [n for n in names if n not in per_name_candidates]
            if missing:
                msg = (
                    f"[{split}] Missing index files for {len(missing)} classes; "
                    f"cannot enforce per-class sampling. Missing examples: {missing[:5]}"
                )
                if strict_per_name:
                    raise ValueError(msg)
                print("WARNING:", msg)

            # Filter to existing paths, then enforce a uniform per-class count.
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
                self.stats["selected_paths_total"] += len(take)
                self.stats["selected_paths_exist"] += len(take)
                for p in take:
                    self.samples.append((p, self.name_to_idx[name]))
                self.selected_paths_by_name[name].extend(take)

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
            ds_len = len(loader.dataset)
        except Exception:
            pass
        raise ValueError(
            "No embeddings were extracted because the dataloader yielded 0 batches "
            f"(dataset_len={ds_len}). This usually means your dataset is empty; "
            "check --index-dir and that image paths in the index files exist on disk."
        )
    return torch.cat(embs, dim=0), torch.cat(ys, dim=0)


def train_cosine_head(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    scale: float,
    seed: int,
    device: torch.device,
):
    torch.manual_seed(seed)
    head = CosineClassifier(train_X.shape[1], num_classes, scale=scale).to(device)
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

    # Restore best
    head.load_state_dict(best["state"])
    head = head.to(device)
    head.eval()
    with torch.no_grad():
        val_logits = head(val_X)
        val_probs = torch.softmax(val_logits, dim=1).cpu().numpy()
        val_preds = val_logits.argmax(dim=1).cpu().numpy()
        weight_norms = head.get_weight_norms().cpu().numpy()

    return best["acc"], best["epoch"], val_preds, val_probs, weight_norms


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
    classifier_type: str
    num_classes: int
    val_size: int
    overall_accuracy: float
    random_baseline: float
    pred_count_min: int
    pred_count_max: int
    pred_count_cv: float
    weight_norm_min: float
    weight_norm_max: float
    weight_norm_cv: float
    best_epoch: int
    scale: float


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index-dir",
        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001",
        help="Directory containing per-name index files like index_<name>.json",
    )
    p.add_argument("--names-json", default="/home/leann/face-detection/results/scale_up_results/names.json")
    p.add_argument("--names", nargs="*", default=None, help="Override names list explicitly")
    p.add_argument("--tag", default="cosine_baseline")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--max-per-name", type=int, default=500)
    p.add_argument(
        "--per-name",
        type=int,
        default=0,
        help="If >0, enforce a uniform per-class sample count per split (after train/val split) "
             "for a truly balanced evaluation set. If some class has fewer images, the per-class "
             "count is reduced to the minimum across classes (unless --strict-per-name).",
    )
    p.add_argument(
        "--strict-per-name",
        action="store_true",
        help="If set with --per-name, error out instead of reducing per-class counts when data is insufficient.",
    )
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--scale", type=float, default=10.0, help="Temperature scale for cosine similarity")
    p.add_argument(
        "--image-source",
        choices=["chips", "original"],
        default="chips",
        help="Choose which images to use from the index files: "
        "'chips' uses index['good']; 'original' uses index['meta'][chip].src_path.",
    )
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

    out_dir = Path("/home/leann/face-detection/results/cosine_classifier") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("COSINE CLASSIFIER TEST: NORMALIZED WEIGHT VECTORS")
    print("=" * 70)
    print(f"Tag: {args.tag}")
    print(f"Classes: {len(names)}")
    print(f"Random baseline: {1/len(names):.4f}")
    print(f"Output dir: {out_dir}")
    print(f"Scale (temperature): {args.scale}")
    print()

    # Validate index dir
    index_dir = Path(args.index_dir).expanduser()
    if not index_dir.is_dir():
        suggestions = []
        repo_default = (
            Path(__file__).resolve().parent
            / "data"
            / "index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001"
        )
        if repo_default.is_dir():
            suggestions.append(str(repo_default))
        hardcoded_default = Path("/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001")
        if hardcoded_default.is_dir() and str(hardcoded_default) not in suggestions:
            suggestions.append(str(hardcoded_default))
        deprecated_og = Path("/home/leann/face-detection/data/deprecated_index_dirs_2026-01-13/index_files_og")
        if deprecated_og.is_dir() and str(deprecated_og) not in suggestions:
            suggestions.append(str(deprecated_og))
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
        names=names,
        transform=preprocess,
        split="train",
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_per_name=args.max_per_name,
        per_name=args.per_name,
        strict_per_name=args.strict_per_name,
        image_source=args.image_source,  # type: ignore[arg-type]
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
            print("Example: /home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001")
        elif train_ds.stats["selected_paths_total"] > 0 and train_ds.stats["selected_paths_exist"] == 0:
            print("Cause: index files were found, but none of the selected image paths exist on disk.")
            print(
                "Fix: check that the original images directory is present and paths in index files are still valid.\n"
                "     New default location: /home/leann/face-detection/data/original ppl images\n"
                "     Back-compat symlink:  /home/leann/ppl-images"
            )
        raise SystemExit(1)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Save the exact split membership so you can inspect/verify "held-out" images.
    # JSONL is used so it's easy to stream/grep and scales to larger splits.
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
            },
            indent=2,
        )
    )

    train_X, train_y = extract_embeddings(model, train_loader, device)
    val_X, val_y = extract_embeddings(model, val_loader, device)

    print()
    print(f"Training cosine classifier ({len(names)} classes) ...")
    best_acc, best_epoch, preds, probs, weight_norms = train_cosine_head(
        train_X=train_X,
        train_y=train_y,
        val_X=val_X,
        val_y=val_y,
        num_classes=len(names),
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        scale=args.scale,
        seed=args.seed,
        device=device,
    )

    true = val_y.numpy()
    overall_acc = float((preds == true).mean())

    pred_counts = np.bincount(preds, minlength=len(names))
    pred_cv = float(pred_counts.std() / pred_counts.mean())
    weight_norm_cv = float(weight_norms.std() / weight_norms.mean())

    # Save artifacts
    (out_dir / "names.json").write_text(json.dumps(names))
    np.save(out_dir / "predictions.npy", preds)
    np.save(out_dir / "true_labels.npy", true)

    # Save weight norms for verification
    weight_norm_dict = {name: float(norm) for name, norm in zip(names, weight_norms)}
    (out_dir / "weight_norms.json").write_text(json.dumps(weight_norm_dict, indent=2))

    rows = per_class_prf(preds, true, names)
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
        classifier_type="cosine",
        num_classes=len(names),
        val_size=int(len(true)),
        overall_accuracy=overall_acc,
        random_baseline=float(1 / len(names)),
        pred_count_min=int(pred_counts.min()),
        pred_count_max=int(pred_counts.max()),
        pred_count_cv=pred_cv,
        weight_norm_min=float(weight_norms.min()),
        weight_norm_max=float(weight_norms.max()),
        weight_norm_cv=weight_norm_cv,
        best_epoch=int(best_epoch),
        scale=args.scale,
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))

    print()
    print("=" * 70)
    print("COSINE CLASSIFIER RESULT SUMMARY")
    print("=" * 70)
    print(f"Overall accuracy: {overall_acc:.4f} ({overall_acc*100:.1f}%)")
    print(f"Random baseline:  {1/len(names):.4f} ({100/len(names):.1f}%)")
    print(f"Best epoch:       {best_epoch}")
    print()
    print(f"Weight norms (should be ~equal after normalization):")
    print(f"  Min:  {weight_norms.min():.4f}")
    print(f"  Max:  {weight_norms.max():.4f}")
    print(f"  Mean: {weight_norms.mean():.4f}")
    print(f"  CV:   {weight_norm_cv:.4f}  (baseline had implicit norms)")
    print()
    print(f"Prediction distribution:")
    print(f"  Count range: [{pred_counts.min()}, {pred_counts.max()}]")
    print(f"  Pred CV:     {pred_cv:.3f}  (baseline=0.400; lower=more uniform)")
    print()
    print("Top-5 by F1 (does dominance persist?):")
    for r in rows[:5]:
        print(
            f"  {r['name']:<10s} "
            f"F1={r['f1_score']*100:5.1f}%  "
            f"P={r['precision']*100:5.1f}%  "
            f"R={r['recall']*100:5.1f}%  "
            f"pred/actual={r['pred_bias']:.2f}x"
        )

    print()
    print(f"Results saved to: {out_dir}/")
    print()
    print("INTERPRETATION:")
    print("  - If pred_CV drops significantly → skew was weight-norm artifact")
    print("  - If pred_CV stays high → skew comes from embedding separability")
    print("  - Compare accuracy: if similar → no harm from normalization")


if __name__ == "__main__":
    main()
