#!/usr/bin/env python3
"""
Monkey-Moo ViT: Train a small ViT *from scratch* on the exact same train/val split
files used by the CLIP cosine-classifier experiment, for an apples-to-apples comparison.

Default split files are taken from:
  results/cosine_classifier/balanced_eval_p200_e10/{train_split.jsonl,val_split.jsonl}

Each JSONL line must look like:
  {"path": "/abs/path/to/img.jpg", "y": 12, "name": "maria"}

Outputs:
  results/monkey_moo_vit/<tag>/
    - config.json
    - log.csv
    - best.pt
    - final.pt
    - predictions.npy
    - true_labels.npy
    - summary.json
"""

from __future__ import annotations

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
from torchvision import transforms
from tqdm import tqdm

try:
    import timm
except ImportError as e:
    raise SystemExit("Missing dependency 'timm'. Install it with: pip install timm") from e


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_split_jsonl(path: str) -> List[Tuple[str, int]]:
    items: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            items.append((rec["path"], int(rec["y"])))
    return items


class SplitDataset(Dataset):
    def __init__(self, items: List[Tuple[str, int]], image_size: int):
        self.items = items
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        path, y = self.items[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, y


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    correct = 0
    total = 0
    preds_all: List[int] = []
    ys_all: List[int] = []
    use_cuda = device.type == "cuda"
    ctx = torch.autocast(device_type="cuda", dtype=torch.float16) if use_cuda else torch.autocast(device_type="cpu", enabled=False)
    with ctx:
        for x, y in loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            logits = model(x)
            preds = logits.argmax(dim=1)
            correct += int((preds == y).sum().item())
            total += int(y.numel())
            preds_all.extend(preds.detach().cpu().tolist())
            ys_all.extend(y.detach().cpu().tolist())
    acc = correct / max(1, total)
    return acc, np.asarray(preds_all, dtype=np.int64), np.asarray(ys_all, dtype=np.int64)


@dataclass
class Summary:
    tag: str
    num_classes: int
    train_size: int
    val_size: int
    best_epoch: int
    best_val_acc: float
    final_val_acc: float
    random_baseline: float


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", default="moo_vit_baseline")
    p.add_argument(
        "--train-split",
        default="/home/leann/face-detection/results/cosine_classifier/balanced_eval_p200_e10/train_split.jsonl",
    )
    p.add_argument(
        "--val-split",
        default="/home/leann/face-detection/results/cosine_classifier/balanced_eval_p200_e10/val_split.jsonl",
    )
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=384)
    p.add_argument("--model", default="vit_tiny_patch16_384")
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--compile", action="store_true", help="Enable torch.compile() (can speed up on some setups).")
    args = p.parse_args()

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_items = load_split_jsonl(args.train_split)
    val_items = load_split_jsonl(args.val_split)
    if not train_items or not val_items:
        raise SystemExit("Train/val split is empty. Check --train-split/--val-split paths.")

    # Basic split sanity
    train_paths = {p for p, _ in train_items}
    val_paths = {p for p, _ in val_items}
    overlap = len(train_paths & val_paths)
    if overlap:
        print(f"WARNING: train/val splits overlap on {overlap} paths (this should be 0).")

    # Infer number of classes from labels present
    ys = [y for _, y in train_items] + [y for _, y in val_items]
    num_classes = int(max(ys) + 1)

    out_dir = Path("/home/leann/face-detection/results/monkey_moo_vit") / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    (out_dir / "config.json").write_text(
        json.dumps(
            {
                "tag": args.tag,
                "train_split": args.train_split,
                "val_split": args.val_split,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "seed": args.seed,
                "num_workers": args.num_workers,
                "image_size": args.image_size,
                "model": args.model,
                "depth": args.depth,
                "device": str(device),
            },
            indent=2,
        )
    )

    train_ds = SplitDataset(train_items, image_size=args.image_size)
    val_ds = SplitDataset(val_items, image_size=args.image_size)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    model = timm.create_model(
        args.model,
        pretrained=False,
        num_classes=num_classes,
        depth=args.depth,
        global_pool="avg",
        class_token=False,
    ).to(device)
    if args.compile:
        model = torch.compile(model)

    print("=" * 70)
    print("MONKEY-MOO VIT: SCRATCH TRAINING ON CLIP SPLITS")
    print("=" * 70)
    print(f"Output dir: {out_dir}")
    print(f"Device:     {device}")
    print(f"Classes:    {num_classes}")
    print(f"Train size: {len(train_ds)}")
    print(f"Val size:   {len(val_ds)}")
    print(f"Random baseline: {1/num_classes:.4f} ({100/num_classes:.1f}%)")
    print(f"Model: {args.model} depth={args.depth} (pretrained=False, GAP, no-CLS)")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
    print()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    best = {"epoch": 0, "val_acc": -1.0, "state": None}

    log_path = out_dir / "log.csv"
    with open(log_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "train_acc", "val_acc"])
        w.writeheader()

        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0

            pbar = tqdm(train_loader, desc=f"epoch {epoch}/{args.epochs}", leave=False)
            for x, y in pbar:
                x = x.to(device, non_blocking=True)
                y = torch.as_tensor(y, device=device)
                opt.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                    logits = model(x)
                    loss = loss_fn(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()

                total_loss += float(loss.item())
                preds = logits.argmax(dim=1)
                correct += int((preds == y).sum().item())
                total += int(y.numel())
                pbar.set_postfix(loss=f"{loss.item():.3f}")

            train_loss = total_loss / max(1, len(train_loader))
            train_acc = correct / max(1, total)

            val_acc, val_preds, val_true = evaluate(model, val_loader, device)

            w.writerow(
                {
                    "epoch": epoch,
                    "train_loss": f"{train_loss:.6f}",
                    "train_acc": f"{train_acc:.6f}",
                    "val_acc": f"{val_acc:.6f}",
                }
            )
            f.flush()

            print(f"Epoch {epoch:3d} | train_loss={train_loss:.4f} | train_acc={train_acc*100:5.1f}% | val_acc={val_acc*100:5.1f}%")

            if val_acc > best["val_acc"]:
                best = {
                    "epoch": epoch,
                    "val_acc": float(val_acc),
                    "state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                }
                torch.save({"epoch": epoch, "model_state_dict": best["state"], "val_acc": float(val_acc)}, out_dir / "best.pt")

    # Save final checkpoint
    torch.save({"epoch": args.epochs, "model_state_dict": model.state_dict()}, out_dir / "final.pt")

    # Evaluate best checkpoint for a stable reported number
    if best["state"] is not None:
        model.load_state_dict(best["state"])
    final_val_acc, val_preds, val_true = evaluate(model, val_loader, device)
    np.save(out_dir / "predictions.npy", val_preds)
    np.save(out_dir / "true_labels.npy", val_true)

    summary = Summary(
        tag=args.tag,
        num_classes=num_classes,
        train_size=len(train_ds),
        val_size=len(val_ds),
        best_epoch=int(best["epoch"]),
        best_val_acc=float(best["val_acc"]),
        final_val_acc=float(final_val_acc),
        random_baseline=float(1 / num_classes),
    )
    (out_dir / "summary.json").write_text(json.dumps(asdict(summary), indent=2))

    print()
    print("=" * 70)
    print("MONKEY-MOO VIT SUMMARY")
    print("=" * 70)
    print(f"Best epoch: {best['epoch']}")
    print(f"Best val acc: {best['val_acc']:.4f} ({best['val_acc']*100:.1f}%)")
    print(f"Final val acc (best ckpt re-eval): {final_val_acc:.4f} ({final_val_acc*100:.1f}%)")
    print(f"Random baseline: {1/num_classes:.4f} ({100/num_classes:.1f}%)")
    print(f"Results saved to: {out_dir}/")


if __name__ == "__main__":
    main()

