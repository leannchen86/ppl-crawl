"""
Track 2: Vision Transformer Training from Scratch

Train a ViT-Base/16 from scratch on the full 435k face dataset.
Supports class-weighted loss, mixup/cutmix, label smoothing, and more.

Usage:
    # Experiment 2.1: Baseline ViT-Base
    python train_vit_scratch.py --experiment baseline

    # Experiment 2.2: With Mixup/CutMix
    python train_vit_scratch.py --experiment mixup

    # Experiment 2.3: Label smoothing ablation
    python train_vit_scratch.py --experiment label_smooth --label-smoothing 0.2

    # Experiment 2.4: Subset (top-30 names for comparison)
    python train_vit_scratch.py --experiment subset --num-names 30

    # Full custom run
    python train_vit_scratch.py --epochs 100 --batch-size 256 --model vit_base_patch16_224
"""
import argparse
import json
import math
import os
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

try:
    import timm
    from timm.data.mixup import Mixup
    from timm.loss import SoftTargetCrossEntropy
except ImportError:
    print("Please install timm: pip install timm")
    exit(1)


def seed_everything(seed=42):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FaceNameDataset(Dataset):
    """Dataset for face images with name labels."""

    def __init__(
        self,
        index_dir: str,
        names: list,
        transform=None,
        split: str = "train",
        seed: int = 42,
        max_per_name: int = None,
    ):
        """
        Args:
            index_dir: Directory with index_*.json files
            names: List of name strings (defines class order)
            transform: Torchvision transforms
            split: "train" or "val"
            seed: Random seed for reproducible splits
            max_per_name: Limit samples per class (None for unlimited)
        """
        self.transform = transform
        self.samples = []  # (image_path, label_idx)
        self.class_counts = []  # Count per class

        rng = random.Random(seed)

        for idx, name in enumerate(names):
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                print(f"Warning: No index for '{name}'")
                self.class_counts.append(0)
                continue

            with open(index_path) as f:
                data = json.load(f)

            images = data.get("good", [])

            # Deterministic split using hash
            train_imgs = [img for img in images if hash(img) % 10 < 8]
            val_imgs = [img for img in images if hash(img) % 10 >= 8]

            selected = train_imgs if split == "train" else val_imgs

            # Apply max_per_name limit if specified
            if max_per_name and len(selected) > max_per_name:
                rng.shuffle(selected)
                selected = selected[:max_per_name]

            # Filter to existing files
            valid_paths = [p for p in selected if os.path.exists(p)]
            self.samples.extend([(p, idx) for p in valid_paths])
            self.class_counts.append(len(valid_paths))

        print(f"[{split}] Loaded {len(self.samples)} samples across {len(names)} classes")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            image = Image.open(path).convert("RGB")
        except Exception as e:
            print(f"Error loading {path}: {e}")
            # Return a black image on error
            image = Image.new("RGB", (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label

    def get_class_weights(self, method="effective"):
        """Compute class weights for imbalanced data.

        Methods:
            - inverse: 1/count (aggressive)
            - sqrt: sqrt(total/count) (moderate)
            - effective: (1-beta)/(1-beta^n) from Class-Balanced Loss paper
        """
        counts = np.array(self.class_counts, dtype=np.float32)
        counts = np.maximum(counts, 1)  # Avoid division by zero
        total = counts.sum()

        if method == "inverse":
            weights = total / (len(counts) * counts)
        elif method == "sqrt":
            weights = np.sqrt(total / counts)
        elif method == "effective":
            beta = 0.999
            weights = (1 - beta) / (1 - np.power(beta, counts))
        else:
            weights = np.ones_like(counts)

        # Normalize so mean = 1
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)


def discover_names(index_dir: str, min_samples: int = 100, max_names: int = None):
    """Discover available names and their sample counts."""
    names_with_counts = []

    for filepath in Path(index_dir).glob("index_*.json"):
        name = filepath.stem.replace("index_", "")
        try:
            with open(filepath) as f:
                data = json.load(f)
            count = data.get("counts", {}).get("good", len(data.get("good", [])))
            if count >= min_samples:
                names_with_counts.append((name, count))
        except Exception:
            pass

    # Sort by count (descending)
    names_with_counts.sort(key=lambda x: -x[1])

    if max_names:
        names_with_counts = names_with_counts[:max_names]

    return names_with_counts


def create_model(model_name: str, num_classes: int, drop_rate: float = 0.1,
                 drop_path_rate: float = 0.1):
    """Create ViT model from scratch (no pretrained weights)."""
    model = timm.create_model(
        model_name,
        pretrained=False,  # FROM SCRATCH
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )

    # Initialize weights properly for training from scratch
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    model.apply(_init_weights)
    return model


def get_transforms(img_size: int = 224, is_train: bool = True):
    """Get data augmentation transforms."""
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomGrayscale(p=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
            transforms.RandomErasing(p=0.25, scale=(0.02, 0.33)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(int(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup and cosine decay."""

    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * self.current_epoch / self.warmup_epochs
        else:
            # Cosine decay
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

    def get_lr(self):
        return self.optimizer.param_groups[0]['lr']


def train_one_epoch(model, loader, optimizer, criterion, scaler, device,
                    mixup_fn=None, grad_clip=1.0):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for images, labels in pbar:
        images, labels = images.to(device), labels.to(device)

        # Apply mixup/cutmix if enabled
        if mixup_fn is not None:
            images, labels = mixup_fn(images, labels)

        optimizer.zero_grad()

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()

        # Gradient clipping
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

        # For accuracy, only count when not using mixup
        if mixup_fn is None:
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total if total > 0 else 0
    return avg_loss, accuracy


@torch.no_grad()
def evaluate(model, loader, criterion, device, num_classes):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []
    class_correct = torch.zeros(num_classes)
    class_total = torch.zeros(num_classes)

    for images, labels in tqdm(loader, desc="Evaluating", leave=False):
        images, labels = images.to(device), labels.to(device)

        with autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        # Per-class accuracy
        for i in range(labels.size(0)):
            label = labels[i].item()
            class_total[label] += 1
            if predicted[i] == label:
                class_correct[label] += 1

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    # Compute per-class metrics
    per_class_acc = class_correct / (class_total + 1e-6)

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "predictions": np.array(all_preds),
        "labels": np.array(all_labels),
        "per_class_acc": per_class_acc.numpy(),
    }


def compute_prediction_cv(predictions):
    """Compute coefficient of variation of prediction distribution."""
    unique, counts = np.unique(predictions, return_counts=True)
    return counts.std() / counts.mean() if counts.mean() > 0 else 0


def save_results(output_dir, names, history, best_metrics, config):
    """Save all results in standardized format."""
    os.makedirs(output_dir, exist_ok=True)

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    # Save names
    with open(os.path.join(output_dir, "names.json"), "w") as f:
        json.dump(names, f)

    # Save training history
    with open(os.path.join(output_dir, "training_log.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Save predictions and labels
    if "predictions" in best_metrics:
        np.save(os.path.join(output_dir, "predictions.npy"), best_metrics["predictions"])
        np.save(os.path.join(output_dir, "true_labels.npy"), best_metrics["labels"])

    # Generate per-name results CSV
    if "per_class_acc" in best_metrics:
        import csv
        with open(os.path.join(output_dir, "results.csv"), "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["name", "accuracy"])
            for name, acc in zip(names, best_metrics["per_class_acc"]):
                writer.writerow([name, f"{acc*100:.2f}"])

    print(f"Results saved to {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Train ViT from scratch on face data")

    # Experiment presets
    parser.add_argument("--experiment", type=str, default=None,
                        choices=["baseline", "mixup", "label_smooth", "subset"],
                        help="Predefined experiment configuration")

    # Model
    parser.add_argument("--model", type=str, default="vit_base_patch16_224",
                        choices=["vit_tiny_patch16_224", "vit_small_patch16_224",
                                "vit_base_patch16_224", "vit_large_patch16_224"],
                        help="ViT model variant")

    # Data
    parser.add_argument("--index-dir", type=str,
                        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001",
                        help="Directory with index files")
    parser.add_argument("--num-names", type=int, default=500,
                        help="Number of names/classes to use")
    parser.add_argument("--min-samples", type=int, default=50,
                        help="Minimum samples per name")
    parser.add_argument("--max-per-name", type=int, default=None,
                        help="Maximum samples per name (None for unlimited)")

    # Training
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="Base learning rate (scaled by batch size)")
    parser.add_argument("--warmup-epochs", type=int, default=10)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--drop-rate", type=float, default=0.1)
    parser.add_argument("--drop-path-rate", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    # Regularization
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--mixup-alpha", type=float, default=0.0,
                        help="Mixup alpha (0 to disable)")
    parser.add_argument("--cutmix-alpha", type=float, default=0.0,
                        help="CutMix alpha (0 to disable)")
    parser.add_argument("--class-weight-method", type=str, default="effective",
                        choices=["none", "inverse", "sqrt", "effective"],
                        help="Class weighting method for imbalance")

    # Output
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory (auto-generated if not specified)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=8)

    args = parser.parse_args()

    # Apply experiment presets
    if args.experiment == "baseline":
        args.mixup_alpha = 0.0
        args.cutmix_alpha = 0.0
        args.label_smoothing = 0.1
        if args.output_dir is None:
            args.output_dir = "results/track2_vit_scratch/exp2.1_baseline"
    elif args.experiment == "mixup":
        args.mixup_alpha = 0.8
        args.cutmix_alpha = 1.0
        args.label_smoothing = 0.1
        if args.output_dir is None:
            args.output_dir = "results/track2_vit_scratch/exp2.2_mixup"
    elif args.experiment == "label_smooth":
        args.mixup_alpha = 0.0
        args.cutmix_alpha = 0.0
        # label_smoothing should be set via command line
        if args.output_dir is None:
            args.output_dir = f"results/track2_vit_scratch/exp2.3_ls{args.label_smoothing}"
    elif args.experiment == "subset":
        args.num_names = 30 if args.num_names == 500 else args.num_names
        if args.output_dir is None:
            args.output_dir = f"results/track2_vit_scratch/exp2.4_subset{args.num_names}"

    if args.output_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output_dir = f"results/track2_vit_scratch/run_{timestamp}"

    # Setup
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("Track 2: Vision Transformer from Scratch")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Output: {args.output_dir}")

    # Discover names
    print("\nDiscovering names...")
    names_with_counts = discover_names(
        args.index_dir,
        min_samples=args.min_samples,
        max_names=args.num_names
    )
    names = [n for n, c in names_with_counts]
    total_samples = sum(c for n, c in names_with_counts)

    print(f"Found {len(names)} names with {total_samples:,} total samples")
    print(f"Top 5: {names_with_counts[:5]}")
    print(f"Bottom 5: {names_with_counts[-5:]}")

    # Create datasets
    print("\nCreating datasets...")
    train_transform = get_transforms(224, is_train=True)
    val_transform = get_transforms(224, is_train=False)

    train_dataset = FaceNameDataset(
        args.index_dir, names, train_transform,
        split="train", seed=args.seed, max_per_name=args.max_per_name
    )
    val_dataset = FaceNameDataset(
        args.index_dir, names, val_transform,
        split="val", seed=args.seed, max_per_name=args.max_per_name
    )

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True
    )

    print(f"Train samples: {len(train_dataset):,}")
    print(f"Val samples: {len(val_dataset):,}")
    print(f"Random baseline: {100/len(names):.2f}%")

    # Create model
    print(f"\nCreating model: {args.model}")
    model = create_model(
        args.model,
        num_classes=len(names),
        drop_rate=args.drop_rate,
        drop_path_rate=args.drop_path_rate
    )
    model = model.to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")

    # Loss function with class weights
    if args.class_weight_method != "none":
        class_weights = train_dataset.get_class_weights(args.class_weight_method).to(device)
        print(f"Using {args.class_weight_method} class weighting")
        print(f"Weight range: [{class_weights.min():.3f}, {class_weights.max():.3f}]")
    else:
        class_weights = None

    # Setup mixup if enabled
    mixup_fn = None
    if args.mixup_alpha > 0 or args.cutmix_alpha > 0:
        mixup_fn = Mixup(
            mixup_alpha=args.mixup_alpha,
            cutmix_alpha=args.cutmix_alpha,
            prob=1.0,
            switch_prob=0.5,
            mode="batch",
            label_smoothing=args.label_smoothing,
            num_classes=len(names)
        )
        criterion = SoftTargetCrossEntropy()
        print(f"Using Mixup (alpha={args.mixup_alpha}) and CutMix (alpha={args.cutmix_alpha})")
    else:
        criterion = nn.CrossEntropyLoss(
            weight=class_weights,
            label_smoothing=args.label_smoothing
        )
        print(f"Using CrossEntropyLoss with label_smoothing={args.label_smoothing}")

    # For validation, always use standard CE
    val_criterion = nn.CrossEntropyLoss()

    # Optimizer
    scaled_lr = args.lr * args.batch_size / 256
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=scaled_lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    print(f"Learning rate: {scaled_lr:.6f} (base {args.lr} scaled by batch size)")

    # Scheduler
    scheduler = WarmupCosineScheduler(
        optimizer, args.warmup_epochs, args.epochs, scaled_lr
    )

    # Mixed precision
    scaler = GradScaler()

    # Training loop
    print("\n" + "=" * 70)
    print(f"Starting training for {args.epochs} epochs")
    print("=" * 70)

    history = {
        "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [],
        "learning_rate": [], "prediction_cv": []
    }

    best_val_acc = 0
    best_metrics = {}
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, optimizer, criterion, scaler, device,
            mixup_fn=mixup_fn, grad_clip=args.grad_clip
        )

        # Update LR
        current_lr = scheduler.step()

        # Evaluate
        val_metrics = evaluate(model, val_loader, val_criterion, device, len(names))
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        # Compute prediction CV
        pred_cv = compute_prediction_cv(val_metrics["predictions"])

        # Record history
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["learning_rate"].append(current_lr)
        history["prediction_cv"].append(pred_cv)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_metrics = val_metrics.copy()
            torch.save(model.state_dict(), os.path.join(args.output_dir, "best_model.pth"))

        # Progress
        epoch_time = time.time() - epoch_start
        overfit_gap = train_acc - val_acc if train_acc > 0 else 0
        warning = " [OVERFIT]" if overfit_gap > 30 else ""

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:5.1f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:5.1f}% | "
              f"lr={current_lr:.2e} | cv={pred_cv:.3f} | "
              f"time={epoch_time:.1f}s{warning}")

        # Save checkpoint periodically
        if epoch % 10 == 0:
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "history": history,
            }, os.path.join(args.output_dir, f"checkpoint_epoch{epoch}.pth"))

    total_time = time.time() - start_time

    # Save final results
    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)

    config = vars(args)
    config["total_time_hours"] = total_time / 3600
    config["best_val_acc"] = best_val_acc
    config["final_train_acc"] = history["train_acc"][-1] if history["train_acc"] else 0
    config["final_val_acc"] = history["val_acc"][-1] if history["val_acc"] else 0
    config["num_params"] = num_params
    config["train_samples"] = len(train_dataset)
    config["val_samples"] = len(val_dataset)
    config["num_classes"] = len(names)

    save_results(args.output_dir, names, history, best_metrics, config)

    # Print summary
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    print(f"Final train accuracy: {config['final_train_acc']:.2f}%")
    print(f"Final val accuracy: {config['final_val_acc']:.2f}%")
    print(f"Prediction CV: {history['prediction_cv'][-1]:.3f}")
    print(f"Random baseline: {100/len(names):.2f}%")
    print(f"Total training time: {total_time/3600:.2f} hours")

    # Compare with baselines
    print("\n" + "-" * 70)
    print("Comparison with baselines:")
    print("-" * 70)
    print(f"{'Method':<35} {'Accuracy':<15}")
    print(f"{'Random baseline':<35} {100/len(names):.1f}%")
    print(f"{'CLIP linear probe (30 names)':<35} 13.9%")
    print(f"{'Fine-tune pretrained ViT':<35} 11.7%")
    print(f"{'Train CNN from scratch':<35} 10.9%")
    print(f"{'This run (ViT from scratch)':<35} {best_val_acc:.1f}%")


if __name__ == "__main__":
    main()
