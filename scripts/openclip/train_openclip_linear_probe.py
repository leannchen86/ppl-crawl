"""
OpenCLIP Linear Probe Training for Face-Name Classification.

This script trains a linear classifier on top of frozen OpenCLIP embeddings
for the "Guessing Your English Name" project.

Key design decisions:
1. Linear probe (frozen backbone) - proven to outperform fine-tuning for this task
2. OpenCLIP ViT-B-32 trained on LAION-2B - widely used open-source CLIP
3. Uses pre-created held-out validation split for fair comparison
4. Supports both balanced and imbalanced training data

Usage:
    python scripts/openclip/train_openclip_linear_probe.py \
        --data-dir /home/leann/face-detection/data/siglip_balanced \
        --output-dir ./results/openclip/balanced

Model info:
- ViT-B-32 (laion2b_s34b_b79k): ~150M params, embedding dim 512
- Trained on LAION-2B dataset with contrastive loss
"""
import argparse
import json
import os
import random
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
import open_clip


def seed_everything(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class FaceDataset(Dataset):
    """Simple dataset loading images and labels from JSON manifest."""

    def __init__(self, json_path: str, transform=None):
        with open(json_path) as f:
            self.samples = json.load(f)
        self.transform = transform

        # Filter out missing files and report
        valid_samples = []
        missing = 0
        for s in self.samples:
            if os.path.exists(s["image"]):
                valid_samples.append(s)
            else:
                missing += 1
        if missing > 0:
            print(f"  Warning: {missing} images not found, using {len(valid_samples)}/{len(self.samples)}")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img = Image.open(sample["image"]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, sample["label"]


class LinearProbeClassifier(nn.Module):
    """Linear classifier on top of frozen embeddings."""

    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        return self.classifier(x)


def load_openclip_model(model_name: str = "ViT-B-32", pretrained: str = "laion2b_s34b_b79k", device: str = "cuda"):
    """Load OpenCLIP model and preprocessing transforms."""

    print(f"Loading OpenCLIP: {model_name} (pretrained={pretrained})")
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained
    )
    model = model.to(device)
    model.eval()

    # Get embedding dimension
    with torch.no_grad():
        dummy = torch.randn(1, 3, 224, 224).to(device)
        emb = model.encode_image(dummy)
        embed_dim = emb.shape[-1]

    return model, preprocess, embed_dim


def extract_embeddings(
    model, dataloader, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract normalized image embeddings from the model."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            emb = model.encode_image(images)

            # L2 normalize
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_linear_probe(
    train_X: torch.Tensor,
    train_y: torch.Tensor,
    val_X: torch.Tensor,
    val_y: torch.Tensor,
    num_classes: int,
    num_epochs: int = 100,
    lr: float = 0.01,
    weight_decay: float = 1e-4,
    device: str = "cuda",
    verbose: bool = True,
) -> Tuple[float, Dict[int, float], nn.Module, List[Dict]]:
    """Train a linear classifier on extracted embeddings."""

    classifier = LinearProbeClassifier(train_X.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Move data to device
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    best_val_acc = 0.0
    best_epoch = 0
    best_per_class = None
    best_classifier_state = None
    history = []

    for epoch in range(num_epochs):
        # Training step
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        train_acc = (logits.argmax(dim=1) == train_y).float().mean().item()

        # Validation step
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_X)
            val_loss = criterion(val_logits, val_y)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

            # Per-class accuracy
            per_class = {}
            for c in range(num_classes):
                mask = val_y == c
                if mask.sum() > 0:
                    per_class[c] = (val_preds[mask] == c).float().mean().item()
                else:
                    per_class[c] = 0.0

        history.append({
            "epoch": epoch + 1,
            "train_loss": loss.item(),
            "train_acc": train_acc,
            "val_loss": val_loss.item(),
            "val_acc": val_acc,
        })

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
            best_per_class = per_class.copy()
            best_classifier_state = classifier.state_dict().copy()

        if verbose and ((epoch + 1) % 10 == 0 or epoch == 0):
            print(f"  Epoch {epoch+1:3d}: train_loss={loss.item():.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}")

    # Load best model
    classifier.load_state_dict(best_classifier_state)

    return best_val_acc, best_per_class, classifier, history


def main():
    parser = argparse.ArgumentParser(description="OpenCLIP Linear Probe Training")

    # Data paths
    parser.add_argument("--data-dir", type=str, required=True,
                        help="Directory containing train.json, val.json, labels.json")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Output directory for results and checkpoints")

    # Model settings
    parser.add_argument("--model-name", type=str, default="ViT-B-32",
                        help="OpenCLIP model variant (default: ViT-B-32)")
    parser.add_argument("--pretrained", type=str, default="laion2b_s34b_b79k",
                        help="Pretrained weights (default: laion2b_s34b_b79k)")

    # Training settings
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate (default: 0.01)")
    parser.add_argument("--weight-decay", type=float, default=1e-4,
                        help="Weight decay (default: 1e-4)")
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for embedding extraction (default: 64)")

    # Other
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=4)

    args = parser.parse_args()

    # Setup
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 70)
    print("OPENCLIP LINEAR PROBE TRAINING")
    print("=" * 70)
    print(f"Model: {args.model_name} (pretrained={args.pretrained})")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {device}")
    print(f"Seed: {args.seed}")
    print("-" * 70)

    # Load dataset config
    config_path = Path(args.data_dir) / "dataset_config.json"
    with open(config_path) as f:
        dataset_config = json.load(f)

    labels_path = Path(args.data_dir) / "labels.json"
    with open(labels_path) as f:
        names = json.load(f)

    num_classes = len(names)
    print(f"Dataset mode: {dataset_config.get('mode', 'unknown')}")
    print(f"Number of classes: {num_classes}")
    print(f"Total training samples: {dataset_config.get('total_train', 'unknown')}")
    print(f"Total validation samples: {dataset_config.get('total_val', 'unknown')}")
    print(f"Random baseline: {100/num_classes:.2f}%")
    print("-" * 70)

    # Load OpenCLIP model
    print("\nLoading OpenCLIP model...")
    start_time = time.time()
    model, preprocess, embed_dim = load_openclip_model(
        model_name=args.model_name,
        pretrained=args.pretrained,
        device=device
    )
    print(f"Model loaded in {time.time()-start_time:.1f}s")
    print(f"Embedding dimension: {embed_dim}")

    # Freeze model
    for param in model.parameters():
        param.requires_grad = False

    # Create data loaders
    print("\nLoading datasets...")
    train_dataset = FaceDataset(
        str(Path(args.data_dir) / "train.json"),
        transform=preprocess
    )
    val_dataset = FaceDataset(
        str(Path(args.data_dir) / "val.json"),
        transform=preprocess
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Extract embeddings (one-time cost)
    print("\nExtracting embeddings...")
    start_time = time.time()
    train_X, train_y = extract_embeddings(model, train_loader, device)
    val_X, val_y = extract_embeddings(model, val_loader, device)
    embed_time = time.time() - start_time
    print(f"Embeddings extracted in {embed_time:.1f}s")
    print(f"Train embeddings shape: {train_X.shape}")
    print(f"Val embeddings shape: {val_X.shape}")

    # Train linear probe
    print(f"\nTraining linear probe for {args.epochs} epochs...")
    print("-" * 70)
    start_time = time.time()
    best_acc, per_class_acc, classifier, history = train_linear_probe(
        train_X, train_y, val_X, val_y,
        num_classes=num_classes,
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device,
        verbose=True
    )
    train_time = time.time() - start_time
    print("-" * 70)
    print(f"Training completed in {train_time:.1f}s")

    # Create results dataframe
    results = []
    for idx, name in enumerate(names):
        acc = per_class_acc.get(idx, 0)
        stats = dataset_config.get("per_name_stats", {}).get(name, {})
        results.append({
            "name": name,
            "accuracy": acc,
            "train_samples": stats.get("train", 0),
            "val_samples": stats.get("val", 0),
            "above_random": acc - 1/num_classes
        })

    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)

    # Print summary
    print("\n" + "=" * 70)
    print("OPENCLIP LINEAR PROBE RESULTS")
    print("=" * 70)
    print(f"Best validation accuracy: {100*best_acc:.2f}%")
    print(f"Random baseline: {100/num_classes:.2f}%")
    print(f"Improvement over random: +{100*(best_acc - 1/num_classes):.2f}%")
    print(f"Relative improvement: +{100*(best_acc - 1/num_classes)/(1/num_classes):.1f}%")

    print(f"\nTop 10 names by accuracy:")
    print("-" * 50)
    for _, row in df.head(10).iterrows():
        bar = "*" * int(row['accuracy'] * 20)
        print(f"  {row['name']:15s} {100*row['accuracy']:5.1f}% {bar}")

    print(f"\nBottom 10 names:")
    print("-" * 50)
    for _, row in df.tail(10).iterrows():
        bar = "*" * int(row['accuracy'] * 20)
        print(f"  {row['name']:15s} {100*row['accuracy']:5.1f}% {bar}")

    # Save results
    output_path = Path(args.output_dir)

    # Save rankings CSV
    df.to_csv(output_path / "rankings.csv", index=False)

    # Save training history
    with open(output_path / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Save embeddings for later analysis
    np.save(output_path / "train_embeddings.npy", train_X.numpy())
    np.save(output_path / "train_labels.npy", train_y.numpy())
    np.save(output_path / "val_embeddings.npy", val_X.numpy())
    np.save(output_path / "val_labels.npy", val_y.numpy())

    # Save classifier checkpoint
    torch.save({
        "model_state_dict": classifier.state_dict(),
        "names": names,
        "num_classes": num_classes,
        "embed_dim": embed_dim,
        "best_val_acc": best_acc,
    }, output_path / "linear_probe_checkpoint.pt")

    # Save full experiment config
    experiment_config = {
        "timestamp": datetime.now().isoformat(),
        "model": {
            "name": args.model_name,
            "pretrained": args.pretrained,
            "embed_dim": embed_dim,
            "type": "OpenCLIP",
        },
        "training": {
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "data": {
            "data_dir": args.data_dir,
            "mode": dataset_config.get("mode", "unknown"),
            "num_classes": num_classes,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
            "holdout_checksum": dataset_config.get("holdout_checksum", "unknown"),
        },
        "results": {
            "best_val_acc": best_acc,
            "random_baseline": 1/num_classes,
            "improvement_over_random": best_acc - 1/num_classes,
            "relative_improvement": (best_acc - 1/num_classes)/(1/num_classes),
            "embed_time_seconds": embed_time,
            "train_time_seconds": train_time,
        },
        "paths": {
            "train_data": str(Path(args.data_dir) / "train.json"),
            "val_data": str(Path(args.data_dir) / "val.json"),
            "holdout_manifest": dataset_config.get("source_manifest_dir", "unknown") + "/holdout_manifest.json",
        }
    }
    with open(output_path / "experiment_config.json", "w") as f:
        json.dump(experiment_config, f, indent=2)

    print(f"\nResults saved to: {output_path}")
    print(f"  - rankings.csv: Per-name accuracy rankings")
    print(f"  - training_history.json: Epoch-by-epoch metrics")
    print(f"  - linear_probe_checkpoint.pt: Trained classifier")
    print(f"  - experiment_config.json: Full reproducibility info")
    print(f"  - *_embeddings.npy: Cached embeddings for analysis")
    print("=" * 70)

    return {
        "best_val_acc": best_acc,
        "random_baseline": 1/num_classes,
        "improvement": best_acc - 1/num_classes,
    }


if __name__ == "__main__":
    main()
