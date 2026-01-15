"""
Phase 2B: Quality Filtering + Focal Loss

This script addresses two confounds identified in Phase 2A:
1. Blur/sharpness confound (r≈0.43-0.45 with recall/predicted_count)
2. Brightness confound (r≈0.54 with precision)

Solutions:
- Quality filtering: Remove bottom 25% blur + extreme brightness outliers per name
- Focal loss: Reduce dominance of easy/over-predicted classes

Usage:
    python phase2b_quality_filtered.py --num-names 30 --epochs 50 --use-focal-loss
"""
import argparse
import json
import os
import random
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from collections import defaultdict, Counter

from clip_dataset import create_name_gender_mapping


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_quality_metrics(quality_csv_path):
    """Load quality metrics and return per-image quality data."""
    df = pd.read_csv(quality_csv_path)
    return df


def filter_images_by_quality(quality_df, blur_percentile=25, brightness_std_threshold=2.0):
    """
    Filter out poor quality images per name.

    Args:
        quality_df: DataFrame with columns [name, path, blur_score, brightness, ...]
        blur_percentile: Remove bottom X% by blur (higher blur = blurrier)
        brightness_std_threshold: Remove images beyond X std from mean brightness

    Returns:
        set of valid image paths
    """
    valid_paths = set()

    # Group by name
    for name, group in quality_df.groupby('name'):
        # Calculate blur threshold (remove bottom X%)
        blur_threshold = group['blur_score'].quantile(blur_percentile / 100.0)

        # Calculate brightness outlier bounds
        bright_mean = group['brightness'].mean()
        bright_std = group['brightness'].std()
        bright_lower = bright_mean - brightness_std_threshold * bright_std
        bright_upper = bright_mean + brightness_std_threshold * bright_std

        # Filter
        filtered = group[
            (group['blur_score'] >= blur_threshold) &  # Keep sharp images (low blur)
            (group['brightness'] >= bright_lower) &
            (group['brightness'] <= bright_upper)
        ]

        valid_paths.update(filtered['path'].tolist())

        print(f"  {name:12s}: {len(group):4d} → {len(filtered):4d} images "
              f"({100*len(filtered)/len(group):.1f}% kept)")

    return valid_paths


class FocalLoss(nn.Module):
    """
    Focal Loss: Reduces contribution from easy examples, focuses on hard negatives.

    FL(p_t) = -α(1 - p_t)^γ * log(p_t)

    Where p_t is the probability of the correct class.
    - α: Weighting factor (typically 0.25)
    - γ: Focusing parameter (typically 2.0)
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: Logits (B, num_classes)
            targets: Class labels (B,)
        """
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        p_t = torch.exp(-ce_loss)  # Probability of correct class
        focal_loss = self.alpha * (1 - p_t) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class QualityFilteredDataset(torch.utils.data.Dataset):
    """Dataset with quality filtering."""

    def __init__(self, index_dir, names, transform, valid_paths,
                 split="train", train_ratio=0.8, seed=42, max_per_name=None):
        self.transform = transform
        self.samples = []  # (path, name_idx)
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}

        random.seed(seed)

        for name in names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                continue

            with open(index_path) as f:
                data = json.load(f)

            good_images = data.get("good", [])

            # FILTER BY QUALITY
            if valid_paths:
                good_images = [p for p in good_images if p in valid_paths]

            random.shuffle(good_images)

            # Split
            split_idx = int(len(good_images) * train_ratio)
            if split == "train":
                selected = good_images[:split_idx]
            else:
                selected = good_images[split_idx:]

            # Optional limit per name (for balanced datasets)
            if max_per_name and len(selected) > max_per_name:
                selected = selected[:max_per_name]

            for path in selected:
                if os.path.exists(path):
                    self.samples.append((path, self.name_to_idx[name]))

        random.shuffle(self.samples)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def discover_names(index_dir: str, min_samples: int = 100):
    """Find all names with enough samples."""
    names_with_counts = []

    pattern = os.path.join(index_dir, "index_*.json")
    for filepath in glob.glob(pattern):
        name = os.path.basename(filepath).replace("index_", "").replace(".json", "")

        try:
            with open(filepath) as f:
                data = json.load(f)
            count = data.get("counts", {}).get("good", 0)
            if count >= min_samples:
                names_with_counts.append((name, count))
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")

    names_with_counts.sort(key=lambda x: -x[1])
    return names_with_counts


def extract_embeddings(model, dataloader, device):
    """Extract CLIP embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []

    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, labels in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)

    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_linear_probe(train_X, train_y, val_X, val_y, num_classes,
                       num_epochs=50, lr=0.01, weight_decay=1e-4,
                       use_focal_loss=False, device="cuda"):
    """Train linear probe with optional focal loss."""
    classifier = nn.Linear(train_X.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)

    if use_focal_loss:
        criterion = FocalLoss(alpha=0.25, gamma=2.0)
        print("  Using Focal Loss (α=0.25, γ=2.0)")
    else:
        criterion = nn.CrossEntropyLoss()
        print("  Using Cross-Entropy Loss")

    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)

    best_val_acc = 0
    best_per_class = None
    best_preds = None

    for epoch in range(num_epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()

        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_X)
            val_preds = val_logits.argmax(dim=1)
            val_acc = (val_preds == val_y).float().mean().item()

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_preds = val_preds.cpu()

                # Per-class accuracy
                per_class = {}
                for c in range(num_classes):
                    mask = val_y == c
                    if mask.sum() > 0:
                        per_class[c] = (val_preds[mask] == c).float().mean().item()
                    else:
                        per_class[c] = 0.0
                best_per_class = per_class

        if (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1}: val_acc={val_acc:.4f}")

    return best_val_acc, best_per_class, best_preds, val_y.cpu()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-names", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--max-per-name", type=int, default=500)
    parser.add_argument("--blur-filter-percentile", type=int, default=25,
                        help="Remove bottom X% by blur score (higher=blurrier)")
    parser.add_argument("--brightness-std-threshold", type=float, default=2.0,
                        help="Remove images beyond X std from mean brightness")
    parser.add_argument("--use-focal-loss", action="store_true",
                        help="Use focal loss instead of cross-entropy")
    parser.add_argument("--index-dir",
                        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001")
    parser.add_argument("--quality-csv",
                        default="/home/leann/face-detection/results/confound_analysis/quality_metrics.csv")
    parser.add_argument("--output-dir",
                        default="./results/phase2b_quality_filtered")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 2B: QUALITY FILTERING + FOCAL LOSS")
    print("="*70)
    print(f"Quality filtering:")
    print(f"  - Remove bottom {args.blur_filter_percentile}% by blur score")
    print(f"  - Remove brightness outliers (>{args.brightness_std_threshold} std)")
    print(f"Loss function: {'Focal Loss' if args.use_focal_loss else 'Cross-Entropy'}")
    print("="*70 + "\n")

    # Load quality metrics
    print("Loading quality metrics...")
    quality_df = load_quality_metrics(args.quality_csv)
    print(f"Loaded quality data for {len(quality_df)} images\n")

    # Filter images by quality
    print("Filtering images by quality...")
    valid_paths = filter_images_by_quality(
        quality_df,
        blur_percentile=args.blur_filter_percentile,
        brightness_std_threshold=args.brightness_std_threshold
    )
    print(f"\nTotal images after filtering: {len(valid_paths)}\n")

    # Discover available names
    print("Discovering names...")
    all_names = discover_names(args.index_dir, min_samples=args.min_samples)
    print(f"Found {len(all_names)} names with >= {args.min_samples} samples")

    if len(all_names) < args.num_names:
        print(f"Warning: Only {len(all_names)} names available, using all")
        args.num_names = len(all_names)

    selected = all_names[:args.num_names]
    names = [n for n, c in selected]

    print(f"\nSelected {len(names)} names:")
    for name, count in selected[:10]:
        print(f"  {name}: {count} samples")
    if len(selected) > 10:
        print(f"  ... and {len(selected) - 10} more")

    # Load CLIP
    print("\nLoading CLIP ViT-B-32...")
    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    # Create datasets with quality filtering
    train_dataset = QualityFilteredDataset(
        args.index_dir, names, preprocess, valid_paths,
        split="train", seed=args.seed, max_per_name=args.max_per_name
    )
    val_dataset = QualityFilteredDataset(
        args.index_dir, names, preprocess, valid_paths,
        split="val", seed=args.seed, max_per_name=args.max_per_name
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Random baseline: {100/len(names):.1f}%")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=False, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    # Extract embeddings
    print("\nExtracting embeddings...")
    train_X, train_y = extract_embeddings(model, train_loader, device)
    val_X, val_y = extract_embeddings(model, val_loader, device)

    # Train linear probe
    print(f"\nTraining linear probe for {args.epochs} epochs...")
    best_acc, per_class_acc, predictions, true_labels = train_linear_probe(
        train_X, train_y, val_X, val_y,
        num_classes=len(names),
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        use_focal_loss=args.use_focal_loss,
        device=device
    )

    # Analyze prediction bias
    pred_counts = Counter(predictions.numpy())
    expected_count = len(predictions) / len(names)
    prediction_ratios = {names[i]: pred_counts.get(i, 0) / expected_count
                        for i in range(len(names))}

    # Calculate prediction CV (coefficient of variation)
    pred_count_values = [pred_counts.get(i, 0) for i in range(len(names))]
    pred_mean = np.mean(pred_count_values)
    pred_std = np.std(pred_count_values)
    pred_cv = pred_std / pred_mean if pred_mean > 0 else 0

    # Create results dataframe
    results = []
    for idx, name in enumerate(names):
        acc = per_class_acc.get(idx, 0)
        pred_count = pred_counts.get(idx, 0)
        results.append({
            "name": name,
            "recall": acc,
            "predicted_count": pred_count,
            "prediction_ratio": prediction_ratios[name],
            "support": (true_labels == idx).sum().item()
        })

    df = pd.DataFrame(results)
    df = df.sort_values("recall", ascending=False)

    # Save results
    csv_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    np.save(os.path.join(args.output_dir, "predictions.npy"), predictions.numpy())
    np.save(os.path.join(args.output_dir, "true_labels.npy"), true_labels.numpy())
    with open(os.path.join(args.output_dir, "names.json"), "w") as f:
        json.dump(names, f)

    # Save config
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("PHASE 2B RESULTS")
    print("="*70)
    print(f"\nOverall accuracy: {100*best_acc:.1f}%")
    print(f"Random baseline: {100/len(names):.1f}%")
    print(f"Improvement: +{100*(best_acc - 1/len(names)):.1f}%")

    print(f"\nPrediction Skew Metrics:")
    print(f"  Prediction CV: {pred_cv:.3f}")
    print(f"  Expected: {expected_count:.1f} predictions/name")
    print(f"  Mean: {pred_mean:.1f}, Std: {pred_std:.1f}")

    print(f"\n📊 Top 5 Names (Highest Recall):")
    print("-"*50)
    for i, row in df.head(5).iterrows():
        ratio = row['prediction_ratio']
        bar = "█" * int(row['recall'] * 20)
        print(f"  {row['name']:12s} {100*row['recall']:5.1f}% "
              f"(pred={row['predicted_count']:3.0f}, ratio={ratio:.2f}x) {bar}")

    print(f"\n📊 Bottom 5 Names (Lowest Recall):")
    print("-"*50)
    for i, row in df.tail(5).iterrows():
        ratio = row['prediction_ratio']
        bar = "█" * int(row['recall'] * 20)
        print(f"  {row['name']:12s} {100*row['recall']:5.1f}% "
              f"(pred={row['predicted_count']:3.0f}, ratio={ratio:.2f}x) {bar}")

    print(f"\n📁 Results saved to: {args.output_dir}/")
    print(f"   - results.csv: Per-name metrics")
    print(f"   - config.json: Experiment configuration")


if __name__ == "__main__":
    main()
