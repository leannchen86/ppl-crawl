"""
Phase 3: ArcFace Embeddings for Face-Name Association

Replace CLIP embeddings with ArcFace (face-specific) embeddings.
ArcFace is trained with angular margin loss specifically designed for
face recognition - should have better inter-class separation.

Usage:
    python phase3_arcface.py --num-names 30 --epochs 50
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
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from collections import Counter
from PIL import Image
import cv2

# InsightFace for ArcFace embeddings
from insightface.app import FaceAnalysis


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FaceChipDataset(Dataset):
    """Dataset that loads face chips for ArcFace embedding extraction."""

    def __init__(self, index_dir, names, split="train", train_ratio=0.8,
                 seed=42, max_per_name=None):
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
            random.shuffle(good_images)

            # Split
            split_idx = int(len(good_images) * train_ratio)
            if split == "train":
                selected = good_images[:split_idx]
            else:
                selected = good_images[split_idx:]

            # Optional limit per name
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
        return path, label


def extract_arcface_embeddings(app, dataset, batch_size=32):
    """Extract ArcFace embeddings for all images in dataset."""
    all_embeddings = []
    all_labels = []
    failed_count = 0

    # Process in batches for progress tracking
    paths_labels = [(dataset.samples[i][0], dataset.samples[i][1])
                    for i in range(len(dataset))]

    for i in tqdm(range(0, len(paths_labels), batch_size), desc="Extracting ArcFace embeddings"):
        batch = paths_labels[i:i+batch_size]

        for path, label in batch:
            # Load image (ArcFace expects BGR)
            img = cv2.imread(path)
            if img is None:
                failed_count += 1
                continue

            # Get face embedding
            # The face chip is already cropped, but ArcFace still needs to detect
            # We'll use get() which returns faces with embeddings
            faces = app.get(img)

            if len(faces) == 0:
                # If no face detected in chip, try resizing
                # Sometimes the face is too large/small for detection
                failed_count += 1
                continue

            # Take the first (should be only) face
            embedding = faces[0].embedding
            all_embeddings.append(embedding)
            all_labels.append(label)

    if failed_count > 0:
        print(f"  Warning: Failed to extract embeddings for {failed_count} images")

    embeddings = np.array(all_embeddings)
    labels = np.array(all_labels)

    return torch.tensor(embeddings, dtype=torch.float32), torch.tensor(labels, dtype=torch.long)


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


def train_linear_probe(train_X, train_y, val_X, val_y, num_classes,
                       num_epochs=50, lr=0.01, weight_decay=1e-4, device="cuda"):
    """Train linear probe on embeddings."""
    # Normalize embeddings (ArcFace embeddings are usually already normalized, but ensure)
    train_X = train_X / train_X.norm(dim=-1, keepdim=True)
    val_X = val_X / val_X.norm(dim=-1, keepdim=True)

    classifier = nn.Linear(train_X.shape[1], num_classes).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

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

                # Per-class accuracy (recall)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=200)
    parser.add_argument("--max-per-name", type=int, default=500)
    parser.add_argument("--index-dir",
                        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001")
    parser.add_argument("--output-dir", default="./results/phase3_arcface")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print("PHASE 3: ARCFACE EMBEDDINGS")
    print("="*70)
    print("Using InsightFace ArcFace model for face-specific embeddings")
    print("Expected: Better inter-class separation than CLIP")
    print("="*70 + "\n")

    # Initialize ArcFace model
    print("Loading ArcFace model...")
    app = FaceAnalysis(name='buffalo_l', providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    app.prepare(ctx_id=0, det_size=(512, 512))
    print("ArcFace model loaded!\n")

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

    # Create datasets
    train_dataset = FaceChipDataset(
        args.index_dir, names, split="train",
        seed=args.seed, max_per_name=args.max_per_name
    )
    val_dataset = FaceChipDataset(
        args.index_dir, names, split="val",
        seed=args.seed, max_per_name=args.max_per_name
    )

    print(f"\nTraining samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Random baseline: {100/len(names):.1f}%")

    # Extract ArcFace embeddings
    print("\nExtracting ArcFace embeddings (this may take a while)...")
    train_X, train_y = extract_arcface_embeddings(app, train_dataset)
    val_X, val_y = extract_arcface_embeddings(app, val_dataset)

    print(f"\nTrain embeddings shape: {train_X.shape}")
    print(f"Val embeddings shape: {val_X.shape}")

    # Train linear probe
    print(f"\nTraining linear probe for {args.epochs} epochs...")
    best_acc, per_class_acc, predictions, true_labels = train_linear_probe(
        train_X, train_y, val_X, val_y,
        num_classes=len(names),
        num_epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=device
    )

    # Analyze prediction bias
    pred_counts = Counter(predictions.numpy())
    expected_count = len(predictions) / len(names)
    prediction_ratios = {names[i]: pred_counts.get(i, 0) / expected_count
                        for i in range(len(names))}

    # Calculate prediction CV
    pred_count_values = [pred_counts.get(i, 0) for i in range(len(names))]
    pred_mean = np.mean(pred_count_values)
    pred_std = np.std(pred_count_values)
    pred_cv = pred_std / pred_mean if pred_mean > 0 else 0

    # Create results dataframe
    results = []
    for idx, name in enumerate(names):
        acc = per_class_acc.get(idx, 0)
        pred_count = pred_counts.get(idx, 0)
        support = (true_labels == idx).sum().item()
        results.append({
            "name": name,
            "recall": acc,
            "predicted_count": pred_count,
            "prediction_ratio": prediction_ratios[name],
            "support": support
        })

    df = pd.DataFrame(results)
    df = df.sort_values("recall", ascending=False)

    # Save results
    csv_path = os.path.join(args.output_dir, "results.csv")
    df.to_csv(csv_path, index=False)

    np.save(os.path.join(args.output_dir, "predictions.npy"), predictions.numpy())
    np.save(os.path.join(args.output_dir, "true_labels.npy"), true_labels.numpy())
    np.save(os.path.join(args.output_dir, "train_embeddings.npy"), train_X.numpy())
    np.save(os.path.join(args.output_dir, "val_embeddings.npy"), val_X.numpy())
    with open(os.path.join(args.output_dir, "names.json"), "w") as f:
        json.dump(names, f)
    with open(os.path.join(args.output_dir, "config.json"), "w") as f:
        json.dump(vars(args), f, indent=2)

    # Print summary
    print("\n" + "="*70)
    print("PHASE 3 RESULTS: ARCFACE EMBEDDINGS")
    print("="*70)
    print(f"\nOverall accuracy: {100*best_acc:.1f}%")
    print(f"Random baseline: {100/len(names):.1f}%")
    print(f"Improvement over random: +{100*(best_acc - 1/len(names)):.1f}%")

    print(f"\nPrediction Skew Metrics:")
    print(f"  Prediction CV: {pred_cv:.3f}")
    print(f"  (CLIP Phase 2A CV was ~0.41 - lower is better)")

    print(f"\n📊 Top 10 Names (Highest Recall):")
    print("-"*55)
    for i, row in df.head(10).iterrows():
        ratio = row['prediction_ratio']
        bar = "█" * int(row['recall'] * 20)
        print(f"  {row['name']:12s} {100*row['recall']:5.1f}% "
              f"(pred={row['predicted_count']:3.0f}, ratio={ratio:.2f}x) {bar}")

    print(f"\n📊 Bottom 5 Names (Lowest Recall):")
    print("-"*55)
    for i, row in df.tail(5).iterrows():
        ratio = row['prediction_ratio']
        bar = "█" * int(row['recall'] * 20)
        print(f"  {row['name']:12s} {100*row['recall']:5.1f}% "
              f"(pred={row['predicted_count']:3.0f}, ratio={ratio:.2f}x) {bar}")

    # Compare with CLIP
    print("\n" + "="*70)
    print("COMPARISON: ARCFACE vs CLIP")
    print("="*70)
    print(f"\n{'Metric':<25} {'CLIP (Phase 2A)':<20} {'ArcFace (Phase 3)':<20}")
    print("-"*65)
    print(f"{'Accuracy':<25} {'12.6%':<20} {f'{100*best_acc:.1f}%':<20}")
    print(f"{'Prediction CV':<25} {'0.41':<20} {f'{pred_cv:.3f}':<20}")
    print(f"{'Embedding dim':<25} {'512':<20} {f'{train_X.shape[1]}':<20}")

    print(f"\n📁 Results saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()
