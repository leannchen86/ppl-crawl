"""
Scale-Up Test (30-way by default): linear probe across many names and rank by accuracy.

This script:
1. Discovers all available names from index files
2. Tests linear probe on N names
3. Ranks names by "vibe clarity" (per-class accuracy)
4. Outputs rankings to CSV

Usage:
    python scripts/clip/clip_probe_30way_scaleup.py --epochs 50
    python scripts/clip/clip_probe_30way_scaleup.py --epochs 50 --balanced  # Equal samples per name

Notes:
- This script is named "30way" because the default benchmark is 30 classes.
- You *can* override `--num-names`, but the filename will no longer match the run.
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
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from collections import defaultdict

from clip_dataset import FaceNameDataset, create_name_gender_mapping
from index_utils import ImageSource, resolve_good_images


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


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
    
    # Sort by count descending
    names_with_counts.sort(key=lambda x: -x[1])
    return names_with_counts


def get_gender(name: str, name_to_gender: dict) -> str:
    """Get gender for a name, with fallback."""
    return name_to_gender.get(name, "unknown")


class MultiNameDataset(torch.utils.data.Dataset):
    """Dataset for multiple names."""
    
    def __init__(self, index_dir, names, transform, split="train", 
                 train_ratio=0.8, seed=42, max_per_name=None, image_source: ImageSource = "chips"):
        self.transform = transform
        self.samples = []  # (path, name_idx)
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        self.image_source = image_source
        
        random.seed(seed)
        
        for name in names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                continue
                
            with open(index_path) as f:
                data = json.load(f)
            
            good_images = resolve_good_images(data, image_source=self.image_source)
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
                       num_epochs=50, lr=0.01, device="cuda"):
    """Train linear probe and return per-class accuracies."""
    classifier = nn.Linear(train_X.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-samples", type=int, default=200,
                        help="Minimum samples per name to include")
    parser.add_argument("--balanced", action="store_true",
                        help="Use equal samples per name")
    parser.add_argument("--max-per-name", type=int, default=500,
                        help="Max samples per name (for balanced mode)")
    parser.add_argument(
        "--index-dir",
        default="/home/leann/face-detection/data/index_files",
    )
    parser.add_argument(
        "--image-source",
        choices=["chips", "original"],
        default="chips",
        help="Choose which images to train/evaluate on using the same index files: "
        "'chips' uses index['good']; 'original' uses index['meta'][chip].src_path.",
    )
    parser.add_argument("--output-dir", default="./scale_up_results")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    if args.num_names != 30:
        print(
            f"Note: you set --num-names={args.num_names}. "
            "This script is named '30way' because the default benchmark is 30 classes."
        )
    
    # Discover available names
    print("Discovering names...")
    all_names = discover_names(args.index_dir, min_samples=args.min_samples)
    print(f"Found {len(all_names)} names with >= {args.min_samples} samples")
    
    if len(all_names) < args.num_names:
        print(f"Warning: Only {len(all_names)} names available, using all")
        args.num_names = len(all_names)
    
    # Select names (take top N by sample count for reliability)
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
    
    # Create datasets
    max_per = args.max_per_name if args.balanced else None
    
    train_dataset = MultiNameDataset(
        args.index_dir, names, preprocess, 
        split="train", seed=args.seed, max_per_name=max_per, image_source=args.image_source  # type: ignore[arg-type]
    )
    val_dataset = MultiNameDataset(
        args.index_dir, names, preprocess,
        split="val", seed=args.seed, max_per_name=max_per, image_source=args.image_source  # type: ignore[arg-type]
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
        device=device
    )
    
    # Create results dataframe
    results = []
    for idx, name in enumerate(names):
        acc = per_class_acc.get(idx, 0)
        count = dict(selected).get(name, 0)
        results.append({
            "name": name,
            "accuracy": acc,
            "sample_count": count,
            "above_random": acc - 1/len(names)
        })
    
    df = pd.DataFrame(results)
    df = df.sort_values("accuracy", ascending=False)
    
    # Save results
    csv_path = os.path.join(args.output_dir, "rankings.csv")
    df.to_csv(csv_path, index=False)
    
    # Also save predictions for visualization
    np.save(os.path.join(args.output_dir, "predictions.npy"), predictions.numpy())
    np.save(os.path.join(args.output_dir, "true_labels.npy"), true_labels.numpy())
    np.save(os.path.join(args.output_dir, "val_embeddings.npy"), val_X.numpy())
    with open(os.path.join(args.output_dir, "names.json"), "w") as f:
        json.dump(names, f)
    
    # Print summary
    print("\n" + "="*60)
    print("SCALE-UP TEST RESULTS")
    print("="*60)
    print(f"\nOverall accuracy: {100*best_acc:.1f}%")
    print(f"Random baseline: {100/len(names):.1f}%")
    print(f"Improvement: +{100*(best_acc - 1/len(names)):.1f}%")
    
    print(f"\n📊 Top 10 Names by 'Vibe Clarity':")
    print("-"*40)
    for i, row in df.head(10).iterrows():
        bar = "█" * int(row['accuracy'] * 20)
        print(f"  {row['name']:12s} {100*row['accuracy']:5.1f}% {bar}")
    
    print(f"\n📊 Bottom 10 Names (Weakest Vibes):")
    print("-"*40)
    for i, row in df.tail(10).iterrows():
        bar = "█" * int(row['accuracy'] * 20)
        print(f"  {row['name']:12s} {100*row['accuracy']:5.1f}% {bar}")
    
    print(f"\n📁 Results saved to: {args.output_dir}/")
    print(f"   - rankings.csv: Full rankings")
    print(f"   - predictions.npy, true_labels.npy: For visualization")


if __name__ == "__main__":
    main()

