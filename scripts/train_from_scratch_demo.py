"""
Educational Demo: Training a small model from scratch vs fine-tuning.

This demonstrates WHY training from scratch fails with limited data,
and why fine-tuning pretrained models is the standard practice.

Usage:
    python train_from_scratch_demo.py --mode scratch   # Train CNN from scratch
    python train_from_scratch_demo.py --mode finetune  # Fine-tune pretrained ViT
"""
import argparse
import json
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm
from PIL import Image
import glob


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class FaceDataset(Dataset):
    """Simple dataset for face images."""

    def __init__(self, index_dir, names, transform, split="train",
                 train_ratio=0.8, seed=42, max_per_name=500):
        self.transform = transform
        self.samples = []
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

            split_idx = int(len(good_images) * train_ratio)
            if split == "train":
                selected = good_images[:split_idx]
            else:
                selected = good_images[split_idx:]

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
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


# ============================================================================
# Option 1: Small CNN from scratch (will work okay with regularization)
# ============================================================================

class SmallCNN(nn.Module):
    """A small CNN appropriate for ~15K images."""

    def __init__(self, num_classes=30):
        super().__init__()
        # ~500K parameters (vs ViT's 86M)
        self.features = nn.Sequential(
            # 224 -> 112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # 112 -> 56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.1),

            # 56 -> 28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # 28 -> 14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.2),

            # 14 -> 7
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Dropout2d(0.3),

            # Global average pooling
            nn.AdaptiveAvgPool2d(1),
        )

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ============================================================================
# Option 2: Fine-tune pretrained model (the RIGHT way)
# ============================================================================

def get_pretrained_model(num_classes=30):
    """Load pretrained ResNet and replace classifier."""
    from torchvision.models import resnet18, ResNet18_Weights

    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    # Freeze early layers (they learned good features already)
    for param in list(model.parameters())[:-10]:
        param.requires_grad = False

    # Replace final layer
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training", leave=False):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def discover_names(index_dir, min_samples=200):
    """Find all names with enough samples."""
    names_with_counts = []

    for filepath in glob.glob(os.path.join(index_dir, "index_*.json")):
        name = os.path.basename(filepath).replace("index_", "").replace(".json", "")
        try:
            with open(filepath) as f:
                data = json.load(f)
            count = data.get("counts", {}).get("good", 0)
            if count >= min_samples:
                names_with_counts.append((name, count))
        except:
            pass

    names_with_counts.sort(key=lambda x: -x[1])
    return names_with_counts


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["scratch", "finetune"], default="scratch",
                        help="scratch: train CNN from scratch, finetune: fine-tune pretrained")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--num-names", type=int, default=30)
    parser.add_argument("--index-dir",
                        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001")
    parser.add_argument("--output-dir", default="./results/train_from_scratch")
    parser.add_argument("--max-per-name", type=int, default=500,
                        help="Max samples per name (0 for unlimited)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)

    print("\n" + "="*70)
    print(f"TRAINING FROM {'SCRATCH' if args.mode == 'scratch' else 'PRETRAINED'}")
    print("="*70)

    # Discover names
    all_names = discover_names(args.index_dir)[:args.num_names]
    names = [n for n, c in all_names]
    print(f"Using {len(names)} names")

    # Data transforms
    if args.mode == "scratch":
        # More augmentation needed when training from scratch
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        # Less augmentation needed with pretrained features
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Datasets
    max_samples = args.max_per_name if args.max_per_name > 0 else None
    train_dataset = FaceDataset(args.index_dir, names, train_transform, "train", max_per_name=max_samples)
    val_dataset = FaceDataset(args.index_dir, names, val_transform, "val", max_per_name=max_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=4, pin_memory=True)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Random baseline: {100/len(names):.1f}%")

    # Model
    if args.mode == "scratch":
        model = SmallCNN(num_classes=len(names))
        print(f"\nModel: SmallCNN (training from scratch)")
        print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    else:
        model = get_pretrained_model(num_classes=len(names))
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        print(f"\nModel: ResNet18 (pretrained, fine-tuning)")
        print(f"Total parameters: {total:,}")
        print(f"Trainable parameters: {trainable:,}")

    model = model.to(device)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # Training loop
    print(f"\nTraining for {args.epochs} epochs...")
    print("-"*70)

    best_val_acc = 0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    for epoch in range(args.epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{args.output_dir}/best_model.pth")

        # Check for overfitting
        overfit_gap = train_acc - val_acc
        overfit_warning = " ⚠️ OVERFITTING!" if overfit_gap > 20 else ""

        print(f"Epoch {epoch+1:2d}/{args.epochs}: "
              f"train_loss={train_loss:.4f}, train_acc={train_acc:.1f}%, "
              f"val_loss={val_loss:.4f}, val_acc={val_acc:.1f}%{overfit_warning}")

    # Save history
    with open(f"{args.output_dir}/history.json", "w") as f:
        json.dump(history, f, indent=2)

    # Final results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    print(f"Best validation accuracy: {best_val_acc:.1f}%")
    print(f"Random baseline: {100/len(names):.1f}%")
    print(f"Final train accuracy: {history['train_acc'][-1]:.1f}%")
    print(f"Final val accuracy: {history['val_acc'][-1]:.1f}%")
    print(f"Overfit gap: {history['train_acc'][-1] - history['val_acc'][-1]:.1f}%")

    # Compare with our baselines
    print("\n" + "="*70)
    print("COMPARISON WITH EMBEDDING-BASED APPROACHES")
    print("="*70)
    print(f"{'Method':<30} {'Val Accuracy':<15}")
    print("-"*45)
    print(f"{'Random baseline':<30} {'3.3%':<15}")
    print(f"{'ArcFace + Linear Probe':<30} {'9.0%':<15}")
    print(f"{'CLIP + Linear Probe':<30} {'12.6%':<15}")
    print(f"{args.mode.upper() + ' (this run)':<30} {f'{best_val_acc:.1f}%':<15}")


if __name__ == "__main__":
    main()
