#!/usr/bin/env python3
"""Minimal ViT skeleton - now with train/test split."""

import json
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import timm
import os

def load_names_from_file(filepath):
    """Load names from a text file (one per line, ignoring comments and blanks)."""
    names = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                names.append(line)
    return names

# Hardcoded config
INDEX_DIR = "data/index_files"
NAMES_FILE = "data/male_names.txt"  # Male-only names for gender-controlled experiment
EPOCHS = 100
BATCH_SIZE = 1024
LR = 1e-4  # Lower LR to reduce instability
NUM_WORKERS = 8
LOG_FILE = "monkey_vit_log.txt"
EVAL_EVERY = 20


class FaceDataset(Dataset):
    def __init__(self, index_dir, names, split="train"):
        """split: 'train' = first half, 'test' = second half"""
        self.images = []
        self.labels = []
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        for label_idx, name in enumerate(names):
            index_path = f"{index_dir}/index_{name}.json"
            with open(index_path) as f:
                data = json.load(f)
            paths = data["good"]
            mid = len(paths) // 2
            if split == "train":
                paths = paths[:mid]
            else:
                paths = paths[mid:]
            for p in paths:
                self.images.append(p)
                self.labels.append(label_idx)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        img = self.transform(img)
        return img, self.labels[idx]


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load names from file
    names = load_names_from_file(NAMES_FILE)
    print(f"Using {len(names)} male names: {names[:5]}...{names[-5:]}")

    # Datasets
    train_dataset = FaceDataset(INDEX_DIR, names, split="train")
    test_dataset = FaceDataset(INDEX_DIR, names, split="test")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)
    print(f"Train: {len(train_dataset)} images | Test: {len(test_dataset)} images | Classes: {len(names)}")

    # Model - custom ViT: 6 layers, GAP instead of CLS
    model = timm.create_model(
        "vit_tiny_patch16_384",
        pretrained=False,
        num_classes=len(names),
        depth=6,              # 6 layers instead of 12
        global_pool='avg',    # GAP instead of CLS token
        class_token=False,    # No CLS token
    )
    model = model.to(device)
    model = torch.compile(model)
    print(f"Model: ViT-Tiny/16@384 (6 layers, GAP), {sum(p.numel() for p in model.parameters()):,} params (compiled)")

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler()  # AMP for faster training

    # Open log file
    log_f = open(LOG_FILE, "w")
    log_f.write("epoch,train_loss,train_acc,test_acc\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            with torch.amp.autocast(device_type="cuda"):
                outputs = model(imgs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        avg_loss = total_loss / len(train_loader)
        train_acc = correct / total * 100

        # Evaluate every EVAL_EVERY epochs
        if epoch % EVAL_EVERY == 0 or epoch == 1:
            model.eval()
            correct = 0
            total = 0
            with torch.no_grad(), torch.amp.autocast(device_type="cuda"):
                for imgs, labels in test_loader:
                    imgs, labels = imgs.to(device), labels.to(device)
                    outputs = model(imgs)
                    preds = outputs.argmax(dim=1)
                    correct += (preds == labels).sum().item()
                    total += labels.size(0)
            test_acc = correct / total * 100

            line = f"Epoch {epoch:3d} | Loss: {avg_loss:.4f} | Train: {train_acc:.1f}% | Test: {test_acc:.1f}%"
            print(line)
            log_f.write(f"{epoch},{avg_loss:.6f},{train_acc:.1f},{test_acc:.1f}\n")
            log_f.flush()

    log_f.close()
    print(f"\nTraining complete. Log saved to {LOG_FILE}")


if __name__ == "__main__":
    main()
