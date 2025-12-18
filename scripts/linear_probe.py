"""
Linear Probe Experiment: Freeze CLIP, train only a classifier head.

This tests whether CLIP embeddings already contain name-face signal,
without the risk of catastrophic forgetting from full fine-tuning.

Usage:
    python linear_probe.py --names david michael --epochs 50
    EXPERIMENT=2 python linear_probe.py --epochs 50  # Use same experiment selector
"""
import argparse
import json
import os
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
from pathlib import Path

from clip_dataset import FaceNameDataset, create_name_gender_mapping


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class LinearProbeClassifier(nn.Module):
    """Simple linear classifier on top of frozen CLIP embeddings."""
    
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.classifier = nn.Linear(input_dim, num_classes)
    
    def forward(self, x):
        return self.classifier(x)


def extract_embeddings(model, dataloader, device):
    """Extract CLIP image embeddings for all samples."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, _text_tokens, names in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(names)
    
    return torch.cat(all_embeddings, dim=0), all_labels


def train_linear_probe(
    train_embeddings, train_labels,
    val_embeddings, val_labels,
    name_to_idx, num_epochs=50, lr=0.01, device="cuda"
):
    """Train a linear classifier on extracted embeddings."""
    num_classes = len(name_to_idx)
    input_dim = train_embeddings.shape[1]
    
    classifier = LinearProbeClassifier(input_dim, num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    # Convert labels to indices
    train_y = torch.tensor([name_to_idx[n] for n in train_labels])
    val_y = torch.tensor([name_to_idx[n] for n in val_labels])
    
    # Move to device
    train_X = train_embeddings.to(device)
    train_y = train_y.to(device)
    val_X = val_embeddings.to(device)
    val_y = val_y.to(device)
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(num_epochs):
        # Training
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
        
        train_acc = (logits.argmax(dim=1) == train_y).float().mean().item()
        
        # Validation
        classifier.eval()
        with torch.no_grad():
            val_logits = classifier(val_X)
            val_loss = criterion(val_logits, val_y)
            val_acc = (val_logits.argmax(dim=1) == val_y).float().mean().item()
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch + 1
        
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}: train_loss={loss.item():.4f}, train_acc={train_acc:.4f}, "
                  f"val_loss={val_loss.item():.4f}, val_acc={val_acc:.4f}")
    
    return best_val_acc, best_epoch, classifier


def collate_fn(batch, tokenizer):
    images, texts, names = zip(*batch)
    images = torch.stack(images)
    text_tokens = tokenizer(list(texts))
    return images, text_tokens, list(names)


def main():
    parser = argparse.ArgumentParser(description="Linear probe experiment for CLIP")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Names to test (overrides EXPERIMENT env var)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-dir", default="/home/leann/face-detection/data/index_files")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    # Experiment selector (same as test_train_clip.py)
    if args.names:
        target_names = args.names
        exp_name = f"Custom ({', '.join(args.names)})"
    else:
        experiment = os.environ.get("EXPERIMENT", "2")  # Default to same-gender
        if experiment == "1":
            target_names = ["david", "laura"]
            exp_name = "Mixed Gender (david vs laura)"
        elif experiment == "2":
            target_names = ["david", "michael"]
            exp_name = "Same Gender Male (david vs michael)"
        elif experiment == "3":
            target_names = ["maria", "laura"]
            exp_name = "Same Gender Female (maria vs laura)"
        else:
            target_names = ["david", "michael"]
            exp_name = "Same Gender Male (david vs michael)"
    
    print("\n" + "="*60)
    print("LINEAR PROBE EXPERIMENT")
    print(f"Testing: {target_names}")
    print(f"Experiment: {exp_name}")
    print("="*60 + "\n")
    
    # Load CLIP model (frozen)
    print("Loading CLIP ViT-B-32 (frozen)...")
    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    tokenizer = get_tokenizer("ViT-B-32")
    name_to_gender = create_name_gender_mapping()
    
    # Create datasets
    train_dataset = FaceNameDataset(
        index_dir=args.index_dir,
        target_names=target_names,
        name_to_gender=name_to_gender,
        transform=preprocess,
        split="train",
        seed=args.seed,
        prompt_mode="deterministic",
    )
    
    val_dataset = FaceNameDataset(
        index_dir=args.index_dir,
        target_names=target_names,
        name_to_gender=name_to_gender,
        transform=preprocess,
        split="val",
        seed=args.seed,
        prompt_mode="deterministic",
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=True,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Random baseline: {100/len(target_names):.1f}%\n")
    
    # Extract embeddings
    print("Extracting embeddings (this is one-time)...")
    train_embeddings, train_labels = extract_embeddings(model, train_loader, device)
    val_embeddings, val_labels = extract_embeddings(model, val_loader, device)
    
    print(f"Train embeddings shape: {train_embeddings.shape}")
    print(f"Val embeddings shape: {val_embeddings.shape}\n")
    
    # Train linear probe
    name_to_idx = {name: i for i, name in enumerate(target_names)}
    
    print(f"Training linear probe for {args.epochs} epochs...")
    best_acc, best_epoch, classifier = train_linear_probe(
        train_embeddings, train_labels,
        val_embeddings, val_labels,
        name_to_idx,
        num_epochs=args.epochs,
        lr=args.lr,
        device=device,
    )
    
    # Results
    print("\n" + "="*60)
    print("LINEAR PROBE RESULTS")
    print("="*60)
    print(f"Best validation accuracy: {best_acc:.4f} ({100*best_acc:.1f}%)")
    print(f"Achieved at epoch: {best_epoch}")
    print(f"Random baseline: {100/len(target_names):.1f}%")
    print()
    
    # Interpretation
    improvement = (best_acc - 1/len(target_names)) / (1/len(target_names)) * 100
    print(f"Improvement over random: +{improvement:.1f}%")
    print()
    
    if best_acc > 0.70:
        print("→ Linear probe works well!")
        print("  CLIP embeddings contain strong name-face signal.")
        print("  Full fine-tuning was causing overfitting.")
    elif best_acc > 0.55:
        print("→ Linear probe shows modest improvement.")
        print("  CLIP embeddings contain some signal, but it's weak.")
        print("  This might be near the task ceiling.")
    else:
        print("→ Linear probe near random.")
        print("  CLIP embeddings may not capture name-face associations.")
        print("  Consider: data quality issue OR task is too hard.")


if __name__ == "__main__":
    main()

