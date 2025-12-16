"""
Experiment 1: Two-Name Sanity Check
Tests if CLIP can learn to distinguish david vs laura.
"""
import json
import os
import random
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
import argparse
from pathlib import Path

from clip_dataset import FaceNameDataset, create_name_gender_mapping


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def collate_fn(batch, tokenizer):
    images, texts, names = zip(*batch)
    images = torch.stack(images)
    text_tokens = tokenizer(list(texts))
    return images, text_tokens, list(names)


class CLIPTrainer:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        learning_rate: float = 1e-6,
        batch_size: int = 64,
        num_epochs: int = 20,
        warmup_steps: int = 50,
        output_dir: str = "./exp1_2names",
        device: str = "cuda",
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        print(f"Loading {model_name} pretrained on {pretrained}...")
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.tokenizer = get_tokenizer(model_name)
        
        # Single learning rate for simplicity
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=learning_rate, 
            weight_decay=0.01
        )
        
        self.loss_fn = nn.CrossEntropyLoss()
        self.scaler = torch.amp.GradScaler("cuda")
        self.scheduler = None
        self.global_step = 0
    
    def compute_clip_loss(self, image_features, text_features, logit_scale):
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        loss_i2t = self.loss_fn(logits_per_image, labels)
        loss_t2i = self.loss_fn(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def _build_scheduler(self, total_steps: int):
        warmup = self.warmup_steps

        def lr_lambda(step: int):
            if step < warmup:
                return float(step + 1) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_steps - warmup))
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train_epoch(self, dataloader, epoch):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for images, text_tokens, _names in pbar:
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            
            with torch.amp.autocast("cuda"):
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)
                logit_scale = self.model.logit_scale.exp()
                loss = self.compute_clip_loss(image_features, text_features, logit_scale)
            
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            
            with torch.no_grad():
                self.model.logit_scale.clamp_(0, 4.6052)
            
            total_loss += loss.item()
            num_batches += 1
            if num_batches % 10 == 0:
                pbar.set_postfix({"loss": f"{loss.item():.4f}"})
        
        return total_loss / max(1, num_batches)
    
    @torch.no_grad()
    def validate(self, dataloader, candidate_names):
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_top1 = 0
        total = 0

        candidate_texts = [n.capitalize() for n in candidate_names]
        candidate_tokens = self.tokenizer(candidate_texts).to(self.device)
        
        with torch.amp.autocast("cuda"):
            cand_text_features = self.model.encode_text(candidate_tokens)
        cand_text_features = cand_text_features / cand_text_features.norm(dim=-1, keepdim=True)

        name_to_idx = {n: i for i, n in enumerate(candidate_names)}

        for images, text_tokens, names in dataloader:
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)

            with torch.amp.autocast("cuda"):
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)
                logit_scale = self.model.logit_scale.exp()
                loss = self.compute_clip_loss(image_features, text_features, logit_scale)

            total_loss += float(loss.item())
            num_batches += 1

            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            logits = logit_scale * image_features @ cand_text_features.T

            preds = logits.argmax(dim=1).tolist()
            labels = [name_to_idx[n] for n in names]
            correct_top1 += sum(int(p == y) for p, y in zip(preds, labels))
            total += len(names)

        return {
            "val_loss": total_loss / max(1, num_batches),
            "top1_name_acc": correct_top1 / max(1, total),
        }
    
    def save_checkpoint(self, epoch, val_metrics):
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "val_metrics": val_metrics,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def train(self, train_loader, val_loader, candidate_names):
        best_val_acc = 0.0
        
        total_steps = self.num_epochs * len(train_loader)
        self._build_scheduler(total_steps)
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, candidate_names=candidate_names)
            
            print(
                f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"top1_name_acc={val_metrics['top1_name_acc']:.4f}"
            )
            
            # Save based on accuracy for this experiment
            if val_metrics["top1_name_acc"] > best_val_acc:
                best_val_acc = val_metrics["top1_name_acc"]
                self.save_checkpoint(epoch, val_metrics)
        
        print(f"\nExperiment complete! Best accuracy: {best_val_acc:.4f}")
        return self.model


def zero_shot_baseline(model, val_loader, candidate_names, tokenizer, device):
    """Test vanilla CLIP without any fine-tuning."""
    model.eval()
    correct = 0
    total = 0
    
    # Use simple prompt template for zero-shot
    candidate_texts = [f"a photo of {n.capitalize()}" for n in candidate_names]
    candidate_tokens = tokenizer(candidate_texts).to(device)
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        cand_text_features = model.encode_text(candidate_tokens)
        cand_text_features = cand_text_features / cand_text_features.norm(dim=-1, keepdim=True)
        
        name_to_idx = {n: i for i, n in enumerate(candidate_names)}
        
        for images, _text_tokens, names in val_loader:
            images = images.to(device)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            logit_scale = model.logit_scale.exp()
            logits = logit_scale * image_features @ cand_text_features.T
            
            preds = logits.argmax(dim=1).tolist()
            labels = [name_to_idx[n] for n in names]
            correct += sum(int(p == y) for p, y in zip(preds, labels))
            total += len(names)
    
    return correct / max(1, total)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-dir", default="/home/leann/face-detection")
    parser.add_argument("--output-dir", default="./exp1_2names")
    parser.add_argument("--zero-shot-only", action="store_true", help="Only run zero-shot baseline")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    # ============================================
    # EXPERIMENT SELECTOR
    # ============================================
    # Experiment 1: Mixed gender (easy - may just detect gender)
    # Experiment 2: Same gender male (tests true face recognition)
    # Experiment 3: Same gender female (tests true face recognition)
    
    experiment = os.environ.get("EXPERIMENT", "1")
    
    if experiment == "1":
        target_names = ["david", "laura"]
        exp_name = "Mixed Gender (david vs laura)"
        output_dir = args.output_dir
    elif experiment == "2":
        target_names = ["david", "michael"]
        exp_name = "Same Gender Male (david vs michael)"
        output_dir = args.output_dir.replace("2names", "2names_male")
    elif experiment == "3":
        target_names = ["maria", "laura"]
        exp_name = "Same Gender Female (maria vs laura)"
        output_dir = args.output_dir.replace("2names", "2names_female")
    else:
    target_names = ["david", "laura"]
        exp_name = "Mixed Gender (david vs laura)"
        output_dir = args.output_dir
    
    # Override output dir if experiment changes
    args.output_dir = output_dir
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENT {experiment}: {exp_name}")
    print(f"Testing: {target_names}")
    print(f"{'='*60}\n")
    
    trainer = CLIPTrainer(
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.output_dir,
        device=device,
    )

    name_to_gender = create_name_gender_mapping()
    
    train_dataset = FaceNameDataset(
        index_dir=args.index_dir,
        target_names=target_names,
        name_to_gender=name_to_gender,
        transform=trainer.preprocess,
        split="train",
        seed=args.seed,
        prompt_mode="random",
    )
    
    val_dataset = FaceNameDataset(
        index_dir=args.index_dir,
        target_names=target_names,
        name_to_gender=name_to_gender,
        transform=trainer.preprocess,
        split="val",
        seed=args.seed,
        prompt_mode="deterministic",
    )
    
    g = torch.Generator()
    g.manual_seed(args.seed)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, trainer.tokenizer),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, trainer.tokenizer),
        pin_memory=True,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Random baseline: 50% (2 classes)\n")
    
    # Always run zero-shot baseline first
    print("Running zero-shot baseline (no fine-tuning)...")
    zs_acc = zero_shot_baseline(
        trainer.model, val_loader, target_names, trainer.tokenizer, device
    )
    print(f"Zero-shot accuracy: {zs_acc:.4f} ({zs_acc*100:.1f}%)\n")
    
    if args.zero_shot_only:
        print("Zero-shot only mode, skipping training.")
        return
    
    trainer.train(train_loader, val_loader, candidate_names=target_names)


if __name__ == "__main__":
    main()