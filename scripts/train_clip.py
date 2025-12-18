"""
CLIP Fine-tuning for Face-Name Recognition.
Uses OpenCLIP for efficient training with full fine-tuning.

Goals:
- reproducible runs (when --deterministic is enabled)
- stable training (AMP, grad clipping, warmup+cosine)
- correct/meaningful validation (top-1 name classification against all candidate names)
"""
import json
import os
import random
import time
from dataclasses import asdict, dataclass
import math
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
import argparse
from pathlib import Path

from clip_dataset import FaceNameDataset, create_name_gender_mapping


def seed_everything(seed: int, deterministic: bool = False):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        # This can raise on some ops; that's intended for strict reproducibility.
        torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int):
    # Ensures python random + torch RNG are unique per worker but reproducible.
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed)


def collate_fn(batch, tokenizer):
    """Custom collate function to tokenize text on-the-fly."""
    images, texts, names = zip(*batch)
    images = torch.stack(images)
    text_tokens = tokenizer(list(texts))
    return images, text_tokens, list(names)


@dataclass(frozen=True)
class TrainConfig:
    model: str
    pretrained: str
    lr: float
    text_lr_mult: float
    weight_decay: float
    batch_size: int
    epochs: int
    warmup_steps: int
    grad_clip_norm: float
    amp: bool
    seed: int
    deterministic: bool
    num_workers: int
    index_dir: str
    output_dir: str
    log_every: int


class CLIPTrainer:
    def __init__(
        self,
        model_name: str = "ViT-B-32",
        pretrained: str = "openai",
        learning_rate: float = 1e-5,
        text_lr_mult: float = 0.1,
        weight_decay: float = 0.01,
        batch_size: int = 32,
        num_epochs: int = 10,
        warmup_steps: int = 100,
        grad_clip_norm: float = 1.0,
        amp: bool = True,
        output_dir: str = "./experiments/clip_checkpoints",
        device: str = "cuda",
        log_every: int = 25,
    ):
        self.device = device
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.warmup_steps = warmup_steps
        self.grad_clip_norm = grad_clip_norm
        self.amp = amp and (device.startswith("cuda") and torch.cuda.is_available())
        self.log_every = log_every
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Load CLIP model
        print(f"Loading {model_name} pretrained on {pretrained}...")
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.tokenizer = get_tokenizer(model_name)
        
        # Optimizer - include ALL trainable parameters; group by module prefix.
        visual_params = []
        text_params = []
        other_params = []
        for n, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if n.startswith("visual."):
                visual_params.append(p)
            elif n.startswith("transformer.") or n.startswith("token_embedding") or n.startswith("ln_final") or n.startswith("text_projection") or n.startswith("positional_embedding"):
                text_params.append(p)
            else:
                other_params.append(p)  # includes logit_scale and anything model-specific

        param_groups = [
            {"params": visual_params, "lr": learning_rate},
            {"params": text_params, "lr": learning_rate * text_lr_mult},
        ]
        if other_params:
            param_groups.append({"params": other_params, "lr": learning_rate})

        self.optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)
        
        # Loss
        self.loss_fn = nn.CrossEntropyLoss()

        self.scaler = torch.cuda.amp.GradScaler(enabled=self.amp)
        self.scheduler = None
        self.global_step = 0
    
    def compute_clip_loss(self, image_features, text_features, logit_scale):
        """Compute symmetric CLIP contrastive loss."""
        # Normalize features
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        
        # Compute logits
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logits_per_image.T
        
        # Labels are just the diagonal (each image matches its own text)
        batch_size = image_features.shape[0]
        labels = torch.arange(batch_size, device=self.device)
        
        # Symmetric loss
        loss_i2t = self.loss_fn(logits_per_image, labels)
        loss_t2i = self.loss_fn(logits_per_text, labels)
        
        return (loss_i2t + loss_t2i) / 2
    
    def _build_scheduler(self, total_steps: int):
        # Warmup then cosine decay to 0.
        warmup = max(0, int(self.warmup_steps))
        total = max(1, int(total_steps))

        def lr_lambda(step: int):
            if warmup > 0 and step < warmup:
                return float(step + 1) / float(warmup)
            # cosine from 1 -> 0
            progress = float(step - warmup) / float(max(1, total - warmup))
            return 0.5 * (1.0 + math.cos(progress * math.pi))

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lr_lambda)

    def train_epoch(self, dataloader, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        for images, text_tokens, _names in pbar:
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=self.amp):
                image_features = self.model.encode_image(images)
                text_features = self.model.encode_text(text_tokens)
                logit_scale = self.model.logit_scale.exp()
                loss = self.compute_clip_loss(image_features, text_features, logit_scale)
            
            # Backward pass
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            if self.grad_clip_norm and self.grad_clip_norm > 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            if self.scheduler is not None:
                self.scheduler.step()
            self.global_step += 1
            
            # Clamp logit scale
            with torch.no_grad():
                self.model.logit_scale.clamp_(0, 4.6052)  # ln(100)
            
            total_loss += loss.item()
            num_batches += 1
            if num_batches % max(1, self.log_every) == 0:
                lr0 = self.optimizer.param_groups[0]["lr"]
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{lr0:.2e}"})
        
        return total_loss / max(1, num_batches)
    
    @torch.no_grad()
    def validate(self, dataloader, candidate_names):
        """Validate the model with a meaningful metric: top-1 name classification."""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        correct_top1 = 0
        total = 0

        # Encode candidate texts once.
        candidate_texts = [n.capitalize() for n in candidate_names]
        candidate_tokens = self.tokenizer(candidate_texts)
        candidate_tokens = candidate_tokens.to(self.device)
        with torch.cuda.amp.autocast(enabled=self.amp):
            cand_text_features = self.model.encode_text(candidate_tokens)
        cand_text_features = cand_text_features / cand_text_features.norm(dim=-1, keepdim=True)

        name_to_idx = {n: i for i, n in enumerate(candidate_names)}

        for images, text_tokens, names in dataloader:
            images = images.to(self.device)
            text_tokens = text_tokens.to(self.device)

            with torch.cuda.amp.autocast(enabled=self.amp):
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
        """Save model checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "global_step": self.global_step,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler is not None else None,
            "scaler_state_dict": self.scaler.state_dict() if self.scaler is not None else None,
            "val_metrics": val_metrics,
        }
        path = self.output_dir / f"checkpoint_epoch_{epoch+1}.pt"
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")
    
    def train(self, train_loader, val_loader, candidate_names):
        """Full training loop."""
        best_val_loss = float("inf")

        if self.scheduler is None:
            total_steps = self.num_epochs * max(1, len(train_loader))
            self._build_scheduler(total_steps)
        
        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader, candidate_names=candidate_names)
            
            print(
                f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, "
                f"val_loss={val_metrics['val_loss']:.4f}, "
                f"top1_name_acc={val_metrics['top1_name_acc']:.4f}"
            )
            
            # Save best model
            if val_metrics["val_loss"] < best_val_loss:
                best_val_loss = val_metrics["val_loss"]
                self.save_checkpoint(epoch, val_metrics)
        
        print("\nTraining complete!")
        return self.model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="ViT-B-32", help="CLIP model name")
    parser.add_argument("--pretrained", default="openai", help="Pretrained weights")
    parser.add_argument("--lr", type=float, default=1e-5, help="Learning rate")
    parser.add_argument("--text-lr-mult", type=float, default=0.1, help="Multiplier for text encoder LR vs visual LR")
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--no-amp", action="store_true", help="Disable AMP")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--deterministic", action="store_true", help="Enable strict deterministic algorithms")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--index-dir", default="/home/leann/face-detection/data/index_files", help="Directory containing index_*.json files")
    parser.add_argument("--device", default=None, help="e.g. cuda, cuda:0, cpu (default: auto)")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--output-dir", default="./experiments/clip_checkpoints")
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seed_everything(args.seed, deterministic=args.deterministic)
    
    # Target names
    male_names = [
        "david", "john", "michael", "mark", "peter",
        "robert", "james", "paul", "richard", "andrew",
        "thomas", "daniel", "chris", "william", "eric",
        "andreas"  # Add here
    ]
    female_names = [
        "maria", "jennifer", "mary", "susan", "patricia",
        "linda", "sarah", "karen", "jessica", "elizabeth",
        "anne", "lisa", "laura", "andrea"  # Remove andreas
    ]
    target_names = male_names + female_names
    
    # Create trainer
    trainer = CLIPTrainer(
        model_name=args.model,
        pretrained=args.pretrained,
        learning_rate=args.lr,
        text_lr_mult=args.text_lr_mult,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        amp=(not args.no_amp),
        output_dir=args.output_dir,
        device=device,
        log_every=args.log_every,
    )

    # Save run config for reproducibility
    cfg = TrainConfig(
        model=args.model,
        pretrained=args.pretrained,
        lr=args.lr,
        text_lr_mult=args.text_lr_mult,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        grad_clip_norm=args.grad_clip_norm,
        amp=(not args.no_amp),
        seed=args.seed,
        deterministic=args.deterministic,
        num_workers=args.num_workers,
        index_dir=args.index_dir,
        output_dir=args.output_dir,
        log_every=args.log_every,
    )
    (Path(args.output_dir) / "run_config.json").write_text(json.dumps(asdict(cfg), indent=2))
    
    # Create datasets
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
    
    # Create dataloaders
    g = torch.Generator()
    g.manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, trainer.tokenizer),
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=lambda b: collate_fn(b, trainer.tokenizer),
        pin_memory=True,
        worker_init_fn=seed_worker,
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    # Train!
    trainer.train(train_loader, val_loader, candidate_names=target_names)


if __name__ == "__main__":
    main()
