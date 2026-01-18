"""
Qwen 2.5 VL 7B Fine-tuning with 4-bit Quantization + LoRA.

Key features:
- 4-bit quantization (NF4) to fit 7B model in ~18GB VRAM
- LoRA fine-tuning for actual weight updates (not just classifier head)
- Classification head on top of mean-pooled hidden states
- Per-class accuracy tracking
- Proper LoRA adapter saving for reproducibility

Usage:
    # Train with balanced data
    python train_qwen_7b_lora.py \\
        --data-dir /home/leann/face-detection/data/qwen_7b_balanced \\
        --output-dir /home/leann/face-detection/results/qwen_7b/balanced_lora \\
        --epochs 5

    # Train with imbalanced data (use class balancing)
    python train_qwen_7b_lora.py \\
        --data-dir /home/leann/face-detection/data/qwen_7b_imbalanced \\
        --output-dir /home/leann/face-detection/results/qwen_7b/imbalanced_lora \\
        --class-balanced-beta 0.999 \\
        --epochs 5
"""
import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional
from collections import defaultdict

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm

try:
    from transformers import (
        AutoModelForVision2Seq,
        AutoProcessor,
        BitsAndBytesConfig,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, PeftModel
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/peft not installed. Install with:")
    print("  pip install transformers accelerate bitsandbytes peft qwen-vl-utils")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@dataclass
class TrainConfig:
    """Full training configuration for reproducibility."""
    model_id: str
    num_labels: int
    learning_rate: float
    batch_size: int
    gradient_accumulation_steps: int
    num_epochs: int
    warmup_ratio: float
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list
    use_4bit: bool
    use_lora: bool
    freeze_vision: bool
    seed: int
    data_dir: str
    output_dir: str
    class_balanced_beta: Optional[float]
    max_length: int
    num_workers: int


class FaceNameClassificationDataset(Dataset):
    """Dataset for face-name classification."""

    def __init__(
        self,
        data_file: str,
        processor,
        max_length: int = 512,
        target_size: int = 512,
    ):
        with open(data_file) as f:
            self.samples = json.load(f)
        self.processor = processor
        self.max_length = max_length
        self.target_size = target_size
        self.prompt = "What is this person's first name?"

        # Pre-validate paths
        valid_samples = []
        for s in self.samples:
            if Path(s["image"]).exists():
                valid_samples.append(s)
        if len(valid_samples) < len(self.samples):
            print(f"  Warning: {len(self.samples) - len(valid_samples)} images not found, using {len(valid_samples)}")
        self.samples = valid_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image"]
        label = sample["label"]

        try:
            image = Image.open(image_path).convert("RGB")
            # Resize to target size to ensure consistent tensor shapes
            image = image.resize((self.target_size, self.target_size), Image.Resampling.LANCZOS)
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            image = Image.new("RGB", (self.target_size, self.target_size), color="gray")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        inputs = self.processor(
            text=[text],
            images=[image],
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt",
        )

        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.long)

        return inputs


class Qwen2VLForClassification(nn.Module):
    """Qwen2VL with LoRA and classification head."""

    def __init__(
        self,
        model_id: str,
        num_labels: int,
        use_4bit: bool = True,
        use_lora: bool = True,
        freeze_vision: bool = True,
        lora_r: int = 64,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        lora_target_modules: list = None,
    ):
        super().__init__()

        self.model_id = model_id
        self.use_lora = use_lora
        self.num_labels = num_labels

        # Quantization config for 4-bit
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        print(f"Loading {model_id}...")
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Freeze vision encoder
        if freeze_vision:
            for name, param in self.model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

        # Apply LoRA
        if use_lora:
            print("Applying LoRA...")
            self.model = prepare_model_for_kbit_training(self.model)

            if lora_target_modules is None:
                lora_target_modules = [
                    "q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"
                ]

            lora_config = LoraConfig(
                r=lora_r,
                lora_alpha=lora_alpha,
                target_modules=lora_target_modules,
                lora_dropout=lora_dropout,
                bias="none",
                task_type="CAUSAL_LM",
            )
            self.model = get_peft_model(self.model, lora_config)
            self.model.print_trainable_parameters()

        # Classification head
        hidden_size = self.model.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )
        self.classifier = self.classifier.to(self.model.device)

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
    ):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden_states = outputs.hidden_states[-1]

        # Mean pooling
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1)
        pooled = sum_hidden / count.clamp(min=1e-9)

        logits = self.classifier(pooled.to(self.classifier[0].weight.dtype))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}

    def save_pretrained(self, output_dir: str):
        """Save LoRA adapter and classifier weights."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save LoRA adapter
        if self.use_lora:
            self.model.save_pretrained(output_path / "lora_adapter")
            print(f"  Saved LoRA adapter to {output_path / 'lora_adapter'}")

        # Save classifier
        torch.save(self.classifier.state_dict(), output_path / "classifier.pt")
        print(f"  Saved classifier to {output_path / 'classifier.pt'}")


class Trainer:
    """Training loop with per-class metrics."""

    def __init__(
        self,
        model: Qwen2VLForClassification,
        train_dataset: Dataset,
        val_dataset: Dataset,
        label_names: list,
        config: TrainConfig,
        class_weights: Optional[torch.Tensor] = None,
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.label_names = label_names
        self.config = config
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Optimizer
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=config.learning_rate)

        # Scheduler
        total_steps = (
            len(train_dataset) // (config.batch_size * config.gradient_accumulation_steps)
        ) * config.num_epochs
        warmup_steps = int(total_steps * config.warmup_ratio)

        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config.learning_rate,
            total_steps=max(total_steps, 1),
            pct_start=config.warmup_ratio,
        )

        # Loss
        self.class_weights = class_weights
        if class_weights is not None:
            self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn = nn.CrossEntropyLoss()

    def collate_fn(self, batch):
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            if key == "labels":
                collated[key] = torch.stack([b[key] for b in batch])
            elif key in ["pixel_values"]:
                collated[key] = torch.cat([b[key].unsqueeze(0) for b in batch], dim=0)
            else:
                collated[key] = torch.stack([b[key] for b in batch])
        return collated

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    labels=batch["labels"],
                )

                # Use class-weighted loss if provided
                if self.class_weights is not None:
                    loss = self.loss_fn(outputs["logits"], batch["labels"])
                else:
                    loss = outputs["loss"]

                loss = loss / self.config.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1

            if num_batches % 10 == 0:
                pbar.set_postfix({
                    "loss": f"{total_loss/num_batches:.4f}",
                    "lr": f"{self.scheduler.get_last_lr()[0]:.2e}"
                })

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        # Per-class tracking
        per_class_correct = defaultdict(int)
        per_class_total = defaultdict(int)

        for batch in tqdm(dataloader, desc="Validating"):
            batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    labels=batch["labels"],
                )

            total_loss += outputs["loss"].item()
            preds = outputs["logits"].argmax(dim=-1)
            labels = batch["labels"]

            correct += (preds == labels).sum().item()
            total += len(labels)

            # Per-class
            for pred, label in zip(preds.cpu().tolist(), labels.cpu().tolist()):
                per_class_total[label] += 1
                if pred == label:
                    per_class_correct[label] += 1

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        # Compute per-class accuracy
        per_class_acc = {}
        for label_idx in range(len(self.label_names)):
            if per_class_total[label_idx] > 0:
                acc = per_class_correct[label_idx] / per_class_total[label_idx]
                per_class_acc[self.label_names[label_idx]] = {
                    "accuracy": acc,
                    "correct": per_class_correct[label_idx],
                    "total": per_class_total[label_idx],
                }

        return {
            "val_loss": total_loss / len(dataloader),
            "accuracy": correct / total,
            "per_class_accuracy": per_class_acc,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        best_accuracy = 0
        training_log = []

        for epoch in range(self.config.num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "accuracy": val_metrics["accuracy"],
                "per_class_accuracy": val_metrics["per_class_accuracy"],
            }
            training_log.append(log_entry)

            print(f"\nEpoch {epoch+1}:")
            print(f"  Train loss: {train_loss:.4f}")
            print(f"  Val loss: {val_metrics['val_loss']:.4f}")
            print(f"  Accuracy: {val_metrics['accuracy']:.4f}")

            # Top-5 and bottom-5 per-class accuracies
            sorted_classes = sorted(
                val_metrics["per_class_accuracy"].items(),
                key=lambda x: x[1]["accuracy"],
                reverse=True
            )
            print(f"  Top-5 classes: {[(n, f'{v['accuracy']:.2%}') for n, v in sorted_classes[:5]]}")
            print(f"  Bottom-5 classes: {[(n, f'{v['accuracy']:.2%}') for n, v in sorted_classes[-5:]]}")

            # Save checkpoint
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                self.save_checkpoint(epoch, val_metrics, is_best=True)

            # Always save latest
            self.save_checkpoint(epoch, val_metrics, is_best=False)

        # Save final training log
        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        return training_log

    def save_checkpoint(self, epoch: int, val_metrics: dict, is_best: bool = False):
        if is_best:
            checkpoint_dir = self.output_dir / "best_model"
        else:
            checkpoint_dir = self.output_dir / f"checkpoint_epoch_{epoch+1}"

        checkpoint_dir.mkdir(exist_ok=True)

        # Save model (LoRA adapter + classifier)
        self.model.save_pretrained(str(checkpoint_dir))

        # Save metrics
        metrics = {
            "epoch": epoch + 1,
            "val_loss": val_metrics["val_loss"],
            "accuracy": val_metrics["accuracy"],
            "per_class_accuracy": val_metrics["per_class_accuracy"],
        }
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"  Saved {'best ' if is_best else ''}checkpoint to {checkpoint_dir}")


def compute_class_weights(data_file: str, num_labels: int, beta: float = 0.999) -> torch.Tensor:
    """Compute class-balanced weights using effective number of samples."""
    with open(data_file) as f:
        samples = json.load(f)

    counts = [0] * num_labels
    for sample in samples:
        counts[sample["label"]] += 1

    effective_num = [(1.0 - beta**c) / (1.0 - beta) if c > 0 else 1.0 for c in counts]
    weights = [1.0 / en for en in effective_num]

    total = sum(weights)
    weights = [w / total * len(weights) for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen 2.5 VL 7B with 4-bit + LoRA")

    # Model
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--use-4bit", action="store_true", default=True)
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit")
    parser.add_argument("--use-lora", action="store_true", default=True)
    parser.add_argument("--no-lora", action="store_false", dest="use_lora")
    parser.add_argument("--freeze-vision", action="store_true", default=True)

    # LoRA
    parser.add_argument("--lora-r", type=int, default=64)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--class-balanced-beta", type=float, default=None)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=4)

    # Data
    parser.add_argument("--data-dir", required=True, help="Directory with train.json, val.json, labels.json")
    parser.add_argument("--output-dir", required=True, help="Output directory for checkpoints")

    # Other
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Please install required packages")
        return

    seed_everything(args.seed)

    # Load labels
    data_dir = Path(args.data_dir)
    with open(data_dir / "labels.json") as f:
        labels = json.load(f)
    num_labels = len(labels)

    print(f"=" * 60)
    print(f"Qwen 2.5 VL 7B Fine-tuning with 4-bit + LoRA")
    print(f"=" * 60)
    print(f"Model: {args.model_id}")
    print(f"Data dir: {args.data_dir}")
    print(f"Output dir: {args.output_dir}")
    print(f"Num labels: {num_labels}")
    print(f"4-bit: {args.use_4bit}, LoRA: {args.use_lora}")
    print(f"LoRA config: r={args.lora_r}, alpha={args.lora_alpha}")
    print(f"=" * 60)

    # Load processor
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Create model
    lora_target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    model = Qwen2VLForClassification(
        model_id=args.model_id,
        num_labels=num_labels,
        use_4bit=args.use_4bit,
        use_lora=args.use_lora,
        freeze_vision=args.freeze_vision,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
    )

    # Load datasets
    print("Loading datasets...")
    train_dataset = FaceNameClassificationDataset(
        data_file=str(data_dir / "train.json"),
        processor=processor,
        max_length=args.max_length,
    )
    val_dataset = FaceNameClassificationDataset(
        data_file=str(data_dir / "val.json"),
        processor=processor,
        max_length=args.max_length,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Class weights
    class_weights = None
    if args.class_balanced_beta:
        print(f"Computing class-balanced weights (beta={args.class_balanced_beta})...")
        class_weights = compute_class_weights(
            str(data_dir / "train.json"),
            num_labels,
            beta=args.class_balanced_beta,
        )
        class_weights = class_weights.cuda()

    # Save config
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = TrainConfig(
        model_id=args.model_id,
        num_labels=num_labels,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        lora_target_modules=lora_target_modules,
        use_4bit=args.use_4bit,
        use_lora=args.use_lora,
        freeze_vision=args.freeze_vision,
        seed=args.seed,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        class_balanced_beta=args.class_balanced_beta,
        max_length=args.max_length,
        num_workers=args.num_workers,
    )

    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)
    with open(output_dir / "labels.json", "w") as f:
        json.dump(labels, f, indent=2)

    # Copy dataset config for reference
    if (data_dir / "dataset_config.json").exists():
        import shutil
        shutil.copy(data_dir / "dataset_config.json", output_dir / "dataset_config.json")

    # Create trainer
    trainer = Trainer(
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        label_names=labels,
        config=config,
        class_weights=class_weights,
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()
    training_log = trainer.train()
    elapsed = time.time() - start_time

    print(f"\n{'=' * 60}")
    print(f"Training complete in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {max(log['accuracy'] for log in training_log):.4f}")
    print(f"Results saved to: {args.output_dir}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
