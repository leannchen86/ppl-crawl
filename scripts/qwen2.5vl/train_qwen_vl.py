"""
Qwen 2.5 VL Fine-tuning for Face-Name Association.

Supports:
- Classification head training (recommended)
- LoRA fine-tuning with 4-bit quantization
- Multiple training approaches (classification, generative)

Based on: https://github.com/2U1/Qwen-VL-Series-Finetune
"""
import argparse
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

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
        TrainingArguments,
        Trainer,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False
    print("Warning: transformers/peft not installed. Install with:")
    print("  pip install transformers accelerate bitsandbytes peft trl qwen-vl-utils")


def seed_everything(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class TrainConfig:
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
    use_4bit: bool
    use_lora: bool
    freeze_vision: bool
    seed: int
    output_dir: str
    class_balanced_beta: Optional[float]


class FaceNameClassificationDataset(Dataset):
    """Dataset for classification training."""

    def __init__(
        self,
        data_file: str,
        processor,
        max_length: int = 512,
    ):
        with open(data_file) as f:
            self.samples = json.load(f)
        self.processor = processor
        self.max_length = max_length

        # Fixed prompt for classification
        self.prompt = "What is this person's first name?"

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        image_path = sample["image"]
        label = sample["label"]

        # Load and process image
        try:
            image = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            # Return a dummy sample
            image = Image.new("RGB", (512, 512), color="gray")

        # Process with Qwen2VL processor
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

        # Squeeze batch dimension
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label, dtype=torch.long)

        return inputs


class Qwen2VLForClassification(nn.Module):
    """Qwen2VL with classification head."""

    def __init__(
        self,
        model_id: str,
        num_labels: int,
        use_4bit: bool = True,
        freeze_vision: bool = True,
    ):
        super().__init__()

        # Quantization config
        if use_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )
        else:
            quantization_config = None

        # Load base model (AutoModelForVision2Seq handles both Qwen2-VL and Qwen2.5-VL)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            quantization_config=quantization_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Freeze vision encoder if specified
        if freeze_vision:
            for name, param in self.model.named_parameters():
                if "visual" in name:
                    param.requires_grad = False

        # Get hidden size from config
        hidden_size = self.model.config.hidden_size

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels),
        )

        # Move classifier to same device as model
        self.classifier = self.classifier.to(self.model.device)

        self.num_labels = num_labels

    def forward(
        self,
        input_ids,
        attention_mask,
        pixel_values=None,
        image_grid_thw=None,
        labels=None,
    ):
        # Get model outputs
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            output_hidden_states=True,
            return_dict=True,
        )

        # Use last hidden state, take the last token (or mean pool)
        hidden_states = outputs.hidden_states[-1]

        # Mean pooling over sequence (excluding padding)
        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size())
        sum_hidden = (hidden_states * mask).sum(dim=1)
        count = mask.sum(dim=1)
        pooled = sum_hidden / count.clamp(min=1e-9)

        # Classify
        logits = self.classifier(pooled.to(self.classifier[0].weight.dtype))

        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)

        return {"loss": loss, "logits": logits}


class ClassificationTrainer:
    """Trainer for classification approach."""

    def __init__(
        self,
        model,
        processor,
        train_dataset,
        val_dataset,
        output_dir: str,
        learning_rate: float = 2e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 8,
        num_epochs: int = 3,
        warmup_ratio: float = 0.03,
        class_weights: Optional[torch.Tensor] = None,
        device: str = "cuda",
    ):
        self.model = model
        self.processor = processor
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # Optimizer - only train classifier and unfrozen params
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate)

        # Scheduler
        total_steps = (len(train_dataset) // (batch_size * gradient_accumulation_steps)) * num_epochs
        warmup_steps = int(total_steps * warmup_ratio)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=learning_rate,
            total_steps=total_steps,
            pct_start=warmup_ratio,
        )

        # Loss with optional class weights
        self.loss_fn = nn.CrossEntropyLoss(weight=class_weights)

        # Note: GradScaler not needed for bfloat16 (only for float16)

    def collate_fn(self, batch):
        """Custom collate function."""
        keys = batch[0].keys()
        collated = {}
        for key in keys:
            if key == "labels":
                collated[key] = torch.stack([b[key] for b in batch])
            elif key in ["pixel_values"]:
                # These may have variable shapes
                collated[key] = torch.cat([b[key].unsqueeze(0) for b in batch], dim=0)
            else:
                collated[key] = torch.stack([b[key] for b in batch])
        return collated

    def train_epoch(self, dataloader, epoch: int):
        self.model.train()
        total_loss = 0
        num_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{self.num_epochs}")
        self.optimizer.zero_grad()

        for step, batch in enumerate(pbar):
            # Move to device
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    pixel_values=batch.get("pixel_values"),
                    image_grid_thw=batch.get("image_grid_thw"),
                    labels=batch["labels"],
                )
                loss = outputs["loss"] / self.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % self.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()

            total_loss += loss.item() * self.gradient_accumulation_steps
            num_batches += 1

            if num_batches % 10 == 0:
                pbar.set_postfix({"loss": f"{total_loss/num_batches:.4f}"})

        return total_loss / max(1, num_batches)

    @torch.no_grad()
    def validate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for batch in tqdm(dataloader, desc="Validating"):
            batch = {k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

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
            correct += (preds == batch["labels"]).sum().item()
            total += len(batch["labels"])

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["labels"].cpu().tolist())

        return {
            "val_loss": total_loss / len(dataloader),
            "accuracy": correct / total,
            "predictions": all_preds,
            "labels": all_labels,
        }

    def train(self):
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=4,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

        best_accuracy = 0
        training_log = []

        for epoch in range(self.num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            val_metrics = self.validate(val_loader)

            log_entry = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_metrics["val_loss"],
                "accuracy": val_metrics["accuracy"],
            }
            training_log.append(log_entry)

            print(f"\nEpoch {epoch+1}: train_loss={train_loss:.4f}, "
                  f"val_loss={val_metrics['val_loss']:.4f}, "
                  f"accuracy={val_metrics['accuracy']:.4f}")

            # Save best model
            if val_metrics["accuracy"] > best_accuracy:
                best_accuracy = val_metrics["accuracy"]
                self.save_checkpoint(epoch, val_metrics)

        # Save training log
        with open(self.output_dir / "training_log.json", "w") as f:
            json.dump(training_log, f, indent=2)

        return training_log

    def save_checkpoint(self, epoch: int, val_metrics: dict):
        checkpoint_dir = self.output_dir / f"checkpoint_epoch_{epoch+1}"
        checkpoint_dir.mkdir(exist_ok=True)

        # Save classifier weights
        torch.save(
            self.model.classifier.state_dict(),
            checkpoint_dir / "classifier.pt"
        )

        # Save metrics
        with open(checkpoint_dir / "metrics.json", "w") as f:
            json.dump({
                "epoch": epoch + 1,
                "val_loss": val_metrics["val_loss"],
                "accuracy": val_metrics["accuracy"],
            }, f, indent=2)

        print(f"Saved checkpoint to {checkpoint_dir}")


def compute_class_weights(data_file: str, num_labels: int, beta: float = 0.999) -> torch.Tensor:
    """Compute class-balanced weights using effective number of samples."""
    with open(data_file) as f:
        samples = json.load(f)

    counts = [0] * num_labels
    for sample in samples:
        counts[sample["label"]] += 1

    # Effective number of samples: (1 - beta^n) / (1 - beta)
    effective_num = [(1.0 - beta**c) / (1.0 - beta) if c > 0 else 1.0 for c in counts]
    weights = [1.0 / en for en in effective_num]

    # Normalize
    total = sum(weights)
    weights = [w / total * len(weights) for w in weights]

    return torch.tensor(weights, dtype=torch.float32)


def main():
    parser = argparse.ArgumentParser(description="Train Qwen 2.5 VL for face-name classification")

    # Model args
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-VL-7B-Instruct",
                        help="HuggingFace model ID")
    parser.add_argument("--use-4bit", action="store_true", default=True,
                        help="Use 4-bit quantization")
    parser.add_argument("--no-4bit", action="store_false", dest="use_4bit",
                        help="Disable 4-bit quantization")
    parser.add_argument("--use-lora", action="store_true",
                        help="Use LoRA fine-tuning")
    parser.add_argument("--freeze-vision", action="store_true", default=True,
                        help="Freeze vision encoder")

    # LoRA args
    parser.add_argument("--lora-r", type=int, default=64, help="LoRA rank")
    parser.add_argument("--lora-alpha", type=int, default=16, help="LoRA alpha")
    parser.add_argument("--lora-dropout", type=float, default=0.05)

    # Training args
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--class-balanced-beta", type=float, default=None,
                        help="Beta for class-balanced loss (e.g., 0.999)")

    # Data args
    parser.add_argument("--data-dir", default="/home/leann/face-detection/data/qwen_dataset",
                        help="Directory with train.json and val.json")
    parser.add_argument("--output-dir", default="/home/leann/face-detection/results/track1_qwen_vl",
                        help="Output directory for checkpoints and logs")

    # Other args
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    if not HAS_TRANSFORMERS:
        print("Please install required packages:")
        print("  pip install transformers accelerate bitsandbytes peft trl qwen-vl-utils")
        return

    seed_everything(args.seed)

    # Load labels
    labels_file = Path(args.data_dir) / "labels.json"
    with open(labels_file) as f:
        labels = json.load(f)
    num_labels = len(labels)
    print(f"Number of labels: {num_labels}")

    # Load processor
    print(f"Loading processor for {args.model_id}...")
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True)

    # Create model
    print(f"Loading model {args.model_id}...")
    model = Qwen2VLForClassification(
        model_id=args.model_id,
        num_labels=num_labels,
        use_4bit=args.use_4bit,
        freeze_vision=args.freeze_vision,
    )

    # Apply LoRA if requested
    if args.use_lora:
        print("Applying LoRA...")
        model.model = prepare_model_for_kbit_training(model.model)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model.model = get_peft_model(model.model, lora_config)
        model.model.print_trainable_parameters()

    # Load datasets
    print("Loading datasets...")
    train_dataset = FaceNameClassificationDataset(
        data_file=str(Path(args.data_dir) / "train.json"),
        processor=processor,
    )
    val_dataset = FaceNameClassificationDataset(
        data_file=str(Path(args.data_dir) / "val.json"),
        processor=processor,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")

    # Compute class weights if requested
    class_weights = None
    if args.class_balanced_beta:
        print(f"Computing class-balanced weights with beta={args.class_balanced_beta}...")
        class_weights = compute_class_weights(
            str(Path(args.data_dir) / "train.json"),
            num_labels,
            beta=args.class_balanced_beta,
        )
        class_weights = class_weights.to(model.classifier[0].weight.device)

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
        use_4bit=args.use_4bit,
        use_lora=args.use_lora,
        freeze_vision=args.freeze_vision,
        seed=args.seed,
        output_dir=args.output_dir,
        class_balanced_beta=args.class_balanced_beta,
    )
    with open(output_dir / "config.json", "w") as f:
        json.dump(asdict(config), f, indent=2)

    # Save labels
    with open(output_dir / "names.json", "w") as f:
        json.dump(labels, f, indent=2)

    # Create trainer
    trainer = ClassificationTrainer(
        model=model,
        processor=processor,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir=args.output_dir,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_epochs=args.epochs,
        warmup_ratio=args.warmup_ratio,
        class_weights=class_weights,
    )

    # Train
    print("\nStarting training...")
    start_time = time.time()
    training_log = trainer.train()
    elapsed = time.time() - start_time

    print(f"\nTraining complete in {elapsed/60:.1f} minutes")
    print(f"Best accuracy: {max(log['accuracy'] for log in training_log):.4f}")
    print(f"Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
