# Track 2: Vision Transformer (ViT) Training from Scratch

## Project Context

### What We're Trying to Do
Train a Vision Transformer from scratch on our full 435k face dataset to learn face-name associations without relying on pretrained weights that may have embedded biases.

### Why This Matters
- Previous experiments used frozen CLIP/ArcFace embeddings with linear probes
- Those embeddings have baked-in biases (CLIP: web aesthetics, ArcFace: identity verification)
- Training from scratch lets the model learn task-specific features
- 435k images is substantial - original ViT paper showed good results with ~1M+ images

### What We've Tried (Phase 1-3 Summary)
| Approach | Result | Problem |
|----------|--------|---------|
| CLIP linear probe (30 names) | 13.9% acc | Prediction skew, aspect ratio confound |
| Fine-tune pretrained ViT | 11.7% val acc | Catastrophic forgetting (98% train, 9% val) |
| Train CNN from scratch | 10.9% val acc | Stable but low ceiling |
| ArcFace embeddings | ~12% acc | Face-ID features, not name features |

**Key insight:** Pretrained models carry biases. Training from scratch with enough data might learn different (better?) features.

---

## Data Available

### Location
```
/home/leann/face-detection/data/index_files/          # Original images
/home/leann/face-detection/data/index_files_facechips512_m0.5_reflect/  # 512x512 face chips (recommended)
```

### Statistics
- **Total images:** 434,528
- **Total names (classes):** 500
- **Distribution:** Highly unbalanced (power law)
  ```
  Top 10: alex(5960), david(5674), michael(4801), laura(4645), sarah(4473),
          daniel(4217), john(3794), chris(3782), james(3736), amanda(3428)

  Names with >1000 images: 153
  Names with >500 images: 283
  Names with >200 images: 437
  ```

### Recommended: Use ALL 435k images
- Don't balance - use natural distribution with class-weighted loss
- This gives model maximum signal to learn from
- Weight minority classes higher to prevent ignoring them

---

## Why ViT from Scratch

### Hypothesis
1. **Clean slate:** No pretrained biases about faces, compositions, or web aesthetics
2. **Task-specific features:** Model learns exactly what distinguishes names
3. **Scale advantage:** 435k is enough for ViT (papers show ~1M is sweet spot)
4. **Attention patterns:** ViT attention might reveal what features matter for each name

### Architecture Choice

| Model | Params | Patch Size | Notes |
|-------|--------|------------|-------|
| ViT-Tiny | 5.7M | 16 | Too small for 500 classes |
| ViT-Small | 22M | 16 | Good for experimentation |
| **ViT-Base** | 86M | 16 | **Recommended** |
| ViT-Large | 304M | 16 | If Base underfits |

**Recommended: ViT-Base/16** (86M params, 12 layers, 768 hidden dim, 12 heads)

---

## Implementation Plan

### Step 1: Environment Setup
```bash
pip install timm torch torchvision
pip install pytorch-lightning  # Optional but recommended
pip install wandb  # For experiment tracking
```

### Step 2: Dataset Implementation
```python
class FaceNameDataset(Dataset):
    def __init__(self, index_dir, names, transform, split="train", train_ratio=0.8):
        self.samples = []  # (image_path, label_idx)
        self.transform = transform

        for idx, name in enumerate(names):
            index_path = os.path.join(index_dir, f"index_{name}.json")
            with open(index_path) as f:
                data = json.load(f)

            images = data.get("good", [])
            # Split by hash for reproducibility
            train_imgs = [img for img in images if hash(img) % 10 < 8]
            val_imgs = [img for img in images if hash(img) % 10 >= 8]

            selected = train_imgs if split == "train" else val_imgs
            self.samples.extend([(img, idx) for img in selected])

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
```

### Step 3: Model Architecture
```python
import timm

# Option A: Using timm (recommended)
model = timm.create_model(
    'vit_base_patch16_224',
    pretrained=False,  # FROM SCRATCH
    num_classes=500,
    img_size=224,
    drop_rate=0.1,
    drop_path_rate=0.1,
)

# Option B: Custom implementation for more control
from vit_pytorch import ViT

model = ViT(
    image_size=224,
    patch_size=16,
    num_classes=500,
    dim=768,
    depth=12,
    heads=12,
    mlp_dim=3072,
    dropout=0.1,
    emb_dropout=0.1
)
```

### Step 4: Training Configuration

#### Class-Weighted Loss (Critical for Imbalance)
```python
# Compute class weights
class_counts = [count_for_class[i] for i in range(500)]
total = sum(class_counts)

# Option A: Inverse frequency
weights = [total / (500 * count) for count in class_counts]

# Option B: Square root scaling (less aggressive)
weights = [math.sqrt(total / count) for count in class_counts]

# Option C: Effective number (from Class-Balanced Loss paper)
beta = 0.999
weights = [(1 - beta) / (1 - beta ** count) for count in class_counts]

criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights))
```

#### Data Augmentation (Important for Generalization)
```python
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomGrayscale(p=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),  # Cutout-like augmentation
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

#### Training Hyperparameters
```python
# ViT-specific training recipe
config = {
    "batch_size": 256,              # Larger batches help ViT
    "epochs": 100,                  # ViT needs more epochs from scratch
    "base_lr": 1e-3,                # Will be scaled by batch size
    "warmup_epochs": 10,            # Critical for ViT stability
    "weight_decay": 0.05,           # AdamW regularization
    "label_smoothing": 0.1,         # Helps with many classes
    "mixup_alpha": 0.8,             # Optional: mixup augmentation
    "cutmix_alpha": 1.0,            # Optional: cutmix augmentation
    "drop_path_rate": 0.1,          # Stochastic depth
}

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=config["base_lr"] * config["batch_size"] / 256,  # Linear scaling
    weight_decay=config["weight_decay"],
    betas=(0.9, 0.999)
)

# Scheduler (cosine with warmup)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=config["epochs"] - config["warmup_epochs"],
    eta_min=1e-6
)
```

### Step 5: Training Loop Key Points

1. **Warmup:** Linear LR warmup for first 10 epochs (critical for ViT)
2. **Gradient clipping:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`
3. **Mixed precision:** Use `torch.cuda.amp` for faster training
4. **Checkpointing:** Save best model by validation accuracy

### Step 6: Evaluation
Save results in standardized format:
```
results/track2_vit_scratch/
├── config.json           # Training hyperparameters
├── best_model.pth        # Best checkpoint
├── predictions.npy       # (N,) predicted class indices
├── true_labels.npy       # (N,) ground truth indices
├── names.json            # List of 500 name strings
├── results.csv           # Per-name precision/recall/F1
├── attention_maps/       # Optional: visualize attention
├── embeddings.npy        # Optional: CLS token embeddings
└── training_log.json     # Loss/accuracy curves
```

---

## Experiments to Run

### Experiment 2.1: Baseline ViT-Base (Full Data)
- All 435k images, class-weighted cross-entropy
- 100 epochs, cosine LR schedule
- **Goal:** Establish baseline for from-scratch training

### Experiment 2.2: With Mixup/CutMix
- Same as 2.1 but with mixup_alpha=0.8, cutmix_alpha=1.0
- **Goal:** Test if regularization helps with class imbalance

### Experiment 2.3: Label Smoothing Ablation
- Test α = 0.0, 0.1, 0.2
- **Goal:** Find optimal smoothing for 500-class problem

### Experiment 2.4: Subset Experiments (if compute limited)
- Train on top-100 names only (most data)
- Train on top-30 names (for comparison with CLIP baseline)
- **Goal:** Faster iteration, compare with prior results

---

## Key References

1. [lucidrains/vit-pytorch](https://github.com/lucidrains/vit-pytorch) - Clean ViT implementation
2. [timm library](https://github.com/huggingface/pytorch-image-models) - Production-ready models
3. [DeiT paper](https://arxiv.org/abs/2012.12877) - Training ViT without large-scale pretraining
4. [Class-Balanced Loss](https://arxiv.org/abs/1901.05555) - Effective weighting for imbalanced data

---

## Success Criteria

| Metric | Baseline (CLIP) | CNN from Scratch | Target |
|--------|-----------------|------------------|--------|
| Accuracy (500 names) | N/A | ~3% | >5% |
| Accuracy (30 names) | 13.9% | 10.9% | >15% |
| Prediction CV | 0.40 | ~0.35 | <0.30 |
| Training stability | N/A | Stable | No divergence |

---

## Sync Protocol

When done, update:
1. `results/track2_vit_scratch/` with all outputs
2. Create `results/track2_vit_scratch/summary.md` with findings
3. Key metrics to report:
   - Final train/val accuracy curves
   - Best validation accuracy and epoch
   - Per-class accuracy distribution
   - Prediction CV
   - Top 5 / Bottom 5 names by F1
   - Training time, GPU memory, throughput (images/sec)
   - Attention visualization for a few examples (optional but interesting)
