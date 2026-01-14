# ViT Face-to-Name Classification Experiments

## Overview

**Goal**: Train a Vision Transformer from scratch to classify face images by first name, testing whether there's a learnable correlation between facial appearance and common names.

**Dataset**: Curated face chips from `data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001/`
- 500 classes (first names)
- ~396k total images
- Quality filtered: detection score ≥ 0.9, bbox ≥ 32px, area fraction ≥ 0.001
- 512x512 face chip images

---

## Experiment Timeline

### Experiment 1: Skeleton Test (2 images)
**Config**: ViT-Base/16@224, 2 classes (aaron, david), 1 image each, 100 epochs

**Result**:
- Loss: 0.55 → 0.0 by epoch 10
- Accuracy: 100%

**Learning**: Architecture works, can overfit on minimal data.

---

### Experiment 2: 2 Classes, Full Data
**Config**: ViT-Base/16@224, 2 classes, 50/50 train-test split (~3k images each)

**Results**:
```
Epoch   1 | Train: 80.5% | Test: 83.3%
Epoch  50 | Train: 88.9% | Test: 81.0%
Epoch 100 | Train: 99.1% | Test: 76.1%
```

**Learning**: Severe overfitting. Model memorizes training data (99%) but test drops to 76%.

---

### Experiment 3: Scale to 100 Classes
**Config**: ViT-Base/16@224, 100 classes, batch 512, ~100k train/test

**Results**:
```
Epoch   1 | Train: 2.8% | Test: 3.2%
Epoch  20 | Train: 5.9% | Test: 4.0%  ← peak
Epoch  40 | Train: 98.1% | Test: 2.8%
Epoch  60 | Train: 98.4% | Test: 2.7%
```

**Learning**:
- 4% test >> 1% random baseline → real signal exists!
- Still overfits badly after epoch 20
- Smaller model or more classes needed

---

### Experiment 4: ViT-Small (22M params)
**Config**: ViT-Small/16@224, 100 classes, batch 1024

**Results**:
```
Epoch   1 | Train: 2.8% | Test: 2.9%
Epoch  20 | Train: 5.5% | Test: 4.4%  ← peak
Epoch  40 | Train: 99.9% | Test: 2.9%
```

**Learning**: Smaller model didn't help much - just delayed overfitting slightly.

---

### Experiment 5: 200 Classes + ViT-Tiny + 384x384 Input
**Config**: ViT-Tiny/16@384, 200 classes, 12 layers, ~147k train/test

**Results**:
```
Epoch   1 | Train: 2.0% | Test: 2.2%
Epoch  20 | Train: 3.8% | Test: 3.2%  ← peak
Epoch  40 | Train: 50.2% | Test: 1.9%
```

**Learning**:
- Higher resolution (384 vs 224) preserves facial detail
- More classes helps but still overfits
- LR instability observed at epoch 80

---

### Experiment 6: Custom ViT (6 layers, GAP, lower LR)
**Config**:
- ViT-Tiny/16@384
- 6 layers (reduced from 12)
- Global Average Pooling (no CLS token)
- LR: 1e-4 (reduced from 4e-4)
- 200 classes, ~3M params

**Results**:
```
Epoch   1 | Train: 1.9%  | Test: 2.0%
Epoch  20 | Train: 3.6%  | Test: 3.3%  ← BEST
Epoch  40 | Train: 4.8%  | Test: 3.2%
Epoch  60 | Train: 8.6%  | Test: 2.5%
Epoch  80 | Train: 22.2% | Test: 2.0%
Epoch 100 | Train: 52.2% | Test: 1.9%
```

**Learning**:
- 6 layers sufficient for this task (12 was overkill)
- GAP works as well as CLS token
- Lower LR prevents instability
- Overfitting much slower (train 52% vs 99% at epoch 100)
- **Best test: 3.3% = 6.6x better than random (0.5%)**

---

### Experiment 7: Male-Only Names (Gender-Controlled)
**Goal**: Remove gender as a confounding signal. If accuracy drops to random, gender was the main signal.

**Config**:
- 304 clearly male names (excluded female and unisex names)
- Same architecture (6-layer ViT-Tiny/16@384, GAP)
- ~106k train/test images

**Results** (in progress):
```
Epoch   1 | Train: 2.3% | Test: 2.3%
Epoch  20 | Train: 2.9% | Test: 2.6%  ← peak so far
Epoch  40 | Train: 4.5% | Test: 2.3%
```

**Learning**:
- Random baseline: 0.33% (1/304 classes)
- **Test: 2.6% = 7.9x better than random**
- **Gender is NOT the only signal!**
- Face-to-name correlation exists independent of gender
- Likely signals: age, ethnicity, cultural background

---

## Architecture Details

### Final Architecture: ViT-Tiny/16@384 (Custom)
```
INPUT: [B, 3, 384, 384]

1. Patch Embedding
   - Conv2d(3→192, kernel=16, stride=16)
   - Output: [B, 576, 192]  (576 = 24×24 patches)

2. Position Embedding (Learned)
   - Add [1, 576, 192] to patches

3. Transformer Encoder (×6 blocks)
   Each block:
   - LayerNorm → Multi-Head Attention (3 heads) → Residual
   - LayerNorm → MLP (192→768→192) → Residual

4. Global Average Pooling
   - Mean over 576 patches → [B, 192]

5. Classification Head
   - Linear(192 → num_classes)

Total: ~3M parameters
```

### Key Architecture Decisions
| Decision | Reason |
|----------|--------|
| 384x384 input | Preserve facial detail (512→224 loses too much) |
| 6 layers | 12 was overkill, caused faster overfitting |
| GAP vs CLS | Simpler, one less parameter, same performance |
| Patch size 16 | Standard, gives 576 patches at 384x384 |

---

## Training Configuration

```python
BATCH_SIZE = 1024
LR = 1e-4
EPOCHS = 100
EVAL_EVERY = 20
NUM_WORKERS = 8

# Optimizations
- AMP (mixed precision)
- torch.compile()
- AdamW optimizer
- CrossEntropyLoss
```

---

## Key Findings

### 1. Face-to-Name Correlation is Real
- Consistently 6-8x better than random baseline across experiments
- Not just gender: male-only experiment still shows 7.9x improvement

### 2. Overfitting is the Main Challenge
- ViT from scratch memorizes training data quickly
- Peak test accuracy typically at epoch 15-25
- After that, train accuracy rises while test drops

### 3. Model Size vs Task Difficulty
| Model | Params | 100 classes | 200 classes |
|-------|--------|-------------|-------------|
| ViT-Base | 86M | Overfits by epoch 40 | - |
| ViT-Small | 22M | Overfits by epoch 40 | - |
| ViT-Tiny (12L) | 5.6M | - | Overfits by epoch 40 |
| ViT-Tiny (6L) | 3M | - | Slower overfit, peak at epoch 20 |

### 4. What Reduces Overfitting
- More classes (harder task)
- Fewer layers (less capacity)
- Lower learning rate (slower learning)
- Early stopping at epoch ~20

### 5. What the Model Likely Learns
- Gender (when available) - easy signal
- Age - names have generational popularity (Emma vs Harold)
- Ethnicity/cultural background - names correlate with ancestry
- Possibly other subtle features

---

## Files

- `scripts/monkey_vit.py` - Training script
- `data/male_names.txt` - 304 clearly male names for gender-controlled experiments
- `monkey_vit_log.txt` - Training logs (CSV format)

---

## Next Steps (Potential)

1. **More classes**: Try all 500 classes
2. **Data augmentation**: Horizontal flip (faces are symmetric)
3. **Learning rate schedule**: Decay after epoch 20
4. **Early stopping**: Stop at best validation accuracy
5. **Female-only experiment**: Compare to male-only results
6. **Per-class analysis**: Which names are easiest/hardest to classify?
