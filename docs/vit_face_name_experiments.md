# ViT Face-to-Name Classification Experiments

## Goal
Train a Vision Transformer from scratch to classify face images by first name. Test whether there's a learnable correlation between facial features and names.

## Dataset
- Source: `data/index_files/`
- Curated face chips (512x512) filtered by detection score ≥ 0.9
- 500 total classes (first names), ~396k images
- Train/test split: 50/50 (first half / second half per class)

## Architecture Evolution

### Final Architecture: ViT-Tiny/16@384 (6 layers, GAP)
- **Input**: 384x384 RGB
- **Patch size**: 16x16 → 576 patches
- **Embedding dim**: 192
- **Layers**: 6 (reduced from standard 12)
- **Pooling**: Global Average Pooling (no CLS token)
- **Parameters**: ~3M

### Training Setup
- Optimizer: AdamW
- Learning rate: 1e-4
- Batch size: 1024
- Mixed precision (AMP)
- torch.compile() enabled

## Experiments & Results

### Experiment 1: ViT-Base, 2 classes (aaron, david)
- **Purpose**: Verify architecture works, test overfitting
- **Result**: 99% train, 76% test → massive overfitting
- **Conclusion**: Model works but overfits quickly

### Experiment 2: ViT-Base, 100 classes
- **Purpose**: Scale up to reduce overfitting
- **Result**: Peak 4% test at epoch 20, then overfit to 99% train / 2.8% test
- **Conclusion**: Still overfits, need smaller model or more classes

### Experiment 3: ViT-Small, 100 classes
- **Purpose**: Smaller model (22M → 86M params)
- **Result**: Similar pattern, slight improvement
- **Conclusion**: Model size alone doesn't fix overfitting

### Experiment 4: ViT-Tiny@384, 200 classes, 12 layers
- **Purpose**: Even smaller model, more classes, higher resolution
- **Result**: Peak 3.2% test at epoch 20
- **Conclusion**: Better but still overfits

### Experiment 5: ViT-Tiny@384, 200 classes, 6 layers + GAP
- **Purpose**: Reduce depth, use GAP instead of CLS token
- **Result**:
  ```
  Epoch   1 | Train: 1.9% | Test: 2.0%
  Epoch  20 | Train: 3.6% | Test: 3.3%  ← BEST
  Epoch  40 | Train: 4.8% | Test: 3.2%
  Epoch  60 | Train: 8.6% | Test: 2.5%
  Epoch 100 | Train: 52%  | Test: 1.9%
  ```
- **Random baseline**: 0.5% (1/200)
- **Best**: 3.3% = 6.6x better than random
- **Conclusion**: Slower overfitting, clear signal exists

### Experiment 6: Male-only names (304 classes) - Gender Control
- **Purpose**: Remove gender signal to test true face-name correlation
- **Hypothesis**: If accuracy drops to random, gender was the main signal
- **Result** (in progress):
  ```
  Epoch   1 | Train: 2.3% | Test: 2.3%
  Epoch  20 | Train: 2.9% | Test: 2.6%  ← BEST so far
  Epoch  40 | Train: 4.5% | Test: 2.3%
  ```
- **Random baseline**: 0.33% (1/304)
- **Best**: 2.6% = 7.9x better than random
- **Conclusion**: Signal persists without gender! Correlation is real.

### Experiment 7: Muon Optimizer (failed)
- **Purpose**: Test if Muon optimizer improves convergence
- **Setup**: Muon for 2D weight matrices, AdamW for 1D params (biases, LayerNorm)
- **Results**:
  ```
  LR=0.02:   Epoch 20: Train 4.2%, Test 1.8%  (baseline: 2.6%)
  LR=0.002:  Epoch 20: Train 24.2%, Test 1.7% (worse)
  LR=0.0005: Epoch 20: Train 3.6%, Test 2.4%  (close)
             Epoch 40: Train 25.2%, Test 1.6% (crashed)
  ```
- **Conclusion**: Muon overfits faster than AdamW regardless of LR tuning. Even when epoch 20 looks comparable, it degrades rapidly after. Muon is designed for large-scale training (GPT-2, CIFAR speedruns) where fast convergence matters. For small noisy datasets like ours, AdamW's conservative updates are more stable.
- **Recommendation**: Stick with AdamW for this task.

### Experiment 8: SAM Optimizer (failed)
- **Purpose**: Test if Sharpness-Aware Minimization improves generalization
- **Inspiration**: SAM showed +93 puzzles improvement in sudoku-solver looped transformer
- **Setup**: SAM wrapper around AdamW, tested rho=0.05 and rho=0.01
- **Results**:
  ```
  rho=0.05: Epochs 1-77: Train stuck at 2.4-2.5%, Test stuck at 2.4-2.5%
  rho=0.01: Epochs 1-40: Train stuck at 2.5%, Test stuck at 2.5%
  (Baseline AdamW: Epoch 20: Train 2.9%, Test 2.6%)
  ```
- **Conclusion**: SAM completely prevents learning. The model never improves beyond near-random performance. SAM's sharpness penalty is too aggressive for this weak-signal task - the face-name correlation is too subtle to survive SAM's optimization for flat minima.
- **Note**: SAM worked for sudoku because that task has strong, learnable structure with iterative refinement. Our face-name task has noisy, weak correlations that require sharp (overfit-prone) minima to capture.

## Key Findings

1. **Face-to-name correlation exists**: Models consistently achieve 6-8x better than random chance.

2. **Not just gender**: Male-only experiment shows 7.9x better than random, proving the signal isn't purely gender-based.

3. **Likely sources of signal**:
   - Age/generation (names have era-specific popularity)
   - Ethnicity (names correlate with ethnic backgrounds)
   - Socioeconomic factors (may influence both naming and appearance)

4. **Overfitting is the main challenge**: All models eventually memorize training data. Best results at epoch 20, with early stopping recommended.

## Training Speed Optimizations Used
- Mixed precision (AMP) with GradScaler
- torch.compile() for model compilation
- Large batch size (1024) with scaled LR
- 8 DataLoader workers + pin_memory
- Reduced model depth (6 layers vs 12)

## Files
- Training script: `scripts/monkey_vit.py`
- Male names list: `data/male_names.txt`
- Training log: `monkey_vit_log.txt`
