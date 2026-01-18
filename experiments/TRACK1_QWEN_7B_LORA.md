# Qwen 2.5 VL 7B Fine-tuning with 4-bit + LoRA

## Experiment Overview

This experiment fine-tunes **Qwen/Qwen2.5-VL-7B-Instruct** for face-name classification using:
- **4-bit quantization (NF4)**: Reduces VRAM from ~40GB to ~14GB
- **LoRA**: Enables actual weight updates (190M trainable params / 8.5B total = 2.24%)
- **Classification head**: 2-layer MLP on mean-pooled hidden states

### Key Improvements Over Previous 3B Experiment
1. Larger model capacity (7B vs 3B)
2. LoRA fine-tuning (previously classifier-only)
3. **Explicit held-out validation** with zero train/val overlap
4. Both balanced and imbalanced data configurations
5. Per-class accuracy tracking

## Data Configuration

### Held-out Validation Strategy
The validation set is **completely independent** from training:
- Created once via `prepare_holdout_dataset.py --create-holdout`
- Same val set used for both balanced and imbalanced experiments
- Enables fair comparison between configurations
- Checksum verification ensures no data leakage

### Datasets Created

| Configuration | Train Samples | Val Samples | Description |
|--------------|---------------|-------------|-------------|
| Balanced | 54,000 | 9,054 | 1,800 per name (equal distribution) |
| Imbalanced | 81,581 | 9,054 | Natural distribution (alex: 4937, ryan: 1887) |

**Paths:**
- Holdout manifest: `/home/leann/face-detection/data/qwen_7b_holdout/`
- Balanced: `/home/leann/face-detection/data/qwen_7b_balanced/`
- Imbalanced: `/home/leann/face-detection/data/qwen_7b_imbalanced/`

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Model | Qwen/Qwen2.5-VL-7B-Instruct |
| Quantization | 4-bit NF4 |
| LoRA rank (r) | 64 |
| LoRA alpha | 16 |
| LoRA targets | q, k, v, o, gate, up, down projections |
| Learning rate | 2e-5 |
| Batch size | 2 |
| Gradient accumulation | 16 |
| Effective batch size | 32 |
| Epochs | 5 |
| Image size | 512x512 (resized) |
| Class-balanced beta | None (balanced) / 0.999 (imbalanced) |

### Trainable Parameters
- LoRA adapters: ~190M params
- Classification head: ~2M params
- Total trainable: ~192M / 8.5B = 2.26%

## Running the Experiments

### Single GPU Sequential (Recommended)
```bash
cd /home/leann/face-detection/scripts/qwen2.5vl
./run_sequential_experiments.sh
```

### Individual Runs
```bash
# Balanced only
./run_sequential_experiments.sh balanced

# Imbalanced only (after balanced completes)
./run_sequential_experiments.sh imbalanced
```

### Manual Training Command
```bash
source /home/leann/face-detection/venv/bin/activate
python train_qwen_7b_lora.py \
    --data-dir /home/leann/face-detection/data/qwen_7b_balanced \
    --output-dir /home/leann/face-detection/results/qwen_7b_lora/balanced \
    --epochs 5 \
    --batch-size 2 \
    --gradient-accumulation-steps 16 \
    --lr 2e-5 \
    --lora-r 64 \
    --lora-alpha 16
```

## Monitoring Training

### Check Progress
```bash
# View recent output
tail -50 /home/leann/face-detection/results/qwen_7b_lora/balanced_training.log

# Check GPU usage
nvidia-smi

# Check if running
ps aux | grep train_qwen
```

### Expected Timeline
- ~1.8 batches/second on H100
- ~4 hours per epoch
- ~20 hours per experiment
- ~40 hours total for both experiments

## Results Location

```
/home/leann/face-detection/results/qwen_7b_lora/
├── balanced/
│   ├── config.json
│   ├── labels.json
│   ├── training_log.json
│   ├── best_model/
│   │   ├── lora_adapter/
│   │   └── classifier.pt
│   └── checkpoint_epoch_*/
└── imbalanced/
    └── (same structure)
```

## Comparing Results

```python
import json

# Load training logs
balanced = json.load(open('results/qwen_7b_lora/balanced/training_log.json'))
imbalanced = json.load(open('results/qwen_7b_lora/imbalanced/training_log.json'))

# Best accuracies
print(f"Balanced best: {max(e['accuracy'] for e in balanced):.4f}")
print(f"Imbalanced best: {max(e['accuracy'] for e in imbalanced):.4f}")

# Per-class comparison
best_balanced = max(balanced, key=lambda x: x['accuracy'])
best_imbalanced = max(imbalanced, key=lambda x: x['accuracy'])
```

## Baselines to Beat

| Model | Accuracy | Notes |
|-------|----------|-------|
| Random | 3.3% | 1/30 |
| CLIP (baseline) | 13.9% | Zero-shot |
| Qwen 3B (classifier only) | 11.8% | Previous experiment |
| **Qwen 7B + LoRA** | TBD | This experiment |

## Research Questions

1. **Does 7B + LoRA outperform 3B classifier-only?**
   - Hypothesis: Yes, due to larger capacity and actual weight updates

2. **Does balanced vs imbalanced data matter with class-balanced loss?**
   - Hypothesis: Class-balanced loss should compensate for imbalance

3. **Which names are hardest to classify?**
   - Check per-class accuracy in training_log.json

4. **Does the model generalize to held-out images?**
   - Validation accuracy on completely unseen images answers this

## Files Created

| File | Description |
|------|-------------|
| `prepare_holdout_dataset.py` | Creates explicit train/val splits with manifests |
| `train_qwen_7b_lora.py` | Training script with LoRA and per-class metrics |
| `run_sequential_experiments.sh` | Runs both experiments sequentially |
| `run_7b_lora_experiments.sh` | Alternative full pipeline script |

## Notes

- Images are resized to 512x512 for consistent tensor shapes
- Using original images (not face chips) since chips directory was removed
- Both experiments use identical held-out validation for fair comparison
- LoRA adapters saved separately for potential model merging later
