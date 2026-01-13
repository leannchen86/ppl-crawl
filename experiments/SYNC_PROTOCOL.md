# Multi-Branch Experiment Sync Protocol

## Overview

This project is running **3 parallel experiment tracks** in separate chat branches. This document defines how to coordinate and sync results.

---

## Active Tracks

| Track | Focus | Branch | Status |
|-------|-------|--------|--------|
| Track 1 | Qwen 2.5 VL Fine-tuning | Branch 1 | Pending |
| Track 2 | ViT from Scratch (435k images) | Branch 2 | Pending |
| Track 3 | Quick Experiments (DINOv2, etc.) | Branch 3 | Pending |

---

## Shared Context

### Data Location (CANONICAL)
```
/home/leann/face-detection/data/index_files_facechips512_m0.5_reflect/
```

**All tracks MUST use this dataset** for fair comparison. This contains:
- 512x512 face chips (standardized, face-focused)
- Reflect padding (no black borders)
- 0.5 margin around detected face

Alternative (DO NOT USE for main experiments):
```
/home/leann/face-detection/data/index_files/  # Original variable-size images
```

### Data Statistics
- **Total images:** 445,356
- **Total names:** 500
- **Image size:** 512x512 (all standardized)
- **Split:** 80% train / 20% val (by hash for reproducibility)

### Baseline Performance (for comparison)
| Model | Names | Accuracy | Pred CV | Notes |
|-------|-------|----------|---------|-------|
| CLIP ViT-B-32 linear probe | 30 | 13.9% | 0.400 | William dominates (51%) |
| CLIP ViT-B-32 linear probe | 500 | ~3% | ~0.40 | Not formally tested |
| ArcFace linear probe | 30 | ~12% | ~0.30 | More balanced |
| CNN from scratch | 30 | 10.9% | ~0.35 | Stable training |

---

## Output Directory Structure

Each track saves results to its designated directory:

```
results/
├── track1_qwen_vl/
│   ├── config.json
│   ├── predictions.npy
│   ├── true_labels.npy
│   ├── names.json
│   ├── results.csv
│   └── summary.md
│
├── track2_vit_scratch/
│   ├── config.json
│   ├── best_model.pth
│   ├── predictions.npy
│   ├── true_labels.npy
│   ├── names.json
│   ├── results.csv
│   ├── training_log.json
│   └── summary.md
│
├── track3_dinov2/
├── track3_dinov3/
├── track3_arcface_500/
├── track3_supcon/
├── track3_ensemble/
├── track3_siglip/
└── track3_comparison.md
```

---

## Standardized Output Files

### config.json
```json
{
  "model": "model_name_and_version",
  "num_classes": 500,
  "train_samples": 347622,
  "val_samples": 86906,
  "epochs": 100,
  "batch_size": 256,
  "learning_rate": 0.001,
  "optimizer": "AdamW",
  "loss": "CrossEntropyLoss",
  "class_weights": "sqrt_inverse",
  "augmentation": ["RandomResizedCrop", "HorizontalFlip", "ColorJitter"],
  "hardware": "NVIDIA A100 80GB",
  "training_time_hours": 12.5
}
```

### predictions.npy / true_labels.npy
```python
# Shape: (num_val_samples,)
# dtype: int64
# Values: class indices 0-499
predictions = np.load("predictions.npy")
true_labels = np.load("true_labels.npy")
```

### names.json
```json
["alex", "david", "michael", "laura", ...]  // 500 names, index = class label
```

### results.csv
```csv
name,recall,precision,f1,support,predicted_count,prediction_ratio
william,0.24,0.18,0.20,488,676,1.39
mark,0.17,0.15,0.16,474,629,1.33
...
```

### summary.md
```markdown
# Track X: [Experiment Name] Summary

## Key Results
- Overall accuracy: X.X%
- Prediction CV: X.XX
- Training time: X hours

## Top 5 Names (by F1)
1. name1: F1=X.X%, P=X.X%, R=X.X%
...

## Bottom 5 Names (by F1)
...

## Key Observations
- [What worked]
- [What didn't work]
- [Surprising findings]

## Recommendations
- [Next steps based on results]
```

---

## Merge Checklist

When merging branch results back to main:

### 1. Verify Outputs Exist
```bash
ls results/trackX_*/config.json
ls results/trackX_*/predictions.npy
ls results/trackX_*/summary.md
```

### 2. Run Comparison Script
```python
# compare_all_tracks.py
import json
import numpy as np
from pathlib import Path

tracks = [
    "track1_qwen_vl",
    "track2_vit_scratch",
    "track3_dinov2",
    "track3_arcface_500",
    # ... etc
]

results = []
for track in tracks:
    track_dir = Path(f"results/{track}")
    if not track_dir.exists():
        continue

    config = json.load(open(track_dir / "config.json"))
    preds = np.load(track_dir / "predictions.npy")
    labels = np.load(track_dir / "true_labels.npy")

    accuracy = (preds == labels).mean()
    pred_cv = np.bincount(preds).std() / np.bincount(preds).mean()

    results.append({
        "track": track,
        "model": config.get("model", "unknown"),
        "accuracy": accuracy,
        "pred_cv": pred_cv,
        "training_time": config.get("training_time_hours", "N/A")
    })

# Print comparison table
print("| Track | Model | Accuracy | Pred CV | Time |")
print("|-------|-------|----------|---------|------|")
for r in sorted(results, key=lambda x: -x["accuracy"]):
    print(f"| {r['track']} | {r['model']} | {r['accuracy']:.1%} | {r['pred_cv']:.3f} | {r['training_time']} |")
```

### 3. Update Main README
Add results summary to project README after all tracks complete.

---

## Communication Protocol

### Blocking Issues
If a track encounters a blocking issue:
1. Document the issue in `results/trackX/BLOCKED.md`
2. Include: error message, attempted solutions, what's needed

### Early Wins
If a track finds a significant improvement:
1. Document immediately in `results/trackX/BREAKTHROUGH.md`
2. Include: what worked, metrics, reproducibility steps

### Requesting Resources
If a track needs more compute/time:
1. Document in `results/trackX/NEEDS.md`
2. Include: what's needed, expected benefit

---

## Timeline

| Phase | Duration | Activities |
|-------|----------|------------|
| Setup | Day 1 | Environment setup, data loading verification |
| Training | Days 2-5 | Main training runs |
| Evaluation | Day 6 | Generate all metrics, summaries |
| Merge | Day 7 | Combine results, comparison analysis |

---

## Final Deliverables

After all tracks complete:

1. **Comparison Report** (`results/FINAL_COMPARISON.md`)
   - Table of all experiments
   - Best performing approach
   - Statistical significance tests

2. **Best Model Checkpoint**
   - Save to `experiments/best_model/`
   - Include inference script

3. **Recommendations Document**
   - Which direction to pursue further
   - What didn't work and why
   - Estimated ceiling for this task
