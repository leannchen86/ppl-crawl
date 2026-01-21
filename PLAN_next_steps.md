# Face-Name Classification: Next Steps Plan

## Context Summary

### What Was Completed
- Unified linear probe comparison framework created in `scripts/unified/`
- All 6 experiments completed (CLIP, SigLIP, OpenCLIP × balanced/imbalanced)
- Results committed and pushed to GitHub

### Current Best Results
| Model | Architecture | Balanced | Imbalanced | Params (Visual) |
|-------|-------------|----------|------------|-----------------|
| **CLIP** | ViT-B-32 (openai) | 9.24% | **10.98%** | 87.8M |
| SigLIP | ViT-B-16-SigLIP (webli) | 8.06% | 10.36% | 92.9M |
| OpenCLIP | ViT-B-32 (laion2b) | 8.27% | 10.28% | 87.8M |

- Random baseline: 2% (1/50 classes)
- All models used same data: `data/face_chips_512_m0.5_reflect/`
- Validation set: 12,778 samples, holdout checksum `36ad6b2f`
- Training: AdamW, lr=0.01, weight_decay=1e-4, 100 epochs, seed=42

### Key Files
- Training script: `scripts/unified/train_linear_probe.py`
- Runner script: `scripts/unified/run_all_experiments.sh`
- Comparison generator: `scripts/unified/generate_comparison.py`
- Results: `results/linear_probe_comparison/{model}/{balanced,imbalanced}/`
- Cached embeddings: `results/linear_probe_comparison/{model}/{mode}/*_embeddings.npy`

### Hardware Available
- **Zeus (A100)**: 216.218.141.226 - faster for ViT-B-16 models (more patches)
- **Odin (RTX 6000)**: 172.16.1.250 - good for ViT-B-32 models

---

## Phase 1: Analyze Results

**Goal:** Understand current model behavior to inform improvement strategies.

### Task 1.1: Confusion Matrices
**File to create:** `scripts/analysis/generate_confusion_matrices.py`

- Load cached embeddings and labels from `results/linear_probe_comparison/`
- Load trained linear probe checkpoints
- Generate confusion matrix for each model
- Save as heatmap images and CSV
- Identify most confused name pairs

### Task 1.2: Per-Class Analysis
**File to create:** `scripts/analysis/per_class_analysis.py`

- Correlate per-class accuracy with training sample count
- Identify names that are easy (high acc) vs hard (low acc)
- Check if certain names are systematically confused
- Generate report with recommendations

### Task 1.3: Embedding Visualization
**File to create:** `scripts/analysis/visualize_embeddings.py`

- Load validation embeddings (already cached as .npy)
- Apply t-SNE or UMAP dimensionality reduction
- Color by class label
- Generate interactive HTML plot (plotly) or static PNG
- Compare cluster quality across models

### Task 1.4: Error Analysis
**File to create:** `scripts/analysis/error_analysis.py`

- Identify misclassified samples
- Check if errors are consistent across models
- Look for patterns (e.g., similar-looking names, low-quality images)
- Sample and save example error cases for manual review

**Parallelization:** Tasks 1.1-1.4 can all run in parallel (independent analyses)

---

## Phase 2: Try Other Models

**Goal:** Test if larger models improve accuracy significantly.

### Task 2.1: Add Large Models to Registry
**File to modify:** `scripts/unified/train_linear_probe.py`

Add to MODEL_REGISTRY:
```python
"clip-large": {
    "model_name": "ViT-L-14",
    "pretrained": "openai",
    "type": "CLIP-Large",
    "expected_embed_dim": 768,
},
"siglip-large": {
    "model_name": "ViT-L-16-SigLIP",
    "pretrained": "webli",
    "type": "SigLIP-Large",
    "expected_embed_dim": 1024,
},
"openclip-large": {
    "model_name": "ViT-L-14",
    "pretrained": "laion2b_s32b_b82k",
    "type": "OpenCLIP-Large",
    "expected_embed_dim": 768,
},
```

### Task 2.2: Run Large Model Experiments
**Commands:**
```bash
# On Zeus (A100) - better for large models
python scripts/unified/train_linear_probe.py --model clip-large --data-dir data/siglip_imbalanced --output-dir results/linear_probe_comparison/clip-large/imbalanced
python scripts/unified/train_linear_probe.py --model siglip-large --data-dir data/siglip_imbalanced --output-dir results/linear_probe_comparison/siglip-large/imbalanced
python scripts/unified/train_linear_probe.py --model openclip-large --data-dir data/siglip_imbalanced --output-dir results/linear_probe_comparison/openclip-large/imbalanced
```

### Task 2.3: Compare Base vs Large
- Update `generate_comparison.py` to include large models
- Create accuracy vs parameters plot
- Decide if large models are worth the compute cost

**Parallelization:**
- Phase 2 can run in parallel with Phase 1 (different tasks)
- Large model training can run on Zeus while analysis runs on Odin

---

## Phase 3: Improve Accuracy

**Goal:** Squeeze more performance from the best model(s).

### Task 3.1: Hyperparameter Sweep
**File to create:** `scripts/unified/hyperparam_sweep.py`

Test combinations:
- Learning rate: [0.001, 0.005, 0.01, 0.05, 0.1]
- Weight decay: [0, 1e-5, 1e-4, 1e-3]
- Epochs: [50, 100, 200]

Use best model from Phase 2 results.

### Task 3.2: MLP Probe (Instead of Linear)
**File to create:** `scripts/unified/train_mlp_probe.py`

- Copy `train_linear_probe.py` as base
- Replace LinearProbeClassifier with:
```python
class MLPProbeClassifier(nn.Module):
    def __init__(self, input_dim, num_classes, hidden_dim=512):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_classes)
        )
```
- Test hidden_dim: [256, 512, 1024]

### Task 3.3: Data Augmentation During Embedding
**File to create:** `scripts/unified/train_with_augmentation.py`

- Apply augmentations before embedding extraction:
  - Random horizontal flip
  - Random rotation (±10°)
  - Color jitter
- Extract multiple augmented embeddings per image
- Average or use all for training

### Task 3.4: Model Ensemble
**File to create:** `scripts/unified/ensemble_inference.py`

- Load multiple trained probes (CLIP, SigLIP, OpenCLIP)
- Average logits or probabilities
- Test if ensemble beats best single model

**Parallelization:**
- Tasks 3.1-3.4 can run in parallel (independent experiments)
- Each task can use different GPU

**Dependencies:**
- Phase 3 should wait for Phase 2 to identify best base model
- But can start Task 3.2-3.4 with current best (CLIP) while Phase 2 runs

---

## Phase 4: Deploy for Inference

**Goal:** Create usable inference pipeline for new face images.

### Task 4.1: Single-Image Inference Script
**File to create:** `scripts/inference/predict_name.py`

```bash
# Usage
python scripts/inference/predict_name.py --image path/to/face.jpg --model clip
# Output: Top-5 predicted names with probabilities
```

Features:
- Load model and probe from checkpoint
- Preprocess image (resize, normalize)
- Return top-k predictions with confidence scores

### Task 4.2: Batch Inference Script
**File to create:** `scripts/inference/batch_predict.py`

```bash
# Usage
python scripts/inference/batch_predict.py --input-dir faces/ --output predictions.csv
```

Features:
- Process directory of images
- Output CSV with filename, predicted name, confidence
- Progress bar for large batches

### Task 4.3: Package Best Model
**File to create:** `scripts/inference/export_model.py`

- Combine vision encoder + linear probe into single checkpoint
- Save class labels mapping
- Create minimal inference code that doesn't need training dependencies

### Task 4.4: Simple Demo (Optional)
**File to create:** `scripts/demo/gradio_app.py`

- Gradio web interface for quick testing
- Upload image → get name predictions
- Show confidence bar chart

**Dependencies:**
- Phase 4 depends on Phase 3 to identify final best model
- But Task 4.1-4.2 can be developed with current CLIP model, then swap checkpoint later

---

## Execution Timeline

```
Week 1:
├── Phase 1 (Analysis) ──────────────────────────►
│   ├── Task 1.1: Confusion matrices
│   ├── Task 1.2: Per-class analysis
│   ├── Task 1.3: Embedding visualization
│   └── Task 1.4: Error analysis
│
└── Phase 2 (Large Models) ──────────────────────►
    ├── Task 2.1: Add to registry
    ├── Task 2.2: Run experiments
    └── Task 2.3: Compare results

Week 2:
├── Phase 3 (Improve Accuracy) ──────────────────►
│   ├── Task 3.1: Hyperparam sweep
│   ├── Task 3.2: MLP probe
│   ├── Task 3.3: Data augmentation
│   └── Task 3.4: Ensemble
│
└── Phase 4 (Deploy) ────────────────────────────►
    ├── Task 4.1: Single inference
    ├── Task 4.2: Batch inference
    ├── Task 4.3: Package model
    └── Task 4.4: Demo (optional)
```

---

## Quick Start Commands

### To continue from this point:

1. **Start Phase 1 analysis:**
```bash
cd /mnt/disk1/home/leann/face-detection
# Create analysis scripts (see Task 1.1-1.4)
```

2. **Check current results:**
```bash
cat results/linear_probe_comparison/comparison/summary.json
```

3. **View cached embeddings:**
```bash
ls -lh results/linear_probe_comparison/clip/imbalanced/*.npy
```

4. **Re-run comparison after new experiments:**
```bash
python scripts/unified/generate_comparison.py --results-dir results/linear_probe_comparison
```

---

## Notes

- All experiments should use seed=42 for reproducibility
- Imbalanced dataset (115K samples) consistently outperforms balanced (25K)
- CLIP (OpenAI pretrained) beats OpenCLIP (LAION pretrained) despite same architecture
- Laura has anomalously high accuracy (47%) - worth investigating
- Several names have 0% accuracy (mike, julia, rachel, etc.) - need more samples or better features
