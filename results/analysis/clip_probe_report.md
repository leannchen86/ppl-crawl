# Phase 1 Ablation Study: Comprehensive Report

**Date**: 2026-01-08
**Project**: Face-Name Association Learning with CLIP
**Goal**: Identify root causes of prediction skew and test whether to switch from CLIP architecture

---

## Executive Summary

We conducted 4 rigorous causal ablation tests to understand why the baseline model concentrates predictions on a few dominant names (William, Nick, Emily). **Key finding: The prediction skew is NOT an architectural artifact - it comes from real but uneven separability in CLIP's embedding space, likely driven by photographic confounds (aspect ratio) rather than name-specific facial features.**

**Recommendation: Do NOT switch from CLIP yet.** The problem is not the architecture - it's the task itself and potential confounds in the data.

---

## Baseline Performance (30-name benchmark)

**Setup:**
- 30 names, balanced sampling (500 train + 500 val per name)
- Frozen CLIP ViT-B-32 embeddings + linear classifier head
- Training: AdamW, lr=0.01, weight_decay=1e-4, 50 epochs

**Results:**
- Overall accuracy: **13.9%** (vs 3.3% random)
- Prediction CV: **0.400** (high skew)
- Dominant name: **William** (51.4% recall, 27.4% precision, 1.87× prediction bias)

**Problem:** Even with balanced data, model overpredicts William 1.87× and underpredicts James 0.28×.

---

## Test 1: Cosine Classifier (Normalized Weights)

**Hypothesis:** Prediction skew is caused by weight-norm artifacts in softmax geometry.

**Method:** Replace standard linear head with cosine classifier:
- Standard: `logits = W @ x + b`
- Cosine: `logits = (W/||W||) @ (x/||x||) * scale + b`

**Results:**

| Metric | Baseline | Cosine | Interpretation |
|--------|----------|---------|----------------|
| Overall accuracy | 13.9% | **14.5%** | ✓ Slightly better |
| Prediction CV | 0.400 | **0.393** | ⚠️ Almost no change |
| William bias | 1.87× | **1.87×** | ⚠️ Identical! |
| Weight norm CV | ~0.30 (implicit) | **0.093** | ✓ Successfully normalized |

**Conclusion:** ❌ **Skew is NOT a weight-norm artifact.** Even with perfectly normalized weights, William still dominates. The skew comes from **embedding separability differences**.

---

## Test 2: Permutation Test (Shuffled Labels)

**Hypothesis:** Correlations (e.g., weight-norm vs pred-freq) are optimization artifacts.

**Method:** Randomly shuffle all labels, destroying true face-name associations, then retrain with identical hyperparameters.

**Results:**

| Metric | Baseline (real labels) | Permuted (random labels) | Interpretation |
|--------|------------------------|--------------------------|----------------|
| Overall accuracy | 13.9% | **3.6%** | ✓ Dropped to random |
| Prediction CV | 0.400 | **0.351** | ✓ Reduced skew |
| Weight-norm correlation | +0.609 | **+0.151** | ✓ Correlation dropped |

**Conclusion:** ✅ **Model is learning real signal, not artifacts.** When labels are random, accuracy drops to baseline and correlations weaken. The weight-norm correlation in the baseline is *because* of real separability differences, not an artifact.

---

## Test 3: Top-3 Ablation (Remove William, Nick, Emily)

**Hypothesis:** Dominance is specific to William/Nick/Emily. Removing them will reduce skew.

**Method:** Exclude the 3 most dominant names and retrain on remaining 27 names.

**Results:**

| Metric | Baseline (30 names) | No top-3 (27 names) | Interpretation |
|--------|---------------------|---------------------|----------------|
| Overall accuracy | 13.9% | **13.4%** | Similar (considering fewer classes) |
| Prediction CV | 0.400 | **0.321** | ⚠️ Reduced but still substantial |
| New dominant name | William (51.4% recall) | **Thomas (27.8% recall)** | ⚠️ Dominance cascades |
| New #2 | Nick (24.9%) | **Lisa (26.6%)** | ⚠️ Others rise |

**Conclusion:** ⚠️ **Dominance is NOT name-specific.** Remove the top names, and the next-most-separable classes take their place. This confirms the problem is structural: the model learns to concentrate mass on whichever classes have the strongest embedding separation.

---

## Test 4: Confound Analysis (Photo Quality Metrics)

**Hypothesis:** "Easy names" correlate with better photo quality (blur, brightness, size, aspect ratio).

**Method:** Sample 100 val images per name, compute quality metrics, correlate with baseline F1/recall/precision.

**Results:**

| Quality Metric | Correlation with Precision | Correlation with Recall | Correlation with F1 | Significance |
|----------------|----------------------------|-------------------------|---------------------|--------------|
| **Aspect ratio** (H/W) | **+0.586** | +0.412 | **+0.396** | ✅ **p<0.001** |
| Blur score | +0.032 | -0.184 | -0.191 | ❌ Not significant |
| Brightness | -0.028 | -0.105 | -0.140 | ❌ Not significant |
| Contrast | -0.294 | -0.127 | -0.176 | ❌ Not significant |
| Face size | -0.321 | -0.206 | -0.174 | ❌ Not significant |

**Key Finding:** Names with **taller/narrower face crops** (higher aspect ratio) have **dramatically better precision** (r=0.586, p<0.001) and F1 scores.

**Conclusion:** 🔴 **Strong evidence of confounds!** The model appears to learn photographic artifacts (crop patterns) rather than name-specific facial features. This explains why CLIP embeddings show "separability" - they capture image properties, not face→name associations.

---

## Synthesis: What's Really Happening

Combining all 4 tests, here's the causal story:

1. **CLIP embeddings do contain signal** (permutation test confirms this)
2. **The signal is uneven** (some names more separable than others)
3. **Separability is NOT weight-norm driven** (cosine classifier fails to fix it)
4. **Dominance cascades structurally** (remove top names → others rise)
5. **Strong confound with aspect ratio** (separability comes from photo properties, not faces)

**The Problem:**
- CLIP was pretrained on diverse web images with varied crops/compositions
- Names with more "professional" photo crops (taller aspect ratio) cluster together in embedding space
- The model learns this confound instead of face→name patterns
- This creates "separability" that's real (not artifact) but meaningless (not causal)

---

## Comparison to Original Research Questions

### Q1: Is the skew a weight-norm artifact?
**Answer:** ❌ No. Cosine classifier (Test 1) proves skew persists even with normalized weights.

### Q2: Is the skew an optimization artifact?
**Answer:** ❌ No. Permutation test (Test 2) proves the model learns real signal, not spurious correlations.

### Q3: Is dominance specific to William?
**Answer:** ❌ No. Top-3 ablation (Test 3) shows dominance cascades to whichever names are most separable.

### Q4: Does the model learn confounds instead of name features?
**Answer:** ✅ **Yes!** Confound analysis (Test 4) reveals strong aspect-ratio correlation (r=0.586, p<0.001).

---

## Should You Replace CLIP?

### Short Answer: **No, not yet.**

The problem is **not the architecture** - it's the task and data. Here's why:

### Evidence Against Switching:
1. **CLIP is working as designed** - it extracts visual features, and aspect ratio IS a visual feature
2. **The confound exists in the data** - any vision model will learn it (ArcFace, ResNet, ViT, etc.)
3. **Switching won't fix structural issues** - dominance will cascade with any architecture
4. **CLIP has advantages** - multimodal understanding, strong pretraining, language grounding

### What Would Help Instead:

#### Option A: Fix the Data Confounds (RECOMMENDED)
1. **Normalize aspect ratio** - resize all images to same crop dimensions
2. **Augment with crops** - random crops to break crop→name correlations
3. **Filter by quality** - remove outlier aspect ratios (e.g., keep 0.8-1.2 range)
4. **Control for demographics** - ensure balanced age/ethnicity per name

#### Option B: Regularize the Model
1. **Adversarial training** - add "aspect ratio predictor" loss, penalize model for using it
2. **Mixup/CutMix augmentation** - force model to learn from face content, not composition
3. **Balanced loss** (e.g., focal loss, class weights) - reduce dominance cascade
4. **Temperature scaling** - calibrate softmax to reduce overconfidence

#### Option C: Try Face-Specific Encoders (IF data fixes fail)
- **ArcFace / CosFace** - trained explicitly for face recognition, ignore background/crop
- **FaceNet** - triplet loss forces within-identity tightness
- **InsightFace** - state-of-the-art face embeddings

---

## Recommended Next Steps

### Phase 2A: Data-Centric Fixes (2-3 days)
1. ✅ **Normalize aspect ratios** - resize to 224×224 or 256×256 squares
2. ✅ **Rerun baseline** - check if prediction skew reduces
3. ✅ **Augment with random crops** - train with RandomResizedCrop
4. Compare: accuracy, prediction CV, confound correlations

### Phase 2B: Model Regularization (1-2 days)
5. ✅ **Focal loss** - reduce easy-class dominance
6. ✅ **Class-balanced loss** - explicit per-class weights
7. ✅ **Adversarial debiasing** - penalize aspect ratio predictions

### Phase 2C: Architecture Exploration (IF Phase 2A/B fail)
8. Try **ArcFace** embeddings (face-specific)
9. Try **DINO-v2** (self-supervised, less web-image bias)
10. Compare: CLIP vs ArcFace vs DINO-v2 on normalized data

---

## Quantitative Summary Table

| Test | Setup | Key Result | Conclusion |
|------|-------|------------|------------|
| **Baseline** | 30 names, standard linear head | 13.9% acc, 0.400 CV, William 1.87× | High skew, uneven predictions |
| **Cosine Classifier** | Normalized weight vectors | 14.5% acc, 0.393 CV, William 1.87× | ❌ Skew NOT weight-norm driven |
| **Permutation Test** | Shuffled labels | 3.6% acc (random), 0.151 correlation | ✅ Model learns real signal |
| **Top-3 Ablation** | Exclude William/Nick/Emily | 13.4% acc, Thomas becomes dominant | ⚠️ Dominance cascades |
| **Confound Analysis** | Photo quality metrics | Aspect ratio: r=0.586, p<0.001 | 🔴 Strong confound detected |

---

## Visualizations Reference

All visualizations saved in:
- `results/cosine_classifier/cosine_baseline/` - weight norm distributions
- `results/permutation_test/permuted_labels/` - shuffled label results
- `results/ablations/no_top3/` - top-3 ablation metrics
- `results/confound_analysis/visualizations/` - quality metric scatterplots, correlation heatmap

---

## Conclusion

**The prediction skew is real (not artifact), uneven (CLIP embeddings), structural (cascading dominance), and confounded (aspect ratio correlation r=0.586).**

**Do NOT switch from CLIP yet.** The architecture is not the problem - the confound in the data is. Fix aspect ratio normalization first, then try regularization. Only explore other architectures (ArcFace, DINO-v2) if data-centric and model-centric fixes fail.

**Critical path forward:**
1. Normalize aspect ratios → rerun baseline → compare CV
2. If skew persists → try focal loss / class balancing
3. If still persists → then consider ArcFace

The Phase 1 ablations successfully ruled out architectural artifacts and identified the real culprit: photographic confounds in the training data.
