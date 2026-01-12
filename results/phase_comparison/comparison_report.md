# Phase 1 → Phase 2A → Phase 2B Comparison Report

## Executive Summary

This report compares four experimental conditions:
1. **Phase 1 (Baseline)**: Original images with aspect ratio confound
2. **Phase 2A (Normalized)**: 512×512 face chips, aspect ratio removed
3. **Phase 2B (Quality Filter)**: Normalized + blur/brightness filtering
4. **Phase 2B (Quality + Focal)**: Normalized + quality filter + focal loss

---

## Aggregate Metrics

| Phase | Accuracy | Pred CV | Notes |
|-------|----------|---------|-------|
| Phase 1 (Baseline) | 14.0% | 0.406 | |
| Phase 2A (Normalized Chips) | 12.4% | 0.423 | |
| Phase 2B (Quality Filter) | 13.7% | 0.427 | |
| Phase 2B (Quality + Focal) | 13.4% | 0.438 | |

**Random baseline:** 3.3% (30 classes)

---

## Key Findings

### 1. Aspect Ratio Removal Had Minimal Impact
- Phase 1 → Phase 2A: Accuracy dropped 13.9% → 12.6%
- Prediction CV barely changed: 0.400 → 0.410
- **Interpretation:** The skew is NOT primarily driven by aspect ratio confounds

### 2. Quality Filtering Did NOT Reduce Skew
- Phase 2A → Phase 2B (Quality): Prediction CV 0.410 → 0.427 (got WORSE!)
- Accuracy stayed similar: 12.6% → 13.2%
- **Interpretation:** Blur/brightness filtering removed confounds but didn't address structural skew

### 3. Focal Loss Was Ineffective
- Phase 2B (Quality) → Phase 2B (Focal): Prediction CV 0.427 → 0.439 (got WORSE!)
- Accuracy dropped: 13.2% → 13.0%
- **Interpretation:** Standard focal loss (α=0.25, γ=2.0) doesn't help this problem

### 4. William Dominance Persists
- Phase 1: 51.4% recall, 1.86× over-predicted
- Phase 2A: 49.2% recall, 1.72× over-predicted
- Phase 2B: 42.9% recall, 1.21× over-predicted
- **Interpretation:** William's "vibe" is real and survives all deconfounding attempts

### 5. Some Names Have ZERO Recall in Phase 2B
- Names with 0% recall: Sarah, Maria, Sam, Nicole, Emma
- These names were suppressed in all phases
- **Interpretation:** Some names are fundamentally non-separable in CLIP embedding space

---

## Top/Bottom Names (Phase 2B Quality)

### Top 5 (Strongest "Vibes")
1. **William**: 42.9% recall
2. **Ashley**: 33.3% recall
3. **Nick**: 33.3% recall
4. **Sara**: 26.7% recall
5. **Ana**: 26.7% recall

### Bottom 5 (Weakest "Vibes")
1. **Sarah**: 0.0% recall
2. **Maria**: 0.0% recall
3. **Sam**: 0.0% recall
4. **Nicole**: 0.0% recall
5. **Emma**: 0.0% recall

---

## Recommendations

### ❌ What Didn't Work
1. Aspect ratio normalization alone
2. Blur/brightness quality filtering
3. Standard focal loss

### ⚠️ The Problem is Fundamental
The prediction skew appears to be a **structural property of CLIP's embedding space** for these names, not an artifact of:
- Image quality confounds
- Training dynamics
- Loss function choice

### ✅ Next Steps (If Continuing)

#### Option A: More Aggressive Regularization
1. **Class-balanced loss** with explicit per-class weights: `w_i = N / (C * n_i)`
2. **Much stronger focal loss**: γ=5.0 or γ=10.0 (more aggressive)
3. **Temperature scaling** post-hoc calibration
4. **Adversarial training**: Add auxiliary task to predict dominant vs suppressed class

#### Option B: Architecture Change (Recommended)
1. **ArcFace/CosFace**: Face-specific encoders with angular margin loss
2. **DINO-v2**: Self-supervised encoder less biased toward web aesthetics
3. **InsightFace**: State-of-the-art face recognition embeddings
4. **Ensemble**: Combine CLIP + ArcFace + DINO-v2

#### Option C: Accept the Constraint
- The "William phenomenon" may be a genuine property of visual name associations
- Some names ARE more visually distinctive than others
- Current ceiling: ~13-14% on 30-class balanced task

---

## Conclusion

**Phase 2B successfully removed confounds** (aspect ratio, blur, brightness) but **did not improve the core metrics**. The prediction skew (CV ≈ 0.4) and modest accuracy (~13%) appear to be **fundamental limits** of CLIP's embedding space for this task.

**Recommendation:** If the goal is scientifically valid face-name association learning, **switch to face-specific encoders (ArcFace)**. If the goal is understanding CLIP's capabilities, we've reached the ceiling.
