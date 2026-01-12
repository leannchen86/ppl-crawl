"""
Compare Phase 1, Phase 2A, and Phase 2B results.

Generates comprehensive comparison report with visualizations.
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def load_phase_data():
    """Load results from all phases."""

    # Phase 1 (baseline)
    phase1 = pd.read_csv("results/scale_up_results/precision_recall_metrics.csv")
    phase1 = phase1.rename(columns={'name': 'name', 'recall': 'phase1_recall',
                                     'predicted_count': 'phase1_pred_count'})

    # Phase 2A (normalized face chips)
    phase2a = pd.read_csv("results/scale_up_results_facechips512/precision_recall_metrics.csv")
    phase2a = phase2a.rename(columns={'recall': 'phase2a_recall',
                                       'predicted_count': 'phase2a_pred_count'})

    # Phase 2B Quality Filtering Only
    phase2b_quality = pd.read_csv("results/phase2b_quality_only/results.csv")
    phase2b_quality = phase2b_quality.rename(columns={'recall': 'phase2b_quality_recall',
                                                       'predicted_count': 'phase2b_quality_pred_count'})

    # Phase 2B Quality + Focal Loss
    phase2b_focal = pd.read_csv("results/phase2b_quality_filtered/results.csv")
    phase2b_focal = phase2b_focal.rename(columns={'recall': 'phase2b_focal_recall',
                                                   'predicted_count': 'phase2b_focal_pred_count'})

    # Merge all
    df = phase1[['name', 'phase1_recall', 'phase1_pred_count', 'support']].copy()
    df = df.merge(phase2a[['name', 'phase2a_recall', 'phase2a_pred_count']], on='name', how='left')
    df = df.merge(phase2b_quality[['name', 'phase2b_quality_recall', 'phase2b_quality_pred_count']],
                  on='name', how='left')
    df = df.merge(phase2b_focal[['name', 'phase2b_focal_recall', 'phase2b_focal_pred_count']],
                  on='name', how='left')

    return df


def calculate_metrics(df):
    """Calculate aggregate metrics for each phase."""
    metrics = {}

    # Phase 1
    metrics['Phase 1 (Baseline)'] = {
        'accuracy': df['phase1_recall'].mean(),
        'pred_cv': df['phase1_pred_count'].std() / df['phase1_pred_count'].mean()
    }

    # Phase 2A
    metrics['Phase 2A (Normalized Chips)'] = {
        'accuracy': df['phase2a_recall'].mean(),
        'pred_cv': df['phase2a_pred_count'].std() / df['phase2a_pred_count'].mean()
    }

    # Phase 2B Quality
    metrics['Phase 2B (Quality Filter)'] = {
        'accuracy': df['phase2b_quality_recall'].mean(),
        'pred_cv': df['phase2b_quality_pred_count'].std() / df['phase2b_quality_pred_count'].mean()
    }

    # Phase 2B Focal
    metrics['Phase 2B (Quality + Focal)'] = {
        'accuracy': df['phase2b_focal_recall'].mean(),
        'pred_cv': df['phase2b_focal_pred_count'].std() / df['phase2b_focal_pred_count'].mean()
    }

    return metrics


def plot_recall_comparison(df, output_dir):
    """Plot recall comparison across phases."""
    fig, ax = plt.subplots(figsize=(16, 10))

    # Sort by Phase 1 recall
    df_sorted = df.sort_values('phase1_recall', ascending=True)

    x = np.arange(len(df_sorted))
    width = 0.2

    ax.barh(x - 1.5*width, df_sorted['phase1_recall'] * 100, width,
            label='Phase 1 (Baseline)', alpha=0.8, color='#1f77b4')
    ax.barh(x - 0.5*width, df_sorted['phase2a_recall'] * 100, width,
            label='Phase 2A (Normalized)', alpha=0.8, color='#ff7f0e')
    ax.barh(x + 0.5*width, df_sorted['phase2b_quality_recall'] * 100, width,
            label='Phase 2B (Quality)', alpha=0.8, color='#2ca02c')
    ax.barh(x + 1.5*width, df_sorted['phase2b_focal_recall'] * 100, width,
            label='Phase 2B (Quality+Focal)', alpha=0.8, color='#d62728')

    ax.set_yticks(x)
    ax.set_yticklabels(df_sorted['name'])
    ax.set_xlabel('Recall (%)', fontsize=12)
    ax.set_title('Recall Comparison Across All Phases', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(axis='x', alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{output_dir}/recall_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_prediction_bias_comparison(df, output_dir):
    """Plot prediction bias comparison."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    phases = [
        ('Phase 1 (Baseline)', 'phase1_pred_count'),
        ('Phase 2A (Normalized)', 'phase2a_pred_count'),
        ('Phase 2B (Quality)', 'phase2b_quality_pred_count'),
        ('Phase 2B (Quality+Focal)', 'phase2b_focal_pred_count')
    ]

    for idx, (title, col) in enumerate(phases):
        ax = axes[idx // 2, idx % 2]

        df_sorted = df.sort_values(col, ascending=False)
        expected = df_sorted['support'].mean()

        x = np.arange(len(df_sorted))
        bars = ax.bar(x, df_sorted[col], alpha=0.7, color='steelblue', edgecolor='black', linewidth=0.5)
        ax.axhline(expected, color='red', linestyle='--', linewidth=2, label=f'Expected ({expected:.0f})')

        # Color top/bottom
        max_idx = df_sorted[col].idxmax()
        min_idx = df_sorted[col].idxmin()
        bars[df_sorted.index.get_loc(max_idx)].set_color('darkred')
        bars[df_sorted.index.get_loc(min_idx)].set_color('darkgreen')

        ax.set_xticks(x[::2])
        ax.set_xticklabels(df_sorted['name'].iloc[::2], rotation=45, ha='right', fontsize=8)
        ax.set_ylabel('Predicted Count', fontsize=10)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.legend()
        ax.grid(axis='y', alpha=0.3)

        # Add CV annotation
        cv = df_sorted[col].std() / df_sorted[col].mean()
        ax.text(0.95, 0.95, f'CV = {cv:.3f}', transform=ax.transAxes,
                ha='right', va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_bias_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()


def plot_william_analysis(df, output_dir):
    """Special analysis of William's dominance across phases."""
    william = df[df['name'] == 'william'].iloc[0]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Recall
    recalls = [
        william['phase1_recall'] * 100,
        william['phase2a_recall'] * 100,
        william['phase2b_quality_recall'] * 100,
        william['phase2b_focal_recall'] * 100
    ]

    phases = ['Phase 1\n(Baseline)', 'Phase 2A\n(Normalized)',
              'Phase 2B\n(Quality)', 'Phase 2B\n(Qual+Focal)']

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    bars1 = ax1.bar(phases, recalls, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax1.set_ylabel('Recall (%)', fontsize=12)
    ax1.set_title('William Recall Across Phases', fontsize=13, fontweight='bold')
    ax1.set_ylim([0, 60])
    ax1.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars1, recalls):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                f'{val:.1f}%', ha='center', va='bottom', fontweight='bold')

    # Prediction ratio
    pred_ratios = [
        william['phase1_pred_count'] / (william['support']),
        william['phase2a_pred_count'] / (william['support']),
        william['phase2b_quality_pred_count'] / (william['support']),
        william['phase2b_focal_pred_count'] / (william['support'])
    ]

    bars2 = ax2.bar(phases, pred_ratios, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    ax2.axhline(1.0, color='red', linestyle='--', linewidth=2, label='Fair (1.0×)')
    ax2.set_ylabel('Prediction Ratio (actual/expected)', fontsize=12)
    ax2.set_title('William Prediction Bias Across Phases', fontsize=13, fontweight='bold')
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars2, pred_ratios):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                f'{val:.2f}×', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.savefig(f"{output_dir}/william_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()


def generate_report(df, metrics, output_dir):
    """Generate markdown report."""
    report = """# Phase 1 → Phase 2A → Phase 2B Comparison Report

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
"""

    for phase, m in metrics.items():
        acc = m['accuracy'] * 100
        cv = m['pred_cv']
        report += f"| {phase} | {acc:.1f}% | {cv:.3f} | |\n"

    report += f"""
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
"""

    with open(f"{output_dir}/comparison_report.md", "w") as f:
        f.write(report)

    return report


def main():
    output_dir = "results/phase_comparison"
    os.makedirs(output_dir, exist_ok=True)

    print("Loading data from all phases...")
    df = load_phase_data()

    print("Calculating metrics...")
    metrics = calculate_metrics(df)

    print("Generating visualizations...")
    plot_recall_comparison(df, output_dir)
    plot_prediction_bias_comparison(df, output_dir)
    plot_william_analysis(df, output_dir)

    print("Generating report...")
    report = generate_report(df, metrics, output_dir)

    print("\n" + "="*70)
    print("COMPARISON COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {output_dir}/")
    print("  - comparison_report.md")
    print("  - recall_comparison.png")
    print("  - prediction_bias_comparison.png")
    print("  - william_analysis.png")

    # Print summary
    print("\n" + "="*70)
    print("SUMMARY METRICS")
    print("="*70)
    for phase, m in metrics.items():
        print(f"\n{phase}:")
        print(f"  Accuracy: {m['accuracy']*100:.1f}%")
        print(f"  Prediction CV: {m['pred_cv']:.3f}")


if __name__ == "__main__":
    main()
