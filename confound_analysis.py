"""
Confound Analysis: Check if dominant names correlate with photo quality metrics.

Why:
  - If "easy names" (high F1) have systematically better photo quality, the model
    may be learning confounds (e.g., professional photos → better features)
    rather than name-specific signals
  - We measure: blur, brightness, contrast, face size, aspect ratio

Design:
  - Sample N images per name (default: 100) from validation split
  - Compute quality metrics using OpenCV
  - Correlate per-name metrics with per-name F1/recall/precision from baseline

Outputs:
  - results/confound_analysis/
      - quality_metrics.csv (per-image metrics)
      - per_name_summary.csv (aggregated per name)
      - correlations.json
      - visualizations/*.png

Usage:
  python confound_analysis.py --samples-per-name 100
"""

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def compute_blur_score(image: np.ndarray) -> float:
    """Laplacian variance (higher = sharper)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(image: np.ndarray) -> float:
    """Mean pixel intensity (0-255)."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(gray.mean())


def compute_contrast(image: np.ndarray) -> float:
    """Standard deviation of pixel intensities."""
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return float(gray.std())


def compute_face_size(image: np.ndarray) -> int:
    """Number of pixels in image (proxy for face size if images are cropped faces)."""
    return image.shape[0] * image.shape[1]


def compute_aspect_ratio(image: np.ndarray) -> float:
    """Height / width."""
    h, w = image.shape[:2]
    return float(h / w) if w > 0 else 1.0


@dataclass
class ImageQualityMetrics:
    name: str
    path: str
    blur_score: float
    brightness: float
    contrast: float
    face_size: int
    aspect_ratio: float


def analyze_image(path: str, name: str) -> ImageQualityMetrics:
    """Compute all quality metrics for one image."""
    img = Image.open(path).convert("RGB")
    img_np = np.array(img)

    return ImageQualityMetrics(
        name=name,
        path=path,
        blur_score=compute_blur_score(img_np),
        brightness=compute_brightness(img_np),
        contrast=compute_contrast(img_np),
        face_size=compute_face_size(img_np),
        aspect_ratio=compute_aspect_ratio(img_np),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--index-dir",
        default="/home/leann/face-detection/data/index_files",
    )
    p.add_argument("--names-json", default="/home/leann/face-detection/results/scale_up_results/names.json")
    p.add_argument(
        "--baseline-metrics",
        default="/home/leann/face-detection/results/scale_up_results/precision_recall_metrics.csv",
    )
    p.add_argument("--samples-per-name", type=int, default=100)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--train-ratio", type=float, default=0.8)
    args = p.parse_args()

    seed_everything(args.seed)

    with open(args.names_json) as f:
        names = json.load(f)
    names = [n.strip().lower() for n in names]

    out_dir = Path("/home/leann/face-detection/results/confound_analysis")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("CONFOUND ANALYSIS: PHOTO QUALITY METRICS")
    print("=" * 70)
    print(f"Names: {len(names)}")
    print(f"Samples per name: {args.samples_per_name}")
    print(f"Output dir: {out_dir}")
    print()

    # Collect validation images
    all_metrics = []
    rng = random.Random(args.seed)

    for name in tqdm(names, desc="Analyzing images"):
        index_path = os.path.join(args.index_dir, f"index_{name}.json")
        if not os.path.exists(index_path):
            print(f"  Warning: {index_path} not found")
            continue

        with open(index_path) as f:
            data = json.load(f)

        good_images = list(data.get("good", []))
        rng.shuffle(good_images)

        # Use validation split
        split_idx = int(len(good_images) * args.train_ratio)
        val_images = good_images[split_idx:]

        # Sample up to N
        sampled = val_images[: args.samples_per_name]

        for path in sampled:
            if not os.path.exists(path):
                continue
            try:
                metrics = analyze_image(path, name)
                all_metrics.append(metrics)
            except Exception as e:
                print(f"  Error processing {path}: {e}")
                continue

    # Save per-image metrics
    df = pd.DataFrame([vars(m) for m in all_metrics])
    df.to_csv(out_dir / "quality_metrics.csv", index=False)

    print(f"\nAnalyzed {len(all_metrics)} images across {len(names)} names")

    # Aggregate per name
    per_name = df.groupby("name").agg(
        {
            "blur_score": ["mean", "std"],
            "brightness": ["mean", "std"],
            "contrast": ["mean", "std"],
            "face_size": ["mean", "std"],
            "aspect_ratio": ["mean", "std"],
        }
    )
    per_name.columns = ["_".join(col).strip() for col in per_name.columns.values]
    per_name = per_name.reset_index()

    # Load baseline performance metrics
    baseline_df = pd.read_csv(args.baseline_metrics)
    baseline_df = baseline_df[["name", "recall", "precision", "f1_score", "predicted_count", "support"]]

    # Merge
    merged = per_name.merge(baseline_df, on="name", how="inner")
    merged.to_csv(out_dir / "per_name_summary.csv", index=False)

    # Compute correlations
    from scipy.stats import pearsonr

    correlations = {}
    quality_cols = [
        "blur_score_mean",
        "brightness_mean",
        "contrast_mean",
        "face_size_mean",
        "aspect_ratio_mean",
    ]
    perf_cols = ["recall", "precision", "f1_score", "predicted_count"]

    for qcol in quality_cols:
        correlations[qcol] = {}
        for pcol in perf_cols:
            valid = merged[[qcol, pcol]].dropna()
            if len(valid) > 2:
                r, p = pearsonr(valid[qcol], valid[pcol])
                correlations[qcol][pcol] = {"r": float(r), "p": float(p)}
            else:
                correlations[qcol][pcol] = {"r": 0.0, "p": 1.0}

    (out_dir / "correlations.json").write_text(json.dumps(correlations, indent=2))

    # Print summary
    print()
    print("=" * 70)
    print("CONFOUND ANALYSIS RESULTS")
    print("=" * 70)
    print()
    print("Correlations between photo quality and model performance:")
    print("(Significant if |r| > 0.3 and p < 0.05)")
    print()

    for qcol in quality_cols:
        print(f"{qcol}:")
        for pcol in perf_cols:
            r = correlations[qcol][pcol]["r"]
            p = correlations[qcol][pcol]["p"]
            sig = "**" if abs(r) > 0.3 and p < 0.05 else "  "
            print(f"  vs {pcol:20s}: r={r:+.3f}, p={p:.4f} {sig}")
        print()

    print("INTERPRETATION:")
    print("  - Strong correlations suggest model learns photo quality confounds")
    print("  - Weak correlations suggest model learns name-specific features")
    print()
    print(f"Results saved to: {out_dir}/")

    # Create simple visualizations
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        viz_dir = out_dir / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        # Scatter: blur vs F1
        plt.figure(figsize=(10, 6))
        plt.scatter(merged["blur_score_mean"], merged["f1_score"], alpha=0.7)
        for _, row in merged.iterrows():
            plt.annotate(row["name"], (row["blur_score_mean"], row["f1_score"]), fontsize=8, alpha=0.7)
        plt.xlabel("Blur Score (higher = sharper)")
        plt.ylabel("F1 Score")
        plt.title("Photo Sharpness vs Model Performance")
        r = correlations["blur_score_mean"]["f1_score"]["r"]
        plt.text(0.05, 0.95, f"r = {r:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top")
        plt.tight_layout()
        plt.savefig(viz_dir / "blur_vs_f1.png", dpi=150)
        plt.close()

        # Scatter: brightness vs predicted_count
        plt.figure(figsize=(10, 6))
        plt.scatter(merged["brightness_mean"], merged["predicted_count"], alpha=0.7)
        for _, row in merged.iterrows():
            plt.annotate(row["name"], (row["brightness_mean"], row["predicted_count"]), fontsize=8, alpha=0.7)
        plt.xlabel("Brightness (mean pixel intensity)")
        plt.ylabel("Predicted Count")
        plt.title("Photo Brightness vs Prediction Frequency")
        r = correlations["brightness_mean"]["predicted_count"]["r"]
        plt.text(
            0.05, 0.95, f"r = {r:.3f}", transform=plt.gca().transAxes, fontsize=12, verticalalignment="top"
        )
        plt.tight_layout()
        plt.savefig(viz_dir / "brightness_vs_pred_count.png", dpi=150)
        plt.close()

        # Heatmap of all correlations
        corr_matrix = []
        for qcol in quality_cols:
            row = [correlations[qcol][pcol]["r"] for pcol in perf_cols]
            corr_matrix.append(row)

        plt.figure(figsize=(10, 6))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=perf_cols,
            yticklabels=[q.replace("_mean", "") for q in quality_cols],
            cbar_kws={"label": "Pearson r"},
        )
        plt.title("Quality Metrics vs Model Performance")
        plt.tight_layout()
        plt.savefig(viz_dir / "correlation_heatmap.png", dpi=150)
        plt.close()

        print(f"Visualizations saved to: {viz_dir}/")

    except ImportError:
        print("(matplotlib not available, skipping visualizations)")


if __name__ == "__main__":
    main()
