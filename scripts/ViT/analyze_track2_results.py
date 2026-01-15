"""
Track 2: ViT from Scratch - Results Analysis

Analyzes results from all Track 2 experiments and generates summary report.

Usage:
    python analyze_track2_results.py
    python analyze_track2_results.py --results-dir results/track2_vit_scratch
    python analyze_track2_results.py --compare exp2.1_baseline exp2.2_mixup
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def load_experiment(exp_dir):
    """Load all results from an experiment directory."""
    results = {"name": os.path.basename(exp_dir)}

    # Load config
    config_path = os.path.join(exp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            results["config"] = json.load(f)

    # Load training history
    history_path = os.path.join(exp_dir, "training_log.json")
    if os.path.exists(history_path):
        with open(history_path) as f:
            results["history"] = json.load(f)

    # Load predictions
    pred_path = os.path.join(exp_dir, "predictions.npy")
    if os.path.exists(pred_path):
        results["predictions"] = np.load(pred_path)

    # Load labels
    labels_path = os.path.join(exp_dir, "true_labels.npy")
    if os.path.exists(labels_path):
        results["labels"] = np.load(labels_path)

    # Load names
    names_path = os.path.join(exp_dir, "names.json")
    if os.path.exists(names_path):
        with open(names_path) as f:
            results["names"] = json.load(f)

    return results


def compute_metrics(results):
    """Compute additional metrics from predictions."""
    if "predictions" not in results or "labels" not in results:
        return {}

    preds = results["predictions"]
    labels = results["labels"]
    num_classes = len(results.get("names", []))

    metrics = {}

    # Accuracy
    metrics["accuracy"] = 100.0 * np.mean(preds == labels)

    # Prediction distribution
    unique, counts = np.unique(preds, return_counts=True)
    metrics["prediction_cv"] = counts.std() / counts.mean() if counts.mean() > 0 else 0
    metrics["num_predicted_classes"] = len(unique)

    # Per-class metrics
    if num_classes > 0:
        class_correct = np.zeros(num_classes)
        class_total = np.zeros(num_classes)
        class_predicted = np.zeros(num_classes)

        for p, l in zip(preds, labels):
            class_total[l] += 1
            class_predicted[p] += 1
            if p == l:
                class_correct[l] += 1

        per_class_acc = class_correct / (class_total + 1e-10)
        per_class_precision = class_correct / (class_predicted + 1e-10)

        # Find best and worst performing classes
        valid_mask = class_total > 0
        if valid_mask.any():
            sorted_idx = np.argsort(per_class_acc[valid_mask])
            names = results.get("names", [f"class_{i}" for i in range(num_classes)])

            valid_names = [names[i] for i in range(num_classes) if valid_mask[i]]
            valid_accs = per_class_acc[valid_mask]

            metrics["top5_classes"] = [
                (valid_names[i], f"{valid_accs[sorted_idx[-1-j]]*100:.1f}%")
                for j, i in enumerate(sorted_idx[-5:][::-1])
            ]
            metrics["bottom5_classes"] = [
                (valid_names[sorted_idx[j]], f"{valid_accs[sorted_idx[j]]*100:.1f}%")
                for j in range(min(5, len(sorted_idx)))
            ]

        metrics["per_class_acc"] = per_class_acc
        metrics["per_class_precision"] = per_class_precision

    return metrics


def plot_training_curves(experiments, output_path):
    """Plot training curves for all experiments."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    colors = plt.cm.tab10(np.linspace(0, 1, len(experiments)))

    for i, (exp, color) in enumerate(zip(experiments, colors)):
        history = exp.get("history", {})
        name = exp["name"]

        epochs = range(1, len(history.get("train_loss", [])) + 1)

        if "train_loss" in history:
            axes[0, 0].plot(epochs, history["train_loss"], color=color, label=name)
        if "val_loss" in history:
            axes[0, 1].plot(epochs, history["val_loss"], color=color, label=name)
        if "train_acc" in history:
            axes[1, 0].plot(epochs, history["train_acc"], color=color, label=name)
        if "val_acc" in history:
            axes[1, 1].plot(epochs, history["val_acc"], color=color, label=name)

    axes[0, 0].set_xlabel("Epoch")
    axes[0, 0].set_ylabel("Train Loss")
    axes[0, 0].set_title("Training Loss")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].set_xlabel("Epoch")
    axes[0, 1].set_ylabel("Val Loss")
    axes[0, 1].set_title("Validation Loss")
    axes[0, 1].legend(fontsize=8)
    axes[0, 1].grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Epoch")
    axes[1, 0].set_ylabel("Train Accuracy (%)")
    axes[1, 0].set_title("Training Accuracy")
    axes[1, 0].legend(fontsize=8)
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].set_xlabel("Epoch")
    axes[1, 1].set_ylabel("Val Accuracy (%)")
    axes[1, 1].set_title("Validation Accuracy")
    axes[1, 1].legend(fontsize=8)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved training curves to {output_path}")


def plot_prediction_distribution(experiments, output_path):
    """Plot prediction distribution for each experiment."""
    n_exp = len(experiments)
    fig, axes = plt.subplots(1, n_exp, figsize=(5 * n_exp, 4))
    if n_exp == 1:
        axes = [axes]

    for ax, exp in zip(axes, experiments):
        if "predictions" not in exp:
            continue

        preds = exp["predictions"]
        unique, counts = np.unique(preds, return_counts=True)

        # Sort by count
        sorted_idx = np.argsort(counts)[::-1]
        sorted_counts = counts[sorted_idx]

        ax.bar(range(len(sorted_counts)), sorted_counts, width=1.0)
        ax.set_xlabel("Class (sorted by frequency)")
        ax.set_ylabel("Prediction count")
        ax.set_title(f"{exp['name']}\nCV={counts.std()/counts.mean():.3f}")

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved prediction distribution to {output_path}")


def generate_summary_report(experiments, output_path):
    """Generate markdown summary report."""
    lines = [
        "# Track 2: ViT from Scratch - Results Summary\n",
        f"Generated: {__import__('datetime').datetime.now().strftime('%Y-%m-%d %H:%M')}\n",
        "## Experiment Comparison\n",
        "| Experiment | Val Acc | Train Acc | Pred CV | Epochs | Time (h) |",
        "|------------|---------|-----------|---------|--------|----------|",
    ]

    for exp in experiments:
        config = exp.get("config", {})
        history = exp.get("history", {})
        metrics = compute_metrics(exp)

        val_acc = config.get("best_val_acc", "N/A")
        if isinstance(val_acc, float):
            val_acc = f"{val_acc:.2f}%"

        train_acc = config.get("final_train_acc", "N/A")
        if isinstance(train_acc, float):
            train_acc = f"{train_acc:.2f}%"

        pred_cv = metrics.get("prediction_cv", "N/A")
        if isinstance(pred_cv, float):
            pred_cv = f"{pred_cv:.3f}"

        epochs = config.get("epochs", "N/A")
        time_h = config.get("total_time_hours", "N/A")
        if isinstance(time_h, float):
            time_h = f"{time_h:.2f}"

        lines.append(
            f"| {exp['name']} | {val_acc} | {train_acc} | {pred_cv} | {epochs} | {time_h} |"
        )

    lines.append("\n## Configuration Details\n")

    for exp in experiments:
        config = exp.get("config", {})
        lines.append(f"### {exp['name']}\n")
        lines.append("```json")
        lines.append(json.dumps({
            "model": config.get("model", "N/A"),
            "num_classes": config.get("num_classes", "N/A"),
            "batch_size": config.get("batch_size", "N/A"),
            "lr": config.get("lr", "N/A"),
            "label_smoothing": config.get("label_smoothing", "N/A"),
            "mixup_alpha": config.get("mixup_alpha", "N/A"),
            "cutmix_alpha": config.get("cutmix_alpha", "N/A"),
            "class_weight_method": config.get("class_weight_method", "N/A"),
        }, indent=2))
        lines.append("```\n")

    lines.append("## Per-Class Analysis\n")

    for exp in experiments:
        metrics = compute_metrics(exp)
        if not metrics:
            continue

        lines.append(f"### {exp['name']}\n")

        if "top5_classes" in metrics:
            lines.append("**Top 5 Classes:**\n")
            for name, acc in metrics["top5_classes"]:
                lines.append(f"- {name}: {acc}")
            lines.append("")

        if "bottom5_classes" in metrics:
            lines.append("**Bottom 5 Classes:**\n")
            for name, acc in metrics["bottom5_classes"]:
                lines.append(f"- {name}: {acc}")
            lines.append("")

    # Comparison with baselines
    lines.append("## Comparison with Baselines\n")
    lines.append("| Method | Val Accuracy |")
    lines.append("|--------|--------------|")
    lines.append("| Random baseline (500 classes) | 0.2% |")
    lines.append("| Random baseline (30 classes) | 3.3% |")
    lines.append("| CLIP linear probe (30 names) | 13.9% |")
    lines.append("| Fine-tune pretrained ViT | 11.7% |")
    lines.append("| Train CNN from scratch | 10.9% |")

    for exp in experiments:
        config = exp.get("config", {})
        val_acc = config.get("best_val_acc", "N/A")
        if isinstance(val_acc, float):
            val_acc = f"{val_acc:.2f}%"
        lines.append(f"| **{exp['name']}** | **{val_acc}** |")

    with open(output_path, "w") as f:
        f.write("\n".join(lines))

    print(f"Saved summary report to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Analyze Track 2 experiment results")
    parser.add_argument("--results-dir", type=str,
                        default="results/track2_vit_scratch",
                        help="Base directory containing experiment results")
    parser.add_argument("--compare", nargs="+", type=str, default=None,
                        help="Specific experiments to compare (directory names)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Output directory for plots and reports")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        print("Run experiments first using: ./scripts/ViT/run_track2_experiments.sh")
        return

    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find experiment directories
    if args.compare:
        exp_dirs = [results_dir / name for name in args.compare]
    else:
        exp_dirs = [d for d in results_dir.iterdir()
                    if d.is_dir() and (d / "config.json").exists()]

    if not exp_dirs:
        print("No experiment results found!")
        print(f"Looking in: {results_dir}")
        return

    exp_dirs.sort()
    print(f"Found {len(exp_dirs)} experiments:")
    for d in exp_dirs:
        print(f"  - {d.name}")

    # Load all experiments
    print("\nLoading experiment results...")
    experiments = []
    for exp_dir in exp_dirs:
        try:
            exp = load_experiment(str(exp_dir))
            experiments.append(exp)
            print(f"  Loaded: {exp['name']}")
        except Exception as e:
            print(f"  Failed to load {exp_dir.name}: {e}")

    if not experiments:
        print("No valid experiments loaded!")
        return

    # Generate analysis
    print("\nGenerating analysis...")

    # Training curves
    plot_training_curves(
        experiments,
        str(output_dir / "training_curves.png")
    )

    # Prediction distribution
    plot_prediction_distribution(
        experiments,
        str(output_dir / "prediction_distribution.png")
    )

    # Summary report
    generate_summary_report(
        experiments,
        str(output_dir / "summary.md")
    )

    # Print quick summary
    print("\n" + "=" * 70)
    print("QUICK SUMMARY")
    print("=" * 70)
    print(f"{'Experiment':<30} {'Val Acc':<12} {'Pred CV':<10}")
    print("-" * 52)

    for exp in experiments:
        config = exp.get("config", {})
        metrics = compute_metrics(exp)

        val_acc = config.get("best_val_acc", 0)
        pred_cv = metrics.get("prediction_cv", 0)

        print(f"{exp['name']:<30} {val_acc:>8.2f}%   {pred_cv:>8.3f}")

    print("-" * 52)
    print(f"\nFull report: {output_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
