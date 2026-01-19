"""
Generate comparison report for CLIP/SigLIP/OpenCLIP linear probe experiments.

This script reads experiment results from all 6 experiments and generates:
1. summary.json - Aggregated results with validation checks
2. comparison_table.csv - Easy-to-read comparison table

Usage:
    python scripts/unified/generate_comparison.py --results-dir results/linear_probe_comparison

The script validates that all experiments used:
- Same holdout checksum (36ad6b2f)
- Same validation set size (12,778 samples)
- Same hyperparameters (seed=42, epochs=100, lr=0.01, weight_decay=1e-4)
"""
import argparse
import json
import os
from pathlib import Path
from datetime import datetime

import pandas as pd


# Expected configurations for validation
EXPECTED_HOLDOUT_CHECKSUM = "36ad6b2f"
EXPECTED_VAL_SAMPLES = 12778
EXPECTED_SEED = 42
EXPECTED_EPOCHS = 100
EXPECTED_LR = 0.01
EXPECTED_WEIGHT_DECAY = 1e-4

EXPECTED_EMBED_DIMS = {
    "clip": 512,
    "siglip": 768,
    "openclip": 512,
}


def load_experiment_config(experiment_dir: Path) -> dict:
    """Load experiment config from a results directory."""
    config_path = experiment_dir / "experiment_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"experiment_config.json not found in {experiment_dir}")

    with open(config_path) as f:
        return json.load(f)


def validate_experiment(config: dict, model: str, mode: str) -> list:
    """Validate that an experiment meets expected criteria."""
    issues = []

    # Check holdout checksum
    checksum = config.get("data", {}).get("holdout_checksum", "unknown")
    if checksum != EXPECTED_HOLDOUT_CHECKSUM:
        issues.append(f"holdout_checksum mismatch: {checksum} != {EXPECTED_HOLDOUT_CHECKSUM}")

    # Check validation samples
    val_samples = config.get("data", {}).get("val_samples", 0)
    if val_samples != EXPECTED_VAL_SAMPLES:
        issues.append(f"val_samples mismatch: {val_samples} != {EXPECTED_VAL_SAMPLES}")

    # Check seed
    seed = config.get("training", {}).get("seed", 0)
    if seed != EXPECTED_SEED:
        issues.append(f"seed mismatch: {seed} != {EXPECTED_SEED}")

    # Check epochs
    epochs = config.get("training", {}).get("epochs", 0)
    if epochs != EXPECTED_EPOCHS:
        issues.append(f"epochs mismatch: {epochs} != {EXPECTED_EPOCHS}")

    # Check learning rate
    lr = config.get("training", {}).get("lr", 0)
    if lr != EXPECTED_LR:
        issues.append(f"lr mismatch: {lr} != {EXPECTED_LR}")

    # Check weight decay
    weight_decay = config.get("training", {}).get("weight_decay", 0)
    if weight_decay != EXPECTED_WEIGHT_DECAY:
        issues.append(f"weight_decay mismatch: {weight_decay} != {EXPECTED_WEIGHT_DECAY}")

    # Check embedding dimension
    embed_dim = config.get("model", {}).get("embed_dim", 0)
    expected_dim = EXPECTED_EMBED_DIMS.get(model, 0)
    if embed_dim != expected_dim:
        issues.append(f"embed_dim mismatch: {embed_dim} != {expected_dim}")

    return issues


def main():
    parser = argparse.ArgumentParser(description="Generate comparison report for linear probe experiments")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Directory containing experiment results (e.g., results/linear_probe_comparison)")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(exist_ok=True)

    print("=" * 70)
    print("LINEAR PROBE COMPARISON REPORT GENERATOR")
    print("=" * 70)
    print(f"Results directory: {results_dir}")
    print(f"Output directory: {comparison_dir}")
    print()

    # Define expected experiments
    models = ["clip", "siglip", "openclip"]
    modes = ["balanced", "imbalanced"]

    # Load all experiments
    experiments = {}
    all_issues = []
    missing = []

    for model in models:
        for mode in modes:
            exp_key = f"{model}/{mode}"
            exp_dir = results_dir / model / mode

            if not exp_dir.exists():
                missing.append(exp_key)
                continue

            try:
                config = load_experiment_config(exp_dir)
                experiments[exp_key] = config

                # Validate
                issues = validate_experiment(config, model, mode)
                if issues:
                    all_issues.append({
                        "experiment": exp_key,
                        "issues": issues
                    })
            except Exception as e:
                missing.append(f"{exp_key} (error: {e})")

    # Report missing experiments
    if missing:
        print("WARNING: Missing experiments:")
        for m in missing:
            print(f"  - {m}")
        print()

    # Report validation issues
    if all_issues:
        print("WARNING: Validation issues found:")
        for item in all_issues:
            print(f"  {item['experiment']}:")
            for issue in item['issues']:
                print(f"    - {issue}")
        print()
    else:
        print("All experiments passed validation checks.")
        print()

    # Build comparison data
    comparison_data = []

    for model in models:
        for mode in modes:
            exp_key = f"{model}/{mode}"
            if exp_key not in experiments:
                continue

            config = experiments[exp_key]
            results = config.get("results", {})
            model_info = config.get("model", {})
            training_info = config.get("training", {})
            data_info = config.get("data", {})

            comparison_data.append({
                "model": model,
                "mode": mode,
                "model_type": model_info.get("type", ""),
                "architecture": model_info.get("name", ""),
                "pretrained": model_info.get("pretrained", ""),
                "embed_dim": model_info.get("embed_dim", 0),
                "best_val_acc": results.get("best_val_acc", 0),
                "best_epoch": results.get("best_epoch", 0),
                "random_baseline": results.get("random_baseline", 0),
                "improvement_over_random": results.get("improvement_over_random", 0),
                "relative_improvement": results.get("relative_improvement", 0),
                "train_samples": data_info.get("train_samples", 0),
                "val_samples": data_info.get("val_samples", 0),
                "holdout_checksum": data_info.get("holdout_checksum", ""),
                "epochs": training_info.get("epochs", 0),
                "lr": training_info.get("lr", 0),
                "weight_decay": training_info.get("weight_decay", 0),
                "seed": training_info.get("seed", 0),
                "embed_time_seconds": results.get("embed_time_seconds", 0),
                "train_time_seconds": results.get("train_time_seconds", 0),
            })

    # Create DataFrame
    df = pd.DataFrame(comparison_data)

    # Print comparison table
    print("COMPARISON TABLE")
    print("-" * 70)

    if len(df) > 0:
        # Summary by model and mode
        print("\nValidation Accuracy by Model and Data Mode:")
        print()

        for mode in modes:
            mode_df = df[df["mode"] == mode].sort_values("best_val_acc", ascending=False)
            print(f"  {mode.upper()} DATA:")
            for _, row in mode_df.iterrows():
                acc_pct = row['best_val_acc'] * 100
                rel_imp = row['relative_improvement'] * 100
                print(f"    {row['model']:8s}: {acc_pct:5.2f}% (rel. improvement: +{rel_imp:.1f}%)")
            print()

        # Best overall
        best_row = df.loc[df["best_val_acc"].idxmax()]
        print(f"Best overall: {best_row['model']} on {best_row['mode']} data")
        print(f"             Accuracy: {best_row['best_val_acc']*100:.2f}%")
        print()

    # Save comparison table CSV
    csv_path = comparison_dir / "comparison_table.csv"
    df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to: {csv_path}")

    # Build summary JSON
    summary = {
        "generated_at": datetime.now().isoformat(),
        "results_dir": str(results_dir),
        "experiments_found": len(experiments),
        "experiments_expected": len(models) * len(modes),
        "validation": {
            "all_passed": len(all_issues) == 0 and len(missing) == 0,
            "missing_experiments": missing,
            "issues": all_issues,
            "expected": {
                "holdout_checksum": EXPECTED_HOLDOUT_CHECKSUM,
                "val_samples": EXPECTED_VAL_SAMPLES,
                "seed": EXPECTED_SEED,
                "epochs": EXPECTED_EPOCHS,
                "lr": EXPECTED_LR,
                "weight_decay": EXPECTED_WEIGHT_DECAY,
                "embed_dims": EXPECTED_EMBED_DIMS,
            }
        },
        "results": {},
    }

    # Organize results by model
    for model in models:
        summary["results"][model] = {}
        for mode in modes:
            exp_key = f"{model}/{mode}"
            if exp_key in experiments:
                config = experiments[exp_key]
                summary["results"][model][mode] = {
                    "best_val_acc": config.get("results", {}).get("best_val_acc", 0),
                    "best_epoch": config.get("results", {}).get("best_epoch", 0),
                    "embed_dim": config.get("model", {}).get("embed_dim", 0),
                    "train_samples": config.get("data", {}).get("train_samples", 0),
                    "val_samples": config.get("data", {}).get("val_samples", 0),
                    "holdout_checksum": config.get("data", {}).get("holdout_checksum", ""),
                    "embed_time_seconds": config.get("results", {}).get("embed_time_seconds", 0),
                    "train_time_seconds": config.get("results", {}).get("train_time_seconds", 0),
                }

    # Add rankings
    if len(df) > 0:
        for mode in modes:
            mode_df = df[df["mode"] == mode].sort_values("best_val_acc", ascending=False)
            summary[f"ranking_{mode}"] = [
                {"rank": i+1, "model": row["model"], "accuracy": row["best_val_acc"]}
                for i, (_, row) in enumerate(mode_df.iterrows())
            ]

    # Save summary JSON
    summary_path = comparison_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Summary saved to: {summary_path}")

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)

    # Return exit code based on validation
    if len(missing) > 0 or len(all_issues) > 0:
        return 1
    return 0


if __name__ == "__main__":
    exit(main())
