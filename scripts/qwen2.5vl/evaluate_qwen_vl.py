"""
Evaluation script for Qwen 2.5 VL face-name classification.

Generates standardized results format:
- predictions.npy: Predicted class indices
- true_labels.npy: Ground truth indices
- results.csv: Per-name precision/recall/F1
- summary.md: Human-readable summary

Also computes metrics for comparison with CLIP baselines.
"""
import argparse
import json
from pathlib import Path
from collections import Counter
import numpy as np
import pandas as pd
from typing import Optional

import torch
from torch.utils.data import DataLoader
from PIL import Image
from tqdm import tqdm

try:
    from transformers import AutoProcessor
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False


def load_model_and_processor(checkpoint_dir: str, config: dict):
    """Load trained model from checkpoint."""
    from scripts.train_qwen_vl import Qwen2VLForClassification

    # Load model
    model = Qwen2VLForClassification(
        model_id=config["model_id"],
        num_labels=config["num_labels"],
        use_4bit=config["use_4bit"],
        freeze_vision=config["freeze_vision"],
    )

    # Load classifier weights
    classifier_path = Path(checkpoint_dir) / "classifier.pt"
    if classifier_path.exists():
        model.classifier.load_state_dict(torch.load(classifier_path))
        print(f"Loaded classifier weights from {classifier_path}")

    model.eval()
    return model


def compute_metrics(predictions: np.ndarray, labels: np.ndarray, names: list[str]) -> dict:
    """Compute comprehensive metrics."""
    num_classes = len(names)

    # Overall accuracy
    accuracy = (predictions == labels).mean()

    # Per-class metrics
    per_class = []
    for i, name in enumerate(names):
        mask = labels == i
        pred_mask = predictions == i

        tp = ((predictions == i) & (labels == i)).sum()
        fp = ((predictions == i) & (labels != i)).sum()
        fn = ((predictions != i) & (labels == i)).sum()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        per_class.append({
            "name": name,
            "support": int(mask.sum()),
            "predicted": int(pred_mask.sum()),
            "true_positives": int(tp),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    # Prediction distribution metrics
    pred_counts = Counter(predictions)
    pred_values = [pred_counts.get(i, 0) for i in range(num_classes)]
    pred_cv = np.std(pred_values) / np.mean(pred_values) if np.mean(pred_values) > 0 else 0

    # Top predictions analysis
    most_predicted = sorted(enumerate(pred_values), key=lambda x: -x[1])[:5]
    least_predicted = sorted(enumerate(pred_values), key=lambda x: x[1])[:5]

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "prediction_cv": pred_cv,
        "most_predicted": [(names[i], c) for i, c in most_predicted],
        "least_predicted": [(names[i], c) for i, c in least_predicted],
    }


def compute_comparison_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    names: list[str],
    top_n: int = 30
) -> dict:
    """Compute metrics comparable to CLIP baseline (top N names only)."""
    # Get top N names by support
    label_counts = Counter(labels)
    top_names_idx = sorted(label_counts.keys(), key=lambda x: -label_counts[x])[:top_n]
    top_names = [names[i] for i in top_names_idx]

    # Filter to only samples with top N names
    mask = np.isin(labels, top_names_idx)
    filtered_preds = predictions[mask]
    filtered_labels = labels[mask]

    # Also filter predictions to map to new indices
    idx_map = {old: new for new, old in enumerate(top_names_idx)}

    # Re-index
    new_labels = np.array([idx_map.get(l, -1) for l in filtered_labels])
    new_preds = np.array([idx_map.get(p, -1) for p in filtered_preds])

    # Accuracy (predictions within top N names)
    valid_mask = (new_preds >= 0) & (new_labels >= 0)
    accuracy = (new_preds[valid_mask] == new_labels[valid_mask]).mean() if valid_mask.sum() > 0 else 0

    return {
        f"accuracy_top{top_n}": accuracy,
        "top_names": top_names,
        "num_samples": int(mask.sum()),
    }


def generate_summary(
    metrics: dict,
    comparison_metrics: dict,
    config: dict,
    output_dir: Path,
):
    """Generate human-readable summary markdown."""
    summary = f"""# Track 1: Qwen 2.5 VL Results

## Configuration
- **Model:** {config.get('model_id', 'N/A')}
- **Num Labels:** {config.get('num_labels', 'N/A')}
- **Use 4-bit:** {config.get('use_4bit', 'N/A')}
- **Use LoRA:** {config.get('use_lora', 'N/A')}
- **LoRA Rank:** {config.get('lora_r', 'N/A')}
- **Learning Rate:** {config.get('learning_rate', 'N/A')}
- **Epochs:** {config.get('num_epochs', 'N/A')}
- **Class Balanced Beta:** {config.get('class_balanced_beta', 'N/A')}

## Results

### Overall Metrics
| Metric | Value | CLIP Baseline | Target |
|--------|-------|---------------|--------|
| Accuracy (all names) | {metrics['accuracy']*100:.2f}% | ~3% | >5% |
| Accuracy (top 30 names) | {comparison_metrics['accuracy_top30']*100:.2f}% | 13.9% | >18% |
| Prediction CV | {metrics['prediction_cv']:.3f} | 0.40 | <0.35 |

### Prediction Distribution
**Most Predicted Names:**
"""
    for name, count in metrics['most_predicted']:
        summary += f"- {name}: {count} predictions\n"

    summary += "\n**Least Predicted Names:**\n"
    for name, count in metrics['least_predicted']:
        summary += f"- {name}: {count} predictions\n"

    # Top/Bottom by F1
    per_class_sorted = sorted(metrics['per_class'], key=lambda x: -x['f1'])

    summary += "\n### Top 5 Names by F1 Score\n"
    summary += "| Name | Precision | Recall | F1 | Support |\n"
    summary += "|------|-----------|--------|----|---------|\n"
    for item in per_class_sorted[:5]:
        summary += f"| {item['name']} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['support']} |\n"

    summary += "\n### Bottom 5 Names by F1 Score\n"
    summary += "| Name | Precision | Recall | F1 | Support |\n"
    summary += "|------|-----------|--------|----|---------|\n"
    for item in per_class_sorted[-5:]:
        summary += f"| {item['name']} | {item['precision']:.3f} | {item['recall']:.3f} | {item['f1']:.3f} | {item['support']} |\n"

    summary += """
## Interpretation

### Key Findings
[To be filled after reviewing results]

### Next Steps
[To be filled based on results]

---
*Generated by evaluate_qwen_vl.py*
"""

    with open(output_dir / "summary.md", "w") as f:
        f.write(summary)

    print(f"Summary saved to {output_dir / 'summary.md'}")


def evaluate_from_predictions(
    predictions_file: str,
    labels_file: str,
    names_file: str,
    output_dir: str,
    config_file: Optional[str] = None,
):
    """Evaluate from saved prediction files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load data
    predictions = np.load(predictions_file)
    labels = np.load(labels_file)
    with open(names_file) as f:
        names = json.load(f)

    print(f"Loaded {len(predictions)} predictions")
    print(f"Number of names: {len(names)}")

    # Load config if available
    config = {}
    if config_file and Path(config_file).exists():
        with open(config_file) as f:
            config = json.load(f)

    # Compute metrics
    print("\nComputing metrics...")
    metrics = compute_metrics(predictions, labels, names)
    comparison_metrics = compute_comparison_metrics(predictions, labels, names, top_n=30)

    print(f"\nOverall accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"Top-30 accuracy: {comparison_metrics['accuracy_top30']*100:.2f}%")
    print(f"Prediction CV: {metrics['prediction_cv']:.3f}")

    # Save results
    df = pd.DataFrame(metrics['per_class'])
    df.to_csv(output_path / "results.csv", index=False)
    print(f"Per-class results saved to {output_path / 'results.csv'}")

    # Save metrics JSON
    metrics_summary = {
        "accuracy": metrics['accuracy'],
        "prediction_cv": metrics['prediction_cv'],
        "accuracy_top30": comparison_metrics['accuracy_top30'],
        "most_predicted": metrics['most_predicted'],
        "least_predicted": metrics['least_predicted'],
    }
    with open(output_path / "metrics.json", "w") as f:
        json.dump(metrics_summary, f, indent=2)

    # Generate summary
    generate_summary(metrics, comparison_metrics, config, output_path)


def run_inference(
    checkpoint_dir: str,
    data_dir: str,
    output_dir: str,
    batch_size: int = 8,
):
    """Run inference on validation set and save predictions."""
    if not HAS_TRANSFORMERS:
        print("Please install transformers: pip install transformers")
        return

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Load config
    config_file = Path(checkpoint_dir).parent / "config.json"
    if not config_file.exists():
        config_file = Path(checkpoint_dir) / "config.json"
    with open(config_file) as f:
        config = json.load(f)

    # Load names
    names_file = Path(data_dir) / "labels.json"
    with open(names_file) as f:
        names = json.load(f)

    # Load processor
    processor = AutoProcessor.from_pretrained(config["model_id"], trust_remote_code=True)

    # Load model
    model = load_model_and_processor(checkpoint_dir, config)

    # Load validation data
    from scripts.train_qwen_vl import FaceNameClassificationDataset

    val_dataset = FaceNameClassificationDataset(
        data_file=str(Path(data_dir) / "val.json"),
        processor=processor,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
    )

    # Run inference
    all_preds = []
    all_labels = []

    print("Running inference...")
    for batch in tqdm(val_loader):
        batch = {k: v.to("cuda") if isinstance(v, torch.Tensor) else v for k, v in batch.items()}

        with torch.no_grad(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
            )

        preds = outputs["logits"].argmax(dim=-1).cpu().numpy()
        labels = batch["labels"].cpu().numpy()

        all_preds.extend(preds)
        all_labels.extend(labels)

    predictions = np.array(all_preds)
    labels = np.array(all_labels)

    # Save predictions
    np.save(output_path / "predictions.npy", predictions)
    np.save(output_path / "true_labels.npy", labels)
    with open(output_path / "names.json", "w") as f:
        json.dump(names, f, indent=2)

    print(f"Saved predictions to {output_path}")

    # Run evaluation
    evaluate_from_predictions(
        predictions_file=str(output_path / "predictions.npy"),
        labels_file=str(output_path / "true_labels.npy"),
        names_file=str(output_path / "names.json"),
        output_dir=output_dir,
        config_file=str(config_file),
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate Qwen 2.5 VL face-name model")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference and evaluate")
    infer_parser.add_argument("--checkpoint-dir", required=True,
                              help="Directory containing model checkpoint")
    infer_parser.add_argument("--data-dir", required=True,
                              help="Directory containing val.json")
    infer_parser.add_argument("--output-dir", required=True,
                              help="Output directory for results")
    infer_parser.add_argument("--batch-size", type=int, default=8)

    # Evaluate from files command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate from prediction files")
    eval_parser.add_argument("--predictions", required=True,
                             help="Path to predictions.npy")
    eval_parser.add_argument("--labels", required=True,
                             help="Path to true_labels.npy")
    eval_parser.add_argument("--names", required=True,
                             help="Path to names.json")
    eval_parser.add_argument("--output-dir", required=True,
                             help="Output directory for results")
    eval_parser.add_argument("--config", default=None,
                             help="Optional config.json for summary")

    args = parser.parse_args()

    if args.command == "infer":
        run_inference(
            checkpoint_dir=args.checkpoint_dir,
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    elif args.command == "evaluate":
        evaluate_from_predictions(
            predictions_file=args.predictions,
            labels_file=args.labels,
            names_file=args.names,
            output_dir=args.output_dir,
            config_file=args.config,
        )
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
