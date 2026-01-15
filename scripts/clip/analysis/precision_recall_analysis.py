"""
Precision/Recall Analysis for Name-Face Association.

Computes per-name precision, recall, F1-score, and confusion patterns
from saved predictions.

Usage:
    python precision_recall_analysis.py --input-dir ./scale_up_results
"""
import argparse
import json
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, classification_report
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns


def load_results(input_dir):
    """Load predictions and labels from scale_up_test results."""
    predictions = np.load(f"{input_dir}/predictions.npy")
    true_labels = np.load(f"{input_dir}/true_labels.npy")
    
    with open(f"{input_dir}/names.json") as f:
        names = json.load(f)
    
    return predictions, true_labels, names


def compute_confusion_for_name(predictions, true_labels, target_idx, all_names):
    """Find what a specific name gets confused with."""
    # Find all instances where true label is target
    mask = true_labels == target_idx
    actual_instances = true_labels[mask]
    predicted_for_instances = predictions[mask]
    
    # Count predictions
    from collections import Counter
    confusion_counts = Counter(predicted_for_instances)
    
    # Convert to name-based confusion
    confusions = []
    total = len(actual_instances)
    
    for pred_idx, count in confusion_counts.most_common(10):
        if pred_idx != target_idx:  # Skip correct predictions
            confusions.append({
                'confused_as': all_names[pred_idx],
                'count': count,
                'percentage': 100 * count / total
            })
    
    return confusions


def compute_per_name_metrics(predictions, true_labels, names):
    """Compute precision, recall, F1 for each name."""
    # Get metrics from sklearn
    precision, recall, f1, support = precision_recall_fscore_support(
        true_labels, predictions, labels=range(len(names)), zero_division=0
    )
    
    # Count predictions per name
    pred_counts = np.bincount(predictions, minlength=len(names))
    
    results = []
    for i, name in enumerate(names):
        # Compute TP, FP, FN
        tp = np.sum((predictions == i) & (true_labels == i))
        fp = np.sum((predictions == i) & (true_labels != i))
        fn = np.sum((predictions != i) & (true_labels == i))
        
        results.append({
            'name': name,
            'recall': recall[i],
            'precision': precision[i],
            'f1_score': f1[i],
            'support': support[i],  # Actual count in validation
            'predicted_count': pred_counts[i],  # Times model predicted this
            'true_positives': tp,
            'false_positives': fp,
            'false_negatives': fn,
        })
    
    return pd.DataFrame(results)


def plot_precision_recall_scatter(df, output_path):
    """Scatter plot of precision vs recall for all names."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Color by F1 score
    scatter = ax.scatter(df['recall'], df['precision'], 
                        s=100, c=df['f1_score'], 
                        cmap='RdYlGn', alpha=0.7, edgecolors='black')
    
    # Add name labels
    for idx, row in df.iterrows():
        ax.annotate(row['name'].capitalize(), 
                   (row['recall'], row['precision']),
                   fontsize=8, ha='center', va='bottom')
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('F1-Score', fontsize=11)
    
    # Add diagonal reference line (precision = recall)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Precision = Recall')
    
    # Random baseline
    random_baseline = 1 / len(df)
    ax.axhline(y=random_baseline, color='red', linestyle=':', alpha=0.5, 
               label=f'Random Precision ({100*random_baseline:.1f}%)')
    ax.axvline(x=random_baseline, color='red', linestyle=':', alpha=0.5,
               label=f'Random Recall ({100*random_baseline:.1f}%)')
    
    ax.set_xlabel('Recall (Sensitivity)', fontsize=12)
    ax.set_ylabel('Precision (Positive Predictive Value)', fontsize=12)
    ax.set_title('Precision-Recall Trade-off by Name', fontsize=14)
    ax.legend(loc='lower left')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_prediction_bias(df, output_path):
    """Show prediction bias: predicted count vs actual count."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    df_sorted = df.sort_values('support', ascending=True)
    x = np.arange(len(df_sorted))
    width = 0.35
    
    bars1 = ax.barh(x - width/2, df_sorted['support'], width, 
                    label='Actual Count', color='steelblue', alpha=0.8)
    bars2 = ax.barh(x + width/2, df_sorted['predicted_count'], width,
                    label='Predicted Count', color='coral', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels([n.capitalize() for n in df_sorted['name']])
    ax.set_xlabel('Count', fontsize=11)
    ax.set_title('Prediction Bias: Actual vs Predicted Counts', fontsize=13)
    ax.legend()
    
    # Add vertical line at balanced point
    if df_sorted['support'].nunique() == 1:  # All equal (balanced)
        ax.axvline(x=df_sorted['support'].iloc[0], color='green', 
                  linestyle='--', alpha=0.5, label='Balanced')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_confusion_top_names(predictions, true_labels, names, output_path, top_n=5):
    """Show detailed confusion for top/bottom performers."""
    # Get top and bottom performers by F1
    _, _, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, labels=range(len(names)), zero_division=0
    )
    
    sorted_indices = np.argsort(f1)
    bottom_indices = sorted_indices[:top_n]
    top_indices = sorted_indices[-top_n:]
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    for idx, (indices, title, ax) in enumerate([
        (top_indices, "Top 5 Performers", axes[0]),
        (bottom_indices, "Bottom 5 Performers", axes[1])
    ]):
        confusion_data = []
        for i in indices:
            confusions = compute_confusion_for_name(predictions, true_labels, i, names)
            for conf in confusions[:3]:  # Top 3 confusions
                confusion_data.append({
                    'Name': names[i].capitalize(),
                    'Confused As': conf['confused_as'].capitalize(),
                    'Percentage': conf['percentage']
                })
        
        if confusion_data:
            df_conf = pd.DataFrame(confusion_data)
            pivot = df_conf.pivot(index='Name', columns='Confused As', values='Percentage')
            pivot = pivot.fillna(0)
            
            sns.heatmap(pivot, annot=True, fmt='.1f', cmap='Reds', 
                       ax=ax, cbar_kws={'label': 'Confusion %'})
            ax.set_title(f'{title} - Top Confusions', fontsize=12)
            ax.set_xlabel('Confused As', fontsize=11)
            ax.set_ylabel('True Name', fontsize=11)
        else:
            ax.text(0.5, 0.5, 'No confusion data', ha='center', va='center')
            ax.set_title(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Precision/Recall analysis for name-face task")
    parser.add_argument("--input-dir", default="./scale_up_results",
                       help="Directory with predictions from clip_probe_30way_scaleup.py")
    parser.add_argument("--output-dir", default=None,
                       help="Output directory (defaults to input-dir)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("="*60)
    print("PRECISION/RECALL ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading predictions...")
    predictions, true_labels, names = load_results(args.input_dir)
    print(f"Loaded {len(predictions)} predictions for {len(names)} names")
    
    # Compute metrics
    print("\nComputing per-name metrics...")
    df = compute_per_name_metrics(predictions, true_labels, names)
    
    # Sort by F1 for display
    df_sorted = df.sort_values('f1_score', ascending=False)
    
    # Save to CSV
    csv_path = f"{args.output_dir}/precision_recall_metrics.csv"
    df_sorted.to_csv(csv_path, index=False)
    print(f"Saved metrics to: {csv_path}")
    
    # Generate visualizations
    print("\nGenerating visualizations...")
    plot_precision_recall_scatter(df, f"{args.output_dir}/precision_recall_scatter.png")
    plot_prediction_bias(df, f"{args.output_dir}/prediction_bias.png")
    plot_confusion_top_names(predictions, true_labels, names, 
                            f"{args.output_dir}/top_confusions.png")
    
    # Print summary
    print("\n" + "="*60)
    print("PER-NAME METRICS SUMMARY")
    print("="*60)
    
    # Overall stats
    overall_acc = np.mean(predictions == true_labels)
    print(f"\nOverall Accuracy: {100*overall_acc:.1f}%")
    print(f"Random Baseline: {100/len(names):.1f}%")
    print(f"Mean Precision: {100*df['precision'].mean():.1f}%")
    print(f"Mean Recall: {100*df['recall'].mean():.1f}%")
    print(f"Mean F1-Score: {100*df['f1_score'].mean():.1f}%")
    
    # Top performers
    print(f"\n📊 Top 10 Names by F1-Score:")
    print("-"*80)
    print(f"{'Name':<12} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Support':<10} {'Predicted':<10}")
    print("-"*80)
    for _, row in df_sorted.head(10).iterrows():
        print(f"{row['name'].capitalize():<12} "
              f"{100*row['recall']:>6.1f}%   "
              f"{100*row['precision']:>6.1f}%    "
              f"{100*row['f1_score']:>6.1f}%   "
              f"{int(row['support']):>6}     "
              f"{int(row['predicted_count']):>6}")
    
    # Bottom performers
    print(f"\n📊 Bottom 10 Names by F1-Score:")
    print("-"*80)
    print(f"{'Name':<12} {'Recall':<10} {'Precision':<10} {'F1':<10} {'Support':<10} {'Predicted':<10}")
    print("-"*80)
    for _, row in df_sorted.tail(10).iterrows():
        print(f"{row['name'].capitalize():<12} "
              f"{100*row['recall']:>6.1f}%   "
              f"{100*row['precision']:>6.1f}%    "
              f"{100*row['f1_score']:>6.1f}%   "
              f"{int(row['support']):>6}     "
              f"{int(row['predicted_count']):>6}")
    
    # Interesting patterns
    print(f"\n📊 Notable Patterns:")
    print("-"*60)
    
    # High precision, low recall (conservative)
    conservative = df[(df['precision'] > 0.3) & (df['recall'] < 0.15)]
    if len(conservative) > 0:
        print(f"\n⚡ Conservative (High Precision, Low Recall):")
        for _, row in conservative.iterrows():
            print(f"   {row['name'].capitalize()}: "
                  f"P={100*row['precision']:.1f}%, R={100*row['recall']:.1f}% "
                  f"(Rarely predicts, but usually correct)")
    
    # Low precision, high recall (over-predicts)
    overpredicts = df[(df['precision'] < 0.15) & (df['recall'] > 0.15)]
    if len(overpredicts) > 0:
        print(f"\n⚠ Over-predicts:")
        for _, row in overpredicts.iterrows():
            print(f"   {row['name'].capitalize()}: "
                  f"P={100*row['precision']:.1f}%, R={100*row['recall']:.1f}% "
                  f"(Predicts often, often wrong)")
    
    # Balanced (precision ≈ recall)
    balanced = df[abs(df['precision'] - df['recall']) < 0.05]
    if len(balanced) > 0 and len(balanced) <= 10:
        print(f"\n✓ Balanced (Precision ≈ Recall):")
        for _, row in balanced.iterrows():
            print(f"   {row['name'].capitalize()}: "
                  f"P={100*row['precision']:.1f}%, R={100*row['recall']:.1f}%")
    
    # Never predicted
    never_predicted = df[df['predicted_count'] == 0]
    if len(never_predicted) > 0:
        print(f"\n❌ Never Predicted (Model Ignores These):")
        for _, row in never_predicted.iterrows():
            print(f"   {row['name'].capitalize()}: "
                  f"{int(row['support'])} actual instances, 0 predictions")
    
    # Prediction bias
    print(f"\n📊 Prediction Bias:")
    df['pred_bias'] = df['predicted_count'] / df['support']
    most_overpredicted = df.nlargest(3, 'pred_bias')
    most_underpredicted = df.nsmallest(3, 'pred_bias')
    
    print(f"\n   Most Over-predicted (pred/actual ratio):")
    for _, row in most_overpredicted.iterrows():
        if row['pred_bias'] > 1.1:
            print(f"      {row['name'].capitalize()}: "
                  f"{row['pred_bias']:.2f}x "
                  f"({int(row['predicted_count'])} pred / {int(row['support'])} actual)")
    
    print(f"\n   Most Under-predicted:")
    for _, row in most_underpredicted.iterrows():
        if row['pred_bias'] < 0.9:
            print(f"      {row['name'].capitalize()}: "
                  f"{row['pred_bias']:.2f}x "
                  f"({int(row['predicted_count'])} pred / {int(row['support'])} actual)")
    
    # Detailed confusion for top performers
    print(f"\n📊 Top Confusions for Best/Worst Names:")
    print("-"*60)
    
    # Best performer
    best_name_idx = df_sorted.iloc[0].name
    best_name = df_sorted.iloc[0]['name']
    confusions_best = compute_confusion_for_name(predictions, true_labels, 
                                                 names.index(best_name), names)
    print(f"\n   {best_name.capitalize()} (best) gets confused with:")
    for conf in confusions_best[:5]:
        print(f"      {conf['confused_as'].capitalize()}: "
              f"{conf['percentage']:.1f}% ({conf['count']} times)")
    
    # Worst performer
    worst_name_idx = df_sorted.iloc[-1].name
    worst_name = df_sorted.iloc[-1]['name']
    confusions_worst = compute_confusion_for_name(predictions, true_labels,
                                                  names.index(worst_name), names)
    print(f"\n   {worst_name.capitalize()} (worst) gets confused with:")
    for conf in confusions_worst[:5]:
        print(f"      {conf['confused_as'].capitalize()}: "
              f"{conf['percentage']:.1f}% ({conf['count']} times)")
    
    print(f"\n📁 Files saved to: {args.output_dir}/")
    print("   - precision_recall_metrics.csv")
    print("   - precision_recall_scatter.png")
    print("   - prediction_bias.png")
    print("   - top_confusions.png")


if __name__ == "__main__":
    main()

