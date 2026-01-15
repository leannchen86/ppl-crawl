"""
Advanced Visualizations for Name-Face Association Analysis.

Creates:
1. ROC curves (per-name discriminability)
2. Confusion heatmap (which names get confused)
3. LDA projection (honest 2D visualization)
4. Per-name accuracy bar chart with gender coloring

Usage:
    python scripts/clip/analysis/advanced_viz.py  # Uses saved data from clip_probe_30way_scaleup.py
    python scripts/clip/analysis/advanced_viz.py --input-dir ./scale_up_results
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.metrics import roc_curve, auc, confusion_matrix
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import label_binarize
import seaborn as sns

from clip_dataset import create_name_gender_mapping


def load_data(input_dir):
    """Load saved data from clip_scale_up_probe outputs."""
    predictions = np.load(os.path.join(input_dir, "predictions.npy"))
    true_labels = np.load(os.path.join(input_dir, "true_labels.npy"))
    embeddings = np.load(os.path.join(input_dir, "val_embeddings.npy"))
    
    with open(os.path.join(input_dir, "names.json")) as f:
        names = json.load(f)
    
    rankings = pd.read_csv(os.path.join(input_dir, "rankings.csv"))
    
    return predictions, true_labels, embeddings, names, rankings


def plot_roc_curves(true_labels, embeddings, names, output_path, top_n=10):
    """Plot ROC curves for top N and bottom N names."""
    n_classes = len(names)
    
    # Binarize labels for ROC
    y_bin = label_binarize(true_labels, classes=range(n_classes))
    
    # Train a simple classifier to get probabilities
    from sklearn.linear_model import LogisticRegression
    # Use solver that handles multi-class automatically
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(embeddings, true_labels)
    y_score = clf.predict_proba(embeddings)
    
    # Compute ROC for each class
    fpr = {}
    tpr = {}
    roc_auc = {}
    
    for i, name in enumerate(names):
        if y_bin[:, i].sum() > 0:  # Has positive samples
            fpr[name], tpr[name], _ = roc_curve(y_bin[:, i], y_score[:, i])
            roc_auc[name] = auc(fpr[name], tpr[name])
        else:
            roc_auc[name] = 0.5
    
    # Sort by AUC
    sorted_names = sorted(roc_auc.keys(), key=lambda x: -roc_auc[x])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Top N
    ax = axes[0]
    colors = plt.cm.Greens(np.linspace(0.4, 0.9, top_n))
    for i, name in enumerate(sorted_names[:top_n]):
        if name in fpr:
            ax.plot(fpr[name], tpr[name], color=colors[i], lw=2,
                   label=f'{name.capitalize()} (AUC={roc_auc[name]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Top {top_n} Names (Strongest Vibes)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    # Bottom N
    ax = axes[1]
    colors = plt.cm.Reds(np.linspace(0.4, 0.9, top_n))
    for i, name in enumerate(sorted_names[-top_n:]):
        if name in fpr:
            ax.plot(fpr[name], tpr[name], color=colors[i], lw=2,
                   label=f'{name.capitalize()} (AUC={roc_auc[name]:.2f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=11)
    ax.set_ylabel('True Positive Rate', fontsize=11)
    ax.set_title(f'Bottom {top_n} Names (Weakest Vibes)', fontsize=12)
    ax.legend(loc='lower right', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")
    
    return roc_auc


def plot_confusion_heatmap(predictions, true_labels, names, output_path, top_n=15):
    """Plot confusion matrix as heatmap."""
    # Get most common names for readability
    from collections import Counter
    label_counts = Counter(true_labels)
    top_labels = [l for l, _ in label_counts.most_common(top_n)]
    
    # Filter to top labels
    mask = np.isin(true_labels, top_labels)
    filtered_true = true_labels[mask]
    filtered_pred = predictions[mask]
    
    # Remap to sequential indices
    label_map = {l: i for i, l in enumerate(top_labels)}
    filtered_true_mapped = np.array([label_map[l] for l in filtered_true])
    filtered_pred_mapped = np.array([label_map.get(p, -1) for p in filtered_pred])
    
    # Only keep valid predictions
    valid_mask = filtered_pred_mapped >= 0
    filtered_true_mapped = filtered_true_mapped[valid_mask]
    filtered_pred_mapped = filtered_pred_mapped[valid_mask]
    
    # Compute confusion matrix
    cm = confusion_matrix(filtered_true_mapped, filtered_pred_mapped, 
                         labels=range(len(top_labels)))
    
    # Normalize by row
    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-10)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    top_names = [names[l].capitalize() for l in top_labels]
    
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=top_names, yticklabels=top_names,
                ax=ax, cbar_kws={'label': 'Proportion'})
    
    ax.set_xlabel('Predicted Name', fontsize=12)
    ax.set_ylabel('True Name', fontsize=12)
    ax.set_title(f'Confusion Matrix (Top {top_n} Names)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_lda_projection(embeddings, true_labels, names, output_path, n_samples=2000):
    """Create LDA 2D projection (honest visualization)."""
    # Subsample for speed
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings = embeddings[idx]
        true_labels = true_labels[idx]
    
    # Fit LDA
    n_components = min(2, len(set(true_labels)) - 1)
    if n_components < 2:
        print("Warning: Not enough classes for 2D LDA projection")
        return
    
    lda = LinearDiscriminantAnalysis(n_components=2)
    X_lda = lda.fit_transform(embeddings, true_labels)
    
    # Get gender for coloring
    name_to_gender = create_name_gender_mapping()
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by gender
    colors = []
    for label in true_labels:
        name = names[label]
        gender = name_to_gender.get(name, "unknown")
        if gender == "male":
            colors.append('#3498db')  # Blue
        elif gender == "female":
            colors.append('#e74c3c')  # Red
        else:
            colors.append('#95a5a6')  # Gray
    
    scatter = ax.scatter(X_lda[:, 0], X_lda[:, 1], c=colors, alpha=0.4, s=10)
    
    # Add name centroids
    unique_labels = list(set(true_labels))
    for label in unique_labels[:15]:  # Top 15 only for readability
        mask = true_labels == label
        centroid = X_lda[mask].mean(axis=0)
        name = names[label]
        gender = name_to_gender.get(name, "unknown")
        color = '#2980b9' if gender == "male" else '#c0392b' if gender == "female" else '#7f8c8d'
        ax.annotate(name.capitalize(), centroid, fontsize=9, fontweight='bold',
                   color=color, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
    
    ax.set_xlabel('LDA Component 1', fontsize=11)
    ax.set_ylabel('LDA Component 2', fontsize=11)
    ax.set_title('LDA Projection of CLIP Embeddings\n(Blue=Male, Red=Female)', fontsize=13)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#3498db', label='Male names'),
        Patch(facecolor='#e74c3c', label='Female names'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_accuracy_by_gender(rankings, names, output_path):
    """Plot per-name accuracy with gender coloring."""
    name_to_gender = create_name_gender_mapping()
    
    # Add gender column
    rankings = rankings.copy()
    rankings['gender'] = rankings['name'].apply(
        lambda n: name_to_gender.get(n, 'unknown')
    )
    
    # Sort by accuracy
    rankings = rankings.sort_values('accuracy', ascending=True)
    
    # Colors
    colors = rankings['gender'].map({
        'male': '#3498db',
        'female': '#e74c3c', 
        'unknown': '#95a5a6'
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(rankings) * 0.25)))
    
    bars = ax.barh(range(len(rankings)), rankings['accuracy'], color=colors)
    
    ax.set_yticks(range(len(rankings)))
    ax.set_yticklabels([n.capitalize() for n in rankings['name']])
    ax.set_xlabel('Accuracy', fontsize=11)
    ax.set_title('Per-Name Accuracy (Blue=Male, Red=Female)', fontsize=13)
    
    # Add random baseline
    random_baseline = 1 / len(rankings)
    ax.axvline(x=random_baseline, color='black', linestyle='--', 
               label=f'Random ({100*random_baseline:.1f}%)')
    ax.legend(loc='lower right')
    
    # Add value labels
    for i, (idx, row) in enumerate(rankings.iterrows()):
        ax.text(row['accuracy'] + 0.01, i, f"{100*row['accuracy']:.1f}%", 
                va='center', fontsize=8)
    
    ax.set_xlim(0, max(rankings['accuracy']) * 1.15)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def plot_gender_comparison(rankings, output_path):
    """Compare male vs female name accuracy distributions."""
    name_to_gender = create_name_gender_mapping()
    
    rankings = rankings.copy()
    rankings['gender'] = rankings['name'].apply(
        lambda n: name_to_gender.get(n, 'unknown')
    )
    
    male_acc = rankings[rankings['gender'] == 'male']['accuracy']
    female_acc = rankings[rankings['gender'] == 'female']['accuracy']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Box plot
    data = [male_acc.values, female_acc.values]
    bp = ax.boxplot(data, labels=['Male Names', 'Female Names'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    # Add individual points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.5, s=30, 
                  color='#2980b9' if i == 0 else '#c0392b')
    
    # Stats
    ax.axhline(y=1/len(rankings), color='black', linestyle='--', 
               label=f'Random ({100/len(rankings):.1f}%)')
    
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Accuracy Distribution by Gender', fontsize=13)
    ax.legend()
    
    # Add means
    ax.text(1, male_acc.mean() + 0.02, f'μ={100*male_acc.mean():.1f}%', 
            ha='center', fontsize=10)
    ax.text(2, female_acc.mean() + 0.02, f'μ={100*female_acc.mean():.1f}%', 
            ha='center', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./scale_up_results",
                        help="Directory with clip_probe_30way_scaleup.py outputs")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (defaults to input-dir)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("="*60)
    print("ADVANCED VISUALIZATIONS")
    print("="*60)
    
    # Load data
    print("\nLoading data...")
    try:
        predictions, true_labels, embeddings, names, rankings = load_data(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in {args.input_dir}")
        print("Please run scripts/clip/clip_probe_30way_scaleup.py first!")
        return
    
    print(f"Loaded {len(embeddings)} embeddings for {len(names)} names")
    
    # Generate visualizations
    print("\n1. Generating ROC curves...")
    roc_auc = plot_roc_curves(
        true_labels, embeddings, names,
        os.path.join(args.output_dir, "roc_curves.png")
    )
    
    print("\n2. Generating confusion heatmap...")
    plot_confusion_heatmap(
        predictions, true_labels, names,
        os.path.join(args.output_dir, "confusion_heatmap.png")
    )
    
    print("\n3. Generating LDA projection...")
    plot_lda_projection(
        embeddings, true_labels, names,
        os.path.join(args.output_dir, "lda_projection.png")
    )
    
    print("\n4. Generating accuracy by gender...")
    plot_accuracy_by_gender(
        rankings, names,
        os.path.join(args.output_dir, "accuracy_by_name.png")
    )
    
    print("\n5. Generating gender comparison...")
    plot_gender_comparison(
        rankings,
        os.path.join(args.output_dir, "gender_comparison.png")
    )
    
    # Summary
    print("\n" + "="*60)
    print("VISUALIZATION SUMMARY")
    print("="*60)
    
    # Top/bottom by ROC AUC
    sorted_auc = sorted(roc_auc.items(), key=lambda x: -x[1])
    
    print("\n📊 Top 5 Names by ROC AUC (most discriminable):")
    for name, auc_val in sorted_auc[:5]:
        print(f"   {name.capitalize():12s} AUC={auc_val:.3f}")
    
    print("\n📊 Bottom 5 Names by ROC AUC (least discriminable):")
    for name, auc_val in sorted_auc[-5:]:
        print(f"   {name.capitalize():12s} AUC={auc_val:.3f}")
    
    # Gender comparison
    name_to_gender = create_name_gender_mapping()
    male_aucs = [v for k, v in roc_auc.items() if name_to_gender.get(k) == 'male']
    female_aucs = [v for k, v in roc_auc.items() if name_to_gender.get(k) == 'female']
    
    if male_aucs and female_aucs:
        print(f"\n📊 Gender Comparison:")
        print(f"   Male names avg AUC:   {np.mean(male_aucs):.3f}")
        print(f"   Female names avg AUC: {np.mean(female_aucs):.3f}")
    
    print(f"\n📁 Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

