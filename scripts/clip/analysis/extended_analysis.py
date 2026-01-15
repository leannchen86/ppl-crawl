"""
Extended Analysis for Name-Face Association Model.

Additional analyses beyond the basic visualizations:
1. Calibration curves (confidence reliability)
2. t-SNE projection (non-linear structure)
3. Inter-class similarity matrix (name-to-name confusion potential)
4. Prototype visualization (typical faces per name)
5. Sample difficulty analysis (easy vs hard examples)
6. Statistical significance tests
7. Silhouette scores (cluster quality)

Usage:
    python scripts/clip/analysis/extended_analysis.py --input-dir ./scale_up_results
"""
import argparse
import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from collections import defaultdict

from sklearn.manifold import TSNE
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.model_selection import cross_val_score, StratifiedKFold
from scipy.stats import mannwhitneyu, spearmanr
from scipy.spatial.distance import cosine
import warnings
warnings.filterwarnings('ignore')

from clip_dataset import create_name_gender_mapping


def load_data(input_dir):
    """Load saved data from clip_scale_up_probe outputs."""
    predictions = np.load(os.path.join(input_dir, "predictions.npy"))
    true_labels = np.load(os.path.join(input_dir, "true_labels.npy"))
    embeddings = np.load(os.path.join(input_dir, "val_embeddings.npy"))
    
    with open(os.path.join(input_dir, "names.json")) as f:
        names = json.load(f)
    
    return predictions, true_labels, embeddings, names


def plot_calibration_curve(true_labels, embeddings, names, output_path):
    """
    Plot calibration curve to check if predicted probabilities match reality.
    
    A well-calibrated model: when it says 80% confident, it's right 80% of the time.
    """
    print("\n📊 Calibration Analysis")
    print("-" * 40)
    
    # Train classifier to get probabilities
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    clf.fit(embeddings, true_labels)
    y_prob = clf.predict_proba(embeddings)
    y_pred = clf.predict(embeddings)
    
    # Get max probability (confidence) for each prediction
    confidences = y_prob.max(axis=1)
    correct = (y_pred == true_labels).astype(int)
    
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(correct, confidences, n_bins=10, strategy='uniform')
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Calibration curve
    ax = axes[0]
    ax.plot([0, 1], [0, 1], 'k--', label='Perfectly calibrated')
    ax.plot(prob_pred, prob_true, 'o-', color='#e74c3c', linewidth=2, markersize=8,
            label='Model calibration')
    ax.fill_between(prob_pred, prob_true, prob_pred, alpha=0.2, color='#e74c3c')
    ax.set_xlabel('Mean Predicted Confidence', fontsize=11)
    ax.set_ylabel('Fraction Actually Correct', fontsize=11)
    ax.set_title('Calibration Curve\n(How reliable is model confidence?)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    
    # Confidence distribution
    ax = axes[1]
    ax.hist(confidences[correct == 1], bins=30, alpha=0.7, label='Correct predictions',
            color='#2ecc71', density=True)
    ax.hist(confidences[correct == 0], bins=30, alpha=0.7, label='Incorrect predictions',
            color='#e74c3c', density=True)
    ax.set_xlabel('Prediction Confidence', fontsize=11)
    ax.set_ylabel('Density', fontsize=11)
    ax.set_title('Confidence Distribution\n(Can we trust high-confidence predictions?)', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")
    
    # Statistics
    high_conf_mask = confidences > 0.5
    high_conf_acc = correct[high_conf_mask].mean() if high_conf_mask.sum() > 0 else 0
    low_conf_mask = confidences <= 0.5
    low_conf_acc = correct[low_conf_mask].mean() if low_conf_mask.sum() > 0 else 0
    
    print(f"   High confidence (>50%) accuracy: {100*high_conf_acc:.1f}%")
    print(f"   Low confidence (≤50%) accuracy: {100*low_conf_acc:.1f}%")
    print(f"   → {'Good' if high_conf_acc > low_conf_acc + 0.1 else 'Poor'} confidence separation")
    
    return confidences, correct


def plot_tsne_projection(embeddings, true_labels, names, output_path, n_samples=3000):
    """
    t-SNE projection - non-linear dimensionality reduction.
    
    Unlike LDA, t-SNE doesn't assume linear separability and can reveal
    complex clustering structure.
    """
    print("\n📊 t-SNE Projection")
    print("-" * 40)
    
    # Subsample for speed
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sub = embeddings[idx]
        labels_sub = true_labels[idx]
    else:
        embeddings_sub = embeddings
        labels_sub = true_labels
    
    print(f"   Running t-SNE on {len(embeddings_sub)} samples...")
    
    tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(embeddings_sub)
    
    # Get gender for coloring
    name_to_gender = create_name_gender_mapping()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    # Color by gender
    ax = axes[0]
    colors = []
    for label in labels_sub:
        name = names[label]
        gender = name_to_gender.get(name, "unknown")
        if gender == "male":
            colors.append('#3498db')
        elif gender == "female":
            colors.append('#e74c3c')
        else:
            colors.append('#95a5a6')
    
    ax.scatter(X_tsne[:, 0], X_tsne[:, 1], c=colors, alpha=0.4, s=8)
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('t-SNE Projection (by Gender)\n(Blue=Male, Red=Female)', fontsize=12)
    
    # Add centroids for top names
    unique_labels, counts = np.unique(labels_sub, return_counts=True)
    top_labels = unique_labels[np.argsort(-counts)[:10]]
    
    for label in top_labels:
        mask = labels_sub == label
        centroid = X_tsne[mask].mean(axis=0)
        name = names[label]
        gender = name_to_gender.get(name, "unknown")
        color = '#2980b9' if gender == "male" else '#c0392b' if gender == "female" else '#7f8c8d'
        ax.annotate(name.capitalize(), centroid, fontsize=9, fontweight='bold',
                   color=color, ha='center', va='center',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
    
    # Color by individual names (top 10)
    ax = axes[1]
    cmap = plt.cm.get_cmap('tab10')
    
    # Gray background for other names
    other_mask = ~np.isin(labels_sub, top_labels)
    ax.scatter(X_tsne[other_mask, 0], X_tsne[other_mask, 1], c='#cccccc', alpha=0.2, s=5, label='Other')
    
    for i, label in enumerate(top_labels):
        mask = labels_sub == label
        ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1], c=[cmap(i)], alpha=0.6, s=15,
                  label=names[label].capitalize())
    
    ax.set_xlabel('t-SNE 1', fontsize=11)
    ax.set_ylabel('t-SNE 2', fontsize=11)
    ax.set_title('t-SNE Projection (Top 10 Names)', fontsize=12)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")


def plot_interclass_similarity(embeddings, true_labels, names, output_path, top_n=20):
    """
    Compute and plot similarity between name centroids.
    
    Shows which names are inherently similar in embedding space,
    beyond what the confusion matrix shows.
    """
    print("\n📊 Inter-Class Similarity Matrix")
    print("-" * 40)
    
    # Compute centroids
    centroids = {}
    for i, name in enumerate(names):
        mask = true_labels == i
        if mask.sum() > 0:
            centroids[name] = embeddings[mask].mean(axis=0)
    
    # Get top N names by sample count
    label_counts = np.bincount(true_labels, minlength=len(names))
    top_indices = np.argsort(-label_counts)[:top_n]
    top_names = [names[i] for i in top_indices]
    
    # Compute cosine similarity matrix
    n = len(top_names)
    sim_matrix = np.zeros((n, n))
    
    for i, name_i in enumerate(top_names):
        for j, name_j in enumerate(top_names):
            if name_i in centroids and name_j in centroids:
                # Cosine similarity
                sim = 1 - cosine(centroids[name_i], centroids[name_j])
                sim_matrix[i, j] = sim
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Mask diagonal for better visualization
    mask = np.eye(n, dtype=bool)
    sim_display = sim_matrix.copy()
    sim_display[mask] = np.nan
    
    sns.heatmap(sim_display, annot=True, fmt='.2f', cmap='RdYlGn',
                xticklabels=[n.capitalize() for n in top_names],
                yticklabels=[n.capitalize() for n in top_names],
                ax=ax, cbar_kws={'label': 'Cosine Similarity'},
                vmin=0.7, vmax=1.0, mask=mask)
    
    ax.set_title('Inter-Name Centroid Similarity\n(High similarity → likely to confuse)', fontsize=13)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")
    
    # Find most similar pairs
    print("\n   Most similar name pairs (confusion risk):")
    pairs = []
    for i in range(n):
        for j in range(i+1, n):
            pairs.append((top_names[i], top_names[j], sim_matrix[i, j]))
    
    pairs.sort(key=lambda x: -x[2])
    for name1, name2, sim in pairs[:5]:
        print(f"   {name1.capitalize():10s} ↔ {name2.capitalize():10s}: {sim:.3f}")
    
    return sim_matrix, top_names


def compute_silhouette_scores(embeddings, true_labels, names, output_path, n_samples=5000):
    """
    Compute silhouette scores to measure cluster quality.
    
    Silhouette score ranges from -1 to 1:
    - +1: Sample is well-matched to its cluster
    - 0: Sample is on boundary between clusters  
    - -1: Sample is misclassified/in wrong cluster
    """
    print("\n📊 Silhouette Score Analysis")
    print("-" * 40)
    
    # Subsample for speed
    if len(embeddings) > n_samples:
        idx = np.random.choice(len(embeddings), n_samples, replace=False)
        embeddings_sub = embeddings[idx]
        labels_sub = true_labels[idx]
    else:
        embeddings_sub = embeddings
        labels_sub = true_labels
    
    print(f"   Computing silhouette scores for {len(embeddings_sub)} samples...")
    
    # Overall silhouette score
    overall_score = silhouette_score(embeddings_sub, labels_sub)
    print(f"   Overall silhouette score: {overall_score:.3f}")
    
    # Per-sample silhouette scores
    sample_scores = silhouette_samples(embeddings_sub, labels_sub)
    
    # Per-class silhouette scores
    class_scores = {}
    for i, name in enumerate(names):
        mask = labels_sub == i
        if mask.sum() > 10:  # Need enough samples
            class_scores[name] = sample_scores[mask].mean()
    
    # Sort by score
    sorted_scores = sorted(class_scores.items(), key=lambda x: -x[1])
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Per-class silhouette scores
    ax = axes[0]
    names_sorted = [n.capitalize() for n, _ in sorted_scores]
    scores_sorted = [s for _, s in sorted_scores]
    colors = ['#2ecc71' if s > 0 else '#e74c3c' for s in scores_sorted]
    
    bars = ax.barh(range(len(names_sorted)), scores_sorted, color=colors)
    ax.set_yticks(range(len(names_sorted)))
    ax.set_yticklabels(names_sorted, fontsize=8)
    ax.axvline(x=0, color='black', linewidth=1)
    ax.axvline(x=overall_score, color='blue', linestyle='--', 
               label=f'Overall ({overall_score:.2f})')
    ax.set_xlabel('Silhouette Score', fontsize=11)
    ax.set_title('Per-Name Cluster Quality\n(Higher = better separated)', fontsize=12)
    ax.legend(loc='lower right')
    ax.grid(alpha=0.3, axis='x')
    
    # Silhouette score distribution
    ax = axes[1]
    ax.hist(sample_scores, bins=50, color='#3498db', alpha=0.7, edgecolor='white')
    ax.axvline(x=0, color='black', linewidth=2, label='Boundary')
    ax.axvline(x=overall_score, color='red', linewidth=2, linestyle='--',
               label=f'Mean ({overall_score:.2f})')
    ax.set_xlabel('Silhouette Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Distribution of Sample Silhouette Scores', fontsize=12)
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")
    
    # Interpretation
    print(f"\n   Interpretation:")
    if overall_score > 0.5:
        print("   → Strong cluster structure - names are well-separated")
    elif overall_score > 0.25:
        print("   → Moderate cluster structure - some overlap between names")
    elif overall_score > 0:
        print("   → Weak cluster structure - significant overlap")
    else:
        print("   → No cluster structure - names are not separable in embedding space")
    
    return class_scores, overall_score


def statistical_significance_tests(embeddings, true_labels, names, output_path):
    """
    Run statistical tests to determine if findings are significant.
    """
    print("\n📊 Statistical Significance Tests")
    print("-" * 40)
    
    name_to_gender = create_name_gender_mapping()
    
    # Train classifier with cross-validation
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    
    # Get per-fold accuracies per class
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_accuracies = defaultdict(list)
    
    for fold, (train_idx, val_idx) in enumerate(cv.split(embeddings, true_labels)):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = true_labels[train_idx], true_labels[val_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        # Per-class accuracy
        for i, name in enumerate(names):
            mask = y_val == i
            if mask.sum() > 0:
                acc = (y_pred[mask] == y_val[mask]).mean()
                fold_accuracies[name].append(acc)
    
    # Compute mean and std per name
    name_stats = {}
    for name, accs in fold_accuracies.items():
        if len(accs) >= 3:
            name_stats[name] = {
                'mean': np.mean(accs),
                'std': np.std(accs),
                'accs': accs
            }
    
    # Test: Male vs Female accuracy difference
    male_accs = [s['mean'] for n, s in name_stats.items() 
                 if name_to_gender.get(n) == 'male']
    female_accs = [s['mean'] for n, s in name_stats.items() 
                   if name_to_gender.get(n) == 'female']
    
    if male_accs and female_accs:
        stat, p_value = mannwhitneyu(male_accs, female_accs, alternative='two-sided')
        print(f"\n   Gender Comparison (Mann-Whitney U test):")
        print(f"   Male names mean accuracy:   {100*np.mean(male_accs):.2f}% ± {100*np.std(male_accs):.2f}%")
        print(f"   Female names mean accuracy: {100*np.mean(female_accs):.2f}% ± {100*np.std(female_accs):.2f}%")
        print(f"   p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"   → Statistically significant difference (p < 0.05)")
        else:
            print(f"   → No significant difference (p ≥ 0.05)")
    
    # Test: Correlation between sample count and accuracy
    sample_counts = np.bincount(true_labels, minlength=len(names))
    name_accs = [name_stats.get(n, {}).get('mean', 0) for n in names]
    
    corr, p_corr = spearmanr(sample_counts, name_accs)
    print(f"\n   Sample Count vs Accuracy (Spearman correlation):")
    print(f"   Correlation: {corr:.3f}, p-value: {p_corr:.4f}")
    if abs(corr) > 0.3 and p_corr < 0.05:
        print(f"   → {'More' if corr > 0 else 'Fewer'} samples associated with {'higher' if corr > 0 else 'lower'} accuracy")
    else:
        print(f"   → No significant correlation")
    
    # Plot CV stability
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Accuracy with error bars
    ax = axes[0]
    sorted_names = sorted(name_stats.items(), key=lambda x: -x[1]['mean'])[:20]
    
    y_pos = range(len(sorted_names))
    means = [s['mean'] for _, s in sorted_names]
    stds = [s['std'] for _, s in sorted_names]
    name_labels = [n.capitalize() for n, _ in sorted_names]
    
    colors = [('#3498db' if name_to_gender.get(n) == 'male' else 
               '#e74c3c' if name_to_gender.get(n) == 'female' else '#95a5a6')
              for n, _ in sorted_names]
    
    ax.barh(y_pos, means, xerr=stds, color=colors, alpha=0.8, capsize=3)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(name_labels)
    ax.set_xlabel('Accuracy (5-fold CV)', fontsize=11)
    ax.set_title('Cross-Validation Stability\n(Error bars = std across folds)', fontsize=12)
    ax.grid(alpha=0.3, axis='x')
    
    # Gender comparison boxplot
    ax = axes[1]
    data = [male_accs, female_accs]
    bp = ax.boxplot(data, labels=['Male Names', 'Female Names'], patch_artist=True)
    bp['boxes'][0].set_facecolor('#3498db')
    bp['boxes'][1].set_facecolor('#e74c3c')
    
    # Add points
    for i, d in enumerate(data):
        x = np.random.normal(i+1, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, s=40, 
                  color='#2980b9' if i == 0 else '#c0392b')
    
    ax.set_ylabel('Mean Accuracy (5-fold CV)', fontsize=11)
    ax.set_title(f'Gender Accuracy Comparison\n(p={p_value:.3f})', fontsize=12)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n   Saved: {output_path}")
    
    return name_stats


def identify_hard_samples(embeddings, true_labels, names, output_path, top_n=50):
    """
    Identify consistently misclassified samples (hard examples).
    
    Uses multiple random splits to find samples that are always wrong.
    """
    print("\n📊 Hard Sample Analysis")
    print("-" * 40)
    
    n_splits = 5
    sample_correct_count = np.zeros(len(embeddings))
    sample_tested_count = np.zeros(len(embeddings))
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    clf = LogisticRegression(max_iter=1000, solver='lbfgs')
    
    for train_idx, val_idx in cv.split(embeddings, true_labels):
        X_train, X_val = embeddings[train_idx], embeddings[val_idx]
        y_train, y_val = true_labels[train_idx], true_labels[val_idx]
        
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_val)
        
        correct = (y_pred == y_val)
        sample_correct_count[val_idx] += correct
        sample_tested_count[val_idx] += 1
    
    # Difficulty score (0 = always wrong, 1 = always correct)
    difficulty = sample_correct_count / np.maximum(sample_tested_count, 1)
    
    # Find hardest samples (always wrong)
    hardest_idx = np.argsort(difficulty)[:top_n]
    
    print(f"   Found {(difficulty == 0).sum()} samples always misclassified")
    print(f"   Found {(difficulty == 1).sum()} samples always correct")
    
    # Analyze hard samples by class
    hard_by_class = defaultdict(int)
    for idx in hardest_idx:
        hard_by_class[names[true_labels[idx]]] += 1
    
    print(f"\n   Names with most hard samples:")
    for name, count in sorted(hard_by_class.items(), key=lambda x: -x[1])[:10]:
        print(f"   {name.capitalize():12s}: {count} hard samples")
    
    # Plot difficulty distribution
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    ax = axes[0]
    ax.hist(difficulty, bins=n_splits+1, color='#3498db', alpha=0.7, edgecolor='white')
    ax.set_xlabel('Fraction of Folds Correct', fontsize=11)
    ax.set_ylabel('Number of Samples', fontsize=11)
    ax.set_title('Sample Difficulty Distribution', fontsize=12)
    ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.grid(alpha=0.3)
    
    # Hard samples by class
    ax = axes[1]
    sorted_classes = sorted(hard_by_class.items(), key=lambda x: -x[1])[:15]
    class_names = [n.capitalize() for n, _ in sorted_classes]
    class_counts = [c for _, c in sorted_classes]
    
    ax.barh(range(len(class_names)), class_counts, color='#e74c3c', alpha=0.7)
    ax.set_yticks(range(len(class_names)))
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Number of Hard Samples', fontsize=11)
    ax.set_title('Names with Most Difficult Samples\n(Always misclassified)', fontsize=12)
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved: {output_path}")
    
    return difficulty, hard_by_class


def generate_summary_report(results, output_path):
    """Generate a text summary of all analyses."""
    
    with open(output_path, 'w') as f:
        f.write("=" * 60 + "\n")
        f.write("EXTENDED ANALYSIS SUMMARY REPORT\n")
        f.write("=" * 60 + "\n\n")
        
        # Calibration
        f.write("1. CALIBRATION ANALYSIS\n")
        f.write("-" * 40 + "\n")
        f.write(f"   High confidence (>50%) accuracy: {results.get('high_conf_acc', 'N/A')}\n")
        f.write(f"   Interpretation: Model confidence is ")
        f.write("reliable\n\n" if results.get('calibration_good', False) else "unreliable\n\n")
        
        # Silhouette
        f.write("2. CLUSTER QUALITY (Silhouette Score)\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Overall silhouette score: {results.get('silhouette', 'N/A'):.3f}\n")
        f.write(f"   Interpretation: ")
        sil = results.get('silhouette', 0)
        if sil > 0.5:
            f.write("Strong cluster structure\n\n")
        elif sil > 0.25:
            f.write("Moderate cluster structure\n\n")
        else:
            f.write("Weak cluster structure\n\n")
        
        # Statistical significance
        f.write("3. STATISTICAL SIGNIFICANCE\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Gender difference p-value: {results.get('gender_p', 'N/A')}\n")
        f.write(f"   Interpretation: ")
        if results.get('gender_p', 1) < 0.05:
            f.write("Significant gender difference\n\n")
        else:
            f.write("No significant gender difference\n\n")
        
        # Hard samples
        f.write("4. HARD SAMPLES\n")
        f.write("-" * 40 + "\n")
        f.write(f"   Always misclassified: {results.get('always_wrong', 'N/A')} samples\n")
        f.write(f"   Always correct: {results.get('always_correct', 'N/A')} samples\n\n")
        
        f.write("=" * 60 + "\n")
        f.write("END OF REPORT\n")
        f.write("=" * 60 + "\n")
    
    print(f"\n📄 Summary report saved: {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="./scale_up_results",
                        help="Directory with clip_probe_30way_scaleup.py outputs")
    parser.add_argument("--output-dir", default=None,
                        help="Output directory (defaults to input-dir)")
    args = parser.parse_args()
    
    if args.output_dir is None:
        args.output_dir = args.input_dir
    
    print("=" * 60)
    print("EXTENDED ANALYSIS")
    print("=" * 60)
    
    # Load data
    print("\nLoading data...")
    try:
        predictions, true_labels, embeddings, names = load_data(args.input_dir)
    except FileNotFoundError as e:
        print(f"Error: Could not find data files in {args.input_dir}")
        print("Please run scripts/clip/clip_probe_30way_scaleup.py first!")
        return
    
    print(f"Loaded {len(embeddings)} embeddings for {len(names)} names")
    
    results = {}
    
    # 1. Calibration analysis
    print("\n" + "=" * 60)
    confidences, correct = plot_calibration_curve(
        true_labels, embeddings, names,
        os.path.join(args.output_dir, "calibration_curve.png")
    )
    results['high_conf_acc'] = f"{100*correct[confidences > 0.5].mean():.1f}%"
    results['calibration_good'] = correct[confidences > 0.5].mean() > correct[confidences <= 0.5].mean() + 0.1
    
    # 2. t-SNE projection
    print("\n" + "=" * 60)
    plot_tsne_projection(
        embeddings, true_labels, names,
        os.path.join(args.output_dir, "tsne_projection.png")
    )
    
    # 3. Inter-class similarity
    print("\n" + "=" * 60)
    sim_matrix, top_names = plot_interclass_similarity(
        embeddings, true_labels, names,
        os.path.join(args.output_dir, "interclass_similarity.png")
    )
    
    # 4. Silhouette scores
    print("\n" + "=" * 60)
    class_scores, overall_sil = compute_silhouette_scores(
        embeddings, true_labels, names,
        os.path.join(args.output_dir, "silhouette_scores.png")
    )
    results['silhouette'] = overall_sil
    
    # 5. Statistical significance
    print("\n" + "=" * 60)
    name_stats = statistical_significance_tests(
        embeddings, true_labels, names,
        os.path.join(args.output_dir, "statistical_tests.png")
    )
    
    # 6. Hard sample analysis
    print("\n" + "=" * 60)
    difficulty, hard_by_class = identify_hard_samples(
        embeddings, true_labels, names,
        os.path.join(args.output_dir, "hard_samples.png")
    )
    results['always_wrong'] = (difficulty == 0).sum()
    results['always_correct'] = (difficulty == 1).sum()
    
    # Generate summary report
    generate_summary_report(results, os.path.join(args.output_dir, "extended_analysis_report.txt"))
    
    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"\n📁 All outputs saved to: {args.output_dir}/")
    print("\nNew visualizations:")
    print("   • calibration_curve.png     - Is model confidence reliable?")
    print("   • tsne_projection.png       - Non-linear clustering structure")
    print("   • interclass_similarity.png - Which names are inherently similar?")
    print("   • silhouette_scores.png     - Cluster quality per name")
    print("   • statistical_tests.png     - Significance of findings")
    print("   • hard_samples.png          - Which samples are always wrong?")
    print("   • extended_analysis_report.txt - Summary report")


if __name__ == "__main__":
    main()






















