"""
Embedding Analysis: Visualize CLIP embeddings to understand separability.

Creates:
1. Cosine similarity distributions (intra-class vs inter-class)
2. Confusion matrix from nearest-neighbor classification
3. Per-name accuracy breakdown
4. Silhouette score (cluster quality metric)

Usage:
    python embedding_analysis.py --names david michael laura maria
    EXPERIMENT=2 python embedding_analysis.py
"""
import argparse
import os
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, silhouette_score
from collections import defaultdict

from clip_dataset import FaceNameDataset, create_name_gender_mapping


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def extract_embeddings(model, dataloader, device):
    """Extract CLIP image embeddings for all samples."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, _text_tokens, names in tqdm(dataloader, desc="Extracting embeddings"):
            images = images.to(device)
            embeddings = model.encode_image(images)
            embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
            all_embeddings.append(embeddings.cpu())
            all_labels.extend(names)
    
    return torch.cat(all_embeddings, dim=0), all_labels


def compute_similarity_distributions(embeddings, labels, sample_size=5000):
    """Compute cosine similarity distributions for same-class and different-class pairs."""
    embeddings = embeddings.numpy()
    n = len(labels)
    
    same_class_sims = []
    diff_class_sims = []
    
    # Sample random pairs (for efficiency)
    indices = list(range(n))
    random.shuffle(indices)
    
    count = 0
    for i in range(min(sample_size, n)):
        for j in range(i + 1, min(sample_size, n)):
            idx_i, idx_j = indices[i], indices[j]
            sim = np.dot(embeddings[idx_i], embeddings[idx_j])
            
            if labels[idx_i] == labels[idx_j]:
                same_class_sims.append(sim)
            else:
                diff_class_sims.append(sim)
            
            count += 1
            if count >= sample_size * 10:  # Limit total pairs
                break
        if count >= sample_size * 10:
            break
    
    return same_class_sims, diff_class_sims


def plot_similarity_distributions(same_sims, diff_sims, output_path):
    """Plot overlapping histograms of similarity distributions."""
    plt.figure(figsize=(10, 6))
    
    plt.hist(same_sims, bins=50, alpha=0.6, label=f'Same name (n={len(same_sims)})', color='green')
    plt.hist(diff_sims, bins=50, alpha=0.6, label=f'Different name (n={len(diff_sims)})', color='red')
    
    plt.xlabel('Cosine Similarity', fontsize=12)
    plt.ylabel('Count', fontsize=12)
    plt.title('Embedding Similarity Distributions', fontsize=14)
    plt.legend(fontsize=11)
    
    # Add statistics
    same_mean = np.mean(same_sims)
    diff_mean = np.mean(diff_sims)
    separation = same_mean - diff_mean
    
    stats_text = f"Same-name mean: {same_mean:.4f}\nDiff-name mean: {diff_mean:.4f}\nSeparation: {separation:.4f}"
    plt.text(0.02, 0.98, stats_text, transform=plt.gca().transAxes, 
             verticalalignment='top', fontsize=10, 
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")
    
    return same_mean, diff_mean, separation


def nearest_neighbor_classify(embeddings, labels, name_to_idx):
    """Classify each sample by its nearest neighbor (leave-one-out)."""
    embeddings = embeddings.numpy()
    n = len(labels)
    predictions = []
    
    # Compute all pairwise similarities
    sims = embeddings @ embeddings.T
    
    for i in range(n):
        # Mask self-similarity
        sims[i, i] = -1
        
        # Find nearest neighbor
        nn_idx = np.argmax(sims[i])
        predictions.append(labels[nn_idx])
    
    # Compute per-class accuracy
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, true in zip(predictions, labels):
        class_total[true] += 1
        if pred == true:
            class_correct[true] += 1
    
    per_class_acc = {name: class_correct[name] / class_total[name] 
                     for name in class_total}
    
    overall_acc = sum(class_correct.values()) / sum(class_total.values())
    
    return predictions, per_class_acc, overall_acc


def plot_confusion_matrix(labels, predictions, names, output_path):
    """Plot confusion matrix."""
    cm = confusion_matrix(labels, predictions, labels=names)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    plt.imshow(cm_normalized, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Nearest Neighbor)', fontsize=14)
    plt.colorbar(label='Proportion')
    
    tick_marks = np.arange(len(names))
    plt.xticks(tick_marks, [n.capitalize() for n in names], rotation=45)
    plt.yticks(tick_marks, [n.capitalize() for n in names])
    
    # Add text annotations
    thresh = cm_normalized.max() / 2.
    for i in range(len(names)):
        for j in range(len(names)):
            plt.text(j, i, f'{cm_normalized[i, j]:.2f}',
                    ha="center", va="center",
                    color="white" if cm_normalized[i, j] > thresh else "black")
    
    plt.ylabel('True Name', fontsize=12)
    plt.xlabel('Predicted Name', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def plot_per_class_accuracy(per_class_acc, output_path):
    """Plot per-class accuracy bar chart."""
    names = list(per_class_acc.keys())
    accs = [per_class_acc[n] for n in names]
    
    plt.figure(figsize=(10, 5))
    bars = plt.bar([n.capitalize() for n in names], accs, color='steelblue')
    
    # Add value labels on bars
    for bar, acc in zip(bars, accs):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{100*acc:.1f}%', ha='center', va='bottom', fontsize=10)
    
    plt.axhline(y=1/len(names), color='red', linestyle='--', 
                label=f'Random baseline ({100/len(names):.1f}%)')
    
    plt.xlabel('Name', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Per-Name Accuracy (Nearest Neighbor)', fontsize=14)
    plt.legend()
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved: {output_path}")


def collate_fn(batch, tokenizer):
    images, texts, names = zip(*batch)
    images = torch.stack(images)
    text_tokens = tokenizer(list(texts))
    return images, text_tokens, list(names)


def main():
    parser = argparse.ArgumentParser(description="Embedding analysis for CLIP")
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--index-dir", default="/home/leann/face-detection/data/index_files")
    parser.add_argument("--output-dir", default="./embedding_analysis")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(args.seed)
    
    # Experiment selector
    if args.names:
        target_names = args.names
    else:
        experiment = os.environ.get("EXPERIMENT", "2")
        if experiment == "1":
            target_names = ["david", "laura"]
        elif experiment == "2":
            target_names = ["david", "michael"]
        elif experiment == "3":
            target_names = ["maria", "laura"]
        elif experiment == "all":
            target_names = ["david", "michael", "maria", "laura"]
        else:
            target_names = ["david", "michael"]
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("\n" + "="*60)
    print("EMBEDDING ANALYSIS")
    print(f"Names: {target_names}")
    print("="*60 + "\n")
    
    # Load CLIP model
    print("Loading CLIP ViT-B-32...")
    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    tokenizer = get_tokenizer("ViT-B-32")
    name_to_gender = create_name_gender_mapping()
    
    # Use validation set for analysis
    val_dataset = FaceNameDataset(
        index_dir=args.index_dir,
        target_names=target_names,
        name_to_gender=name_to_gender,
        transform=preprocess,
        split="val",
        seed=args.seed,
        prompt_mode="deterministic",
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=lambda b: collate_fn(b, tokenizer),
        pin_memory=True,
    )
    
    print(f"Validation samples: {len(val_dataset)}")
    print()
    
    # Extract embeddings
    embeddings, labels = extract_embeddings(model, val_loader, device)
    print(f"Embeddings shape: {embeddings.shape}\n")
    
    # 1. Similarity distributions
    print("Computing similarity distributions...")
    same_sims, diff_sims = compute_similarity_distributions(embeddings, labels)
    same_mean, diff_mean, separation = plot_similarity_distributions(
        same_sims, diff_sims, 
        os.path.join(args.output_dir, "similarity_distributions.png")
    )
    
    # 2. Nearest neighbor classification
    print("\nRunning nearest neighbor classification...")
    name_to_idx = {name: i for i, name in enumerate(target_names)}
    predictions, per_class_acc, overall_acc = nearest_neighbor_classify(
        embeddings, labels, name_to_idx
    )
    
    # 3. Confusion matrix
    plot_confusion_matrix(
        labels, predictions, target_names,
        os.path.join(args.output_dir, "confusion_matrix.png")
    )
    
    # 4. Per-class accuracy
    plot_per_class_accuracy(
        per_class_acc,
        os.path.join(args.output_dir, "per_class_accuracy.png")
    )
    
    # 5. Silhouette score
    print("\nComputing silhouette score...")
    label_indices = [name_to_idx[l] for l in labels]
    sil_score = silhouette_score(embeddings.numpy(), label_indices)
    
    # Summary
    print("\n" + "="*60)
    print("ANALYSIS SUMMARY")
    print("="*60)
    print(f"\n📊 Similarity Distributions:")
    print(f"   Same-name mean similarity: {same_mean:.4f}")
    print(f"   Diff-name mean similarity: {diff_mean:.4f}")
    print(f"   Separation (higher=better): {separation:.4f}")
    
    print(f"\n📊 Nearest Neighbor Classification:")
    print(f"   Overall accuracy: {100*overall_acc:.1f}%")
    print(f"   Random baseline: {100/len(target_names):.1f}%")
    print(f"   Per-name breakdown:")
    for name, acc in per_class_acc.items():
        print(f"      {name.capitalize():12s}: {100*acc:.1f}%")
    
    print(f"\n📊 Silhouette Score: {sil_score:.4f}")
    print(f"   (Range: -1 to 1, higher = better cluster separation)")
    
    # Interpretation
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    if separation > 0.05:
        print("✓ Meaningful separation in embedding space.")
        print("  CLIP embeddings do capture some name-face signal.")
    elif separation > 0.01:
        print("~ Weak separation in embedding space.")
        print("  Some signal exists, but it's subtle.")
    else:
        print("✗ No meaningful separation in embedding space.")
        print("  CLIP embeddings don't distinguish these names.")
    
    if overall_acc > 0.6:
        print(f"\n✓ NN accuracy ({100*overall_acc:.1f}%) suggests learnable signal.")
    elif overall_acc > 1/len(target_names) + 0.05:
        print(f"\n~ NN accuracy ({100*overall_acc:.1f}%) shows weak signal.")
    else:
        print(f"\n✗ NN accuracy ({100*overall_acc:.1f}%) near random chance.")
    
    print(f"\n📁 Visualizations saved to: {args.output_dir}/")


if __name__ == "__main__":
    main()

