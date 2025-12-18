"""
Debug Prediction Bias in Name-Face Association Model.

Diagnoses why the model over-predicts some names (William, Nick) 
and under-predicts others (James, David).

Checks:
1. Classifier weight magnitudes
2. Embedding space structure (centrality)
3. Per-class confidence distributions
4. Proposes solutions

Usage:
    python debug_prediction_bias.py
"""
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from open_clip import create_model_and_transforms, get_tokenizer
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from collections import defaultdict
import os
import random

from clip_dataset import FaceNameDataset, create_name_gender_mapping


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class MultiNameDataset(torch.utils.data.Dataset):
    """Dataset for multiple names."""
    def __init__(self, index_dir, names, transform, split="train", 
                 train_ratio=0.8, seed=42, max_per_name=500):
        self.transform = transform
        self.samples = []
        self.names = names
        self.name_to_idx = {n: i for i, n in enumerate(names)}
        
        random.seed(seed)
        
        for name in names:
            index_path = os.path.join(index_dir, f"index_{name}.json")
            if not os.path.exists(index_path):
                continue
                
            with open(index_path) as f:
                data = json.load(f)
            
            good_images = data.get("good", [])
            random.shuffle(good_images)
            
            split_idx = int(len(good_images) * train_ratio)
            selected = good_images[:split_idx] if split == "train" else good_images[split_idx:]
            
            if max_per_name and len(selected) > max_per_name:
                selected = selected[:max_per_name]
            
            for path in selected:
                if os.path.exists(path):
                    self.samples.append((path, self.name_to_idx[name]))
        
        random.shuffle(self.samples)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


def extract_embeddings(model, dataloader, device):
    """Extract CLIP embeddings."""
    model.eval()
    all_embeddings = []
    all_labels = []
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for images, labels in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)
    
    return torch.cat(all_embeddings), torch.cat(all_labels)


def train_classifier_with_diagnostics(train_X, train_y, val_X, val_y, 
                                      num_classes, epochs=50, lr=0.01, device="cuda"):
    """Train classifier and return detailed diagnostics."""
    classifier = nn.Linear(train_X.shape[1], num_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    train_X = train_X.to(device)
    train_y = train_y.to(device)
    val_X = val_X.to(device)
    val_y = val_y.to(device)
    
    for epoch in range(epochs):
        classifier.train()
        optimizer.zero_grad()
        logits = classifier(train_X)
        loss = criterion(logits, train_y)
        loss.backward()
        optimizer.step()
    
    # Get final predictions and confidence
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_X)
        val_probs = torch.softmax(val_logits, dim=1)
        val_preds = val_logits.argmax(dim=1)
        val_confidence = val_probs.max(dim=1).values
    
    return classifier, val_preds.cpu(), val_probs.cpu(), val_confidence.cpu()


def analyze_classifier_weights(classifier, names):
    """Analyze the learned classifier weights."""
    weights = classifier.weight.detach().cpu().numpy()
    biases = classifier.bias.detach().cpu().numpy()
    
    # Weight magnitude per class
    weight_norms = np.linalg.norm(weights, axis=1)
    
    return {
        'weight_norms': weight_norms,
        'biases': biases,
        'weights': weights
    }


def analyze_embedding_centrality(embeddings, labels, num_classes):
    """Check if some classes are more "central" in embedding space."""
    # Global centroid
    global_centroid = embeddings.mean(dim=0)
    
    # Per-class analysis
    class_stats = {}
    for c in range(num_classes):
        mask = labels == c
        class_emb = embeddings[mask]
        
        if len(class_emb) == 0:
            continue
        
        class_centroid = class_emb.mean(dim=0)
        
        # Distance from global centroid
        dist_to_global = torch.norm(class_centroid - global_centroid).item()
        
        # Intra-class spread (how tight is the cluster)
        spread = torch.std(class_emb, dim=0).mean().item()
        
        # Average distance of samples to their class centroid
        avg_dist_to_centroid = torch.norm(class_emb - class_centroid, dim=1).mean().item()
        
        class_stats[c] = {
            'dist_to_global_centroid': dist_to_global,
            'intra_class_spread': spread,
            'avg_dist_to_class_centroid': avg_dist_to_centroid
        }
    
    return class_stats


def analyze_confidence_by_class(probs, labels, preds, num_classes):
    """Analyze prediction confidence patterns."""
    results = {}
    
    for c in range(num_classes):
        # Confidence when predicting this class
        pred_mask = preds == c
        if pred_mask.sum() > 0:
            conf_when_predicting = probs[pred_mask, c].mean().item()
        else:
            conf_when_predicting = 0
        
        # Confidence on actual samples of this class
        actual_mask = labels == c
        if actual_mask.sum() > 0:
            conf_on_actual = probs[actual_mask, c].mean().item()
            # Max probability assigned to correct class
            correct_mask = (labels == c) & (preds == c)
            if correct_mask.sum() > 0:
                conf_when_correct = probs[correct_mask, c].mean().item()
            else:
                conf_when_correct = 0
        else:
            conf_on_actual = 0
            conf_when_correct = 0
        
        results[c] = {
            'confidence_when_predicting': conf_when_predicting,
            'confidence_on_actual_samples': conf_on_actual,
            'confidence_when_correct': conf_when_correct,
            'times_predicted': pred_mask.sum().item(),
            'actual_count': actual_mask.sum().item()
        }
    
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-names", type=int, default=30)
    parser.add_argument("--index-dir", default="/home/leann/face-detection/data/index_files")
    parser.add_argument("--output-dir", default="./bias_debug")
    args = parser.parse_args()
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seed_everything(42)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*60)
    print("PREDICTION BIAS DEBUGGING")
    print("="*60)
    
    # Load names (same as scale_up_test)
    from scale_up_test import discover_names
    all_names = discover_names(args.index_dir, min_samples=200)
    names = [n for n, c in all_names[:args.num_names]]
    
    print(f"\nAnalyzing {len(names)} names...")
    
    # Load CLIP
    print("\nLoading CLIP...")
    model, _, preprocess = create_model_and_transforms("ViT-B-32", pretrained="openai")
    model = model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    
    # Create datasets
    train_dataset = MultiNameDataset(args.index_dir, names, preprocess, 
                                     split="train", max_per_name=500)
    val_dataset = MultiNameDataset(args.index_dir, names, preprocess,
                                   split="val", max_per_name=500)
    
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    train_X, train_y = extract_embeddings(model, train_loader, device)
    val_X, val_y = extract_embeddings(model, val_loader, device)
    
    # Train classifier
    print("\nTraining classifier...")
    classifier, preds, probs, confidence = train_classifier_with_diagnostics(
        train_X, train_y, val_X, val_y, len(names), device=device
    )
    
    # === DIAGNOSTIC 1: Classifier Weights ===
    print("\n" + "="*60)
    print("DIAGNOSTIC 1: Classifier Weight Analysis")
    print("="*60)
    
    weight_info = analyze_classifier_weights(classifier, names)
    
    # Sort by weight norm
    sorted_indices = np.argsort(weight_info['weight_norms'])[::-1]
    
    print(f"\n{'Name':<12} {'Weight Norm':<12} {'Bias':<10} {'Interpretation'}")
    print("-"*55)
    for idx in sorted_indices[:10]:
        norm = weight_info['weight_norms'][idx]
        bias = weight_info['biases'][idx]
        interpretation = "Strong feature detector" if norm > np.median(weight_info['weight_norms']) else "Weak"
        print(f"{names[idx].capitalize():<12} {norm:>8.3f}     {bias:>+7.3f}    {interpretation}")
    
    print(f"\n...Bottom 5...")
    for idx in sorted_indices[-5:]:
        norm = weight_info['weight_norms'][idx]
        bias = weight_info['biases'][idx]
        print(f"{names[idx].capitalize():<12} {norm:>8.3f}     {bias:>+7.3f}")
    
    # === DIAGNOSTIC 2: Embedding Centrality ===
    print("\n" + "="*60)
    print("DIAGNOSTIC 2: Embedding Space Centrality")
    print("="*60)
    
    centrality = analyze_embedding_centrality(val_X, val_y, len(names))
    
    # Sort by distance to global centroid
    centrality_sorted = sorted(centrality.items(), 
                               key=lambda x: x[1]['dist_to_global_centroid'])
    
    print(f"\n{'Name':<12} {'Dist to Center':<15} {'Spread':<10} {'Interpretation'}")
    print("-"*55)
    print("Most CENTRAL (near global mean - default predictions):")
    for idx, stats in centrality_sorted[:5]:
        print(f"  {names[idx].capitalize():<12} {stats['dist_to_global_centroid']:>8.4f}       "
              f"{stats['intra_class_spread']:>6.4f}     Close to 'average face'")
    
    print("\nMost PERIPHERAL (far from global mean):")
    for idx, stats in centrality_sorted[-5:]:
        print(f"  {names[idx].capitalize():<12} {stats['dist_to_global_centroid']:>8.4f}       "
              f"{stats['intra_class_spread']:>6.4f}     Distinctive region")
    
    # === DIAGNOSTIC 3: Confidence Analysis ===
    print("\n" + "="*60)
    print("DIAGNOSTIC 3: Confidence Patterns")
    print("="*60)
    
    conf_analysis = analyze_confidence_by_class(probs, val_y, preds, len(names))
    
    # Sort by times predicted
    conf_sorted = sorted(conf_analysis.items(), 
                        key=lambda x: x[1]['times_predicted'], reverse=True)
    
    print(f"\n{'Name':<12} {'Predicted':<10} {'Actual':<8} {'Conf When Pred':<15} {'Bias'}")
    print("-"*65)
    for idx, stats in conf_sorted[:10]:
        ratio = stats['times_predicted'] / max(stats['actual_count'], 1)
        bias_str = f"{ratio:.2f}x" if ratio > 1.1 else f"{ratio:.2f}x" if ratio < 0.9 else "balanced"
        print(f"{names[idx].capitalize():<12} {stats['times_predicted']:<10} "
              f"{stats['actual_count']:<8} {stats['confidence_when_predicting']*100:>6.1f}%        {bias_str}")
    
    # === DIAGNOSTIC 4: Root Cause Analysis ===
    print("\n" + "="*60)
    print("ROOT CAUSE ANALYSIS")
    print("="*60)
    
    # Correlate weight norms with prediction frequency
    pred_counts = np.bincount(preds.numpy(), minlength=len(names))
    correlation = np.corrcoef(weight_info['weight_norms'], pred_counts)[0, 1]
    
    print(f"\n1. Correlation: Weight Norm ↔ Prediction Frequency: {correlation:.3f}")
    if correlation > 0.5:
        print("   → HIGH correlation: Model favors classes with larger weight vectors")
        print("   → Solution: Apply L2 regularization or class-balanced loss")
    
    # Check bias term correlation
    bias_corr = np.corrcoef(weight_info['biases'], pred_counts)[0, 1]
    print(f"\n2. Correlation: Bias Term ↔ Prediction Frequency: {bias_corr:.3f}")
    if bias_corr > 0.5:
        print("   → HIGH correlation: Bias terms are driving over-prediction")
        print("   → Solution: Use bias-free classifier or regularize biases")
    
    # Check centrality correlation
    centrality_dists = [centrality[i]['dist_to_global_centroid'] for i in range(len(names))]
    cent_corr = np.corrcoef(centrality_dists, pred_counts)[0, 1]
    print(f"\n3. Correlation: Centrality ↔ Prediction Frequency: {cent_corr:.3f}")
    if abs(cent_corr) > 0.3:
        print("   → Embedding space structure affects predictions")
    
    # === SOLUTIONS ===
    print("\n" + "="*60)
    print("RECOMMENDED SOLUTIONS")
    print("="*60)
    
    print("""
1. CLASS-BALANCED LOSS
   Weight the loss inversely by prediction frequency.
   Names that are over-predicted get higher loss penalty.

2. TEMPERATURE SCALING
   Apply temperature to soften the softmax before prediction.
   Reduces confidence gap between "easy" and "hard" names.

3. FOCAL LOSS
   Down-weight easy examples, focus on hard ones.
   Would reduce confidence on William, increase on James.

4. REMOVE BIAS TERM
   Train classifier without bias (just weights).
   Removes systematic preference for certain classes.

5. BALANCED PREDICTION CONSTRAINT
   Post-hoc adjust predictions to enforce equal frequencies.
   Simple but loses some discriminative power.
""")
    
    # Save diagnostics
    results = {
        'weight_norms': {names[i]: float(weight_info['weight_norms'][i]) for i in range(len(names))},
        'biases': {names[i]: float(weight_info['biases'][i]) for i in range(len(names))},
        'prediction_counts': {names[i]: int(pred_counts[i]) for i in range(len(names))},
        'correlations': {
            'weight_norm_vs_pred_freq': float(correlation),
            'bias_vs_pred_freq': float(bias_corr),
            'centrality_vs_pred_freq': float(cent_corr)
        }
    }
    
    with open(f"{args.output_dir}/bias_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Diagnostics saved to: {args.output_dir}/bias_diagnostics.json")


if __name__ == "__main__":
    main()

