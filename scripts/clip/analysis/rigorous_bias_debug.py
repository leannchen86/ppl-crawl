"""
Rigorous Prediction Bias Debugging Tool

Diagnoses whether prediction bias is due to:
1. Real signal differences (some names genuinely more learnable)
2. Dataset quality confounds (image quality, demographics)
3. Optimization artifacts (softmax geometry, norm blow-up)

Key Tests:
- Permutation test (shuffle labels)
- Cosine classifier comparison (force equal norms)
- Logit decomposition (bias vs norm vs alignment)
- Balanced inference probe (prior correction without retraining)
- Confidence distribution analysis
- Image quality proxy correlations

Usage:
    python scripts/clip/analysis/rigorous_bias_debug.py --num-names 30
"""
import argparse
import json
import os
import random
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from contextlib import nullcontext
from open_clip import create_model_and_transforms, get_tokenizer
from sklearn.metrics import roc_auc_score
from collections import defaultdict
import matplotlib.pyplot as plt
import cv2
from PIL import Image


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def discover_names(index_dir: str, min_samples: int = 100):
    """Find all names with enough samples (from index_*.json files)."""
    names_with_counts = []

    pattern = os.path.join(index_dir, "index_*.json")
    for filepath in glob.glob(pattern):
        name = os.path.basename(filepath).replace("index_", "").replace(".json", "")
        try:
            with open(filepath) as f:
                data = json.load(f)
            count = data.get("counts", {}).get("good", 0)
            if count >= min_samples:
                names_with_counts.append((name, count))
        except Exception as e:
            print(f"Warning: Could not read {filepath}: {e}")

    names_with_counts.sort(key=lambda x: -x[1])
    return names_with_counts


class MultiNameDataset(torch.utils.data.Dataset):
    """Dataset for multiple names with image path tracking."""
    def __init__(self, index_dir, names, transform, split="train", 
                 train_ratio=0.8, seed=42, max_per_name=500):
        self.transform = transform
        self.samples = []  # (path, name_idx)
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
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label, path


def extract_embeddings_with_paths(model, dataloader, device):
    """Extract CLIP embeddings with image paths."""
    model.eval()
    all_embeddings = []
    all_labels = []
    all_paths = []
    
    device_type = "cuda" if torch.cuda.is_available() and "cuda" in str(device) else "cpu"
    ctx = torch.autocast(device_type=device_type, dtype=torch.float16) if device_type == "cuda" else nullcontext()
    
    with torch.no_grad(), ctx:
        for images, labels, paths in tqdm(dataloader, desc="Extracting"):
            images = images.to(device)
            emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embeddings.append(emb.cpu())
            all_labels.append(labels)
            all_paths.extend(paths)
    
    return torch.cat(all_embeddings), torch.cat(all_labels), all_paths


def compute_image_quality_proxies(paths, sample_size=100):
    """Compute image quality proxies: blur, brightness, face size."""
    results = defaultdict(list)
    
    # Sample subset for efficiency
    if len(paths) > sample_size:
        indices = random.sample(range(len(paths)), sample_size)
        paths = [paths[i] for i in indices]
    
    for path in tqdm(paths, desc="Computing quality", leave=False):
        try:
            img = cv2.imread(path)
            if img is None:
                continue
            
            # Blur score (Laplacian variance)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            blur = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Brightness (mean intensity)
            brightness = gray.mean()
            
            # Face size proxy (image dimensions)
            h, w = img.shape[:2]
            size = (h * w) ** 0.5
            
            results['blur'].append(blur)
            results['brightness'].append(brightness)
            results['size'].append(size)
        except:
            pass
    
    return {k: np.mean(v) if v else np.nan for k, v in results.items()}


class StandardLinearClassifier(nn.Module):
    """Standard linear classifier with unconstrained weights."""
    def __init__(self, input_dim, num_classes, weight_decay=0.0):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)
        self.weight_decay = weight_decay
    
    def forward(self, x):
        return self.fc(x)
    
    def get_weights_biases(self):
        return self.fc.weight.detach(), self.fc.bias.detach()


class CosineClassifier(nn.Module):
    """Cosine classifier with normalized weights (equal norms)."""
    def __init__(self, input_dim, num_classes, scale=20.0):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(num_classes, input_dim))
        self.scale = nn.Parameter(torch.tensor(scale))
    
    def forward(self, x):
        # Normalize both weights and inputs
        w_norm = F.normalize(self.weight, p=2, dim=1)
        x_norm = F.normalize(x, p=2, dim=1)
        return self.scale * (x_norm @ w_norm.T)
    
    def get_weights_biases(self):
        return F.normalize(self.weight.detach(), p=2, dim=1), None


def train_classifier(classifier, train_X, train_y, val_X, val_y, 
                    epochs=50, lr=0.01, weight_decay=0.0, device="cuda"):
    """Train classifier and return diagnostics."""
    classifier = classifier.to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=weight_decay)
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
    
    # Get final predictions
    classifier.eval()
    with torch.no_grad():
        val_logits = classifier(val_X)
        val_probs = torch.softmax(val_logits, dim=1)
        val_preds = val_logits.argmax(dim=1)
    
    return classifier, val_preds.cpu(), val_probs.cpu(), val_logits.cpu()


def permutation_test(train_X, train_y, val_X, val_y, num_classes, n_permutations=5):
    """Test if weight norm correlation is an artifact by shuffling labels."""
    results = []
    
    for i in range(n_permutations):
        # Shuffle training labels
        perm_train_y = train_y[torch.randperm(len(train_y))]
        
        # Train classifier on shuffled data
        classifier = StandardLinearClassifier(train_X.shape[1], num_classes)
        classifier, preds, probs, logits = train_classifier(
            classifier, train_X, perm_train_y, val_X, val_y,
            epochs=50, device=train_X.device
        )
        
        # Compute correlation
        weights, biases = classifier.get_weights_biases()
        weight_norms = weights.norm(dim=1).cpu().numpy()
        pred_counts = np.bincount(preds.numpy(), minlength=num_classes)
        
        corr = np.corrcoef(weight_norms, pred_counts)[0, 1]
        results.append(corr)
    
    return results


def logit_decomposition(classifier, embeddings, labels, num_classes):
    """Decompose logits into bias, norm, and alignment components."""
    weights, biases = classifier.get_weights_biases()
    
    results = {}
    for c in range(num_classes):
        mask = labels == c
        non_mask = labels != c
        
        if mask.sum() == 0:
            results[c] = {
                'bias': np.nan,
                'weight_norm': np.nan,
                'mean_alignment_on_class': np.nan,
                'mean_alignment_on_nonclass': np.nan,
                'mean_logit_on_class': np.nan,
                'mean_logit_on_nonclass': np.nan,
            }
            continue
        
        w_c = weights[c].cpu()
        b_c = biases[c].item() if biases is not None else 0.0
        
        # Cosine similarity
        emb_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
        w_c_norm = w_c / w_c.norm()
        cosine = (emb_norm @ w_c_norm).numpy()
        
        # Logits (if we had them)
        logits_c = (embeddings @ w_c).numpy() + b_c
        
        results[c] = {
            'bias': b_c,
            'weight_norm': w_c.norm().item(),
            'mean_alignment_on_class': cosine[mask].mean(),
            'mean_alignment_on_nonclass': cosine[non_mask].mean() if non_mask.sum() > 0 else np.nan,
            'mean_logit_on_class': logits_c[mask].mean(),
            'mean_logit_on_nonclass': logits_c[non_mask].mean() if non_mask.sum() > 0 else np.nan,
        }
    
    return results


def balanced_inference_probe(logits, labels, num_classes):
    """Test if prior correction reduces bias without retraining."""
    # Original predictions
    original_preds = logits.argmax(dim=1)
    original_counts = np.bincount(original_preds.numpy(), minlength=num_classes)
    
    # Compute class priors from training (approximate from predictions)
    # In practice, use actual training counts
    prior_log = torch.log(torch.tensor(original_counts + 1.0))
    
    # Adjusted logits
    adjusted_logits = logits - prior_log
    adjusted_preds = adjusted_logits.argmax(dim=1)
    adjusted_counts = np.bincount(adjusted_preds.numpy(), minlength=num_classes)
    
    # Compute skew (coefficient of variation)
    original_skew = np.std(original_counts) / np.mean(original_counts)
    adjusted_skew = np.std(adjusted_counts) / np.mean(adjusted_counts)
    
    # Accuracy
    original_acc = (original_preds == labels).float().mean().item()
    adjusted_acc = (adjusted_preds == labels).float().mean().item()
    
    return {
        'original_counts': original_counts,
        'adjusted_counts': adjusted_counts,
        'original_skew': original_skew,
        'adjusted_skew': adjusted_skew,
        'original_acc': original_acc,
        'adjusted_acc': adjusted_acc,
        'skew_reduction': (original_skew - adjusted_skew) / original_skew * 100
    }


def confidence_distribution_analysis(probs, preds, labels, num_classes):
    """Analyze confidence distributions for overpredicted classes."""
    results = {}
    
    for c in range(num_classes):
        pred_mask = preds == c
        label_mask = labels == c
        correct_mask = pred_mask & label_mask
        wrong_mask = pred_mask & ~label_mask
        
        if pred_mask.sum() == 0:
            results[c] = {
                'mean_conf_when_correct': np.nan,
                'mean_conf_when_wrong': np.nan,
                'p50_conf_when_predicting': np.nan,
                'p90_conf_when_predicting': np.nan,
            }
            continue
        
        conf_on_preds = probs[pred_mask, c].numpy()
        
        results[c] = {
            'mean_conf_when_correct': probs[correct_mask, c].mean().item() if correct_mask.sum() > 0 else np.nan,
            'mean_conf_when_wrong': probs[wrong_mask, c].mean().item() if wrong_mask.sum() > 0 else np.nan,
            'p50_conf_when_predicting': np.median(conf_on_preds),
            'p90_conf_when_predicting': np.percentile(conf_on_preds, 90),
        }
    
    return results


def compute_per_class_auc(logits, labels, num_classes):
    """Compute one-vs-rest AUC for each class."""
    from sklearn.preprocessing import label_binarize
    
    labels_bin = label_binarize(labels.numpy(), classes=range(num_classes))
    probs = torch.softmax(logits, dim=1).numpy()
    
    aucs = {}
    for c in range(num_classes):
        if labels_bin[:, c].sum() == 0:
            aucs[c] = np.nan
        else:
            try:
                aucs[c] = roc_auc_score(labels_bin[:, c], probs[:, c])
            except:
                aucs[c] = np.nan
    
    return aucs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-names", type=int, default=30)
    parser.add_argument(
        "--index-dir",
        default="/home/leann/face-detection/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001",
    )
    parser.add_argument("--output-dir", default="./rigorous_bias_debug")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("="*70)
    print("RIGOROUS PREDICTION BIAS DEBUGGING")
    print("="*70)
    
    # Load names
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
    
    # Create datasets (with path tracking)
    train_dataset = MultiNameDataset(args.index_dir, names, preprocess, 
                                     split="train", max_per_name=500, seed=args.seed)
    val_dataset = MultiNameDataset(args.index_dir, names, preprocess,
                                   split="val", max_per_name=500, seed=args.seed)
    
    train_loader = DataLoader(train_dataset, batch_size=64, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, num_workers=4)
    
    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Extract embeddings
    print("\nExtracting embeddings...")
    train_X, train_y, train_paths = extract_embeddings_with_paths(model, train_loader, device)
    val_X, val_y, val_paths = extract_embeddings_with_paths(model, val_loader, device)
    
    # ===== TEST 1: Standard Classifier =====
    print("\n" + "="*70)
    print("TEST 1: Standard Linear Classifier (Baseline)")
    print("="*70)
    
    standard_clf = StandardLinearClassifier(train_X.shape[1], len(names), weight_decay=1e-4)
    standard_clf, std_preds, std_probs, std_logits = train_classifier(
        standard_clf, train_X, train_y, val_X, val_y,
        epochs=50, lr=0.01, weight_decay=1e-4, device=device
    )
    
    std_weights, std_biases = standard_clf.get_weights_biases()
    std_weight_norms = std_weights.norm(dim=1).cpu().numpy()
    std_pred_counts = np.bincount(std_preds.numpy(), minlength=len(names))
    std_acc = (std_preds == val_y).float().mean().item()
    
    print(f"\nAccuracy: {100*std_acc:.1f}%")
    print(f"Weight norm range: [{std_weight_norms.min():.3f}, {std_weight_norms.max():.3f}]")
    print(f"Prediction count range: [{std_pred_counts.min()}, {std_pred_counts.max()}]")
    
    # ===== TEST 2: Cosine Classifier =====
    print("\n" + "="*70)
    print("TEST 2: Cosine Classifier (Equal Norms)")
    print("="*70)
    
    cosine_clf = CosineClassifier(train_X.shape[1], len(names), scale=20.0)
    cosine_clf, cos_preds, cos_probs, cos_logits = train_classifier(
        cosine_clf, train_X, train_y, val_X, val_y,
        epochs=50, lr=0.01, weight_decay=1e-4, device=device
    )
    
    cos_pred_counts = np.bincount(cos_preds.numpy(), minlength=len(names))
    cos_acc = (cos_preds == val_y).float().mean().item()
    
    print(f"\nAccuracy: {100*cos_acc:.1f}%")
    print(f"Prediction count range: [{cos_pred_counts.min()}, {cos_pred_counts.max()}]")
    
    # Compare skew
    std_skew = np.std(std_pred_counts) / np.mean(std_pred_counts)
    cos_skew = np.std(cos_pred_counts) / np.mean(cos_pred_counts)
    
    print(f"\nPrediction Skew:")
    print(f"  Standard: {std_skew:.3f}")
    print(f"  Cosine:   {cos_skew:.3f}")
    print(f"  Reduction: {(std_skew - cos_skew)/std_skew*100:.1f}%")
    
    if cos_skew < std_skew * 0.7 and abs(cos_acc - std_acc) < 0.03:
        print("\n✓ CONCLUSION: Prediction bias is largely NORM-DRIVEN")
        print("  (Equal norms reduce skew without hurting accuracy)")
    else:
        print("\n⚠ CONCLUSION: Prediction bias reflects REAL SIGNAL differences")
        print("  (Equalizing norms either doesn't reduce skew or hurts accuracy)")
    
    # ===== TEST 3: Permutation Test =====
    print("\n" + "="*70)
    print("TEST 3: Permutation Test (5 runs with shuffled labels)")
    print("="*70)
    
    print("\nRunning permutations (this may take a few minutes)...")
    perm_correlations = permutation_test(train_X, train_y, val_X, val_y, len(names), n_permutations=5)
    
    # Real correlation
    real_corr = np.corrcoef(std_weight_norms, std_pred_counts)[0, 1]
    
    print(f"\nReal data correlation: {real_corr:.3f}")
    print(f"Permuted correlations: {[f'{c:.3f}' for c in perm_correlations]}")
    print(f"Mean permuted:         {np.mean(perm_correlations):.3f}")
    
    if real_corr > np.mean(perm_correlations) + 2*np.std(perm_correlations):
        print("\n✓ CONCLUSION: Correlation is DATA-DRIVEN (not just optimization artifact)")
    else:
        print("\n⚠ CONCLUSION: Correlation is largely OPTIMIZATION ARTIFACT")
    
    # ===== TEST 4: Logit Decomposition =====
    print("\n" + "="*70)
    print("TEST 4: Logit Decomposition (Why do some classes win?)")
    print("="*70)
    
    decomp = logit_decomposition(standard_clf, val_X, val_y, len(names))
    
    # Sort by prediction frequency
    sorted_indices = np.argsort(std_pred_counts)[::-1]
    
    print(f"\n{'Name':<12} {'Pred':<6} {'Norm':<8} {'Bias':<8} {'Align(class)':<12} {'Align(other)'}")
    print("-"*70)
    for idx in sorted_indices[:10]:
        if idx not in decomp:
            continue
        d = decomp[idx]
        print(f"{names[idx].capitalize():<12} {std_pred_counts[idx]:<6} "
              f"{d['weight_norm']:>7.3f} {d['bias']:>+7.3f} "
              f"{d['mean_alignment_on_class']:>11.3f} {d['mean_alignment_on_nonclass']:>11.3f}")
    
    # ===== TEST 5: Balanced Inference Probe =====
    print("\n" + "="*70)
    print("TEST 5: Balanced Inference (Prior Correction Without Retraining)")
    print("="*70)
    
    balanced_results = balanced_inference_probe(std_logits, val_y, len(names))
    
    print(f"\nPrediction skew reduction: {balanced_results['skew_reduction']:.1f}%")
    print(f"Accuracy change: {100*(balanced_results['adjusted_acc'] - balanced_results['original_acc']):.1f}%")
    
    if balanced_results['skew_reduction'] > 30:
        print("\n✓ CONCLUSION: Bias is largely PRIOR-DRIVEN")
        print("  (Simple logit adjustment reduces skew significantly)")
    
    # ===== TEST 6: Confidence Distribution =====
    print("\n" + "="*70)
    print("TEST 6: Confidence Distribution Analysis")
    print("="*70)
    
    conf_analysis = confidence_distribution_analysis(std_probs, std_preds, val_y, len(names))
    
    print(f"\n{'Name':<12} {'Pred':<6} {'Conf(correct)':<14} {'Conf(wrong)':<14} {'Pattern'}")
    print("-"*70)
    for idx in sorted_indices[:10]:
        if idx not in conf_analysis or std_pred_counts[idx] == 0:
            continue
        ca = conf_analysis[idx]
        
        # Determine pattern
        if not np.isnan(ca['mean_conf_when_wrong']):
            if ca['mean_conf_when_wrong'] < 0.15:
                pattern = "Prior effect"
            elif ca['mean_conf_when_wrong'] > 0.25:
                pattern = "Spurious feature"
            else:
                pattern = "Mixed"
        else:
            pattern = "Always correct"
        
        print(f"{names[idx].capitalize():<12} {std_pred_counts[idx]:<6} "
              f"{ca['mean_conf_when_correct']*100:>6.1f}%        "
              f"{ca['mean_conf_when_wrong']*100:>6.1f}%        {pattern}")
    
    # ===== TEST 7: Per-Class AUC =====
    print("\n" + "="*70)
    print("TEST 7: Per-Class AUC (Signal Strength Independent of Argmax)")
    print("="*70)
    
    aucs = compute_per_class_auc(std_logits, val_y, len(names))
    
    # Correlation with prediction counts
    valid_indices = [i for i in range(len(names)) if not np.isnan(aucs.get(i, np.nan))]
    auc_values = [aucs[i] for i in valid_indices]
    pred_counts_valid = [std_pred_counts[i] for i in valid_indices]
    
    auc_corr = np.corrcoef(auc_values, pred_counts_valid)[0, 1] if len(valid_indices) > 2 else np.nan
    
    print(f"\nCorrelation: AUC ↔ Prediction Frequency: {auc_corr:.3f}")
    print("\nTop 5 by AUC (true signal strength):")
    sorted_by_auc = sorted([(i, aucs.get(i, 0)) for i in range(len(names))], key=lambda x: -x[1])
    for idx, auc_val in sorted_by_auc[:5]:
        print(f"  {names[idx].capitalize():<12} AUC={auc_val:.3f}, Predicted {std_pred_counts[idx]} times")
    
    # ===== TEST 8: Image Quality Proxies =====
    print("\n" + "="*70)
    print("TEST 8: Image Quality Proxy Correlations")
    print("="*70)
    
    print("\nComputing quality metrics (sampling 100 images per name)...")
    quality_by_name = {}
    for idx, name in enumerate(names):
        name_paths = [p for p, l in zip(val_paths, val_y.tolist()) if l == idx]
        if len(name_paths) > 0:
            quality_by_name[idx] = compute_image_quality_proxies(name_paths, sample_size=100)
    
    # Correlation with prediction counts
    quality_metrics = ['blur', 'brightness', 'size']
    for metric in quality_metrics:
        values = [quality_by_name[i][metric] for i in range(len(names)) if i in quality_by_name]
        counts = [std_pred_counts[i] for i in range(len(names)) if i in quality_by_name]
        
        if len(values) > 2:
            corr = np.corrcoef(values, counts)[0, 1]
            print(f"\nCorrelation: {metric.capitalize()} ↔ Prediction Frequency: {corr:.3f}")
            if abs(corr) > 0.3:
                print(f"  → {metric.capitalize()} may be a CONFOUND")
    
    # ===== FINAL SUMMARY =====
    print("\n" + "="*70)
    print("DIAGNOSIS SUMMARY")
    print("="*70)
    
    print("\nKey Findings:")
    print(f"1. Cosine classifier skew reduction: {(std_skew - cos_skew)/std_skew*100:.1f}%")
    print(f"2. Permutation test: Real corr {real_corr:.3f} vs Permuted mean {np.mean(perm_correlations):.3f}")
    print(f"3. Balanced inference skew reduction: {balanced_results['skew_reduction']:.1f}%")
    print(f"4. AUC ↔ Prediction correlation: {auc_corr:.3f}")
    
    print("\nRecommended Actions:")
    if (std_skew - cos_skew)/std_skew > 0.3:
        print("✓ Use cosine classifier or weight normalization")
    if balanced_results['skew_reduction'] > 30:
        print("✓ Apply logit adjustment (prior correction)")
    if real_corr < np.mean(perm_correlations) + np.std(perm_correlations):
        print("⚠ Consider that bias is mainly optimization artifact")
    
    # Save diagnostics
    # Convert numpy types to native Python types
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        return obj
    
    results = {
        'standard_accuracy': float(std_acc),
        'cosine_accuracy': float(cos_acc),
        'standard_skew': float(std_skew),
        'cosine_skew': float(cos_skew),
        'skew_reduction_pct': float((std_skew - cos_skew)/std_skew*100),
        'real_correlation': float(real_corr),
        'permuted_correlations': [float(x) for x in perm_correlations],
        'balanced_inference': convert_to_serializable(balanced_results),
        'names': names,
        'prediction_counts': std_pred_counts.tolist(),
    }
    
    with open(f"{args.output_dir}/rigorous_diagnostics.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📁 Results saved to: {args.output_dir}/rigorous_diagnostics.json")


if __name__ == "__main__":
    main()

