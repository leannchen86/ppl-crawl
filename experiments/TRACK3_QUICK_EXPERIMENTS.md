# Track 3: Quick Experiments (DINOv2/v3, Contrastive Learning, etc.)

## Project Context

### What We're Trying to Do
Run multiple quick experiments with different embedding models and training approaches to find promising directions for face-name association learning.

### Why This Track
- These experiments are **low effort, potentially high reward**
- Most use frozen pretrained encoders + simple classifiers
- Can run several in parallel on a single GPU
- Results inform whether to invest more in any particular direction

### What We've Tried (Phase 1-3 Summary)
| Approach | Result | Problem |
|----------|--------|---------|
| CLIP ViT-B-32 linear probe | 13.9% acc (30 names) | William dominance, web-bias |
| ArcFace linear probe | ~12% acc | More balanced but face-ID focused |
| Full fine-tuning | Catastrophic forgetting | Train 98%, val 9% |

---

## Data Available

```
/home/leann/face-detection/data/index_files/          # Original images
/home/leann/face-detection/data/index_files_facechips512_m0.5_reflect/  # 512x512 face chips
```

- **Total images:** 434,528
- **Total names:** 500
- **Names with >200 images:** 437

---

## Experiments in This Track

### Experiment 3.1: DINOv2 Linear Probe

**Hypothesis:** DINOv2 is self-supervised (no text), so less biased toward web aesthetics than CLIP.

```python
from transformers import AutoModel, AutoImageProcessor

# Load DINOv2
model = AutoModel.from_pretrained("facebook/dinov2-base")
processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")

# Extract embeddings (freeze encoder)
model.eval()
with torch.no_grad():
    inputs = processor(images, return_tensors="pt")
    outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0]  # CLS token (768-dim)

# Train linear classifier on top
classifier = nn.Linear(768, 500)
```

**Variants to try:**
- `dinov2-small` (384-dim) - faster
- `dinov2-base` (768-dim) - recommended
- `dinov2-large` (1024-dim) - if base works well
- `dinov2-giant` (1536-dim) - best features but slower

**Output:** `results/track3_dinov2/`

---

### Experiment 3.2: DINOv3 Linear Probe (NEW - Aug 2025)

**Hypothesis:** Latest version may have improvements.

```python
# DINOv3 just released
model = AutoModel.from_pretrained("facebook/dinov3-vitb16-pretrain-lvd1689m")
```

**Output:** `results/track3_dinov3/`

---

### Experiment 3.3: ArcFace 500-Class Classifier

**Hypothesis:** We already have ArcFace embeddings from Phase 3. Scaling to 500 classes might work.

```python
# Load existing embeddings (if saved) or extract new ones
from insightface.app import FaceAnalysis

app = FaceAnalysis(providers=['CUDAExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Extract 512-dim ArcFace embeddings
faces = app.get(image)
embedding = faces[0].embedding  # 512-dim

# Train classifier
classifier = nn.Linear(512, 500)
```

**Note:** ArcFace is trained for face identity verification, so embeddings cluster by person, not by name. But there might be residual signal.

**Output:** `results/track3_arcface_500/`

---

### Experiment 3.4: Supervised Contrastive Learning (SupCon)

**Hypothesis:** Instead of cross-entropy, train embeddings where same-name faces are close, different-name faces are far.

```python
# SupCon loss (from https://arxiv.org/abs/2004.11362)
class SupConLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        # features: (batch, dim), L2 normalized
        # labels: (batch,)
        batch_size = features.shape[0]

        # Compute similarity matrix
        sim = torch.matmul(features, features.T) / self.temperature

        # Mask for same-class pairs
        mask = labels.unsqueeze(0) == labels.unsqueeze(1)
        mask = mask.float() - torch.eye(batch_size).to(mask.device)

        # Compute loss
        exp_sim = torch.exp(sim)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True))
        mean_log_prob = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)
        loss = -mean_log_prob.mean()
        return loss

# Training: encoder + projection head
encoder = timm.create_model('resnet50', pretrained=True, num_classes=0)
projector = nn.Sequential(
    nn.Linear(2048, 2048),
    nn.ReLU(),
    nn.Linear(2048, 128)  # 128-dim embedding
)

# After training, use encoder features for linear probe
```

**Output:** `results/track3_supcon/`

---

### Experiment 3.5: CLIP with Different Prompts

**Hypothesis:** Zero-shot CLIP performance depends heavily on prompt engineering.

```python
import open_clip

model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-B-32')

# Try different prompt templates
templates = [
    "a photo of a person named {}",
    "a face photo of someone called {}",
    "{}, a common first name",
    "this person's name is {}",
    "a portrait of {}",
]

# Ensemble multiple prompts
text_features = []
for template in templates:
    texts = [template.format(name) for name in names]
    text_tokens = tokenizer(texts)
    with torch.no_grad():
        feats = model.encode_text(text_tokens)
        feats /= feats.norm(dim=-1, keepdim=True)
        text_features.append(feats)

# Average across prompts
text_features = torch.stack(text_features).mean(dim=0)
text_features /= text_features.norm(dim=-1, keepdim=True)

# Zero-shot classification
image_features = model.encode_image(images)
image_features /= image_features.norm(dim=-1, keepdim=True)
logits = image_features @ text_features.T
```

**Output:** `results/track3_clip_prompts/`

---

### Experiment 3.6: Ensemble (CLIP + ArcFace + DINOv2)

**Hypothesis:** Different models capture different aspects. Ensemble might be better than any single model.

```python
# Extract features from each model
clip_features = clip_model.encode_image(images)      # 512-dim
arcface_features = arcface_model.get(images)         # 512-dim
dino_features = dino_model(images).last_hidden_state[:, 0]  # 768-dim

# Option A: Concatenate
combined = torch.cat([clip_features, arcface_features, dino_features], dim=1)  # 1792-dim
classifier = nn.Linear(1792, 500)

# Option B: Late fusion (train separate classifiers, average logits)
clip_logits = clip_classifier(clip_features)
arcface_logits = arcface_classifier(arcface_features)
dino_logits = dino_classifier(dino_features)
final_logits = (clip_logits + arcface_logits + dino_logits) / 3

# Option C: Learned weighting
weights = nn.Parameter(torch.ones(3) / 3)
final_logits = weights[0] * clip_logits + weights[1] * arcface_logits + weights[2] * dino_logits
```

**Output:** `results/track3_ensemble/`

---

### Experiment 3.7: SigLIP (Better CLIP Alternative)

**Hypothesis:** SigLIP uses sigmoid loss instead of softmax, might be better calibrated.

```python
from transformers import AutoModel, AutoProcessor

model = AutoModel.from_pretrained("google/siglip-base-patch16-224")
processor = AutoProcessor.from_pretrained("google/siglip-base-patch16-224")

# Extract image features, train linear classifier
```

**Output:** `results/track3_siglip/`

---

## Priority Order

Run in this order (highest impact first):

1. **DINOv2-base** - Different inductive bias than CLIP
2. **ArcFace 500-class** - Already have infrastructure
3. **Ensemble** - Combine best models
4. **SupCon** - Different training objective
5. **SigLIP** - CLIP alternative
6. **CLIP prompts** - Low effort
7. **DINOv3** - If DINOv2 works well

---

## Shared Evaluation Code

```python
def evaluate_model(predictions, true_labels, names):
    """Standardized evaluation for all experiments."""
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np

    # Overall accuracy
    accuracy = (predictions == true_labels).mean()

    # Prediction CV (coefficient of variation)
    pred_counts = np.bincount(predictions, minlength=len(names))
    pred_cv = pred_counts.std() / pred_counts.mean()

    # Per-class metrics
    report = classification_report(true_labels, predictions,
                                   target_names=names, output_dict=True)

    # Top/bottom names
    f1_scores = [(name, report[name]['f1-score']) for name in names]
    f1_scores.sort(key=lambda x: -x[1])

    return {
        'accuracy': accuracy,
        'prediction_cv': pred_cv,
        'top5_names': f1_scores[:5],
        'bottom5_names': f1_scores[-5:],
        'per_class_report': report
    }
```

---

## Output Format (Same for All)

```
results/track3_{experiment}/
├── config.json           # Model and training config
├── predictions.npy       # (N,) predicted class indices
├── true_labels.npy       # (N,) ground truth indices
├── names.json            # List of 500 name strings
├── results.csv           # Per-name precision/recall/F1
├── embeddings.npy        # Optional: extracted embeddings for analysis
└── summary.json          # Key metrics summary
```

---

## Success Criteria

| Metric | CLIP Baseline | Target |
|--------|---------------|--------|
| Accuracy (30 names) | 13.9% | >15% |
| Accuracy (500 names) | ~3% | >4% |
| Prediction CV | 0.40 | <0.35 |

**Note:** Even small improvements are significant given the task difficulty.

---

## Sync Protocol

When done, update:
1. Create comparison table across all Track 3 experiments
2. Identify best performing approach
3. Report in `results/track3_comparison.md`:
   - Table of all experiments with key metrics
   - Which embeddings capture most name-relevant signal
   - Recommendations for further investigation
