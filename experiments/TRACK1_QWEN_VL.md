# Track 1: Qwen 2.5 VL Fine-tuning for Face-Name Association

## Project Context

### What We're Trying to Do
Train a model to learn "name vibes" - the subtle statistical correlations between facial features and first names. Given a face image, predict the most likely first name.

### Why This Matters
- Psychology research shows humans can match faces to names at ~60-70% for binary choices
- CLIP embeddings contain weak but real signal (~14% accuracy on 30 names vs 3.3% random)
- We want to push beyond CLIP's ceiling by trying different architectures

### What We've Tried (Phase 1-3 Summary)
| Approach | Result | Problem |
|----------|--------|---------|
| CLIP linear probe (30 names) | 13.9% acc | Prediction skew to William/Nick/Lisa |
| Cosine classifier | 14.5% acc | Skew persisted (not weight-norm artifact) |
| Focal loss | 13.4% acc | Made skew worse |
| ArcFace embeddings | ~12% acc | More balanced but lower overall |
| Face chips (512x512) | 12.4% acc | Aspect ratio wasn't main confound |

**Key insight:** The skew is structural in CLIP's embedding space, not an artifact.

---

## Data Available

### Location
```
/home/leann/face-detection/data/index_files/          # Original images
/home/leann/face-detection/data/index_files_facechips512_m0.5_reflect/  # Face chips
```

### Statistics
- **Total images:** 434,528 (good quality, single face detected)
- **Total names:** 500
- **Distribution:** Highly unbalanced
  - Top: alex (5,960), david (5,674), michael (4,801)
  - Bottom: ~200 images per name

### Index File Format
Each `index_{name}.json` contains:
```json
{
  "good": ["/path/to/image1.jpg", "/path/to/image2.jpg", ...],
  "bad": []
}
```

---

## Why Qwen 2.5 VL

### Hypothesis
Vision-Language Models can leverage:
1. **Language priors** about names (cultural associations, era, demographics)
2. **Multimodal reasoning** - not just visual features but semantic understanding
3. **In-context learning** - potentially learn from examples in the prompt

### Model Options
| Model | Params | VRAM Required | Notes |
|-------|--------|---------------|-------|
| Qwen2.5-VL-3B | 3B | ~8GB (4-bit) | Fast iteration |
| Qwen2.5-VL-7B | 7B | ~18GB (4-bit) | Best balance |
| Qwen2.5-VL-72B | 72B | Multi-GPU | If 7B fails |

### Recommended: Qwen2.5-VL-7B with LoRA + 4-bit quantization

---

## Implementation Plan

### Step 1: Environment Setup
```bash
pip install transformers accelerate bitsandbytes peft trl
pip install qwen-vl-utils  # For image processing
```

### Step 2: Data Preparation
Create a dataset in conversation format:
```python
{
    "messages": [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/path/to/face.jpg"},
                {"type": "text", "text": "What first name does this person look like? Choose from: [list of names]"}
            ]
        },
        {
            "role": "assistant",
            "content": "William"
        }
    ]
}
```

### Step 3: Training Configuration
```python
# LoRA config (keep vision encoder frozen)
lora_config = LoraConfig(
    r=64,
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# Training args
training_args = TrainingArguments(
    output_dir="./results/qwen_vl_finetune",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    learning_rate=2e-5,
    warmup_ratio=0.03,
    lr_scheduler_type="cosine",
    bf16=True,
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="epoch",
)
```

### Step 4: Training Approaches to Try

#### Approach A: Classification Head (Recommended First)
Use the [Qwen-VL-Series-Finetune](https://github.com/2U1/Qwen-VL-Series-Finetune) repo's classification script:
```bash
python train_classification.py \
    --model_id Qwen/Qwen2.5-VL-7B-Instruct \
    --num_labels 500 \
    --class_balanced_beta 0.999 \
    --use_lora \
    --lora_r 64
```

#### Approach B: Generative (Name as Text Output)
Frame as text generation - model outputs the name directly. More flexible but harder to evaluate.

#### Approach C: Multiple Choice
Give model 4-5 name options per image, ask it to choose. Easier task, good for sanity check.

### Step 5: Evaluation
Save results in standardized format for comparison:
```
results/track1_qwen_vl/
├── config.json           # Training hyperparameters
├── predictions.npy       # (N,) predicted class indices
├── true_labels.npy       # (N,) ground truth indices
├── names.json            # List of name strings
├── results.csv           # Per-name precision/recall/F1
├── val_embeddings.npy    # Optional: extracted embeddings
└── training_log.json     # Loss curves
```

---

## Key References

1. [Qwen-VL-Series-Finetune GitHub](https://github.com/2U1/Qwen-VL-Series-Finetune) - Has classification training script
2. [Roboflow Tutorial](https://blog.roboflow.com/fine-tune-qwen-2-5/) - LoRA + 4-bit setup
3. [HuggingFace Cookbook](https://huggingface.co/learn/cookbook/en/fine_tuning_vlm_trl) - TRL-based fine-tuning
4. [Skywork Tutorial](https://skywork.ai/blog/llm/fine-tune-qwen2-5-vl-32b-in-3-days-complete-hands-on-tutorial/) - Keep vision encoder frozen

---

## Success Criteria

| Metric | Baseline (CLIP) | Target |
|--------|-----------------|--------|
| Overall accuracy (500 names) | ~3% | >5% |
| Overall accuracy (30 names) | 13.9% | >18% |
| Prediction CV | 0.40 | <0.35 |
| Top-1 name recall | 51% (William) | <40% (more balanced) |

---

## Sync Protocol

When done, update:
1. `results/track1_qwen_vl/` with all outputs
2. Create `results/track1_qwen_vl/summary.md` with findings
3. Key metrics to report:
   - Overall accuracy (all 500 names)
   - Overall accuracy (top 30 names for comparison)
   - Prediction CV (coefficient of variation)
   - Top 5 / Bottom 5 names by F1
   - Training time and GPU memory used
