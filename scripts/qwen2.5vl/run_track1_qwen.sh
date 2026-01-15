#!/bin/bash
# Track 1: Qwen 2.5 VL Fine-tuning Pipeline
# Run this script step-by-step or all at once

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
DATA_DIR="$PROJECT_DIR/data"
# Canonical data: 512x512 face chips with reflect padding
INDEX_DIR="$DATA_DIR/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001"
RESULTS_DIR="$PROJECT_DIR/results/track1_qwen_vl"
SCRIPTS_DIR="$PROJECT_DIR/scripts/qwen2.5vl"

# Ensure imports like `index_utils` resolve correctly
cd "$PROJECT_DIR"

# ============================================================
# STEP 0: Install dependencies (run once)
# ============================================================
install_deps() {
    echo "Installing dependencies..."
    pip install transformers accelerate bitsandbytes peft trl
    pip install qwen-vl-utils
    echo "Dependencies installed!"
}

# ============================================================
# STEP 1: Prepare Dataset
# ============================================================

# Option A: Quick test with top 30 names (matches CLIP baseline)
prepare_dataset_small() {
    echo "Preparing dataset with top 30 names..."
    python "$SCRIPTS_DIR/prepare_qwen_dataset.py" \
        --index-dir "$INDEX_DIR" \
        --output-dir "$DATA_DIR/qwen_dataset_30names" \
        --format classification \
        --top-n-names 30 \
        --max-per-name 1000 \
        --val-ratio 0.1 \
        --seed 42
}

# Option B: Full dataset with all 500 names
prepare_dataset_full() {
    echo "Preparing full dataset with all 500 names..."
    python "$SCRIPTS_DIR/prepare_qwen_dataset.py" \
        --index-dir "$INDEX_DIR" \
        --output-dir "$DATA_DIR/qwen_dataset_full" \
        --format classification \
        --val-ratio 0.1 \
        --seed 42
}

# Option C: Using face chips (preprocessed crops)
prepare_dataset_facechips() {
    echo "Preparing dataset with face chips..."
    python "$SCRIPTS_DIR/prepare_qwen_dataset.py" \
        --index-dir "$INDEX_DIR" \
        --output-dir "$DATA_DIR/qwen_dataset_facechips" \
        --format classification \
        --top-n-names 30 \
        --max-per-name 1000 \
        --use-face-chips \
        --face-chips-dir "$DATA_DIR/face_chips_512_m0.5_reflect" \
        --val-ratio 0.1 \
        --seed 42
}

# ============================================================
# STEP 2: Training
# ============================================================

# Approach A: Classification head with 7B model (Recommended)
train_classification_7b() {
    echo "Training Qwen2.5-VL-7B with classification head..."
    python "$SCRIPTS_DIR/train_qwen_vl.py" \
        --model-id "Qwen/Qwen2.5-VL-7B-Instruct" \
        --data-dir "$DATA_DIR/qwen_dataset_30names" \
        --output-dir "$RESULTS_DIR/7b_classification" \
        --use-4bit \
        --freeze-vision \
        --lr 2e-5 \
        --batch-size 4 \
        --gradient-accumulation-steps 8 \
        --epochs 3 \
        --warmup-ratio 0.03 \
        --class-balanced-beta 0.999 \
        --seed 42
}

# Approach B: With LoRA fine-tuning
train_classification_lora() {
    echo "Training Qwen2.5-VL-7B with LoRA + classification head..."
    python "$SCRIPTS_DIR/train_qwen_vl.py" \
        --model-id "Qwen/Qwen2.5-VL-7B-Instruct" \
        --data-dir "$DATA_DIR/qwen_dataset_30names" \
        --output-dir "$RESULTS_DIR/7b_classification_lora" \
        --use-4bit \
        --use-lora \
        --lora-r 64 \
        --lora-alpha 16 \
        --freeze-vision \
        --lr 2e-5 \
        --batch-size 4 \
        --gradient-accumulation-steps 8 \
        --epochs 3 \
        --warmup-ratio 0.03 \
        --class-balanced-beta 0.999 \
        --seed 42
}

# Approach C: Smaller 3B model for faster iteration
train_classification_3b() {
    echo "Training Qwen2.5-VL-3B with classification head..."
    python "$SCRIPTS_DIR/train_qwen_vl.py" \
        --model-id "Qwen/Qwen2.5-VL-3B-Instruct" \
        --data-dir "$DATA_DIR/qwen_dataset_30names" \
        --output-dir "$RESULTS_DIR/3b_classification" \
        --use-4bit \
        --freeze-vision \
        --lr 2e-5 \
        --batch-size 8 \
        --gradient-accumulation-steps 4 \
        --epochs 3 \
        --warmup-ratio 0.03 \
        --class-balanced-beta 0.999 \
        --seed 42
}

# ============================================================
# STEP 3: Evaluation
# ============================================================

evaluate_model() {
    CHECKPOINT_DIR=$1
    OUTPUT_DIR=$2
    DATA_DIR_ARG=$3

    echo "Evaluating model from $CHECKPOINT_DIR..."
    python "$SCRIPTS_DIR/evaluate_qwen_vl.py" infer \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --data-dir "$DATA_DIR_ARG" \
        --output-dir "$OUTPUT_DIR" \
        --batch-size 8
}

# ============================================================
# FULL PIPELINES
# ============================================================

# Quick test pipeline (30 names, 3B model)
run_quick_test() {
    echo "=== Running Quick Test Pipeline ==="
    prepare_dataset_small
    train_classification_3b
    evaluate_model \
        "$RESULTS_DIR/3b_classification/checkpoint_epoch_3" \
        "$RESULTS_DIR/3b_classification" \
        "$DATA_DIR/qwen_dataset_30names"
    echo "=== Quick Test Complete ==="
    echo "Results: $RESULTS_DIR/3b_classification/summary.md"
}

# Full pipeline (30 names, 7B model)
run_full_pipeline() {
    echo "=== Running Full Pipeline ==="
    prepare_dataset_small
    train_classification_7b
    evaluate_model \
        "$RESULTS_DIR/7b_classification/checkpoint_epoch_3" \
        "$RESULTS_DIR/7b_classification" \
        "$DATA_DIR/qwen_dataset_30names"
    echo "=== Pipeline Complete ==="
    echo "Results: $RESULTS_DIR/7b_classification/summary.md"
}

# ============================================================
# USAGE
# ============================================================
print_usage() {
    echo "Usage: $0 <command>"
    echo ""
    echo "Commands:"
    echo "  install_deps           Install required Python packages"
    echo ""
    echo "  prepare_dataset_small  Prepare dataset with top 30 names"
    echo "  prepare_dataset_full   Prepare dataset with all 500 names"
    echo "  prepare_dataset_facechips  Prepare using face chip images"
    echo ""
    echo "  train_classification_7b     Train 7B model with classification head"
    echo "  train_classification_lora   Train 7B model with LoRA"
    echo "  train_classification_3b     Train 3B model (faster)"
    echo ""
    echo "  run_quick_test         Run quick test pipeline (3B model)"
    echo "  run_full_pipeline      Run full pipeline (7B model)"
    echo ""
    echo "Examples:"
    echo "  $0 install_deps"
    echo "  $0 prepare_dataset_small"
    echo "  $0 train_classification_7b"
    echo "  $0 run_quick_test"
}

# Run command
if [ -z "$1" ]; then
    print_usage
else
    "$1"
fi
