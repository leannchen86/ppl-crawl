#!/bin/bash
# ==============================================================================
# Qwen 2.5 VL 7B Fine-tuning Experiments
# ==============================================================================
# This script runs the complete pipeline:
# 1. Create held-out validation split (fixed, used by both experiments)
# 2. Create balanced training dataset (equal samples per name)
# 3. Create imbalanced training dataset (natural distribution)
# 4. Train with balanced data
# 5. Train with imbalanced data + class-balanced loss
#
# Usage:
#   ./run_7b_lora_experiments.sh           # Run everything
#   ./run_7b_lora_experiments.sh --prep    # Only data preparation
#   ./run_7b_lora_experiments.sh --train   # Only training (assumes data exists)
# ==============================================================================

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/home/leann/face-detection"
VENV_PATH="${PROJECT_DIR}/venv/bin/activate"

# Data paths
INDEX_DIR="${PROJECT_DIR}/data/index_files"
HOLDOUT_DIR="${PROJECT_DIR}/data/qwen_7b_holdout"
BALANCED_DATA="${PROJECT_DIR}/data/qwen_7b_balanced"
IMBALANCED_DATA="${PROJECT_DIR}/data/qwen_7b_imbalanced"

# Output paths
RESULTS_DIR="${PROJECT_DIR}/results/qwen_7b_lora"
BALANCED_RESULTS="${RESULTS_DIR}/balanced"
IMBALANCED_RESULTS="${RESULTS_DIR}/imbalanced"

# Training parameters
NUM_NAMES=30
MAX_PER_NAME_BALANCED=900  # 90% of 1000 max (10% held out)
EPOCHS=5
BATCH_SIZE=2
GRAD_ACCUM=16
LEARNING_RATE=2e-5
LORA_R=64
LORA_ALPHA=16

# ==============================================================================
# Helper functions
# ==============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

activate_venv() {
    if [ -f "${VENV_PATH}" ]; then
        source "${VENV_PATH}"
        log "Activated virtual environment"
    else
        log "Warning: Virtual environment not found at ${VENV_PATH}"
    fi
}

# ==============================================================================
# Data Preparation
# ==============================================================================

prepare_data() {
    log "=========================================="
    log "STEP 1: Creating held-out validation split"
    log "=========================================="

    cd "${SCRIPT_DIR}"

    python prepare_holdout_dataset.py \
        --create-holdout \
        --index-dir "${INDEX_DIR}" \
        --manifest-dir "${HOLDOUT_DIR}" \
        --top-n-names ${NUM_NAMES} \
        --holdout-ratio 0.1 \
        --seed 42 \
        --image-source chips

    log "=========================================="
    log "STEP 2: Creating BALANCED training dataset"
    log "=========================================="

    python prepare_holdout_dataset.py \
        --mode balanced \
        --manifest-dir "${HOLDOUT_DIR}" \
        --output-dir "${BALANCED_DATA}" \
        --max-per-name ${MAX_PER_NAME_BALANCED} \
        --seed 42

    log "=========================================="
    log "STEP 3: Creating IMBALANCED training dataset"
    log "=========================================="

    python prepare_holdout_dataset.py \
        --mode imbalanced \
        --manifest-dir "${HOLDOUT_DIR}" \
        --output-dir "${IMBALANCED_DATA}" \
        --seed 42

    log "Data preparation complete!"
    log "  Holdout manifest: ${HOLDOUT_DIR}"
    log "  Balanced data: ${BALANCED_DATA}"
    log "  Imbalanced data: ${IMBALANCED_DATA}"
}

# ==============================================================================
# Training
# ==============================================================================

train_balanced() {
    log "=========================================="
    log "STEP 4: Training with BALANCED data"
    log "=========================================="

    cd "${SCRIPT_DIR}"

    python train_qwen_7b_lora.py \
        --data-dir "${BALANCED_DATA}" \
        --output-dir "${BALANCED_RESULTS}" \
        --model-id "Qwen/Qwen2.5-VL-7B-Instruct" \
        --use-4bit \
        --use-lora \
        --freeze-vision \
        --lora-r ${LORA_R} \
        --lora-alpha ${LORA_ALPHA} \
        --lr ${LEARNING_RATE} \
        --batch-size ${BATCH_SIZE} \
        --gradient-accumulation-steps ${GRAD_ACCUM} \
        --epochs ${EPOCHS} \
        --seed 42

    log "Balanced training complete!"
    log "  Results: ${BALANCED_RESULTS}"
}

train_imbalanced() {
    log "=========================================="
    log "STEP 5: Training with IMBALANCED data + class-balanced loss"
    log "=========================================="

    cd "${SCRIPT_DIR}"

    python train_qwen_7b_lora.py \
        --data-dir "${IMBALANCED_DATA}" \
        --output-dir "${IMBALANCED_RESULTS}" \
        --model-id "Qwen/Qwen2.5-VL-7B-Instruct" \
        --use-4bit \
        --use-lora \
        --freeze-vision \
        --lora-r ${LORA_R} \
        --lora-alpha ${LORA_ALPHA} \
        --lr ${LEARNING_RATE} \
        --batch-size ${BATCH_SIZE} \
        --gradient-accumulation-steps ${GRAD_ACCUM} \
        --epochs ${EPOCHS} \
        --class-balanced-beta 0.999 \
        --seed 42

    log "Imbalanced training complete!"
    log "  Results: ${IMBALANCED_RESULTS}"
}

# ==============================================================================
# Summary
# ==============================================================================

print_summary() {
    log "=========================================="
    log "EXPERIMENT SUMMARY"
    log "=========================================="

    echo ""
    echo "Data Paths:"
    echo "  Holdout manifest: ${HOLDOUT_DIR}"
    echo "  Balanced data:    ${BALANCED_DATA}"
    echo "  Imbalanced data:  ${IMBALANCED_DATA}"
    echo ""
    echo "Results Paths:"
    echo "  Balanced model:   ${BALANCED_RESULTS}"
    echo "  Imbalanced model: ${IMBALANCED_RESULTS}"
    echo ""
    echo "To compare results:"
    echo "  cat ${BALANCED_RESULTS}/training_log.json"
    echo "  cat ${IMBALANCED_RESULTS}/training_log.json"
    echo ""
}

# ==============================================================================
# Main
# ==============================================================================

main() {
    log "Starting Qwen 2.5 VL 7B LoRA experiments"

    activate_venv

    case "${1:-all}" in
        --prep)
            prepare_data
            ;;
        --train)
            train_balanced
            train_imbalanced
            ;;
        --train-balanced)
            train_balanced
            ;;
        --train-imbalanced)
            train_imbalanced
            ;;
        all|*)
            prepare_data
            train_balanced
            train_imbalanced
            ;;
    esac

    print_summary
    log "All experiments complete!"
}

main "$@"
