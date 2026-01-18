#!/bin/bash
# ==============================================================================
# Sequential Training Script for Qwen 2.5 VL 7B Experiments
# ==============================================================================
# This script runs balanced and imbalanced training sequentially on a single GPU.
#
# Usage:
#   ./run_sequential_experiments.sh        # Run both experiments
#   ./run_sequential_experiments.sh balanced    # Run only balanced
#   ./run_sequential_experiments.sh imbalanced  # Run only imbalanced
# ==============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="/home/leann/face-detection"
LOG_DIR="${PROJECT_DIR}/results/qwen_7b_lora"

# Activate venv
source "${PROJECT_DIR}/venv/bin/activate"
cd "${SCRIPT_DIR}"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

train_balanced() {
    log "Starting BALANCED training..."
    python train_qwen_7b_lora.py \
        --data-dir /home/leann/face-detection/data/qwen_7b_balanced \
        --output-dir /home/leann/face-detection/results/qwen_7b_lora/balanced \
        --epochs 5 \
        --batch-size 2 \
        --gradient-accumulation-steps 16 \
        --lr 2e-5 \
        --lora-r 64 \
        --lora-alpha 16 \
        2>&1 | tee "${LOG_DIR}/balanced_training.log"
    log "BALANCED training complete!"
}

train_imbalanced() {
    log "Starting IMBALANCED training with class-balanced loss..."
    python train_qwen_7b_lora.py \
        --data-dir /home/leann/face-detection/data/qwen_7b_imbalanced \
        --output-dir /home/leann/face-detection/results/qwen_7b_lora/imbalanced \
        --epochs 5 \
        --batch-size 2 \
        --gradient-accumulation-steps 16 \
        --lr 2e-5 \
        --lora-r 64 \
        --lora-alpha 16 \
        --class-balanced-beta 0.999 \
        2>&1 | tee "${LOG_DIR}/imbalanced_training.log"
    log "IMBALANCED training complete!"
}

print_summary() {
    log "=========================================="
    log "EXPERIMENT SUMMARY"
    log "=========================================="
    echo ""
    echo "Results:"
    if [ -f "${LOG_DIR}/balanced/training_log.json" ]; then
        echo "Balanced:"
        python3 -c "import json; log=json.load(open('${LOG_DIR}/balanced/training_log.json')); print(f'  Best accuracy: {max(e[\"accuracy\"] for e in log):.4f}')"
    fi
    if [ -f "${LOG_DIR}/imbalanced/training_log.json" ]; then
        echo "Imbalanced:"
        python3 -c "import json; log=json.load(open('${LOG_DIR}/imbalanced/training_log.json')); print(f'  Best accuracy: {max(e[\"accuracy\"] for e in log):.4f}')"
    fi
    echo ""
    echo "Full logs:"
    echo "  ${LOG_DIR}/balanced_training.log"
    echo "  ${LOG_DIR}/imbalanced_training.log"
}

# Main
mkdir -p "${LOG_DIR}"

case "${1:-both}" in
    balanced)
        train_balanced
        ;;
    imbalanced)
        train_imbalanced
        ;;
    both|*)
        train_balanced
        train_imbalanced
        ;;
esac

print_summary
log "All experiments complete!"
