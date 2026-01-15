#!/bin/bash
# Track 2: ViT from Scratch - Experiment Runner
#
# Runs all experiments from TRACK2_VIT_SCRATCH.md:
#   2.1: Baseline ViT-Base (Full Data)
#   2.2: With Mixup/CutMix
#   2.3: Label Smoothing Ablation
#   2.4: Subset Experiments
#
# Usage:
#   ./run_track2_experiments.sh all        # Run all experiments
#   ./run_track2_experiments.sh baseline   # Run only baseline
#   ./run_track2_experiments.sh mixup      # Run only mixup
#   ./run_track2_experiments.sh label      # Run label smoothing ablation
#   ./run_track2_experiments.sh subset     # Run subset experiments

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
RESULTS_BASE="$PROJECT_DIR/results/track2_vit_scratch"
TRAIN_SCRIPT="$SCRIPT_DIR/train_vit_scratch.py"

# Ensure relative paths (and imports, if any) resolve from project root
cd "$PROJECT_DIR"

# Hyperparameters (adjust based on GPU memory)
BATCH_SIZE=128  # Use 256 if you have 40GB+ VRAM
EPOCHS=100
NUM_WORKERS=8

# Create results directory
mkdir -p "$RESULTS_BASE"

# Logging
LOG_FILE="$RESULTS_BASE/experiments.log"
echo "=" | tee -a "$LOG_FILE"
echo "Track 2 Experiments - $(date)" | tee -a "$LOG_FILE"
echo "=" | tee -a "$LOG_FILE"

run_baseline() {
    echo "" | tee -a "$LOG_FILE"
    echo "[Experiment 2.1] Baseline ViT-Base (Full Data)" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    python "$TRAIN_SCRIPT" \
        --experiment baseline \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$RESULTS_BASE/exp2.1_baseline" \
        2>&1 | tee -a "$LOG_FILE"
}

run_mixup() {
    echo "" | tee -a "$LOG_FILE"
    echo "[Experiment 2.2] With Mixup/CutMix" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    python "$TRAIN_SCRIPT" \
        --experiment mixup \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$RESULTS_BASE/exp2.2_mixup" \
        2>&1 | tee -a "$LOG_FILE"
}

run_label_smoothing() {
    echo "" | tee -a "$LOG_FILE"
    echo "[Experiment 2.3] Label Smoothing Ablation" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    # Test different label smoothing values
    for LS in 0.0 0.1 0.2; do
        echo "" | tee -a "$LOG_FILE"
        echo "Running with label_smoothing=$LS" | tee -a "$LOG_FILE"

        python "$TRAIN_SCRIPT" \
            --experiment label_smooth \
            --label-smoothing "$LS" \
            --epochs "$EPOCHS" \
            --batch-size "$BATCH_SIZE" \
            --num-workers "$NUM_WORKERS" \
            --output-dir "$RESULTS_BASE/exp2.3_ls${LS}" \
            2>&1 | tee -a "$LOG_FILE"
    done
}

run_subset() {
    echo "" | tee -a "$LOG_FILE"
    echo "[Experiment 2.4] Subset Experiments" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    # Top-30 names (for comparison with CLIP baseline)
    echo "" | tee -a "$LOG_FILE"
    echo "Running with 30 names (CLIP comparison)" | tee -a "$LOG_FILE"

    python "$TRAIN_SCRIPT" \
        --experiment subset \
        --num-names 30 \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$RESULTS_BASE/exp2.4_subset30" \
        2>&1 | tee -a "$LOG_FILE"

    # Top-100 names (faster iteration with more data)
    echo "" | tee -a "$LOG_FILE"
    echo "Running with 100 names" | tee -a "$LOG_FILE"

    python "$TRAIN_SCRIPT" \
        --experiment subset \
        --num-names 100 \
        --epochs "$EPOCHS" \
        --batch-size "$BATCH_SIZE" \
        --num-workers "$NUM_WORKERS" \
        --output-dir "$RESULTS_BASE/exp2.4_subset100" \
        2>&1 | tee -a "$LOG_FILE"
}

# Quick test run (for debugging)
run_quick_test() {
    echo "" | tee -a "$LOG_FILE"
    echo "[Quick Test] Sanity check with small data" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    python "$TRAIN_SCRIPT" \
        --num-names 10 \
        --epochs 3 \
        --batch-size 32 \
        --num-workers 4 \
        --output-dir "$RESULTS_BASE/quick_test" \
        2>&1 | tee -a "$LOG_FILE"
}

print_summary() {
    echo "" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"
    echo "EXPERIMENT SUMMARY" | tee -a "$LOG_FILE"
    echo "=============================================" | tee -a "$LOG_FILE"

    echo ""
    echo "Results directories:"
    ls -la "$RESULTS_BASE"/ 2>/dev/null || echo "No results yet"

    echo ""
    echo "Best validation accuracies:"
    for dir in "$RESULTS_BASE"/exp*; do
        if [ -f "$dir/config.json" ]; then
            name=$(basename "$dir")
            acc=$(python -c "import json; print(json.load(open('$dir/config.json')).get('best_val_acc', 'N/A'))" 2>/dev/null || echo "N/A")
            echo "  $name: $acc%"
        fi
    done
}

# Main dispatcher
case "${1:-all}" in
    baseline)
        run_baseline
        ;;
    mixup)
        run_mixup
        ;;
    label)
        run_label_smoothing
        ;;
    subset)
        run_subset
        ;;
    quick|test)
        run_quick_test
        ;;
    summary)
        print_summary
        ;;
    all)
        echo "Running all experiments (this will take a while)..."
        run_baseline
        run_mixup
        run_label_smoothing
        run_subset
        print_summary
        ;;
    *)
        echo "Usage: $0 {all|baseline|mixup|label|subset|quick|summary}"
        echo ""
        echo "Experiments:"
        echo "  baseline  - Exp 2.1: Baseline ViT-Base (full data)"
        echo "  mixup     - Exp 2.2: With Mixup/CutMix augmentation"
        echo "  label     - Exp 2.3: Label smoothing ablation (0.0, 0.1, 0.2)"
        echo "  subset    - Exp 2.4: Subset experiments (30, 100 names)"
        echo "  quick     - Quick sanity check (10 names, 3 epochs)"
        echo "  summary   - Print summary of completed experiments"
        echo "  all       - Run all experiments sequentially"
        exit 1
        ;;
esac

echo ""
echo "Done! Results saved to: $RESULTS_BASE"
