#!/bin/bash
#
# Run all 6 linear probe experiments for CLIP/SigLIP/OpenCLIP comparison
#
# Usage:
#   ./scripts/unified/run_all_experiments.sh
#
# This will train linear probes for all 3 models on both balanced and imbalanced data.
# Results will be saved to results/linear_probe_comparison/
#
# Estimated time: ~30-60 minutes total (depending on GPU)
#

set -e  # Exit on error

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TRAIN_SCRIPT="$PROJECT_ROOT/scripts/unified/train_linear_probe.py"

BALANCED_DATA="$PROJECT_ROOT/data/siglip_balanced"
IMBALANCED_DATA="$PROJECT_ROOT/data/siglip_imbalanced"
OUTPUT_BASE="$PROJECT_ROOT/results/linear_probe_comparison"

# Models to train
MODELS=("clip" "siglip" "openclip")
DATA_MODES=("balanced" "imbalanced")

# Hyperparameters (standardized across all models)
EPOCHS=100
LR=0.01
WEIGHT_DECAY=1e-4
BATCH_SIZE=64
SEED=42

echo "========================================================================"
echo "UNIFIED LINEAR PROBE COMPARISON"
echo "========================================================================"
echo "Project root: $PROJECT_ROOT"
echo "Training script: $TRAIN_SCRIPT"
echo ""
echo "Balanced data: $BALANCED_DATA"
echo "Imbalanced data: $IMBALANCED_DATA"
echo "Output directory: $OUTPUT_BASE"
echo ""
echo "Hyperparameters:"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Weight decay: $WEIGHT_DECAY"
echo "  Batch size: $BATCH_SIZE"
echo "  Seed: $SEED"
echo "========================================================================"
echo ""

# Check data directories exist
if [ ! -d "$BALANCED_DATA" ]; then
    echo "ERROR: Balanced data directory not found: $BALANCED_DATA"
    exit 1
fi

if [ ! -d "$IMBALANCED_DATA" ]; then
    echo "ERROR: Imbalanced data directory not found: $IMBALANCED_DATA"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_BASE"

# Track successful and failed experiments
SUCCESSFUL=()
FAILED=()

# Run all experiments
TOTAL_EXPERIMENTS=$((${#MODELS[@]} * ${#DATA_MODES[@]}))
CURRENT=0

for model in "${MODELS[@]}"; do
    for mode in "${DATA_MODES[@]}"; do
        CURRENT=$((CURRENT + 1))

        # Set data directory based on mode
        if [ "$mode" == "balanced" ]; then
            DATA_DIR="$BALANCED_DATA"
        else
            DATA_DIR="$IMBALANCED_DATA"
        fi

        OUTPUT_DIR="$OUTPUT_BASE/$model/$mode"

        echo ""
        echo "========================================================================"
        echo "[$CURRENT/$TOTAL_EXPERIMENTS] Training $model on $mode data"
        echo "========================================================================"
        echo "Output: $OUTPUT_DIR"
        echo ""

        # Run training
        if python "$TRAIN_SCRIPT" \
            --model "$model" \
            --data-dir "$DATA_DIR" \
            --output-dir "$OUTPUT_DIR" \
            --epochs "$EPOCHS" \
            --lr "$LR" \
            --weight-decay "$WEIGHT_DECAY" \
            --batch-size "$BATCH_SIZE" \
            --seed "$SEED"; then
            SUCCESSFUL+=("$model/$mode")
            echo ""
            echo "SUCCESS: $model/$mode completed"
        else
            FAILED+=("$model/$mode")
            echo ""
            echo "FAILED: $model/$mode encountered an error"
        fi
    done
done

# Print summary
echo ""
echo "========================================================================"
echo "EXPERIMENT SUMMARY"
echo "========================================================================"
echo ""

if [ ${#SUCCESSFUL[@]} -gt 0 ]; then
    echo "Successful experiments (${#SUCCESSFUL[@]}/$TOTAL_EXPERIMENTS):"
    for exp in "${SUCCESSFUL[@]}"; do
        echo "  - $exp"
    done
fi

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "Failed experiments (${#FAILED[@]}/$TOTAL_EXPERIMENTS):"
    for exp in "${FAILED[@]}"; do
        echo "  - $exp"
    done
fi

echo ""
echo "Results saved to: $OUTPUT_BASE"
echo ""

# Generate comparison report if all experiments succeeded
if [ ${#FAILED[@]} -eq 0 ]; then
    echo "All experiments completed successfully!"
    echo ""
    echo "Generating comparison report..."
    if python "$PROJECT_ROOT/scripts/unified/generate_comparison.py" \
        --results-dir "$OUTPUT_BASE"; then
        echo ""
        echo "Comparison report generated:"
        echo "  - $OUTPUT_BASE/comparison/summary.json"
        echo "  - $OUTPUT_BASE/comparison/comparison_table.csv"
    else
        echo "WARNING: Failed to generate comparison report"
    fi
else
    echo ""
    echo "WARNING: Some experiments failed. Comparison report not generated."
    echo "Fix the errors and re-run, or run generate_comparison.py manually."
    exit 1
fi

echo ""
echo "========================================================================"
echo "DONE"
echo "========================================================================"
