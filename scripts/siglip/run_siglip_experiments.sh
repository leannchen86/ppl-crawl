#!/bin/bash
# =============================================================================
# SigLIP Linear Probe Experiments
# =============================================================================
# This script runs the full SigLIP experiment pipeline:
# 1. Install required packages (if missing)
# 2. Create held-out validation split (10% of data, never used for training)
# 3. Create balanced training dataset (equal samples per class)
# 4. Create imbalanced training dataset (natural distribution)
# 5. Train SigLIP linear probe on balanced data
# 6. Train SigLIP linear probe on imbalanced data
#
# Key design:
# - Both experiments use the SAME held-out validation set for fair comparison
# - Linear probe only (frozen SigLIP backbone) - proven better than fine-tuning
# - Results saved with full reproducibility information
#
# Usage:
#   cd /home/leann/face-detection
#   bash scripts/siglip/run_siglip_experiments.sh
#
# Expected runtime: ~30-60 minutes depending on GPU
# =============================================================================

set -e  # Exit on error

# Configuration
PROJECT_ROOT="/home/leann/face-detection"
VENV_PATH="${PROJECT_ROOT}/venv"
INDEX_DIR="${PROJECT_ROOT}/data/index_files_facechips512_filtered_score0.9_bbox32_areafrac0.001"
MANIFEST_DIR="${PROJECT_ROOT}/data/siglip_holdout"
RESULTS_DIR="${PROJECT_ROOT}/results/siglip"

# Data settings
NUM_CLASSES=30
HOLDOUT_RATIO=0.1  # 10% held out for validation
MIN_SAMPLES=200    # Minimum samples per name
MAX_PER_NAME=500   # For balanced mode
SEED=42

# Training settings
EPOCHS=100
LR=0.01
BATCH_SIZE=64

# SigLIP model - using the base variant
# Available via open_clip: ViT-B-16-SigLIP (webli pretrained)
SIGLIP_MODEL="ViT-B-16-SigLIP"
SIGLIP_PRETRAINED="webli"

echo "============================================================================="
echo "SIGLIP LINEAR PROBE EXPERIMENTS"
echo "============================================================================="
echo "Project root: ${PROJECT_ROOT}"
echo "Index directory: ${INDEX_DIR}"
echo "Manifest directory: ${MANIFEST_DIR}"
echo "Results directory: ${RESULTS_DIR}"
echo "Model: ${SIGLIP_MODEL} (pretrained=${SIGLIP_PRETRAINED})"
echo "Number of classes: ${NUM_CLASSES}"
echo "Holdout ratio: ${HOLDOUT_RATIO}"
echo "Seed: ${SEED}"
echo "============================================================================="

# Change to project root
cd "${PROJECT_ROOT}"

# Activate virtual environment
echo ""
echo "[Step 0] Activating virtual environment..."
source "${VENV_PATH}/bin/activate"

# Check and install required packages
echo ""
echo "[Step 1] Checking required packages..."
PACKAGES_NEEDED=""

python -c "import open_clip" 2>/dev/null || PACKAGES_NEEDED="${PACKAGES_NEEDED} open_clip_torch"
python -c "import pandas" 2>/dev/null || PACKAGES_NEEDED="${PACKAGES_NEEDED} pandas"
python -c "import tqdm" 2>/dev/null || PACKAGES_NEEDED="${PACKAGES_NEEDED} tqdm"
python -c "import numpy" 2>/dev/null || PACKAGES_NEEDED="${PACKAGES_NEEDED} numpy"

if [ -n "${PACKAGES_NEEDED}" ]; then
    echo "Installing missing packages:${PACKAGES_NEEDED}"
    pip install ${PACKAGES_NEEDED}
else
    echo "All required packages already installed."
fi

# Verify open_clip has SigLIP
echo ""
echo "[Step 1.1] Verifying SigLIP model availability..."
python -c "
import open_clip
models = [m for m in open_clip.list_pretrained() if 'siglip' in m[0].lower()]
print(f'Available SigLIP models: {len(models)}')
for m in models[:5]:
    print(f'  - {m[0]} ({m[1]})')
if len(models) > 5:
    print(f'  ... and {len(models)-5} more')
"

# Create directory structure
echo ""
echo "[Step 2] Creating directory structure..."
mkdir -p "${MANIFEST_DIR}"
mkdir -p "${RESULTS_DIR}/balanced"
mkdir -p "${RESULTS_DIR}/imbalanced"
mkdir -p "${PROJECT_ROOT}/data/siglip_balanced"
mkdir -p "${PROJECT_ROOT}/data/siglip_imbalanced"

# Create held-out split (run once, reuse for both experiments)
echo ""
echo "[Step 3] Creating held-out validation split..."
if [ -f "${MANIFEST_DIR}/holdout_manifest.json" ]; then
    echo "Holdout manifest already exists, skipping creation."
    echo "Delete ${MANIFEST_DIR}/holdout_manifest.json to recreate."
else
    python scripts/siglip/prepare_siglip_holdout.py \
        --create-holdout \
        --index-dir "${INDEX_DIR}" \
        --manifest-dir "${MANIFEST_DIR}" \
        --holdout-ratio ${HOLDOUT_RATIO} \
        --min-samples ${MIN_SAMPLES} \
        --top-n-names ${NUM_CLASSES} \
        --seed ${SEED}
fi

# Create balanced training dataset
echo ""
echo "[Step 4] Creating BALANCED training dataset..."
python scripts/siglip/prepare_siglip_holdout.py \
    --mode balanced \
    --index-dir "${INDEX_DIR}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --output-dir "${PROJECT_ROOT}/data/siglip_balanced" \
    --max-per-name ${MAX_PER_NAME} \
    --seed ${SEED}

# Create imbalanced training dataset
echo ""
echo "[Step 5] Creating IMBALANCED training dataset..."
python scripts/siglip/prepare_siglip_holdout.py \
    --mode imbalanced \
    --index-dir "${INDEX_DIR}" \
    --manifest-dir "${MANIFEST_DIR}" \
    --output-dir "${PROJECT_ROOT}/data/siglip_imbalanced" \
    --seed ${SEED}

# Train SigLIP linear probe - BALANCED
echo ""
echo "============================================================================="
echo "[Step 6] TRAINING SIGLIP LINEAR PROBE - BALANCED DATA"
echo "============================================================================="
python scripts/siglip/train_siglip_linear_probe.py \
    --data-dir "${PROJECT_ROOT}/data/siglip_balanced" \
    --output-dir "${RESULTS_DIR}/balanced" \
    --model-name "${SIGLIP_MODEL}" \
    --pretrained "${SIGLIP_PRETRAINED}" \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED}

# Train SigLIP linear probe - IMBALANCED
echo ""
echo "============================================================================="
echo "[Step 7] TRAINING SIGLIP LINEAR PROBE - IMBALANCED DATA"
echo "============================================================================="
python scripts/siglip/train_siglip_linear_probe.py \
    --data-dir "${PROJECT_ROOT}/data/siglip_imbalanced" \
    --output-dir "${RESULTS_DIR}/imbalanced" \
    --model-name "${SIGLIP_MODEL}" \
    --pretrained "${SIGLIP_PRETRAINED}" \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch-size ${BATCH_SIZE} \
    --seed ${SEED}

# Summary comparison
echo ""
echo "============================================================================="
echo "EXPERIMENT SUMMARY"
echo "============================================================================="

# Extract results from both experiments
echo ""
echo "Balanced results:"
cat "${RESULTS_DIR}/balanced/experiment_config.json" | python -c "
import sys, json
data = json.load(sys.stdin)
results = data['results']
print(f\"  Best validation accuracy: {100*results['best_val_acc']:.2f}%\")
print(f\"  Random baseline: {100*results['random_baseline']:.2f}%\")
print(f\"  Improvement over random: +{100*results['improvement_over_random']:.2f}%\")
"

echo ""
echo "Imbalanced results:"
cat "${RESULTS_DIR}/imbalanced/experiment_config.json" | python -c "
import sys, json
data = json.load(sys.stdin)
results = data['results']
print(f\"  Best validation accuracy: {100*results['best_val_acc']:.2f}%\")
print(f\"  Random baseline: {100*results['random_baseline']:.2f}%\")
print(f\"  Improvement over random: +{100*results['improvement_over_random']:.2f}%\")
"

echo ""
echo "============================================================================="
echo "KEY PATHS FOR REPRODUCIBILITY"
echo "============================================================================="
echo ""
echo "Held-out validation manifest (NEVER modify this):"
echo "  ${MANIFEST_DIR}/holdout_manifest.json"
echo ""
echo "Training data (balanced):"
echo "  ${PROJECT_ROOT}/data/siglip_balanced/train.json"
echo ""
echo "Training data (imbalanced):"
echo "  ${PROJECT_ROOT}/data/siglip_imbalanced/train.json"
echo ""
echo "Validation data (same for both experiments, from holdout):"
echo "  ${PROJECT_ROOT}/data/siglip_balanced/val.json"
echo "  ${PROJECT_ROOT}/data/siglip_imbalanced/val.json"
echo ""
echo "Results:"
echo "  ${RESULTS_DIR}/balanced/"
echo "  ${RESULTS_DIR}/imbalanced/"
echo ""
echo "============================================================================="
echo "EXPERIMENTS COMPLETE!"
echo "============================================================================="
