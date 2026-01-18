#!/bin/bash
# =============================================================================
# SigLIP Linear Probe - Complete Pipeline
# =============================================================================
# Run this script to execute both balanced and imbalanced experiments
#
# Usage:
#   cd /home/leann/face-detection
#   bash scripts/siglip/run_all.sh
# =============================================================================

set -e
export CUDA_VISIBLE_DEVICES=0  # Use A100 GPU

cd /home/leann/face-detection
source venv/bin/activate

echo "============================================="
echo "Step 1: Installing dependencies..."
echo "============================================="
pip install open_clip_torch pandas tqdm numpy -q

echo ""
echo "============================================="
echo "Step 2: Verifying SigLIP model availability"
echo "============================================="
python3 -c "
import open_clip
models = [m for m in open_clip.list_pretrained() if 'siglip' in m[0].lower()]
print(f'Found {len(models)} SigLIP models')
for m in models[:3]:
    print(f'  {m}')
"

echo ""
echo "============================================="
echo "Step 3: Training BALANCED (separate run)"
echo "============================================="
python3 scripts/siglip/train_siglip_linear_probe.py \
    --data-dir /home/leann/face-detection/data/siglip_balanced \
    --output-dir /home/leann/face-detection/results/siglip/balanced \
    --model-name "ViT-B-16-SigLIP" \
    --pretrained "webli" \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 64 \
    --seed 42

echo ""
echo "============================================="
echo "Step 4: Training IMBALANCED (separate run)"
echo "============================================="
python3 scripts/siglip/train_siglip_linear_probe.py \
    --data-dir /home/leann/face-detection/data/siglip_imbalanced \
    --output-dir /home/leann/face-detection/results/siglip/imbalanced \
    --model-name "ViT-B-16-SigLIP" \
    --pretrained "webli" \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 64 \
    --seed 42

echo ""
echo "============================================="
echo "COMPLETE! Results in:"
echo "  Balanced:   results/siglip/balanced/"
echo "  Imbalanced: results/siglip/imbalanced/"
echo "============================================="
