#!/bin/bash
# OpenCLIP Linear Probe Experiments
# Runs both balanced and imbalanced training with ViT-B-32 (LAION-2B)

set -e

BASE_DIR="/home/leann/face-detection"
SCRIPT_DIR="$BASE_DIR/scripts/openclip"
LOG_DIR="$BASE_DIR/logs"

mkdir -p "$LOG_DIR"
mkdir -p "$BASE_DIR/results/openclip/balanced"
mkdir -p "$BASE_DIR/results/openclip/imbalanced"

# Use GPU 0 (A100)
export CUDA_VISIBLE_DEVICES=0

echo "========================================================================"
echo "OpenCLIP LINEAR PROBE EXPERIMENTS"
echo "========================================================================"
echo "Model: ViT-B-32 (laion2b_s34b_b79k)"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

# Train BALANCED
echo "[1/2] Training BALANCED dataset..."
python3 "$SCRIPT_DIR/train_openclip_linear_probe.py" \
    --data-dir "$BASE_DIR/data/siglip_balanced" \
    --output-dir "$BASE_DIR/results/openclip/balanced" \
    --model-name "ViT-B-32" \
    --pretrained "laion2b_s34b_b79k" \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 64 \
    --seed 42
touch "$LOG_DIR/openclip_balanced_done.flag"

echo ""
echo "[2/2] Training IMBALANCED dataset..."
python3 "$SCRIPT_DIR/train_openclip_linear_probe.py" \
    --data-dir "$BASE_DIR/data/siglip_imbalanced" \
    --output-dir "$BASE_DIR/results/openclip/imbalanced" \
    --model-name "ViT-B-32" \
    --pretrained "laion2b_s34b_b79k" \
    --epochs 100 \
    --lr 0.01 \
    --batch-size 64 \
    --seed 42
touch "$LOG_DIR/openclip_imbalanced_done.flag"

echo ""
echo "========================================================================"
echo "ALL OPENCLIP EXPERIMENTS COMPLETE!"
echo "========================================================================"
