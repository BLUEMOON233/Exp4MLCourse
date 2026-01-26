#!/bin/bash

# Define the project root and current directory
# This script is expected to be located in code/exp2/

# Navigate to the exp2 directory if not already there
# cd "$(dirname "$0")" || exit

# Training Configuration
# Adjust 'epochs' and 'batch_size' based on your hardware capabilities
# Current setting: 50 epochs, batch size 48 (optimized for RTX 3090, whole image 224x224)
EPOCHS=50
BATCH_SIZE=32
NUM_WORKERS=8
LR=2e-4

# Checkpoint Directory
# Following the user's request to match exp1 naming convention and location
CKPT_DIR="../../checkpoints/exp2_MambaFeatureEnhancer_CUBC"

# Create checkpoint directory if it doesn't exist
mkdir -p $CKPT_DIR

# Execute the training script
# We filter warnings to keep output clean, but you can remove PYTHONWARNINGS if needed
export PYTHONWARNINGS="ignore"

echo "Starting Feature Enhancer Training..."
echo "Dataset Root: ../../datasets/CUB-C" # Assuming this is the intended value for DATASET_ROOT_CUB
echo "Checkpoints: $CKPT_DIR"
echo "Batch Size: $BATCH_SIZE, Epochs: $EPOCHS"

# Run Training Script
python train_feature_enhancer.py \
    --dataset_root "../../datasets/CUB-C" \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --ckpt_dir "$CKPT_DIR"

echo "检查点和损失曲线 (loss_curve.png) 应位于 '$CKPT_DIR'"
