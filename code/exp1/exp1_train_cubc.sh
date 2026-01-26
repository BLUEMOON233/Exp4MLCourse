#!/bin/bash

# Define the project root and current directory
# This script is expected to be located in /Users/wardenliu/Develop/MLC/code/exp1/

# Navigate to the AdaIR project directory
cd AdaIR || exit

# Training Configuration
# Adjust 'epochs' and 'batch_size' based on your hardware capabilities
# Adjust 'epochs' and 'batch_size' based on your hardware capabilities
# Current setting: 50 epochs (adjusted for smaller dataset), batch size 6
# EPOCHS=30
EPOCHS=10
BATCH_SIZE=6
NUM_WORKERS=4
LR=1e-6
PATCH_SIZE=128

# Execute the training script
# We filter warnings to keep output clean, but you can remove PYTHONWARNINGS if needed
export PYTHONWARNINGS="ignore"

echo "步骤 1: 开始在 CUB-C 数据集上训练 AdaIR..."
echo "配置: 轮次(Epochs)=$EPOCHS, 批大小(BatchSize)=$BATCH_SIZE"

python train_cub.py \
    --epochs $EPOCHS \
    --batch_size $BATCH_SIZE \
    --num_workers $NUM_WORKERS \
    --lr $LR \
    --patch_size $PATCH_SIZE \
    --wblogger "" \
    --ckpt_dir "../../../checkpoints/exp1_AdaIR_CUBC"

echo "步骤 2: 训练完成。"
echo "检查点和损失曲线 (loss_curve.png) 应位于 'MLC/checkpoints/exp1_AdaIR_CUBC/'"
