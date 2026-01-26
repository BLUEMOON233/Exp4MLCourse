#!/bin/bash

# Exp3 Decoder 训练脚本 (ImageNet-C 版本)
# 使用 Pixel Shuffle 上采样 + 感知损失

# Training Configuration
EPOCHS=30
BATCH_SIZE=128
NUM_WORKERS=8
LR=2e-4
NUM_RES_BLOCKS=4

# 数据集路径
DATASET_ROOT="../../datasets/ImageNet-C"

# Checkpoint Directory (新模型保存路径)
CKPT_DIR="../../checkpoints/exp3_Decoder_ImageNet"

# 可选：限制训练图像数量 (用于快速测试)
# MAX_IMAGES=10000

# Create checkpoint directory if it doesn't exist
mkdir -p $CKPT_DIR

# Execute the training script
export PYTHONWARNINGS="ignore"

echo "=============================================="
echo "Starting Decoder Training (Pixel Shuffle)"
echo "=============================================="
echo "Dataset: $DATASET_ROOT/origin"
echo "Checkpoints: $CKPT_DIR"
echo "Batch Size: $BATCH_SIZE, Epochs: $EPOCHS"
echo "Num Res Blocks: $NUM_RES_BLOCKS"
echo "=============================================="

# Run Training Script
python train_decoder.py \
    --dataset_root "$DATASET_ROOT" \
    --lr $LR \
    --batch_size $BATCH_SIZE \
    --epochs $EPOCHS \
    --num_workers $NUM_WORKERS \
    --ckpt_dir "$CKPT_DIR" \
    --num_res_blocks $NUM_RES_BLOCKS
    # --max_images $MAX_IMAGES  # 取消注释以限制图像数量

echo "=============================================="
echo "Training finished."
echo "Checkpoints saved to: $CKPT_DIR"
echo "=============================================="
