# -*- coding: utf-8 -*-
"""
VGG 特征解码器训练脚本 (Exp3)
改进版：使用 ImageNet-C/origin 训练，Pixel Shuffle 上采样
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, Callback
from pytorch_lightning.loggers import CSVLogger
import argparse
import os
import matplotlib.pyplot as plt

# 使用 ImageNet 数据集
from utils.dataset_imagenet import ImageNetOriginDataset
from archs.feature_mambair import VGGPerceptualLossExtractor
from archs.decoder import FeatureDecoder

# =============================================================================
# 参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='VGG 特征解码器训练脚本 (Exp3 - ImageNet)')
parser.add_argument('--dataset_root', type=str, default='../../datasets/ImageNet-C', 
                    help='ImageNet-C 数据集路径')
parser.add_argument('--lr', type=float, default=2e-4, help='初始学习率')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小')
parser.add_argument('--epochs', type=int, default=30, help='训练总轮数')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 线程数')
parser.add_argument('--ckpt_dir', type=str, default='../../checkpoints/exp3_Decoder_ImageNet', 
                    help='权重保存路径')
parser.add_argument('--resume', type=str, default='', help='恢复训练的 checkpoint 路径')
parser.add_argument('--max_images', type=int, default=None, 
                    help='限制训练图像数量 (用于快速测试)')
parser.add_argument('--num_res_blocks', type=int, default=4, 
                    help='Decoder 中的残差块数量')

args = parser.parse_args()


class LossPlotCallback(Callback):
    """训练结束后绘制 Loss 曲线"""
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.losses.append(metrics["train_loss"].item())

    def on_fit_end(self, trainer, pl_module):
        if len(self.losses) > 0:
            plt.figure(figsize=(10, 6))
            plt.plot(self.losses, label='Train Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Decoder Training Loss (ImageNet)')
            plt.legend()
            plt.grid(True)
            os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
            plt.savefig(self.save_path)
            plt.close()
            print(f"Loss 曲线已保存至: {self.save_path}")


class PerceptualLoss(nn.Module):
    """
    感知损失：结合像素级 L1 和 VGG 特征匹配
    """
    def __init__(self, vgg_extractor, l1_weight=1.0, perceptual_weight=0.1):
        super().__init__()
        self.vgg = vgg_extractor
        self.l1_loss = nn.L1Loss()
        self.l1_weight = l1_weight
        self.perceptual_weight = perceptual_weight
        
        # ImageNet 归一化
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def normalize(self, x):
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        # 像素级 L1 损失
        l1 = self.l1_loss(pred, target)
        
        # 感知损失 (VGG 特征匹配)
        pred_norm = self.normalize(pred)
        target_norm = self.normalize(target)
        
        with torch.no_grad():
            target_feat = self.vgg.block1(target_norm)
        pred_feat = self.vgg.block1(pred_norm)
        
        perceptual = self.l1_loss(pred_feat, target_feat)
        
        return self.l1_weight * l1 + self.perceptual_weight * perceptual


class DecoderSystem(pl.LightningModule):
    def __init__(self, num_res_blocks=4, use_perceptual_loss=True):
        super().__init__()
        
        # 1. 初始化 VGG 特征提取器
        self.vgg = VGGPerceptualLossExtractor(requires_grad=False)
        self.vgg.eval()
        
        # 2. 初始化改进版解码器 (Pixel Shuffle)
        self.decoder = FeatureDecoder(in_channels=128, out_channels=3, num_res_blocks=num_res_blocks)
        
        # 3. 损失函数
        if use_perceptual_loss:
            # 使用感知损失
            self.criterion = PerceptualLoss(self.vgg, l1_weight=1.0, perceptual_weight=0.1)
        else:
            # 仅使用 L1 损失
            self.criterion = nn.L1Loss()
        
        self.use_perceptual_loss = use_perceptual_loss
        
        # ImageNet 归一化 (用于输入 VGG)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, x):
        return self.decoder(x)
    
    def training_step(self, batch, batch_idx):
        _, _, clean_img = batch
        # 流程: Clean Img -> VGG Feat -> Decoder -> Recon Img -> Loss(Recon, Clean)
        
        clean_norm = self.normalize(clean_img)
        
        with torch.no_grad():
            feat = self.vgg.block1(clean_norm)
            
        recon_img = self.decoder(feat)
        
        if self.use_perceptual_loss:
            loss = self.criterion(recon_img, clean_img)
        else:
            loss = self.criterion(recon_img, clean_img)
        
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.AdamW(self.decoder.parameters(), lr=args.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
        return [optimizer], [scheduler]


def main():
    print("=" * 60)
    print("Exp3 Decoder 训练 (ImageNet-C Origin)")
    print("=" * 60)
    print(f"使用 Pixel Shuffle 上采样")
    print(f"残差块数量: {args.num_res_blocks}")
    print(f"学习率: {args.lr}")
    print(f"批次大小: {args.batch_size}")
    print(f"训练轮数: {args.epochs}")
    print("=" * 60)
    
    # 1. 准备数据集
    root = os.path.abspath(args.dataset_root)
    if not os.path.exists(root):
        fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/ImageNet-C'))
        if os.path.exists(fallback):
            root = fallback
        else:
            print(f"错误: 找不到数据集目录 {root}")
            return

    print(f"加载数据集: {root}")
    trainset = ImageNetOriginDataset(
        root_dir=root, 
        mode='train', 
        image_size=224,
        max_images=args.max_images
    )
    
    if len(trainset) == 0:
        print("错误: 数据集为空!")
        return
    
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 2. 初始化模型
    model = DecoderSystem(num_res_blocks=args.num_res_blocks, use_perceptual_loss=True)
    
    # 3. 配置 Checkpoint
    ckpt_dir = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='decoder_pixelshuffle-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True
    )
    
    loss_plot_callback = LossPlotCallback(os.path.join(ckpt_dir, 'loss_curve.png'))
    logger = CSVLogger(save_dir=ckpt_dir, name="logs")
    
    # 4. 启动 Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, loss_plot_callback],
        log_every_n_steps=50,
        precision='16-mixed'  # 使用混合精度加速训练
    )
    
    print("\n开始训练...")
    if args.resume and os.path.exists(args.resume):
        print(f"恢复 Checkpoint: {args.resume}")
        trainer.fit(model, trainloader, ckpt_path=args.resume)
    else:
        trainer.fit(model, trainloader)
    
    print(f"\n训练完成! 权重保存在: {ckpt_dir}")


if __name__ == '__main__':
    main()
