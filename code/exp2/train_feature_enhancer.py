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

from utils.dataset_cub import CUBCDataset
from utils.schedulers import LinearWarmupCosineAnnealingLR
from archs.feature_mambair import MambaFeatureEnhancer, VGGPerceptualLossExtractor
class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 Loss 的一种鲁棒变体)"""
    def __init__(self, eps=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps

    def forward(self, x, y):
        diff = x - y
        loss = torch.sqrt(diff * diff + self.eps * self.eps)
        return torch.mean(loss)

# =============================================================================
# 参数配置
# =============================================================================
parser = argparse.ArgumentParser(description='Mamba 特征增强网络训练脚本')
# ... [lines 20-28 skipped, assumed unchanged by tool context awareness, but wait.
# I must provide context. StartLine and EndLine logic is strict.] 
# The tool replaces specific lines. I cannot insert the class randomly in the middle of a function unless I am careful.
# Better to define the class at top level imports and then use it.
# However, I will define it before parser or FeatureEnhancerSystem.

# Actually, I can put the class definition near imports or before `FeatureEnhancerSystem`.
# Let's insert the Class definition first separately to be safe.

parser.add_argument('--dataset_root', type=str, default='../../datasets/CUB-C', help='CUB-C 数据集路径')
parser.add_argument('--lr', type=float, default=2e-4, help='初始学习率')
parser.add_argument('--batch_size', type=int, default=32, help='批次大小 (建议 >16)')
parser.add_argument('--epochs', type=int, default=50, help='训练总轮数')
parser.add_argument('--num_workers', type=int, default=4, help='DataLoader 线程数')
parser.add_argument('--ckpt_dir', type=str, default='checkpoints/exp2_MambaFeatureEnhancer_CUBC', help='权重保存路径')
parser.add_argument('--resume', type=str, default='', help='恢复训练的 checkpoint 路径')

args = parser.parse_args()

class LossPlotCallback(Callback):
    """用于在训练结束后绘制 Loss 曲线的回调函数"""
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # 记录每个 epoch 的平均 loss
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.losses.append(metrics["train_loss"].item())

    def on_fit_end(self, trainer, pl_module):
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='Train Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Feature Enhancement Training Loss')
        plt.legend()
        plt.grid(True)
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path)
        plt.close()
        print(f"Loss 曲线已保存至: {self.save_path}")

class FeatureEnhancerSystem(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        # 1. 初始化 VGG 特征提取器 (作为 Loss Network 和 前置特征提取)
        # 包含 block1, block2, block3
        self.vgg = VGGPerceptualLossExtractor(requires_grad=False)
        self.vgg.eval() # 冻结状态
        
        # 2. 初始化 Mamba 增强模块
        # 输入通道为 128 (对应 VGG Block 1 - conv2_2 的输出)
        # 使用 FDM 风格的配置: 3 个分支, 不同维度
        self.net = MambaFeatureEnhancer(
            in_channels=128, 
            img_channels=3, 
            dims=[128, 64, 32], 
            blocks_per_group=4
        )
        
        # 3. 损失函数：使用 Charbonnier Loss
        # 针对特征空间稀疏性和抗异常值（如亮度截断）需求，优于 MSE
        self.criterion = CharbonnierLoss()
        
        # 4. 注册 ImageNet 归一化参数 (用于输入预处理)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def normalize(self, x):
        """将 [0, 1] 的图像归一化到 ImageNet 分布"""
        return (x - self.mean) / self.std

    def forward(self, x, img):
        # 推理接口: 输入 VGG 特征 + 图像 -> 输出增强特征
        return self.net(x, img)
    
    def training_step(self, batch, batch_idx):
        _, degrad_img, clean_img = batch
        
        # --- 步骤 1: 数据归一化 ---
        degrad_norm = self.normalize(degrad_img)
        clean_norm = self.normalize(clean_img)
        
        # --- 步骤 2: 提取基础特征 (VGG Block 1) ---
        with torch.no_grad():
            # 获取 Clean 图像的浅层特征 (作为 Ground Truth)
            # f1_c: Block 1 Output (64 ch)
            f1_c = self.vgg.block1(clean_norm)
            
            # 获取 Degraded 图像的基础特征 (Block 1)
            f1_d = self.vgg.block1(degrad_norm)
            
        # --- 步骤 3: Mamba 特征增强 ---
        # 将降质特征和降质图像一同输入 Mamba 模块
        f1_enhanced = self.net(f1_d, degrad_norm)
        
        # --- 步骤 4: 计算浅层特征 MSE 损失 ---
        loss = self.criterion(f1_enhanced, f1_c)
        
        # 记录日志
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.net.parameters(), lr=args.lr, betas=(0.9, 0.999))
        
        # 预热 + 余弦退火调度器
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, 
            warmup_epochs=min(5, args.epochs//10), 
            max_epochs=args.epochs
        )
        return [optimizer], [scheduler]

def main():
    print("=== 开始初始化训练流程 ===")
    
    # 1. 准备数据集
    root = os.path.abspath(args.dataset_root)
    if not os.path.exists(root):
        # 尝试自动寻找路径
        fallback = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../datasets/CUB-C'))
        if os.path.exists(fallback):
            root = fallback
        else:
            print(f"错误: 无法找到数据集目录 {root}")
            return
            
    print(f"加载数据集: {root}")
    # 移除了 patch_size 参数
    trainset = CUBCDataset(root_dir=root)
    
    trainloader = DataLoader(
        trainset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    # 2. 初始化模型系统
    model = FeatureEnhancerSystem()
    
    # 3. 配置 Checkpoint 保存逻辑
    os.makedirs(args.ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='mamba_enhancer-{epoch:02d}-{train_loss:.4f}',
        save_top_k=3,
        monitor='train_loss',
        mode='min',
        save_last=True
    )
    
    loss_plot_callback = LossPlotCallback(os.path.join(args.ckpt_dir, 'loss_curve.png'))
    logger = CSVLogger(save_dir=args.ckpt_dir, name="logs")
    
    # 4. 启动 Trainer
    trainer = pl.Trainer(
        max_epochs=args.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        logger=logger,
        callbacks=[checkpoint_callback, loss_plot_callback],
        log_every_n_steps=10
    )
    
    print("=== 开始训练 ===")
    if args.resume and os.path.exists(args.resume):
        print(f"恢复 Checkpoint: {args.resume}")
        trainer.fit(model, trainloader, ckpt_path=args.resume)
    else:
        trainer.fit(model, trainloader)

if __name__ == '__main__':
    main()
