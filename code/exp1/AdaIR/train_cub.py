import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Import our new dataset
from utils.dataset_cub import CUBCDataset 
from net.model import AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
import numpy as np
import os
import matplotlib.pyplot as plt

# Using options from existing file, but we might override paths
from options import options as opt
# import lightning.pytorch as pl
# from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
# from lightning.pytorch.callbacks import ModelCheckpoint, Callback
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger, CSVLogger
from pytorch_lightning.callbacks import ModelCheckpoint, Callback

class LossPlotCallback(Callback):
    def __init__(self, save_path):
        super().__init__()
        self.save_path = save_path
        self.losses = []

    def on_train_epoch_end(self, trainer, pl_module):
        # Retrieve the logged train_loss. Note: this fetches the value logged in training_step 
        # aggregated over the epoch (default logic for on_step=False, on_epoch=True)
        # However, in the original code: self.log("train_loss", loss) 
        # creates a metric. We need to ensure it's logged per epoch.
        metrics = trainer.callback_metrics
        if "train_loss" in metrics:
            self.losses.append(metrics["train_loss"].item())
        else:
            print("警告：metrics 中未找到 train_loss")

    def on_fit_end(self, trainer, pl_module):
        # Plot and save
        # 尝试设置中文字体支持
        plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(10, 6))
        plt.plot(self.losses, label='训练损失')
        plt.xlabel('轮次 (Epoch)')
        plt.ylabel('L1 损失 (L1 Loss)')
        plt.title('CUB-C 数据集训练损失曲线')
        plt.legend()
        plt.grid(True)
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        plt.savefig(self.save_path)
        print(f"损失曲线已保存至 {self.save_path}")
        plt.close()

class AdaIRModel(pl.LightningModule):
    def __init__(self, pretrained_path=None):
        super().__init__()
        self.net = AdaIR(decoder=True)
        self.loss_fn  = nn.L1Loss()
        
        if pretrained_path and os.path.exists(pretrained_path):
            print(f"正在加载预训练权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location='cpu')
            
            # Check format: wrapper dictionary or direct state_dict
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            # Remove 'net.' prefix if present (from PL saving) or add it if model expects it?
            # Self.net is the model.
            # If checkpoint keys start with 'net.', we can load directly into self?
            # Or if we load into self.net, we should remove 'net.' prefix.
            
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('net.'):
                    new_state_dict[k[4:]] = v
                else:
                    new_state_dict[k] = v
            
            # Load into self.net
            try:
                self.net.load_state_dict(new_state_dict, strict=True)
                print("预训练权重加载成功 (Strict Mode)")
            except Exception as e:
                print(f"严格加载失败，尝试非严格加载: {e}")
                self.net.load_state_dict(new_state_dict, strict=False)
                print("预训练权重加载成功 (Non-Strict Mode)")
        else:
            if pretrained_path:
                print(f"警告: 预训练权重文件不存在: {pretrained_path}")
            else:
                print("未指定预训练权重，使用随机初始化。")
    
    def forward(self,x):
        return self.net(x)
    
    def training_step(self, batch, batch_idx):
        # Desctructure batch based on what CUBCDataset returns
        # Return format: [clean_name, de_id], degrad_patch, clean_patch
        ([clean_name, de_id], degrad_patch, clean_patch) = batch
        
        restored = self.net(degrad_patch)

        loss = self.loss_fn(restored, clean_patch)
        
        # Log with on_epoch=True to get epoch average for the callback
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss
    
    def lr_scheduler_step(self, scheduler, optimizer_idx, metric):
        scheduler.step(self.current_epoch)
        # lr = scheduler.get_lr() # unused
    
    def configure_optimizers(self):
        # Match paper: Adam optimizer (beta1=0.9, beta2=0.999)
        optimizer = optim.Adam(self.parameters(), lr=opt.lr, betas=(0.9, 0.999)) 
        # Adjust max_epochs to match opt.epochs
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=min(15, opt.epochs//5), max_epochs=opt.epochs)

        return [optimizer],[scheduler]


def main():
    print("开始在 CUB-C 数据集上进行训练...")
    
    # Setup Paths
    # Assuming code run from code/exp1/
    # CUB-C at ../../../datasets/CUB-C (Up from AdaIR -> exp1 -> code -> root)
    dataset_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../datasets/CUB-C'))
    print(f"数据集根目录: {dataset_root}")
    
    # Initialize Dataset
    trainset = CUBCDataset(root_dir=dataset_root, patch_size=opt.patch_size)
    
    # Check directory
    # Use opt.ckpt_dir from arguments
    ckpt_dir = opt.ckpt_dir
    print(f"检查点保存目录: {ckpt_dir}")
    os.makedirs(ckpt_dir, exist_ok=True)
    
    # Checkpoint Callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='adair-{epoch:02d}-{train_loss:.4f}',
        every_n_epochs=1,
        save_top_k=-1,
        save_last=True
    )
    
    # Loss Plot Callback
    loss_plot_callback = LossPlotCallback(save_path=os.path.join(ckpt_dir, "loss_curve.png"))

    # Logger
    # Use CSVLogger to avoid tensorboard dependency issues
    logger = CSVLogger(save_dir="logs/", name="AdaIR_CUBC")

    trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                             drop_last=True, num_workers=opt.num_workers)
    
    # Check for pretrained model
    # Assuming code run from code/exp1/AdaIR/
    # Pretrained path: ckpt/adair5d.ckpt
    # pretrained_path = os.path.join(os.path.dirname(__file__), 'ckpt/adair5d.ckpt')
    pretrained_path = os.path.join(os.path.dirname(__file__), 'ckpt/best.ckpt')
    
    model = AdaIRModel(pretrained_path=pretrained_path)
    
    # Trainer config
    # Users requested to "re-train parameters", implying scratch or finetune.
    # We load fresh model (init from implementation).
    
    # accelerator="auto" might not work in 1.9.0 depending on setup, but typically works. 
    # If not, use accelerator='gpu', devices=1 etc.
    # For safe compatibility with 1.9:
    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1, 
        # strategy="auto",  # Removing auto strategy to be safe if ddp not needed or complex
        logger=logger,
        callbacks=[checkpoint_callback, loss_plot_callback],
        log_every_n_steps=10
    )
    
    trainer.fit(model=model, train_dataloaders=trainloader)
    print(f"训练完成。模型权重和损失曲线已保存至 {ckpt_dir}")

if __name__ == '__main__':
    main()
