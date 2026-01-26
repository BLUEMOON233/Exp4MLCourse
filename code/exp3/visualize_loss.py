import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import glob

def visualize_loss(ckpt_dir, output_file="loss_curve.png"):
    # 寻找 logs 目录
    logs_root = os.path.join(ckpt_dir, "logs")
    if not os.path.exists(logs_root):
        print(f"错误: 找不到日志目录 {logs_root}")
        return

    # 寻找最新的 version 目录
    versions = sorted(glob.glob(os.path.join(logs_root, "version_*")), key=os.path.getmtime)
    if not versions:
        print(f"错误: 在 {logs_root} 中未找到任何 version_* 目录")
        return
    
    # 默认使用最新的日志
    latest_version = versions[-1]
    metrics_csv = os.path.join(latest_version, "metrics.csv")
    
    if not os.path.exists(metrics_csv):
        print(f"错误: 在 {latest_version} 中未找到 metrics.csv")
        return
        
    print(f"正在读取日志文件: {metrics_csv}")
    
    # 读取 CSV
    try:
        df = pd.read_csv(metrics_csv)
    except Exception as e:
        print(f"读取 CSV 失败: {e}")
        return

    # 绘制 Loss 曲线
    plt.figure(figsize=(10, 6))
    
    # 过滤掉 NaN 值 (PL 有时会记录 step loss 但 epoch loss 为 NaN，反之亦然)
    # 我们优先绘制 epoch 级别的 loss，如果没有则绘制 step 级别的
    
    if 'train_loss_epoch' in df.columns:
        # 清洗数据，去除 NaN
        data = df[['epoch', 'train_loss_epoch']].dropna()
        plt.plot(data['epoch'], data['train_loss_epoch'], label='Train Loss (Epoch)', marker='o')
        plt.xlabel('Epoch')
    elif 'train_loss' in df.columns:
        # 可能是 step 级别的
        data = df['train_loss'].dropna().reset_index()
        plt.plot(data.index, data['train_loss'], label='Train Loss (Step)', alpha=0.6)
        plt.xlabel('Step')
    else:
        print("警告: 无法在 CSV 中找到 'train_loss_epoch' 或 'train_loss' 列")
        print(f"可用列名: {df.columns.tolist()}")
        return

    plt.ylabel('Loss')
    plt.title(f'Training Loss Curve (Source: {os.path.basename(latest_version)})')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 保存图片
    save_path = os.path.join(ckpt_dir, output_file)
    plt.savefig(save_path)
    print(f"Loss 曲线已保存至: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="绘制 PyTorch Lightning CSV 日志的 Loss 曲线")
    parser.add_argument('--ckpt_dir', type=str, default='../../checkpoints/exp3_Decoder_CUBC', 
                        help='Checkpoints 根目录 (包含 logs 文件夹)')
    parser.add_argument('--output', type=str, default='loss_curve.png', help='输出图片文件名')
    
    args = parser.parse_args()
    
    # 处理相对路径
    if not os.path.isabs(args.ckpt_dir):
        args.ckpt_dir = os.path.abspath(args.ckpt_dir)
        
    visualize_loss(args.ckpt_dir, args.output)
