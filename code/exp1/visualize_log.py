import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
import glob

def plot_metrics(log_dir, version=None):
    # Determine the target directory
    if version is not None:
        target_dir = os.path.join(log_dir, f"version_{version}")
    else:
        # Find the latest version if not specified
        versions = glob.glob(os.path.join(log_dir, "version_*"))
        if not versions:
            print(f"找不到任何版本目录: {log_dir}")
            return
        # Sort by version number
        versions.sort(key=lambda x: int(x.split('_')[-1]))
        target_dir = versions[-1]
        print(f"自动选择最新版本: {os.path.basename(target_dir)}")

    csv_path = os.path.join(target_dir, "metrics.csv")
    if not os.path.exists(csv_path):
        print(f"在 {target_dir} 中未找到 metrics.csv")
        return

    print(f"正在读取: {csv_path}")
    df = pd.read_csv(csv_path)

    # Plotting
    plt.figure(figsize=(10, 6))

    # Check for train_loss
    # Lightning CSVLogger often saves multiple rows per epoch (some with step metrics, some with epoch metrics)
    # Usually 'train_loss' (epoch average) has non-null values at the end of epoch rows.
    # We filter rows where 'train_loss' is not NaN.
    
    # We prefer 'train_loss_step' as requested
    metric_name = 'train_loss'
    if 'train_loss_step' in df.columns:
        metric_name = 'train_loss_step'
    elif 'train_loss_epoch' in df.columns:
        metric_name = 'train_loss_epoch'
    elif 'train_loss' in df.columns:
        metric_name = 'train_loss'
    else:
        print(f"CSV中未找到 Loss 相关列，可用列: {df.columns.tolist()}")
        return

    # Filter out NaNs (CSVLogger creates sparse rows when multiple metrics are logged at different frequencies)
    df_plot = df.dropna(subset=[metric_name])
    
    # User requested to use row number (step) as x-axis
    x_axis = range(len(df_plot))
    x_label = 'Step'

    plt.plot(x_axis, df_plot[metric_name], marker='o', label='Train Loss')
    
    plt.title(f'Training Loss - {os.path.basename(target_dir)}')
    plt.xlabel(x_label)
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()

    # Support for Chinese characters (just in case)
    plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False

    output_path = os.path.join(target_dir, "loss_visualization.png")
    plt.savefig(output_path)
    print(f"图表已保存至: {output_path}")
    # plt.show() # Disabled for remote server

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize metrics.csv from PyTorch Lightning logs")
    # Default path relative to script execution
    default_log_dir = os.path.join(os.path.dirname(__file__), "AdaIR/logs/AdaIR_CUBC")
    
    parser.add_argument("--log_dir", type=str, default=default_log_dir, help="Base directory of logs")
    parser.add_argument("--version", type=int, default=None, help="Specific version to plot (e.g., 0). Default is latest.")
    
    args = parser.parse_args()
    
    # Allow running from different CWDs
    if not os.path.exists(args.log_dir):
        # Try relative to code/exp1 if script is run from there
        alt_dir = "AdaIR/logs/AdaIR_CUBC" 
        if os.path.exists(alt_dir):
            args.log_dir = alt_dir
        else:
             # Try absolute path based on user environment if standard path fails
             pass

    plot_metrics(args.log_dir, args.version)
