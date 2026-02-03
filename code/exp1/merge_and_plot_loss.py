import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.font_manager as fm

def configure_plotting_style():
    # Attempt to use a scientific style if available, otherwise default
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
        
    # Configure fonts for Chinese support on macOS
    # Prioritize fonts we confirmed exist
    font_preferences = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'Songti SC', 'SimHei', 'DejaVu Sans']
    
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    selected_font = 'sans-serif' # Fallback
    
    for f in font_preferences:
        if f in available_fonts:
            selected_font = f
            break
            
    print(f"Selected font: {selected_font}")
    plt.rcParams['font.family'] = [selected_font, 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False # Solve minus sign issue
    
    # Increase font sizes
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

def merge_and_plot_loss_refined():
    configure_plotting_style()
    
    base_dir = "/Users/wardenliu/Develop/Projects/Exp4MLCourse/code/exp1/AdaIR/logs/AdaIR_CUBC"
    file_v0 = os.path.join(base_dir, "version_0/metrics.csv")
    file_v1 = os.path.join(base_dir, "version_1/metrics.csv")
    
    if not os.path.exists(file_v0) or not os.path.exists(file_v1):
        print("Missing metric files.")
        return

    # Process Version 0
    print("Processing Version 0...")
    df0 = pd.read_csv(file_v0)
    if 'train_loss_step' in df0.columns:
        df0_clean = df0[['epoch', 'train_loss_step']].dropna()
        df0_avg = df0_clean.groupby('epoch')['train_loss_step'].mean().reset_index()
    else:
        df0_clean = df0[['epoch', 'train_loss_epoch']].dropna()
        df0_avg = df0_clean.rename(columns={'train_loss_epoch': 'train_loss_step'}).reset_index(drop=True)
    
    max_epoch_v0 = df0_avg['epoch'].max()
    offset = max_epoch_v0 + 1

    # Process Version 1
    print("Processing Version 1...")
    df1 = pd.read_csv(file_v1)
    if 'train_loss_step' in df1.columns:
        df1_clean = df1[['epoch', 'train_loss_step']].dropna()
        df1_avg = df1_clean.groupby('epoch')['train_loss_step'].mean().reset_index()
    else:
        df1_clean = df1[['epoch', 'train_loss_epoch']].dropna()
        df1_avg = df1_clean.rename(columns={'train_loss_epoch': 'train_loss_step'}).reset_index(drop=True)

    # Shift epochs
    df1_avg['epoch'] = df1_avg['epoch'] + offset
    
    # Concatenate
    final_df = pd.concat([df0_avg, df1_avg]).sort_values('epoch').reset_index(drop=True)
    
    # Plotting
    plt.figure(figsize=(10, 8), dpi=300) # High DPI for academic quality
    
    # Scientific color using hex
    scientific_color = '#1f77b4' 
    
    # Updated: removed marker='o'
    plt.plot(final_df['epoch'], final_df['train_loss_step'], 
             linestyle='-', linewidth=2.5, 
             color=scientific_color, label='训练损失')
    
    plt.title('训练曲线', pad=20, weight='bold') # Chinese Title
    plt.xlabel('Epoch', labelpad=10) # Keep English
    plt.ylabel('Loss', labelpad=10)  # Keep English
    
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
    
    # Optimize layout
    plt.tight_layout()
    
    output_path = os.path.join(base_dir, "loss_curve_refined_zh.png")
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Refined plot saved to: {output_path}")

if __name__ == "__main__":
    merge_and_plot_loss_refined()
