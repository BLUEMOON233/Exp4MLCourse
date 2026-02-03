import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
import matplotlib.font_manager as fm

def configure_plotting_style():
    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except:
        plt.style.use('ggplot')
        
    font_preferences = ['Arial Unicode MS', 'Heiti TC', 'PingFang HK', 'Songti SC', 'SimHei', 'DejaVu Sans']
    available_fonts = set([f.name for f in fm.fontManager.ttflist])
    selected_font = 'sans-serif'
    for f in font_preferences:
        if f in available_fonts:
            selected_font = f
            break
            
    print(f"Selected font: {selected_font}")
    plt.rcParams['font.family'] = [selected_font, 'sans-serif']
    plt.rcParams['axes.unicode_minus'] = False
    
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.titlesize'] = 20
    plt.rcParams['axes.labelsize'] = 16
    plt.rcParams['xtick.labelsize'] = 14
    plt.rcParams['ytick.labelsize'] = 14
    plt.rcParams['legend.fontsize'] = 14

def process_and_plot(log_dir, output_name="loss_curve_refined_zh.png", specific_versions=None):
    print(f"Processing directory: {log_dir}")
    
    # Locate version directories
    if specific_versions:
        # User specified specific versions (list of strings like "version_1")
        version_dirs = []
        for v_name in specific_versions:
            v_path = os.path.join(log_dir, v_name)
            if os.path.exists(v_path):
                version_dirs.append(v_path)
            else:
                print(f"  Warning: Specified {v_name} not found in {log_dir}")
    else:
        # Default: auto-discover all versions sorted numerically
        version_dirs = sorted(glob.glob(os.path.join(log_dir, "version_*")), 
                              key=lambda x: int(os.path.basename(x).split('_')[-1]))
    
    if not version_dirs:
        print(f"No versions found in {log_dir}")
        return

    combined_df = pd.DataFrame()
    last_max_epoch = -1
    
    for i, v_dir in enumerate(version_dirs):
        csv_path = os.path.join(v_dir, "metrics.csv")
        if not os.path.exists(csv_path):
            continue
            
        print(f"  Reading {os.path.basename(v_dir)}...")
        df = pd.read_csv(csv_path)
        
        # Extract epoch and loss
        if 'train_loss_step' in df.columns:
            df_subset = df[['epoch', 'train_loss_step']].dropna()
            df_avg = df_subset.groupby('epoch')['train_loss_step'].mean().reset_index()
        elif 'train_loss_epoch' in df.columns:
            df_subset = df[['epoch', 'train_loss_epoch']].dropna()
            df_avg = df_subset.rename(columns={'train_loss_epoch': 'train_loss_step'}).reset_index(drop=True)
        else:
            print(f"    Skipping {v_dir}: No suitable loss column found.")
            continue
            
        if df_avg.empty:
            print(f"    Skipping {v_dir}: Empty data.")
            continue

        # Shift epochs logic
        # If specific_versions is provided and has only 1 item, we probably don't want to offset blindly based on previous
        # But if specific_versions has multiple, we probably DO want to concatenate them.
        # However, typically if we select just one version (like version_1), we might want to keep its original epochs 
        # OR shift it if it was a continuation.
        # The prompt says "只需绘制version1", implying we look at that training run. 
        # Usually version numbers imply restarts. If I just plot version 1, I should probably plot it starting from its own epoch 0 
        # UNLESS it was a resume. 
        # But for safety, if we are cherry-picking one version, let's treat it as the ground truth for that plot unless told otherwise.
        # BUT, standard PyTorch Lightning logging usually resets epoch to 0 in new versions unless explicitly handled, 
        # OR if it resumes, it might continue.
        # Let's inspect the data logic: 
        # If we concatenate, we shift.
        # If we just have one file, we don't shift based on 'previous' files because there are no previous files in this run.
        
        if not combined_df.empty:
             offset = last_max_epoch + 1
             df_avg['epoch'] = df_avg['epoch'] + offset
             
        combined_df = pd.concat([combined_df, df_avg])
        last_max_epoch = df_avg['epoch'].max()

    if combined_df.empty:
        print("No valid data to plot.")
        return
        
    combined_df = combined_df.sort_values('epoch').reset_index(drop=True)
    
    # Plotting
    plt.figure(figsize=(10, 8), dpi=300)
    scientific_color = '#1f77b4' 
    
    # Removed markers as requested
    plt.plot(combined_df['epoch'], combined_df['train_loss_step'], 
             linestyle='-', linewidth=2.5, 
             color=scientific_color, label='训练损失')
    
    plt.title('训练曲线', pad=20, weight='bold') # Chinese
    plt.xlabel('Epoch', labelpad=10)
    plt.ylabel('Loss', labelpad=10)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(frameon=True, fancybox=True, framealpha=0.9, loc='upper right')
    plt.tight_layout()
    
    output_path = os.path.join(log_dir, output_name)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved plot to {output_path}")

def main():
    configure_plotting_style()
    
    # Exp2: Only use version_1
    dir_exp2 = "/Users/wardenliu/Develop/Projects/Exp4MLCourse/checkpoints/exp2_MambaFeatureEnhancer_CUBC/logs"
    if os.path.exists(dir_exp2):
        process_and_plot(dir_exp2, specific_versions=['version_1'])
    else:
        print(f"Directory not found: {dir_exp2}")

    # Exp3: Default (use all/whatever is there, which is currently just version_0)
    dir_exp3 = "/Users/wardenliu/Develop/Projects/Exp4MLCourse/checkpoints/exp3_Decoder_ImageNet/logs"
    if os.path.exists(dir_exp3):
        process_and_plot(dir_exp3)
    else:
        print(f"Directory not found: {dir_exp3}")

if __name__ == "__main__":
    main()
