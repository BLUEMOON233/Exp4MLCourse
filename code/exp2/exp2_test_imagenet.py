# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose, Resize, CenterCrop
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from tqdm import tqdm
import warnings
import logging
import datetime

# 添加当前目录到路径确保能找到 archs
sys.path.append(os.path.dirname(__file__))
from archs.feature_mambair import MambaFeatureEnhancer

# 忽略警告
warnings.filterwarnings("ignore", category=UserWarning)

# ImageNet 标准归一化
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# =============================================================================
# 工具类与数据加载
# =============================================================================

class AverageMeter(object):
    """计算并存储平均值和当前值"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_synset_mapping(mapping_path):
    """加载 synset_mapping.txt，构建 wnid -> class_index 的映射"""
    wnid_to_idx = {}
    if not os.path.exists(mapping_path):
        print(f"Warning: Mapping file not found at {mapping_path}")
        return {}
    with open(mapping_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            wnid = line.strip().split(' ')[0]
            wnid_to_idx[wnid] = idx
    return wnid_to_idx

class ImageNetCDataset(Dataset):
    """
    ImageNet-C 数据集加载器
    适配扁平化目录结构: root_dir/corruption_type/class_wnid/*.jpeg
    """
    def __init__(self, root_dir, corruption, max_images=None, wnid_mapping=None):
        self.root_dir = root_dir
        self.corruption = corruption
        self.wnid_mapping = wnid_mapping
        
        # 定义图像预处理
        # 用户确认数据集已是 224x224，直接转 Tensor，避免任何插值
        self.transform = Compose([
            ToTensor()
        ])

        # 处理 origin 目录可能位于 root 或 root/origin 的情况
        if corruption == 'origin':
            potential_path = os.path.join(root_dir, 'origin')
            if os.path.exists(potential_path):
                self.images_dir = potential_path
            else:
                self.images_dir = os.path.join(root_dir, 'origin') # Fallback
        else:
            self.images_dir = os.path.join(root_dir, corruption)
        
        self.image_paths = []
        if not os.path.exists(self.images_dir):
            print(f"警告: 数据目录不存在 {self.images_dir}")
        else:
            self._scan_directory()

        if max_images is not None and len(self.image_paths) > max_images:
            # 简单随机采样或截断，这里直接截断
            self.image_paths = self.image_paths[:max_images]
            print(f"[{corruption}] 限制测试数量: {len(self.image_paths)}")
        else:
             print(f"[{corruption}] 已加载图像数: {len(self.image_paths)}")

    def _scan_directory(self):
        try:
            subdirs = sorted(os.listdir(self.images_dir))
            for cls in subdirs:
                cls_dir = os.path.join(self.images_dir, cls)
                if not os.path.isdir(cls_dir): continue
                
                # 跳过数字目录（如果存在），只处理 wnid 目录
                if cls.isdigit(): continue

                class_idx = -1
                if self.wnid_mapping and cls in self.wnid_mapping:
                    class_idx = self.wnid_mapping[cls]
                
                fnames = sorted(os.listdir(cls_dir))
                for fname in fnames:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append({
                            'path': os.path.join(cls_dir, fname),
                            'label': class_idx
                        })
        except Exception as e:
            print(f"扫描目录出错: {e}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = self.image_paths[idx]
        try:
            img = Image.open(item['path']).convert('RGB')
            img_tensor = self.transform(img)
            return img_tensor, item['label']
        except Exception as e:
            print(f"Error loading {item['path']}: {e}")
            # 返回一个全黑图像防止 crash
            return torch.zeros(3, 224, 224), item['label']

# =============================================================================
# 模型定义
# =============================================================================

class SplitVGG16(nn.Module):
    """
    拆分后的 VGG16。
    Part 1: 前10层 (Feature Extraction until Block 2) -> Output 128 channels
    Part 2: 剩余层 (Feature Extraction Block 3-5 + Classifier)
    """
    def __init__(self, split_idx=10):
        super().__init__()
        original_vgg = vgg16(weights=VGG16_Weights.DEFAULT)
        features = list(original_vgg.features.children())
        
        self.part1 = nn.Sequential(*features[:split_idx])
        self.part2_features = nn.Sequential(*features[split_idx:])
        self.avgpool = original_vgg.avgpool
        self.classifier = original_vgg.classifier
        
        self.eval() # 默认 Eval 模式
        
        # 冻结参数
        for p in self.parameters():
            p.requires_grad = False

    def forward_part1(self, x):
        return self.part1(x)

    def forward_part2(self, x):
        x = self.part2_features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def load_enhancer(ckpt_path, device):
    """加载 Mamba Enhancer 模型"""
    print(f"Loading Enhancer from: {ckpt_path}")
    # 关键修改: in_channels=128 (适配 VGG Block 2 输出)
    model = MambaFeatureEnhancer(in_channels=128, img_channels=3, blocks_per_group=4)
    
    if os.path.exists(ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location='cpu')
        state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
        
        # 只加载属于 'net' 子模块 (MambaFeatureEnhancer) 的权重
        # 忽略 LightningModule 层的 'mean', 'std' 以及 'vgg' 相关权重
        clean_state_dict = {k[4:]: v for k, v in state_dict.items() if k.startswith('net.')}
        
        try:
            model.load_state_dict(clean_state_dict, strict=True)
            print("Enhancer weights loaded successfully.")
        except Exception as e:
            print(f"Error loading weights: {e}")
            print("Attempting lenient loading (strict=False)...")
            model.load_state_dict(clean_state_dict, strict=False)
    else:
        print(f"Warning: Checkpoint not found at {ckpt_path}, using random init.")

    model.to(device).eval()
    return model

# =============================================================================
# 测试主逻辑
# =============================================================================

def test_corruption(vgg, enhancer, dataset_root, corruption, wnid_mapping, device, args):
    dataset = ImageNetCDataset(dataset_root, corruption, max_images=args.max_images, wnid_mapping=wnid_mapping)
    if len(dataset) == 0:
        return 0.0, 0.0

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    meter_orig = AverageMeter()
    meter_enh = AverageMeter()
    
    # 归一化层
    normalizer = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)

    with torch.no_grad():
        for img, label in tqdm(dataloader, desc=f"Testing {corruption}", leave=False):
            img = img.to(device)
            label = label.to(device)
            
            # 对输入图像进行归一化，送入 VGG
            img_norm = normalizer(img)
            
            # --- 1. VGG Part 1 ---
            feat_degraded = vgg.forward_part1(img_norm)
            
            # --- 2. 原始路径 ---
            logits_orig = vgg.forward_part2(feat_degraded)
            acc_orig = (logits_orig.argmax(dim=1) == label).float().mean().item()
            meter_orig.update(acc_orig, len(img))
            
            # --- 3. 增强路径 ---
            # Enhancer 接收 [Part1 Features] 和 [Normalized Image]
            feat_enhanced = enhancer(feat_degraded, img_norm)
            logits_enh = vgg.forward_part2(feat_enhanced)
            acc_enh = (logits_enh.argmax(dim=1) == label).float().mean().item()
            meter_enh.update(acc_enh, len(img))
            
    return meter_orig.avg * 100, meter_enh.avg * 100

def setup_logger(output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"test_results_{timestamp}.txt")
    
    logger = logging.getLogger("Exp2_Test")
    logger.setLevel(logging.INFO)
    
    # File Handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(fh)
    
    # Stream Handler
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(ch)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="ImageNet-C Mamba Feature Enhancer Test")
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_images', type=int, default=None, help="Debug mode: limit images per corruption")
    args = parser.parse_args()

    # 路径配置
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '../../'))
    dataset_root = os.path.join(project_root, 'datasets', 'ImageNet-C')
    ckpt_root = os.path.join(project_root, 'checkpoints', 'exp2_MambaFeatureEnhancer_CUBC')
    mapping_path = os.path.join(dataset_root, 'synset_mapping.txt')
    
    # 日志
    logger = setup_logger(os.path.join(current_dir, 'results'))
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    
    # 加载映射
    wnid_mapping = load_synset_mapping(mapping_path)
    
    # 加载模型
    vgg = SplitVGG16(split_idx=10).to(device) # Split at layer 10 (after Block 2)
    
    ckpt_path = os.path.join(ckpt_root, 'last.ckpt')
    # 如果 last.ckpt 不存在，找最新的作为 Fallback
    if not os.path.exists(ckpt_path) and os.path.exists(ckpt_root):
        ckpts = sorted([f for f in os.listdir(ckpt_root) if f.endswith('.ckpt')])
        if ckpts: 
            ckpt_path = os.path.join(ckpt_root, ckpts[-1])
            
    enhancer = load_enhancer(ckpt_path, device)
    
    # 定义测试集
    corruptions = ['contrast', 'fog', 'motion_blur', 'snow', 'brightness', 'origin']
    
    # 表头
    logger.info("\n" + "="*85)
    logger.info(f"{'Corruption':<20} {'Orig Acc (%)':<20} {'Enh Acc (%)':<20} {'Diff':<10}")
    logger.info("="*85)
    
    results = {}
    
    for corr in corruptions:
        acc_orig, acc_enh = test_corruption(vgg, enhancer, dataset_root, corr, wnid_mapping, device, args)
        diff = acc_enh - acc_orig
        logger.info(f"{corr:<20} {acc_orig:<20.2f} {acc_enh:<20.2f} {diff:<+10.2f}")
        results[corr] = (acc_orig, acc_enh)
        
    logger.info("="*85)
    
    # Summary of Shifts (excluding origin)
    logger.info("\nSummary (Excluding Origin):")
    total_diff = 0
    count = 0
    for k, v in results.items():
        if k == 'origin': continue
        total_diff += (v[1] - v[0])
        count += 1
        
    if count > 0:
        avg_diff = total_diff / count
        logger.info(f"Average Improvement on Corruptions: {avg_diff:+.2f}%")
    
    logger.info(f"Results saved to logs.")

if __name__ == "__main__":
    main()
