# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import logging
import datetime
import warnings

# 忽略警告
warnings.filterwarnings("ignore")

# 导入 Exp3 定义的模块
from archs.feature_mambair import MambaFeatureEnhancer, VGGPerceptualLossExtractor
from archs.decoder import FeatureDecoder
from utils.image_utils import crop_img

# ImageNet 标准归一化 (用于分类器)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

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

def setup_logger(output_dir):
    """设置日志记录器"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"test_results_{timestamp}.txt")
    
    logger = logging.getLogger("Exp3_Test")
    logger.setLevel(logging.INFO)
    
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def compute_metrics_safe(recoverd, clean):
    """
    计算 PSNR 和 SSIM
    """
    assert recoverd.shape == clean.shape
    recoverd = np.clip(recoverd.detach().cpu().numpy(), 0, 1)
    clean = np.clip(clean.detach().cpu().numpy(), 0, 1)

    # (B, C, H, W) -> (B, H, W, C)
    recoverd = recoverd.transpose(0, 2, 3, 1)
    clean = clean.transpose(0, 2, 3, 1)
    psnr = 0
    ssim = 0
    
    B = recoverd.shape[0]

    for i in range(B):
        # PSNR
        psnr += peak_signal_noise_ratio(clean[i], recoverd[i], data_range=1)
        
        # SSIM
        try:
            ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=-1)
        except TypeError:
            ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)
            
    return psnr / B, ssim / B, B

def load_synset_mapping(mapping_path):
    """加载 synset_mapping.txt"""
    wnid_to_idx = {}
    with open(mapping_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            wnid = line.split(' ')[0]
            wnid_to_idx[wnid] = idx
    return wnid_to_idx

# 定义 ImageNet-C 测试数据集类 (与 Exp1 一致)
class ImageNetCDataset(Dataset):
    def __init__(self, root_dir, corruption, mapping_path, max_images=None):
        self.root_dir = root_dir
        self.corruption = corruption
        if corruption == 'origin':
            self.images_dir = os.path.join(root_dir, 'origin')
        else:
            self.images_dir = os.path.join(root_dir, corruption)
            
        self.clean_root = os.path.join(root_dir, 'origin')
        self.wnid_mapping = load_synset_mapping(mapping_path)
        self.to_tensor = ToTensor()
        
        # 收集图像
        self.image_paths = []
        if os.path.exists(self.images_dir):
            subdirs = sorted(os.listdir(self.images_dir))
            for cls in subdirs:
                cls_dir = os.path.join(self.images_dir, cls)
                if not os.path.isdir(cls_dir): continue
                
                class_idx = self.wnid_mapping.get(cls, -1)
                
                fnames = sorted(os.listdir(cls_dir))
                for fname in fnames:
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.image_paths.append({
                            'path': os.path.join(cls_dir, fname),
                            'class': cls,
                            'filename': fname,
                            'label': class_idx
                        })
        else:
             print(f"警告: 目录不存在 {self.images_dir}")

        # 数量限制
        if max_images is not None and len(self.image_paths) > max_images:
            print(f"[{corruption}] 限制测试图像数量: {len(self.image_paths)} -> {max_images}")
            self.image_paths = self.image_paths[:max_images]
        else:
             print(f"[{corruption}] 加载了 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = self.image_paths[idx]
        degraded_path = item['path']
        clean_path = os.path.join(self.clean_root, item['class'], item['filename'])
        
        try:
            degraded_img = Image.open(degraded_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
        except Exception as e:
            print(f"读取失败: {e}")
            degraded_img = Image.new('RGB', (224, 224))
            clean_img = Image.new('RGB', (224, 224))

        # 裁剪对齐
        degraded_np = crop_img(np.array(degraded_img), base=16)
        clean_np = crop_img(np.array(clean_img), base=16)
        
        return item['filename'], self.to_tensor(degraded_np), self.to_tensor(clean_np), item['label']

class Exp3System(nn.Module):
    """
    Exp3 完整系统封装:
    Image -> VGG提取 -> Mamba增强 -> Decoder解码 -> Image
    """
    def __init__(self, mamba_ckpt, decoder_ckpt):
        super().__init__()
        
        # 1. 特征提取 (固定)
        self.vgg = VGGPerceptualLossExtractor(requires_grad=False)
        self.vgg.eval()
        
        # 2. Mamba 增强模块 (Exp2)
        self.enhancer = MambaFeatureEnhancer(in_channels=128, img_channels=3, dims=[128, 64, 32], blocks_per_group=4)
        print(f"正在加载 Mamba 增强模型: {mamba_ckpt}")
        self._load_ckpt(self.enhancer, mamba_ckpt, allowed_prefixes=['net.'])
        self.enhancer.eval()
        
        # 3. 特征解码器 (Exp3) - 使用 Pixel Shuffle 版本
        self.decoder = FeatureDecoder(in_channels=128, out_channels=3, num_res_blocks=4)
        print(f"正在加载 Decoder 模型: {decoder_ckpt}")
        self._load_ckpt(self.decoder, decoder_ckpt, allowed_prefixes=['decoder.'])
        self.decoder.eval()
        
        # 归一化参数
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _load_ckpt(self, model, path, allowed_prefixes=[]):
        checkpoint = torch.load(path, map_location='cpu')
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
            
        new_sd = {}
        for k, v in state_dict.items():
            # 尝试去除前缀
            for prefix in allowed_prefixes:
                if k.startswith(prefix):
                    k = k[len(prefix):]
                    break
            new_sd[k] = v
            
        keys = model.load_state_dict(new_sd, strict=False)
        print(f"加载权重结果: {keys}")

    def normalize(self, x):
        return (x - self.mean) / self.std

    def forward(self, img_degrad):
        # 1. 预处理
        norm_img = self.normalize(img_degrad)
        
        # 2. 提取降质特征 (Block 1)
        with torch.no_grad():
            feat_d = self.vgg.block1(norm_img)
            
        # 3. 特征增强 (输入: 降质特征 + 降质图像)
        feat_enhanced = self.enhancer(feat_d, norm_img)
        
        # 4. 解码特征 -> 图像
        img_restored = self.decoder(feat_enhanced)
        
        return img_restored

def test_corruption(sys_model, classifier, dataset_root, corruption, device, mapping_path, batch_size=1, max_images=None):
    """测试单个腐蚀类型"""
    dataset = ImageNetCDataset(dataset_root, corruption, mapping_path, max_images=max_images)
    if len(dataset) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    acc_meter = AverageMeter()      # Enhanced Accuracy
    orig_acc_meter = AverageMeter() # Original Accuracy
    
    normalizer = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    with torch.no_grad():
        for filename, degraded, clean, label in tqdm(dataloader, desc=f"Testing {corruption}", leave=False):
            degraded = degraded.to(device)
            clean = clean.to(device)
            label = label.to(device)
            
            # --- 1. 图像恢复 (Exp3 System) ---
            restored = sys_model(degraded)
            restored = torch.clamp(restored, 0.0, 1.0)
            
            # --- 2. 计算质量指标 (PSNR/SSIM) ---
            p, s, n = compute_metrics_safe(restored, clean)
            psnr_meter.update(p, n)
            ssim_meter.update(s, n)
            
            # --- 3. 计算分类准确率 (Enhanced) ---
            classifier_input = torch.stack([normalizer(img) for img in restored])
            logits = classifier(classifier_input)
            _, preds = torch.max(logits, 1)
            
            correct = (preds == label).float().sum()
            acc_meter.update(correct.item() / n, n)
            
            # --- 4. 计算原始分类准确率 (Original) ---
            classifier_input_orig = torch.stack([normalizer(img) for img in degraded])
            logits_orig = classifier(classifier_input_orig)
            _, preds_orig = torch.max(logits_orig, 1)
            
            correct_orig = (preds_orig == label).float().sum()
            orig_acc_meter.update(correct_orig.item() / n, n)
            
    return psnr_meter.avg, ssim_meter.avg, acc_meter.avg * 100.0, orig_acc_meter.avg * 100.0

def main():
    parser = argparse.ArgumentParser(description="在 ImageNet-C 上测试 Exp3 模型 (Feature Inner Enhancement)")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--max_images', type=int, default=None, help='每个腐蚀类型的最大测试图像数')
    parser.add_argument('--mamba_ckpt', type=str, required=True, help='Exp2 Mamba Enhancer 权重路径')
    parser.add_argument('--decoder_ckpt', type=str, required=True, help='Exp3 Decoder 权重路径')
    args = parser.parse_args()
    
    # 路径设置
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dataset_root = os.path.join(project_root, 'datasets', 'ImageNet-C')
    mapping_path = os.path.join(dataset_root, 'synset_mapping.txt')
    
    # 初始化日志
    logger = setup_logger(os.path.join(os.path.dirname(__file__), 'results'))
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    if not os.path.exists(mapping_path):
        logger.error(f"错误: 找不到映射文件 {mapping_path}")
        return
        
    # 加载完整模型
    try:
        sys_model = Exp3System(args.mamba_ckpt, args.decoder_ckpt)
        sys_model.to(device)
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        return

    # 加载分类器
    logger.info("正在加载预训练 VGG16 分类器...")
    classifier = vgg16(weights=VGG16_Weights.DEFAULT)
    classifier.to(device)
    classifier.eval()
    
    corruptions = ['contrast', 'fog', 'motion_blur', 'snow', 'brightness', 'origin']
    
    logger.info("\n开始测试 (PSNR, SSIM, Top-1 Acc)...")
    if args.max_images:
        logger.info(f"注: 使用 max_images={args.max_images} 进行测试")
    
    logger.info("-" * 80)
    logger.info(f"{'Corruption':<15} {'PSNR':<10} {'SSIM':<10} {'Orig Acc%':<12} {'Enh Acc%':<12}")
    logger.info("-" * 80)
    
    results = {}
    
    for corr in corruptions:
        p, s, a_enh, a_orig = test_corruption(sys_model, classifier, dataset_root, corr, device, mapping_path, args.batch_size, args.max_images)
        logger.info(f"{corr:<15} {p:.4f}     {s:.4f}     {a_orig:<12.2f} {a_enh:<12.2f}")
        results[corr] = (p, s, a_enh, a_orig)
        
    logger.info("-" * 80)
    logger.info("测试完成。")
    
    # --- 结果总结 ---
    logger.info("\n" + "=" * 60)
    logger.info("测试结果总结 (Summary)")
    logger.info("=" * 60)
    
    logger.info("-" * 90)
    logger.info(f"{'Corruption':<15} {'Orig AvgErr':<15} {'Model AvgErr':<15}")
    logger.info("-" * 90)
    
    valid_corruptions = 0
    total_unnormalized_error = 0
    total_orig_error = 0
    
    for corr in corruptions:
        if corr == 'origin': continue
        
        if corr in results:
            avg_err_enh = 100.0 - results[corr][2]
            avg_err_orig = 100.0 - results[corr][3]
            
            total_unnormalized_error += avg_err_enh
            total_orig_error += avg_err_orig
            valid_corruptions += 1
            
            logger.info(f"{corr:<15} {avg_err_orig:<15.2f} {avg_err_enh:<15.2f}")
    
    logger.info("-" * 60)
    
    if valid_corruptions > 0:
        mean_err_abs = total_unnormalized_error / valid_corruptions
        logger.info(f"Absolute Mean Error:            {mean_err_abs:.2f}%")
        
    logger.info("=" * 60)
    logger.info(f"详细结果已保存至日志文件。")

if __name__ == '__main__':
    main()
