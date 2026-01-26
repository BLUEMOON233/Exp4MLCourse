
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision.models import vgg16, VGG16_Weights
from PIL import Image
from tqdm import tqdm
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import warnings

# 忽略 sigmoid 警告
warnings.filterwarnings("ignore", category=UserWarning)

# 将 AdaIR 目录添加到 Python 路径，以便导入模型
sys.path.append(os.path.join(os.path.dirname(__file__), 'AdaIR'))

from net.model import AdaIR
from utils.val_utils import AverageMeter
from utils.image_utils import crop_img

# ImageNet 标准归一化 (用于分类器)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

def compute_metrics_safe(recoverd, clean):
    """
    计算 PSNR 和 SSIM，兼容不同版本的 skimage
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
            # 尝试新版本参数 (scikit-image >= 0.19)
            ssim += structural_similarity(clean[i], recoverd[i], data_range=1, channel_axis=-1)
        except TypeError:
            # 回退到旧版本参数
            ssim += structural_similarity(clean[i], recoverd[i], data_range=1, multichannel=True)
            
    return psnr / B, ssim / B, B

def load_synset_mapping(mapping_path):
    """
    加载 synset_mapping.txt，构建 wnid -> class_index 的映射
    ImageNet-C 的目录名是 wnid (例如 n01440764)
    PyTorch ResNet50 使用标准的 1000 类索引
    synset_mapping.txt 的行号 (0-999) 对应 PyTorch 模型的输出索引吗？
    通常 PyTorch 的 ResNet50 权重是对应 ILSVRC2012 的 label 顺序。
    标准的 synset_mapping.txt 顺序即为 class index。
    """
    wnid_to_idx = {}
    with open(mapping_path, 'r') as f:
        lines = f.readlines()
        for idx, line in enumerate(lines):
            # line format: n01440764 tench, Tinca tinca
            wnid = line.split(' ')[0]
            wnid_to_idx[wnid] = idx
    return wnid_to_idx

# 定义 ImageNet-C 测试数据集类
class ImageNetCDataset(Dataset):
    def __init__(self, root_dir, corruption, max_images=None, wnid_mapping=None):
        """
        初始化数据集
        :param root_dir: 数据集根目录 (e.g., .../datasets/ImageNet-C)
        :param corruption: 腐蚀类型 (e.g., 'contrast', 'origin')
        :param max_images: 最大测试图像数量 (用于快速测试)
        :param wnid_mapping: wnid -> class_index 的字典
        """
        self.root_dir = root_dir
        self.corruption = corruption
        self.wnid_mapping = wnid_mapping
        self.to_tensor = ToTensor()

        # 确定图像目录
        # 直接使用 corruption 目录名称，预期已是扁平化结构
        if corruption == 'origin':
            self.images_dir = os.path.join(root_dir, 'origin')
        else:
            self.images_dir = os.path.join(root_dir, corruption)
            
        self.clean_root = os.path.join(root_dir, 'origin')
        
        # 收集所有图像路径
        self.image_paths = []
        # 遍历类别目录
        if not os.path.exists(self.images_dir):
             print(f"警告: 目录不存在 {self.images_dir}")
        else:
            # 获取类别列表
            try:
                # 检查是否包含子目录
                subdirs = sorted(os.listdir(self.images_dir))
                
                # 简单检查一下是否意外包含了分级目录 '1', '2' 等 (虽然用户说已经扁平化了)
                if '1' in subdirs and os.path.isdir(os.path.join(self.images_dir, '1')):
                    print(f"警告: [{corruption}] 目录下似乎包含 '1' 等分级目录，但代码期望扁平化结构。")
                    print("将尝试直接遍历所有子目录，如果 '1' 是类别名则无影响，如果是分级目录则可能找不到图片。")

                for cls in subdirs:
                    # 确保是有效的 wnid (目录)
                    cls_dir = os.path.join(self.images_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue
                    
                    # 获取该类的 index
                    class_idx = -1
                    if wnid_mapping and cls in wnid_mapping:
                        class_idx = wnid_mapping[cls]
                    
                    # 获取图像文件
                    fnames = sorted(os.listdir(cls_dir))
                    for fname in fnames:
                        if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                            self.image_paths.append({
                                'path': os.path.join(cls_dir, fname),
                                'class': cls,
                                'filename': fname,
                                'label': class_idx
                            })
            except Exception as e:
                print(f"遍历目录出错: {e}")

        # 应用数量限制
        if max_images is not None and len(self.image_paths) > max_images:
            print(f"[{corruption}] 限制图像数量: {len(self.image_paths)} -> {max_images}")
            self.image_paths = self.image_paths[:max_images]
        else:
             print(f"[{corruption}] 加载了 {len(self.image_paths)} 张图像")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        item = self.image_paths[idx]
        degraded_path = item['path']
        
        # 构造 clean 图像路径 (origin)
        clean_path = os.path.join(self.clean_root, item['class'], item['filename'])
        
        # 加载图像
        try:
            degraded_img = Image.open(degraded_path).convert('RGB')
            clean_img = Image.open(clean_path).convert('RGB')
        except Exception as e:
            print(f"读取图像失败: {e}")
            degraded_img = Image.new('RGB', (224, 224))
            clean_img = Image.new('RGB', (224, 224))

        # 裁剪为 16 的倍数 (AdaIR要求)
        degraded_np = np.array(degraded_img)
        clean_np = np.array(clean_img)
        
        # 使用 utils.image_utils 中的 crop_img
        degraded_np = crop_img(degraded_np, base=16)
        clean_np = crop_img(clean_np, base=16)
        
        # 转为 Tensor
        degraded_tensor = self.to_tensor(degraded_np)
        clean_tensor = self.to_tensor(clean_np)
        
        return item['filename'], degraded_tensor, clean_tensor, item['label']

def load_adair_model(ckpt_path, device):
    """加载 AdaIR 模型"""
    print(f"正在加载 AdaIR 模型权重: {ckpt_path}")
    model = AdaIR(decoder=True)
    
    # 加载权重
    checkpoint = torch.load(ckpt_path, map_location='cpu')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
        
    # 处理 'net.' 前缀
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('net.'):
            new_state_dict[k[4:]] = v
        else:
            new_state_dict[k] = v
            
    model.load_state_dict(new_state_dict, strict=True)
    model.to(device)
    model.eval()
    return model

def load_classifier(device):
    """加载预训练的 VGG16 分类器"""
    print("正在加载预训练 VGG16 分类器...")
    # 使用最新的默认权重
    model = vgg16(weights=VGG16_Weights.DEFAULT)
    model.to(device)
    model.eval()
    return model

def test_corruption(adair_model, classifier, root_dir, corruption, device, wnid_mapping, batch_size=1, max_images=None):
    """测试单个腐蚀类型 (包含原始图像分类测试)"""
    # 不再传入 severity
    dataset = ImageNetCDataset(root_dir, corruption, max_images=max_images, wnid_mapping=wnid_mapping)
    if len(dataset) == 0:
        return 0.0, 0.0, 0.0, 0.0
        
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    
    psnr_meter = AverageMeter()
    ssim_meter = AverageMeter()
    acc_meter = AverageMeter()      # Enhanced Accuracy
    orig_acc_meter = AverageMeter() # Original Accuracy
    
    # 定义分类器的预处理 (归一化)
    # 输入已经是 Tensor [0, 1]
    normalizer = Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    
    with torch.no_grad():
        for filename, degraded, clean, label in tqdm(dataloader, desc=f"Testing {corruption}", leave=False):
            degraded = degraded.to(device)
            clean = clean.to(device)
            label = label.to(device)
            
            # --- 1. 图像恢复 (AdaIR) ---
            restored = adair_model(degraded)
            # 裁剪值域
            restored = torch.clamp(restored, 0.0, 1.0)
            
            # --- 2. 计算质量指标 (PSNR/SSIM) ---
            p, s, n = compute_metrics_safe(restored, clean)
            psnr_meter.update(p, n)
            ssim_meter.update(s, n)
            
            # --- 3. 计算分类准确率 (Enhanced) ---
            # ResNet 需要归一化
            # 注意: restored 形状为 (B, C, H, W)
            classifier_input = torch.stack([normalizer(img) for img in restored])
            
            logits = classifier(classifier_input)
            _, preds = torch.max(logits, 1)
            
            correct = (preds == label).float().sum()
            acc = correct / n
            acc_meter.update(acc.item(), n)
            
            # --- 4. 计算原始分类准确率 (Original) ---
            # 直接使用 degraded 图像
            classifier_input_orig = torch.stack([normalizer(img) for img in degraded])
            logits_orig = classifier(classifier_input_orig)
            _, preds_orig = torch.max(logits_orig, 1)
            
            correct_orig = (preds_orig == label).float().sum()
            acc_orig = correct_orig / n
            orig_acc_meter.update(acc_orig.item(), n)
            
    return psnr_meter.avg, ssim_meter.avg, acc_meter.avg * 100.0, orig_acc_meter.avg * 100.0

import logging
import datetime

def setup_logger(output_dir):
    """设置日志记录器，同时输出到控制台和文件"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(output_dir, f"test_results_{timestamp}.txt")
    
    # 配置 logging
    logger = logging.getLogger("AdaIR_Test")
    logger.setLevel(logging.INFO)
    
    # 文件 Handler
    fh = logging.FileHandler(log_file, encoding='utf-8')
    fh.setLevel(logging.INFO)
    
    # 控制台 Handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    
    # 格式
    formatter = logging.Formatter('%(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    
    logger.addHandler(fh)
    logger.addHandler(ch)
    
    return logger

def main():
    parser = argparse.ArgumentParser(description="在 ImageNet-C 上测试 AdaIR 模型 (含分类准确率)")
    parser.add_argument('--gpu', type=int, default=0, help='GPU ID')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch Size')
    parser.add_argument('--max_images', type=int, default=None, help='每个腐蚀类型的最大测试图像数')
    args = parser.parse_args()
    
    # 路径设置
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
    dataset_root = os.path.join(project_root, 'datasets', 'ImageNet-C')
    ckpt_dir = os.path.join(project_root, 'checkpoints', 'exp1_AdaIR_CUBC')
    mapping_path = os.path.join(dataset_root, 'synset_mapping.txt')
    
    # 初始化日志
    logger = setup_logger(os.path.join(os.path.dirname(__file__), 'results'))
    
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    logger.info(f"使用设备: {device}")
    
    # 加载 Label 映射
    if not os.path.exists(mapping_path):
        logger.error(f"错误: 找不到映射文件 {mapping_path}，无法进行分类测试")
        return
    wnid_mapping = load_synset_mapping(mapping_path)
    
    # 查找最佳 checkpoint
    ckpt_path = os.path.join(ckpt_dir, 'best.ckpt')
    if not os.path.exists(ckpt_path):
        logger.info(f"best.ckpt 未找到，尝试查找最后保存的模型...")
        files = [f for f in os.listdir(ckpt_dir) if f.endswith('.ckpt')]
        if not files:
            logger.error("错误: 未找到任何 checkpoint (.ckpt) 文件")
            return
        if 'last.ckpt' in files:
            ckpt_path = os.path.join(ckpt_dir, 'last.ckpt')
        else:
            files.sort()
            ckpt_path = os.path.join(ckpt_dir, files[-1])
            
    # 加载模型
    adair_model = load_adair_model(ckpt_path, device)
    
    # 加载分类器
    classifier = load_classifier(device)
    
    # 定义测试的腐蚀类型
    corruptions = ['contrast', 'fog', 'motion_blur', 'snow', 'brightness', 'origin']
    
    logger.info("\n开始测试 (PSNR, SSIM, Top-1 Acc)...")
    if args.max_images:
        logger.info(f"注: 使用 max_images={args.max_images} 进行测试")
    else:
        logger.info("注: 全量测试 (可能较慢)")
        
    logger.info("-" * 80)
    logger.info(f"{'Corruption':<15} {'PSNR':<10} {'SSIM':<10} {'Orig Acc%':<12} {'Enh Acc%':<12}")
    logger.info("-" * 80)
    
    # structure: results[corruption] = (psnr, ssim, acc_enh, acc_orig)
    results = {}
    
    for corr in corruptions:
        p, s, a_enh, a_orig = test_corruption(adair_model, classifier, dataset_root, corr, device, wnid_mapping, args.batch_size, args.max_images)
        logger.info(f"{corr:<15} {p:.4f}     {s:.4f}     {a_orig:<12.2f} {a_enh:<12.2f}")
        results[corr] = (p, s, a_enh, a_orig)
            
    logger.info("-" * 80)
    logger.info("测试完成。")
    
    # --- 计算总结指标 ---
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
            avg_acc_enh = results[corr][2]
            avg_acc_orig = results[corr][3]
            
            avg_err_enh = 100.0 - avg_acc_enh
            avg_err_orig = 100.0 - avg_acc_orig
            
            total_unnormalized_error += avg_err_enh
            total_orig_error += avg_err_orig
            valid_corruptions += 1
            
            logger.info(f"{corr:<15} {avg_err_orig:<15.2f} {avg_err_enh:<15.2f}")
    
    logger.info("-" * 60)
    
    # 平均错误率
    if valid_corruptions > 0:
        mean_err_abs = total_unnormalized_error / valid_corruptions
        logger.info(f"Absolute Mean Error:            {mean_err_abs:.2f}%")
        
    logger.info("=" * 60)
    logger.info(f"详细结果已保存至日志文件。")

if __name__ == '__main__':
    main()
