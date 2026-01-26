# -*- coding: utf-8 -*-
"""
ImageNet-C 数据集加载器
用于从 ImageNet-C/origin 目录加载清晰图像训练 Decoder
"""
import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import Compose, ToTensor, RandomHorizontalFlip

class ImageNetOriginDataset(Dataset):
    """
    ImageNet-C Origin 数据集加载器
    仅加载清晰图像用于训练 Decoder（特征反演任务）
    """
    def __init__(self, root_dir, mode='train', image_size=224, max_images=None):
        """
        Args:
            root_dir: ImageNet-C 数据集根目录 (包含 origin 子目录)
            mode: 'train' 或 'val'
            image_size: 输出图像大小
            max_images: 最大图像数量限制 (用于快速测试)
        """
        super(ImageNetOriginDataset, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        
        # origin 目录路径
        self.origin_dir = os.path.join(root_dir, 'origin')
        
        # 收集所有图片路径
        self.image_paths = []
        if os.path.exists(self.origin_dir):
            categories = sorted(os.listdir(self.origin_dir))
            for category in categories:
                cat_path = os.path.join(self.origin_dir, category)
                if os.path.isdir(cat_path):
                    for img_name in sorted(os.listdir(cat_path)):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPEG')):
                            self.image_paths.append(os.path.join(cat_path, img_name))
        else:
            print(f"警告: 未找到 origin 图像目录 {self.origin_dir}")

        # 数量限制
        if max_images is not None and len(self.image_paths) > max_images:
            print(f"限制图像数量: {len(self.image_paths)} -> {max_images}")
            # 随机采样以保证多样性
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_images)
        
        print(f"[ImageNetOriginDataset] 加载了 {len(self.image_paths)} 张清晰图像")

        # 定义变换
        # ImageNet-C 图片已经是 224×224，无需 Resize 和 Crop
        if mode == 'train':
            self.transform = Compose([
                RandomHorizontalFlip(p=0.5),    # 随机水平翻转 (数据增强)
                ToTensor()
            ])
        else:
            self.transform = Compose([
                ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        try:
            # 读取图片
            img_pil = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"读取图像出错 {img_path}: {e}")
            # 随机重试另一张图
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 应用变换
        img_tensor = self.transform(img_pil)
        
        # 返回格式与 CUBCDataset 兼容: [info, degrad, clean]
        # 对于 Decoder 训练，我们只需要 clean
        # degrad 和 clean 相同（因为只用 origin）
        img_name = os.path.basename(img_path).split('.')[0]
        
        return [img_name, 0], img_tensor, img_tensor


class ImageNetCDataset(Dataset):
    """
    ImageNet-C 完整数据集加载器
    加载降质图像和对应的清晰图像
    用于训练需要 (降质, 清晰) 图像对的模型
    """
    def __init__(self, root_dir, mode='train', image_size=224, max_images=None,
                 degrad_types=['contrast', 'fog', 'motion_blur', 'snow', 'brightness']):
        """
        Args:
            root_dir: ImageNet-C 数据集根目录
            mode: 'train' 或 'val'
            image_size: 输出图像大小
            max_images: 最大图像数量限制
            degrad_types: 要使用的降质类型列表
        """
        super(ImageNetCDataset, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        self.image_size = image_size
        self.degrad_types = degrad_types
        
        # origin 目录路径
        self.origin_dir = os.path.join(root_dir, 'origin')
        
        # 收集所有图片的相对路径 (类别/文件名)
        self.image_paths = []
        if os.path.exists(self.origin_dir):
            categories = sorted(os.listdir(self.origin_dir))
            for category in categories:
                cat_path = os.path.join(self.origin_dir, category)
                if os.path.isdir(cat_path):
                    for img_name in sorted(os.listdir(cat_path)):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.JPEG')):
                            # 保存相对路径: category/img_name
                            self.image_paths.append(os.path.join(category, img_name))
        else:
            print(f"警告: 未找到 origin 图像目录 {self.origin_dir}")

        # 数量限制
        if max_images is not None and len(self.image_paths) > max_images:
            print(f"限制图像数量: {len(self.image_paths)} -> {max_images}")
            random.seed(42)
            self.image_paths = random.sample(self.image_paths, max_images)
        
        print(f"[ImageNetCDataset] 加载了 {len(self.image_paths)} 张图像, 降质类型: {degrad_types}")

        # 定义变换
        # ImageNet-C 图片已经是 224×224，无需 Resize 和 Crop
        if mode == 'train':
            self.transform = Compose([
                RandomHorizontalFlip(p=0.5),    # 随机水平翻转
                ToTensor()
            ])
        else:
            self.transform = Compose([
                ToTensor()
            ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        
        # 随机选择一种降质类型
        de_type = random.choice(self.degrad_types)
        
        clean_path = os.path.join(self.origin_dir, rel_path)
        degrad_path = os.path.join(self.root_dir, de_type, rel_path)
        
        try:
            clean_img_pil = Image.open(clean_path).convert('RGB')
            degrad_img_pil = Image.open(degrad_path).convert('RGB')
        except Exception as e:
            print(f"读取图像出错 {clean_path} 或 {degrad_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 为了保证 clean 和 degrad 使用相同的随机翻转
        # 需要手动控制随机种子
        seed = random.randint(0, 2**32)
        
        random.seed(seed)
        torch.manual_seed(seed)
        clean_tensor = self.transform(clean_img_pil)
        
        random.seed(seed)
        torch.manual_seed(seed)
        degrad_tensor = self.transform(degrad_img_pil)
        
        img_name = os.path.basename(rel_path).split('.')[0]
        
        return [img_name, de_type], degrad_tensor, clean_tensor
