import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, Resize, ToTensor

from .image_utils import random_augmentation

class CUBCDataset(Dataset):
    def __init__(self, root_dir, mode='train'):
        """
        CUBC 数据集加载器
        :param root_dir: 数据集根目录
        :param mode: 模式 (train/val)
        """
        super(CUBCDataset, self).__init__()
        self.root_dir = root_dir
        self.mode = mode
        
        self.origin_dir = os.path.join(root_dir, 'origin')
        self.degrad_types = ['snow', 'contrast', 'brightness', 'motion_blur', 'fog']
        
        # 收集所有图片的相对路径
        self.image_paths = []
        if os.path.exists(self.origin_dir):
            for category in os.listdir(self.origin_dir):
                cat_path = os.path.join(self.origin_dir, category)
                if os.path.isdir(cat_path):
                    for img_name in os.listdir(cat_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(category, img_name))
        else:
            print(f"警告: 未找到原始图像目录 {self.origin_dir}")

        print(f"共找到清晰图像: {len(self.image_paths)} 张")

        # 定义变换：强制调整大小为 224x224 并转为 Tensor
        self.transform = Compose([
            ToPILImage(),
            Resize((224, 224)),
            ToTensor()
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        
        # 混合数据训练策略: 20% 概率使用清晰图像作为输入 (Identity Mapping)
        # 这有助于模型学习恒等映射，解决清晰图像掉点问题，并提供正确的残差零基准
        if self.mode == 'train' and random.random() < 0.2:
            clean_path = os.path.join(self.root_dir, 'origin', rel_path)
            degrad_path = clean_path
            de_type = 'origin'
        else:
            # 正常降质流程
            de_type = random.choice(self.degrad_types)
            clean_path = os.path.join(self.root_dir, 'origin', rel_path)
            degrad_path = os.path.join(self.root_dir, de_type, rel_path)
        
        try:
            # 读取图片
            clean_img_pil = Image.open(clean_path).convert('RGB')
            degrad_img_pil = Image.open(degrad_path).convert('RGB')
        except Exception as e:
            print(f"读取图像出错 {clean_path}: {e}")
            # 随机重试另一张图
            return self.__getitem__(random.randint(0, len(self) - 1))

        # 转换为 Numpy 数组以便应用 random_augmentation (如果包含颜色抖动等，目前 random_augmentation 主要是翻转/旋转)
        clean_np = np.array(clean_img_pil)
        degrad_np = np.array(degrad_img_pil)
        
        # 数据增强 (随机翻转/旋转)
        # 注意：这里我们传入的是整图，但 random_augmentation 是通用的，只要是 numpy 数组即可
        degrad_np, clean_np = random_augmentation(degrad_np, clean_np)
        
        # 转换为 Tensor 并且 Resize
        # 注意: random_augmentation 返回的是 numpy，我们需要重新变回 PIL 或直接处理
        # 为了使用 torchvision 的 Resize，我们先转回 PIL 或使用 transform 的 ToPILImage
        
        clean_tensor = self.transform(clean_np)
        degrad_tensor = self.transform(degrad_np)
        
        clean_name = os.path.basename(rel_path).split('.')[0]
        de_id = 0 
        
        return [clean_name, de_id], degrad_tensor, clean_tensor
