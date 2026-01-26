import os
import random
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.transforms import ToPILImage, Compose, RandomCrop, ToTensor

from utils.image_utils import random_augmentation, crop_img

class CUBCDataset(Dataset):
    def __init__(self, root_dir, patch_size=128, mode='train'):
        super(CUBCDataset, self).__init__()
        self.root_dir = root_dir
        self.patch_size = patch_size
        self.mode = mode
        
        self.origin_dir = os.path.join(root_dir, 'origin')
        self.degrad_types = ['snow', 'contrast', 'brightness', 'motion_blur', 'fog']
        
        # Collect all image relative paths
        self.image_paths = []
        if os.path.exists(self.origin_dir):
            for category in os.listdir(self.origin_dir):
                cat_path = os.path.join(self.origin_dir, category)
                if os.path.isdir(cat_path):
                    for img_name in os.listdir(cat_path):
                        if img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                            self.image_paths.append(os.path.join(category, img_name))
        else:
            print(f"警告: 在 {self.origin_dir} 未找到原始图像目录")

        print(f"共找到清晰图像: {len(self.image_paths)} 张")

        self.crop_transform = Compose([
            ToPILImage(),
            RandomCrop(patch_size),
        ])
        self.toTensor = ToTensor()

    def __len__(self):
        # We can define epoch length as (num_files * num_degradations) or just num_files
        # To make it robust, we'll iterate over all files, and randomly pick degradation each time
        return len(self.image_paths)

    def _crop_patch(self, img_1, img_2):
        H = img_1.shape[0]
        W = img_1.shape[1]
        
        # Check if image is smaller than patch size
        if H < self.patch_size or W < self.patch_size:
            # Resize or pad if needed, but for now let's just resize to patch_size if smaller
            # Or better, random crop with smaller size if image is small?
            # Ideally, we skip small images or resize them up. 
            # Assuming CUB images are generally large enough (>128).
            # If not, let's just resize the smaller edge to patch_size
            pass 

        ind_H = random.randint(0, max(0, H - self.patch_size))
        ind_W = random.randint(0, max(0, W - self.patch_size))

        patch_1 = img_1[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]
        patch_2 = img_2[ind_H:ind_H + self.patch_size, ind_W:ind_W + self.patch_size]

        return patch_1, patch_2

    def __getitem__(self, idx):
        rel_path = self.image_paths[idx]
        
        # Randomly select a degradation type
        de_type = random.choice(self.degrad_types)
        
        clean_path = os.path.join(self.root_dir, 'origin', rel_path)
        degrad_path = os.path.join(self.root_dir, de_type, rel_path)
        
        # Load images
        # Use crop_img with base=1 to just read as array, or base=16 if model requires multiple of 16
        # The original code used base=16, let's stick to it.
        try:
            clean_img = crop_img(np.array(Image.open(clean_path).convert('RGB')), base=16)
            degrad_img = crop_img(np.array(Image.open(degrad_path).convert('RGB')), base=16)
        except Exception as e:
            print(f"加载图像出错 {clean_path} 或 {degrad_path}: {e}")
            # Return a random other image to avoid crashing
            return self.__getitem__(random.randint(0, len(self) - 1))

        # Crop patches
        degrad_patch, clean_patch = self._crop_patch(degrad_img, clean_img)
        
        # Augmentation
        degrad_patch, clean_patch = random_augmentation(degrad_patch, clean_patch)
        
        # Convert to Tensor
        clean_patch = self.toTensor(clean_patch)
        degrad_patch = self.toTensor(degrad_patch)
        
        # Return format matching original: ([clean_name, de_id], degrad_patch, clean_patch)
        # We'll use a dummy de_id since model doesn't use it, but train loop expects it unpacked
        clean_name = os.path.basename(rel_path).split('.')[0]
        de_id = 0 # Dummy
        
        return [clean_name, de_id], degrad_patch, clean_patch
