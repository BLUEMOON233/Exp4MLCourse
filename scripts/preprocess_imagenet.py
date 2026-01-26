import os
from PIL import Image
try:
    # Pillow >= 10
    Resample = Image.Resampling.BILINEAR
except AttributeError:
    # Pillow < 10
    Resample = Image.BILINEAR

# 目标目录 (自动修正用户输入 orgin -> origin)
TARGET_DIR = os.path.join("datasets", "ImageNet-C", "origin")

def process_image(img_path):
    try:
        # 打开图片
        with Image.open(img_path) as img:
            # 转换为 RGB 防止 RGBA 保存为 JPG 出错
            img = img.convert('RGB')
            w, h = img.size
            
            # 1. Resize (等比缩放短边至 256)
            if w < h:
                new_w = 256
                new_h = int(h * (256 / w))
            else:
                new_h = 256
                new_w = int(w * (256/h))
            
            # 使用双线性插值进行缩放 (标准 ImageNet 预处理)
            img = img.resize((new_w, new_h), Resample)
            
            # 2. Center Crop (中心裁剪至 224x224)
            left = (new_w - 224) // 2
            top = (new_h - 224) // 2
            right = left + 224
            bottom = top + 224
            
            img = img.crop((left, top, right, bottom))
            
            # 3. Save (直接覆盖原文件)
            # 使用 quality=95 尽可能保留质量，虽然对于数据集修改来说已经是有损的
            img.save(img_path, quality=95)
            return True
    except Exception as e:
        print(f"Error processing {img_path}: {e}")
        return False

def main():
    abs_target_dir = os.path.abspath(TARGET_DIR)
    
    if not os.path.exists(abs_target_dir):
        print(f"Error: Directory not found: {abs_target_dir}")
        print("Please check the path.")
        return

    print(f"Target Directory: {abs_target_dir}")
    print("Starting processing (Resize 256 -> CenterCrop 224)...")
    
    count = 0
    errors = 0
    
    # 递归遍历目录
    for root, dirs, files in os.walk(abs_target_dir):
        for file in files:
            # 检查常见的图片扩展名
            if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')):
                full_path = os.path.join(root, file)
                if process_image(full_path):
                    count += 1
                    if count % 100 == 0:
                        print(f"Processed {count} images...", end='\r')
                else:
                    errors += 1
                    
    print(f"\nCompleted. Processed {count} images. {errors} errors.")

if __name__ == "__main__":
    main()
