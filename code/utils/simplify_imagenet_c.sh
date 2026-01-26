#!/bin/bash

# 设置数据根目录
DATA_ROOT="/home/wardenliu/develop/projects/ML-Course/datasets/ImageNet-C"
# 定义需要处理的降质类型
CORRUPTIONS=("fog" "motion_blur" "brightness" "contrast" "snow")
# 输出目录后缀
SUFFIX="_simple"
# 降质等级列表
LEVELS=(1 2 3 4 5)

echo "开始 ImageNet-C 数据集降质等级随机采样简化任务..."
echo "数据根目录: $DATA_ROOT"

for corruption in "${CORRUPTIONS[@]}"; do
    src_base="$DATA_ROOT/$corruption"
    dst_dir="$DATA_ROOT/${corruption}${SUFFIX}"
    ref_dir="$src_base/1"

    echo "--------------------------------------------------"
    echo "正在处理降质类型: $corruption"
    
    # 检查源目录是否存在
    if [ ! -d "$src_base" ]; then
        echo "警告: 源目录 $src_base 不存在，跳过。"
        continue
    fi

    # 检查 Level 1 目录是否存在（用于获取文件列表）
    if [ ! -d "$ref_dir" ]; then
        echo "警告: 参考目录 $ref_dir 不存在，无法获取文件列表，跳过。"
        continue
    fi

    # 清理并创建目标目录
    if [ -d "$dst_dir" ]; then
        echo "目录 $dst_dir 已存在，正在清理..."
        rm -rf "$dst_dir"
    fi
    mkdir -p "$dst_dir"

    echo "正在扫描图片文件..."
    # 获取 Level 1 下所有 JPEG/PNG 图片的相对路径 (例如: n01440764/ILSVRC2012_val_00000293.JPEG)
    # 使用 cd 进入目录以获取相对路径
    # 使用 find 查找所有 .JPEG, .jpg, .png 文件
    files=$(cd "$ref_dir" && find . -type f \( -name "*.JPEG" -o -name "*.jpg" -o -name "*.png" \))
    
    # 计算文件总数
    total_files=$(echo "$files" | wc -l | xargs)
    echo "找到 $total_files 张图片，开始随机采样复制..."
    
    count=0
    
    # 设置字段分隔符为换行，以处理文件名中可能包含的空格（虽然ImageNet通常没有）
    IFS=$'\n'
    for rel_path in $files; do
        # 去掉路径开头的 ./
        rel_path="${rel_path#./}"
        
        # 随机选择一个等级 1-5
        # $RANDOM % 5 生成 0-4，+1 得到 1-5
        rand_idx=$((RANDOM % 5))
        level=${LEVELS[$rand_idx]}
        
        # 构造源文件路径和目标文件路径
        src_file="$src_base/$level/$rel_path"
        dst_file="$dst_dir/$rel_path"
        
        # 确保目标文件的父目录存在
        dst_subdir=$(dirname "$dst_file")
        if [ ! -d "$dst_subdir" ]; then
            mkdir -p "$dst_subdir"
        fi
        
        # 复制文件
        if [ -f "$src_file" ]; then
            cp "$src_file" "$dst_file"
        else
            echo "错误: 文件缺失 $src_file"
        fi
        
        # 进度显示
        ((count++))
        if ((count % 1000 == 0)); then
            echo "已处理 $count / $total_files ..."
        fi
    done
    unset IFS

    echo "完成 $corruption 的处理。结果保存在: $dst_dir"
done

echo "--------------------------------------------------"
echo "所有任务已完成。"
