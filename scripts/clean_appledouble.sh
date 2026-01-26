#!/bin/bash
# 递归查找并删除当前目录及子目录下所有以 ._ 开头的文件
# 这些通常是 macOS 在不支持扩展属性的文件系统上创建的 AppleDouble 文件

echo "正在扫描并删除 ._ 文件..."
find . -type f -name "._*" -print -delete
echo "清理完成。"
