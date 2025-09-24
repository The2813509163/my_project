#!/bin/bash

# 设置源目录和目标目录
source_dir="/home/kris/workspace/qianxuzhen/Pruning-LLMs/LLaMA-Factory-main/saves"
target_dir="/data/kris/qianxuzhen/Pruning-LLMs/LLaMA-Factory-main/saves"

# 确保目标目录存在，如果不存在则创建
mkdir -p "$target_dir"

# 使用 find 命令查找源目录下的所有文件
find "$source_dir" -type f -print0 | while IFS= read -r -d $'\0' file; do
  # 获取文件相对于源目录的路径
  relative_path=$(echo "$file" | sed "s#^$(realpath "$source_dir")/##")

  # 构建目标路径
  target_path="$target_dir/$relative_path"

  # 创建目标目录（如果不存在）
  mkdir -p "$(dirname "$target_path")"

  # 移动文件到目标路径
  mv "$file" "$target_path"
done

echo "文件移动完成。"