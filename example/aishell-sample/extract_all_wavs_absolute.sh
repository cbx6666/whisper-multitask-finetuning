#!/bin/bash

# --- 配置 ---
# 源目录，包含 Sxxxx.tar.gz 文件
# 假设此脚本在 /mnt/d/WSL/asr_large_model/whisper-multitask-finetuning/example/aishell-sample/ 目录下运行
SOURCE_ARCHIVE_DIR="data_aishell/wav"

# 目标目录，用于存放所有提取出的 .wav 文件 (使用你提供的绝对路径)
TARGET_WAV_DIR_ABSOLUTE="/mnt/d/WSL/asr_large_model/whisper-multitask-finetuning/example/aishell-sample/data/my_wav"

# --- 主逻辑 ---
START_DIR=$(pwd) # 保存脚本开始时的工作目录

echo "脚本开始执行，当前工作目录: $START_DIR"
echo "源压缩包相对路径: $SOURCE_ARCHIVE_DIR"
echo "目标WAV文件绝对路径: $TARGET_WAV_DIR_ABSOLUTE"

# 检查源目录是否存在 (相对于当前工作目录)
if [ ! -d "$SOURCE_ARCHIVE_DIR" ]; then
  echo "错误：源压缩包目录 '$SOURCE_ARCHIVE_DIR' (在 '$START_DIR' 下) 不存在。"
  echo "请确保你位于 'aishell-sample' 目录下运行此脚本，"
  echo "并且 '$SOURCE_ARCHIVE_DIR' 路径正确 (即 './data_aishell/wav/')"
  exit 1
fi

# 创建目标目录，如果它不存在 (-p 会创建必要的父目录)
mkdir -p "$TARGET_WAV_DIR_ABSOLUTE"
if [ $? -ne 0 ]; then
    echo "错误：无法创建目标目录 '$TARGET_WAV_DIR_ABSOLUTE'。请检查权限或路径。"
    exit 1
fi
echo "确保目标目录 '$TARGET_WAV_DIR_ABSOLUTE' 已创建或已存在。"

# 准备进入源压缩包目录
echo "准备进入源目录: $START_DIR/$SOURCE_ARCHIVE_DIR"
cd "$SOURCE_ARCHIVE_DIR" || { echo "错误：无法进入目录 '$START_DIR/$SOURCE_ARCHIVE_DIR'"; exit 1; }

CURRENT_PROCESSING_DIR=$(pwd)
echo "当前处理工作目录: $CURRENT_PROCESSING_DIR"
echo "将从这里的 .tar.gz 文件中提取 .wav 文件到 '$TARGET_WAV_DIR_ABSOLUTE'"

# 遍历所有 Sxxxx.tar.gz 文件
# shopt -s nullglob 确保在没有匹配文件时循环不执行
shopt -s nullglob
archive_files=(S*.tar.gz)
shopt -u nullglob #恢复默认行为

if [ ${#archive_files[@]} -eq 0 ]; then
    echo "在 '$CURRENT_PROCESSING_DIR' 中没有找到 S*.tar.gz 文件。"
    cd "$START_DIR" # 在退出前返回到开始目录
    exit 0
fi

for archive_file in "${archive_files[@]}"; do
  echo "------------------------------------"
  echo "正在处理压缩包: $archive_file"

  # 为当前压缩包创建一个临时解压目录 (基于压缩包名，去掉 .tar.gz)
  # temp_extract_dir 是相对于 CURRENT_PROCESSING_DIR (即 data_aishell/wav/) 的
  temp_extract_dir="${archive_file%.tar.gz}_temp_extract"
  mkdir -p "$temp_extract_dir"

  echo "解压 '$archive_file' 到临时目录 './$temp_extract_dir'..."
  tar -zxvf "$archive_file" -C "$temp_extract_dir" # -v 显示解压过程
  if [ $? -ne 0 ]; then
      echo "警告：解压 '$archive_file' 失败。跳过此压缩包。"
      # 清理可能部分创建的临时目录
      rm -rf "$temp_extract_dir" 
      continue
  fi

  echo "在 './$temp_extract_dir' 中查找并移动 .wav 文件到 '$TARGET_WAV_DIR_ABSOLUTE'..."
  # 查找临时解压目录中的所有 .wav 文件 (不区分路径深度和文件名大小写)
  # 并将它们移动到最终的目标目录 (absolute path)
  # 使用 -print0 和 while read 循环以安全处理可能包含特殊字符的文件名
  find "$temp_extract_dir" -type f -iname "*.wav" -print0 | while IFS= read -r -d $'\0' wav_file_path_in_temp; do
    # wav_file_path_in_temp 是类似 'S0002_temp_extract/S0002/BAC009S0002W0122.wav' 的路径
    # base_wav_name=$(basename "$wav_file_path_in_temp") # 获取纯文件名，用于日志或冲突处理
    echo "  找到: $wav_file_path_in_temp -> 准备移动到 $TARGET_WAV_DIR_ABSOLUTE/"
    mv -v "$wav_file_path_in_temp" "$TARGET_WAV_DIR_ABSOLUTE/" # mv 到绝对路径的目标目录
    if [ $? -ne 0 ]; then
        echo "  警告：移动 '$wav_file_path_in_temp' 失败。"
    fi
  done

  echo "清理临时目录 './$temp_extract_dir'..."
  rm -rf "$temp_extract_dir"

  # 可选：如果确认提取成功，可以删除原始的 Sxxxx.tar.gz 压缩包
  # 注意：取消注释前请务必确认脚本行为符合预期
  # echo "可选：删除原始压缩包 '$archive_file'..."
  # rm "$archive_file"
done

echo "------------------------------------"
echo "所有 .wav 文件提取操作完成。"
echo "请检查目录 '$TARGET_WAV_DIR_ABSOLUTE'"

# 返回到脚本开始时的目录
cd "$START_DIR"
echo "已返回到初始目录: $(pwd)"
