#!/bin/bash

# --- 配置 ---
# 包含 .wav 文件的目录 (绝对路径)
WAV_FILES_DIR="/mnt/d/WSL/asr_large_model/whisper-multitask-finetuning/example/aishell-sample/data/wav"

# 输出 wav.scp 文件的路径 (绝对路径)
OUTPUT_WAV_SCP="/mnt/d/WSL/asr_large_model/whisper-multitask-finetuning/example/aishell-sample/data/wav.scp"

# 固定语言和任务类型
LANGUAGE="chinese"
TASK_TRANSCRIBE="transcribe"
TASK_TRANSLATE="translate"

# --- 主逻辑 ---

echo "开始生成 wav.scp 文件..."
echo "WAV 文件来源目录: $WAV_FILES_DIR"
echo "输出文件路径: $OUTPUT_WAV_SCP"

# 检查 WAV 文件目录是否存在
if [ ! -d "$WAV_FILES_DIR" ]; then
  echo "错误：WAV 文件目录 '$WAV_FILES_DIR' 不存在。"
  exit 1
fi

# 创建或清空输出文件
# > "$OUTPUT_WAV_SCP" # 这样会直接覆盖，如果文件不存在则创建
# 或者先检查父目录是否存在
OUTPUT_DIR=$(dirname "$OUTPUT_WAV_SCP")
if [ ! -d "$OUTPUT_DIR" ]; then
    echo "创建输出目录: $OUTPUT_DIR"
    mkdir -p "$OUTPUT_DIR"
    if [ $? -ne 0 ]; then
        echo "错误：无法创建输出目录 '$OUTPUT_DIR'。"
        exit 1
    fi
fi
# 现在可以安全地创建或覆盖文件了
: > "$OUTPUT_WAV_SCP" # 清空或创建文件


# 查找所有 .wav 文件 (不区分大小写，以防万一有 .WAV)
# 并为每个文件生成两行记录
# 使用 find ... -print0 和 while read ... 来安全处理可能包含特殊字符的文件名
find "$WAV_FILES_DIR" -type f -iname "*.wav" -print0 | while IFS= read -r -d $'\0' wav_file_path; do
  # 从完整路径中提取文件名 (例如: BAC009S0150W0001.wav)
  filename=$(basename "$wav_file_path")
  
  # 从文件名中提取 UTTERANCE_ID (去掉 .wav 后缀)
  # ${filename%.*} 会去掉最后一个点及其之后的部分
  utterance_id="${filename%.wav}" 
  # 如果可能是 .WAV 等，可以更通用地去掉最后一个点和后缀
  # utterance_id=$(echo "$filename" | sed 's/\.[^.]*$//')


  # 确保 wav_file_path 是绝对路径 (find 默认会给出相对或绝对路径，取决于起始点)
  # 为保险起见，如果不是绝对路径，转换为绝对路径
  # 不过，由于我们给 find 的起始路径是绝对的，所以 wav_file_path 应该也是绝对的
  # absolute_wav_path=$(realpath "$wav_file_path") # realpath 可以确保是绝对规范路径

  echo "处理文件: $wav_file_path, Utterance ID: $utterance_id"
  
  # 写入 transcribe 任务的行
  echo "${utterance_id}|${LANGUAGE}|${TASK_TRANSCRIBE} ${wav_file_path}" >> "$OUTPUT_WAV_SCP"
done

echo "------------------------------------"
if [ -s "$OUTPUT_WAV_SCP" ]; then # 检查文件是否存在且非空
  num_lines=$(wc -l < "$OUTPUT_WAV_SCP")
  num_wav_files=$num_lines
  echo "wav.scp 文件已成功生成于: $OUTPUT_WAV_SCP"
  echo "共找到 $num_wav_files 个 .wav 文件，生成了 $num_lines 行记录。"
else
  echo "警告：没有找到 .wav 文件或未能成功生成 wav.scp 文件于: $OUTPUT_WAV_SCP"
fi


