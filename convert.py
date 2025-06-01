import os
import shutil
import torch
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel
from whisper.utils.common_utils import (
    load_whisper_config,
    log_info
)

# ========== 加载配置 ==========
log_info("加载配置文件")
config = load_whisper_config(
    "/mnt/d/WSL/asr_large_model/whisper-multitask-finetuning/example/aishell-sample/config/whisper_multitask.yaml"
)

# ========== 从配置读取路径 ==========
peft_model_path = config['predict']['model_path']
base_model_path = config['model']['model_path']
merged_model_path = config['dev_env']['merged_model_path']

# ========== 合并模型 ==========
log_info(f"开始合并模型...")
log_info(f"PEFT适配器模型: {peft_model_path}")
log_info(f"基础模型: {base_model_path}")
log_info(f"输出目录: {merged_model_path}")

# 创建输出目录
os.makedirs(merged_model_path, exist_ok=True)

log_info("加载基础 Whisper 模型...")
base_model = WhisperForConditionalGeneration.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
    device_map="auto" if torch.cuda.is_available() else None
)

log_info("加载 PEFT 微调模型...")
peft_model = PeftModel.from_pretrained(base_model, peft_model_path)

log_info("合并模型...")
merged_model = peft_model.merge_and_unload()

# 转换回float32以确保兼容性
merged_model = merged_model.float()

log_info(f"保存合并后的模型到 {merged_model_path}")
merged_model.save_pretrained(merged_model_path)

log_info("保存 Processor 和 Tokenizer...")
processor = WhisperProcessor.from_pretrained(base_model_path)
processor.save_pretrained(merged_model_path)

# ========== 检查并复制 tokenizer 相关文件 ==========
required_files = [
    "tokenizer.json",
    "tokenizer_config.json",
    "generation_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt"
]

for file_name in required_files:
    src = os.path.join(base_model_path, file_name)
    dst = os.path.join(merged_model_path, file_name)
    if not os.path.exists(dst) and os.path.exists(src):
        shutil.copy(src, dst)
        log_info(f"复制缺失文件: {file_name}")
    elif os.path.exists(dst):
        log_info(f"文件已存在: {file_name}")
    else:
        log_info(f"⚠️ 找不到文件: {file_name}")

# ========== 列出合并后的模型文件 ==========
log_info(f"合并模型目录内容:")
for file_name in os.listdir(merged_model_path):
    file_path = os.path.join(merged_model_path, file_name)
    file_size = os.path.getsize(file_path) / (1024 * 1024)  # 转换为MB
    log_info(f"  {file_name} ({file_size:.2f} MB)")

log_info("✅ 模型合并完成!")
log_info(f"✅ 合并模型保存在: {merged_model_path}")
log_info("现在可以使用合并后的模型进行推理了")
