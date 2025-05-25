from whisper.bin.whisper_trainer import WhisperTrainer
from whisper.utils.common_utils import (
    load_whisper_config,
    log_info
)

# 配置参数
log_info("加载配置文件")
config = load_whisper_config(
    "/mnt/d/WSL/asr_large_model/whisper-multitask-finetuning/example/aishell-sample/config/whisper_multitask.yaml"
)

# 加载模型
log_info("加载模型训练器")
whisper_trainer = WhisperTrainer(config)

# 训练
log_info("开始训练")
whisper_trainer.train_start()