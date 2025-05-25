from whisper.bin.whisper_predicter import WhisperPredicter
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
log_info("加载模型测试器")
whisper_predicter = WhisperPredicter(config)

# 测试
log_info("测试")
whisper_predicter.predict_start_gpu()