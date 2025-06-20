import yaml
import logging

# 日志
logger = logging.getLogger(__name__)
logging.getLogger().setLevel(logging.INFO)

def load_whisper_config(config_path):
    """
        读取配置
    """
    with open(config_path, 'r', encoding='utf-8') as f:
        result = yaml.load(f.read(), Loader=yaml.FullLoader)
        return result
    
def log_info(str_log):
    logging.info(str_log)