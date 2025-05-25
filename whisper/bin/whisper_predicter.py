from peft import PeftConfig,PeftModel
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    Seq2SeqTrainer,
)
from tqdm import tqdm
import torch
import numpy as np
import torchaudio
from whisper.tokenizer.tokenization_whisper import WhisperTokenizer
from whisper.utils.common_utils import (
    log_info
)
from whisper.utils.data_utils import (
    IterWhisperDataset
)


class WhisperPredicter:
    def __init__(self,config):
        # 加载配置
        self.config = config
        # 加载模型
        self.load_whisper_model()
        # 加载数据
        self.load_data_wav_and_text()
        pass

    def predict_start_gpu(self):
        """
            模型的预测 - gpu解码
        """
        out_file = open(self.config['predict']['result_file'],'w',encoding="utf-8")
        self.model = self.model.cuda()
        eos_token_id = self.whisper_tokenizer.eos_token_id
        # 遍历数据集中的每一个样本，tqdm用于显示进度条
        for step, batch in enumerate(tqdm(self.eval_data_list)):
            # 获取语言和任务
            language = batch['idx'].split("|")[1]
            task = batch['idx'].split("|")[2]

            # 获取forced_decoder_ids（控制生成模式）
            forced_decoder_ids = self.whisper_tokenizer.get_decoder_prompt_ids(
                language=language,
                task=task
            )
            # 数据预处理（全部迁移到 GPU）
            input_features = torch.from_numpy(batch["input_features"][np.newaxis,:,:]).float().cuda()
            # 在CUDA上自动混合精度（AMP），可加速推理同时节省显存
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    # 使用训练好的模型对输入的音频特征进行自动生成，得到预测的token序列
                    generated_tokens = (
                        self.model.generate(
                            input_features=torch.from_numpy(batch["input_features"][np.newaxis,:,:]).to("cuda"),
                            eos_token_id=eos_token_id,
                            forced_decoder_ids=forced_decoder_ids,
                            max_new_tokens=255
                        )
                        .cpu()
                        .numpy()
                    )

                    # 获取当前样本的真实标签
                    labels = batch["labels"]
                    labels = np.where(labels != -100, labels, self.whisper_tokenizer.pad_token_id)
                    
                    # 解码预测的token序列，得到模型输出的文本
                    decoded_preds = self.whisper_tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
                    out_file.write(batch['idx']+" "+decoded_preds[0]+"\n")
                    pass
                pass
            
            # 显存清理
            del generated_tokens, labels, batch
            torch.cuda.empty_cache()
        pass

    def load_data_wav_and_text(self):
        """
            准备数据
        """
        log_info("加载eval集")
        self.eval_data_list = IterWhisperDataset(
            self.config['predict']['eval']['wav_scp'],
            self.config['predict']['eval']['text'],
            self.whisper_feature_extractor,
            self.whisper_tokenizer,
            False
        )
        pass

    def load_whisper_model(self):
        """
            加载模型
        """
        whisper_model = self.config['model']['model_path']

        # 特征提取器
        log_info("加载whisper_feature_extractor")
        self.whisper_feature_extractor = WhisperFeatureExtractor.from_pretrained(whisper_model)
        
        # 分词器
        log_info("加载whisper_tokenizer")
        self.whisper_tokenizer = WhisperTokenizer.from_pretrained(whisper_model)

        # 加载大模型
        if self.config['model']['is_large_model']:
            log_info("加载 whisper模型 - 经过peft微调")
            peft_model_id = self.config['predict']['model_path']
            peft_config = PeftConfig.from_pretrained(peft_model_id)
            self.model = WhisperForConditionalGeneration.from_pretrained(
                peft_config.base_model_name_or_path
            )
            self.model = PeftModel.from_pretrained(self.model, peft_model_id)
            pass
        else:
            log_info("加载 whisper模型")
            self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model)

        log_info("模型结构：")
        log_info(self.model)
        self.model.eval()
        pass
    
    pass