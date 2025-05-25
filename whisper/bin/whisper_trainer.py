from peft import prepare_model_for_kbit_training
from peft import LoraConfig, PeftModel, LoraModel, LoraConfig, get_peft_model
from transformers import (
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments
)
from whisper.tokenizer.tokenization_whisper import WhisperTokenizer
from whisper.utils.common_utils import (
    log_info
)
from whisper.utils.data_utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    IterWhisperDataset
)

class WhisperTrainer:
    def __init__(self, config):
        # 加载配置
        self.config = config
        # 加载模型
        self.load_whisper_model()
        # 加载数据
        self.load_data_wav_and_text()
        # 模型参数
        self.load_whisper_model_argv()
        pass
    
    def train_start(self):
        """
            模型训练
        """
        trainer = Seq2SeqTrainer(
        model=self.model,                   # 模型
        args=self.training_args,            # 训练参数
        train_dataset=self.train_data_list, # 训练数据集
        eval_dataset=self.test_data_list,   # 评估数据集
        tokenizer=self.whisper_tokenizer,   # 分词器
        data_collator=self.data_collator    # 数据整理器
        )

        # 训练
        check_point = self.config['model']['model_train_argv']['resume_from_checkpoint']
        if check_point != "":
            log_info("模型恢复中....."+check_point)
            train_result = trainer.train(resume_from_checkpoint=check_point)
        else:
            train_result = trainer.train()
        trainer.save_model(
            self.config['model']['model_train_argv']['out_model_path']
        )
        self.processor.save_pretrained(
            self.config['model']['model_train_argv']['out_model_path']
        )
        pass
        
    def load_data_wav_and_text(self):
        """
            加载数据
        """
        log_info("加载训练集")
        self.train_data_list = IterWhisperDataset(
            self.config['data']['train']['wav_scp'],
            self.config['data']['train']['text'],
            self.whisper_feature_extractor,
            self.whisper_tokenizer
        )
        log_info("加载测试集")
        self.test_data_list = IterWhisperDataset(
            self.config['data']['test']['wav_scp'],
            self.config['data']['test']['text'],
            self.whisper_feature_extractor,
            self.whisper_tokenizer
        )
        pass
    
    def load_whisper_model_argv(self):
        """
            加载参数
        """
        log_info("加载模型训练的参数")
        self.training_args = Seq2SeqTrainingArguments(
            output_dir = self.config['model']['model_train_argv']["out_model_path"],  
            per_device_train_batch_size = self.config['model']['model_train_argv']["per_device_train_batch_size"],
            gradient_accumulation_steps = self.config['model']['model_train_argv']["gradient_accumulation_steps"],  
            # 学习率
            learning_rate = self.config['model']['model_train_argv']["learning_rate"],
            warmup_steps = self.config['model']['model_train_argv']["warmup_steps"],
            num_train_epochs = self.config['model']['model_train_argv']["num_train_epochs"],
            evaluation_strategy = self.config['model']['model_train_argv']["evaluation_strategy"],
            fp16 = self.config['model']['model_train_argv']["fp16"],
            per_device_eval_batch_size = self.config['model']['model_train_argv']["per_device_eval_batch_size"],
            generation_max_length = self.config['model']['model_train_argv']["generation_max_length"],
            logging_steps = self.config['model']['model_train_argv']["logging_steps"],
            remove_unused_columns = self.config['model']['model_train_argv']["remove_unused_columns"],
            # 模型训练时，把example中的labels字段当作目标标签，评估是对模型生成的token序列（文本）和真实文本进行比较  
            label_names = self.config['model']['model_train_argv']["label_names"]  
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
        
        # whisper模型
        log_info("加载whisper模型")
        self.model = WhisperForConditionalGeneration.from_pretrained(whisper_model)
        
        # 集成的处理器类，将音频数据预处理成模型可以理解的输入格式
        log_info("加载processor")
        self.processor = WhisperProcessor.from_pretrained(whisper_model)
        
        log_info("模型结构：")
        log_info(self.model)
        
        # 是否加载大模型
        if self.config['model']['is_large_model']:
            log_info("加载 whisper模型 - 经过peft微调")
            self.model = self._load_large_model_peft(self.model)
            
        # 配置解码参数
        # 禁用强制解码标记：防止在生成文本时强制模型以特定Token开头
        self.model.config.forced_decode_ids = None
        # 解除生成时的Token抑制：允许模型生成所有可能的 Token，包括特殊符号、标点、多语言字符等。
        self.model.config.suppress_tokens = []
        
        # 定义数据整理器
        self.data_collator = DataCollatorSpeechSeq2SeqWithPadding(
            processor=self.processor,
            decoder_start_token_id=self.model.config.decoder_start_token_id,
            forward_attention_mask=self.config['model']['data_collator']['forward_attention_mask']
        ) 
        pass      
        
    def _load_large_model_peft(self, model):
        """
            加载大模型
        """
        # 准备量化训练
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        # 配置LoRA参数（低秩适应），一种参数高效微调技术，旨在通过引入少量可训练参数，高效调整大型预训练模型
        config = LoraConfig(
            r=32, 
            lora_alpha=64, 
            target_modules=".*decoder.*(self_attn|encoder_attn).*(q_proj|v_proj)$",#["q_proj", "v_proj"],
            lora_dropout=0.05, 
            bias="none"
        )
        # 创建可训练的 PEFT 模型
        model = get_peft_model(model, config)
        log_info("查看训练参数和模型原始参数量")
        model.print_trainable_parameters()
        
        return model
    
    pass