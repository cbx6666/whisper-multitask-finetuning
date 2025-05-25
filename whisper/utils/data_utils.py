import random
from tqdm import tqdm
from random import shuffle
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
import torch
import torchaudio
from torch.utils.data import IterableDataset
from whisper.utils.common_utils import (
    log_info
)

# 打乱字典
def random_dic(dicts):
    dict_ls_key = list(dicts.keys())
    random.shuffle(dict_ls_key)
    new_dic = {}
    for key in dict_ls_key:
        new_dic[key] = dicts.get(key)
    return new_dic

# 数据整理器，专门用于处理语音序列到文本序列任务的数据批次化
# 核心作用是将多个样本（语音特征和文本标签）整理成一个统一的批次
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any # 包含特征提取器和分词器的处理器对象
    decoder_start_token_id: int # 解码器的起始Token ID
    forward_attention_mask: bool # 是否生成语音特征的注意力掩码

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need
        # different padding methods
        model_input_name = self.processor.model_input_names[0]
        input_features = [{model_input_name: feature[model_input_name]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        if self.forward_attention_mask:
            batch["attention_mask"] = torch.LongTensor([feature["attention_mask"] for feature in features])

        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

# 准备数据
# 可迭代数据集
class IterWhisperDataset(IterableDataset):
    def __init__(self, wav_scp, text, whisper_feature_extractor, whisper_tokenizer, shuffle=True):
        # 处理为字典
        self.data_list = {}
        # 音频路径
        with open(wav_scp, 'r', encoding="utf-8") as file:
            # 使用 tqdm 显示进度条
            for line in tqdm(file.readlines()):
                # 去除换行符和空格
                line = line.strip()  
                # 将行内容按空格分割为列表，取第一个元素作为ID
                idx = line.split(" ")[0]
                # 将剩余部分合并为音频路径字符串
                wav_path = " ".join(line.split(" ")[1:])
                # 为当前ID创建空列表
                self.data_list[idx] = []
                # 将音频路径添加到列表
                self.data_list[idx].append(wav_path)
                
        # 音频文本
        with open(text, 'r', encoding="utf-8") as file:
            for line in tqdm(file.readlines()):
                line = line.strip()
                # 将行内容按空格分割为列表，取第一个元素作为ID
                idx = line.split(" ")[0]
                # 将剩余部分合并为文本内容字符串
                text = " ".join(line.split(" ")[1:])
                # 将文本内容追加到对应ID的列表中
                if idx in self.data_list:
                    if len(self.data_list[idx]) == 1:
                        self.data_list[idx].append(text)
        
        # 过滤掉不完整的条目 (即列表长度不为2的条目)
        filtered_data_list = {}
        incomplete_entry_count = 0
        for idx, content_list in self.data_list.items():
            if len(content_list) == 2: # 确保同时有音频路径和文本
                filtered_data_list[idx] = content_list
            else:
                incomplete_entry_count += 1

        self.data_list = filtered_data_list

        if incomplete_entry_count > 0:
            print(f"Info: Removed {incomplete_entry_count} incomplete entries from the dataset.")
        print(f"Info: Loaded {len(self.data_list)} complete entries.")
        
        self.whisper_feature_extractor = whisper_feature_extractor
        self.whisper_tokenizer = whisper_tokenizer
        if shuffle:
            log_info("打乱文本，全部个数为：" + str(len(self.data_list)))
            # 打乱
            self.data_list = random_dic(self.data_list)
            pass
        pass
                
    def __len__(self):
        return len(self.data_list)
    
    # 每次读取一对音频+文本，提取音频特征、对文本分词，返回一个example字典用于模型输入
    def __iter__(self):
        # 遍历所有的数据
        for idx in self.data_list:
            # 获取语种
            language = idx.split("|")[1]
            # 获取任务类型
            task = idx.split("|")[2]
            # 音频的路径
            wav_path = self.data_list[idx][0]
            # 音频的文本
            text = self.data_list[idx][1]
            
            example = {}
            example['idx'] = idx
            
            # 提取特征
            # 读取音频
            data_audio = torchaudio.load(wav_path)
            example['input_features'] = self.whisper_feature_extractor(
                data_audio[0].numpy(), 
                sampling_rate=16000 
            ).input_features[0]
            
            # token
            self.whisper_tokenizer.set_prefix_tokens(language=language, task=task)
            example['labels'] = self.whisper_tokenizer(text).input_ids[1:]
            
            # 生成当前样本字典，最终传递给DataCollator进行批处理
            yield example