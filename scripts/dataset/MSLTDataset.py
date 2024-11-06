import sys
import os

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

import torch
from transformers import BertTokenizer
from torch.utils.data import Dataset, DataLoader
from utils.text_processing import preprocess_text_pairs, filter_long_sequences


class MSLTDataset(Dataset):
    def __init__(self, data_dir_en, data_dir_zh, tokenizer, max_length=64):
        self.tokenizer = tokenizer
        self.max_length = max_length
        # self.data_pairs = self._load_and_prepare_combined_data(data_dir_en, data_dir_zh)
        self.data_pairs = self._load_and_prepare_data(data_dir_en)

    def _load_and_prepare_combined_data(self, directory_en, directory_zh):
        data_pairs = []

        # Load English to Chinese (from English dataset)
        for root, _, files in os.walk(directory_en):
            for file in files:
                if file.endswith(".T2.en.snt"):
                    en_file_path = os.path.join(root, file)
                    zh_file_path = en_file_path.replace(".T2.en.snt", ".T3.zh.snt")

                    if os.path.exists(zh_file_path):
                        en_text = self._read_snt_file(en_file_path)
                        zh_text = self._read_snt_file(zh_file_path)
                        if not en_text or not zh_text:
                            continue
                        data_pairs.append(("[EN-TO-ZH] " + en_text, zh_text))  # Add direction token

        # Load Chinese to English (from Chinese dataset)
        for root, _, files in os.walk(directory_zh):
            for file in files:
                if file.endswith(".T2.ch.snt"):
                    zh_file_path = os.path.join(root, file)
                    en_file_path = zh_file_path.replace(".T2.ch.snt", ".T3.en.snt")

                    if os.path.exists(en_file_path):
                        zh_text = self._read_snt_file(zh_file_path)
                        en_text = self._read_snt_file(en_file_path)
                        if not zh_text or not en_text:  # Skip if either file is empty
                            continue
                        data_pairs.append(("[ZH-TO-EN] " + zh_text, en_text))  # Add direction token
        
        print("Reformatting text...")
        data_pairs = preprocess_text_pairs(data_pairs)
        
        print("Dropping long sequences...")
        data_pairs = filter_long_sequences(data_pairs, self.tokenizer, self.max_length)
        
        return data_pairs
    
    def _load_and_prepare_data(self, directory_en):
        data_pairs = []

        # Load English to Chinese
        for root, _, files in os.walk(directory_en):
            for file in files:
                if file.endswith(".T2.en.snt"):
                    en_file_path = os.path.join(root, file)
                    zh_file_path = en_file_path.replace(".T2.en.snt", ".T3.zh.snt")

                    if os.path.exists(zh_file_path):
                        en_text = self._read_snt_file(en_file_path)
                        zh_text = self._read_snt_file(zh_file_path)
                        if not en_text or not zh_text:
                            continue
                        data_pairs.append((en_text, zh_text))
        
        print("Reformatting text...")
        data_pairs = preprocess_text_pairs(data_pairs)
        
        print("Dropping long sequences...")
        data_pairs = filter_long_sequences(data_pairs, self.tokenizer, self.max_length)
        
        return data_pairs

    @staticmethod
    def _read_snt_file(file_path):
        with open(file_path, 'r', encoding='utf-16') as file:
            return file.read().strip()

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        src_text, tgt_text = self.data_pairs[idx]

        # Tokenize the source and target texts
        src_tokens = self.tokenizer(
            src_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        tgt_tokens = self.tokenizer(
            tgt_text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Extract input_ids and attention_mask for source and target
        src_input_ids = src_tokens['input_ids'].view(-1)
        src_attention_mask = src_tokens['attention_mask'].view(-1)

        tgt_input_ids = tgt_tokens['input_ids'].view(-1)
        tgt_attention_mask = tgt_tokens['attention_mask'].view(-1)

        return {
            'src_text': src_text,
            'tgt_text': tgt_text,
            'src_input_ids': src_input_ids,
            'src_attention_mask': src_attention_mask,
            'tgt_input_ids': tgt_input_ids,
            'tgt_attention_mask': tgt_attention_mask
        }


# ================================================================================================
# Set the directories and parameters
# data_dir_en = "../../data/MSLT_Corpus/Data/MSLT_Dev_EN"
# data_dir_zh = "../../data/MSLT_Corpus/Data/MSLT_Dev_ZH"
# tokenizer = BertTokenizer.from_pretrained("../../bert-base/bert-base-multilingual-cased")

# # Avoid reinitialization
# if __name__ == "__main__":
#     print("Initializing dataset...")
#     dataset = MSLTDataset(data_dir_en, data_dir_zh, tokenizer)
    
#     sample = dataset[0]
    
#     sample = dataset[0]
#     print("\n=== 单个样本详细信息 ===")
#     for key, value in sample.items():
#         if torch.is_tensor(value):
#             print(f"{key}:")
#             print(f"- 形状: {value.shape}")
#             print(f"- 类型: {value.dtype}")
#             print(f"- 设备: {value.device}")
#             print(f"- 值: {value[:]}...")  # 只显示前50个元素
#         else:
#             print(f"{key}: {value[:50]}...")  # 文本只显示前50个字符
#         print()
    
#     print("\n:")
#     print(f"src_text: {sample['src_text']}")
#     print(f"tgt_text: {sample['tgt_text']}")
#     print(f"\n:")
#     print(f"src_input_ids: {sample['src_input_ids'].shape}")
#     print(f"src_attention_mask: {sample['src_attention_mask'].shape}")
#     print(f"tgt_input_ids: {sample['tgt_input_ids'].shape}")
#     print(f"tgt_attention_mask: {sample['tgt_attention_mask'].shape}")
    
#     dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

#     for batch in dataloader:
#         print(f"\nSource Input IDs:{batch['src_input_ids'].shape}\n {batch['src_input_ids']}\n")
#         print(f"\nSource Attention Mask:{batch['src_attention_mask'].shape}\n, {batch['src_attention_mask']}\n")
#         print(f"\nTarget Input IDs:{batch['src_attention_mask'].shape}\n, {batch['tgt_input_ids']}\n")
#         print(f"\nTarget Attention Mask:{batch['src_attention_mask'].shape}\n, {batch['tgt_attention_mask']}\n")
#         break

