"""
PyTorch Dataset 定义
分类数据集: {text, label} → {input_ids, attention_mask, label}
NER数据集:  {tokens, tags} → {input_ids, attention_mask, labels}
"""
import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


class ClassificationDataset(Dataset):
    """文本分类数据集"""

    def __init__(self, data_path: str, tokenizer: BertTokenizer, max_length: int = 256):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item["text"]
        label = item["label"]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "label": torch.tensor(label, dtype=torch.long),
        }


class NERDataset(Dataset):
    """命名实体识别数据集"""

    def __init__(
        self,
        data_path: str,
        tokenizer: BertTokenizer,
        tag2id: dict,
        max_length: int = 256,
    ):
        with open(data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)
        self.tokenizer = tokenizer
        self.tag2id = tag2id
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        tokens = item["tokens"]
        raw_tags = item["tags"]

        # 对 tokens 进行 BERT tokenization
        # BERT 中文分词器可能把一个词拆成多个 subtoken
        # 需要对齐标签: 第一个 subtoken 保留原标签, 后续 subtoken 用相同标签
        aligned_tokens = []
        aligned_tags = []
        for token, tag in zip(tokens, raw_tags):
            subtokens = self.tokenizer.tokenize(token)
            if len(subtokens) == 0:
                continue
            aligned_tokens.extend(subtokens)
            # 第一个 subtoken 保留原标签，后续 subtoken 用相同标签
            tag_id = self.tag2id.get(tag, 0)
            aligned_tags.append(tag_id)
            for _ in range(len(subtokens) - 1):
                aligned_tags.append(tag_id)

        # 截断（留空间给 [CLS] 和 [SEP]）
        max_len = self.max_length - 2
        aligned_tokens = aligned_tokens[:max_len]
        aligned_tags = aligned_tags[:max_len]

        # 添加特殊 token
        input_tokens = ["[CLS]"] + aligned_tokens + ["[SEP]"]
        label_ids = [-100] + aligned_tags + [-100]  # [CLS]和[SEP]用-100忽略

        # 转成 ID
        input_ids = self.tokenizer.convert_tokens_to_ids(input_tokens)
        attention_mask = [1] * len(input_ids)

        # Padding
        pad_len = self.max_length - len(input_ids)
        input_ids += [self.tokenizer.pad_token_id] * pad_len
        attention_mask += [0] * pad_len
        label_ids += [-100] * pad_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(label_ids, dtype=torch.long),
        }


def create_dataloaders(processed_dir: str, config: dict):
    """
    创建所有 DataLoader
    返回: {
        "cls_train": DataLoader, "cls_val": DataLoader, "cls_test": DataLoader,
        "ner_train": DataLoader, "ner_val": DataLoader, "ner_test": DataLoader,
    }
    """
    tokenizer = BertTokenizer.from_pretrained(
        config["encoder"]["model_name"], local_files_only=True
    )
    max_length = config["encoder"]["max_length"]

    dataloaders = {}

    # 分类 DataLoader
    for split in ["train", "val", "test"]:
        ds = ClassificationDataset(
            os.path.join(processed_dir, f"news_{split}.json"),
            tokenizer,
            max_length,
        )
        batch_size = config["classifier"]["batch_size"]
        dataloaders[f"cls_{split}"] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"分类 {split}: {len(ds)} 条, {len(dataloaders[f'cls_{split}'])} batches")

    # NER DataLoader
    for split in ["train", "val", "test"]:
        ds = NERDataset(
            os.path.join(processed_dir, f"ner_{split}.json"),
            tokenizer,
            config["ner"]["tag2id"],
            max_length,
        )
        batch_size = config["ner"]["batch_size"]
        dataloaders[f"ner_{split}"] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=0,
        )
        print(f"NER {split}: {len(ds)} 条, {len(dataloaders[f'ner_{split}'])} batches")

    return dataloaders, tokenizer
