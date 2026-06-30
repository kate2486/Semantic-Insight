# Semantic-Insight — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 5天内构建 Semantic-Insight：基于共享BERT编码器的双任务NLP系统（文本分类 + 命名实体识别）+ Gradio演示界面

**Architecture:** 共享 `bert-base-chinese` 作为编码器，上层分接分类头（[CLS]向量→Linear→10类别）和NER标注头（每Token向量→Linear→CRF→BIO标签序列），通过 `main.py` 统一训练/评估/推理入口，`Gradio` 提供Web交互界面。

**Tech Stack:** Python 3.10+, PyTorch 2.0+, HuggingFace Transformers 4.30+, Gradio 4.0+, pytorch-crf, scikit-learn

## Global Constraints

- torch>=2.0.0, transformers>=4.30.0, gradio>=4.0.0
- BERT编码器使用 `bert-base-chinese`，冻结 embedding 层和前6层 Transformer
- 最大序列长度 512，超长截断
- NER标签共7个: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
- 特殊token ([CLS], [SEP], [PAD]) 标签置 -100（loss计算时忽略）
- 分类准确率目标 ≥90%，NER Macro F1 ≥80%
- 所有模块文件以 `src/` 为根，`from src.xxx import yyy` 方式导入
- 配置文件统一用 `configs/config.yaml`，YAML格式

---

## Day 1 — 数据准备

### Task 1: 项目脚手架 + 依赖 + 配置

**Files:**
- Create: `requirements.txt`
- Create: `configs/config.yaml`
- Create: `src/__init__.py`
- Create: `src/data/__init__.py`
- Create: `src/models/__init__.py`
- Create: `src/train/__init__.py`
- Create: `src/evaluate/__init__.py`
- Create: `src/app/__init__.py`
- Create: `tests/__init__.py`
- Create: `tests/test_data.py`
- Create: `tests/test_models.py`
- Create: `.gitignore`

**Interfaces:**
- Produces: `config.yaml` 被所有后续任务导入使用
- Produces: `requirements.txt` 定义全部依赖

- [ ] **Step 1: 创建 requirements.txt**

```txt
torch>=2.0.0
transformers>=4.30.0
gradio>=4.0.0
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
pyyaml
pytorch-crf
jieba
datasets
```

- [ ] **Step 2: 创建 configs/config.yaml**

```yaml
# 共享编码器配置
encoder:
  model_name: "bert-base-chinese"
  max_length: 256
  freeze_embeddings: true
  freeze_layers: 6  # 冻结前6层Transformer

# 文本分类配置
classifier:
  num_classes: 10
  label_names: ["体育", "财经", "科技", "教育", "时尚", "军事", "游戏", "房产", "娱乐", "时政"]
  dropout: 0.1
  learning_rate: 2e-5
  batch_size: 32
  epochs: 5
  warmup_ratio: 0.1

# NER配置
ner:
  num_tags: 7
  tag2id:
    O: 0
    B-PER: 1
    I-PER: 2
    B-LOC: 3
    I-LOC: 4
    B-ORG: 5
    I-ORG: 6
  id2tag: ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
  dropout: 0.1
  learning_rate: 3e-5
  batch_size: 16
  epochs: 10
  warmup_ratio: 0.1

# 训练通用配置
training:
  seed: 42
  gradient_accumulation_steps: 2
  max_grad_norm: 1.0
  eval_steps: 200
  save_steps: 500
  logging_steps: 50

# 数据配置
data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1
```

- [ ] **Step 3: 创建所有 `__init__.py` 空文件**

```bash
touch src/__init__.py src/data/__init__.py src/models/__init__.py
touch src/train/__init__.py src/evaluate/__init__.py src/app/__init__.py
touch tests/__init__.py tests/test_data.py tests/test_models.py
```

- [ ] **Step 4: 创建 .gitignore**

```gitignore
# Python
__pycache__/
*.py[cod]
*.egg-info/
dist/
build/
.eggs/

# Virtual env
venv/
.venv/
env/

# Data
data/raw/*
data/processed/*
!data/raw/.gitkeep
!data/processed/.gitkeep

# Models
checkpoints/*.pt
checkpoints/*.bin
!checkpoints/.gitkeep

# IDE
.vscode/
.idea/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# OS
.DS_Store
Thumbs.db

# Misc
*.log
wandb/
```

- [ ] **Step 5: 创建占位文件**

```bash
touch data/raw/.gitkeep data/processed/.gitkeep checkpoints/.gitkeep
```

- [ ] **Step 6: 安装依赖并验证**

Run: `pip install -r requirements.txt`
Expected: 所有包安装成功，无版本冲突

- [ ] **Step 7: 提交**

```bash
git add -A && git commit -m "feat: scaffold project with configs and dependencies

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 2: 数据下载脚本

**Files:**
- Create: `src/data/download.py`
- Create: `data/raw/.gitkeep` (already exists)

**Interfaces:**
- Consumes: `configs/config.yaml` 中的 `data.raw_dir`
- Produces: `data/raw/thucnews/` 目录（含10个子文件夹，每个含txt文件）
- Produces: `data/raw/peopledaily/` 目录（含BIO标注文件）

- [ ] **Step 1: 编写 download.py**

```python
"""
数据下载模块
下载 THUCNews 子集（10分类文本）和 PeopleDaily NER 数据集
"""
import os
import sys
import tarfile
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest: str, desc: str = "Downloading"):
    """带进度条的文件下载"""
    response = requests.get(url, stream=True)
    total = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def download_thucnews(raw_dir: str):
    """
    下载 THUCNews 子集（10分类，每类6000条）
    数据来源: THUCTC (thuctc.thunlp.org) 精简版
    """
    thucnews_dir = os.path.join(raw_dir, "thucnews")
    if os.path.exists(thucnews_dir) and len(os.listdir(thucnews_dir)) >= 10:
        print(f"THUCNews 已存在于 {thucnews_dir}")
        return thucnews_dir

    # THUCNews 子集下载链接（GitHub mirror）
    url = "https://github.com/649453932/Chinese-Text-Classification-Pytorch/releases/download/v1.0/THUCNews.zip"

    # 备用: 使用清华源
    backup_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip"

    zip_path = os.path.join(raw_dir, "thucnews.zip")
    try:
        print("尝试从 GitHub mirror 下载 THUCNews...")
        download_file(url, zip_path, "下载 THUCNews")
    except Exception:
        print("GitHub mirror 失败，尝试备用源...")
        download_file(backup_url, zip_path, "下载 THUCNews")

    print("解压 THUCNews...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    # 清理zip文件
    os.remove(zip_path)

    # THUCNews解压后结构为 THUCNews/类别/xxx.txt
    extracted = os.path.join(raw_dir, "THUCNews")
    if os.path.exists(extracted):
        # 只保留前10个类别
        import shutil
        target_dir = thucnews_dir
        os.makedirs(target_dir, exist_ok=True)
        categories = sorted(os.listdir(extracted))[:10]
        for cat in categories:
            cat_src = os.path.join(extracted, cat)
            cat_dst = os.path.join(target_dir, cat)
            if os.path.isdir(cat_src) and not os.path.exists(cat_dst):
                shutil.move(cat_src, cat_dst)
        shutil.rmtree(extracted)

    print(f"THUCNews 下载完成: {thucnews_dir}")
    return thucnews_dir


def download_peopledaily(raw_dir: str):
    """
    下载 PeopleDaily NER 数据集
    使用 HuggingFace datasets 库直接加载
    """
    ner_dir = os.path.join(raw_dir, "peopledaily")
    if os.path.exists(ner_dir) and os.path.exists(os.path.join(ner_dir, "train.txt")):
        print(f"PeopleDaily 已存在于 {ner_dir}")
        return ner_dir

    os.makedirs(ner_dir, exist_ok=True)
    try:
        from datasets import load_dataset
        print("从 HuggingFace 加载 PeopleDaily NER 数据集...")
        dataset = load_dataset("shibing624/peoples_daily_ner")
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                output_path = os.path.join(ner_dir, f"{split_name}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in dataset[split_name]:
                        tokens = item["tokens"]
                        tags = item["ner_tags"]
                        for token, tag in zip(tokens, tags):
                            f.write(f"{token}\t{tag}\n")
                        f.write("\n")  # 空行分隔句子
                print(f"  {split_name}.txt 已保存 ({len(dataset[split_name])} 条)")

    except Exception as e:
        print(f"HuggingFace 加载失败 ({e})，使用内置数据生成器...")
        _generate_sample_ner_data(ner_dir)

    return ner_dir


def _generate_sample_ner_data(ner_dir: str):
    """
    生成示例NER数据（含人名、地名、机构名）
    用于在无法访问网络时仍能跑通项目流程
    """
    samples = [
        ("中国 政府 在 北京 举行 记者招待会", "B-ORG I-ORG O O B-LOC O O O"),
        ("李克强 总理 访问 了 上海 和 杭州", "B-PER I-PER O O O B-LOC O B-LOC"),
        ("华为 公司 在 深圳 发布 新 产品", "B-ORG I-ORG O B-LOC O O O"),
        ("北京 大学 和 清华 大学 联合 举办 论坛", "B-ORG I-ORG O B-ORG I-ORG O O O"),
        ("张三 来到 了 纽约 参加 联合国 大会", "B-PER I-PER O O B-LOC O B-ORG I-ORG O"),
        # ... 更多样本 ...
        ("腾讯 和 阿里巴巴 是 中国 最大 的 互联网 公司", "B-ORG O B-ORG O B-LOC O O O O"),
        ("习近平 主席 在 人民大会堂 发表 重要 讲话", "B-PER I-PER O B-LOC I-LOC I-LOC O O O"),
        ("微软 公司 总部 位于 美国 西雅图", "B-ORG I-ORG O O B-LOC B-LOC I-LOC"),
        ("王 明 昨天 去了 广州 和 深圳", "B-PER I-PER O O B-LOC O B-LOC"),
        ("国务院 近日 在 北京 召开 常务会议", "B-ORG O O B-LOC O O I-ORG"),
    ]

    train_file = os.path.join(ner_dir, "train.txt")
    test_file = os.path.join(ner_dir, "test.txt")

    # 生成更多样本（简单扩充）
    import random
    random.seed(42)
    all_samples = samples.copy()
    for _ in range(200):
        base = random.choice(samples)
        all_samples.append(base)

    split = int(len(all_samples) * 0.8)
    for fname, subset in [(train_file, all_samples[:split]), (test_file, all_samples[split:])]:
        with open(fname, "w", encoding="utf-8") as f:
            for text, tags_str in subset:
                words = text.split()
                tags = tags_str.split()
                for w, t in zip(words, tags):
                    f.write(f"{w}\t{t}\n")
                f.write("\n")
    print(f"  已生成 {len(all_samples)} 条示例NER数据")
    print("  ⚠ 这是示例数据，用于跑通流程。真实训练建议连接网络下载完整数据集。")


def main():
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw")
    raw_dir = os.path.abspath(raw_dir)

    print("=" * 50)
    print("Step 1/2: 下载 THUCNews 文本分类数据集")
    print("=" * 50)
    download_thucnews(raw_dir)

    print()
    print("=" * 50)
    print("Step 2/2: 下载 PeopleDaily NER 数据集")
    print("=" * 50)
    download_peopledaily(raw_dir)

    print()
    print("数据集下载完成!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行下载脚本**

Run: `cd "c:/Users/ASUS/Desktop/神经网络实训/-" && python -m src.data.download`
Expected: 下载THUCNews和PeopleDaily数据（或生成备用示例数据）

- [ ] **Step 3: 提交**

```bash
git add src/data/download.py data/raw/.gitkeep
git commit -m "feat: add data download module (THUCNews + PeopleDaily NER)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 3: 数据预处理脚本

**Files:**
- Create: `src/data/preprocess.py`

**Interfaces:**
- Consumes: `data/raw/thucnews/` 和 `data/raw/peopledaily/`
- Produces: `data/processed/news_train.json`, `news_val.json`, `news_test.json`, `ner_train.json`, `ner_val.json`, `ner_test.json`
- 所有输出JSON格式：分类为 `[{text, label}]`，NER为 `[{tokens, tags}]`

- [ ] **Step 1: 编写 preprocess.py**

```python
"""
数据预处理模块
清洗文本 → 格式统一 → train/val/test 划分 → 保存为 JSON
"""
import os
import json
import re
import random
from collections import Counter
from pathlib import Path


def clean_text(text: str) -> str:
    """统一文本清洗"""
    # 去HTML标签
    text = re.sub(r"<[^>]+>", "", text)
    # 去多余空白
    text = re.sub(r"\s+", " ", text)
    # 统一全角半角
    text = text.strip()
    # 截断过长文本（>512字符大概率超出BERT限制）
    if len(text) > 512:
        text = text[:510] + "..."
    return text


def process_thucnews(raw_dir: str, processed_dir: str, ratios=(0.8, 0.1, 0.1)):
    """
    处理 THUCNews 数据集
    输入: raw_dir/thucnews/类别/xxx.txt
    输出: news_train.json, news_val.json, news_test.json
    """
    thucnews_dir = os.path.join(raw_dir, "thucnews")
    if not os.path.exists(thucnews_dir):
        print(f"THUCNews 原始数据未找到: {thucnews_dir}")
        return None

    # 获取所有类别文件夹
    categories = sorted([
        d for d in os.listdir(thucnews_dir)
        if os.path.isdir(os.path.join(thucnews_dir, d))
    ])
    label2id = {cat: i for i, cat in enumerate(categories)}
    print(f"类别映射: {label2id}")

    samples = []
    for cat in categories:
        cat_dir = os.path.join(thucnews_dir, cat)
        txt_files = [f for f in os.listdir(cat_dir) if f.endswith(".txt")]
        for fname in txt_files:
            fpath = os.path.join(cat_dir, fname)
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                text = clean_text(text)
                if len(text) < 10:  # 过滤太短的文本
                    continue
                samples.append({"text": text, "label": label2id[cat]})
            except Exception as e:
                print(f"  读取失败 {fpath}: {e}")

    print(f"总样本数: {len(samples)}")

    # 打乱并划分
    random.seed(42)
    random.shuffle(samples)
    n = len(samples)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    splits = {
        "news_train.json": samples[:train_end],
        "news_val.json": samples[train_end:val_end],
        "news_test.json": samples[val_end:],
    }

    os.makedirs(processed_dir, exist_ok=True)
    for fname, data in splits.items():
        fpath = os.path.join(processed_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {fname}: {len(data)} 条")

    return label2id


def process_peopledaily(raw_dir: str, processed_dir: str, ratios=(0.8, 0.1, 0.1)):
    """
    处理 PeopleDaily NER 数据集
    输入: raw_dir/peopledaily/train.txt 等（BIO格式: 词语\t标签）
    输出: ner_train.json, ner_val.json, ner_test.json
    """
    ner_raw_dir = os.path.join(raw_dir, "peopledaily")

    # 收集所有txt文件
    source_files = []
    for fname in ["train.txt", "validation.txt", "test.txt"]:
        fpath = os.path.join(ner_raw_dir, fname)
        if os.path.exists(fpath):
            source_files.append(fpath)

    if not source_files:
        print(f"NER原始数据未找到: {ner_raw_dir}")
        return None

    # 合并所有数据
    all_sentences = []
    for fpath in source_files:
        with open(fpath, "r", encoding="utf-8") as f:
            tokens, tags = [], []
            for line in f:
                line = line.strip()
                if not line:
                    if tokens:
                        all_sentences.append({"tokens": tokens, "tags": tags})
                        tokens, tags = [], []
                    continue
                parts = line.split()
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    # 清理标签（有些数据集用 B-ORG 有些用 B_ORG）
                    tag = parts[1].replace("_", "-").strip()
                    tags.append(tag)

    print(f"NER总句子数: {len(all_sentences)}")

    # 打乱并划分
    random.seed(42)
    random.shuffle(all_sentences)
    n = len(all_sentences)
    train_end = int(n * ratios[0])
    val_end = train_end + int(n * ratios[1])

    splits = {
        "ner_train.json": all_sentences[:train_end],
        "ner_val.json": all_sentences[train_end:val_end],
        "ner_test.json": all_sentences[val_end:],
    }

    os.makedirs(processed_dir, exist_ok=True)
    for fname, data in splits.items():
        fpath = os.path.join(processed_dir, fname)
        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"  {fname}: {len(data)} 条")

    # 统计标签分布
    all_tags = []
    for s in all_sentences:
        all_tags.extend(s["tags"])
    tag_counts = Counter(all_tags)
    print(f"  标签分布: {dict(tag_counts)}")

    return None


def main():
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    raw_dir = os.path.join(base_dir, "data", "raw")
    processed_dir = os.path.join(base_dir, "data", "processed")

    print("=" * 50)
    print("处理 THUCNews 文本分类数据集")
    print("=" * 50)
    process_thucnews(raw_dir, processed_dir)

    print()
    print("=" * 50)
    print("处理 PeopleDaily NER 数据集")
    print("=" * 50)
    process_peopledaily(raw_dir, processed_dir)

    print()
    print("预处理完成!")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 运行预处理脚本**

Run: `python -m src.data.preprocess`
Expected: 生成6个JSON文件在 `data/processed/` 下，每个文件打印样本数量

- [ ] **Step 3: 提交**

```bash
git add src/data/preprocess.py
git commit -m "feat: add data preprocessing (clean + split + JSON export)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 4: PyTorch Dataset 定义

**Files:**
- Create: `src/data/dataset.py`

**Interfaces:**
- Consumes: `data/processed/news_*.json` 和 `data/processed/ner_*.json`
- Produces: `ClassificationDataset` 和 `NERDataset` 类，供 DataLoader 使用
- 每个 Dataset 返回 `dict` 包含 tokenized tensors

- [ ] **Step 1: 编写 dataset.py**

```python
"""
PyTorch Dataset 定义
分类数据集: {text, label} → {input_ids, attention_mask, label}
NER数据集:  {tokens, tags} → {input_ids, attention_mask, labels}
"""
import os
import json
import torch
from torch.utils.data import Dataset
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
        # BERT中文分词器可能把一个词拆成多个subtoken
        # 需要对齐标签
        aligned_tokens = []
        aligned_tags = []
        for token, tag in zip(tokens, raw_tags):
            subtokens = self.tokenizer.tokenize(token)
            if len(subtokens) == 0:
                continue
            aligned_tokens.extend(subtokens)
            # 第一个subtoken保留原标签，后续subtoken用X标记（或I标签）
            aligned_tags.append(self.tag2id.get(tag, 0))
            for _ in range(len(subtokens) - 1):
                aligned_tags.append(self.tag2id.get(tag, 0))

        # 截断（留空间给[CLS]和[SEP]）
        max_len = self.max_length - 2
        aligned_tokens = aligned_tokens[:max_len]
        aligned_tags = aligned_tags[:max_len]

        # 添加特殊token
        input_tokens = ["[CLS]"] + aligned_tokens + ["[SEP]"]
        label_ids = [-100] + aligned_tags + [-100]  # [CLS]和[SEP]用-100忽略

        # 转成ID
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
    from torch.utils.data import DataLoader

    tokenizer = BertTokenizer.from_pretrained(config["encoder"]["model_name"])
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
```

- [ ] **Step 2: 编写测试 test_data.py**

```python
"""测试数据模块"""
import os
import sys
import json
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import ClassificationDataset, NERDataset
from transformers import BertTokenizer


@pytest.fixture
def tokenizer():
    return BertTokenizer.from_pretrained("bert-base-chinese")


@pytest.fixture
def sample_cls_data(tmp_path):
    """创建临时分类测试数据"""
    data = [
        {"text": "今天股市大涨", "label": 0},
        {"text": "体育新闻播报", "label": 1},
    ]
    fpath = tmp_path / "news_test.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return str(fpath)


@pytest.fixture
def sample_ner_data(tmp_path):
    """创建临时NER测试数据"""
    data = [
        {"tokens": ["中国", "北京", "举行"], "tags": ["B-LOC", "B-LOC", "O"]},
        {"tokens": ["张三", "出席"], "tags": ["B-PER", "O"]},
    ]
    fpath = tmp_path / "ner_test.json"
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    return str(fpath)


def test_classification_dataset(tokenizer, sample_cls_data):
    ds = ClassificationDataset(sample_cls_data, tokenizer, max_length=128)
    assert len(ds) == 2
    batch = ds[0]
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch
    assert batch["input_ids"].shape[0] == 128  # padded
    assert batch["attention_mask"].sum() > 0  # 有真实token


def test_ner_dataset(tokenizer, sample_ner_data):
    tag2id = {"O": 0, "B-PER": 1, "B-LOC": 2}
    ds = NERDataset(sample_ner_data, tokenizer, tag2id, max_length=128)
    assert len(ds) == 2
    batch = ds[0]
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["labels"].shape[0] == 128
    # [CLS]标签是-100
    assert batch["labels"][0] == -100
```

- [ ] **Step 3: 运行测试验证**

Run: `pytest tests/test_data.py -v`
Expected: 2 tests pass

- [ ] **Step 4: 提交**

```bash
git add src/data/dataset.py tests/test_data.py
git commit -m "feat: add PyTorch Dataset classes and data tests

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 5: 数据探索 Notebook

**Files:**
- Create: `notebooks/01_data_exploration.ipynb`

**Interfaces:** 独立 notebook，不产生代码接口，用于可视化数据分布

- [ ] **Step 1: 编写 notebook**

```python
# Cell 1 (markdown): # 数据探索 — 了解你的数据
# Cell 2 (code):
import json
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# 加载分类数据
with open("../data/processed/news_train.json", "r", encoding="utf-8") as f:
    cls_data = json.load(f)

# 类别分布
labels = [item["label"] for item in cls_data]
label_names = ["体育", "财经", "科技", "教育", "时尚", "军事", "游戏", "房产", "娱乐", "时政"]
label_counts = Counter(labels)

plt.figure(figsize=(10, 5))
plt.bar([label_names[k] for k in sorted(label_counts.keys())],
        [label_counts[k] for k in sorted(label_counts.keys())])
plt.title("训练集类别分布")
plt.xticks(rotation=45)
plt.xlabel("类别")
plt.ylabel("样本数")
plt.tight_layout()
plt.show()

# Cell 3 (code): 文本长度分布
text_lens = [len(item["text"]) for item in cls_data]
plt.figure(figsize=(10, 5))
plt.hist(text_lens, bins=50, edgecolor="black")
plt.title("文本长度分布")
plt.xlabel("字符数")
plt.ylabel("样本数")
plt.axvline(x=256, color="red", linestyle="--", label="BERT max_length=256")
plt.legend()
plt.show()

# Cell 4 (code): NER数据标签分布
with open("../data/processed/ner_train.json", "r", encoding="utf-8") as f:
    ner_data = json.load(f)

all_tags = []
for sent in ner_data:
    all_tags.extend(sent["tags"])
tag_counts = Counter(all_tags)

plt.figure(figsize=(10, 5))
tags_sorted = sorted(tag_counts.keys())
plt.bar(tags_sorted, [tag_counts[t] for t in tags_sorted])
plt.title("NER标签分布")
plt.xticks(rotation=45)
plt.ylabel("出现次数")
plt.tight_layout()
plt.show()
```

- [ ] **Step 2: 提交**

```bash
git add notebooks/01_data_exploration.ipynb
git commit -m "docs: add data exploration notebook

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Day 2 — 共享编码器 + 训练器

### Task 6: 共享BERT编码器

**Files:**
- Create: `src/models/encoder.py`

**Interfaces:**
- Consumes: `config.yaml` 的 `encoder` 节
- Produces: `SharedEncoder` 类
  - `__init__(model_name, freeze_embeddings, freeze_layers)` → 加载并冻结BERT
  - `forward(input_ids, attention_mask, token_type_ids=None)` → `{"last_hidden_state": (B,S,768), "pooler_output": (B,768)}`

- [ ] **Step 1: 编写 encoder.py**

```python
"""
共享BERT编码器
加载预训练bert-base-chinese，冻结指定层
"""
import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class SharedEncoder(nn.Module):
    """共享BERT编码器，提供给分类和NER任务共用"""

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        freeze_embeddings: bool = True,
        freeze_layers: int = 6,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size  # 768

        # 冻结 embedding 层
        if freeze_embeddings:
            for param in self.bert.embeddings.parameters():
                param.requires_grad = False

        # 冻结前 N 层 Transformer encoder
        if freeze_layers > 0:
            for layer_idx in range(freeze_layers):
                for param in self.bert.encoder.layer[layer_idx].parameters():
                    param.requires_grad = False

        # 打印可训练参数
        total = sum(p.numel() for p in self.bert.parameters())
        trainable = sum(p.numel() for p in self.bert.parameters() if p.requires_grad)
        print(f"BERT 总参数: {total:,}")
        print(f"可训练参数: {trainable:,} ({100 * trainable / total:.1f}%)")

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        return {
            "last_hidden_state": outputs.last_hidden_state,  # (batch, seq_len, 768)
            "pooler_output": outputs.pooler_output,          # (batch, 768) — [CLS] after tanh
        }
```

- [ ] **Step 2: 编写测试 test_models.py**

```python
"""测试模型模块"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.encoder import SharedEncoder


def test_encoder_output_shape():
    encoder = SharedEncoder(model_name="bert-base-chinese")
    batch_size, seq_len = 4, 256
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = encoder(input_ids, attention_mask)

    assert output["last_hidden_state"].shape == (batch_size, seq_len, 768)
    assert output["pooler_output"].shape == (batch_size, 768)


def test_encoder_frozen_params():
    encoder = SharedEncoder(
        model_name="bert-base-chinese",
        freeze_embeddings=True,
        freeze_layers=6,
    )
    # Embedding层参数应冻结
    for name, param in encoder.bert.embeddings.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"
```

- [ ] **Step 3: 运行测试**

Run: `pytest tests/test_models.py::test_encoder_output_shape -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add src/models/encoder.py tests/test_models.py
git commit -m "feat: add shared BERT encoder with layer freezing

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 7: 通用训练器

**Files:**
- Create: `src/train/trainer.py`

**Interfaces:**
- Consumes: 任意 `nn.Module` 模型 + DataLoader + config
- Produces: `Trainer` 类
  - `train(model, train_dl, val_dl, ...)` → 返回训练历史 dict
  - 支持: 梯度累积、warmup、梯度裁剪、定期评估、checkpoint保存

- [ ] **Step 1: 编写 trainer.py**

```python
"""
通用训练器
支持分类和NER两种任务，统一训练循环
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np


class Trainer:
    """通用训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        config: dict,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.config = config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 2e-5),
        )

        # 学习率调度器
        total_steps = len(train_dataloader) * config.get("epochs", 5)
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.epochs = config.get("epochs", 5)
        self.eval_steps = config.get("eval_steps", 200)

        # 训练记录
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}

    def train(self, task_type: str = "classification"):
        """
        训练循环
        task_type: "classification" 或 "ner"
        """
        global_step = 0
        best_metric = 0.0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                # 将batch移到GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 前向传播
                outputs = self.model(**batch)
                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                # 梯度累积
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # 定期评估
                if global_step > 0 and global_step % self.eval_steps == 0:
                    val_loss, val_metric = self.evaluate(task_type)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_metric"].append(val_metric)

                    metric_name = "acc" if task_type == "classification" else "f1"
                    if val_metric > best_metric:
                        best_metric = val_metric
                        self.save_checkpoint(f"best_{task_type}.pt")
                        print(f"  ✓ 新最佳模型! {metric_name}: {val_metric:.4f}")

                    self.model.train()

            avg_loss = epoch_loss / len(self.train_dl)
            self.history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch + 1} 完成, 平均loss: {avg_loss:.4f}")

        return self.history

    def evaluate(self, task_type: str = "classification"):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                total_loss += loss.item()

                if task_type == "classification":
                    logits = outputs["logits"]
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(batch["label"].cpu().tolist())
                else:
                    # NER: 收集非-100位置的预测
                    predictions = outputs["predictions"]
                    labels = batch["labels"]
                    for pred_seq, label_seq, mask in zip(
                        predictions, labels, batch["attention_mask"]
                    ):
                        valid_len = mask.sum().item()
                        all_preds.extend(pred_seq[:valid_len])
                        all_labels.extend(
                            [l for l in label_seq[:valid_len].cpu().tolist() if l != -100]
                        )

        avg_loss = total_loss / len(self.val_dl)
        metric = self._compute_metric(all_preds, all_labels, task_type)
        return avg_loss, metric

    def _compute_metric(self, preds, labels, task_type):
        """计算评估指标"""
        from sklearn.metrics import accuracy_score, f1_score

        if task_type == "classification":
            return accuracy_score(labels, preds)
        else:
            # Macro F1，忽略O标签（标签0）
            labels_filtered = [(p, l) for p, l in zip(preds, labels) if l != 0 or p != 0]
            if not labels_filtered:
                return 0.0
            p_filtered, l_filtered = zip(*labels_filtered)
            return f1_score(l_filtered, p_filtered, average="macro", zero_division=0)

    def save_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename: str):
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
```

- [ ] **Step 2: 提交**

```bash
git add src/train/trainer.py
git commit -m "feat: add generic trainer with gradient accumulation and warmup

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Day 3 — 文本分类任务

### Task 8: 分类头模型

**Files:**
- Create: `src/models/classifier.py`

**Interfaces:**
- Consumes: `SharedEncoder` 的输出 `pooler_output` (B, 768)
- Produces: `TextClassifier` 类
  - `__init__(encoder, num_classes, dropout)` 
  - `forward(input_ids, attention_mask, label)` → `{"loss": tensor, "logits": (B, 10)}`

- [ ] **Step 1: 编写 classifier.py**

```python
"""
文本分类模型
BERT编码器 → [CLS]池化 → Dropout → Linear → 10类别
"""
import torch
import torch.nn as nn
from src.models.encoder import SharedEncoder


class TextClassifier(nn.Module):
    """BERT + 分类头"""

    def __init__(
        self,
        encoder: SharedEncoder,
        num_classes: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(encoder.hidden_size, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, input_ids, attention_mask, token_type_ids=None, label=None):
        # 编码
        encoded = self.encoder(input_ids, attention_mask, token_type_ids)
        pooled = encoded["pooler_output"]  # (batch, 768)

        # 分类
        logits = self.classifier(self.dropout(pooled))  # (batch, num_classes)

        # 损失
        loss = None
        if label is not None:
            loss = self.loss_fn(logits, label)

        return {"loss": loss, "logits": logits}
```

- [ ] **Step 2: 扩展 tests/test_models.py**

```python
def test_classifier_forward():
    from src.models.classifier import TextClassifier
    encoder = SharedEncoder(model_name="bert-base-chinese")
    model = TextClassifier(encoder, num_classes=10)

    batch_size, seq_len = 4, 128
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 10, (batch_size,))

    output = model(input_ids, attention_mask, label=labels)
    assert "loss" in output
    assert "logits" in output
    assert output["logits"].shape == (batch_size, 10)
    assert output["loss"].requires_grad
```

- [ ] **Step 3: 提交**

```bash
git add src/models/classifier.py tests/test_models.py
git commit -m "feat: add text classification head with [CLS] pooling

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 9: 分类任务训练脚本

**Files:**
- Create: `src/train/train_classifier.py`

**Interfaces:**
- Consumes: `configs/config.yaml`, `data/processed/news_*.json`, `SharedEncoder`, `TextClassifier`, `Trainer`
- Produces: `checkpoints/best_classification.pt` + 训练日志

- [ ] **Step 1: 编写 train_classifier.py**

```python
"""
训练文本分类模型
"""
import os
import sys
import yaml
import torch
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.data.dataset import ClassificationDataset, create_dataloaders
from src.train.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # 加载配置
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")

    # 创建 DataLoader
    dataloaders, tokenizer = create_dataloaders(processed_dir, config)

    # 创建模型
    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=config["encoder"]["freeze_embeddings"],
        freeze_layers=config["encoder"]["freeze_layers"],
    )
    model = TextClassifier(
        encoder=encoder,
        num_classes=config["classifier"]["num_classes"],
        dropout=config["classifier"]["dropout"],
    )

    print(f"分类模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["cls_train"],
        val_dataloader=dataloaders["cls_val"],
        config=config["classifier"],
        checkpoint_dir=os.path.join(base_dir, "checkpoints"),
    )

    history = trainer.train(task_type="classification")

    # 测试集最终评估
    print("\n" + "=" * 50)
    print("测试集评估")
    print("=" * 50)

    trainer.load_checkpoint("best_classification.pt")
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib.pyplot as plt
    import seaborn as sns

    trainer.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloaders["cls_test"]:
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(**batch)
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    label_names = config["classifier"]["label_names"]
    print(classification_report(all_labels, all_preds, target_names=label_names))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
    plt.title("分类混淆矩阵")
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "checkpoints", "confusion_matrix.png"))
    print("混淆矩阵已保存至 checkpoints/confusion_matrix.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/train/train_classifier.py
git commit -m "feat: add classification training script with evaluation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 10: 分类实验 Notebook

**Files:**
- Create: `notebooks/02_text_classification.ipynb`

- [ ] **Step 1: 编写 notebook**

```python
# Cell 1 (markdown): # 文本分类实验 — BERT微调
# Cell 2 (code):
# 加载训练好的模型
from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
import torch

encoder = SharedEncoder(model_name="bert-base-chinese")
model = TextClassifier(encoder, num_classes=10)
model.load_state_dict(torch.load("../checkpoints/best_classification.pt", map_location="cpu"))
model.eval()

# Cell 3 (code):
# 单条预测测试
from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

texts = [
    "今日NBA总决赛，湖人队大胜凯尔特人",
    "央行宣布降息50个基点，A股大幅反弹",
    "新款手机发布，搭载最新AI芯片",
]
label_names = ["体育", "财经", "科技", "教育", "时尚", "军事", "游戏", "房产", "娱乐", "时政"]

for text in texts:
    encoding = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
        probs = torch.softmax(output["logits"], dim=-1)[0]
        pred = torch.argmax(probs).item()
    print(f"文本: {text}")
    print(f"预测: {label_names[pred]} (置信度: {probs[pred]:.2%})")
    print()
```

- [ ] **Step 2: 提交**

```bash
git add notebooks/02_text_classification.ipynb
git commit -m "docs: add classification experiment notebook

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Day 4 — NER任务

### Task 11: 评估指标模块

**Files:**
- Create: `src/evaluate/metrics.py`

**Interfaces:**
- Consumes: 预测标签列表 + 真实标签列表
- Produces: 各种指标函数
  - `compute_classification_metrics(y_true, y_pred, label_names)` → dict
  - `compute_ner_metrics(y_true, y_pred, id2tag)` → dict (per-entity + macro F1)

- [ ] **Step 1: 编写 metrics.py**

```python
"""
评估指标模块
支持分类和NER两种任务的指标计算
"""
from typing import List, Dict, Tuple
from collections import defaultdict
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)


def compute_classification_metrics(
    y_true: List[int],
    y_pred: List[int],
    label_names: List[str] = None,
) -> Dict:
    """计算分类指标"""
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )

    metrics = {
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
    }

    if label_names:
        metrics["per_class"] = classification_report(
            y_true, y_pred,
            target_names=label_names,
            output_dict=True,
            zero_division=0,
        )
        metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()

    return metrics


def compute_ner_metrics(
    y_true: List[List[str]],
    y_pred: List[List[str]],
    id2tag: List[str],
) -> Dict:
    """
    计算NER指标（per-entity F1 + macro F1）
    使用BIO标注的严格匹配：实体边界和类型都正确才算正确
    """
    def extract_entities(tag_seq: List[str]) -> List[Tuple[str, int, int]]:
        """从BIO标签序列中提取实体"""
        entities = []
        current_entity = None
        start_idx = -1

        for i, tag in enumerate(tag_seq):
            if tag.startswith("B-"):
                if current_entity:
                    entities.append((current_entity, start_idx, i))
                current_entity = tag[2:]  # 去掉B-前缀
                start_idx = i
            elif tag.startswith("I-"):
                entity_type = tag[2:]
                if current_entity != entity_type:
                    if current_entity:
                        entities.append((current_entity, start_idx, i))
                    current_entity = None
            else:  # O标签
                if current_entity:
                    entities.append((current_entity, start_idx, i))
                    current_entity = None

        if current_entity:
            entities.append((current_entity, start_idx, len(tag_seq)))

        return entities

    # 收集所有实体
    true_entities_all = defaultdict(list)  # type → list of (type, start, end)
    pred_entities_all = defaultdict(list)

    for true_tags, pred_tags in zip(y_true, y_pred):
        for ent_type, start, end in extract_entities(true_tags):
            true_entities_all[ent_type].append((ent_type, start, end))
        for ent_type, start, end in extract_entities(pred_tags):
            pred_entities_all[ent_type].append((ent_type, start, end))

    # 计算每种实体的P/R/F1
    entity_types = set(list(true_entities_all.keys()) + list(pred_entities_all.keys()))
    per_entity = {}

    all_true_flat = []
    all_pred_flat = []

    for etype in entity_types:
        true_set = set(true_entities_all[etype])
        pred_set = set(pred_entities_all[etype])

        tp = len(true_set & pred_set)
        fp = len(pred_set - true_set)
        fn = len(true_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_entity[etype] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": len(true_set),
        }

        all_true_flat.extend([etype] * len(true_set))
        all_pred_flat.extend([etype] * len(true_set & pred_set) + ["O"] * (len(pred_set) - len(true_set & pred_set)))

    # Macro F1
    f1s = [v["f1"] for v in per_entity.values() if v["support"] > 0]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "per_entity": per_entity,
        "macro_f1": macro_f1,
    }


def bio_decode(tag_ids: List[int], id2tag: List[str]) -> List[str]:
    """将标签ID序列转为标签字符串序列"""
    return [id2tag[t] if t >= 0 else "O" for t in tag_ids]
```

- [ ] **Step 2: 提交**

```bash
git add src/evaluate/metrics.py
git commit -m "feat: add evaluation metrics (classification + NER BIO scoring)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 12: NER CRF标注头模型

**Files:**
- Create: `src/models/ner_tagger.py`

**Interfaces:**
- Consumes: `SharedEncoder` 的输出 `last_hidden_state` (B, S, 768)
- Produces: `NERTagger` 类
  - `forward(input_ids, attention_mask, labels=None)` → `{"loss": tensor, "predictions": List[List[int]]}`
  - 使用 `torchcrf.CRF` 做序列解码

- [ ] **Step 1: 编写 ner_tagger.py**

```python
"""
NER序列标注模型
BERT编码器 → 每Token Linear → CRF → BIO标签序列
"""
import torch
import torch.nn as nn
from torchcrf import CRF
from src.models.encoder import SharedEncoder


class NERTagger(nn.Module):
    """BERT + Linear + CRF 序列标注"""

    def __init__(
        self,
        encoder: SharedEncoder,
        num_tags: int = 7,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_tags = num_tags
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(encoder.hidden_size, num_tags)
        self.crf = CRF(num_tags, batch_first=True)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        # 编码
        encoded = self.encoder(input_ids, attention_mask, token_type_ids)
        hidden = encoded["last_hidden_state"]  # (batch, seq_len, 768)

        # 每Token发射分数
        emissions = self.linear(self.dropout(hidden))  # (batch, seq_len, num_tags)

        # CRF mask
        mask = attention_mask.bool()

        loss = None
        predictions = None

        if labels is not None:
            # 训练模式: 计算CRF负对数似然
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
        else:
            # 推理模式: CRF解码
            predictions = self.crf.decode(emissions, mask=mask)

        return {"loss": loss, "predictions": predictions, "emissions": emissions}
```

- [ ] **Step 2: 扩展 tests/test_models.py**

```python
def test_ner_tagger_forward():
    from src.models.ner_tagger import NERTagger

    encoder = SharedEncoder(model_name="bert-base-chinese")
    model = NERTagger(encoder, num_tags=7)

    batch_size, seq_len = 4, 64
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 7, (batch_size, seq_len))
    # [CLS]和[SEP]标签置-100
    labels[:, 0] = -100
    labels[:, -1] = -100

    # 训练模式
    output = model(input_ids, attention_mask, labels=labels)
    assert "loss" in output
    assert output["loss"].requires_grad

    # 推理模式
    with torch.no_grad():
        output_infer = model(input_ids, attention_mask, labels=None)
    assert "predictions" in output_infer
    assert len(output_infer["predictions"]) == batch_size
```

- [ ] **Step 3: 提交**

```bash
git add src/models/ner_tagger.py tests/test_models.py
git commit -m "feat: add NER sequence tagger with CRF decoding

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 13: NER训练脚本

**Files:**
- Create: `src/train/train_ner.py`

**Interfaces:**
- Consumes: `config.yaml`, `data/processed/ner_*.json`, `SharedEncoder`, `NERTagger`, `Trainer`
- Produces: `checkpoints/best_ner.pt`

- [ ] **Step 1: 编写 train_ner.py**

```python
"""
训练NER序列标注模型
"""
import os
import sys
import yaml
import torch
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.encoder import SharedEncoder
from src.models.ner_tagger import NERTagger
from src.data.dataset import create_dataloaders
from src.train.trainer import Trainer
from src.evaluate.metrics import compute_ner_metrics


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs", "config.yaml"
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")

    # 创建 DataLoader
    dataloaders, tokenizer = create_dataloaders(processed_dir, config)

    # 创建模型
    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=config["encoder"]["freeze_embeddings"],
        freeze_layers=config["encoder"]["freeze_layers"],
    )
    model = NERTagger(
        encoder=encoder,
        num_tags=config["ner"]["num_tags"],
        dropout=config["ner"]["dropout"],
    )

    print(f"NER模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["ner_train"],
        val_dataloader=dataloaders["ner_val"],
        config=config["ner"],
        checkpoint_dir=os.path.join(base_dir, "checkpoints"),
    )

    history = trainer.train(task_type="ner")

    # 测试集最终评估
    print("\n" + "=" * 50)
    print("NER测试集评估")
    print("=" * 50)

    trainer.load_checkpoint("best_ner.pt")
    trainer.model.eval()
    id2tag = config["ner"]["id2tag"]

    all_true_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for batch in dataloaders["ner_test"]:
            batch_gpu = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(**batch_gpu)

            labels = batch["labels"]
            for pred_ids, label_ids, mask in zip(
                outputs["predictions"], labels, batch["attention_mask"]
            ):
                valid_len = mask.sum().item()
                true_seq = []
                pred_seq = []
                for i in range(valid_len):
                    lid = label_ids[i].item()
                    pid = pred_ids[i] if i < len(pred_ids) else 0
                    if lid != -100:
                        true_seq.append(id2tag[lid])
                        pred_seq.append(id2tag[pid])
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    metrics = compute_ner_metrics(all_true_tags, all_pred_tags, id2tag)

    print(f"\nMacro F1: {metrics['macro_f1']:.4f}")
    print("\nPer-entity 指标:")
    for entity, scores in metrics["per_entity"].items():
        print(f"  {entity:10s}: P={scores['precision']:.3f}  R={scores['recall']:.3f}  F1={scores['f1']:.3f}  (support={scores['support']})")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 提交**

```bash
git add src/train/train_ner.py
git commit -m "feat: add NER training script with per-entity evaluation

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 14: NER实验 Notebook

**Files:**
- Create: `notebooks/03_ner_training.ipynb`

- [ ] **Step 1: 编写 notebook（NER预测可视化）**

```python
# Cell 1 (markdown): # NER序列标注实验 — BERT+CRF
# Cell 2 (code):
from src.models.encoder import SharedEncoder
from src.models.ner_tagger import NERTagger
from transformers import BertTokenizer
import torch

encoder = SharedEncoder(model_name="bert-base-chinese")
model = NERTagger(encoder, num_tags=7)
model.load_state_dict(torch.load("../checkpoints/best_ner.pt", map_location="cpu"))
model.eval()
tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

# Cell 3 (code):
def predict_ner(text):
    id2tag = ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
    encoding = tokenizer(text, max_length=256, padding="max_length", truncation=True, return_tensors="pt")
    with torch.no_grad():
        output = model(**encoding)
    tokens = tokenizer.convert_ids_to_tokens(encoding["input_ids"][0])
    pred_tags = [id2tag[t] for t in output["predictions"][0]]
    # 只显示非padding部分
    valid_len = encoding["attention_mask"][0].sum().item()
    return list(zip(tokens[:valid_len], pred_tags[:valid_len]))

texts = [
    "华为公司在深圳举办了开发者大会",
    "李克强总理访问了上海浦东新区",
]
for text in texts:
    print(f"输入: {text}")
    result = predict_ner(text)
    entities = [(t, tag) for t, tag in result if tag != "O"]
    print(f"实体: {entities}")
    print()
```

- [ ] **Step 2: 提交**

```bash
git add notebooks/03_ner_training.ipynb
git commit -m "docs: add NER experiment notebook

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## Day 5 — 整合 + 演示

### Task 15: 多任务组合模型

**Files:**
- Create: `src/models/multitask_model.py`

**Interfaces:**
- Consumes: `SharedEncoder`, `TextClassifier`, `NERTagger`
- Produces: `MultiTaskModel` 类
  - `__init__(encoder, classifier_head, ner_head)` 
  - `predict(text, tokenizer, max_length)` → `{"classification": (label_name, confidence), "entities": [(word, entity_type, position)]}`
  - `save(path)` / `load(path)` 统一存储/加载

- [ ] **Step 1: 编写 multitask_model.py**

```python
"""
多任务组合模型
一个共享编码器 + 分类头 + NER头，统一推理接口
"""
import torch
from typing import List, Tuple, Dict
from transformers import BertTokenizer
from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.models.ner_tagger import NERTagger


class MultiTaskModel:
    """多任务推理接口"""

    def __init__(
        self,
        encoder: SharedEncoder,
        classifier_head: TextClassifier = None,
        ner_head: NERTagger = None,
        label_names: List[str] = None,
        id2tag: List[str] = None,
        max_length: int = 256,
    ):
        self.encoder = encoder
        self.classifier_head = classifier_head
        self.ner_head = ner_head
        self.label_names = label_names or ["体育", "财经", "科技", "教育", "时尚", "军事", "游戏", "房产", "娱乐", "时政"]
        self.id2tag = id2tag or ["O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"]
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)
        if self.classifier_head:
            self.classifier_head.to(self.device)
        if self.ner_head:
            self.ner_head.to(self.device)

    def predict(self, text: str) -> Dict:
        """
        单条文本推理
        返回: {
            "text": 原始文本,
            "classification": {"label": "体育", "confidence": 0.95},
            "entities": [{"word": "湖人", "type": "ORG", "start": 0, "end": 2}],
        }
        """
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        result = {"text": text}

        # 分类预测
        if self.classifier_head is not None:
            with torch.no_grad():
                encoded = self.encoder(input_ids, attention_mask)
                logits = self.classifier_head.classifier(
                    self.classifier_head.dropout(encoded["pooler_output"])
                )
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
            result["classification"] = {
                "label": self.label_names[pred_idx],
                "confidence": round(probs[pred_idx].item(), 4),
            }

        # NER预测
        if self.ner_head is not None:
            with torch.no_grad():
                encoded = self.encoder(input_ids, attention_mask)
                emissions = self.ner_head.linear(
                    self.ner_head.dropout(encoded["last_hidden_state"])
                )
                mask = attention_mask.bool()
                predictions = self.ner_head.crf.decode(emissions, mask=mask)[0]

            # 解析实体
            tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
            valid_len = attention_mask[0].sum().item()
            entities = self._parse_bio(tokens[1:valid_len-1], predictions[1:valid_len-1])
            result["entities"] = entities

        return result

    def _parse_bio(
        self, tokens: List[str], tag_ids: List[int]
    ) -> List[Dict]:
        """从BIO标签序列中解析实体"""
        entities = []
        current_tokens = []
        current_type = None
        char_pos = 0

        for i, (token, tag_id) in enumerate(zip(tokens, tag_ids)):
            tag = self.id2tag[tag_id] if tag_id < len(self.id2tag) else "O"

            if tag.startswith("B-"):
                # 保存前一个实体
                if current_tokens:
                    entities.append({
                        "word": "".join(current_tokens).replace("##", ""),
                        "type": current_type,
                        "start": i - len(current_tokens),
                        "end": i,
                    })
                current_type = tag[2:]
                current_tokens = [token]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "word": "".join(current_tokens).replace("##", ""),
                        "type": current_type,
                        "start": i - len(current_tokens),
                        "end": i,
                    })
                current_tokens = []
                current_type = None

        if current_tokens:
            entities.append({
                "word": "".join(current_tokens).replace("##", ""),
                "type": current_type,
                "start": len(tokens) - len(current_tokens),
                "end": len(tokens),
            })

        return entities

    def save(self, path: str):
        """保存完整模型"""
        checkpoint = {
            "encoder": self.encoder.state_dict(),
            "classifier": self.classifier_head.state_dict() if self.classifier_head else None,
            "ner": self.ner_head.state_dict() if self.ner_head else None,
        }
        torch.save(checkpoint, path)
        print(f"模型已保存至 {path}")

    @classmethod
    def load(cls, path: str, config: dict):
        """加载完整模型"""
        encoder = SharedEncoder(
            model_name=config["encoder"]["model_name"],
            freeze_embeddings=False,  # 推理时不需要冻结
            freeze_layers=0,
        )
        classifier_head = TextClassifier(encoder, num_classes=config["classifier"]["num_classes"])
        ner_head = NERTagger(encoder, num_tags=config["ner"]["num_tags"])

        checkpoint = torch.load(path, map_location="cpu")
        encoder.load_state_dict(checkpoint["encoder"])
        if checkpoint["classifier"]:
            classifier_head.load_state_dict(checkpoint["classifier"])
        if checkpoint["ner"]:
            ner_head.load_state_dict(checkpoint["ner"])

        return cls(
            encoder=encoder,
            classifier_head=classifier_head,
            ner_head=ner_head,
            label_names=config["classifier"]["label_names"],
            id2tag=config["ner"]["id2tag"],
            max_length=config["encoder"]["max_length"],
        )
```

- [ ] **Step 2: 提交**

```bash
git add src/models/multitask_model.py
git commit -m "feat: add multitask model with unified inference interface

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 16: 统一入口 main.py

**Files:**
- Create: `main.py`

**Interfaces:**
- CLI入口，支持三个子命令: `train`, `evaluate`, `predict`
- `python main.py train --task cls` 训练分类
- `python main.py train --task ner` 训练NER
- `python main.py predict --text "华为在深圳开发布会"`

- [ ] **Step 1: 编写 main.py**

```python
"""
统一入口脚本
用法:
  python main.py train --task cls         # 训练分类模型
  python main.py train --task ner         # 训练NER模型
  python main.py train --task all         # 训练两个模型
  python main.py predict --text "..."     # 单条推理
  python main.py demo                     # 启动Gradio界面
"""
import os
import sys
import yaml
import argparse


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_train(args):
    """训练命令"""
    config = load_config()
    if args.task in ("cls", "all"):
        print("\n" + "=" * 60)
        print("训练文本分类模型")
        print("=" * 60)
        from src.train.train_classifier import main as train_cls
        train_cls()

    if args.task in ("ner", "all"):
        print("\n" + "=" * 60)
        print("训练NER模型")
        print("=" * 60)
        from src.train.train_ner import main as train_ner
        train_ner()


def cmd_predict(args):
    """预测命令"""
    config = load_config()
    from src.models.multitask_model import MultiTaskModel

    # 先尝试加载分类模型
    cls_path = os.path.join("checkpoints", "best_classification.pt")
    if not os.path.exists(cls_path):
        print("分类模型未找到，仅加载NER模型")
        cls_path = None

    # 尝试加载NER模型
    ner_path = os.path.join("checkpoints", "best_ner.pt")
    if not os.path.exists(ner_path):
        print("NER模型未找到，仅加载分类模型")
        ner_path = None

    if not cls_path and not ner_path:
        print("错误: 未找到任何已训练的模型。请先运行 python main.py train")
        return

    # 简单加载方式（使用各自checkpoint）
    from src.models.encoder import SharedEncoder
    from src.models.classifier import TextClassifier
    from src.models.ner_tagger import NERTagger
    import torch

    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )

    classifier_head = TextClassifier(encoder, num_classes=config["classifier"]["num_classes"])
    if cls_path:
        classifier_head.load_state_dict(torch.load(cls_path, map_location="cpu"))

    ner_head = NERTagger(encoder, num_tags=config["ner"]["num_tags"])
    if ner_path:
        ner_head.load_state_dict(torch.load(ner_path, map_location="cpu"))

    model = MultiTaskModel(
        encoder=encoder,
        classifier_head=classifier_head if cls_path else None,
        ner_head=ner_head if ner_path else None,
        label_names=config["classifier"]["label_names"],
        id2tag=config["ner"]["id2tag"],
        max_length=config["encoder"]["max_length"],
    )

    result = model.predict(args.text)
    print("\n" + "=" * 50)
    print(f"输入: {result['text']}")
    print("=" * 50)

    if "classification" in result:
        cls_result = result["classification"]
        print(f"\n📊 分类结果: {cls_result['label']} (置信度: {cls_result['confidence']:.2%})")

    if "entities" in result:
        entities = result["entities"]
        print(f"\n🏷 识别实体 ({len(entities)}个):")
        if entities:
            for ent in entities:
                print(f"  [{ent['type']}] {ent['word']} (位置 {ent['start']}-{ent['end']})")
        else:
            print("  未识别到实体")


def cmd_demo(args):
    """启动Gradio演示"""
    from src.app.demo import main as run_demo
    run_demo()


def main():
    parser = argparse.ArgumentParser(description="NLP多任务学习系统")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # train
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument("--task", choices=["cls", "ner", "all"], default="all")

    # predict
    pred_parser = subparsers.add_parser("predict", help="单条推理")
    pred_parser.add_argument("--text", type=str, required=True, help="输入文本")

    # demo
    subparsers.add_parser("demo", help="启动Gradio Web演示")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试 CLI**

Run: `python main.py --help`
Expected: 显示帮助信息

- [ ] **Step 3: 提交**

```bash
git add main.py
git commit -m "feat: add unified CLI entry (train/predict/demo)

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 17: Gradio Web演示界面

**Files:**
- Create: `src/app/demo.py`

**Interfaces:**
- Produces: 本地Web服务（默认 http://127.0.0.1:7860）
  - 输入框: 任意中文文本
  - 输出区1: 分类结果（类别 + 置信度）
  - 输出区2: NER高亮文本 + 实体列表

- [ ] **Step 1: 编写 demo.py**

```python
"""
Gradio Web演示界面
输入文本 → 同时展示分类结果和NER实体
"""
import os
import sys
import yaml
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.models.ner_tagger import NERTagger
from src.models.multitask_model import MultiTaskModel


def load_models():
    """加载已训练的模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )

    cls_path = os.path.join(base_dir, "checkpoints", "best_classification.pt")
    ner_path = os.path.join(base_dir, "checkpoints", "best_ner.pt")

    classifier_head = TextClassifier(encoder, num_classes=config["classifier"]["num_classes"])
    if os.path.exists(cls_path):
        classifier_head.load_state_dict(torch.load(cls_path, map_location="cpu"))
        print(f"✓ 分类模型已加载")
    else:
        classifier_head = None
        print("⚠ 分类模型未找到")

    ner_head = NERTagger(encoder, num_tags=config["ner"]["num_tags"])
    if os.path.exists(ner_path):
        ner_head.load_state_dict(torch.load(ner_path, map_location="cpu"))
        print(f"✓ NER模型已加载")
    else:
        ner_head = None
        print("⚠ NER模型未找到")

    model = MultiTaskModel(
        encoder=encoder,
        classifier_head=classifier_head,
        ner_head=ner_head,
        label_names=config["classifier"]["label_names"],
        id2tag=config["ner"]["id2tag"],
        max_length=config["encoder"]["max_length"],
    )
    return model


# 全局模型实例
model = None


def analyze_text(text: str):
    """分析文本，返回分类和NER结果"""
    global model
    if model is None:
        model = load_models()

    if not text.strip():
        return "请输入文本", "请输入文本", ""

    result = model.predict(text)

    # 分类结果
    if "classification" in result:
        cls_info = f"### 📊 文本分类\n\n**{result['classification']['label']}** (置信度: {result['classification']['confidence']:.2%})"
    else:
        cls_info = "分类模型未加载"

    # NER结果
    if "entities" in result:
        entities = result["entities"]
        if entities:
            ner_info = "### 🏷 命名实体识别\n\n"
            ner_info += "| 实体 | 类型 |\n|------|------|\n"
            for ent in entities:
                ner_info += f"| {ent['word']} | {ent['type']} |\n"

            # 高亮文本
            highlighted = text
            entity_parts = []
            for ent in sorted(entities, key=lambda x: x["start"], reverse=True):
                start, end = ent["start"], ent["end"]
                entity_text = highlighted[start:end]
                highlighted = f"{highlighted[:start]}[{entity_text}]({ent['type']}){highlighted[end:]}"
            ner_info += f"\n**原文标注**: {highlighted}"
        else:
            ner_info = "### 🏷 命名实体识别\n\n未识别到实体"
    else:
        ner_info = "NER模型未加载"

    return cls_info, ner_info


def create_demo():
    with gr.Blocks(title="NLP多任务学习系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # 🧠 Semantic-Insight
            基于 BERT 的文本分类 + 命名实体识别 联合推理
            """
        )

        with gr.Row():
            input_text = gr.Textbox(
                label="输入文本",
                placeholder="请输入中文文本进行分析...",
                lines=3,
            )

        with gr.Row():
            submit_btn = gr.Button("🔍 分析", variant="primary")
            clear_btn = gr.Button("清除")

        with gr.Row():
            with gr.Column():
                cls_output = gr.Markdown(label="分类结果")

        with gr.Row():
            ner_output = gr.Markdown(label="NER结果")

        # 示例
        gr.Examples(
            examples=[
                ["华为技术有限公司在深圳举办了2024年开发者大会"],
                ["今天NBA总决赛，洛杉矶湖人队大胜波士顿凯尔特人"],
                ["央行宣布降息50个基点，A股市场大幅反弹"],
                ["北京大学和清华大学联合举办人工智能论坛"],
            ],
            inputs=input_text,
        )

        submit_btn.click(
            fn=analyze_text,
            inputs=input_text,
            outputs=[cls_output, ner_output],
        )
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[input_text, cls_output, ner_output],
        )

    return demo


def main():
    global model
    model = load_models()
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 测试界面启动**

Run: `python main.py demo`
Expected: Gradio启动在 http://127.0.0.1:7860

- [ ] **Step 3: 提交**

```bash
git add src/app/demo.py
git commit -m "feat: add Gradio web demo interface

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

### Task 18: 文档与收尾

**Files:**
- Modify: `README.md` — 写项目说明
- Create: `docs/report.md` — 实训报告框架

- [ ] **Step 1: 编写 README.md**

```markdown
# Semantic-Insight

基于 PyTorch + BERT 的中文双任务NLP系统：**文本分类** + **命名实体识别**。

## 项目结构

\`\`\`
src/
├── data/          # 数据下载、预处理、Dataset定义
├── models/        # BERT编码器、分类头、NER标注头、多任务模型
├── train/         # 通用训练器 + 各任务训练脚本
├── evaluate/      # 评估指标（准确率、F1、混淆矩阵）
└── app/           # Gradio Web演示界面
\`\`\`

## 快速开始

\`\`\`bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载并处理数据
python -m src.data.download
python -m src.data.preprocess

# 3. 训练模型
python main.py train --task cls   # 分类
python main.py train --task ner   # NER
python main.py train --task all   # 全部

# 4. 推理
python main.py predict --text "华为在深圳举办开发者大会"

# 5. 启动Web演示
python main.py demo
\`\`\`

## 技术栈

- **框架**: PyTorch 2.0 + HuggingFace Transformers
- **编码器**: bert-base-chinese（12层Transformer）
- **分类头**: [CLS] → Dropout → Linear(768→10)
- **NER头**: 每Token → Linear → CRF序列解码
- **界面**: Gradio 4.0

## 性能

| 任务 | 指标 | 目标 |
|------|------|------|
| 文本分类 | Accuracy | ≥90% |
| NER | Macro F1 | ≥80% |
```

- [ ] **Step 2: 编写 docs/report.md**

```markdown
# Semantic-Insight 实训报告

## 摘要
本项目基于共享BERT编码器，实现了文本分类和命名实体识别两个NLP任务的联合训练和推理...

## 1. 项目背景
（说明NLP多任务学习的意义）

## 2. 技术方案
### 2.1 模型架构
### 2.2 数据集
### 2.3 训练策略

## 3. 实验结果
### 3.1 文本分类
### 3.2 命名实体识别
### 3.3 消融实验

## 4. 总结与展望

## 附录: 代码仓库
```

- [ ] **Step 3: 最终提交**

```bash
git add README.md docs/report.md
git commit -m "docs: complete README and training report template

Co-Authored-By: Claude <noreply@anthropic.com>"
```

---

## 任务依赖关系

```
Task 1 (脚手架)
 ├─→ Task 2 (数据下载)
 │    └─→ Task 3 (预处理)
 │         └─→ Task 4 (Dataset) ──→ Task 5 (数据Notebook)
 │              └─→ Task 7 (Trainer依赖Dataset接口)
 │
 ├─→ Task 6 (Encoder) ──→ Task 8 (分类头) ──→ Task 9 (分类训练)
 │              │              │                  └─→ Task 10 (分类Notebook)
 │              │              │
 │              ├──────────────┤
 │              │              │
 │              └─→ Task 12 (NER头) ──→ Task 13 (NER训练)
 │                     │                     └─→ Task 14 (NER Notebook)
 │                     │
 │              Task 11 (指标模块) ←── 被 Task 9, 13 使用
 │
 └─→ Task 15 (多任务模型) ←── 依赖 Task 8 + Task 12
      └─→ Task 16 (main.py) ←── 依赖 Task 15
           └─→ Task 17 (Gradio) ←── 依赖 Task 15
                └─→ Task 18 (文档)
```

---

## 验收检查清单

- [ ] `python -m src.data.download` 可成功下载/生成数据
- [ ] `python -m src.data.preprocess` 可成功处理并生成JSON
- [ ] `pytest tests/ -v` 所有测试通过
- [ ] `python main.py train --task cls` 可完整训练分类模型，accuracy ≥ 90%
- [ ] `python main.py train --task ner` 可完整训练NER模型，macro F1 ≥ 80%
- [ ] `python main.py predict --text "华为在深圳开发布会"` 返回正确JSON结果
- [ ] `python main.py demo` 启动Gradio，浏览器可访问并正确推理
- [ ] README.md 包含完整的快速开始指南
- [ ] 代码有中文注释