"""
数据预处理模块
清洗文本 → 格式统一 → train/val/test 划分 → 保存为 JSON
"""
import os
import json
import re
import random
from collections import Counter


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
                if len(text) < 2:  # 过滤太短的文本
                    continue
                samples.append({"text": text, "label": label2id[cat]})
            except Exception as e:
                print(f"  读取失败 {fpath}: {e}")

    print(f"THUCNews 总样本数: {len(samples)}")

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
                # 用tab分隔
                parts = line.split("\t")
                if len(parts) >= 2:
                    tokens.append(parts[0])
                    # 清理标签（有些数据集用 B_ORG 有些用 B-ORG）
                    tag = parts[1].replace("_", "-").strip()
                    tags.append(tag)
            # 处理文件末尾可能没有空行的情况
            if tokens:
                all_sentences.append({"tokens": tokens, "tags": tags})

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
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
