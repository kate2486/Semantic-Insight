"""测试数据模块"""
import os
import sys
import json
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.data.dataset import ClassificationDataset, NERDataset
from transformers import BertTokenizer


@pytest.fixture(scope="module")
def tokenizer():
    return BertTokenizer.from_pretrained("bert-base-chinese", local_files_only=True)


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


def test_classification_dataset_length(tokenizer, sample_cls_data):
    """测试分类数据集长度"""
    ds = ClassificationDataset(sample_cls_data, tokenizer, max_length=128)
    assert len(ds) == 2


def test_classification_dataset_batch(tokenizer, sample_cls_data):
    """测试分类数据集返回结构"""
    ds = ClassificationDataset(sample_cls_data, tokenizer, max_length=128)
    batch = ds[0]
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "label" in batch
    assert batch["input_ids"].shape[0] == 128  # padded to max_length
    assert batch["attention_mask"].sum() > 0  # 有真实token


def test_ner_dataset_length(tokenizer, sample_ner_data):
    """测试NER数据集长度"""
    tag2id = {"O": 0, "B-PER": 1, "B-LOC": 2}
    ds = NERDataset(sample_ner_data, tokenizer, tag2id, max_length=128)
    assert len(ds) == 2


def test_ner_dataset_cls_label(tokenizer, sample_ner_data):
    """测试NER数据集[CLS]标签为O（0）"""
    tag2id = {"O": 0, "B-PER": 1, "B-LOC": 2}
    ds = NERDataset(sample_ner_data, tokenizer, tag2id, max_length=128)
    batch = ds[0]
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["labels"].shape[0] == 128
    # [CLS]是第一个token，标签应为O（0）
    assert batch["labels"][0] == 0
