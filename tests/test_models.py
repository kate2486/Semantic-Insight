"""测试模型模块"""
import os
import sys
import torch
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.models.encoder import SharedEncoder


@pytest.fixture(scope="module")
def encoder():
    return SharedEncoder(model_name="bert-base-chinese")


def test_encoder_output_shape(encoder):
    """测试编码器输出维度"""
    batch_size, seq_len = 4, 256
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = encoder(input_ids, attention_mask)

    assert output["last_hidden_state"].shape == (batch_size, seq_len, 768)
    assert output["pooler_output"].shape == (batch_size, 768)


def test_encoder_frozen_params(encoder):
    """测试 Embedding 层参数已冻结"""
    for name, param in encoder.bert.embeddings.named_parameters():
        assert not param.requires_grad, f"{name} should be frozen"


def test_encoder_trainable_params_exist(encoder):
    """测试存在可训练参数（后6层）"""
    trainable = sum(1 for p in encoder.bert.parameters() if p.requires_grad)
    assert trainable > 0, "Should have some trainable parameters (later layers)"


def test_classifier_forward(encoder):
    """测试分类头前向传播"""
    from src.models.classifier import TextClassifier

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


def test_classifier_inference(encoder):
    """测试分类头推理模式（无标签）"""
    from src.models.classifier import TextClassifier

    model = TextClassifier(encoder, num_classes=10)
    model.eval()

    batch_size, seq_len = 4, 128
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = model(input_ids, attention_mask)

    assert output["loss"] is None
    assert output["logits"].shape == (batch_size, 10)


def test_ner_tagger_train_mode(encoder):
    """测试NER标注头训练模式"""
    from src.models.ner_tagger import NERTagger

    model = NERTagger(encoder, num_tags=7)

    batch_size, seq_len = 4, 64
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    labels = torch.randint(0, 7, (batch_size, seq_len))

    output = model(input_ids, attention_mask, labels=labels)
    assert "loss" in output
    assert output["loss"].requires_grad


def test_ner_tagger_inference_mode(encoder):
    """测试NER标注头推理模式"""
    from src.models.ner_tagger import NERTagger

    model = NERTagger(encoder, num_tags=7)
    model.eval()

    batch_size, seq_len = 4, 64
    input_ids = torch.randint(100, 20000, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)

    with torch.no_grad():
        output = model(input_ids, attention_mask, labels=None)

    assert "predictions" in output
    assert len(output["predictions"]) == batch_size
    # 每个预测序列长度应 ≤ seq_len
    for pred_seq in output["predictions"]:
        assert len(pred_seq) <= seq_len
