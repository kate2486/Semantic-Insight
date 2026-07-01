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
