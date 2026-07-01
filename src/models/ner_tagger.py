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

        # CRF mask: True 表示有效位置
        mask = attention_mask.bool()

        loss = None
        predictions = None

        if labels is not None:
            # 训练模式: 计算 CRF 负对数似然
            loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
        else:
            # 推理模式: CRF 维特比解码
            predictions = self.crf.decode(emissions, mask=mask)

        return {"loss": loss, "predictions": predictions, "emissions": emissions}
