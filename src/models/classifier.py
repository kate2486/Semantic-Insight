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
