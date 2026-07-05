"""
NER序列标注模型
BERT编码器 → 每Token Linear → CRF (可选) → BIO标签序列
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from src.models.encoder import SharedEncoder


class NERTagger(nn.Module):
    """BERT + Linear + [CRF | CrossEntropy] 序列标注"""

    def __init__(
        self,
        encoder: SharedEncoder,
        num_tags: int = 7,
        dropout: float = 0.1,
        use_crf: bool = True,
        class_weights: list = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.num_tags = num_tags
        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(encoder.hidden_size, num_tags)

        # 将 bias 初始化为 0，避免 O 标签主导 emission 分数
        nn.init.zeros_(self.linear.bias)

        self.use_crf = use_crf
        if use_crf:
            self.crf = CRF(num_tags, batch_first=True)
        else:
            # 使用带类别权重的 CrossEntropyLoss 处理标签不平衡
            if class_weights is not None:
                weights = torch.tensor(class_weights, dtype=torch.float)
            else:
                weights = None
            self.ce_loss = nn.CrossEntropyLoss(weight=weights, ignore_index=-100)

    def forward(self, input_ids, attention_mask, labels=None, token_type_ids=None):
        # 编码
        encoded = self.encoder(input_ids, attention_mask, token_type_ids)
        hidden = encoded["last_hidden_state"]  # (batch, seq_len, 768)

        # 每Token发射分数
        emissions = self.linear(self.dropout(hidden))  # (batch, seq_len, num_tags)

        mask = attention_mask.bool()

        loss = None

        if labels is not None:
            if self.use_crf:
                # CRF negative log-likelihood
                loss = -self.crf(emissions, labels, mask=mask, reduction="mean")
            else:
                # CrossEntropy with mask (ignore padding=0 positions)
                # 将 padding 位置的 label 设为 -100 让 CE 忽略
                labels_for_ce = labels.clone()
                labels_for_ce[~mask] = -100
                loss = self.ce_loss(
                    emissions.view(-1, self.num_tags),
                    labels_for_ce.view(-1),
                )

        # Decode
        if self.use_crf:
            predictions = self.crf.decode(emissions, mask=mask)
        else:
            pred_ids = torch.argmax(emissions, dim=-1)  # (batch, seq_len)
            predictions = []
            for i in range(pred_ids.size(0)):
                valid_len = mask[i].sum().item()
                predictions.append(pred_ids[i, :valid_len].tolist())

        return {"loss": loss, "predictions": predictions, "emissions": emissions}
