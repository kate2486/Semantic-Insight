"""
共享BERT编码器
加载预训练bert-base-chinese，冻结指定层
"""
import torch
import torch.nn as nn
from transformers import BertModel


class SharedEncoder(nn.Module):
    """共享BERT编码器，提供给分类和NER任务共用"""

    def __init__(
        self,
        model_name: str = "bert-base-chinese",
        freeze_embeddings: bool = True,
        freeze_layers: int = 6,
    ):
        super().__init__()
        self.bert = BertModel.from_pretrained(model_name, local_files_only=True)
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
