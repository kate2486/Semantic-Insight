"""
多任务组合模型
一个共享编码器 + 分类头 + NER头，统一推理接口
"""
import torch
from typing import List, Dict
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
        self.label_names = label_names or [
            "体育", "财经", "科技", "教育", "时尚", "军事", "游戏", "房产", "娱乐", "时政"
        ]
        self.id2tag = id2tag or [
            "O", "B-PER", "I-PER", "B-LOC", "I-LOC", "B-ORG", "I-ORG"
        ]
        self.max_length = max_length

        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese", local_files_only=True
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder.to(self.device)
        if self.classifier_head:
            self.classifier_head.to(self.device)
        if self.ner_head:
            self.ner_head.to(self.device)

        # 切换到评估模式
        self.encoder.eval()
        if self.classifier_head:
            self.classifier_head.eval()
        if self.ner_head:
            self.ner_head.eval()

    def predict(self, text: str) -> Dict:
        """
        单条文本推理
        返回: {
            "text": 原始文本,
            "classification": {"label": "体育", "confidence": 0.95},
            "entities": [{"word": "湖人", "type": "ORG", "start": 0, "end": 2}],
        }
        """
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

        with torch.no_grad():
            encoded = self.encoder(input_ids, attention_mask)

            # 分类预测
            if self.classifier_head is not None:
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
                emissions = self.ner_head.linear(
                    self.ner_head.dropout(encoded["last_hidden_state"])
                )
                mask = attention_mask.bool()
                predictions = self.ner_head.crf.decode(emissions, mask=mask)[0]

                # 解析实体
                valid_len = attention_mask[0].sum().item()
                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                entities = self._parse_bio(tokens[1:valid_len-1], predictions[1:valid_len-1])
                result["entities"] = entities

        return result

    def _parse_bio(self, tokens: List[str], tag_ids: List[int]) -> List[Dict]:
        """从BIO标签序列中解析实体"""
        entities = []
        current_tokens = []
        current_type = None

        for i, (token, tag_id) in enumerate(zip(tokens, tag_ids)):
            tag = self.id2tag[tag_id] if tag_id < len(self.id2tag) else "O"

            if tag.startswith("B-"):
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
            freeze_embeddings=False,
            freeze_layers=0,
        )
        classifier_head = TextClassifier(
            encoder, num_classes=config["classifier"]["num_classes"]
        )
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
