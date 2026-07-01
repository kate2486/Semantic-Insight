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
                current_entity = tag[2:]  # 去掉 B- 前缀
                start_idx = i
            elif tag.startswith("I-"):
                entity_type = tag[2:]
                if current_entity != entity_type:
                    if current_entity:
                        entities.append((current_entity, start_idx, i))
                    current_entity = None
            else:  # O 标签
                if current_entity:
                    entities.append((current_entity, start_idx, i))
                    current_entity = None

        if current_entity:
            entities.append((current_entity, start_idx, len(tag_seq)))

        return entities

    # 收集所有实体
    true_entities_all = defaultdict(list)
    pred_entities_all = defaultdict(list)

    for true_tags, pred_tags in zip(y_true, y_pred):
        for ent_type, start, end in extract_entities(true_tags):
            true_entities_all[ent_type].append((ent_type, start, end))
        for ent_type, start, end in extract_entities(pred_tags):
            pred_entities_all[ent_type].append((ent_type, start, end))

    # 计算每种实体的 P/R/F1
    entity_types = set(list(true_entities_all.keys()) + list(pred_entities_all.keys()))
    per_entity = {}

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

    # Macro F1
    f1s = [v["f1"] for v in per_entity.values() if v["support"] > 0]
    macro_f1 = sum(f1s) / len(f1s) if f1s else 0.0

    return {
        "per_entity": per_entity,
        "macro_f1": macro_f1,
    }
