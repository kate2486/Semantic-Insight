"""
训练完成后运行此脚本进行完整的测试集评估
用法: python final_evaluation.py
"""
import os
import sys
import json
import time
import yaml
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')


def load_model_and_config():
    """加载最佳模型和配置"""
    config_path = "configs/config.yaml"
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    from src.models.encoder import SharedEncoder
    from src.models.classifier import TextClassifier

    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )

    model = TextClassifier(
        encoder=encoder,
        num_classes=config["classifier"]["num_classes"],
        dropout=config["classifier"]["dropout"],
    )

    checkpoint = "checkpoints/best_classification.pt"
    model.load_state_dict(torch.load(checkpoint, map_location="cpu"))
    model.eval()

    return model, config


def evaluate(model, config):
    """在测试集上评估"""
    from src.data.dataset import create_dataloaders
    from sklearn.metrics import (
        classification_report, confusion_matrix, accuracy_score
    )

    dataloaders, tokenizer = create_dataloaders(
        "data/processed", config
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloaders["cls_test"]:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    accuracy = accuracy_score(all_labels, all_preds)
    label_names = config["classifier"]["label_names"]
    report = classification_report(
        all_labels, all_preds, target_names=label_names, output_dict=True
    )

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)

    return accuracy, report, cm, all_preds, all_labels


def test_real_examples(model, config):
    """测试真实案例（包含 NBA 等英文术语）"""
    from src.models.multitask_model import MultiTaskModel
    from src.models.ner_tagger import NERTagger
    from src.models.encoder import SharedEncoder

    # NER 使用独立的 encoder，避免权重覆盖分类 encoder
    ner_encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )
    ner_head = NERTagger(
        ner_encoder,
        num_tags=config["ner"]["num_tags"],
        use_crf=False,
    )
    ner_path = "checkpoints/best_ner.pt"
    if os.path.exists(ner_path):
        ner_head.load_state_dict(
            torch.load(ner_path, map_location="cpu"), strict=False
        )

    multi_model = MultiTaskModel(
        encoder=model.encoder,
        classifier_head=model,
        ner_head=ner_head,
        label_names=config["classifier"]["label_names"],
        id2tag=config["ner"]["id2tag"],
        max_length=config["encoder"]["max_length"],
    )

    test_cases = [
        "今天NBA总决赛，洛杉矶湖人队大胜波士顿凯尔特人",
        "NBA季后赛正在激烈进行中",
        "湖人队在NBA总决赛中表现优异",
        "科比是NBA历史上最伟大的球员之一",
        "腾讯体育获得了NBA中国区的转播权",
        "CBA联赛新赛季即将开幕",
        "中国男足在世界杯预选赛中取得关键胜利",
        "央行宣布降息50个基点，A股市场大幅反弹",
        "华为发布新一代AI芯片，性能提升3倍",
        "北京大学和清华大学联合举办人工智能论坛",
        "巴黎时装周上，LV发布了2025春夏系列",
        "《黑神话：悟空》全球销量突破2000万套",
        "北京楼市新政：首套房首付比例降至20%",
    ]

    results = []
    for text in test_cases:
        r = multi_model.predict(text)
        results.append({
            "text": text,
            "classification": r.get("classification", {}),
            "entities": [(e["word"], e["type"]) for e in r.get("entities", [])],
        })

    return results


def generate_report(accuracy, report, cm, label_names, examples):
    """生成完整评估报告"""
    lines = []
    lines.append("# Semantic Insight — 最终评估报告")
    lines.append("")
    lines.append(f"> 评估时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"> 评估数据集: THUCNews 测试集 ({sum(cm.flatten())} 条)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 测试集总体准确率")
    lines.append("")
    lines.append(f"### {accuracy:.4f} ({accuracy * 100:.2f}%)")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 各类别详细指标")
    lines.append("")
    lines.append("| 类别 | Precision | Recall | F1-Score | Support |")
    lines.append("|------|-----------|--------|----------|---------|")
    for name in label_names:
        metrics = report[name]
        lines.append(
            f"| {name} | {metrics['precision']:.4f} | "
            f"{metrics['recall']:.4f} | {metrics['f1-score']:.4f} | "
            f"{int(metrics['support'])} |"
        )
    lines.append("")
    lines.append(f"| **Macro Avg** | {report['macro avg']['precision']:.4f} | "
                 f"{report['macro avg']['recall']:.4f} | {report['macro avg']['f1-score']:.4f} | "
                 f"{int(report['macro avg']['support'])} |")
    lines.append(f"| **Weighted Avg** | {report['weighted avg']['precision']:.4f} | "
                 f"{report['weighted avg']['recall']:.4f} | {report['weighted avg']['f1-score']:.4f} | "
                 f"{int(report['weighted avg']['support'])} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 混淆矩阵")
    lines.append("")
    lines.append("```")
    header = "      " + " ".join(f"{n[:4]:>4}" for n in label_names)
    lines.append(header)
    for i, name in enumerate(label_names):
        row = f"{name:>4} " + " ".join(f"{cm[i][j]:4d}" for j in range(len(label_names)))
        lines.append(row)
    lines.append("```")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("## 真实案例推理测试")
    lines.append("")
    lines.append("以下测试包含 NBA、CBA 等英文术语及真实新闻文本：")
    lines.append("")
    lines.append("| 输入文本 | 分类结果 | 置信度 | NER 实体 |")
    lines.append("|----------|----------|--------|----------|")
    for ex in examples:
        cls = ex["classification"]
        label = cls.get("label", "N/A")
        conf = cls.get("confidence", 0)
        entities = ", ".join([f"{w}({t})" for w, t in ex["entities"]]) or "无"
        lines.append(
            f"| {ex['text'][:40]}... | **{label}** | {conf:.2%} | {entities} |"
        )
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append("*报告由 final_evaluation.py 自动生成*")

    return "\n".join(lines)


def main():
    print("=" * 60)
    print("最终评估")
    print("=" * 60)

    print("\n加载模型...")
    model, config = load_model_and_config()
    print("模型已加载")

    print("\n在测试集上评估...")
    accuracy, report, cm, preds, labels = evaluate(model, config)
    print(f"测试集准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)")

    label_names = config["classifier"]["label_names"]
    print("\n各类别 F1-Score:")
    for name in label_names:
        f1 = report[name]["f1-score"]
        support = int(report[name]["support"])
        print(f"  {name}: {f1:.4f} (support: {support})")

    print("\n测试真实案例...")
    examples = test_real_examples(model, config)
    for ex in examples:
        cls = ex["classification"]
        print(f"  {ex['text'][:50]}...")
        print(f"    → {cls.get('label', 'N/A')} ({cls.get('confidence', 0):.2%})")

    # 保存评估报告
    print("\n生成评估报告...")
    report_text = generate_report(accuracy, report, cm, label_names, examples)
    report_path = r"C:\Users\ASUS\Desktop\神经网络实训\最终评估报告.md"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"报告已保存至: {report_path}")

    # 也更新原来的训练验证报告
    training_report_path = r"C:\Users\ASUS\Desktop\神经网络实训\训练验证指标报告.md"
    if os.path.exists(training_report_path):
        with open(training_report_path, "a", encoding="utf-8") as f:
            f.write(f"\n\n---\n\n## 🔄 更新：测试集最终评估 ({time.strftime('%Y-%m-%d %H:%M:%S')})\n\n")
            f.write(f"### 总体准确率: {accuracy:.4f} ({accuracy * 100:.2f}%)\n\n")
            f.write(f"### 各类别 F1-Score\n\n")
            for name in label_names:
                f.write(f"- **{name}**: {report[name]['f1-score']:.4f}\n")
        print(f"原始训练报告已更新: {training_report_path}")

    print("\n✅ 评估完成！")


if __name__ == "__main__":
    main()
