"""
快速训练脚本 — 使用真实数据子集加快训练速度
每类取 6000 条，总计 60000 条，5 epochs，约 2 小时
"""
import os
import sys
import json
import random
import yaml
import torch
import numpy as np

sys.stdout.reconfigure(encoding='utf-8')

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.data.dataset import ClassificationDataset, NERDataset
from src.train.trainer import Trainer
from torch.utils.data import DataLoader
from transformers import BertTokenizer


def create_subset_data(input_path, output_path, max_per_class=6000):
    """创建平衡子集"""
    with open(input_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    random.seed(42)
    by_class = {}
    for item in data:
        label = item["label"]
        by_class.setdefault(label, []).append(item)

    subset = []
    for label, items in by_class.items():
        if len(items) > max_per_class:
            items = random.sample(items, max_per_class)
        subset.extend(items)

    random.shuffle(subset)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(subset, f, ensure_ascii=False, indent=2)

    # 统计
    from collections import Counter
    counts = Counter(s["label"] for s in subset)
    label_names = ["体育", "财经", "科技", "教育", "时尚", "社会", "游戏", "房产", "娱乐", "时政"]
    print(f"子集大小: {len(subset)}")
    for idx, name in enumerate(label_names):
        print(f"  {name}: {counts.get(idx, 0)}")
    return subset


def main():
    print("=" * 60)
    print("快速训练 — 真实数据子集")
    print("=" * 60)

    # 加载配置
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 创建子集数据
    processed_dir = "data/processed"
    print("\n[1/3] 创建训练子集 (每类 6000 条)...")
    create_subset_data(
        os.path.join(processed_dir, "news_train.json"),
        os.path.join(processed_dir, "news_train_quick.json"),
        max_per_class=6000,
    )

    # 初始化
    tokenizer = BertTokenizer.from_pretrained(
        config["encoder"]["model_name"], local_files_only=True
    )
    max_length = config["encoder"]["max_length"]

    # 创建子集 DataLoader
    train_ds = ClassificationDataset(
        os.path.join(processed_dir, "news_train_quick.json"),
        tokenizer, max_length,
    )
    val_ds = ClassificationDataset(
        os.path.join(processed_dir, "news_val.json"),
        tokenizer, max_length,
    )
    test_ds = ClassificationDataset(
        os.path.join(processed_dir, "news_test.json"),
        tokenizer, max_length,
    )

    batch_size = config["classifier"]["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    print(f"训练 batches: {len(train_dl)}, 验证 batches: {len(val_dl)}")

    # 创建模型
    print("\n[2/3] 创建模型...")
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
    print(f"可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练
    print("\n[3/3] 开始训练 (5 epochs)...")
    trainer = Trainer(
        model=model,
        train_dataloader=train_dl,
        val_dataloader=val_dl,
        config={**config["classifier"], "epochs": 5},
        checkpoint_dir="checkpoints",
    )

    history = trainer.train(task_type="classification")

    # 测试集评估
    print("\n" + "=" * 50)
    print("测试集最终评估")
    print("=" * 50)

    trainer.load_checkpoint("best_classification.pt")

    from sklearn.metrics import classification_report, accuracy_score

    trainer.model.eval()
    all_preds, all_labels = [], []
    device = trainer.device
    with torch.no_grad():
        for batch in test_dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = trainer.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    acc = accuracy_score(all_labels, all_preds)
    print(f"\n总体准确率: {acc:.4f} ({acc*100:.2f}%)")

    label_names = config["classifier"]["label_names"]
    print("\n" + classification_report(all_labels, all_preds, target_names=label_names))

    # 保存历史记录
    with open("checkpoints/training_history.json", "w", encoding="utf-8") as f:
        json.dump({
            "train_loss": [float(x) for x in history["train_loss"]],
            "val_loss": [float(x) for x in history["val_loss"]],
            "val_metric": [float(x) for x in history["val_metric"]],
            "test_accuracy": float(acc),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n训练历史已保存至 checkpoints/training_history.json")
    print("✅ 快速训练完成！")


if __name__ == "__main__":
    main()
