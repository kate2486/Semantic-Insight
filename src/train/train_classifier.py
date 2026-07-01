"""
训练文本分类模型
"""
import os
import sys
import yaml
import torch
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.data.dataset import create_dataloaders
from src.train.trainer import Trainer


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main():
    # 加载配置
    config_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "configs", "config.yaml",
    )
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    set_seed(config["training"]["seed"])

    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    processed_dir = os.path.join(base_dir, "data", "processed")

    # 创建 DataLoader
    print("创建 DataLoader...")
    dataloaders, tokenizer = create_dataloaders(processed_dir, config)

    # 创建模型
    print("\n创建模型...")
    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=config["encoder"]["freeze_embeddings"],
        freeze_layers=config["encoder"]["freeze_layers"],
    )
    model = TextClassifier(
        encoder=encoder,
        num_classes=config["classifier"]["num_classes"],
        dropout=config["classifier"]["dropout"],
    )

    print(f"分类模型可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练
    print("\n开始训练...")
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["cls_train"],
        val_dataloader=dataloaders["cls_val"],
        config=config["classifier"],
        checkpoint_dir=os.path.join(base_dir, "checkpoints"),
    )

    history = trainer.train(task_type="classification")

    # 测试集最终评估
    print("\n" + "=" * 50)
    print("测试集评估")
    print("=" * 50)

    trainer.load_checkpoint("best_classification.pt")
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")  # 非交互模式
    import matplotlib.pyplot as plt
    import seaborn as sns

    trainer.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloaders["cls_test"]:
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch.get("token_type_ids", None),
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    label_names = config["classifier"]["label_names"]
    print("\n" + classification_report(all_labels, all_preds, target_names=label_names))

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=label_names, yticklabels=label_names)
    plt.title("分类混淆矩阵")
    plt.xlabel("预测")
    plt.ylabel("真实")
    plt.tight_layout()
    cm_path = os.path.join(base_dir, "checkpoints", "confusion_matrix.png")
    plt.savefig(cm_path)
    print(f"\n混淆矩阵已保存至 {cm_path}")


if __name__ == "__main__":
    main()
