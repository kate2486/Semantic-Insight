"""
训练NER序列标注模型
"""
import os
import sys
import yaml
import torch
import numpy as np
import random

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.encoder import SharedEncoder
from src.models.ner_tagger import NERTagger
from src.data.dataset import create_dataloaders
from src.train.trainer import Trainer
from src.evaluate.metrics import compute_ner_metrics


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
    model = NERTagger(
        encoder=encoder,
        num_tags=config["ner"]["num_tags"],
        dropout=config["ner"]["dropout"],
    )

    print(f"NER模型可训练参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # 训练
    print("\n开始训练...")
    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["ner_train"],
        val_dataloader=dataloaders["ner_val"],
        config=config["ner"],
        checkpoint_dir=os.path.join(base_dir, "checkpoints"),
    )

    history = trainer.train(task_type="ner")

    # 测试集最终评估
    print("\n" + "=" * 50)
    print("NER 测试集评估")
    print("=" * 50)

    trainer.load_checkpoint("best_ner.pt")
    id2tag = config["ner"]["id2tag"]

    trainer.model.eval()
    all_true_tags = []
    all_pred_tags = []

    with torch.no_grad():
        for batch in dataloaders["ner_test"]:
            batch_gpu = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                labels=None,  # 推理模式
            )

            labels = batch["labels"]
            for pred_ids, label_ids, mask in zip(
                outputs["predictions"], labels, batch["attention_mask"]
            ):
                valid_len = mask.sum().item()
                true_seq = []
                pred_seq = []
                for i in range(valid_len):
                    lid = label_ids[i].item()
                    pid = pred_ids[i] if i < len(pred_ids) else 0
                    true_seq.append(id2tag[lid])
                    pred_seq.append(id2tag[pid])
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    metrics = compute_ner_metrics(all_true_tags, all_pred_tags, id2tag)

    print(f"\nMacro F1: {metrics['macro_f1']:.4f}")
    print("\nPer-entity 指标:")
    print(f"{'实体类型':<10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Support':>10}")
    print("-" * 55)
    for entity, scores in metrics["per_entity"].items():
        if scores["support"] > 0:
            print(
                f"{entity:<10} {scores['precision']:>10.3f} "
                f"{scores['recall']:>10.3f} {scores['f1']:>10.3f} "
                f"{scores['support']:>10}"
            )


if __name__ == "__main__":
    main()
