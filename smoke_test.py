"""
冒烟测试 — 用 1 epoch 快速验证训练管线完整性
验证: 数据加载 → 前后向传播 → checkpoint保存 → 评估指标
"""
import os
import sys
import yaml
import time
import torch
import numpy as np
import random

sys.path.insert(0, os.path.dirname(__file__))

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.models.ner_tagger import NERTagger
from src.data.dataset import create_dataloaders
from src.train.trainer import Trainer


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config():
    with open("configs/config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_classification_pipeline(config):
    """冒烟测试: 分类训练管线"""
    print("\n" + "=" * 60)
    print("  冒烟测试: 文本分类管线")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(base_dir, "data", "processed")

    # 1. 数据加载
    print("\n[1/5] 加载数据...")
    t0 = time.time()
    dataloaders, tokenizer = create_dataloaders(processed_dir, config)
    print(f"  [OK] Data loaded ({time.time() - t0:.1f}s)")
    print(f"    Classification train: {len(dataloaders['cls_train'].dataset)} samples")

    # Check one batch
    batch = next(iter(dataloaders["cls_train"]))
    assert "input_ids" in batch and "attention_mask" in batch and "label" in batch
    print(f"    Batch shape: input_ids={batch['input_ids'].shape}, label={batch['label'].shape}")

    # 2. Model creation
    print("\n[2/5] Creating model...")
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  [OK] Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

    # 3. Training (smoke: 1 epoch)
    print("\n[3/5] Training 1 epoch...")
    cls_config = config["classifier"].copy()
    cls_config["epochs"] = 1
    cls_config["eval_steps"] = 100

    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["cls_train"],
        val_dataloader=dataloaders["cls_val"],
        config=cls_config,
        checkpoint_dir=os.path.join(base_dir, "checkpoints"),
    )

    t0 = time.time()
    history = trainer.train(task_type="classification")
    elapsed = time.time() - t0
    print(f"  [OK] Training done ({elapsed:.1f}s, {elapsed/60:.1f}min)")
    print(f"    Train loss: {history['train_loss'][-1]:.4f}")
    if history["val_metric"]:
        print(f"    Val accuracy: {history['val_metric'][-1]:.4f}")

    # 4. Checkpoint verification
    print("\n[4/5] Verifying checkpoint...")
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_classification.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt_size = os.path.getsize(ckpt_path) / 1e6
    print(f"  [OK] Checkpoint saved: {ckpt_path} ({ckpt_size:.1f} MB)")

    # 5. Test set evaluation
    print("\n[5/5] Test set evaluation...")
    trainer.load_checkpoint("best_classification.pt")
    trainer.model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for batch in dataloaders["cls_test"]:
            batch = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            preds = torch.argmax(outputs["logits"], dim=-1)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch["label"].cpu().tolist())

    from sklearn.metrics import accuracy_score
    acc = accuracy_score(all_labels, all_preds)
    print(f"  [OK] Test Accuracy: {acc:.4f} ({acc*100:.1f}%)")

    print("\n" + "-" * 40)
    print("  Classification pipeline smoke test: PASSED")
    print("-" * 40)
    return True


def test_ner_pipeline(config):
    """冒烟测试: NER训练管线"""
    print("\n" + "=" * 60)
    print("  冒烟测试: NER 管线")
    print("=" * 60)

    base_dir = os.path.dirname(__file__)
    processed_dir = os.path.join(base_dir, "data", "processed")

    # 1. Data loading
    print("\n[1/5] Loading data...")
    t0 = time.time()
    dataloaders, tokenizer = create_dataloaders(processed_dir, config)
    print(f"  [OK] Data loaded ({time.time() - t0:.1f}s)")
    print(f"    NER train: {len(dataloaders['ner_train'].dataset)} samples")

    batch = next(iter(dataloaders["ner_train"]))
    assert "input_ids" in batch and "attention_mask" in batch and "labels" in batch
    print(f"    Batch shape: input_ids={batch['input_ids'].shape}, labels={batch['labels'].shape}")

    # 2. Model creation
    print("\n[2/5] Creating model...")
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
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  [OK] Trainable params: {trainable:,}")

    # 3. Training (smoke: 1 epoch)
    print("\n[3/5] Training 1 epoch...")
    ner_config = config["ner"].copy()
    ner_config["epochs"] = 1
    ner_config["eval_steps"] = 50

    trainer = Trainer(
        model=model,
        train_dataloader=dataloaders["ner_train"],
        val_dataloader=dataloaders["ner_val"],
        config=ner_config,
        checkpoint_dir=os.path.join(base_dir, "checkpoints"),
    )

    t0 = time.time()
    history = trainer.train(task_type="ner")
    elapsed = time.time() - t0
    print(f"  [OK] Training done ({elapsed:.1f}s, {elapsed/60:.1f}min)")
    print(f"    Train loss: {history['train_loss'][-1]:.4f}")
    if history["val_metric"]:
        print(f"    Val F1: {history['val_metric'][-1]:.4f}")

    # 4. Checkpoint verification
    print("\n[4/5] Verifying checkpoint...")
    ckpt_path = os.path.join(base_dir, "checkpoints", "best_ner.pt")
    assert os.path.exists(ckpt_path), f"Checkpoint not found: {ckpt_path}"
    ckpt_size = os.path.getsize(ckpt_path) / 1e6
    print(f"  [OK] Checkpoint saved: {ckpt_path} ({ckpt_size:.1f} MB)")

    # 5. Inference verification
    print("\n[5/5] Inference verification...")
    trainer.load_checkpoint("best_ner.pt")
    id2tag = config["ner"]["id2tag"]
    trainer.model.eval()
    all_true_tags, all_pred_tags = [], []

    with torch.no_grad():
        for batch in dataloaders["ner_test"]:
            batch_gpu = {k: v.to(trainer.device) for k, v in batch.items()}
            outputs = trainer.model(
                input_ids=batch_gpu["input_ids"],
                attention_mask=batch_gpu["attention_mask"],
                labels=None,
            )
            labels = batch["labels"]
            for pred_ids, label_ids, mask in zip(
                outputs["predictions"], labels, batch["attention_mask"]
            ):
                valid_len = mask.sum().item()
                true_seq, pred_seq = [], []
                for i in range(valid_len):
                    lid = label_ids[i].item()
                    pid = pred_ids[i] if i < len(pred_ids) else 0
                    true_seq.append(id2tag[lid])
                    pred_seq.append(id2tag[pid])
                all_true_tags.append(true_seq)
                all_pred_tags.append(pred_seq)

    # Verify output format
    assert len(all_true_tags) > 0 and len(all_pred_tags) > 0
    print(f"  [OK] Inferenced {len(all_true_tags)} sentences")
    print(f"    Sample pred: {all_pred_tags[0][:10]}...")
    print(f"    Sample true: {all_true_tags[0][:10]}...")

    print("\n" + "-" * 40)
    print("  NER pipeline smoke test: PASSED")
    print("-" * 40)
    return True


def main():
    print("=" * 60)
    print("  Semantic-Insight Smoke Test")
    print(f"  Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 60)

    config = load_config()
    set_seed(config["training"]["seed"])

    results = {}

    # Classification pipeline
    try:
        results["classification"] = test_classification_pipeline(config)
    except Exception as e:
        print(f"\n  [FAIL] Classification pipeline: {e}")
        import traceback
        traceback.print_exc()
        results["classification"] = False

    # NER pipeline
    try:
        results["ner"] = test_ner_pipeline(config)
    except Exception as e:
        print(f"\n  [FAIL] NER pipeline: {e}")
        import traceback
        traceback.print_exc()
        results["ner"] = False

    # Summary
    print("\n" + "=" * 60)
    print("  Smoke Test Results")
    print("=" * 60)
    for name, passed in results.items():
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")

    all_passed = all(results.values())
    if all_passed:
        print("\n  All smoke tests passed! Ready for full training.")
    else:
        print("\n  Some tests failed. Fix before training.")

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
