"""
通用训练器
支持分类和NER两种任务，统一训练循环
"""
import os
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm
import numpy as np


class Trainer:
    """通用训练器"""

    def __init__(
        self,
        model: nn.Module,
        train_dataloader,
        val_dataloader,
        config: dict,
        device: str = None,
        checkpoint_dir: str = "checkpoints",
    ):
        self.model = model
        self.train_dl = train_dataloader
        self.val_dl = val_dataloader
        self.config = config

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        print(f"使用设备: {self.device}")
        self.model.to(self.device)

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        # 优化器
        self.optimizer = AdamW(
            model.parameters(),
            lr=config.get("learning_rate", 2e-5),
        )

        # 学习率调度器
        total_steps = len(train_dataloader) * config.get("epochs", 5)
        warmup_steps = int(total_steps * config.get("warmup_ratio", 0.1))
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )

        self.gradient_accumulation_steps = config.get("gradient_accumulation_steps", 1)
        self.max_grad_norm = config.get("max_grad_norm", 1.0)
        self.epochs = config.get("epochs", 5)
        self.eval_steps = config.get("eval_steps", 200)

        # 训练记录
        self.history = {"train_loss": [], "val_loss": [], "val_metric": []}

    def train(self, task_type: str = "classification"):
        """
        训练循环
        task_type: "classification" 或 "ner"
        """
        global_step = 0
        best_metric = 0.0

        for epoch in range(self.epochs):
            self.model.train()
            epoch_loss = 0.0

            pbar = tqdm(self.train_dl, desc=f"Epoch {epoch + 1}/{self.epochs}")
            for step, batch in enumerate(pbar):
                # 将batch移到GPU
                batch = {k: v.to(self.device) for k, v in batch.items()}

                # 分类任务用 "label"，NER 任务用 "labels"
                if task_type == "classification":
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids", None),
                        label=batch["label"],
                    )
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs

                # 梯度累积
                loss = loss / self.gradient_accumulation_steps
                loss.backward()

                epoch_loss += loss.item()

                if (step + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.max_grad_norm
                    )
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()
                    global_step += 1

                pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # 定期评估
                if global_step > 0 and global_step % self.eval_steps == 0:
                    val_loss, val_metric = self.evaluate(task_type)
                    self.history["val_loss"].append(val_loss)
                    self.history["val_metric"].append(val_metric)

                    metric_name = "acc" if task_type == "classification" else "f1"
                    if val_metric > best_metric:
                        best_metric = val_metric
                        self.save_checkpoint(f"best_{task_type}.pt")
                        print(f"  [新最佳模型] {metric_name}: {val_metric:.4f}")

                    self.model.train()

            avg_loss = epoch_loss / len(self.train_dl)
            self.history["train_loss"].append(avg_loss)
            print(f"Epoch {epoch + 1} 完成, 平均loss: {avg_loss:.4f}")

        return self.history

    def evaluate(self, task_type: str = "classification"):
        """评估模型"""
        self.model.eval()
        total_loss = 0.0
        all_preds, all_labels = [], []

        with torch.no_grad():
            for batch in self.val_dl:
                batch = {k: v.to(self.device) for k, v in batch.items()}

                if task_type == "classification":
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        token_type_ids=batch.get("token_type_ids", None),
                        label=batch["label"],
                    )
                else:
                    outputs = self.model(
                        input_ids=batch["input_ids"],
                        attention_mask=batch["attention_mask"],
                        labels=batch["labels"],
                    )

                loss = outputs["loss"] if isinstance(outputs, dict) else outputs
                total_loss += loss.item()

                if task_type == "classification":
                    logits = outputs["logits"]
                    preds = torch.argmax(logits, dim=-1)
                    all_preds.extend(preds.cpu().tolist())
                    all_labels.extend(batch["label"].cpu().tolist())
                else:
                    # NER: 只收集非-100位置的预测标签
                    predictions = outputs["predictions"]
                    labels = batch["labels"]
                    for pred_seq, label_seq, mask in zip(
                        predictions, labels, batch["attention_mask"]
                    ):
                        valid_len = mask.sum().item()
                        for i in range(valid_len):
                            lid = label_seq[i].item()
                            if lid != -100:
                                pid = pred_seq[i] if i < len(pred_seq) else 0
                                all_preds.append(pid)
                                all_labels.append(lid)

        avg_loss = total_loss / len(self.val_dl)
        metric = self._compute_metric(all_preds, all_labels, task_type)
        return avg_loss, metric

    def _compute_metric(self, preds, labels, task_type):
        """计算评估指标"""
        from sklearn.metrics import accuracy_score, f1_score

        if task_type == "classification":
            return accuracy_score(labels, preds)
        else:
            # Macro F1, 忽略 O 标签（标签0）占主导的影响
            labels_filtered = [(p, l) for p, l in zip(preds, labels) if l != 0 or p != 0]
            if not labels_filtered:
                return 0.0
            p_filtered, l_filtered = zip(*labels_filtered)
            return f1_score(l_filtered, p_filtered, average="macro", zero_division=0)

    def save_checkpoint(self, filename: str):
        """保存模型权重"""
        path = os.path.join(self.checkpoint_dir, filename)
        torch.save(self.model.state_dict(), path)

    def load_checkpoint(self, filename: str):
        """加载模型权重"""
        path = os.path.join(self.checkpoint_dir, filename)
        self.model.load_state_dict(torch.load(path, map_location=self.device))
