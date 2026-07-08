"""
完整模式训练 → 生成图2（分类曲线）和图3（NER曲线）
记录每个 batch 的 train loss，定期评估 val loss/metric
"""
import os
import sys
import json
import time
import yaml
import torch
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
HIST_DIR = os.path.join(BASE_DIR, "checkpoints")
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(HIST_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, "configs", "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

LABEL_NAMES = CONFIG["classifier"]["label_names"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

from transformers import BertTokenizer
from torch.utils.data import DataLoader


# ============================================================
# 图3: NER 曲线（先跑，快）
# ============================================================
def train_and_plot_ner():
    print("\n" + "=" * 60)
    print("[图3] NER 完整训练 — 20 epochs, ~5000 steps")
    print("=" * 60)

    from src.models.encoder import SharedEncoder
    from src.models.ner_tagger import NERTagger
    from src.data.dataset import NERDataset

    tokenizer = BertTokenizer.from_pretrained(CONFIG["encoder"]["model_name"], local_files_only=True)
    max_length = CONFIG["encoder"]["max_length"]
    tag2id = CONFIG["ner"]["tag2id"]

    train_ds = NERDataset(os.path.join(BASE_DIR, "data", "processed", "ner_train.json"),
                          tokenizer, tag2id, max_length)
    val_ds = NERDataset(os.path.join(BASE_DIR, "data", "processed", "ner_val.json"),
                        tokenizer, tag2id, max_length)

    batch_size = CONFIG["ner"]["batch_size"]
    epochs = CONFIG["ner"]["epochs"]
    eval_steps = 20

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    steps_per_epoch = len(train_dl)  # 250
    total_steps = steps_per_epoch * epochs  # 5000
    print(f"  数据: {len(train_ds)} 条训练 / {len(val_ds)} 条验证")
    print(f"  步数: {steps_per_epoch} batches/epoch × {epochs} epochs = {total_steps} steps")

    # 类别权重
    with open(os.path.join(BASE_DIR, "data", "processed", "ner_train.json"), "r", encoding="utf-8") as f:
        raw = json.load(f)
    tag_counts = Counter()
    for s in raw:
        for t in s["tags"]:
            tag_counts[t] += 1
    total_tags = sum(tag_counts.values())
    class_weights = []
    for tag_name, tag_id in sorted(tag2id.items(), key=lambda x: x[1]):
        class_weights.append(total_tags / (len(tag2id) * tag_counts.get(tag_name, 1)))

    encoder = SharedEncoder(model_name=CONFIG["encoder"]["model_name"],
                            freeze_embeddings=False, freeze_layers=0)
    model = NERTagger(encoder=encoder, num_tags=CONFIG["ner"]["num_tags"],
                      dropout=CONFIG["ner"]["dropout"], use_crf=False, class_weights=class_weights)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  设备: {device}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=float(CONFIG["ner"]["learning_rate"]))
    grad_accum = CONFIG["training"]["gradient_accumulation_steps"]
    opt_steps = total_steps // grad_accum
    warmup = int(opt_steps * float(CONFIG["ner"].get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, opt_steps)

    # 记录器：每个 batch 记一次 train loss
    train_losses = []   # [float] 长度 = total_steps
    val_losses, val_f1s, eval_at_steps = [], [], []

    global_step = 0
    batch_step = 0      # 实际 batch 计数（x 轴）
    last_eval_step = -1  # 防止重复评估
    best_f1 = 0.0
    t_start = time.time()

    print("\n  训练中...")
    for epoch in range(epochs):
        model.train()
        for batch in train_dl:
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch_gpu["input_ids"],
                            attention_mask=batch_gpu["attention_mask"],
                            labels=batch_gpu["labels"])
            loss = outputs["loss"] / grad_accum
            loss.backward()

            train_losses.append(float(outputs["loss"].item()))
            batch_step += 1

            if (batch_step % grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            if global_step > 0 and global_step % eval_steps == 0 and global_step != last_eval_step:
                last_eval_step = global_step
                v_loss, v_f1 = _eval_ner(model, val_dl, device)
                val_losses.append(v_loss)
                val_f1s.append(v_f1)
                eval_at_steps.append(batch_step)
                if v_f1 > best_f1:
                    best_f1 = v_f1
                elapsed = time.time() - t_start
                pct = batch_step / total_steps * 100
                eta = elapsed / batch_step * (total_steps - batch_step) if batch_step > 0 else 0
                print(f"    Step {batch_step:>5d}/{total_steps} ({pct:>4.1f}%) | "
                      f"T Loss: {train_losses[-1]:.4f} | V Loss: {v_loss:.4f} | "
                      f"F1: {v_f1:.4f} | ETA: {eta/60:.0f}m")
                model.train()

        print(f"  Epoch {epoch+1}/{epochs} 完成")

    t_total = time.time() - t_start
    print(f"  训练耗时: {t_total/60:.1f} 分钟")
    print(f"  最佳 F1: {best_f1:.4f}")

    # 保存历史
    ner_hist = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_f1s": val_f1s,
        "eval_at_steps": eval_at_steps,
        "total_steps": total_steps,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "best_f1": float(best_f1),
        "training_time_min": round(t_total / 60, 1),
    }
    with open(os.path.join(HIST_DIR, "ner_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(ner_hist, f, ensure_ascii=False, indent=2)

    # ---- 绘图 ----
    _plot_ner_curves(train_losses, val_losses, val_f1s, eval_at_steps,
                     total_steps, steps_per_epoch, epochs, best_f1)
    return ner_hist


def _eval_ner(model, val_dl, device):
    model.eval()
    total_loss, all_p, all_l = 0.0, [], []
    with torch.no_grad():
        for batch in val_dl:
            bg = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=bg["input_ids"], attention_mask=bg["attention_mask"], labels=bg["labels"])
            total_loss += float(out["loss"].item())
            for ps, ls, mk in zip(out["predictions"], batch["labels"], bg["attention_mask"]):
                vl = mk.sum().item()
                for i in range(vl):
                    lid = ls[i].item()
                    if lid != -100:
                        all_p.append(ps[i] if i < len(ps) else 0)
                        all_l.append(lid)
    from sklearn.metrics import f1_score
    filt = [(p, l) for p, l in zip(all_p, all_l) if not (p == 0 and l == 0)]
    if not filt:
        return total_loss / len(val_dl), 0.0
    pf, lf = zip(*filt)
    return total_loss / len(val_dl), f1_score(lf, pf, average="macro", zero_division=0)


def _plot_ner_curves(train_losses, val_losses, val_f1s, eval_at_steps,
                     total_steps, steps_per_epoch, epochs, best_f1):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 平滑 train loss
    w = max(1, len(train_losses) // 250)
    if w > 1:
        kernel = np.ones(w) / w
        smooth = np.convolve(train_losses, kernel, mode="valid")
        smooth_x = np.arange(w, len(train_losses) + 1)
        t_line, = ax1.plot(smooth_x, smooth, color="#e74c3c", linewidth=1.0, alpha=0.9, label="训练 Loss")
    else:
        t_line, = ax1.plot(np.arange(1, len(train_losses) + 1), train_losses,
                            color="#e74c3c", linewidth=1.0, alpha=0.9, label="训练 Loss")

    vl_line, = ax1.plot(eval_at_steps, val_losses, color="#3498db",
                         linewidth=2.0, marker="o", markersize=3, label="验证 Loss", zorder=5)

    ax1.set_xlabel("训练步数 (batch steps)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_xlim(0, total_steps)

    ax2 = ax1.twinx()
    f1_line, = ax2.plot(eval_at_steps, val_f1s, color="#27ae60",
                         linewidth=2.5, marker="s", markersize=3, label="验证 Macro F1", zorder=6)
    ax2.set_ylabel("Macro F1", fontsize=12, color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(0, 1.05)

    # 三阶段标注
    p1, p2 = int(total_steps * 0.2), int(total_steps * 0.7)
    for start, end, label, bg in [
        (0, p1, "快速学习期", "#e8f8f5"),
        (p1, p2, "精细调整期", "#fef9e7"),
        (p2, total_steps, "收敛平台期", "#fdedec"),
    ]:
        ax1.axvspan(start, end, alpha=0.12, color=bg, zorder=0)
        ax1.text((start + end) / 2, ax1.get_ylim()[1] * 0.97, label,
                 fontsize=10, ha="center", va="top",
                 bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#ccc", alpha=0.85))

    # Epoch 边界
    for ep in range(1, epochs + 1):
        ax1.axvline(x=ep * steps_per_epoch, color="#bdc3c7", linestyle="--", linewidth=0.7, alpha=0.6)

    ax1.legend([t_line, vl_line, f1_line], ["训练 Loss", "验证 Loss", "验证 Macro F1"],
               loc="upper right", fontsize=10, framealpha=0.9)
    ax1.set_title(f"NER 序列标注训练曲线 — Loss & Macro F1 (最佳 F1={best_f1:.4f})",
                  fontsize=14, fontweight="bold", pad=14)
    ax1.grid(axis="both", alpha=0.2, linestyle="--")
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig3_ner_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存: {path}")


# ============================================================
# 图2: 分类曲线（后跑，慢 — 5 epochs × 4835 batches）
# ============================================================
def train_and_plot_classification(resume=False):
    print("\n" + "=" * 60)
    print(f"[图2] 分类完整训练 — 5 epochs, ~24,175 steps{' (续训)' if resume else ''}")
    print("=" * 60)

    from src.models.encoder import SharedEncoder
    from src.models.classifier import TextClassifier
    from src.data.dataset import ClassificationDataset
    from torch.utils.data import Subset

    tokenizer = BertTokenizer.from_pretrained(CONFIG["encoder"]["model_name"], local_files_only=True)
    max_length = CONFIG["encoder"]["max_length"]

    train_ds = ClassificationDataset(os.path.join(BASE_DIR, "data", "processed", "news_train.json"),
                                     tokenizer, max_length)
    val_ds = ClassificationDataset(os.path.join(BASE_DIR, "data", "processed", "news_val.json"),
                                   tokenizer, max_length)

    batch_size = CONFIG["classifier"]["batch_size"]
    epochs = CONFIG["classifier"]["epochs"]
    eval_steps = 400  # 每 400 个 optimizer step（800 batch）快速评估

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)

    # 完整验证集（epoch 结束时用）
    val_dl_full = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    # 快速验证子集：随机抽取 1500 条用于 step 级别评估
    rng = np.random.RandomState(42)
    val_indices = rng.choice(len(val_ds), size=min(1500, len(val_ds)), replace=False)
    val_subset = Subset(val_ds, val_indices)
    val_dl = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=0)
    print(f"  快速评估: {len(val_subset)} 条 / {len(val_ds)} 条验证数据")

    steps_per_epoch = len(train_dl)  # 4835
    total_steps = steps_per_epoch * epochs  # 24175
    print(f"  数据: {len(train_ds):,} 条训练 / {len(val_ds):,} 条验证")
    print(f"  步数: {steps_per_epoch} batches/epoch × {epochs} epochs = {total_steps} steps")
    print(f"  预计耗时: ~3-4 小时 (GPU) / ~8 小时 (CPU)")

    encoder = SharedEncoder(model_name=CONFIG["encoder"]["model_name"],
                            freeze_embeddings=False, freeze_layers=0)
    model = TextClassifier(encoder=encoder, num_classes=CONFIG["classifier"]["num_classes"],
                           dropout=CONFIG["classifier"]["dropout"])

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    print(f"  设备: {device}")
    print(f"  可训练参数: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    from torch.optim import AdamW
    from transformers import get_linear_schedule_with_warmup

    optimizer = AdamW(model.parameters(), lr=float(CONFIG["classifier"]["learning_rate"]))
    grad_accum = CONFIG["training"]["gradient_accumulation_steps"]
    opt_steps = total_steps // grad_accum
    warmup = int(opt_steps * float(CONFIG["classifier"].get("warmup_ratio", 0.1)))
    scheduler = get_linear_schedule_with_warmup(optimizer, warmup, opt_steps)

    train_losses = []
    val_losses, val_accs, eval_at_steps = [], [], []

    global_step = 0
    batch_step = 0
    last_eval_step = -1
    best_acc = 0.0
    start_epoch = 0

    # 断点续训
    resume_path = os.path.join(HIST_DIR, "cls_resume.pt")
    if resume and os.path.exists(resume_path):
        print(f"\n  🔄 加载断点: {resume_path}")
        ckpt = torch.load(resume_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        scheduler.load_state_dict(ckpt["scheduler_state"])
        batch_step = ckpt["batch_step"]
        global_step = ckpt["global_step"]
        start_epoch = ckpt["epoch"]  # 从当前 epoch 继续（跳已完成 batch）
        best_acc = ckpt.get("best_acc", 0.0)
        train_losses = ckpt.get("train_losses", [])
        val_losses = ckpt.get("val_losses", [])
        val_accs = ckpt.get("val_accs", [])
        eval_at_steps = ckpt.get("eval_at_steps", [])
        print(f"  从 Epoch {start_epoch+1}/5, Step {batch_step}/{total_steps} 继续")

    t_start = time.time()

    print("\n  训练中...")
    for epoch in range(start_epoch, epochs):
        model.train()
        # 续训时跳过当前 epoch 已处理的 batch
        batches_to_skip = 0
        if resume and epoch == start_epoch and batch_step > 0:
            # 计算本 epoch 已完成的 batch 数
            completed_in_epoch = batch_step % steps_per_epoch
            batches_to_skip = completed_in_epoch
            if batches_to_skip > 0:
                print(f"    跳过本 epoch 前 {batches_to_skip} 个 batch")

        for batch_idx, batch in enumerate(train_dl):
            if batch_idx < batches_to_skip:
                continue
            batch_gpu = {k: v.to(device) for k, v in batch.items()}
            outputs = model(input_ids=batch_gpu["input_ids"],
                            attention_mask=batch_gpu["attention_mask"],
                            token_type_ids=batch_gpu.get("token_type_ids", None),
                            label=batch_gpu["label"])
            loss = outputs["loss"] / grad_accum
            loss.backward()

            train_losses.append(float(outputs["loss"].item()))
            batch_step += 1

            if (batch_step % grad_accum) == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            # 每 eval_steps 个 optimizer step 评估一次
            if global_step > 0 and global_step % eval_steps == 0 and global_step != last_eval_step:
                last_eval_step = global_step
                v_loss, v_acc = _eval_cls(model, val_dl, device)
                val_losses.append(v_loss)
                val_accs.append(v_acc)
                eval_at_steps.append(batch_step)
                if v_acc > best_acc:
                    best_acc = v_acc
                elapsed = time.time() - t_start
                pct = batch_step / total_steps * 100
                eta = elapsed / batch_step * (total_steps - batch_step) if batch_step > 0 else 0
                print(f"    Step {batch_step:>6d}/{total_steps} ({pct:>4.1f}%) | "
                      f"T Loss: {train_losses[-1]:.4f} | V Loss: {v_loss:.4f} | "
                      f"V Acc: {v_acc:.4f} | ETA: {eta/60:.0f}m")
                model.train()

                # 每次评估都保存 checkpoint（防止断电丢失）
                ckpt_data = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "batch_step": batch_step,
                    "global_step": global_step,
                    "epoch": epoch,
                    "best_acc": best_acc,
                    "train_losses": train_losses,
                    "val_losses": val_losses,
                    "val_accs": val_accs,
                    "eval_at_steps": eval_at_steps,
                }
                ckpt_path = os.path.join(HIST_DIR, "cls_resume.pt")
                torch.save(ckpt_data, ckpt_path)
                with open(os.path.join(HIST_DIR, "cls_training_partial.json"), "w", encoding="utf-8") as f:
                    json.dump({"eval_at_steps": eval_at_steps, "val_losses": val_losses,
                               "val_accs": val_accs, "train_losses": train_losses,
                               "batch_step": batch_step, "epoch": epoch}, f, ensure_ascii=False, indent=2)

        # Epoch 结束：用完整验证集评估
        v_loss, v_acc = _eval_cls(model, val_dl_full, device)
        val_losses.append(v_loss)
        val_accs.append(v_acc)
        eval_at_steps.append(batch_step)
        if v_acc > best_acc:
            best_acc = v_acc
        print(f"  Epoch {epoch+1}/{epochs} 完成 | Full Val Loss: {v_loss:.4f} | Full Val Acc: {v_acc:.4f}")

        # 定期保存中间结果（防止中断丢失）
        mid_hist = {
            "train_losses": train_losses,
            "val_losses": val_losses, "val_accs": val_accs,
            "eval_at_steps": eval_at_steps,
            "total_steps": total_steps, "epochs": epochs,
            "steps_per_epoch": steps_per_epoch,
            "current_epoch": epoch + 1,
        }
        mid_path = os.path.join(HIST_DIR, "cls_training_partial.json")
        with open(mid_path, "w", encoding="utf-8") as f:
            json.dump(mid_hist, f, ensure_ascii=False, indent=2)
        # 保存模型
        torch.save(model.state_dict(), os.path.join(HIST_DIR, f"cls_epoch{epoch+1}.pt"))
        print(f"  [已保存 epoch {epoch+1} checkpoint]")

    t_total = time.time() - t_start
    print(f"  训练耗时: {t_total/60:.1f} 分钟 ({t_total/3600:.1f} 小时)")
    print(f"  最佳验证准确率: {best_acc:.4f}")

    # 保存历史
    cls_hist = {
        "train_losses": train_losses,
        "val_losses": val_losses,
        "val_accs": val_accs,
        "eval_at_steps": eval_at_steps,
        "total_steps": total_steps,
        "epochs": epochs,
        "steps_per_epoch": steps_per_epoch,
        "best_acc": float(best_acc),
        "training_time_min": round(t_total / 60, 1),
    }
    with open(os.path.join(HIST_DIR, "cls_training_history.json"), "w", encoding="utf-8") as f:
        json.dump(cls_hist, f, ensure_ascii=False, indent=2)

    # ---- 绘图 ----
    _plot_cls_curves(train_losses, val_losses, val_accs, eval_at_steps,
                     total_steps, steps_per_epoch, epochs, best_acc)
    return cls_hist


def _eval_cls(model, val_dl, device):
    model.eval()
    total_loss, all_p, all_l = 0.0, [], []
    with torch.no_grad():
        for batch in val_dl:
            bg = {k: v.to(device) for k, v in batch.items()}
            out = model(input_ids=bg["input_ids"], attention_mask=bg["attention_mask"],
                        token_type_ids=bg.get("token_type_ids", None), label=bg["label"])
            total_loss += float(out["loss"].item())
            all_p.extend(torch.argmax(out["logits"], dim=-1).cpu().tolist())
            all_l.extend(bg["label"].cpu().tolist())
    from sklearn.metrics import accuracy_score
    return total_loss / len(val_dl), accuracy_score(all_l, all_p)


def _plot_cls_curves(train_losses, val_losses, val_accs, eval_at_steps,
                     total_steps, steps_per_epoch, epochs, best_acc):
    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 平滑 train loss
    w = max(1, len(train_losses) // 600)
    if w > 1:
        kernel = np.ones(w) / w
        smooth = np.convolve(train_losses, kernel, mode="valid")
        smooth_x = np.arange(w, len(train_losses) + 1)
        t_line, = ax1.plot(smooth_x, smooth, color="#e74c3c", linewidth=0.8, alpha=0.85, label="训练 Loss")
    else:
        t_line, = ax1.plot(np.arange(1, len(train_losses) + 1), train_losses,
                            color="#e74c3c", linewidth=0.8, alpha=0.85, label="训练 Loss")

    vl_line, = ax1.plot(eval_at_steps, val_losses, color="#3498db",
                         linewidth=2.0, marker="o", markersize=3, label="验证 Loss", zorder=5)

    ax1.set_xlabel("训练步数 (batch steps)", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12, color="#e74c3c")
    ax1.tick_params(axis="y", labelcolor="#e74c3c")
    ax1.set_xlim(0, total_steps)

    ax2 = ax1.twinx()
    acc_line, = ax2.plot(eval_at_steps, val_accs, color="#27ae60",
                          linewidth=2.5, marker="s", markersize=3, label="验证 Accuracy", zorder=6)
    ax2.set_ylabel("Accuracy", fontsize=12, color="#27ae60")
    ax2.tick_params(axis="y", labelcolor="#27ae60")
    ax2.set_ylim(0, 1.05)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

    # Epoch 边界
    for ep in range(1, epochs + 1):
        x = ep * steps_per_epoch
        ax1.axvline(x=x, color="#bdc3c7", linestyle="--", linewidth=0.7, alpha=0.6)
        ax1.text(x + total_steps * 0.003, ax1.get_ylim()[1] * 0.96,
                 f"Epoch {ep}", fontsize=8, color="#95a5a6", rotation=90, va="top")

    # 过拟合起点 = 验证 loss 最低点
    if val_losses:
        best_idx = int(np.argmin(val_losses))
        best_step = eval_at_steps[best_idx]
        ax1.axvline(x=best_step, color="#e74c3c", linestyle=":", linewidth=1.5, alpha=0.7)
        ax1.annotate(f"最佳验证点\nStep {best_step}",
                     xy=(best_step, val_losses[best_idx]),
                     xytext=(best_step + total_steps * 0.03, val_losses[best_idx] + 0.15),
                     fontsize=9, color="#c0392b",
                     arrowprops=dict(arrowstyle="->", color="#c0392b", lw=1.2),
                     bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#c0392b", alpha=0.85))

    ax1.legend([t_line, vl_line, acc_line],
               ["训练 Loss", "验证 Loss", "验证 Accuracy"],
               loc="upper right", fontsize=10, framealpha=0.9)
    ax1.set_title(f"文本分类训练曲线 — Loss & Accuracy (最佳 Acc={best_acc:.4f})",
                  fontsize=14, fontweight="bold", pad=14)
    ax1.grid(axis="both", alpha=0.2, linestyle="--")
    ax1.spines["top"].set_visible(False)

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig2_classification_curves.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  已保存: {path}")


# ============================================================
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ner", action="store_true", default=False, help="仅训练NER并绘图3")
    parser.add_argument("--cls", action="store_true", default=False, help="仅训练分类并绘图2")
    parser.add_argument("--all", action="store_true", default=False, help="全部训练")
    parser.add_argument("--resume", action="store_true", default=False, help="从断点续训分类")
    args = parser.parse_args()

    # 如果没指定任何参数，默认 --all
    if not args.ner and not args.cls:
        args.all = True

    do_ner = args.all or args.ner
    do_cls = args.all or args.cls

    print("完整模式训练 → 图2 + 图3")
    print(f"输出目录: {OUTPUT_DIR}\n")

    if do_ner:
        train_and_plot_ner()

    if do_cls:
        train_and_plot_classification(resume=args.resume)

    # 汇总
    print("\n" + "=" * 60)
    print("生成完毕！figures/ 目录:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}  ({os.path.getsize(fp)/1024:.1f} KB)")
    print("=" * 60)
