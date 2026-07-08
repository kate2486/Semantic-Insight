"""
生成图1（类别分布柱状图）和图4（混淆矩阵热力图）
纯推理，不需要重新训练
"""
import os
import sys
import json
import yaml
import torch
import numpy as np
from collections import Counter

sys.stdout.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(os.path.join(BASE_DIR, "configs", "config.yaml"), "r", encoding="utf-8") as f:
    CONFIG = yaml.safe_load(f)

LABEL_NAMES = CONFIG["classifier"]["label_names"]  # ["体育","财经","科技","教育","时尚","社会","游戏","房产","娱乐","时政"]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# 中文字体
plt.rcParams["font.sans-serif"] = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# 学术论文配色
BAR_COLORS = ["#5DADE2", "#48C9B0", "#F4D03F", "#EC7063", "#AF7AC5",
              "#5499C7", "#45B39D", "#F5B041", "#EB984E", "#A569BD"]


# ============================================================
# 图1: 类别分布柱状图
# ============================================================
def generate_fig1():
    print("=" * 50)
    print("[图1] 类别分布柱状图")
    print("=" * 50)

    counts = {}
    for split, filename in [("train", "news_train.json"), ("val", "news_val.json"), ("test", "news_test.json")]:
        with open(os.path.join(BASE_DIR, "data", "processed", filename), "r", encoding="utf-8") as f:
            data = json.load(f)
        c = Counter(item["label"] for item in data)
        counts[split] = [c.get(i, 0) for i in range(10)]

    # 打印统计
    print(f"\n  {'类别':<6} {'训练集':>8} {'验证集':>8} {'测试集':>8} {'总计':>8}")
    print(f"  {'-'*45}")
    for i, name in enumerate(LABEL_NAMES):
        total = counts["train"][i] + counts["val"][i] + counts["test"][i]
        print(f"  {name:<6} {counts['train'][i]:>8,} {counts['val'][i]:>8,} {counts['test'][i]:>8,} {total:>8,}")

    # 三子图
    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))
    fig.suptitle("THUCNews 数据集类别分布", fontsize=16, fontweight="bold", y=1.01)

    titles = [("训练集", "train"), ("验证集", "val"), ("测试集", "test")]
    x = np.arange(len(LABEL_NAMES))

    for ax_idx, (title, key) in enumerate(titles):
        ax = axes[ax_idx]
        data_vals = counts[key]
        total = sum(data_vals)

        bars = ax.bar(x, data_vals, color=BAR_COLORS, edgecolor="white", linewidth=0.5, width=0.72)

        # 数值标注（只在 >15% 最大值的柱子上标）
        threshold = max(data_vals) * 0.12
        for bar, val in zip(bars, data_vals):
            if val > threshold:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + total * 0.006,
                        f"{val:,}", ha="center", va="bottom", fontsize=7, color="#2c3e50")

        ax.set_title(f"{title}\n(n = {total:,})", fontsize=13, fontweight="bold", pad=8)
        ax.set_xticks(x)
        ax.set_xticklabels(LABEL_NAMES, rotation=30, ha="right", fontsize=9.5)
        ax.set_ylabel("样本数量", fontsize=11)
        ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda v, _: f"{v/1000:.0f}k"))
        ax.set_ylim(0, max(data_vals) * 1.18)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.25, linestyle="--", linewidth=0.5)

    plt.tight_layout(pad=2)
    path = os.path.join(OUTPUT_DIR, "fig1_eda_category_distribution.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  已保存: {path}")
    return counts


# ============================================================
# 图4: 混淆矩阵热力图
# ============================================================
def generate_fig4():
    print("\n" + "=" * 50)
    print("[图4] 混淆矩阵热力图")
    print("=" * 50)

    from src.models.encoder import SharedEncoder
    from src.models.classifier import TextClassifier
    from src.data.dataset import create_dataloaders
    from sklearn.metrics import confusion_matrix, accuracy_score

    print("  加载模型...")
    encoder = SharedEncoder(
        model_name=CONFIG["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )
    model = TextClassifier(
        encoder=encoder,
        num_classes=CONFIG["classifier"]["num_classes"],
        dropout=CONFIG["classifier"]["dropout"],
    )
    ckpt = os.path.join(BASE_DIR, "checkpoints", "best_classification.pt")
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=False)
    model.eval()
    print(f"  已加载: {ckpt}")

    # DataLoader
    dataloaders, _ = create_dataloaders(os.path.join(BASE_DIR, "data", "processed"), CONFIG)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    print("  推理测试集...")
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

    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"  测试集准确率: {acc:.4f} ({acc*100:.2f}%)")

    # 各类别召回率
    print(f"\n  各类别指标:")
    print(f"  {'类别':<6} {'正确':>6} {'总数':>6} {'召回率':>8}")
    print(f"  {'-'*30}")
    for i, name in enumerate(LABEL_NAMES):
        row_sum = cm[i].sum()
        correct = cm[i][i]
        print(f"  {name:<6} {correct:>6} {row_sum:>6} {correct/row_sum:>7.2%}")

    # ---- 绘制 ----
    fig, ax = plt.subplots(figsize=(10.5, 9))

    # 用 YlOrRd 让对角线深色、误分类浅色
    cmap = sns.color_palette("YlOrRd", as_cmap=True)

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap=cmap,
        xticklabels=LABEL_NAMES,
        yticklabels=LABEL_NAMES,
        linewidths=0.6,
        linecolor="white",
        cbar_kws={"shrink": 0.78, "label": "样本数", "aspect": 25},
        ax=ax,
        annot_kws={"fontsize": 9, "fontweight": "bold"},
        vmin=0,
        vmax=cm.max(),
    )

    # 绿框标注对角线（正确分类）
    for i in range(len(LABEL_NAMES)):
        ax.add_patch(plt.Rectangle(
            (i, i), 1, 1, fill=False,
            edgecolor="#27ae60", linewidth=2.8, linestyle="-"
        ))

    ax.set_title(f"文本分类混淆矩阵\n(测试集准确率 = {acc*100:.2f}%)",
                 fontsize=14, fontweight="bold", pad=16)
    ax.set_xlabel("预测类别", fontsize=12, labelpad=10)
    ax.set_ylabel("真实类别", fontsize=12, labelpad=10)
    ax.tick_params(axis="both", labelsize=10)
    ax.set_aspect("equal")

    plt.tight_layout()
    path = os.path.join(OUTPUT_DIR, "fig4_confusion_matrix.png")
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"\n  已保存: {path}")
    return {"accuracy": float(acc), "cm": cm.tolist()}


# ============================================================
if __name__ == "__main__":
    print("论文图表生成 — 图1 & 图4")
    print(f"输出目录: {OUTPUT_DIR}\n")

    generate_fig1()
    generate_fig4()

    print(f"\n{'='*50}")
    print("完成！输出文件:")
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fp = os.path.join(OUTPUT_DIR, f)
        print(f"  {f}  ({os.path.getsize(fp)/1024:.1f} KB)")
