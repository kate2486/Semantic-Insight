"""
使用真实 THUCNews 数据准备分类训练集
读取从 HuggingFace Tongjilibo/THUCNews 下载的 jsonl 文件
"""
import os
import sys
import json
import random

sys.stdout.reconfigure(encoding='utf-8')

# 10 个分类及其标签索引（与 config.yaml 一致）
CATEGORIES = {
    "体育": 0,
    "财经": 1,
    "科技": 2,
    "教育": 3,
    "时尚": 4,
    "社会": 5,
    "游戏": 6,
    "房产": 7,
    "娱乐": 8,
    "时政": 9,
}

LABEL_NAMES = ["体育", "财经", "科技", "教育", "时尚", "社会", "游戏", "房产", "娱乐", "时政"]


def load_thucnews(data_dir: str, max_per_class: int = 20000):
    """加载 THUCNews jsonl 文件，提取 title 作为分类文本"""
    all_data = []

    for cat_name, label_idx in CATEGORIES.items():
        filepath = os.path.join(data_dir, f"{cat_name}.jsonl")
        if not os.path.exists(filepath):
            print(f"  ⚠ {cat_name}.jsonl 未找到，跳过")
            continue

        samples = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    title = item.get("title", "").strip()
                    if title and len(title) >= 4:  # 过滤太短的标题
                        samples.append({"text": title, "label": label_idx})
                except json.JSONDecodeError:
                    continue

        # 每类取最多 max_per_class 条
        if len(samples) > max_per_class:
            samples = random.sample(samples, max_per_class)

        print(f"  {cat_name}: {len(samples)} 条")
        all_data.extend(samples)

    return all_data


def save_splits(data, prefix, output_dir="data/processed"):
    """按 80/10/10 划分并保存"""
    random.seed(42)
    random.shuffle(data)
    n = len(data)
    train_end = int(n * 0.8)
    val_end = train_end + int(n * 0.1)

    splits = {
        f"{prefix}_train.json": data[:train_end],
        f"{prefix}_val.json": data[train_end:val_end],
        f"{prefix}_test.json": data[val_end:],
    }

    os.makedirs(output_dir, exist_ok=True)
    for fname, subset in splits.items():
        fpath = os.path.join(output_dir, fname)
        # 备份旧文件
        if os.path.exists(fpath):
            backup = fpath + ".bak"
            if not os.path.exists(backup):
                os.rename(fpath, backup)
                print(f"  📦 旧文件已备份: {fname}.bak")

        with open(fpath, "w", encoding="utf-8") as f:
            json.dump(subset, f, ensure_ascii=False, indent=2)
        print(f"  {fname}: {len(subset)} 条")

    # 统计各分类在训练集中的分布
    from collections import Counter
    label_counts = Counter(s["label"] for s in data[:train_end])
    print("\n  训练集分类分布:")
    for idx, name in enumerate(LABEL_NAMES):
        print(f"    {name}: {label_counts.get(idx, 0)} 条")


def main():
    print("=" * 60)
    print("Semantic-Insight 真实数据准备")
    print("=" * 60)

    data_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "thucnews_real"
    )

    print(f"\n从 {data_dir} 加载真实 THUCNews 数据...")
    cls_data = load_thucnews(data_dir, max_per_class=20000)
    print(f"\n总计: {len(cls_data)} 条")

    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        "data", "processed"
    )
    save_splits(cls_data, "news", output_dir)

    print("\n✅ 真实数据准备完成！可运行 python main.py train --task cls 开始训练")


if __name__ == "__main__":
    main()
