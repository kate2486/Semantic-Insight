"""
数据下载模块
下载 THUCNews 子集（10分类文本）和 PeopleDaily NER 数据集
"""
import os
import random
import shutil
import sys
import tarfile
import zipfile
import requests
from pathlib import Path
from tqdm import tqdm


def download_file(url: str, dest: str, desc: str = "Downloading"):
    """带进度条的文件下载"""
    response = requests.get(url, stream=True, timeout=30)
    total = int(response.headers.get("content-length", 0))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    with open(dest, "wb") as f, tqdm(
        total=total, unit="B", unit_scale=True, desc=desc
    ) as pbar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            pbar.update(len(chunk))


def _is_valid_zip(path: str) -> bool:
    """检查文件是否为有效的ZIP文件"""
    try:
        with zipfile.ZipFile(path, "r") as zf:
            return len(zf.namelist()) > 0
    except zipfile.BadZipFile:
        return False


def _generate_sample_thucnews(thucnews_dir: str):
    """
    生成示例 THUCNews 文本分类数据
    用于在无法访问网络时仍能跑通项目流程
    """
    categories = ["体育", "财经", "科技", "教育", "时尚", "军事", "游戏", "房产", "娱乐", "时政"]
    sample_texts = {
        "体育": ["中国男篮在亚运会比赛中获得金牌", "世界杯预选赛今晚开打"],
        "财经": ["股市震荡走高，沪指突破3500点", "央行下调存款准备金率"],
        "科技": ["人工智能技术取得重大突破", "5G网络覆盖范围进一步扩大"],
        "教育": ["教育部发布新课程改革方案", "高考报名人数再创新高"],
        "时尚": ["春季时装周在巴黎开幕", "环保材料成为服装设计新趋势"],
        "军事": ["国防部发言人答记者问", "联合军演圆满结束"],
        "游戏": ["国产游戏《黑神话》火爆全球", "电子竞技入选亚运会正式项目"],
        "房产": ["一线城市房价趋于稳定", "保障性住房建设加快推进"],
        "娱乐": ["春节档电影票房突破80亿", "热门综艺节目收视率创新高"],
        "时政": ["全国两会顺利召开", "一带一路合作成果丰硕"],
    }

    os.makedirs(thucnews_dir, exist_ok=True)
    for cat in categories:
        cat_dir = os.path.join(thucnews_dir, cat)
        os.makedirs(cat_dir, exist_ok=True)
        texts = sample_texts.get(cat, ["示例文本"])
        for i in range(50):  # 每类生成50条
            text = random.choice(texts)
            file_path = os.path.join(cat_dir, f"{i:04d}.txt")
            with open(file_path, "w", encoding="utf-8") as f:
                f.write(text)
    print(f"  已生成示例 THUCNews 数据（{len(categories)} 类，每类 50 条）")
    print("  ⚠ 这是示例数据，用于跑通流程。真实训练建议连接网络下载完整数据集。")


def download_thucnews(raw_dir: str):
    """
    下载 THUCNews 子集（10分类，每类6000条）
    数据来源: THUCTC (thuctc.thunlp.org) 精简版
    """
    thucnews_dir = os.path.join(raw_dir, "thucnews")
    if os.path.exists(thucnews_dir) and len(os.listdir(thucnews_dir)) >= 10:
        print(f"THUCNews 已存在于 {thucnews_dir}")
        return thucnews_dir

    # THUCNews 子集下载链接（GitHub mirror）
    url = "https://github.com/649453932/Chinese-Text-Classification-Pytorch/releases/download/v1.0/THUCNews.zip"

    # 备用: 使用清华源
    backup_url = "https://thunlp.oss-cn-qingdao.aliyuncs.com/THUCNews.zip"

    zip_path = os.path.join(raw_dir, "thucnews.zip")
    download_succeeded = False
    for source_name, source_url in [("GitHub mirror", url), ("阿里云备用源", backup_url)]:
        try:
            print(f"尝试从 {source_name} 下载 THUCNews...")
            download_file(source_url, zip_path, "下载 THUCNews")
            if _is_valid_zip(zip_path):
                download_succeeded = True
                break
            else:
                print(f"  文件无效（非ZIP格式），尝试其他源...")
                os.remove(zip_path)
        except Exception as e:
            print(f"  {source_name} 失败: {e}")
            if os.path.exists(zip_path):
                os.remove(zip_path)

    if not download_succeeded:
        print("所有下载源均失败，生成示例 THUCNews 数据...")
        _generate_sample_thucnews(thucnews_dir)
        print(f"THUCNews 示例数据已生成: {thucnews_dir}")
        return thucnews_dir

    print("解压 THUCNews...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(raw_dir)

    # 清理zip文件
    os.remove(zip_path)

    # THUCNews解压后结构为 THUCNews/类别/xxx.txt
    extracted = os.path.join(raw_dir, "THUCNews")
    if os.path.exists(extracted):
        # 只保留前10个类别
        target_dir = thucnews_dir
        os.makedirs(target_dir, exist_ok=True)
        categories = sorted(os.listdir(extracted))[:10]
        for cat in categories:
            cat_src = os.path.join(extracted, cat)
            cat_dst = os.path.join(target_dir, cat)
            if os.path.isdir(cat_src) and not os.path.exists(cat_dst):
                shutil.move(cat_src, cat_dst)
        shutil.rmtree(extracted)

    print(f"THUCNews 下载完成: {thucnews_dir}")
    return thucnews_dir


def download_peopledaily(raw_dir: str):
    """
    下载 PeopleDaily NER 数据集
    使用 HuggingFace datasets 库直接加载
    """
    ner_dir = os.path.join(raw_dir, "peopledaily")
    if os.path.exists(ner_dir) and os.path.exists(os.path.join(ner_dir, "train.txt")):
        print(f"PeopleDaily 已存在于 {ner_dir}")
        return ner_dir

    os.makedirs(ner_dir, exist_ok=True)
    try:
        from datasets import load_dataset
        print("从 HuggingFace 加载 PeopleDaily NER 数据集...")
        dataset = load_dataset("shibing624/peoples_daily_ner")
        for split_name in ["train", "validation", "test"]:
            if split_name in dataset:
                output_path = os.path.join(ner_dir, f"{split_name}.txt")
                with open(output_path, "w", encoding="utf-8") as f:
                    for item in dataset[split_name]:
                        tokens = item["tokens"]
                        tags = item["ner_tags"]
                        for token, tag in zip(tokens, tags):
                            f.write(f"{token}\t{tag}\n")
                        f.write("\n")  # 空行分隔句子
                print(f"  {split_name}.txt 已保存 ({len(dataset[split_name])} 条)")

    except Exception as e:
        print(f"HuggingFace 加载失败 ({e})，使用内置数据生成器...")
        _generate_sample_ner_data(ner_dir)

    return ner_dir


def _generate_sample_ner_data(ner_dir: str):
    """
    生成示例NER数据（含人名、地名、机构名）
    用于在无法访问网络时仍能跑通项目流程
    """
    samples = [
        ("中国 政府 在 北京 举行 记者招待会", "B-ORG I-ORG O O B-LOC O O O"),
        ("李克强 总理 访问 了 上海 和 杭州", "B-PER I-PER O O O B-LOC O B-LOC"),
        ("华为 公司 在 深圳 发布 新 产品", "B-ORG I-ORG O B-LOC O O O"),
        ("北京 大学 和 清华 大学 联合 举办 论坛", "B-ORG I-ORG O B-ORG I-ORG O O O"),
        ("张三 来到 了 纽约 参加 联合国 大会", "B-PER I-PER O O B-LOC O B-ORG I-ORG O"),
        # ... 更多样本 ...
        ("腾讯 和 阿里巴巴 是 中国 最大 的 互联网 公司", "B-ORG O B-ORG O B-LOC O O O O"),
        ("习近平 主席 在 人民大会堂 发表 重要 讲话", "B-PER I-PER O B-LOC I-LOC I-LOC O O O"),
        ("微软 公司 总部 位于 美国 西雅图", "B-ORG I-ORG O O B-LOC B-LOC I-LOC"),
        ("王 明 昨天 去了 广州 和 深圳", "B-PER I-PER O O B-LOC O B-LOC"),
        ("国务院 近日 在 北京 召开 常务会议", "B-ORG O O B-LOC O O I-ORG"),
    ]

    train_file = os.path.join(ner_dir, "train.txt")
    test_file = os.path.join(ner_dir, "test.txt")

    # 生成更多样本（简单扩充）
    random.seed(42)
    all_samples = samples.copy()
    for _ in range(200):
        base = random.choice(samples)
        all_samples.append(base)

    split = int(len(all_samples) * 0.8)
    for fname, subset in [(train_file, all_samples[:split]), (test_file, all_samples[split:])]:
        with open(fname, "w", encoding="utf-8") as f:
            for text, tags_str in subset:
                words = text.split()
                tags = tags_str.split()
                for w, t in zip(words, tags):
                    f.write(f"{w}\t{t}\n")
                f.write("\n")
    print(f"  已生成 {len(all_samples)} 条示例NER数据")
    print("  ⚠ 这是示例数据，用于跑通流程。真实训练建议连接网络下载完整数据集。")


def main():
    raw_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw")

    print("=" * 50)
    print("Step 1/2: 下载 THUCNews 文本分类数据集")
    print("=" * 50)
    download_thucnews(raw_dir)

    print()
    print("=" * 50)
    print("Step 2/2: 下载 PeopleDaily NER 数据集")
    print("=" * 50)
    download_peopledaily(raw_dir)

    print()
    print("数据集下载完成!")


if __name__ == "__main__":
    main()
