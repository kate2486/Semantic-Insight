# Semantic-Insight

基于 PyTorch + BERT 的中文双任务NLP系统：**文本分类** + **命名实体识别（NER）**。

## 项目结构

```
Semantic-Insight/
├── src/
│   ├── data/              # 数据下载、预处理、Dataset定义
│   │   ├── download.py    # 数据集下载（THUCNews + PeopleDaily）
│   │   ├── preprocess.py  # 数据清洗与预处理
│   │   └── dataset.py     # PyTorch Dataset + DataLoader
│   ├── models/            # 模型模块
│   │   ├── encoder.py         # 共享BERT编码器
│   │   ├── classifier.py      # 文本分类头
│   │   ├── ner_tagger.py      # NER序列标注头（CRF）
│   │   └── multitask_model.py # 多任务组合推理模型
│   ├── train/             # 训练模块
│   │   ├── trainer.py         # 通用训练器
│   │   ├── train_classifier.py # 分类任务训练
│   │   └── train_ner.py       # NER任务训练
│   ├── evaluate/          # 评估模块
│   │   └── metrics.py     # 准确率/F1/混淆矩阵
│   └── app/               # 演示界面
│       └── demo.py        # Gradio Web界面
├── configs/config.yaml    # 统一超参数配置
├── main.py                # CLI统一入口
├── tests/                 # 单元测试
├── checkpoints/           # 模型权重
└── data/                  # 数据集目录
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 下载并处理数据
python -m src.data.download
python -m src.data.preprocess

# 3. 训练模型
python main.py train --task cls    # 训练分类模型
python main.py train --task ner    # 训练NER模型
python main.py train --task all    # 训练所有模型

# 4. 命令行推理
python main.py predict --text "华为在深圳举办开发者大会"

# 5. 启动Web演示（需要先训练完模型）
python main.py demo
```

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 框架 | PyTorch 2.0 + HuggingFace Transformers | 业界标准 |
| 编码器 | bert-base-chinese | 12层中文Transformer |
| 分类头 | [CLS] → Dropout → Linear(768→10) | 10类文本分类 |
| NER头 | 每Token → Linear → CRF | BIO序列标注 |
| 界面 | Gradio 4.0 | 机器学习专用Web框架 |

## 性能目标

| 任务 | 指标 | 目标 |
|------|------|------|
| 文本分类 | Accuracy | ≥ 90% |
| NER | Macro F1 | ≥ 80% |

## 运行测试

```bash
pytest tests/ -v
```
