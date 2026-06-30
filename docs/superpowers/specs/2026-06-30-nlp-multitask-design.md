# Semantic-Insight — 设计文档

**日期**: 2026-06-30  
**作者**: 芝士番薯  
**状态**: 已确认

---

## 1. 项目概述

基于Python和PyTorch，使用共享BERT编码器同时完成**文本分类**和**命名实体识别(NER)**两个NLP任务，最终用Gradio搭建Web演示界面。

### 1.1 目标

- 5天内完成从数据采集到模型部署的全流程
- 体现多任务学习思想（一个共享编码器 + 两个任务头）
- 作为神经网络实训作品，具备完整性和可展示性

### 1.2 硬件环境

- GPU: NVIDIA RTX 4060 8GB（本地训练）
- 框架: PyTorch + HuggingFace Transformers

---

## 2. 项目目录结构

```
Semantic-Insight/
├── data/                          # 数据目录
│   ├── raw/                       # 原始下载数据
│   ├── processed/                 # 清洗后的数据
│   └── README.md                  # 数据说明
├── src/                           # 源代码
│   ├── __init__.py
│   ├── data/                      # 数据模块
│   │   ├── __init__.py
│   │   ├── download.py            # 数据集下载脚本
│   │   ├── preprocess.py          # 数据清洗与预处理
│   │   └── dataset.py             # PyTorch Dataset定义
│   ├── models/                    # 模型模块
│   │   ├── __init__.py
│   │   ├── encoder.py             # 共享BERT编码器
│   │   ├── classifier.py          # 文本分类头
│   │   ├── ner_tagger.py          # NER序列标注头
│   │   └── multitask_model.py     # 多任务组合模型
│   ├── train/                     # 训练模块
│   │   ├── __init__.py
│   │   ├── trainer.py             # 通用训练器
│   │   ├── train_classifier.py    # 训练分类任务
│   │   └── train_ner.py           # 训练NER任务
│   ├── evaluate/                  # 评估模块
│   │   ├── __init__.py
│   │   └── metrics.py             # 准确率/F1/混淆矩阵
│   └── app/                       # 演示界面
│       ├── __init__.py
│       └── demo.py                # Gradio Web界面
├── notebooks/                     # Jupyter实验笔记
│   ├── 01_data_exploration.ipynb
│   ├── 02_text_classification.ipynb
│   └── 03_ner_training.ipynb
├── checkpoints/                   # 模型权重保存
├── configs/                       # 配置文件
│   └── config.yaml                # 统一超参数配置
├── tests/                         # 单元测试
│   ├── test_data.py
│   └── test_models.py
├── docs/                          # 项目文档
│   └── report.md                  # 实训报告
├── requirements.txt               # Python依赖
├── main.py                        # 统一入口（训练+评估+推理）
└── README.md                      # 项目说明
```

---

## 3. 技术架构

### 3.1 整体架构

```
┌─────────────────────────────────────────────────────┐
│                    演示层 (Gradio)                     │
│           输入文本 → 同时返回分类 + NER结果             │
└──────────────────────┬──────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────┐
│                   推理层 (main.py)                     │
│          加载checkpoint → 统一推理接口                  │
└──────────┬───────────────────────────────┬───────────┘
           │                               │
┌──────────▼──────────┐       ┌────────────▼───────────┐
│  分类头 (classifier)  │       │  标注头 (ner_tagger)    │
│  BERT [CLS] → MLP    │       │  BERT每Token → CRF     │
│  → 类别概率          │       │  → BIO标签序列          │
└──────────┬──────────┘       └────────────┬───────────┘
           │                               │
           └───────────┬───────────────────┘
                       │
           ┌───────────▼───────────────┐
           │  共享BERT编码器 (encoder)   │
           │  bert-base-chinese        │
           │  768维向量 → 双任务共用     │
           └───────────────────────────┘
                       │
           ┌───────────▼───────────────┐
           │   数据加载 (dataset.py)     │
           │   分词 → 编码 → DataLoader  │
           └───────────────────────────┘
```

### 3.2 技术选型

| 层 | 技术 | 选择理由 |
|----|------|---------|
| 编码器 | `bert-base-chinese` (HuggingFace) | 中文预训练，开箱即用，12层Transformer |
| 分类头 | `[CLS]向量 → Dropout → Linear(768→10)` | BERT分类的标准做法 |
| 标注头 | `每Token向量 → Linear(768→num_labels) → CRF` | CRF保证标签序列合理性 |
| 框架 | `PyTorch + Transformers` | 业界标准，社区资源丰富 |
| 界面 | `Gradio` | 5行Python代码出Web界面，专为ML演示设计 |

---

## 4. 数据方案

### 4.1 数据集

| 任务 | 数据集 | 规模 | 类别 |
|------|--------|------|------|
| 文本分类 | THUCNews子集 | 10类 × 6000条 = 60000条 | 体育/财经/科技/教育/时尚/军事/游戏/房产/娱乐/时政 |
| NER | PeopleDaily | ~20000条标注语料 | PER(人名)/LOC(地名)/ORG(机构) |

### 4.2 数据流

```
raw/                    →    processed/              →    DataLoader
─────────────────────────────────────────────────────────────────
THUCNews 目录结构              news_train.json              [CLS]输入
  ├─ 体育/xxx.txt    →         {text, label_id}             ↓
  ├─ 财经/xxx.txt             news_val.json             bert_tokenizer
  └─ ...                     news_test.json              ↓
                                                    input_ids,
PeopleDaily.txt              ner_train.json          attention_mask,
  BIO标注格式        →         {tokens, bio_tags}      token_type_ids
                             ner_val.json
                             ner_test.json
```

### 4.3 预处理步骤

1. **统一清洗**: 去HTML标签、特殊符号、全角半角统一、截断>512字符
2. **分类数据**: 文件夹名 → label_id (0-9), split 8:1:1
3. **NER数据**: BIO格式解析 → tokens与tags对齐, split 8:1:1
4. **Token化**: BERT tokenizer → input_ids + attention_mask + token_type_ids

### 4.4 BIO标注格式

```
标签      含义
B-PER    人名首字 (Begin-Person)
I-PER    人名后续 (Inside-Person)
B-LOC    地名首字
I-LOC    地名后续
B-ORG    机构首字
I-ORG    机构后续
O        非实体 (Outside)

示例: 华/为/在/深/圳/开/发/布/会
     B-ORG I-ORG O B-LOC I-LOC O O O O
```

---

## 5. 标注规范

- 共7个标签: `O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG`
- 每个Token对应一个标签
- [CLS]和[SEP]等特殊Token标签置为-100（loss计算时忽略）
- CRF层保证转移约束（如 I-ORG 不能紧跟在 B-PER 后面）

---

## 6. 5天训练计划

| | Day 1 | Day 2 | Day 3 | Day 4 | Day 5 |
|---|---|---|---|---|---|
| 主题 | 数据准备 | 共享编码器 | 分类任务 | NER任务 | 整合演示 |
| 输入 | 原始文件 | Day1输出 | Day2编码器 | Day2编码器 | Day3+4模型 |
| 输出 | 清洗后数据 | 编码向量验证 | 分类模型 | NER模型 | Gradio应用 |

### Day 1 — 数据准备
- 写 `download.py`: 自动下载THUCNews和PeopleDaily
- 写 `preprocess.py`: 清洗、格式化、split
- 写 `dataset.py`: PyTorch Dataset + DataLoader
- 在 `notebooks/01_data_exploration.ipynb` 可视化数据分布

### Day 2 — BERT编码器搭建
- 写 `encoder.py`: 加载 `bert-base-chinese`，冻结embedding和前6层
- 验证输出维度: `(batch_size, seq_len, 768)`
- 写 `trainer.py`: 通用训练循环

### Day 3 — 分类任务训练
- 写 `classifier.py`: [CLS] → Dropout → Linear(768→10)
- 训练3-5 epoch，记录loss/accuracy曲线
- 在测试集评估，生成混淆矩阵

### Day 4 — NER任务训练
- 写 `ner_tagger.py`: 每Token Linear → CRF解码
- 训练5-10 epoch（NER收敛较慢）
- 评估per-entity F1 score

### Day 5 — 整合与演示
- 写 `multitask_model.py`: 编码器 + 双头统一接口
- 写 `main.py`: 统一入口（训练/评估/推理）
- 写 `demo.py`: Gradio界面
- 写 `docs/report.md`: 实训报告
- 录演示gif

---

## 7. 依赖项

```
torch>=2.0.0
transformers>=4.30.0
gradio>=4.0.0
scikit-learn
pandas
numpy
matplotlib
seaborn
tqdm
pyyaml
pytorch-crf
jieba
```

---

## 8. 成功标准

| 指标 | 目标 |
|------|------|
| 文本分类准确率 | ≥ 90%（10分类随机基线10%） |
| NER Macro F1 | ≥ 80% |
| Gradio推理响应 | < 3秒/条 |
| 代码结构完整性 | 所有模块可独立测试 |
| 文档完整性 | README + 代码注释 + 实训报告 |

---

## 9. 变更记录

| 日期 | 变更内容 | 作者 |
|------|---------|------|
| 2026-06-30 | 初始设计 | 芝士番薯 |