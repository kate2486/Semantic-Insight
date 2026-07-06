# Semantic Insight

基于 PyTorch + BERT 的中文双任务 NLP 系统：**文本分类** + **命名实体识别（NER）**。

## 项目结构

```
Semantic-Insight/
├── src/
│   ├── data/                  # 数据处理
│   │   ├── download.py        # 数据集下载
│   │   ├── preprocess.py      # 数据清洗与预处理
│   │   ├── prepare_data.py    # NER + 分类数据准备（旧：模板生成）
│   │   ├── prepare_real_data.py # 分类数据准备（新：真实 THUCNews）
│   │   └── dataset.py         # PyTorch Dataset + DataLoader
│   ├── models/                # 模型模块
│   │   ├── encoder.py         # 共享 BERT 编码器
│   │   ├── classifier.py      # 文本分类头（Linear）
│   │   ├── ner_tagger.py      # NER 序列标注头（CE / CRF）
│   │   └── multitask_model.py # 多任务组合模型（旧版，共享 encoder）
│   ├── train/                 # 训练模块
│   │   ├── trainer.py         # 通用训练器
│   │   ├── train_classifier.py # 分类任务训练
│   │   └── train_ner.py       # NER 任务训练
│   ├── evaluate/              # 评估模块
│   │   └── metrics.py         # 准确率 / F1 / 混淆矩阵
│   └── app/                   # Web 服务
│       ├── server.py          # FastAPI 服务器 + DualEncoderModel
│       ├── demo.py            # Gradio 演示（旧版）
│       └── static/
│           └── index.html     # 自定义前端页面
├── configs/config.yaml        # 统一超参数配置
├── main.py                    # CLI 统一入口
├── final_evaluation.py        # 测试集评估 + 真实案例测试
├── tests/                     # 单元测试
├── checkpoints/               # 模型权重
└── data/                      # 数据集目录
```

## 快速开始

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 准备数据
# 分类数据：从 HuggingFace 下载 THUCNews（~20 万条真实新闻标题）
python -m src.data.prepare_real_data

# 3. 训练模型
python main.py train --task cls    # 训练分类模型
python main.py train --task ner    # 训练 NER 模型

# 4. 命令行推理
python main.py predict --text "华为在深圳举办开发者大会"

# 5. 启动 Web 服务（自定义前端）
python main.py serve               # 访问 http://127.0.0.1:8000
python main.py serve --reload      # 开发模式（热重载）

# 6. 启动 Gradio 演示（旧版）
python main.py demo
```

## 技术栈

| 层 | 技术 | 说明 |
|----|------|------|
| 框架 | PyTorch 2.0 + HuggingFace Transformers | 业界标准 |
| 编码器 | bert-base-chinese | 12 层中文 Transformer |
| 分类头 | [CLS] → Dropout → Linear(768→10) | 10 类文本分类 |
| NER 头 | 每 Token → Linear → Argmax | BIO 序列标注（PER/LOC/ORG） |
| 推理架构 | **双 Encoder**（各自独立） | 避免权重覆盖，保证两任务独立最优 |
| 前端 | FastAPI + 纯 HTML/CSS/JS | 零外部依赖，可自由定制 |

## 模型性能

### 文本分类（THUCNews 10 分类）

| 类别 | Precision | Recall | F1 |
|------|:---:|:---:|:---:|
| 体育 | 0.974 | 0.953 | **0.963** |
| 财经 | 0.943 | 0.950 | **0.946** |
| 科技 | 0.869 | 0.847 | **0.858** |
| 教育 | 0.958 | 0.967 | **0.963** |
| 时尚 | 0.913 | 0.938 | **0.925** |
| 社会 | 0.919 | 0.910 | **0.915** |
| 游戏 | 0.900 | 0.955 | **0.927** |
| 房产 | 0.972 | 0.925 | **0.948** |
| 娱乐 | 0.932 | 0.904 | **0.918** |
| 时政 | 0.888 | 0.950 | **0.918** |
| **总体** | — | — | **93.29%** |

### 命名实体识别（chinese_common_ner）

| 实体 | Precision | Recall | F1 |
|------|:---:|:---:|:---:|
| PER（人名） | 0.955 | 0.974 | **0.964** |
| LOC（地名） | 0.931 | 0.936 | **0.933** |
| ORG（机构） | 0.923 | 0.899 | **0.911** |
| **Macro** | — | — | **0.936** |

## 推理架构说明

本项目采用**双 Encoder 架构**——分类和 NER 各自使用独立的 BERT encoder：

```
┌─────────────────────┐     ┌─────────────────────┐
│   CLS Encoder       │     │   NER Encoder       │
│   (分类训练权重)     │     │   (NER 训练权重)     │
├─────────────────────┤     ├─────────────────────┤
│   Classifier Head   │     │   NER Linear Head    │
└─────────────────────┘     └─────────────────────┘
```

> 两个 encoder 分别从 `best_classification.pt` 和 `best_ner.pt` 加载各自训练的权重，互不干扰。详见 `src/app/server.py` 中的 `DualEncoderModel` 类。

## 分类标签

`体育` `财经` `科技` `教育` `时尚` `社会` `游戏` `房产` `娱乐` `时政`

## NER 实体类型

`PER`（人名） `LOC`（地名） `ORG`（机构）

## API 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 自定义前端页面 |
| GET | `/api/health` | 健康检查 |
| POST | `/api/predict` | 文本分析（分类 + NER） |
| GET | `/docs` | Swagger API 文档 |

### 推理请求示例

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "今天NBA总决赛，洛杉矶湖人队大胜波士顿凯尔特人"}'
```

响应：
```json
{
  "text": "今天NBA总决赛，洛杉矶湖人队大胜波士顿凯尔特人",
  "classification": { "label": "体育", "confidence": 0.9946 },
  "entities": [
    { "word": "洛杉矶湖人队", "type": "ORG", "start": 8, "end": 14 },
    { "word": "波士顿凯尔特人", "type": "ORG", "start": 17, "end": 24 }
  ]
}
```

## 运行测试

```bash
pytest tests/ -v
```

## 已知限制

- `bert-base-chinese` 词表中不含英文缩写（如 NBA → `[UNK]`），对中英混合文本的 NER 识别有影响
- 模型在 THUCNews 新闻标题风格文本上表现最佳，对话式长文本可能准确率下降
- 时尚类别训练样本较少（~1.3 万），是该类别 F1 略低的原因
