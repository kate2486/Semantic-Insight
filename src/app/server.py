"""
FastAPI 服务器 — 为自定义前端提供模型推理 API

启动: python main.py serve
端口: 8000 (默认)
"""

import os
import sys
from contextlib import asynccontextmanager

import torch
import yaml
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# 路径解析
SERVER_DIR = os.path.dirname(os.path.abspath(__file__))  # src/app/
PROJECT_ROOT = os.path.dirname(os.path.dirname(SERVER_DIR))  # 项目根目录
STATIC_DIR = os.path.join(SERVER_DIR, "static")

sys.path.insert(0, PROJECT_ROOT)

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.models.ner_tagger import NERTagger


# ── 双 Encoder 推理模型 ─────────────────────────────────────────

class DualEncoderModel:
    """分类和 NER 各使用独立的 encoder，避免权重冲突"""

    def __init__(
        self,
        cls_encoder: SharedEncoder,
        classifier_head: TextClassifier,
        ner_encoder: SharedEncoder,
        ner_head: NERTagger,
        label_names: list,
        id2tag: list,
        max_length: int = 256,
    ):
        self.cls_encoder = cls_encoder
        self.classifier_head = classifier_head
        self.ner_encoder = ner_encoder
        self.ner_head = ner_head
        self.label_names = label_names
        self.id2tag = id2tag
        self.max_length = max_length

        from transformers import BertTokenizer
        self.tokenizer = BertTokenizer.from_pretrained(
            "bert-base-chinese", local_files_only=True
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.cls_encoder.to(self.device)
        self.ner_encoder.to(self.device)
        if self.classifier_head:
            self.classifier_head.to(self.device)
        if self.ner_head:
            self.ner_head.to(self.device)

        self.cls_encoder.eval()
        self.ner_encoder.eval()
        if self.classifier_head:
            self.classifier_head.eval()
        if self.ner_head:
            self.ner_head.eval()

    def predict(self, text: str) -> dict:
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        result = {"text": text}

        with torch.no_grad():
            # ── 分类（使用分类 encoder）──
            if self.classifier_head is not None:
                cls_encoded = self.cls_encoder(input_ids, attention_mask)
                logits = self.classifier_head.classifier(
                    self.classifier_head.dropout(cls_encoded["pooler_output"])
                )
                probs = torch.softmax(logits, dim=-1)[0]
                pred_idx = torch.argmax(probs).item()
                result["classification"] = {
                    "label": self.label_names[pred_idx],
                    "confidence": round(probs[pred_idx].item(), 4),
                }

            # ── NER（使用 NER encoder）──
            if self.ner_head is not None:
                ner_encoded = self.ner_encoder(input_ids, attention_mask)
                emissions = self.ner_head.linear(
                    self.ner_head.dropout(ner_encoded["last_hidden_state"])
                )
                mask = attention_mask.bool()
                pred_ids = torch.argmax(emissions, dim=-1)[0]
                valid_len = mask[0].sum().item()
                predictions = pred_ids[:valid_len].tolist()

                tokens = self.tokenizer.convert_ids_to_tokens(input_ids[0])
                entities = self._parse_bio(
                    tokens[1:valid_len - 1], predictions[1:valid_len - 1]
                )
                result["entities"] = entities

        return result

    def _parse_bio(self, tokens: list, tag_ids: list) -> list:
        entities = []
        current_tokens = []
        current_type = None

        for i, (token, tag_id) in enumerate(zip(tokens, tag_ids)):
            tag = self.id2tag[tag_id] if 0 <= tag_id < len(self.id2tag) else "O"

            if tag.startswith("B-"):
                if current_tokens:
                    entities.append({
                        "word": "".join(current_tokens).replace("##", ""),
                        "type": current_type,
                        "start": i - len(current_tokens),
                        "end": i,
                    })
                current_type = tag[2:]
                current_tokens = [token]
            elif tag.startswith("I-") and current_type == tag[2:]:
                current_tokens.append(token)
            else:
                if current_tokens:
                    entities.append({
                        "word": "".join(current_tokens).replace("##", ""),
                        "type": current_type,
                        "start": i - len(current_tokens),
                        "end": i,
                    })
                current_tokens = []
                current_type = None

        if current_tokens:
            entities.append({
                "word": "".join(current_tokens).replace("##", ""),
                "type": current_type,
                "start": len(tokens) - len(current_tokens),
                "end": len(tokens),
            })

        return entities


# ── Pydantic 数据模型 ──────────────────────────────────────────

class PredictRequest(BaseModel):
    """推理请求"""
    text: str


class ClassificationResult(BaseModel):
    """分类结果"""
    label: str
    confidence: float


class EntityResult(BaseModel):
    """实体识别结果"""
    word: str
    type: str
    start: int
    end: int


class PredictResponse(BaseModel):
    """推理响应"""
    text: str
    classification: ClassificationResult | None = None
    entities: list[EntityResult] = []


# ── 模型加载 ────────────────────────────────────────────────────

def load_models() -> "DualEncoderModel":
    """加载已训练的模型 — 分类和NER各用独立的encoder"""
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cls_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_classification.pt")
    ner_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_ner.pt")

    # ── 分类模型（独立 encoder）──
    cls_encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )
    classifier_head = TextClassifier(
        cls_encoder, num_classes=config["classifier"]["num_classes"]
    )
    if os.path.exists(cls_path):
        classifier_head.load_state_dict(
            torch.load(cls_path, map_location="cpu")
        )
        print("分类模型已加载")
    else:
        classifier_head = None
        print("⚠ 分类模型未找到")

    # ── NER 模型（独立 encoder）──
    ner_encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )
    ner_head = NERTagger(
        ner_encoder,
        num_tags=config["ner"]["num_tags"],
        use_crf=False,
    )
    if os.path.exists(ner_path):
        ner_head.load_state_dict(
            torch.load(ner_path, map_location="cpu"),
            strict=False,
        )
        print("NER 模型已加载")
    else:
        ner_head = None
        print("⚠ NER 模型未找到")

    # ── 双 encoder 推理模型 ──
    model = DualEncoderModel(
        cls_encoder=cls_encoder,
        classifier_head=classifier_head,
        ner_encoder=ner_encoder,
        ner_head=ner_head,
        label_names=config["classifier"]["label_names"],
        id2tag=config["ner"]["id2tag"],
        max_length=config["encoder"]["max_length"],
    )
    return model


# ── FastAPI 应用 ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时清理"""
    print("正在加载模型...")
    app.state.model = load_models()
    print("模型加载完成，服务器就绪。")
    yield
    print("服务器关闭...")


app = FastAPI(
    title="Semantic Insight API",
    description="基于 BERT 的中文文本分类 + 命名实体识别 API",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS（开发阶段允许所有来源，部署时请修改）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health_check():
    """健康检查"""
    model_loaded = hasattr(app.state, "model") and app.state.model is not None
    return {
        "status": "ok" if model_loaded else "loading",
        "model_loaded": model_loaded,
    }


@app.post("/api/predict", response_model=PredictResponse)
async def predict(request: PredictRequest):
    """文本分析接口 — 返回分类和NER结果"""
    if not request.text.strip():
        raise HTTPException(status_code=422, detail="文本不能为空")

    try:
        result = app.state.model.predict(request.text)
        return PredictResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"推理失败: {str(e)}"
        )


@app.get("/")
async def serve_frontend():
    """提供前端页面"""
    return FileResponse(
        os.path.join(STATIC_DIR, "index.html"),
        media_type="text/html; charset=utf-8",
    )


# 挂载静态文件目录（供后续添加 CSS/JS/图片等资源）
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
