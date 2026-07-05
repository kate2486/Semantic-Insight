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
from src.models.multitask_model import MultiTaskModel


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

def load_models() -> MultiTaskModel:
    """加载已训练的模型（与 demo.py 逻辑一致）"""
    config_path = os.path.join(PROJECT_ROOT, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )

    cls_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_classification.pt")
    ner_path = os.path.join(PROJECT_ROOT, "checkpoints", "best_ner.pt")

    # 分类头
    classifier_head = TextClassifier(
        encoder, num_classes=config["classifier"]["num_classes"]
    )
    if os.path.exists(cls_path):
        classifier_head.load_state_dict(
            torch.load(cls_path, map_location="cpu")
        )
        print("分类模型已加载")
    else:
        classifier_head = None
        print("⚠ 分类模型未找到")

    # NER 头
    ner_head = NERTagger(
        encoder,
        num_tags=config["ner"]["num_tags"],
        use_crf=False,  # 与训练时保持一致
    )
    if os.path.exists(ner_path):
        ner_head.load_state_dict(
            torch.load(ner_path, map_location="cpu"),
            strict=False,  # 检查点包含 ce_loss.weight buffer
        )
        print("NER 模型已加载")
    else:
        ner_head = None
        print("⚠ NER 模型未找到")

    model = MultiTaskModel(
        encoder=encoder,
        classifier_head=classifier_head,
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
