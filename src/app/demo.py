"""
Gradio Web演示界面
输入文本 → 同时展示分类结果和NER实体
"""
import os
import sys
import yaml
import torch
import gradio as gr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.models.encoder import SharedEncoder
from src.models.classifier import TextClassifier
from src.models.ner_tagger import NERTagger
from src.models.multitask_model import MultiTaskModel


def load_models():
    """加载已训练的模型"""
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    config_path = os.path.join(base_dir, "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )

    cls_path = os.path.join(base_dir, "checkpoints", "best_classification.pt")
    ner_path = os.path.join(base_dir, "checkpoints", "best_ner.pt")

    classifier_head = TextClassifier(encoder, num_classes=config["classifier"]["num_classes"])
    if os.path.exists(cls_path):
        classifier_head.load_state_dict(torch.load(cls_path, map_location="cpu"))
        print("分类模型已加载")
    else:
        classifier_head = None
        print("分类模型未找到")

    ner_head = NERTagger(encoder, num_tags=config["ner"]["num_tags"])
    if os.path.exists(ner_path):
        ner_head.load_state_dict(torch.load(ner_path, map_location="cpu"))
        print("NER模型已加载")
    else:
        ner_head = None
        print("NER模型未找到")

    model = MultiTaskModel(
        encoder=encoder,
        classifier_head=classifier_head,
        ner_head=ner_head,
        label_names=config["classifier"]["label_names"],
        id2tag=config["ner"]["id2tag"],
        max_length=config["encoder"]["max_length"],
    )
    return model


# 全局模型实例（启动时加载一次）
_model = None


def get_model():
    global _model
    if _model is None:
        _model = load_models()
    return _model


def analyze_text(text: str):
    """分析文本，返回分类和NER结果"""
    if not text.strip():
        return "### 请输入文本", "### 请输入文本"

    model = get_model()
    result = model.predict(text)

    # 分类结果
    if "classification" in result:
        cls_info = (
            f"### 文本分类\n\n"
            f"**{result['classification']['label']}** "
            f"(置信度: {result['classification']['confidence']:.2%})"
        )
    else:
        cls_info = "### 分类模型未加载"

    # NER结果
    if "entities" in result:
        entities = result["entities"]
        if entities:
            ner_info = "### 命名实体识别\n\n"
            ner_info += "| 实体 | 类型 |\n|------|------|\n"
            for ent in entities:
                ner_info += f"| {ent['word']} | {ent['type']} |\n"
        else:
            ner_info = "### 命名实体识别\n\n未识别到实体"
    else:
        ner_info = "### NER模型未加载"

    return cls_info, ner_info


def create_demo():
    with gr.Blocks(title="NLP多任务学习系统", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            """
            # Semantic-Insight
            ### 基于 BERT 的文本分类 + 命名实体识别 联合推理
            """
        )

        with gr.Row():
            input_text = gr.Textbox(
                label="输入文本",
                placeholder="请输入中文文本进行分析...",
                lines=3,
            )

        with gr.Row():
            submit_btn = gr.Button("分析", variant="primary")
            clear_btn = gr.Button("清除")

        with gr.Row():
            with gr.Column():
                cls_output = gr.Markdown(label="分类结果")

        with gr.Row():
            ner_output = gr.Markdown(label="NER结果")

        # 示例
        gr.Examples(
            examples=[
                ["华为技术有限公司在深圳举办了2024年开发者大会"],
                ["今天NBA总决赛，洛杉矶湖人队大胜波士顿凯尔特人"],
                ["央行宣布降息50个基点，A股市场大幅反弹"],
                ["北京大学和清华大学联合举办人工智能论坛"],
            ],
            inputs=input_text,
        )

        submit_btn.click(
            fn=analyze_text,
            inputs=input_text,
            outputs=[cls_output, ner_output],
        )
        clear_btn.click(
            fn=lambda: ("", "", ""),
            outputs=[input_text, cls_output, ner_output],
        )

    return demo


def main():
    global _model
    print("正在加载模型...")
    _model = load_models()
    print("启动 Gradio 演示界面...")
    demo = create_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860, share=False)


if __name__ == "__main__":
    main()
