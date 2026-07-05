"""
统一入口脚本
用法:
  python main.py train --task cls         # 训练分类模型
  python main.py train --task ner         # 训练NER模型
  python main.py train --task all         # 训练两个模型
  python main.py predict --text "..."     # 单条推理
  python main.py demo                     # 启动Gradio界面
"""
import os
import sys
import yaml
import argparse
import torch


def load_config():
    config_path = os.path.join(os.path.dirname(__file__), "configs", "config.yaml")
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def cmd_train(args):
    """训练命令"""
    config = load_config()
    if args.task in ("cls", "all"):
        print("\n" + "=" * 60)
        print("训练文本分类模型")
        print("=" * 60)
        from src.train.train_classifier import main as train_cls
        train_cls()

    if args.task in ("ner", "all"):
        print("\n" + "=" * 60)
        print("训练NER模型")
        print("=" * 60)
        from src.train.train_ner import main as train_ner
        train_ner()


def cmd_predict(args):
    """预测命令"""
    config = load_config()

    from src.models.encoder import SharedEncoder
    from src.models.classifier import TextClassifier
    from src.models.ner_tagger import NERTagger
    from src.models.multitask_model import MultiTaskModel

    encoder = SharedEncoder(
        model_name=config["encoder"]["model_name"],
        freeze_embeddings=False,
        freeze_layers=0,
    )

    cls_path = os.path.join("checkpoints", "best_classification.pt")
    ner_path = os.path.join("checkpoints", "best_ner.pt")

    classifier_head = TextClassifier(encoder, num_classes=config["classifier"]["num_classes"])
    if os.path.exists(cls_path):
        classifier_head.load_state_dict(torch.load(cls_path, map_location="cpu"))
        print(f"已加载分类模型: {cls_path}")
    else:
        classifier_head = None
        print("分类模型未找到，仅使用NER模型")

    ner_head = NERTagger(encoder, num_tags=config["ner"]["num_tags"], use_crf=False)
    if os.path.exists(ner_path):
        ner_head.load_state_dict(torch.load(ner_path, map_location="cpu"), strict=False)
        print(f"已加载NER模型: {ner_path}")
    else:
        ner_head = None
        print("NER模型未找到，仅使用分类模型")

    if not classifier_head and not ner_head:
        print("错误: 未找到任何已训练的模型。请先运行 python main.py train")
        return

    model = MultiTaskModel(
        encoder=encoder,
        classifier_head=classifier_head,
        ner_head=ner_head,
        label_names=config["classifier"]["label_names"],
        id2tag=config["ner"]["id2tag"],
        max_length=config["encoder"]["max_length"],
    )

    result = model.predict(args.text)
    print("\n" + "=" * 50)
    print(f"输入: {result['text']}")
    print("=" * 50)

    if "classification" in result:
        cls_result = result["classification"]
        print(f"\n分类结果: {cls_result['label']} (置信度: {cls_result['confidence']:.2%})")

    if "entities" in result:
        entities = result["entities"]
        print(f"\n识别实体 ({len(entities)}个):")
        if entities:
            for ent in entities:
                print(f"  [{ent['type']}] {ent['word']} (位置 {ent['start']}-{ent['end']})")
        else:
            print("  未识别到实体")


def cmd_demo(args):
    """启动Gradio演示"""
    from src.app.demo import main as run_demo
    run_demo()


def main():
    parser = argparse.ArgumentParser(description="NLP多任务学习系统 — Semantic-Insight")
    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # train
    train_parser = subparsers.add_parser("train", help="训练模型")
    train_parser.add_argument(
        "--task", choices=["cls", "ner", "all"], default="all",
        help="训练任务: cls(分类), ner(NER), all(全部)"
    )

    # predict
    pred_parser = subparsers.add_parser("predict", help="单条推理")
    pred_parser.add_argument("--text", type=str, required=True, help="输入文本")

    # demo
    subparsers.add_parser("demo", help="启动Gradio Web演示")

    args = parser.parse_args()

    if args.command == "train":
        cmd_train(args)
    elif args.command == "predict":
        cmd_predict(args)
    elif args.command == "demo":
        cmd_demo(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
