#!/usr/bin/env python3
"""使用配置文件启动 YOLOv8n + AdamW 微调实验。"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="启动 YOLOv8n + AdamW 微调实验。")
    parser.add_argument("--config", type=Path, default=Path("configs/adamw.yaml"), help="实验配置文件。")
    parser.add_argument("--model", default=None, help="覆盖配置中的预训练权重路径。")
    parser.add_argument("--data", default=None, help="覆盖配置中的数据集 yaml 路径。")
    parser.add_argument("--epochs", type=int, default=None, help="覆盖训练轮数。")
    parser.add_argument("--imgsz", type=int, default=None, help="覆盖输入尺寸。")
    parser.add_argument("--batch", type=int, default=None, help="覆盖 batch size。")
    parser.add_argument("--device", default=None, help="训练设备，例如 0、0,1 或 cpu。")
    parser.add_argument("--project", default=None, help="覆盖输出目录。")
    parser.add_argument("--name", default=None, help="覆盖实验名称。")
    parser.add_argument("--fraction", type=float, default=None, help="覆盖训练数据使用比例，用于小样本 smoke test。")
    parser.add_argument("--workers", type=int, default=None, help="覆盖 dataloader workers 数。")
    return parser.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as error:
        raise RuntimeError("缺少 PyYAML，请先安装依赖：pip install pyyaml ultralytics") from error

    if not path.exists():
        raise FileNotFoundError(f"配置文件不存在：{path}")
    with path.open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    if not isinstance(config, dict):
        raise ValueError(f"配置文件必须是 YAML 字典：{path}")
    return config


def apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    overrides = {
        "model": args.model,
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "project": args.project,
        "name": args.name,
        "fraction": args.fraction,
        "workers": args.workers,
    }
    for key, value in overrides.items():
        if value is not None:
            config[key] = value
    return config


def validate_config(config: dict[str, Any]) -> None:
    required_keys = ("model", "data", "optimizer")
    missing = [key for key in required_keys if key not in config]
    if missing:
        raise ValueError(f"配置缺少必要字段：{', '.join(missing)}")
    if str(config["optimizer"]).lower() != "adamw":
        raise ValueError("本脚本用于 AdamW 实验，配置中的 optimizer 必须为 AdamW。")
    data_path = Path(str(config["data"]))
    if not data_path.exists():
        raise FileNotFoundError(f"数据配置不存在：{data_path}。请先运行 scripts/prepare_sku110k.py。")


def epoch_summary(trainer: Any) -> None:
    epoch = int(getattr(trainer, "epoch", -1)) + 1
    total_epochs = int(getattr(trainer, "epochs", 0))
    metrics = getattr(trainer, "metrics", {}) or {}
    metric_parts = []
    for key in ("metrics/precision(B)", "metrics/recall(B)", "metrics/mAP50(B)", "metrics/mAP50-95(B)"):
        value = metrics.get(key)
        if value is not None:
            metric_parts.append(f"{key.split('/')[-1]}={float(value):.6f}")
    best_fitness = getattr(trainer, "best_fitness", None)
    if best_fitness is not None:
        metric_parts.append(f"best_fitness={float(best_fitness):.6f}")
    message = f"[epoch-summary] epoch={epoch}/{total_epochs}"
    if metric_parts:
        message = f"{message} " + " ".join(metric_parts)
    print(message, flush=True)


def main() -> None:
    args = parse_args()
    config = apply_overrides(load_config(args.config), args)
    validate_config(config)

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("缺少 Ultralytics，请先安装依赖：pip install ultralytics") from error

    model_path = config.pop("model")
    model = YOLO(model_path)
    model.add_callback("on_fit_epoch_end", epoch_summary)
    model.train(**config)


if __name__ == "__main__":
    main()
