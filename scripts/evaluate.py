#!/usr/bin/env python3
"""Evaluate a YOLO detection model and save a compact metrics summary."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a YOLO detection model on a dataset split.")
    parser.add_argument("--model", required=True, help="Model weights path, for example models/yolov8n.pt.")
    parser.add_argument("--data", default="data/sku110k.yaml", help="Dataset yaml path.")
    parser.add_argument("--split", default="test", choices=("train", "val", "test"), help="Dataset split.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=16, help="Validation batch size.")
    parser.add_argument("--device", default=None, help="Device string, for example 0 or 0,1,2,3.")
    parser.add_argument("--project", default="runs/eval", help="Ultralytics output project directory.")
    parser.add_argument("--name", default=None, help="Ultralytics run name.")
    parser.add_argument("--summary-dir", default="report/metrics", help="Directory for JSON metric summaries.")
    parser.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--save-json", action="store_true", help="Ask Ultralytics to save COCO-style JSON when supported.")
    parser.add_argument("--plots", action="store_true", help="Save validation plots.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting an existing Ultralytics run directory.")
    return parser.parse_args()


def require_path(path: str | Path, description: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def box_metrics(metrics: Any) -> dict[str, Any]:
    box = getattr(metrics, "box", None)
    if box is None:
        return {}
    return {
        "precision": float(getattr(box, "mp", 0.0)),
        "recall": float(getattr(box, "mr", 0.0)),
        "map50": float(getattr(box, "map50", 0.0)),
        "map50_95": float(getattr(box, "map", 0.0)),
        "maps": [float(value) for value in getattr(box, "maps", [])],
    }


def main() -> None:
    args = parse_args()
    model_path = require_path(args.model, "Model")
    data_path = require_path(args.data, "Dataset yaml")

    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Missing Ultralytics. Install dependencies with: pip install -r requirements.txt") from error

    run_name = args.name or f"{model_path.stem}_{Path(args.data).stem}_{args.split}"
    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        save_json=args.save_json,
        plots=args.plots,
    )

    summary = {
        "model": str(model_path),
        "data": str(data_path),
        "split": args.split,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "save_dir": str(getattr(metrics, "save_dir", "")),
        "speed": getattr(metrics, "speed", {}),
        "box": box_metrics(metrics),
    }

    summary_dir = Path(args.summary_dir)
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"{run_name}.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved metrics summary: {summary_path}")


if __name__ == "__main__":
    main()
