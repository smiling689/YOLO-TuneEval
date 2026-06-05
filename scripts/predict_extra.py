#!/usr/bin/env python3
"""Run before/after YOLO predictions on extra shelf images."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict extra images with pre- and post-finetuning models.")
    parser.add_argument("--source", default="extra_images", help="Image file or directory for extra validation images.")
    parser.add_argument("--before-model", default="models/yolov8n.pt", help="Pre-finetuning model path.")
    parser.add_argument("--after-model", default=None, help="Post-finetuning model path, usually weights/best.pt.")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size.")
    parser.add_argument("--batch", type=int, default=1, help="Prediction batch size.")
    parser.add_argument("--device", default=None, help="Device string, for example 0.")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold.")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold.")
    parser.add_argument("--max-det", type=int, default=300, help="Maximum detections per image.")
    parser.add_argument("--project", default="report/outputs/extra_images", help="Prediction output project directory.")
    parser.add_argument("--save-txt", action="store_true", help="Save YOLO-format prediction txt files.")
    parser.add_argument("--save-conf", action="store_true", help="Include confidence values in saved txt files.")
    parser.add_argument("--exist-ok", action="store_true", help="Allow overwriting existing prediction directories.")
    return parser.parse_args()


def require_path(path: str | Path, description: str) -> Path:
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"{description} does not exist: {resolved}")
    return resolved


def run_predict(model_path: Path, source: Path, run_name: str, args: argparse.Namespace) -> dict[str, str]:
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    results = model.predict(
        source=str(source),
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        conf=args.conf,
        iou=args.iou,
        max_det=args.max_det,
        project=args.project,
        name=run_name,
        exist_ok=args.exist_ok,
        save=True,
        save_txt=args.save_txt,
        save_conf=args.save_conf,
    )
    save_dir = getattr(results[0], "save_dir", "") if results else ""
    return {"model": str(model_path), "source": str(source), "save_dir": str(save_dir)}


def main() -> None:
    args = parse_args()
    source = require_path(args.source, "Source")
    before_model = require_path(args.before_model, "Before model")
    after_model = require_path(args.after_model, "After model") if args.after_model else None

    try:
        import ultralytics  # noqa: F401
    except ImportError as error:
        raise RuntimeError("Missing Ultralytics. Install dependencies with: pip install -r requirements.txt") from error

    summaries = [run_predict(before_model, source, "before", args)]
    if after_model is not None:
        summaries.append(run_predict(after_model, source, "after", args))

    project = Path(args.project)
    project.mkdir(parents=True, exist_ok=True)
    summary_path = project / "summary.json"
    summary_path.write_text(json.dumps(summaries, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Saved prediction summary: {summary_path}")


if __name__ == "__main__":
    main()
