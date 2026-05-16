# YOLO-TuneEval

YOLOv8n + AdamW fine-tuning and evaluation on SKU-110K for dense supermarket shelf product detection.

## Contents

- Dataset: SKU-110K converted to Ultralytics YOLO format.
- Baseline model: COCO-pretrained `models/yolov8n.pt`.
- Fine-tuned model: `runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`.
- Training configuration: `configs/adamw.yaml`.
- Scripts: `scripts/prepare_sku110k.py`, `scripts/train_adamw.py`, `scripts/evaluate.py`, `scripts/predict_extra.py`.
- Documentation and reports: `docs/`.

Large dataset assets, model weights, training outputs, and qualitative results are tracked with Git LFS.

## Setup

```bash
git lfs pull
python -m pip install -r requirements.txt
```

The experiment was run with CUDA on an NVIDIA RTX A6000. CPU execution is not recommended for full training.

## Data

The YOLO dataset config is `data/sku110k.yaml`.

Prepared data is stored under:

```text
data/processed/sku110k/
```

See `docs/data.md` for the data layout and regeneration command.

## Train

```bash
python scripts/train_adamw.py --config configs/adamw.yaml
```

The completed full run is saved at:

```text
runs/train/yolov8n_sku110k_adamw_full/
```

## Evaluate

Baseline YOLOv8n:

```bash
python scripts/evaluate.py --weights models/yolov8n.pt --data data/sku110k.yaml --split test --output report/metrics/baseline_yolov8n_test.json
```

AdamW fine-tuned model:

```bash
python scripts/evaluate.py --weights runs/train/yolov8n_sku110k_adamw_full/weights/best.pt --data data/sku110k.yaml --split test --output report/metrics/adamw_best_test.json
```

## Extra Images

```bash
python scripts/predict_extra.py
```

Inputs are in `extra_images/`. Outputs are in `report/outputs/extra_images/`.

## Results

| Model | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| Baseline `models/yolov8n.pt` | 0.1549 | 0.0030 | 0.0008 | 0.0004 |
| AdamW fine-tuned `best.pt` | 0.9069 | 0.8356 | 0.8858 | 0.5471 |

The fine-tuned model also detects dense products on five external shelf images, while the original COCO-pretrained model mostly fails on the SKU-110K single-class setting.

## Documentation

- `docs/personal_report.md`: personal experiment report.
- `docs/adamw_comparison.md`: metric summary.
- `docs/extra_image_comparison.md`: qualitative comparison on five external images.
- `docs/extra_image_sources.md`: external image sources.
- `docs/data.md`: data layout and preparation notes.
- `docs/team_assignment.md`: group division and experiment design.
- `docs/assignment.md`: original assignment requirements.
