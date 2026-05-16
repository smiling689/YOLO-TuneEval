# YOLOv8n + AdamW 实验指标摘要

## 训练设置

- 数据集：SKU-110K，单类别 `object`
- 模型：YOLOv8n
- 预训练权重：`models/yolov8n.pt`
- 优化器：AdamW
- 输入尺寸：640
- Batch size：32
- 训练设备：GPU 6，NVIDIA RTX A6000
- 计划轮数：100
- 实际轮数：96，因 early stopping 停止
- 最佳 epoch：76
- 训练耗时：约 2.947 小时

## 验证集最佳结果

训练期间最佳模型保存在：

`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`

最佳验证集结果：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.9067 | 0.8359 | 0.8840 | 0.5428 |

## 测试集微调前后对比

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| 微调前 `models/yolov8n.pt` | 0.1549 | 0.0030 | 0.0008 | 0.0004 |
| AdamW 微调后 `best.pt` | 0.9069 | 0.8356 | 0.8858 | 0.5471 |
| 提升 | +0.7519 | +0.8326 | +0.8850 | +0.5467 |

## 结果文件

- 微调前 test 指标：`report/metrics/baseline_yolov8n_test.json`
- AdamW 微调后 test 指标：`report/metrics/adamw_best_test.json`
- 训练结果目录：`runs/train/yolov8n_sku110k_adamw_full`
- 训练曲线：`runs/train/yolov8n_sku110k_adamw_full/results.png`
- 最佳权重：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`

## 简要结论

原始 COCO 预训练 YOLOv8n 几乎不能直接检测 SKU-110K 的密集货架商品，测试集 mAP@0.5 约为 0.0008。使用 SKU-110K 进行 AdamW 微调后，测试集 mAP@0.5 提升到 0.8858，mAP@0.5:0.95 提升到 0.5471，说明针对目标场景的微调显著改善了检测性能。
