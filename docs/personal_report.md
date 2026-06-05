# YOLOv8n + AdamW 微调个人实验报告

## 1. 实验任务

本报告对应皇甫泊宁在小组项目中的个人部分：优化策略调优组。小组统一使用 SKU-110K 数据集进行超市货架密集商品检测，本人的变量为在 YOLOv8n 预训练权重基础上，将优化器设置为 AdamW，观察微调前后模型在 SKU-110K 测试集和额外真实货架图片上的性能变化。

本实验不改变模型结构，重点验证优化器替换和场景微调是否能提升 YOLOv8n 在密集商品检测任务上的表现。

## 2. 数据集与预处理

### 2.1 数据集

- 数据集：SKU-110K
- 任务类型：目标检测
- 场景：超市货架密集商品检测
- 类别设置：单类别 `object`
- 数据配置文件：`data/sku110k.yaml`

### 2.2 数据划分与规模

本地将 SKU-110K 原始 CSV 标注转换为 Ultralytics YOLO 格式后，得到如下划分：

| 划分 | 图片数 | 标注框数 |
|---|---:|---:|
| Train | 8219 | 1208482 |
| Val | 588 | 90968 |
| Test | 2936 | 431546 |

数据转换脚本为 `scripts/prepare_sku110k.py`。转换过程将原始边界框标注转为 YOLO 所需的归一化格式：`class x_center y_center width height`。

## 3. 模型与训练设置

### 3.1 模型

- 基础模型：YOLOv8n
- 预训练权重：`models/yolov8n.pt`
- 微调后最佳权重：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`

### 3.2 核心训练参数

完整训练实际使用参数记录在 `runs/train/yolov8n_sku110k_adamw_full/args.yaml` 中，关键设置如下：

| 参数 | 值 |
|---|---:|
| Optimizer | AdamW |
| Epochs | 100 |
| 实际停止 epoch | 96 |
| Best epoch | 76 |
| Batch size | 32 |
| Image size | 640 |
| Initial learning rate `lr0` | 0.001 |
| Weight decay | 0.01 |
| Patience | 20 |
| Seed | 42 |
| Device | GPU 6, NVIDIA RTX A6000 |
| 训练耗时 | 约 2.947 小时 |

训练脚本为 `scripts/train_adamw.py`。训练前先进行了小样本 smoke test，确认数据路径、CUDA 环境和训练入口可正常运行后，再启动完整训练。多 GPU DDP 曾在初始化阶段卡住，因此最终采用单 GPU 完成完整训练，以保证实验可完成和结果可复现。

训练输出目录为：

`runs/train/yolov8n_sku110k_adamw_full/`

其中主要结果文件包括：

- 训练曲线：`runs/train/yolov8n_sku110k_adamw_full/results.png`
- 训练日志 CSV：`runs/train/yolov8n_sku110k_adamw_full/results.csv`
- 最佳权重：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`
- 混淆矩阵：`runs/train/yolov8n_sku110k_adamw_full/confusion_matrix.png`
- PR 曲线：`runs/train/yolov8n_sku110k_adamw_full/BoxPR_curve.png`

## 4. 微调前后测试集性能对比

### 4.1 评估方式

使用同一 SKU-110K test split 对微调前和微调后的模型进行评估。

- 微调前模型：`models/yolov8n.pt`
- 微调后模型：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`
- 评估脚本：`scripts/evaluate.py`
- 输入尺寸：640
- Batch size：32

指标文件：

- 微调前：`report/metrics/baseline_yolov8n_test.json`
- 微调后：`report/metrics/adamw_best_test.json`

### 4.2 测试集结果

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| 微调前 `models/yolov8n.pt` | 0.1549 | 0.0030 | 0.0008 | 0.0004 |
| AdamW 微调后 `best.pt` | 0.9069 | 0.8356 | 0.8858 | 0.5471 |
| 提升 | +0.7519 | +0.8326 | +0.8850 | +0.5467 |

### 4.3 结果分析

原始 YOLOv8n 权重来自 COCO 预训练，类别体系与 SKU-110K 的单类密集货架商品检测任务不一致。因此，微调前模型在测试集上的召回率仅为 0.0030，mAP@0.5 仅为 0.0008，说明它几乎不能直接用于 SKU-110K 场景。

经过 AdamW 微调后，模型在测试集上的 Precision 达到 0.9069，Recall 达到 0.8356，mAP@0.5 达到 0.8858，mAP@0.5:0.95 达到 0.5471。结果说明，针对 SKU-110K 进行场景微调后，YOLOv8n 已经能够较好地学习货架商品的密集目标特征，漏检问题显著减少，整体检测性能大幅提升。

训练期间验证集最佳结果为：

| Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|
| 0.9067 | 0.8359 | 0.8840 | 0.5428 |

验证集和测试集结果接近，说明本次训练结果较稳定，没有出现明显的测试集性能崩塌。

## 5. 额外五张图片泛化验证

### 5.1 图片来源

额外验证图片均来自 Wikimedia Commons，用于测试模型在非 SKU-110K 来源真实货架场景中的泛化表现。图片来源记录在 `docs/extra_image_sources.md`。

| 图片 | 场景 |
|---|---|
| `01_coffee_shelves_bulgaria.jpg` | 咖啡货架 |
| `02_spar_cereal_shelves.jpg` | 谷物/早餐食品货架 |
| `03_spar_sauce_shelves.jpg` | 调味酱货架 |
| `04_rema_pet_food_shelves.jpg` | 宠物食品货架 |
| `05_mackerel_shelves_norway.jpg` | 罐头商品货架 |

### 5.2 推理设置

- 微调前模型：`models/yolov8n.pt`
- 微调后模型：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`
- Confidence threshold：0.25
- IoU threshold：0.7
- Image size：640
- 推理脚本：`scripts/predict_extra.py`

输出目录：

- 微调前可视化：`report/outputs/extra_images/before/`
- 微调后可视化：`report/outputs/extra_images/after/`
- 对比摘要：`docs/extra_image_comparison.md`

### 5.3 检测数量对比

| 图片 | 微调前检测数 | 微调后检测数 |
|---|---:|---:|
| `01_coffee_shelves_bulgaria.jpg` | 0 | 150 |
| `02_spar_cereal_shelves.jpg` | 2 | 99 |
| `03_spar_sauce_shelves.jpg` | 25 | 179 |
| `04_rema_pet_food_shelves.jpg` | 0 | 161 |
| `05_mackerel_shelves_norway.jpg` | 1 | 47 |

### 5.4 定性分析

微调前，COCO 预训练 YOLOv8n 基本不能将货架上的商品识别为目标对象。少数检测结果主要来自 COCO 类别误匹配，例如 bottles、donuts 或 refrigerator，与 SKU-110K 的单类别 `object` 设置并不一致。

微调后，模型在五张额外真实货架图片上都能输出大量商品检测框，说明模型已经学习到货架商品的通用视觉特征，并能迁移到非 SKU-110K 来源的图片上。对于密集排列、多层货架和重复包装区域，微调后模型明显减少了大面积漏检。

同时，额外图片结果也暴露出一些局限：在极密集或包装高度重复的区域，部分检测框可能重叠；小尺寸商品仍可能漏检；货架结构、价格标签或视觉纹理相似区域可能带来少量过检测。这些问题符合密集商品检测任务的典型难点。

## 6. 实验结论

本次个人实验已经完成微调过程记录、测试集性能评估和额外五张图片验证。主要结论如下：

1. 原始 COCO 预训练 YOLOv8n 不能直接适配 SKU-110K 密集货架商品检测任务，测试集 mAP@0.5 仅为 0.0008。
2. 使用 SKU-110K 对 YOLOv8n 进行 AdamW 微调后，测试集 mAP@0.5 提升到 0.8858，mAP@0.5:0.95 提升到 0.5471，性能提升显著。
3. 微调后的模型在 Wikimedia Commons 收集的五张非同源货架图片上均能检测出大量商品，说明其具备一定真实场景泛化能力。
4. 模型在密集、重复、小目标区域仍存在重叠框、漏检和过检测风险，后续可通过更强模型、更高分辨率训练、针对性数据增强或后处理策略进一步优化。

综上，个人部分的实验目标已经完成：YOLOv8n + AdamW 微调成功运行，测试集量化指标显著提升，额外真实货架图片对比结果已保存并完成分析。

## 7. 交付物清单

| 交付物 | 路径 |
|---|---|
| AdamW 训练脚本 | `scripts/train_adamw.py` |
| 测试集评估脚本 | `scripts/evaluate.py` |
| 额外图片推理脚本 | `scripts/predict_extra.py` |
| AdamW 配置 | `configs/adamw.yaml` |
| 完整训练输出 | `runs/train/yolov8n_sku110k_adamw_full/` |
| 最佳权重 | `runs/train/yolov8n_sku110k_adamw_full/weights/best.pt` |
| 微调前测试指标 | `report/metrics/baseline_yolov8n_test.json` |
| 微调后测试指标 | `report/metrics/adamw_best_test.json` |
| 指标摘要 | `docs/adamw_comparison.md` |
| 额外图片来源 | `docs/extra_image_sources.md` |
| 额外图片对比输出 | `report/outputs/extra_images/` |
| 额外图片对比摘要 | `docs/extra_image_comparison.md` |
| 个人实验报告 | `docs/personal_report.md` |
