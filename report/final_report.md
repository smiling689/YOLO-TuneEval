# YOLOv8 模型微调与性能验证总报告

## 1. 实验背景与任务目标

本项目基于 Ultralytics YOLO 框架完成目标检测模型的微调与性能验证。根据作业要求，实验需要选择非 COCO 的检测数据集，对 YOLO 模型进行微调，记录训练设置，并比较模型在微调前后测试集和额外真实图片上的表现。

本组选择 SKU-110K 数据集作为实验数据。该数据集面向超市货架密集商品检测，图像中商品数量多、目标尺度小、遮挡和重复包装较多，适合验证 YOLO 模型在密集目标检测场景中的适应能力。

小组实验采用四组变量设计：在统一数据集和 YOLOv8 检测框架下，分别比较默认 YOLOv8n、模型容量提升、关闭 Mosaic 数据增强、以及 AdamW 优化器替换对检测性能的影响。本人负责优化策略调优组，重点验证 `YOLOv8n + AdamW` 在 SKU-110K 上的微调效果，并完成微调前后测试集评估和额外五张真实货架图片的泛化验证。

## 2. 数据集与预处理

### 2.1 数据集说明

- 数据集：SKU-110K
- 任务类型：目标检测
- 检测场景：超市货架密集商品检测
- 类别设置：单类别 `object`
- 数据配置文件：`data/sku110k.yaml`

SKU-110K 原始标注为 CSV 格式，其中包含图片名称、边界框坐标、类别和图像尺寸信息。实验中使用 `scripts/prepare_sku110k.py` 将原始标注转换为 Ultralytics YOLO 格式，即每个目标以 `class x_center y_center width height` 的归一化形式存储在 `.txt` 标签文件中。

### 2.2 数据划分

本地转换后的数据规模如下：

| 划分 | 图片数 | 标注框数 |
|---|---:|---:|
| Train | 8219 | 1208482 |
| Val | 588 | 90968 |
| Test | 2936 | 431546 |

数据目录结构遵循 Ultralytics 要求，训练、验证和测试图片分别存放在 `data/processed/sku110k/images/train`、`data/processed/sku110k/images/val` 和 `data/processed/sku110k/images/test` 下，对应标签存放在 `labels` 目录中。

## 3. 小组实验设计

小组成员围绕同一数据集设计了四组实验变量，以便从不同角度分析 YOLOv8 在密集货架商品检测任务上的性能变化。

| 成员 | 分组 | 实验目的 | 预训练权重 | 变量设置 |
|---|---|---|---|---|
| 陈奕莱 | Baseline 调优组 | 建立默认 YOLOv8n 微调基线 | `yolov8n.pt` | 默认超参数 |
| 卓识 | 模型容量对比组 | 比较更大模型容量的效果 | `yolov8s.pt` | 默认超参数 |
| 刘易函 | 数据增强消融组 | 分析 Mosaic 数据增强影响 | `yolov8n.pt` | 关闭 Mosaic |
| 皇甫泊宁 | 优化策略调优组 | 分析优化器替换影响 | `yolov8n.pt` | AdamW 优化器 |

该设计能够分别回答四个问题：默认 YOLOv8n 在 SKU-110K 上能达到怎样的基线水平；更大的 YOLOv8s 是否能带来更高检测精度；关闭 Mosaic 是否会影响密集商品检测；AdamW 优化器是否能改善 YOLOv8n 的训练结果。

## 4. 本人 AdamW 实验设置

### 4.1 模型与训练参数

本人实验使用 COCO 预训练的 YOLOv8n 作为初始权重，并将优化器设置为 AdamW。训练不改变模型结构，重点观察优化策略替换和 SKU-110K 场景微调带来的性能变化。

| 参数 | 设置 |
|---|---:|
| 模型 | YOLOv8n |
| 初始权重 | `models/yolov8n.pt` |
| 数据配置 | `data/sku110k.yaml` |
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
| Device | NVIDIA RTX A6000，GPU 6 |
| 训练耗时 | 约 2.947 小时 |

完整训练参数保存在 `runs/train/yolov8n_sku110k_adamw_full/args.yaml`，训练结果保存在 `runs/train/yolov8n_sku110k_adamw_full/`。主要输出包括 `results.csv`、`results.png`、`weights/best.pt`、混淆矩阵和 PR 曲线。

### 4.2 训练过程观察

训练开始后，模型在前几个 epoch 中快速适应 SKU-110K 的单类别密集检测任务。验证集 mAP@0.5 从第 1 个 epoch 的 0.6546 提升到第 10 个 epoch 的 0.8398，说明预训练特征能够较快迁移到货架商品检测场景。

中后期训练进入缓慢提升阶段，mAP@0.5:0.95 从约 0.50 逐步提升到 0.54 左右。最佳验证集结果出现在 epoch 76，之后指标基本稳定并略有回落，最终由于 early stopping 在 epoch 96 停止。该过程说明模型已经基本收敛，继续训练收益有限。

本人 AdamW 训练期间最佳验证集结果如下：

| Epoch | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---:|---:|---:|---:|---:|
| 76 | 0.9067 | 0.8357 | 0.8840 | 0.5427 |

## 5. 微调前后测试集性能比较

### 5.1 评估设置

为满足作业对“微调前后性能比较”的要求，本人使用同一 SKU-110K test split 对微调前后的模型进行评估。

- 微调前模型：`models/yolov8n.pt`
- 微调后模型：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`
- 评估脚本：`scripts/evaluate.py`
- 输入尺寸：640
- Batch size：32
- 指标文件：`report/metrics/baseline_yolov8n_test.json`、`report/metrics/adamw_best_test.json`

### 5.2 测试集结果

| 模型 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---:|---:|---:|---:|
| 微调前 `models/yolov8n.pt` | 0.1549 | 0.0030 | 0.0008 | 0.0004 |
| AdamW 微调后 `best.pt` | 0.9069 | 0.8356 | 0.8858 | 0.5471 |
| 提升 | +0.7519 | +0.8326 | +0.8850 | +0.5467 |

### 5.3 结果分析

微调前的 YOLOv8n 权重来自 COCO 数据集，其类别体系和 SKU-110K 的单类别商品检测任务不一致。因此，原始模型在 SKU-110K 测试集上的召回率仅为 0.0030，mAP@0.5 仅为 0.0008，几乎不能直接检测货架商品。

经过 SKU-110K 微调后，AdamW 模型在测试集上达到 0.9069 的 Precision、0.8356 的 Recall、0.8858 的 mAP@0.5 和 0.5471 的 mAP@0.5:0.95。该结果说明微调显著改善了模型对密集商品目标的识别能力，尤其是大幅降低了漏检问题。

验证集最佳结果和测试集结果非常接近，说明训练结果较稳定，没有出现明显过拟合或测试集性能崩塌。

## 6. 四组实验结果汇总与横向分析

### 6.1 指标汇总

以下结果来自本人训练输出和 `others/` 目录中三位成员上传的结果。需要注意的是，不同成员上传材料格式不完全一致，其中陈奕莱结果为测试集截图，卓识结果为验证集截图，刘易函结果为验证集 CSV。因此表格中的 split 已单独标注，横向比较应作为趋势分析，而不是完全严格的同 split 排名。

| 成员 | 实验设置 | Split / 来源 | Precision | Recall | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---:|---:|---:|---:|
| 陈奕莱 | YOLOv8n，默认超参数 | Test，`others/陈奕莱/res.png` | 0.896 | 0.825 | 0.887 | 0.536 |
| 卓识 | YOLOv8s，默认超参数 | Val，`others/卓识/res.png` | 0.911 | 0.850 | 0.894 | 0.559 |
| 刘易函 | YOLOv8n，关闭 Mosaic | Val，`others/刘易函/finetune3/results.csv` | 0.9084 | 0.8370 | 0.8847 | 0.5432 |
| 皇甫泊宁 | YOLOv8n，AdamW | Test，`report/metrics/adamw_best_test.json` | 0.9069 | 0.8356 | 0.8858 | 0.5471 |

刘易函关闭 Mosaic 实验的 CSV 中，mAP@0.5:0.95 最佳值出现在 epoch 90；mAP@0.5 最佳值出现在 epoch 79，为 0.8850。其最后一个 epoch 的 mAP@0.5:0.95 为 0.5419，说明训练后期结果较稳定。

### 6.2 Baseline 与 AdamW 对比

陈奕莱的默认 YOLOv8n 测试集结果为 Precision 0.896、Recall 0.825、mAP@0.5 0.887、mAP@0.5:0.95 0.536。本人 AdamW 测试集结果为 Precision 0.9069、Recall 0.8356、mAP@0.5 0.8858、mAP@0.5:0.95 0.5471。

两组结果说明，在相同 YOLOv8n 模型容量下，默认设置和 AdamW 都能将模型训练到较高水平。AdamW 相比默认基线在 Precision、Recall 和 mAP@0.5:0.95 上略高，但 mAP@0.5 与默认基线非常接近。由于两组训练环境和具体超参数并不完全一致，该差异更适合解释为 AdamW 在定位质量更严格的 mAP@0.5:0.95 上具有一定改善趋势，而不是绝对优势结论。

### 6.3 模型容量影响

卓识使用 YOLOv8s 进行训练，验证集结果达到 Precision 0.911、Recall 0.850、mAP@0.5 0.894、mAP@0.5:0.95 0.559，是四组结果中数值最高的一组。YOLOv8s 的参数量和计算量高于 YOLOv8n，具备更强的特征表达能力，因此在密集、小目标、重复纹理较多的货架商品检测中更容易获得更高的召回率和定位质量。

该结果说明，模型容量提升对 SKU-110K 这类复杂密集检测任务是有效的。不过，YOLOv8s 的推理成本也更高，实际部署时需要在精度和速度之间权衡。

### 6.4 Mosaic 数据增强影响

刘易函关闭 Mosaic 后，验证集最佳 mAP@0.5:0.95 为 0.5432，与本人 AdamW 验证集最佳 mAP@0.5:0.95 0.5427 非常接近，mAP@0.5 也保持在约 0.885 的水平。这说明在 SKU-110K 上，即使关闭 Mosaic，YOLOv8n 仍能学习到较强的密集货架商品检测能力。

Mosaic 通常可以增加场景组合和尺度变化，对通用检测任务有帮助。但 SKU-110K 本身已经具有高度密集、重复、多目标的图像特征，关闭 Mosaic 后并没有造成明显性能崩塌。这可能说明该数据集自身的场景复杂度已经足够高，或者默认的其他增强策略仍然提供了足够的泛化约束。

### 6.5 综合比较

综合四组结果可以得到以下趋势：

1. 微调是最关键因素。原始 COCO 预训练 YOLOv8n 几乎无法直接用于 SKU-110K，而经过 SKU-110K 微调后 mAP@0.5 可以稳定达到约 0.885 以上。
2. 模型容量提升带来的收益最明显。YOLOv8s 在验证集上取得最高 mAP@0.5 和 mAP@0.5:0.95，说明更强模型对密集商品检测有帮助。
3. AdamW 相比默认 YOLOv8n 基线在 mAP@0.5:0.95 上略有提升，说明优化器替换可能改善更严格 IoU 阈值下的定位质量。
4. 关闭 Mosaic 后性能仍保持稳定，说明 SKU-110K 对 YOLOv8n 的训练信号较充分，Mosaic 并不是该任务中唯一决定性因素。

## 7. 额外五张真实货架图片验证

### 7.1 图片来源与推理设置

根据作业要求，本人额外选取五张非 SKU-110K 来源的真实货架图片，用于验证模型在真实零售场景中的泛化能力。图片均来自 Wikimedia Commons，来源记录在 `docs/extra_image_sources.md`。

| 图片 | 场景 |
|---|---|
| `01_coffee_shelves_bulgaria.jpg` | 咖啡货架 |
| `02_spar_cereal_shelves.jpg` | 谷物/早餐食品货架 |
| `03_spar_sauce_shelves.jpg` | 调味酱货架 |
| `04_rema_pet_food_shelves.jpg` | 宠物食品货架 |
| `05_mackerel_shelves_norway.jpg` | 罐头商品货架 |

推理设置如下：

- 微调前模型：`models/yolov8n.pt`
- 微调后模型：`runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`
- Confidence threshold：0.25
- IoU threshold：0.7
- Image size：640
- 推理脚本：`scripts/predict_extra.py`
- 输出目录：`report/outputs/extra_images/`

### 7.2 检测数量对比

| 图片 | 微调前检测数 | 微调后检测数 |
|---|---:|---:|
| `01_coffee_shelves_bulgaria.jpg` | 0 | 150 |
| `02_spar_cereal_shelves.jpg` | 2 | 99 |
| `03_spar_sauce_shelves.jpg` | 25 | 179 |
| `04_rema_pet_food_shelves.jpg` | 0 | 161 |
| `05_mackerel_shelves_norway.jpg` | 1 | 47 |

### 7.3 定性分析

微调前，COCO 预训练 YOLOv8n 基本不能将货架上的商品识别为 SKU-110K 所需的 `object` 类。少数检测结果主要来自 COCO 类别误匹配，例如 bottles、donuts 或 refrigerator，与本任务的单类别商品检测目标不一致。

微调后，模型在五张额外真实货架图片上均能输出大量商品检测框，说明模型不仅在 SKU-110K 测试集上取得高指标，也学习到了一定程度的通用货架商品视觉特征。对于多层货架、重复包装和密集排列区域，微调后模型明显减少了大面积漏检。

同时，额外图片也暴露出一些局限：在极密集区域，检测框之间可能存在重叠；小尺寸商品或远处商品仍可能漏检；货架边缘、价格标签和重复纹理区域可能出现少量过检测。这些问题是密集目标检测任务中的典型难点。

`others/` 目录中三位成员上传的材料主要为训练或评估指标，未包含他们各自额外五张图片的完整可视化结果。因此，本报告中的额外图片分析以本人完整实验结果为主。

## 8. 实验可靠性与局限性

本实验整体完成了作业要求中的模型微调、测试集性能比较和额外图片验证，并通过小组四组变量实验提供了更全面的横向分析。不过仍存在以下限制：

1. 小组成员上传结果格式不统一。陈奕莱和卓识主要提供截图，刘易函提供 CSV 和训练目录，因此部分训练细节无法完全统一核查。
2. 横向比较中同时包含测试集和验证集结果。陈奕莱与本人为测试集结果，卓识与刘易函主要为验证集结果，因此不能作为严格同 split 排名。
3. 陈奕莱测试截图显示有 1 张测试图片被识别为 corrupt image 并被忽略，实际评估为 2935 张图片、431419 个实例，与本人测试集规模 2936 张图片、431546 个实例略有差异。
4. 额外真实图片验证具有定性性质。检测数量能够反映微调前后的明显差异，但不能完全替代人工标注后的定量泛化评估。

## 9. 结论

本项目验证了 YOLOv8 在 SKU-110K 密集货架商品检测任务上的微调效果。实验结果表明，原始 COCO 预训练 YOLOv8n 几乎不能直接适配 SKU-110K，测试集 mAP@0.5 仅为 0.0008；经过 SKU-110K 微调后，YOLOv8n 可以达到约 0.886 的测试集 mAP@0.5，检测性能大幅提升。

本人负责的 `YOLOv8n + AdamW` 实验在测试集上达到 Precision 0.9069、Recall 0.8356、mAP@0.5 0.8858、mAP@0.5:0.95 0.5471，并在五张非同源真实货架图片上表现出明显泛化能力。与默认 YOLOv8n 基线相比，AdamW 在更严格的 mAP@0.5:0.95 指标上略有提升，说明优化策略可能对定位质量有一定帮助。

小组横向结果显示，YOLOv8s 的模型容量提升带来了最明显的性能增益；关闭 Mosaic 后 YOLOv8n 仍能保持较稳定表现；默认 YOLOv8n、关闭 Mosaic 和 AdamW 三组结果整体接近，均说明 SKU-110K 场景微调本身是性能提升的核心因素。

综合来看，本实验达成了作业目标：完成了 YOLOv8 模型在非 COCO 数据集上的微调，记录了关键训练参数，比较了微调前后性能差异，并通过额外真实图片分析了模型的泛化表现。后续若继续改进，可尝试更大模型、更高输入分辨率、针对密集小目标的增强策略，以及基于真实货架额外图片的人工标注定量评估。

## 10. 主要结果文件

| 内容 | 路径 |
|---|---|
| 作业要求 | `docs/assignment.md` |
| 小组分工 | `docs/team_assignment.md` |
| 最终提交报告 | `report/final_report.md` |
| 本人个人报告 | `docs/personal_report.md` |
| AdamW 训练输出 | `runs/train/yolov8n_sku110k_adamw_full/` |
| AdamW 最佳权重 | `runs/train/yolov8n_sku110k_adamw_full/weights/best.pt` |
| 微调前测试指标 | `report/metrics/baseline_yolov8n_test.json` |
| AdamW 测试指标 | `report/metrics/adamw_best_test.json` |
| 额外图片对比摘要 | `docs/extra_image_comparison.md` |
| 陈奕莱结果 | `others/陈奕莱/res.png` |
| 卓识结果 | `others/卓识/res.png` |
| 刘易函结果 | `others/刘易函/finetune3/results.csv` |
