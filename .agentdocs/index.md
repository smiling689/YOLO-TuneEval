# Agent 文档索引

## 技术治理

`../CLAUDE.md` - 项目级验证要求与大体积实验产物管理约束，修改代码或配置时必读。

## 当前任务文档

暂无。

## 已完成任务文档

`workflow/done/260511-prepare-data-config.md` - 准备 SKU-110K 数据转换与 YOLOv8n AdamW 实验配置。

## 全局重要记忆

- 本仓库服务于皇甫泊宁个人实验部分，核心变量是 YOLOv8n 将优化器设置为 AdamW。
- 数据集使用 SKU-110K，任务为超市货架密集商品检测；不得把原始数据和训练输出大文件纳入版本管理。
- YOLOv8n 预训练权重已放在 `models/yolov8n.pt`，AdamW 配置默认引用该本地路径。
- SKU-110K 已解压到 `data/raw/SKU110K_fixed`，YOLO 格式数据已生成到 `data/processed/sku110k`，图片采用符号链接组织。
