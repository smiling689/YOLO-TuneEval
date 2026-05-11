# 数据目录说明

本目录用于放置 SKU-110K 数据配置与本地处理结果。原始数据和大体积处理产物不应提交到仓库。

## 推荐目录

```text
data/
├── raw/
│   └── SKU110K_fixed/
│       ├── images/
│       ├── annotations_train.csv
│       ├── annotations_val.csv
│       └── annotations_test.csv
├── processed/
│   └── sku110k/
│       ├── images/
│       │   ├── train/
│       │   ├── val/
│       │   └── test/
│       └── labels/
│           ├── train/
│           ├── val/
│           └── test/
└── sku110k.yaml
```

## 数据准备命令

SKU-110K 官方压缩包约 11.36 GiB，超过 10GB，不由代理自动下载。需要手动下载时，在仓库根目录执行：

```bash
mkdir -p data/raw
curl -L "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz" -o data/raw/SKU110K_fixed.tar.gz
tar -xzf data/raw/SKU110K_fixed.tar.gz -C data/raw
```

如果解压后的目录名不是 `data/raw/SKU110K`，请在下面转换命令中把 `--raw-dir` 改成实际目录。

```bash
python scripts/prepare_sku110k.py --raw-dir data/raw/SKU110K_fixed --train-csv annotations/annotations_train.csv --val-csv annotations/annotations_val.csv --test-csv annotations/annotations_test.csv --output-dir data/processed/sku110k --yaml-path data/sku110k.yaml
```

默认会把原始图片以符号链接方式组织到 `data/processed/sku110k/images/`，避免重复复制大体积图片。若当前文件系统不支持符号链接，可使用 `--link-mode copy`。

如果转换过程中断，可用 `--splits` 只补跑指定划分，例如：

```bash
python scripts/prepare_sku110k.py --raw-dir data/raw/SKU110K_fixed --test-csv annotations/annotations_test.csv --output-dir data/processed/sku110k --yaml-path data/sku110k.yaml --splits test --overwrite-links
```

## 注意事项

- SKU-110K 原始标注通常是 CSV，每行包含图片名、左上角坐标、右下角坐标、类别、图片宽高。
- 本实验按密集商品检测处理，默认使用单类别 `object`。
- 如果标注 CSV 使用不同文件名，可通过 `--train-csv`、`--val-csv`、`--test-csv` 显式指定。
