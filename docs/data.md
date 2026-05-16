# Data Notes

This project uses SKU-110K for dense supermarket shelf product detection. Large dataset files and generated artifacts are tracked with Git LFS, so run `git lfs pull` after cloning.

## Layout

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

## Regeneration Command

If the raw dataset needs to be regenerated from the official archive, download and extract it under `data/raw/`:

```bash
mkdir -p data/raw
curl -L "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz" -o data/raw/SKU110K_fixed.tar.gz
tar -xzf data/raw/SKU110K_fixed.tar.gz -C data/raw
```

Then convert the CSV annotations to YOLO format:

```bash
python scripts/prepare_sku110k.py --raw-dir data/raw/SKU110K_fixed --train-csv annotations/annotations_train.csv --val-csv annotations/annotations_val.csv --test-csv annotations/annotations_test.csv --output-dir data/processed/sku110k --yaml-path data/sku110k.yaml
```

The conversion script organizes images in `data/processed/sku110k/images/` with symbolic links by default to avoid duplicating the raw images. Use `--link-mode copy` if symlinks are not supported.

If conversion is interrupted, use `--splits` to rebuild only the needed split:

```bash
python scripts/prepare_sku110k.py --raw-dir data/raw/SKU110K_fixed --test-csv annotations/annotations_test.csv --output-dir data/processed/sku110k --yaml-path data/sku110k.yaml --splits test --overwrite-links
```

## Notes

- SKU-110K annotations are CSV files with image name, bounding box coordinates, class, image width, and image height.
- This experiment treats the task as single-class dense product detection with class name `object`.
- If the annotation files use different names, pass explicit `--train-csv`, `--val-csv`, and `--test-csv` values.
