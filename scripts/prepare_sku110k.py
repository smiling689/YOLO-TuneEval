#!/usr/bin/env python3
"""将 SKU-110K CSV 标注转换为 Ultralytics YOLO 检测数据集格式。"""

from __future__ import annotations

import argparse
import csv
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_SPLIT_FILES = {
    "train": ("annotations_train.csv",),
    "val": ("annotations_val.csv", "annotations_validation.csv"),
    "test": ("annotations_test.csv",),
}


@dataclass(frozen=True)
class Box:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float


@dataclass
class SplitStats:
    images: int = 0
    boxes: int = 0
    skipped_boxes: int = 0
    clipped_boxes: int = 0
    missing_images: int = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="将 SKU-110K 标注转换为 YOLO 格式，并生成 data.yaml。"
    )
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw/SKU110K"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed/sku110k"))
    parser.add_argument("--yaml-path", type=Path, default=Path("data/sku110k.yaml"))
    parser.add_argument("--train-csv", type=Path, default=None, help="训练集 CSV，默认自动查找。")
    parser.add_argument("--val-csv", type=Path, default=None, help="验证集 CSV，默认自动查找。")
    parser.add_argument("--test-csv", type=Path, default=None, help="测试集 CSV，默认自动查找。")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "val", "test"),
        default=("train", "val", "test"),
        help="要转换的数据划分。用于中断后只补跑指定 split。",
    )
    parser.add_argument("--class-name", default="object", help="单类别模式下的类别名。")
    parser.add_argument(
        "--preserve-classes",
        action="store_true",
        help="保留 CSV 中的类别列并自动建立类别映射；默认把所有框视为单类别 object。",
    )
    parser.add_argument(
        "--link-mode",
        choices=("symlink", "copy", "none"),
        default="symlink",
        help="如何把图片组织到输出目录。none 表示只写标签和 yaml。",
    )
    parser.add_argument(
        "--overwrite-links",
        action="store_true",
        help="当输出图片链接或文件已存在时，覆盖它们。",
    )
    return parser.parse_args()


def resolve_csv(raw_dir: Path, split: str, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        path = explicit_path if explicit_path.is_absolute() else raw_dir / explicit_path
        if not path.exists():
            raise FileNotFoundError(f"找不到 {split} CSV：{path}")
        return path

    for filename in DEFAULT_SPLIT_FILES[split]:
        path = raw_dir / filename
        if path.exists():
            return path
    candidates = ", ".join(DEFAULT_SPLIT_FILES[split])
    raise FileNotFoundError(f"找不到 {split} CSV，已尝试：{candidates}")


def looks_like_header(row: list[str]) -> bool:
    lowered = {value.strip().lower() for value in row}
    known_names = {"image", "image_name", "filename", "x1", "y1", "x2", "y2", "xmin", "ymin", "xmax", "ymax"}
    return bool(lowered & known_names)


def read_csv_rows(path: Path) -> tuple[list[str] | None, list[list[str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        rows = [row for row in csv.reader(file) if row]
    if not rows:
        return None, []
    if looks_like_header(rows[0]):
        return [name.strip().lower() for name in rows[0]], rows[1:]
    return None, rows


def value_by_name(row: list[str], header: list[str] | None, names: tuple[str, ...]) -> str | None:
    if header is None:
        return None
    for name in names:
        if name in header:
            index = header.index(name)
            if index < len(row):
                return row[index].strip()
    return None


def value_by_index(row: list[str], index: int) -> str:
    if index >= len(row):
        raise ValueError(f"CSV 列数不足，无法读取第 {index + 1} 列：{row}")
    return row[index].strip()


def parse_float(value: str, field_name: str) -> float:
    try:
        return float(value)
    except ValueError as error:
        raise ValueError(f"字段 {field_name} 不是有效数字：{value}") from error


def get_image_size(row: list[str], header: list[str] | None, image_path: Path) -> tuple[float, float]:
    width_value = value_by_name(row, header, ("image_width", "width", "img_width", "w"))
    height_value = value_by_name(row, header, ("image_height", "height", "img_height", "h"))
    if width_value is None and len(row) >= 8:
        width_value = value_by_index(row, 6)
    if height_value is None and len(row) >= 8:
        height_value = value_by_index(row, 7)
    if width_value is not None and height_value is not None:
        return parse_float(width_value, "image_width"), parse_float(height_value, "image_height")

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            f"标注中缺少图片宽高，且未安装 Pillow，无法读取图片尺寸：{image_path}"
        ) from error

    with Image.open(image_path) as image:
        return float(image.width), float(image.height)


def parse_annotation(
    row: list[str],
    header: list[str] | None,
    image_path: Path,
    class_mapping: dict[str, int],
    single_class_name: str,
    preserve_classes: bool,
) -> tuple[str, Box, bool] | None:
    image_name = value_by_name(row, header, ("image", "image_name", "filename", "file_name", "path"))
    image_name = image_name or value_by_index(row, 0)

    x1_value = value_by_name(row, header, ("x1", "xmin", "left")) or value_by_index(row, 1)
    y1_value = value_by_name(row, header, ("y1", "ymin", "top")) or value_by_index(row, 2)
    x2_value = value_by_name(row, header, ("x2", "xmax", "right")) or value_by_index(row, 3)
    y2_value = value_by_name(row, header, ("y2", "ymax", "bottom")) or value_by_index(row, 4)

    x1 = parse_float(x1_value, "x1")
    y1 = parse_float(y1_value, "y1")
    x2 = parse_float(x2_value, "x2")
    y2 = parse_float(y2_value, "y2")
    image_width, image_height = get_image_size(row, header, image_path)
    if image_width <= 0 or image_height <= 0:
        return None

    clipped = False
    clipped_x1 = min(max(x1, 0.0), image_width)
    clipped_y1 = min(max(y1, 0.0), image_height)
    clipped_x2 = min(max(x2, 0.0), image_width)
    clipped_y2 = min(max(y2, 0.0), image_height)
    if (clipped_x1, clipped_y1, clipped_x2, clipped_y2) != (x1, y1, x2, y2):
        clipped = True

    box_width = clipped_x2 - clipped_x1
    box_height = clipped_y2 - clipped_y1
    if box_width <= 0 or box_height <= 0:
        return None

    if preserve_classes:
        label = value_by_name(row, header, ("class", "label", "category", "category_name"))
        label = label or (value_by_index(row, 5) if len(row) > 5 else single_class_name)
    else:
        label = single_class_name
    if label not in class_mapping:
        class_mapping[label] = len(class_mapping)

    box = Box(
        class_id=class_mapping[label],
        x_center=((clipped_x1 + clipped_x2) / 2.0) / image_width,
        y_center=((clipped_y1 + clipped_y2) / 2.0) / image_height,
        width=box_width / image_width,
        height=box_height / image_height,
    )
    return image_name, box, clipped


def build_image_index(raw_dir: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = defaultdict(list)
    for path in raw_dir.rglob("*"):
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
            index[path.name].append(path)
            relative = path.relative_to(raw_dir).as_posix()
            index[relative].append(path)
    return index


def find_image(image_index: dict[str, list[Path]], image_name: str) -> Path | None:
    normalized = image_name.replace("\\", "/")
    matches = image_index.get(normalized) or image_index.get(Path(normalized).name)
    if not matches:
        return None
    return matches[0]


def place_image(source: Path, target: Path, mode: str, overwrite: bool) -> None:
    if mode == "none":
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if not overwrite:
            return
        target.unlink()
    if mode == "symlink":
        target.symlink_to(source.resolve())
    elif mode == "copy":
        shutil.copy2(source, target)


def write_labels(labels_dir: Path, image_name: str, boxes: list[Box]) -> None:
    label_path = labels_dir / f"{Path(image_name).stem}.txt"
    label_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"{box.class_id} {box.x_center:.6f} {box.y_center:.6f} {box.width:.6f} {box.height:.6f}"
        for box in boxes
    ]
    label_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def process_split(
    split: str,
    csv_path: Path,
    output_dir: Path,
    image_index: dict[str, list[Path]],
    class_mapping: dict[str, int],
    single_class_name: str,
    preserve_classes: bool,
    link_mode: str,
    overwrite_links: bool,
) -> SplitStats:
    header, rows = read_csv_rows(csv_path)
    boxes_by_image: dict[str, list[Box]] = defaultdict(list)
    source_by_image: dict[str, Path] = {}
    stats = SplitStats()

    for row in rows:
        image_name = value_by_name(row, header, ("image", "image_name", "filename", "file_name", "path"))
        image_name = image_name or value_by_index(row, 0)
        image_path = find_image(image_index, image_name)
        if image_path is None:
            stats.missing_images += 1
            continue

        parsed = parse_annotation(row, header, image_path, class_mapping, single_class_name, preserve_classes)
        if parsed is None:
            stats.skipped_boxes += 1
            continue
        normalized_name, box, clipped = parsed
        output_name = Path(normalized_name).name
        boxes_by_image[output_name].append(box)
        source_by_image[output_name] = image_path
        stats.boxes += 1
        if clipped:
            stats.clipped_boxes += 1

    images_dir = output_dir / "images" / split
    labels_dir = output_dir / "labels" / split
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    for image_name, boxes in boxes_by_image.items():
        source = source_by_image[image_name]
        place_image(source, images_dir / Path(image_name).name, link_mode, overwrite_links)
        write_labels(labels_dir, image_name, boxes)
    stats.images = len(boxes_by_image)
    return stats


def write_dataset_yaml(yaml_path: Path, output_dir: Path, class_mapping: dict[str, int]) -> None:
    names = [name for name, _ in sorted(class_mapping.items(), key=lambda item: item[1])]
    lines = [
        f"path: {output_dir.as_posix()}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        "names:",
    ]
    lines.extend(f"  {index}: {name}" for index, name in enumerate(names))
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    raw_dir = args.raw_dir.resolve()
    output_dir = args.output_dir
    if not raw_dir.exists():
        raise FileNotFoundError(f"原始数据目录不存在：{raw_dir}")

    explicit_csvs = {"train": args.train_csv, "val": args.val_csv, "test": args.test_csv}
    split_csvs = {
        split: resolve_csv(raw_dir, split, explicit_csvs[split])
        for split in args.splits
    }
    image_index = build_image_index(raw_dir)
    if not image_index:
        raise FileNotFoundError(f"在原始数据目录中未找到图片：{raw_dir}")

    class_mapping: dict[str, int] = {} if args.preserve_classes else {args.class_name: 0}
    for split, csv_path in split_csvs.items():
        stats = process_split(
            split=split,
            csv_path=csv_path,
            output_dir=output_dir,
            image_index=image_index,
            class_mapping=class_mapping,
            single_class_name=args.class_name,
            preserve_classes=args.preserve_classes,
            link_mode=args.link_mode,
            overwrite_links=args.overwrite_links,
        )
        print(
            f"{split}: images={stats.images}, boxes={stats.boxes}, "
            f"skipped_boxes={stats.skipped_boxes}, clipped_boxes={stats.clipped_boxes}, "
            f"missing_images={stats.missing_images}"
        )

    write_dataset_yaml(args.yaml_path, output_dir, class_mapping)
    print(f"已写入数据配置：{args.yaml_path}")


if __name__ == "__main__":
    main()
