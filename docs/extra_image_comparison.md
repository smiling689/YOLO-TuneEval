# Extra Image Qualitative Comparison

## Models

- Before fine-tuning: `models/yolov8n.pt`
- After AdamW fine-tuning: `runs/train/yolov8n_sku110k_adamw_full/weights/best.pt`
- Confidence threshold: 0.25
- IoU threshold: 0.7
- Image size: 640

## Output Directories

- Before visualizations: `report/outputs/extra_images/before/`
- After visualizations: `report/outputs/extra_images/after/`
- Before labels: `report/outputs/extra_images/before/labels/`
- After labels: `report/outputs/extra_images/after/labels/`

## Detection Counts

| Image | Before detections | After detections |
|---|---:|---:|
| `01_coffee_shelves_bulgaria.jpg` | 0 | 150 |
| `02_spar_cereal_shelves.jpg` | 2 | 99 |
| `03_spar_sauce_shelves.jpg` | 25 | 179 |
| `04_rema_pet_food_shelves.jpg` | 0 | 161 |
| `05_mackerel_shelves_norway.jpg` | 1 | 47 |

## Brief Analysis

The original COCO-pretrained YOLOv8n mostly fails to treat shelf products as the target class. Its few detections are COCO-category detections such as bottles, donuts, or a refrigerator, which are not aligned with the SKU-110K single-class object setting.

After AdamW fine-tuning, the model detects dense shelf products as `object` across all five images. This indicates clear transfer to non-SKU-110K supermarket shelf scenes, although some qualitative risks remain: dense shelves can lead to overlapping boxes, missed small packages, and occasional over-detection on visually repeated packaging or shelf structures.
