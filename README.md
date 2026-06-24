# Part 1: Glove Detection

## Overview
This project implements a safety compliance system that detects whether workers are wearing gloves using a fine-tuned YOLOv8 object detection model.

## Dataset
- **Source:** Ultralytics Construction-PPE Dataset
- **Direct download:** https://github.com/ultralytics/assets/releases/download/v0.0.0/construction-ppe.zip
- **Classes used:** `gloves` and `no_gloves` (filtered from 11 original PPE classes)
- **Remapped to:** `gloved_hand` (class 0) and `bare_hand` (class 1)
- **Split:**
  - Train: ~700 images
  - Validation: ~80 images
  - Test: ~90 images

## Model
- **Architecture:** YOLOv8s (small) from Ultralytics
- **Training:** Fine-tuned for up to 100 epochs on Google Colab T4 GPU
- **Image size:** 800x800
- **Batch size:** 16
- **Classes:** 2 (`gloved_hand`, `bare_hand`)
- **Key hyperparameters:**
  - `cls=2.0` (increased classification loss weight)
  - `cos_lr=True` (cosine learning rate schedule)
  - `mixup=0.2` (mixup augmentation)
  - `patience=25` (early stopping)

## Preprocessing & Training
1. Downloaded the Construction-PPE dataset (178 MB)
2. Filtered label files to keep only class 1 (`gloves` → `gloved_hand`) and class 9 (`no_gloves` → `bare_hand`)
3. Remapped class indices to 0 and 1
4. Trained YOLOv8s on Google Colab T4 GPU with default augmentations (mosaic, mixup, HSV jitter, fliplr) plus custom `cls=2.0` weighting

## What Worked
- Filtering a multi-class PPE dataset down to glove-specific classes gave a focused dataset quickly
- YOLOv8s at 800x800 resolution significantly outperformed the initial YOLOv8n at 640x640
- The `cls=2.0` classification weighting helped the model pay more attention to class discrimination
- Built-in YOLO augmentations (mosaic, mixup, HSV jitter) improved generalization without extra code
- The compliance flagging in the detection script makes results actionable for safety monitoring

## What Didn't Work / Trade-offs
- **bare_hand detection remains weak** (mAP50 ~0.16) due to dataset imbalance — fewer bare-hand training examples and higher visual variability (skin tones, lighting, poses)
- The Construction-PPE dataset contains construction scenes only; the model may not generalize to factory or medical glove environments without domain-specific data
- CPU training was prohibitively slow (~7-8 min/epoch); GPU training on Colab was essential
- Some test images contain no visible hands, resulting in "NO_HANDS" status rather than a detection

## How to Run

### Prerequisites
```bash
pip install ultralytics opencv-python numpy
```

### Testing / Inference on New Images
Use `detection_script.py` to run inference on a folder of `.jpg` images. The script loads the fine-tuned YOLOv8 model, performs object detection on every image, draws bounding boxes, saves annotated results, and writes JSON logs.

**Quick start:**
```bash
python submission/Part_1_Glove_Detection/detection_script.py \
  --input path/to/your/test/images \
  --output submission/Part_1_Glove_Detection/output \
  --logs submission/Part_1_Glove_Detection/logs \
  --model models/best.pt \
  --confidence 0.25
```

**Example — run from inside the Part_1 folder:**
```bash
cd submission/Part_1_Glove_Detection
python detection_script.py \
  --input path/to/your/images \
  --output output \
  --logs logs \
  --model ../../models/best.pt
```

### Arguments
| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Folder containing `.jpg` images |
| `--output` | required | Folder to save annotated images |
| `--logs` | required | Folder to save JSON detection logs |
| `--model` | `models/best.pt` | Path to trained YOLO weights |
| `--confidence` | `0.25` | Confidence threshold for detections |

## Output Format
Annotated images are saved to `output/` with bounding boxes drawn in green (`gloved_hand`) or red (`bare_hand`), plus a compliance status banner at the top:
- **COMPLIANT** (green) — only gloved hands detected
- **VIOLATION** (red) — bare hand(s) detected
- **NO_HANDS** (orange) — no hands found in image

JSON logs per image (`logs/{filename}.json`):
```json
{
  "filename": "image1.jpg",
  "compliance_status": "COMPLIANT",
  "compliance_message": "2 gloved hand(s) detected",
  "detections": [
    {"label": "gloved_hand", "confidence": 0.92, "bbox": [x1, y1, x2, y2]},
    {"label": "bare_hand", "confidence": 0.85, "bbox": [x1, y1, x2, y2]}
  ]
}
```

## Project Structure
```
glove-detection/
├── models/
│   └── best.pt                    # Trained YOLOv8s weights
├── notebooks/
│   └── fine_tune.ipynb            # Colab training notebook
├── submission/
│   ├── Part_1_Glove_Detection/
│   │   ├── detection_script.py    # Standalone inference script
│   │   ├── output/                # Annotated sample images (127)
│   │   ├── logs/                  # JSON detection logs (127)
│   │   └── README.md              # This file
│   └── Part_2_Answers.md          # Reasoning questions
```
