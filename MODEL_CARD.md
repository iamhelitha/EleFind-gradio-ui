---
library_name: ultralytics
tags:
  - object-detection
  - yolo
  - yolov11
  - sahi
  - elephant-detection
  - wildlife-conservation
  - aerial-imagery
  - computer-vision
license: mit
pipeline_tag: object-detection
model-index:
  - name: EleFind YOLOv11 Elephant Detector
    results:
      - task:
          type: object-detection
        metrics:
          - type: precision
            value: 53.16
          - type: recall
            value: 49.07
          - type: f1
            value: 51.03
---

# EleFind - YOLOv11 Elephant Detection Model

A YOLOv11 model fine-tuned for detecting elephants in high-resolution aerial/drone imagery, designed to work with **SAHI** (Slicing Aided Hyper Inference).

## Model Description

This model detects elephants from aerial photographs taken by drones. It was trained on sliced aerial images (1024x1024 patches) and is optimised for use with SAHI to handle full-resolution drone imagery (typically 5472x3648).

### Training Configuration

| Parameter | Value |
|---|---|
| Base model | YOLOv11 (pretrained) |
| Task | Object detection (single class: elephant) |
| Image size | 1024x1024 |
| Epochs | 100 (with early stopping, patience=20) |
| Batch size | 16 |
| Optimiser | Auto |
| Learning rate | 0.01 |
| AMP | Enabled |
| Augmentation | Mosaic, random augment, erasing (0.4), flip LR (0.5) |

### Performance (Test Set - 50 images)

| Metric | Value |
|---|---|
| Precision | 53.16% |
| Recall | 49.07% |
| F1-Score | 51.03% |
| True Positives | 185 |
| False Positives | 163 |
| False Negatives | 192 |

### Optimised SAHI Parameters

| Parameter | Value |
|---|---|
| Slice size | 1024x1024 |
| Overlap ratio | 0.30 |
| Confidence threshold | 0.30 |
| IoU threshold (NMS) | 0.40 |

## Usage

### With SAHI (recommended for high-res images)

```python
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
from huggingface_hub import hf_hub_download

# Download model
model_path = hf_hub_download(
    repo_id="iamhelitha/EleFind-yolo11-elephant",
    filename="best.pt",
    repo_type="model",
)

# Load with SAHI
model = AutoDetectionModel.from_pretrained(
    model_type="yolov8",  # SAHI uses 'yolov8' for YOLOv8/v11 models
    model_path=model_path,
    confidence_threshold=0.30,
    device="cpu",  # or "cuda:0"
)

# Run sliced prediction
result = get_sliced_prediction(
    image="aerial_image.jpg",
    detection_model=model,
    slice_height=1024,
    slice_width=1024,
    overlap_height_ratio=0.30,
    overlap_width_ratio=0.30,
    postprocess_type="NMS",
    postprocess_match_threshold=0.40,
)

print(f"Detected {len(result.object_prediction_list)} elephants")
```

### With Ultralytics directly

```python
from ultralytics import YOLO
from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="iamhelitha/EleFind-yolo11-elephant",
    filename="best.pt",
    repo_type="model",
)

model = YOLO(model_path)
results = model.predict("aerial_image.jpg", conf=0.30)
```

## Intended Use

This model is designed for wildlife conservation research, specifically for counting and locating elephants in aerial survey imagery. It works best with high-resolution drone photographs.

### Limitations

- Trained on a specific dataset of aerial elephant imagery; may not generalise well to different terrains or camera angles
- Optimised for overhead/nadir aerial views; side-angle photographs will perform poorly
- Small elephants or heavily occluded elephants may be missed
- False positives can occur on rocks, shadows, or other objects of similar size/shape

## Author

Helitha Guruge — Undergraduate Research Project

## License

MIT
