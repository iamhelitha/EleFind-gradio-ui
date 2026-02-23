---
title: EleFind - Aerial Elephant Detection
emoji: "\U0001F418"
colorFrom: green
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
python_version: "3.10"
suggested_hardware: cpu-basic
license: mit
tags:
  - object-detection
  - yolo
  - yolov11
  - sahi
  - computer-vision
  - elephant-detection
  - wildlife-conservation
  - aerial-imagery
pinned: false
---

# EleFind - Aerial Elephant Detection

Detect elephants in aerial/drone imagery using **YOLOv11** with **SAHI** (Slicing Aided Hyper Inference) and Explainable AI heatmap visualisations.

## Features

- Upload aerial images and get elephant detections with bounding boxes
- XAI density heatmap showing model attention areas
- Adjustable SAHI parameters (confidence, slice size, overlap, IoU)
- Automatic model download from HuggingFace Hub

## Model Details

| Property | Value |
|---|---|
| Architecture | YOLOv11 |
| Training data | Sliced aerial elephant imagery (1024x1024 patches) |
| Inference | SAHI with NMS post-processing |
| Precision | 53.2% |
| Recall | 49.1% |
| F1-Score | 51.0% |

### Optimised SAHI Configuration

| Parameter | Value |
|---|---|
| Slice size | 1024x1024 |
| Overlap ratio | 0.30 |
| Confidence threshold | 0.30 |
| IoU threshold | 0.40 |

## Local Setup

```bash
# Clone the repo
git clone https://github.com/helithalochana/EleFind-gradio-ui.git
cd EleFind-gradio-ui

# Install dependencies
pip install -r requirements.txt

# Place best.pt in the project root or models/ directory
# Or set HF_MODEL_REPO env variable

# Run the app
python app.py

# Run tests
pytest test_detection.py -v
pytest test_detection.py -v -m "not slow"   # skip inference tests
```

## Environment Variables

| Variable | Description | Default |
|---|---|---|
| `HF_MODEL_REPO` | HuggingFace model repo (e.g. `helitha/EleFind-yolo11-elephant`) | `""` |
| `HF_MODEL_FILE` | Model filename in the repo | `best.pt` |

## Project Structure

```
EleFind-gradio-ui/
├── app.py               # Gradio web interface (HF Spaces entry point)
├── test_detection.py    # pytest test suite
├── requirements.txt     # Python dependencies
├── packages.txt         # System dependencies (for HF Spaces)
├── pytest.ini           # pytest configuration
├── examples/            # Sample aerial images for the demo
├── README.md            # This file (HF Spaces config + docs)
└── meeting_materials/   # Training docs, evaluation results, visualisations
```

## Tech Stack

- **[Ultralytics YOLOv11](https://docs.ultralytics.com/)** — object detection model
- **[SAHI](https://github.com/obss/sahi)** — slicing aided hyper inference for high-res images
- **[Gradio](https://gradio.app/)** — web UI framework
- **[HuggingFace Hub](https://huggingface.co/)** — model hosting and Spaces deployment

## Author

**Helitha Guruge** — Undergraduate Research Project (EleFind)

## License

MIT
