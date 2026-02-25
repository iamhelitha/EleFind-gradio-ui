# CLAUDE.md - Project Instructions for Claude Code

## Project Overview

EleFind is an aerial elephant detection web application using YOLOv11 and SAHI (Slicing Aided Hyper Inference). It provides a Gradio-based UI for detecting elephants in drone/aerial imagery for wildlife conservation.

## Tech Stack

- **Language:** Python 3.10
- **UI Framework:** Gradio (5.50.0 on HF Spaces, >=4.44 locally)
- **ML Framework:** Ultralytics YOLOv11, SAHI, PyTorch
- **Deployment:** HuggingFace Spaces

## Project Structure

- `app.py` — Main application (Gradio UI, detection pipeline, heatmap generation)
- `test_detection.py` — Pytest test suite
- `requirements.txt` — Python dependencies
- `packages.txt` — System-level dependencies for HF Spaces
- `README.md` — Project docs (contains HF Spaces frontmatter with `sdk_version`)
- `MODEL_CARD.md` — Model card and usage guide
- `examples/` — Sample aerial images for demo
- `assets/` — Training visualization images
- `meeting_materials/` — Local-only directory (gitignored), contains models and docs

## Running Locally

```bash
pip install -r requirements.txt
python app.py
# Access at http://127.0.0.1:7860
```

## Testing

```bash
# Run all tests
pytest test_detection.py -v

# Skip slow inference tests
pytest test_detection.py -v -m "not slow"

# Run specific test
pytest test_detection.py -v -k "test_model"
```

## Deployment to HuggingFace Spaces

HF Space is NOT synced with GitHub. Upload manually:

```python
from huggingface_hub import upload_folder

upload_folder(
    folder_path='.',
    repo_id='iamhelitha/EleFind-gradio-ui',
    repo_type='space',
    ignore_patterns=[
        '.git/*', '.git',
        'meeting_materials/*', 'meeting_materials',
        '.DS_Store',
        '__pycache__/*', '*.pyc',
        '.claude/*', '.claude',
    ],
    commit_message='Your commit message here',
)
```

## Important Rules

- **Do NOT pin gradio in `requirements.txt`** — the version is controlled by `sdk_version` in `README.md` frontmatter for HF Spaces. Pinning causes build conflicts.
- **Max image dimension is 6000px** — images are downscaled to avoid CPU inference timeouts on free HF Spaces tier.
- **Model auto-downloads from HuggingFace Hub** — configured via `HF_MODEL_REPO` and `HF_MODEL_FILE` env vars, falls back to local paths.
- **`meeting_materials/` is gitignored** — never commit files from this directory.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HF_MODEL_REPO` | `iamhelitha/EleFind-yolo11-elephant` | HuggingFace model repository |
| `HF_MODEL_FILE` | `best.pt` | Model filename in the repo |

## Default SAHI Parameters

- Confidence threshold: `0.30`
- Slice size: `1024 x 1024`
- Overlap ratio: `0.30`
- IoU threshold (NMS): `0.40`

## Code Conventions

- Single main file (`app.py`) contains the full application
- Use OpenCV headless (`opencv-python-headless`) — no GUI dependencies
- Green bounding boxes for detections, Gaussian density heatmaps for XAI
- Gradio Soft theme with emerald/green primary colors

## Git Commit Style

Follow the existing commit message style — short imperative sentences describing the change:
- `Fix HF Space build: remove gradio pin from requirements.txt`
- `Redesign UI with tabbed layout, stats dashboard, and confidence charts`
- `Add deployment instructions for HuggingFace Space`

## Claude Settings

- **`includeCoAuthoredBy`: false** — Do not add Claude as co-author to commits
- **`gitAttribution`: false** — Disable git attribution for Claude
