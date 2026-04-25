# EleFind Thesis Q&A (Code-Verified)

Date verified: April 20, 2026
Primary source: app.py (plus workflow and docs files listed below)

## SAHI configuration

### 1) What is the exact slice_height and slice_width used in the SAHI inference call? Is it 640 or 1024?

Answer:
- In the SAHI call, both are set to `slice_size`.
- The default `slice_size` in the app is `1024`, not 640.
- The UI slider allows changing it at runtime (range 256 to 2048), but default is 1024.

### 2) What is the exact overlap_height_ratio and overlap_width_ratio? Is it 0.20 or 0.30?

Answer:
- In the SAHI call, both are set to `overlap_ratio`.
- The default `overlap_ratio` in the app is `0.30`, not 0.20.
- The UI slider allows runtime change (range 0.05 to 0.50), but default is 0.30.

### 3) What is the exact confidence threshold passed to the SAHI inference?

Answer:
- The model confidence is set immediately before inference:
  `detection_model.confidence_threshold = conf_threshold`
- The default `conf_threshold` is `0.30`.
- The UI slider allows runtime change (range 0.05 to 0.95), but default is 0.30.

### 4) What is the IoU threshold used for NMS post-processing?

Answer:
- SAHI uses `postprocess_type="NMS"`.
- IoU is passed as `postprocess_match_threshold=iou_threshold`.
- Default `iou_threshold` is `0.40`.
- The UI slider allows runtime change (range 0.10 to 0.80), but default is 0.40.

### 5) What is the exact model_type and model_path or Hugging Face repo ID used to load the model?

Answer:
- `model_type` is `"yolov8"` (comment notes SAHI uses this for YOLOv8/v11 models).
- `model_path` is a resolved file path from:
  1. Hugging Face Hub download first (preferred)
  2. Local fallback paths if download fails
- Hugging Face model repo ID default: `iamhelitha/EleFind-yolo11-elephant`
- Hugging Face model filename default: `best.pt`

## API endpoint

### 6) What is the exact api_name parameter on the Gradio detection function?

Answer:
- Exact value is `"detect"` in `detect_btn.click(...)`.
- In Gradio client usage, this endpoint is typically called as `/detect`.

### 7) What is the exact Hugging Face Space name/URL?

Answer:
- Space identifier: `iamhelitha/EleFind-gradio-ui`
- Space URL: `https://huggingface.co/spaces/iamhelitha/EleFind-gradio-ui`
- Public runtime URL used for API calls: `https://iamhelitha-elefind-gradio-ui.hf.space`

### 8) What is the Hugging Face Hub model repo ID where weights are loaded from?

Answer:
- `iamhelitha/EleFind-yolo11-elephant`

## CI/CD

### 9) Is there a GitHub Actions workflow file? What does it do on push to main?

Answer:
- Yes: `.github/workflows/deploy-hf.yml`
- Trigger is exactly:
  - `on: push` to branch `main`
- Pipeline behavior:
  1. Checks out repo
  2. Sets up Python 3.10
  3. Installs `huggingface-hub`
  4. Uploads project to HF Space `iamhelitha/EleFind-gradio-ui` using `upload_folder(..., repo_type="space")`
  5. Polls HF Space runtime status until RUNNING (or reports build/runtime error)
- So yes, push to main syncs/deploys to Hugging Face Spaces.

## Gradio interface

### 10) What inputs does the Gradio interface accept?

Answer:
- One image input component (`gr.Image`), sources: upload and clipboard.
- Plus 4 SAHI parameter sliders:
  1. Confidence threshold
  2. Slice size (px)
  3. Tile overlap ratio
  4. IoU threshold (NMS)

### 11) What outputs does the Gradio interface return?

Answer:
The detection function returns 8 outputs:
1. Annotated detection image
2. Elephant count
3. Average confidence
4. Highest confidence
5. Lowest confidence
6. Parameters summary markdown text
7. Confidence chart data (BarPlot data; fallback markdown if pandas unavailable)
8. Detection table data

### 12) Is there a requirements.txt or packages.txt, and what are key dependencies?

Answer:
- `requirements.txt` exists and includes:
  - `ultralytics>=8.0.0`
  - `sahi>=0.11.0`
  - `opencv-python-headless>=4.5.0`
  - `torch>=2.0.0`
  - `torchvision>=0.15.0`
  - `huggingface-hub>=0.20.0`
  - `Pillow>=9.0.0`
  - `numpy>=1.24.0`
  - `pandas>=2.0.0`
- Gradio is not pinned in `requirements.txt`; Spaces uses `sdk_version: 6.8.0` from README frontmatter.
- `packages.txt` exists and is currently empty.

---

## Source files checked

- `app.py`
- `.github/workflows/deploy-hf.yml`
- `README.md`
- `requirements.txt`
- `packages.txt`
- `docs/api-usage.md`
