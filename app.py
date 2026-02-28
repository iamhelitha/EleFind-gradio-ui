"""
EleFind - Aerial Elephant Detection
=====================================

A Gradio 6 web interface for detecting elephants in aerial/drone imagery
using YOLOv11 with SAHI (Slicing Aided Hyper Inference).

Features:
- Upload aerial images and detect elephants with bounding boxes
- Adjustable SAHI parameters (confidence, slice size, overlap)
- Automatic model download from HuggingFace Hub
- Confidence bar chart and detection data table

Author: Helitha Guruge
Project: EleFind (Undergraduate Research Project)
"""

import os
import uuid
import warnings
from pathlib import Path

import cv2
import gradio as gr
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

# Optional pandas for chart data
try:
    import pandas as pd
    _PANDAS = True
    _EMPTY_CHART = pd.DataFrame({"Elephant": pd.Series([], dtype=str), "Confidence": pd.Series([], dtype=float)})
except ImportError:
    _PANDAS = False
    _EMPTY_CHART = None

# ---------------------------------------------------------------------------
# Imports: detection libraries
# ---------------------------------------------------------------------------
try:
    import torch
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
except ImportError as e:
    raise SystemExit(
        f"Missing required packages: {e}\n"
        "Install with: pip install -r requirements.txt"
    )

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# HuggingFace model repository (update after you create the HF repo)
HF_MODEL_REPO = os.environ.get("HF_MODEL_REPO", "iamhelitha/EleFind-yolo11-elephant")
HF_MODEL_FILE = os.environ.get("HF_MODEL_FILE", "best.pt")

# Local fallback: look for model in common locations
LOCAL_MODEL_PATHS = [
    Path(__file__).parent / "best.pt",
    Path(__file__).parent / "models" / "best.pt",
    Path(__file__).parent / "meeting_materials" / "models" / "best.pt",
]

# Default SAHI parameters (optimized for elephant detection)
DEFAULT_CONF = 0.30
DEFAULT_SLICE = 1024
DEFAULT_OVERLAP = 0.30
DEFAULT_IOU = 0.40

# Image size limit for CPU inference (avoid timeouts on free Spaces)
MAX_IMAGE_DIMENSION = 6000


# ---------------------------------------------------------------------------
# Device detection
# ---------------------------------------------------------------------------
def get_device() -> str:
    """Detect the best available compute device."""
    try:
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def _resolve_model_path() -> str:
    """Resolve the model path: try HuggingFace Hub first, then local."""

    # 1. Try downloading from HuggingFace Hub
    if HF_MODEL_REPO:
        try:
            from huggingface_hub import hf_hub_download

            print(f"Downloading model from HuggingFace: {HF_MODEL_REPO}/{HF_MODEL_FILE}")
            path = hf_hub_download(
                repo_id=HF_MODEL_REPO,
                filename=HF_MODEL_FILE,
                repo_type="model",
            )
            print(f"Model downloaded to: {path}")
            return path
        except Exception as e:
            print(f"HuggingFace download failed: {e}. Trying local paths...")

    # 2. Try local paths
    for local_path in LOCAL_MODEL_PATHS:
        if local_path.exists():
            print(f"Using local model: {local_path}")
            return str(local_path)

    raise FileNotFoundError(
        "Model not found. Set HF_MODEL_REPO env var or place best.pt "
        "in the project root or models/ directory."
    )


def load_model() -> AutoDetectionModel:
    """Load the SAHI-wrapped detection model."""
    model_path = _resolve_model_path()
    device = get_device()
    print(f"Loading model on device: {device}")

    model = AutoDetectionModel.from_pretrained(
        model_type="yolov8",  # SAHI uses 'yolov8' for YOLOv8/v11 models
        model_path=model_path,
        confidence_threshold=DEFAULT_CONF,
        device=device,
    )
    print("Model loaded successfully!")
    return model


# Load model at startup (cached for all requests)
print("=" * 60)
print("EleFind - Initializing Elephant Detection Model")
print("=" * 60)
detection_model = load_model()
DEVICE = get_device()
print(f"Device: {DEVICE}")
print("=" * 60)


# ---------------------------------------------------------------------------
# Detection functions
# ---------------------------------------------------------------------------
def validate_image(image_np: np.ndarray) -> np.ndarray:
    """Validate and optionally resize image to avoid CPU timeouts."""
    h, w = image_np.shape[:2]

    if max(h, w) > MAX_IMAGE_DIMENSION:
        scale = MAX_IMAGE_DIMENSION / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        image_np = cv2.resize(image_np, (new_w, new_h), interpolation=cv2.INTER_AREA)
        print(f"Image resized from {w}x{h} to {new_w}x{new_h}")

    return image_np


def run_detection(
    image_np: np.ndarray,
    conf_threshold: float = DEFAULT_CONF,
    slice_size: int = DEFAULT_SLICE,
    overlap_ratio: float = DEFAULT_OVERLAP,
    iou_threshold: float = DEFAULT_IOU,
) -> list:
    """Run SAHI sliced prediction and return a list of detection dicts."""

    # Update model confidence threshold
    detection_model.confidence_threshold = conf_threshold

    # Save temp image for SAHI (requires file path)
    temp_path = Path(__file__).parent / f"temp_input_{uuid.uuid4().hex}.jpg"
    cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))

    try:
        result = get_sliced_prediction(
            image=str(temp_path),
            detection_model=detection_model,
            slice_height=slice_size,
            slice_width=slice_size,
            overlap_height_ratio=overlap_ratio,
            overlap_width_ratio=overlap_ratio,
            postprocess_type="NMS",
            postprocess_match_threshold=iou_threshold,
            verbose=0,
        )

        predictions = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            predictions.append(
                {
                    "x1": int(bbox.minx),
                    "y1": int(bbox.miny),
                    "x2": int(bbox.maxx),
                    "y2": int(bbox.maxy),
                    "confidence": round(obj.score.value, 4),
                }
            )
    finally:
        if temp_path.exists():
            temp_path.unlink()

    return predictions


def draw_detections(image: np.ndarray, predictions: list) -> np.ndarray:
    """Draw bounding boxes and labels on the image."""
    img = image.copy()

    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        conf = pred["confidence"]

        # Green bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Label background + text
        label = f"Elephant {conf:.0%}"
        (lw, lh), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
        )
        cv2.rectangle(img, (x1, y1 - lh - 10), (x1 + lw + 5, y1), (0, 255, 0), -1)
        cv2.putText(
            img, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2
        )

    return img


# ---------------------------------------------------------------------------
# Normalisation helpers (handle various Gradio input types)
# ---------------------------------------------------------------------------
def _to_numpy_rgb(image):
    """Convert Gradio image input to numpy RGB array."""
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
        return image
    if isinstance(image, str):
        return np.array(Image.open(image).convert("RGB"))
    if isinstance(image, dict):
        path = image.get("path") or image.get("name")
        if path:
            return np.array(Image.open(path).convert("RGB"))
    return None


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------
def process_image(
    image,
    conf_threshold: float,
    slice_size: int,
    overlap_ratio: float,
    iou_threshold: float,
    progress=gr.Progress(),
):
    """Run detection pipeline and return annotated image, stat values,
    parameters text, confidence chart data, and detection table data."""

    image_np = _to_numpy_rgb(image)
    if image_np is None:
        return None, 0, 0.0, 0.0, 0.0, "", None, None

    try:
        progress(0.05, desc="Validating image")
        image_np = validate_image(image_np)
        h, w = image_np.shape[:2]

        progress(0.10, desc=f"Running SAHI detection ({w}×{h})")
        predictions = run_detection(
            image_np,
            conf_threshold=conf_threshold,
            slice_size=int(slice_size),
            overlap_ratio=overlap_ratio,
            iou_threshold=iou_threshold,
        )

        progress(0.80, desc="Drawing detections")
        det_image = draw_detections(image_np, predictions)

    except Exception as e:
        import traceback
        err_msg = f"Error: {e}\n\n```\n{traceback.format_exc()}\n```"
        return None, 0, 0.0, 0.0, 0.0, err_msg, None, None

    # Compute stats
    n = len(predictions)
    avg_conf = sum(p["confidence"] for p in predictions) / n if n else 0.0
    max_conf = max((p["confidence"] for p in predictions), default=0.0)
    min_conf = min((p["confidence"] for p in predictions), default=0.0)

    params_text = (
        f"**Parameters:** Slice {int(slice_size)}x{int(slice_size)} px · "
        f"Overlap {overlap_ratio:.0%} · Confidence >= {conf_threshold:.0%} · "
        f"IoU {iou_threshold:.0%} · Image {w}x{h} px · Device `{DEVICE}`"
    )

    progress(1.0, desc="Done")

    # Build chart / table data (pandas optional)
    conf_chart = None
    det_table = None
    if _PANDAS and predictions:
        det_table = pd.DataFrame(
            [
                {
                    "ID": i + 1,
                    "Confidence": f"{p['confidence']:.1%}",
                    "BBox (x1,y1,x2,y2)": f"({p['x1']},{p['y1']},{p['x2']},{p['y2']})",
                    "Width (px)": p["x2"] - p["x1"],
                    "Height (px)": p["y2"] - p["y1"],
                }
                for i, p in enumerate(predictions)
            ]
        )
        conf_chart = pd.DataFrame(
            {
                "Elephant": [f"#{i+1}" for i in range(len(predictions))],
                "Confidence": [round(p["confidence"] * 100, 1) for p in predictions],
            }
        )

    return (
        Image.fromarray(det_image.astype(np.uint8)),
        n,
        avg_conf,
        max_conf,
        min_conf,
        params_text,
        conf_chart,
        det_table,
    )


# ---------------------------------------------------------------------------
# Gradio UI  –  Gradio 6.x
# ---------------------------------------------------------------------------

_THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="green",
    neutral_hue="gray",
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
)

def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    with gr.Blocks(
        theme=_THEME,
        title="EleFind – Aerial Elephant Detection",
        fill_width=False,
    ) as demo:

        # ── Hero banner ─────────────────────────────────────────────────────
        gr.Markdown(
            "# EleFind\n\n"
            "Aerial elephant detection powered by **YOLOv11 + SAHI** "
            "(Slicing Aided Hyper Inference). Upload a drone or satellite image "
            "and get instant detection results."
        )

        # ── Main two-column layout ─────────────────────────────────────────
        with gr.Row(equal_height=False):

            # ── LEFT: Input panel ─────────────────────────────────────────
            with gr.Column(scale=4, min_width=320):

                input_image = gr.Image(
                    label="Upload Aerial / Drone Image",
                    type="pil",
                    sources=["upload", "clipboard"],
                    show_fullscreen_button=True,
                    height=320,
                )

                with gr.Accordion("SAHI Detection Parameters", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=DEFAULT_CONF,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum score for a detection to be kept",
                    )
                    slice_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=DEFAULT_SLICE,
                        step=128,
                        label="Slice Size (px)",
                        info="Width & height of each SAHI tile",
                    )
                    overlap_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.50,
                        value=DEFAULT_OVERLAP,
                        step=0.05,
                        label="Tile Overlap Ratio",
                        info="Fraction of overlap between adjacent tiles",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.80,
                        value=DEFAULT_IOU,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                        info="Suppress duplicate boxes above this overlap",
                    )

                detect_btn = gr.Button(
                    "Detect Elephants",
                    variant="primary",
                    size="lg",
                )


            # ── RIGHT: Output tabs ────────────────────────────────────────
            with gr.Column(scale=6, min_width=400):

                with gr.Tabs() as result_tabs:

                    # ── Tab 1: Detection image ────────────────────────────
                    with gr.Tab("Detections", id="tab_det"):
                        detection_output = gr.Image(
                            label="Annotated detections",
                            type="pil",
                            interactive=False,
                            show_download_button=True,
                            show_fullscreen_button=True,
                            height=420,
                        )

                    # ── Tab 2: Statistics ─────────────────────────────────
                    with gr.Tab("Statistics", id="tab_stats"):
                        with gr.Row():
                            stat_count = gr.Number(
                                label="Elephants Detected", value=0,
                                interactive=False,
                            )
                            stat_avg = gr.Number(
                                label="Avg Confidence", value=0.0,
                                interactive=False, precision=2,
                            )
                            stat_max = gr.Number(
                                label="Highest Confidence", value=0.0,
                                interactive=False, precision=2,
                            )
                            stat_min = gr.Number(
                                label="Lowest Confidence", value=0.0,
                                interactive=False, precision=2,
                            )
                        params_md = gr.Markdown()

                        with gr.Accordion("Detection Table", open=True):
                            det_table_out = gr.Dataframe(
                                headers=["ID", "Confidence", "BBox (x1,y1,x2,y2)",
                                         "Width (px)", "Height (px)"],
                                label=None,
                                interactive=False,
                                wrap=True,
                            )

                        with gr.Accordion("Confidence Chart", open=True):
                            if _PANDAS:
                                conf_chart_out = gr.BarPlot(
                                    value=_EMPTY_CHART,
                                    x="Elephant",
                                    y="Confidence",
                                    title="Detection Confidence per Elephant",
                                    x_title="Elephant ID",
                                    y_title="Confidence (%)",
                                    color="Confidence",
                                    height=280,
                                    label="",
                                    show_label=False,
                                )
                            else:
                                conf_chart_out = gr.Markdown(
                                    "_Install pandas for the confidence chart._"
                                )

        # ── Example images ─────────────────────────────────────────────────
        example_dir = Path(__file__).parent / "examples"
        example_files = sorted(example_dir.glob("*.jpg")) if example_dir.exists() else []
        if example_files:
            with gr.Accordion("Example Aerial Images", open=True):
                gr.Examples(
                    examples=[[str(f)] for f in example_files],
                    inputs=[input_image],
                    label=None,
                )

        # ── Event wiring ───────────────────────────────────────────────────
        _outputs = [
            detection_output,
            stat_count,
            stat_avg,
            stat_max,
            stat_min,
            params_md,
            conf_chart_out,
            det_table_out,
        ]

        detect_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                conf_slider,
                slice_slider,
                overlap_slider,
                iou_slider,
            ],
            outputs=_outputs,
            concurrency_limit=1,
            api_name="detect",
        )

    return demo


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
demo = build_ui()

if __name__ == "__main__":
    demo.queue(max_size=10)
    demo.launch(
        server_name="0.0.0.0",
        share=False,
        show_error=True,
    )
