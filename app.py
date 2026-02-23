"""
EleFind - Aerial Elephant Detection with Explainable AI
========================================================

A Gradio web interface for detecting elephants in aerial/drone imagery
using YOLOv11 with SAHI (Slicing Aided Hyper Inference).

Features:
- Upload aerial images and detect elephants with bounding boxes
- XAI heatmap visualization showing detection density
- Adjustable SAHI parameters (confidence, slice size, overlap)
- Automatic model download from HuggingFace Hub

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
) -> list[dict]:
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


def draw_detections(image: np.ndarray, predictions: list[dict]) -> np.ndarray:
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


def create_heatmap(image: np.ndarray, predictions: list[dict]) -> np.ndarray:
    """Create a Gaussian-blurred density heatmap of detections."""
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pred in predictions:
        x1, y1, x2, y2 = pred["x1"], pred["y1"], pred["x2"], pred["y2"]
        conf = pred["confidence"]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        bw, bh = x2 - x1, y2 - y1
        sx, sy = max(bw / 2, 1), max(bh / 2, 1)

        yr = np.arange(max(0, y1 - bh), min(h, y2 + bh))
        xr = np.arange(max(0, x1 - bw), min(w, x2 + bw))

        if len(xr) > 0 and len(yr) > 0:
            xx, yy = np.meshgrid(xr, yr)
            gaussian = np.exp(
                -((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2))
            )
            gaussian *= conf
            heatmap[yr[0] : yr[-1] + 1, xr[0] : xr[-1] + 1] += gaussian

    if heatmap.max() > 0:
        heatmap /= heatmap.max()

    heatmap_color = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

    blended = cv2.addWeighted(image, 0.4, heatmap_color, 0.6, 0)

    # White detection outlines on heatmap
    for pred in predictions:
        cv2.rectangle(
            blended,
            (pred["x1"], pred["y1"]),
            (pred["x2"], pred["y2"]),
            (255, 255, 255),
            2,
        )

    return blended


# ---------------------------------------------------------------------------
# Normalisation helpers (handle various Gradio input types)
# ---------------------------------------------------------------------------
def _to_numpy_rgb(image) -> np.ndarray | None:
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
    """Run detection pipeline and return annotated image, heatmap, and stats."""

    image_np = _to_numpy_rgb(image)
    if image_np is None:
        return None, None, "Please upload an aerial image to detect elephants."

    try:
        progress(0.05, desc="Validating image")
        image_np = validate_image(image_np)
        h, w = image_np.shape[:2]

        progress(0.10, desc=f"Running SAHI detection ({w}x{h})")
        predictions = run_detection(
            image_np,
            conf_threshold=conf_threshold,
            slice_size=int(slice_size),
            overlap_ratio=overlap_ratio,
            iou_threshold=iou_threshold,
        )

        progress(0.80, desc="Drawing detections")
        det_image = draw_detections(image_np, predictions)

        progress(0.90, desc="Generating heatmap")
        heatmap_image = create_heatmap(image_np, predictions)

    except Exception as e:
        import traceback

        return None, None, f"Error during detection: {e}\n```\n{traceback.format_exc()}\n```"

    # Build statistics markdown
    n = len(predictions)
    if n > 0:
        avg = sum(p["confidence"] for p in predictions) / n
        confs = ", ".join(f'{p["confidence"]:.0%}' for p in predictions[:15])
        if n > 15:
            confs += "..."
        stats = (
            f"### Detection Results\n\n"
            f"- **Elephants detected:** {n}\n"
            f"- **Average confidence:** {avg:.1%}\n"
            f"- **Confidences:** {confs}\n\n"
            f"### Parameters Used\n\n"
            f"- Slice size: {int(slice_size)}x{int(slice_size)}\n"
            f"- Overlap: {overlap_ratio:.0%}\n"
            f"- Confidence threshold: {conf_threshold:.0%}\n"
            f"- IoU threshold: {iou_threshold:.0%}\n"
            f"- Image size: {w}x{h}\n"
            f"- Device: {DEVICE}\n"
        )
    else:
        stats = (
            "### Detection Results\n\n"
            "- **Elephants detected:** 0\n\n"
            "No elephants found. Try lowering the confidence threshold or "
            "upload an aerial image with elephants."
        )

    progress(1.0, desc="Done")
    return (
        Image.fromarray(det_image.astype(np.uint8)),
        Image.fromarray(heatmap_image.astype(np.uint8)),
        stats,
    )


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
.gradio-container { font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
#detection_img img, #heatmap_img img { cursor: zoom-in; transition: opacity 0.2s; }
#detection_img img:hover, #heatmap_img img:hover { opacity: 0.9; }
"""

def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface."""

    with gr.Blocks(css=CSS, title="EleFind - Aerial Elephant Detection") as demo:

        gr.Markdown(
            """
            # EleFind - Aerial Elephant Detection

            Upload an **aerial / drone image** and detect elephants using
            **YOLOv11 + SAHI** (Slicing Aided Hyper Inference).
            Outputs include labelled detections and an XAI density heatmap.

            ---
            """
        )

        with gr.Row():
            # ---- Left column: input + controls ----
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Aerial Image",
                    type="pil",
                    height=400,
                    sources=["upload", "clipboard"],
                )

                with gr.Accordion("Advanced SAHI Parameters", open=False):
                    conf_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=DEFAULT_CONF,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Minimum detection confidence",
                    )
                    slice_slider = gr.Slider(
                        minimum=256,
                        maximum=2048,
                        value=DEFAULT_SLICE,
                        step=128,
                        label="Slice Size (px)",
                        info="Size of each SAHI slice",
                    )
                    overlap_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.50,
                        value=DEFAULT_OVERLAP,
                        step=0.05,
                        label="Overlap Ratio",
                        info="Overlap between adjacent slices",
                    )
                    iou_slider = gr.Slider(
                        minimum=0.10,
                        maximum=0.80,
                        value=DEFAULT_IOU,
                        step=0.05,
                        label="IoU Threshold (NMS)",
                        info="Non-max suppression IoU threshold",
                    )

                detect_btn = gr.Button(
                    "Detect Elephants", variant="primary", size="lg"
                )

                gr.Markdown(
                    """
                    **Tips:**
                    - Best results with high-resolution aerial/drone images
                    - Optimal resolution ~5472x3648 or similar
                    - Lower confidence threshold to find more (but noisier) detections
                    """
                )

        with gr.Row():
            with gr.Column(scale=1):
                detection_output = gr.Image(
                    label="Detections (bounding boxes)",
                    type="pil",
                    height=450,
                    interactive=False,
                    elem_id="detection_img",
                )
            with gr.Column(scale=1):
                heatmap_output = gr.Image(
                    label="XAI Heatmap (detection density)",
                    type="pil",
                    height=450,
                    interactive=False,
                    elem_id="heatmap_img",
                )

        stats_output = gr.Markdown(label="Statistics")

        # Wire up the button
        detect_btn.click(
            fn=process_image,
            inputs=[
                input_image,
                conf_slider,
                slice_slider,
                overlap_slider,
                iou_slider,
            ],
            outputs=[detection_output, heatmap_output, stats_output],
            concurrency_limit=1,
        )

        # Example images
        example_dir = Path(__file__).parent / "examples"
        example_files = sorted(example_dir.glob("*.jpg")) if example_dir.exists() else []
        if example_files:
            gr.Examples(
                examples=[[str(f)] for f in example_files],
                inputs=[input_image],
                label="Example Aerial Images",
            )

        gr.Markdown(
            """
            ---
            ### About

            **EleFind** is an undergraduate research project for detecting
            elephants in aerial imagery.

            - **Model:** YOLOv11 trained on sliced 1024x1024 aerial patches
            - **Inference:** SAHI (Slicing Aided Hyper Inference) for high-res images
            - **XAI:** Gaussian density heatmaps showing detection focus areas
            - **Performance:** Precision 53.2% | Recall 49.1% | F1 51.0%

            [GitHub Repository](https://github.com/helithalochana/EleFind-gradio-ui)
            """
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
