"""
EleFind - Aerial Elephant Detection with Explainable AI
========================================================

A Gradio 5/6 web interface for detecting elephants in aerial/drone imagery
using YOLOv11 with SAHI (Slicing Aided Hyper Inference).

Features:
- Upload aerial images and detect elephants with bounding boxes
- XAI heatmap visualization showing detection density
- Adjustable SAHI parameters (confidence, slice size, overlap)
- Automatic model download from HuggingFace Hub
- Confidence bar chart and detection data table
- Tabbed output with download buttons on every result image

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
    enable_heatmap: bool = False,
    progress=gr.Progress(),
):
    """Run detection pipeline and return annotated image, heatmap, stats HTML,
    confidence chart data, and detection table data."""

    image_np = _to_numpy_rgb(image)
    if image_np is None:
        return None, None, "<p style='color:#ef4444;padding:16px;'>Please upload an aerial image to detect elephants.</p>", None, None

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

        heatmap_image = None
        if enable_heatmap:
            progress(0.90, desc="Generating XAI heatmap")
            heatmap_image = create_heatmap(image_np, predictions)

    except Exception as e:
        import traceback
        err_html = (
            f"<div style='color:#ef4444;padding:16px;'>"
            f"<strong>Error during detection:</strong><br><pre>{e}</pre>"
            f"<details><summary>Traceback</summary><pre>{traceback.format_exc()}</pre></details>"
            f"</div>"
        )
        return None, None, err_html, None, None

    stats = _stats_html(predictions, w, h, slice_size, overlap_ratio,
                        conf_threshold, iou_threshold)

    progress(1.0, desc="Done")
    heatmap_out = Image.fromarray(heatmap_image.astype(np.uint8)) if heatmap_image is not None else None

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
        heatmap_out,
        stats,
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
).set(
    button_primary_background_fill="*primary_500",
    button_primary_background_fill_hover="*primary_600",
    button_primary_text_color="white",
    block_label_text_weight="600",
    block_title_text_weight="700",
)

CSS = """
/* ── Global container ──────────────────────────────────────── */
.gradio-container {
    max-width: 1200px !important;
    margin: 0 auto !important;
}

/* ── Hero banner ────────────────────────────────────────────── */
.elefind-hero {
    background: linear-gradient(135deg, #064e3b 0%, #065f46 40%, #0d9488 100%);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 8px;
    color: white;
    position: relative;
    overflow: hidden;
}
.elefind-hero::before {
    content: "🐘";
    position: absolute;
    right: 32px;
    top: 50%;
    transform: translateY(-50%);
    font-size: 80px;
    opacity: 0.18;
}
.elefind-hero h1 {
    font-size: 2rem !important;
    font-weight: 800 !important;
    margin: 0 0 6px 0 !important;
    color: white !important;
}
.elefind-hero p {
    font-size: 1rem;
    opacity: 0.88;
    margin: 0;
    max-width: 640px;
}
.elefind-hero .badge {
    display: inline-block;
    background: rgba(255,255,255,0.15);
    border: 1px solid rgba(255,255,255,0.3);
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    margin-right: 6px;
    margin-top: 12px;
}

/* ── Stat cards ──────────────────────────────────────────────── */
.stat-cards {
    display: flex;
    gap: 12px;
    flex-wrap: wrap;
    margin: 4px 0;
}
.stat-card {
    flex: 1;
    min-width: 120px;
    background: var(--background-fill-primary, #f9fafb);
    border: 1px solid var(--border-color-primary, #e5e7eb);
    border-radius: 12px;
    padding: 16px 20px;
    text-align: center;
}
.stat-card .value {
    font-size: 2rem;
    font-weight: 800;
    color: #065f46;
    line-height: 1;
    display: block;
}
.stat-card .label {
    font-size: 0.78rem;
    color: var(--body-text-color-subdued, #6b7280);
    margin-top: 4px;
    display: block;
}
.stat-card.highlight .value { color: #0d9488; }

/* ── Param card ──────────────────────────────────────────────── */
.param-card {
    background: var(--background-fill-secondary, #f3f4f6);
    border-radius: 10px;
    padding: 12px 16px;
    margin-top: 8px;
    font-size: 0.85rem;
    line-height: 1.8;
}
.param-card strong { color: #065f46; }

/* ── Tips box ────────────────────────────────────────────────── */
.tips-box {
    background: #ecfdf5;
    border-left: 4px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: 10px 14px;
    font-size: 0.84rem;
    line-height: 1.7;
    color: #064e3b;
}

/* ── Tab active indicator ──────────────────────────────────── */
.gradio-tabs .tab-nav button.selected {
    border-bottom-color: #10b981 !important;
    color: #065f46 !important;
    font-weight: 700 !important;
}

/* ── About accordion ──────────────────────────────────────── */
.about-section { font-size: 0.88rem; line-height: 1.7; }
.about-section a { color: #10b981; }

/* ── Hide empty heatmap placeholder ──────────────────────── */
.heatmap-placeholder {
    background: var(--background-fill-secondary, #f3f4f6);
    border-radius: 12px;
    padding: 48px 24px;
    text-align: center;
    color: var(--body-text-color-subdued, #9ca3af);
    font-size: 0.9rem;
}
"""

# ---------------------------------------------------------------------------
# Helper: build stats HTML
# ---------------------------------------------------------------------------
def _stats_html(predictions: list[dict], w: int, h: int, slice_size: int,
                overlap_ratio: float, conf_threshold: float,
                iou_threshold: float) -> str:
    n = len(predictions)
    avg_conf = sum(p["confidence"] for p in predictions) / n if n else 0.0
    max_conf = max((p["confidence"] for p in predictions), default=0.0)
    min_conf = min((p["confidence"] for p in predictions), default=0.0)

    if n > 0:
        cards_html = f"""
        <div class="stat-cards">
          <div class="stat-card">
            <span class="value">{n}</span>
            <span class="label">Elephants<br>Detected</span>
          </div>
          <div class="stat-card highlight">
            <span class="value">{avg_conf:.0%}</span>
            <span class="label">Average<br>Confidence</span>
          </div>
          <div class="stat-card">
            <span class="value">{max_conf:.0%}</span>
            <span class="label">Highest<br>Confidence</span>
          </div>
          <div class="stat-card">
            <span class="value">{min_conf:.0%}</span>
            <span class="label">Lowest<br>Confidence</span>
          </div>
        </div>
        """
        conf_list = ", ".join(f'<code>{p["confidence"]:.0%}</code>' for p in predictions[:20])
        if n > 20:
            conf_list += f" <em>+ {n - 20} more</em>"
        det_detail = f"""
        <p style="margin:12px 0 4px; font-size:0.85rem; color:var(--body-text-color-subdued)">
          Individual confidences: {conf_list}
        </p>
        """
    else:
        cards_html = """
        <div class="stat-cards">
          <div class="stat-card">
            <span class="value">0</span>
            <span class="label">Elephants<br>Detected</span>
          </div>
        </div>
        <p style="margin:12px 0 4px; color:#6b7280; font-size:0.88rem;">
          No elephants found — try lowering the confidence threshold or upload
          an aerial image with visible elephants.
        </p>
        """
        det_detail = ""

    param_html = f"""
    <div class="param-card">
      <strong>Parameters used</strong><br>
      Slice&nbsp;{int(slice_size)}&thinsp;×&thinsp;{int(slice_size)}&nbsp;px
      &nbsp;·&nbsp; Overlap&nbsp;{overlap_ratio:.0%}
      &nbsp;·&nbsp; Confidence&nbsp;≥&nbsp;{conf_threshold:.0%}
      &nbsp;·&nbsp; IoU&nbsp;{iou_threshold:.0%}
      &nbsp;·&nbsp; Image&nbsp;{w}&thinsp;×&thinsp;{h}&nbsp;px
      &nbsp;·&nbsp; Device&nbsp;<code>{DEVICE}</code>
    </div>
    """

    return cards_html + det_detail + param_html


# ---------------------------------------------------------------------------
# Updated process_image (returns 5 outputs)
# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    """Construct the Gradio Blocks interface (Gradio 6.x)."""

    with gr.Blocks(
        css=CSS,
        theme=_THEME,
        title="EleFind – Aerial Elephant Detection",
        fill_width=False,
    ) as demo:

        # ── Hero banner ─────────────────────────────────────────────────────
        gr.HTML(
            """
            <div class="elefind-hero">
              <h1>🐘 EleFind</h1>
              <p>Aerial elephant detection powered by <strong>YOLOv11 + SAHI</strong>
                 (Slicing Aided Hyper Inference). Upload a drone or satellite image
                 and get instant detection results with XAI heatmaps.</p>
              <span class="badge">YOLOv11</span>
              <span class="badge">SAHI</span>
              <span class="badge">XAI Heatmaps</span>
              <span class="badge">Conservation AI</span>
            </div>
            """
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
                    show_download_button=False,
                    height=320,
                )

                with gr.Accordion("⚙️ SAHI Detection Parameters", open=False):
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

                heatmap_toggle = gr.Checkbox(
                    label="Generate XAI Density Heatmap",
                    value=False,
                    info="Overlay a Gaussian density map on the detection image",
                )

                detect_btn = gr.Button(
                    "🔍 Detect Elephants",
                    variant="primary",
                    size="lg",
                )

                gr.HTML(
                    """
                    <div class="tips-box">
                      <strong>Tips for best results</strong><br>
                      • Use high-resolution aerial / drone images (≥ 4K)<br>
                      • Optimal source resolution: ~5472 × 3648 px<br>
                      • Lower confidence threshold → more detections (noisier)<br>
                      • Increase slice size for larger, spread-out herds
                    </div>
                    """
                )

            # ── RIGHT: Output tabs ────────────────────────────────────────
            with gr.Column(scale=6, min_width=400):

                with gr.Tabs() as result_tabs:

                    # ── Tab 1: Detection image ────────────────────────────
                    with gr.Tab("📷 Detections", id="tab_det"):
                        detection_output = gr.Image(
                            label="Annotated detections",
                            type="pil",
                            interactive=False,
                            show_fullscreen_button=True,
                            show_download_button=True,
                            height=420,
                        )

                    # ── Tab 2: XAI Heatmap ────────────────────────────────
                    with gr.Tab("🌡️ XAI Heatmap", id="tab_hm"):
                        heatmap_output = gr.Image(
                            label="Gaussian density heatmap",
                            type="pil",
                            interactive=False,
                            show_fullscreen_button=True,
                            show_download_button=True,
                            height=420,
                        )
                        gr.HTML(
                            """
                            <p style="font-size:0.8rem; color:#6b7280; margin:4px 0 0;">
                              Enable <em>Generate XAI Density Heatmap</em> in the left panel
                              then run detection to see the heatmap.
                            </p>
                            """
                        )

                    # ── Tab 3: Statistics ─────────────────────────────────
                    with gr.Tab("📊 Statistics", id="tab_stats"):
                        stats_html_out = gr.HTML(
                            value="<p style='color:#9ca3af;padding:24px;text-align:center;'>"
                                  "Run detection to see statistics.</p>"
                        )

                        with gr.Accordion("📋 Detection Table", open=True):
                            det_table_out = gr.Dataframe(
                                headers=["ID", "Confidence", "BBox (x1,y1,x2,y2)",
                                         "Width (px)", "Height (px)"],
                                label=None,
                                interactive=False,
                                wrap=True,
                            )

                        with gr.Accordion("📈 Confidence Chart", open=True):
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
            with gr.Accordion("🖼️ Example Aerial Images", open=True):
                gr.Examples(
                    examples=[[str(f)] for f in example_files],
                    inputs=[input_image],
                    label=None,
                )

        # ── About accordion ────────────────────────────────────────────────
        with gr.Accordion("ℹ️ About EleFind", open=False):
            gr.HTML(
                """
                <div class="about-section">
                  <p><strong>EleFind</strong> is an undergraduate research project for
                  automated elephant detection in aerial imagery to support wildlife
                  conservation efforts.</p>
                  <ul>
                    <li><strong>Model:</strong> YOLOv11 trained on sliced 1024 × 1024
                        aerial patches</li>
                    <li><strong>Inference:</strong> SAHI – tiled inference for
                        high-resolution images without GPU memory overflow</li>
                    <li><strong>XAI:</strong> Gaussian density heatmaps highlight
                        detection hotspot areas</li>
                    <li><strong>Performance:</strong> Precision&nbsp;53.2 %
                        &nbsp;|&nbsp; Recall&nbsp;49.1 %
                        &nbsp;|&nbsp; F1&nbsp;51.0 %</li>
                  </ul>
                  <p>
                    <a href="https://github.com/helithalochana/EleFind-gradio-ui"
                       target="_blank">GitHub Repository</a>
                    &nbsp;·&nbsp;
                    <a href="https://huggingface.co/iamhelitha/EleFind-yolo11-elephant"
                       target="_blank">Model on HuggingFace</a>
                  </p>
                </div>
                """
            )

        # ── Event wiring ───────────────────────────────────────────────────
        _outputs = [
            detection_output,
            heatmap_output,
            stats_html_out,
            conf_chart_out,
            det_table_out,
        ]

        def _run(image, conf, ssize, overlap, iou, hm, progress=gr.Progress()):
            det_img, heatmap_img, raw_stats, conf_data, table_data = process_image(
                image, conf, ssize, overlap, iou, hm, progress
            )
            return det_img, heatmap_img, raw_stats, conf_data, table_data

        detect_btn.click(
            fn=_run,
            inputs=[
                input_image,
                conf_slider,
                slice_slider,
                overlap_slider,
                iou_slider,
                heatmap_toggle,
            ],
            outputs=_outputs,
            concurrency_limit=1,
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
