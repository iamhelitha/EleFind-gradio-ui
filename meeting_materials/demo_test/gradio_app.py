"""
Elephant Detection Gradio Web UI
================================

A web interface for elephant detection using YOLOv11 with SAHI inference.
Features:
- Drag and drop image upload
- Detected elephants with bounding boxes
- Heatmap visualization showing detection density

Usage:
------
python gradio_app.py
"""

import gradio as gr
import numpy as np
import cv2
from pathlib import Path
from PIL import Image
import warnings
import uuid

warnings.filterwarnings("ignore")

# Import detection libraries
try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    import torch
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("\nPlease install:")
    print("  pip install sahi ultralytics opencv-python numpy gradio torch")
    exit(1)

# ============================================================================
# CONFIGURATION
# ============================================================================

script_dir = Path(__file__).parent
MODEL_PATH = script_dir.parent / "models" / "best.pt"

# SAHI Parameters (optimized for elephant detection)
SLICE_SIZE = 1024
OVERLAP_RATIO = 0.30
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.40

# ============================================================================
# MODEL LOADING
# ============================================================================

def get_device():
    """Detect best available device."""
    try:
        if torch.backends.mps.is_available():
            return 'mps'  # Apple Silicon
        elif torch.cuda.is_available():
            return 'cuda:0'  # NVIDIA GPU
        else:
            return 'cpu'
    except:
        return 'cpu'

def load_model():
    """Load the detection model."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

    device = get_device()
    print(f"Loading model on device: {device}")

    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(MODEL_PATH),
        confidence_threshold=CONF_THRESHOLD,
        device=device
    )
    return model

# Load model globally
print("Initializing Elephant Detection Model...")
model = load_model()
print("Model loaded successfully!")

# ============================================================================
# DETECTION FUNCTIONS
# ============================================================================

def run_detection(image):
    """Run SAHI detection on the input image."""
    print("DEBUG: Starting run_detection")
    # Convert PIL to numpy array if needed
    if isinstance(image, Image.Image):
        image_np = np.array(image)
    else:
        image_np = image

    # Ensure RGB format
    if len(image_np.shape) == 2:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2RGB)
    elif image_np.shape[2] == 4:
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2RGB)

    # Save temp image for SAHI using unique name to avoid race conditions
    unique_id = str(uuid.uuid4())
    temp_path = script_dir / f"temp_input_{unique_id}.jpg"
    print(f"DEBUG: Saving temp image to {temp_path}")
    write_ok = cv2.imwrite(str(temp_path), cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR))
    if not write_ok:
        raise RuntimeError(f"Failed to write temp image to {temp_path}")

    try:
        print("DEBUG: Calling get_sliced_prediction")
        # Run SAHI prediction
        result = get_sliced_prediction(
            image=str(temp_path),
            detection_model=model,
            slice_height=SLICE_SIZE,
            slice_width=SLICE_SIZE,
            overlap_height_ratio=OVERLAP_RATIO,
            overlap_width_ratio=OVERLAP_RATIO,
            postprocess_type="NMS",
            postprocess_match_threshold=IOU_THRESHOLD,
            verbose=0
        )
        print(f"DEBUG: get_sliced_prediction returned {len(result.object_prediction_list)} predictions")

        # Extract predictions
        predictions = []
        for obj in result.object_prediction_list:
            bbox = obj.bbox
            predictions.append({
                'x1': int(bbox.minx),
                'y1': int(bbox.miny),
                'x2': int(bbox.maxx),
                'y2': int(bbox.maxy),
                'confidence': obj.score.value
            })
        print(f"DEBUG: Processed {len(predictions)} predictions")

    finally:
        # Clean up temp file
        if temp_path.exists():
            temp_path.unlink()
            print("DEBUG: Temp file deleted")

    return image_np, predictions


def draw_detections(image, predictions):
    """Draw bounding boxes on the image."""
    if image is None or predictions is None:
        return None

    img_draw = image.copy()

    for pred in predictions:
        x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
        conf = pred['confidence']

        # Draw bounding box (green)
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Draw label background
        label = f"Elephant {conf:.0%}"
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img_draw, (x1, y1 - label_h - 10), (x1 + label_w + 5, y1), (0, 255, 0), -1)

        # Draw label text
        cv2.putText(img_draw, label, (x1 + 2, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    return img_draw


def create_heatmap(image, predictions):
    """Create a heatmap visualization showing detection density."""
    if image is None or predictions is None:
        return None

    h, w = image.shape[:2]

    # Create density map
    heatmap = np.zeros((h, w), dtype=np.float32)

    for pred in predictions:
        x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
        conf = pred['confidence']

        # Create a gaussian blob centered on the detection
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_w = x2 - x1
        box_h = y2 - y1

        # Gaussian radius based on box size
        sigma_x = box_w / 2
        sigma_y = box_h / 2

        # Create meshgrid for gaussian
        y_range = np.arange(max(0, y1 - box_h), min(h, y2 + box_h))
        x_range = np.arange(max(0, x1 - box_w), min(w, x2 + box_w))

        if len(x_range) > 0 and len(y_range) > 0:
            xx, yy = np.meshgrid(x_range, y_range)
            gaussian = np.exp(-((xx - center_x) ** 2 / (2 * sigma_x ** 2 + 1e-6) +
                               (yy - center_y) ** 2 / (2 * sigma_y ** 2 + 1e-6)))
            gaussian *= conf  # Weight by confidence

            # Add to heatmap
            heatmap[y_range[0]:y_range[-1]+1, x_range[0]:x_range[-1]+1] += gaussian

    # Normalize heatmap
    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Apply colormap
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    # Blend with original image
    alpha = 0.6
    blended = cv2.addWeighted(image, 1 - alpha, heatmap_colored, alpha, 0)

    # Draw detection boxes on heatmap too (white outlines)
    for pred in predictions:
        x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
        cv2.rectangle(blended, (x1, y1), (x2, y2), (255, 255, 255), 2)

    return blended


def _normalize_input_image(image):
    """Normalize Gradio image input across versions."""
    if image is None:
        return None

    if isinstance(image, Image.Image):
        return image

    if isinstance(image, np.ndarray):
        return Image.fromarray(image)

    if isinstance(image, str):
        return Image.open(image).convert("RGB")

    if isinstance(image, dict):
        path = image.get("path") or image.get("name")
        if path:
            return Image.open(path).convert("RGB")

    return None


def _to_pil(image_np):
    """Convert numpy image to PIL for Gradio output compatibility."""
    if image_np is None:
        return None
    if isinstance(image_np, Image.Image):
        return image_np
    return Image.fromarray(image_np.astype(np.uint8))


def process_image(image, progress=gr.Progress()):
    """Main processing function for Gradio interface."""
    print("DEBUG: process_image called")
    image = _normalize_input_image(image)
    if image is None:
        print("DEBUG: Image is None or unsupported type")
        return None, None, "Please upload an aerial image to detect elephants."

    try:
        progress(0.1, desc="Preparing image")

        # Run detection
        progress(0.2, desc="Running detection")
        image_np, predictions = run_detection(image)

        if image_np is None:
            print("DEBUG: image_np is None")
            return None, None, "Error processing image."
        
        print("DEBUG: Creating visualizations")
        progress(0.7, desc="Rendering outputs")
        # Create visualizations
        detection_image = draw_detections(image_np, predictions)
        heatmap_image = create_heatmap(image_np, predictions)
        print("DEBUG: Visualizations created")
        progress(0.95, desc="Finalizing")

    except Exception as e:
        import traceback
        print(f"DEBUG: Exception in process_image: {e}")
        traceback.print_exc()
        error_msg = f"Error during detection: {str(e)}\n\n```\n{traceback.format_exc()}\n```"
        return None, None, error_msg

    # Generate statistics
    num_detections = len(predictions)
    if num_detections > 0:
        avg_conf = sum(p['confidence'] for p in predictions) / num_detections
        confidences = [f"{p['confidence']:.0%}" for p in predictions]
        stats = f"""
### Detection Results

- **Elephants Detected:** {num_detections}
- **Average Confidence:** {avg_conf:.1%}
- **Individual Confidences:** {', '.join(confidences[:10])}{'...' if len(confidences) > 10 else ''}

### Model Configuration
- Slice Size: {SLICE_SIZE}x{SLICE_SIZE}
- Overlap Ratio: {OVERLAP_RATIO:.0%}
- Confidence Threshold: {CONF_THRESHOLD:.0%}
- IoU Threshold: {IOU_THRESHOLD:.0%}
"""
    else:
        stats = """
### Detection Results

- **Elephants Detected:** 0
- No elephants were detected in this image.

Try uploading an aerial image with elephants or adjusting the confidence threshold.
"""

    return _to_pil(detection_image), _to_pil(heatmap_image), stats


# ============================================================================
# GRADIO INTERFACE
# ============================================================================

# Custom CSS for better styling with lightbox zoom
custom_css = """
.gradio-container {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
}
.output-image {
    border-radius: 8px;
}

/* Make output images clickable with cursor */
#detection_img img, #heatmap_img img {
    cursor: zoom-in;
    transition: transform 0.2s ease;
}

#detection_img img:hover, #heatmap_img img:hover {
    opacity: 0.9;
}

/* Lightbox overlay */
.lightbox-overlay {
    position: fixed;
    top: 0;
    left: 0;
    width: 100vw;
    height: 100vh;
    background: rgba(0, 0, 0, 0.95);
    display: flex;
    justify-content: center;
    align-items: center;
    z-index: 9999;
    cursor: zoom-out;
    padding: 20px;
    box-sizing: border-box;
}

.lightbox-overlay img {
    max-width: 95vw;
    max-height: 95vh;
    object-fit: contain;
    border-radius: 4px;
    box-shadow: 0 0 30px rgba(255,255,255,0.1);
}

.lightbox-close {
    position: absolute;
    top: 20px;
    right: 30px;
    color: white;
    font-size: 40px;
    font-weight: bold;
    cursor: pointer;
    z-index: 10000;
}

.lightbox-close:hover {
    color: #ff6b6b;
}

.lightbox-hint {
    position: absolute;
    bottom: 20px;
    left: 50%;
    transform: translateX(-50%);
    color: rgba(255,255,255,0.7);
    font-size: 14px;
}
"""

# JavaScript for lightbox functionality
custom_js = """
function setupLightbox() {
    // Remove existing lightbox if any
    const existingLightbox = document.querySelector('.lightbox-overlay');
    if (existingLightbox) existingLightbox.remove();

    // Add click handlers to output images
    const imageContainers = ['#detection_img', '#heatmap_img'];

    imageContainers.forEach(selector => {
        const container = document.querySelector(selector);
        if (container) {
            const img = container.querySelector('img');
            if (img && !img.hasAttribute('data-lightbox-ready')) {
                img.setAttribute('data-lightbox-ready', 'true');
                img.addEventListener('click', function(e) {
                    e.preventDefault();
                    e.stopPropagation();
                    openLightbox(this.src);
                });
            }
        }
    });
}

function openLightbox(src) {
    // Create lightbox overlay
    const overlay = document.createElement('div');
    overlay.className = 'lightbox-overlay';

    // Create close button
    const closeBtn = document.createElement('span');
    closeBtn.className = 'lightbox-close';
    closeBtn.innerHTML = '&times;';
    closeBtn.onclick = closeLightbox;

    // Create image
    const img = document.createElement('img');
    img.src = src;

    // Create hint text
    const hint = document.createElement('div');
    hint.className = 'lightbox-hint';
    hint.textContent = 'Click anywhere or press ESC to close';

    overlay.appendChild(closeBtn);
    overlay.appendChild(img);
    overlay.appendChild(hint);
    overlay.onclick = closeLightbox;

    document.body.appendChild(overlay);

    // Add ESC key listener
    document.addEventListener('keydown', handleEscKey);
}

function closeLightbox() {
    const overlay = document.querySelector('.lightbox-overlay');
    if (overlay) overlay.remove();
    document.removeEventListener('keydown', handleEscKey);
}

function handleEscKey(e) {
    if (e.key === 'Escape') closeLightbox();
}

// Setup on load and after updates
if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', setupLightbox);
} else {
    setupLightbox();
}

// Re-setup when Gradio updates the DOM
const observer = new MutationObserver(function(mutations) {
    setupLightbox();
});

observer.observe(document.body, { childList: true, subtree: true });
"""

def build_aerial_xai_interface():
    """Build the Gradio UI for aerial-image XAI detection."""
    with gr.Blocks(css=custom_css, js=custom_js, title="Aerial XAI Elephant Detection") as ui:
        gr.Markdown("""
        # Aerial XAI Elephant Detection

        Upload an **aerial image** and run the detection model to produce:
        - a **labeled image** with bounding boxes, and
        - an **XAI heatmap** highlighting detection focus areas.

        ---
        """)

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(
                    label="Upload Aerial Image",
                    type="pil",
                    height=400,
                    sources=["upload", "clipboard"],
                )

                detect_btn = gr.Button("Run XAI Detection", variant="primary", size="lg")

                gr.Markdown("""
                **Tips:**
                - Works best with high-resolution aerial/drone imagery
                - Optimal image size: 5472x3648 or similar
                - Supports JPG, PNG formats
                """)

        with gr.Row():
            with gr.Column(scale=1):
                detection_output = gr.Image(
                    label="Labeled Image (Click image to zoom)",
                    type="pil",
                    height=450,
                    interactive=False,
                    elem_id="detection_img",
                )

            with gr.Column(scale=1):
                heatmap_output = gr.Image(
                    label="XAI Heatmap (Click image to zoom)",
                    type="pil",
                    height=450,
                    interactive=False,
                    elem_id="heatmap_img",
                )

        stats_output = gr.Markdown(label="Statistics")

        detect_btn.click(
            fn=process_image,
            inputs=[input_image],
            outputs=[detection_output, heatmap_output, stats_output],
            concurrency_limit=1
        )

        gr.Markdown("""
        ---

        ### About

        This system uses:
        - **YOLOv11** trained on sliced aerial elephant imagery
        - **SAHI** for handling high-resolution images through slicing
        - **MPS/CUDA/CPU** automatic device detection

        Model trained for aerial elephant detection from drone imagery.
        """)

    return ui


# Create the Gradio interface
demo = build_aerial_xai_interface()

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("ELEPHANT DETECTION WEB UI")
    print("="*60)
    print(f"Model: {MODEL_PATH}")
    print(f"Device: {get_device()}")
    print("="*60 + "\n")

    # Launch the app
    demo.queue(max_size=10)
    demo.launch(
        server_name="127.0.0.1",
        # server_port=7860, # Let Gradio find an available port
        share=False,
        show_error=True,
        inbrowser=True
    )
