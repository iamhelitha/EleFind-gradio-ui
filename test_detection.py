"""
EleFind - Detection Pipeline Test Suite
========================================

Tests for the YOLO + SAHI elephant detection pipeline.

Usage:
    pytest test_detection.py -v                # run all tests
    pytest test_detection.py -v -m "not slow"  # skip slow tests
    pytest test_detection.py -v -k "test_model" # run model tests only

Requires:
    - best.pt model file (in models/ or meeting_materials/models/)
    - pip install pytest
"""

import os
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Ensure project root is importable
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

MODEL_PATHS = [
    PROJECT_ROOT / "best.pt",
    PROJECT_ROOT / "models" / "best.pt",
    PROJECT_ROOT / "meeting_materials" / "models" / "best.pt",
]


def _model_available() -> bool:
    """Check if a model file is available locally."""
    return any(p.exists() for p in MODEL_PATHS)


def _find_model_path() -> str:
    """Return the first available model path."""
    for p in MODEL_PATHS:
        if p.exists():
            return str(p)
    pytest.skip("No model file found (best.pt). Skipping model tests.")


def _get_test_image_dir() -> Path | None:
    """Find the test images directory."""
    d = PROJECT_ROOT / "meeting_materials" / "demo_test" / "images"
    return d if d.exists() else None


def _get_test_labels_dir() -> Path | None:
    """Find the test labels directory."""
    d = PROJECT_ROOT / "meeting_materials" / "demo_test" / "labels"
    return d if d.exists() else None


# ---------------------------------------------------------------------------
# Synthetic image helpers
# ---------------------------------------------------------------------------

def make_blank_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a blank RGB numpy image."""
    return np.zeros((height, width, 3), dtype=np.uint8)


def make_random_image(width: int = 640, height: int = 480) -> np.ndarray:
    """Create a random noise RGB numpy image."""
    rng = np.random.default_rng(42)
    return rng.integers(0, 256, (height, width, 3), dtype=np.uint8)


# ===========================================================================
# Unit tests: utility functions (no model needed)
# ===========================================================================


class TestImageValidation:
    """Tests for image validation and normalisation helpers."""

    def test_validate_small_image_unchanged(self):
        """Small images should pass through without resizing."""
        from app import validate_image

        img = make_random_image(800, 600)
        result = validate_image(img)
        assert result.shape == (600, 800, 3)

    def test_validate_large_image_resized(self):
        """Images exceeding MAX_IMAGE_DIMENSION should be downscaled."""
        from app import validate_image, MAX_IMAGE_DIMENSION

        # Create an oversized image
        big = make_random_image(7000, 5000)
        result = validate_image(big)
        assert max(result.shape[:2]) <= MAX_IMAGE_DIMENSION

    def test_to_numpy_rgb_from_pil(self):
        """PIL Image should convert to numpy RGB."""
        from app import _to_numpy_rgb

        pil_img = Image.fromarray(make_random_image(100, 100))
        result = _to_numpy_rgb(pil_img)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_to_numpy_rgb_from_ndarray(self):
        """Numpy array should pass through."""
        from app import _to_numpy_rgb

        arr = make_random_image(100, 100)
        result = _to_numpy_rgb(arr)
        assert isinstance(result, np.ndarray)
        assert result.shape == (100, 100, 3)

    def test_to_numpy_rgb_from_rgba(self):
        """RGBA array should be converted to RGB."""
        from app import _to_numpy_rgb

        rgba = np.zeros((100, 100, 4), dtype=np.uint8)
        result = _to_numpy_rgb(rgba)
        assert result.shape == (100, 100, 3)

    def test_to_numpy_rgb_from_grayscale(self):
        """Grayscale array should be converted to RGB."""
        from app import _to_numpy_rgb

        gray = np.zeros((100, 100), dtype=np.uint8)
        result = _to_numpy_rgb(gray)
        assert result.shape == (100, 100, 3)

    def test_to_numpy_rgb_none(self):
        """None input should return None."""
        from app import _to_numpy_rgb

        assert _to_numpy_rgb(None) is None


class TestDrawDetections:
    """Tests for drawing bounding boxes on images."""

    def test_no_predictions(self):
        """Empty predictions should return a clean copy of the image."""
        from app import draw_detections

        img = make_random_image(200, 200)
        result = draw_detections(img, [])
        assert result.shape == img.shape
        # Should be a copy, not the same object
        assert result is not img

    def test_single_prediction(self):
        """A single prediction should draw a box without errors."""
        from app import draw_detections

        img = make_random_image(500, 500)
        preds = [{"x1": 50, "y1": 50, "x2": 150, "y2": 150, "confidence": 0.85}]
        result = draw_detections(img, preds)
        assert result.shape == img.shape
        # The image should be modified (green pixels added)
        assert not np.array_equal(result, img)

    def test_multiple_predictions(self):
        """Multiple predictions should all be drawn."""
        from app import draw_detections

        img = make_random_image(500, 500)
        preds = [
            {"x1": 10, "y1": 10, "x2": 100, "y2": 100, "confidence": 0.90},
            {"x1": 200, "y1": 200, "x2": 300, "y2": 300, "confidence": 0.75},
            {"x1": 350, "y1": 350, "x2": 450, "y2": 450, "confidence": 0.60},
        ]
        result = draw_detections(img, preds)
        assert result.shape == img.shape


class TestHeatmap:
    """Tests for the heatmap generation."""

    def test_heatmap_no_predictions(self):
        """Heatmap with no predictions should still return a valid image."""
        from app import create_heatmap

        img = make_random_image(200, 200)
        result = create_heatmap(img, [])
        assert result.shape == img.shape

    def test_heatmap_with_predictions(self):
        """Heatmap with predictions should produce a blended image."""
        from app import create_heatmap

        img = make_random_image(500, 500)
        preds = [{"x1": 100, "y1": 100, "x2": 200, "y2": 200, "confidence": 0.8}]
        result = create_heatmap(img, preds)
        assert result.shape == img.shape
        # Should differ from original
        assert not np.array_equal(result, img)


# ===========================================================================
# Integration tests: require the model file (best.pt)
# ===========================================================================

@pytest.mark.skipif(not _model_available(), reason="Model file (best.pt) not found")
class TestModelLoading:
    """Tests that require the actual model weights."""

    def test_model_loads_successfully(self):
        """The SAHI model should load without errors."""
        from sahi import AutoDetectionModel

        model_path = _find_model_path()
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.30,
            device="cpu",
        )
        assert model is not None

    def test_model_type_is_yolov8(self):
        """SAHI should recognise the model as yolov8 type."""
        from sahi import AutoDetectionModel

        model_path = _find_model_path()
        model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.30,
            device="cpu",
        )
        assert model.type == "yolov8"


@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="Model file (best.pt) not found")
class TestInference:
    """End-to-end inference tests (slower, require model)."""

    @pytest.fixture(autouse=True)
    def _load_model(self):
        """Load model once for all tests in this class."""
        from sahi import AutoDetectionModel

        model_path = _find_model_path()
        self.model = AutoDetectionModel.from_pretrained(
            model_type="yolov8",
            model_path=model_path,
            confidence_threshold=0.25,
            device="cpu",
        )

    def test_inference_on_blank_image(self):
        """Inference on blank image should return zero detections."""
        from app import run_detection

        blank = make_blank_image(640, 480)
        preds = run_detection(blank, conf_threshold=0.30, slice_size=640)
        assert isinstance(preds, list)
        # Blank image should have 0 or very few spurious detections
        assert len(preds) <= 5

    def test_inference_on_random_noise(self):
        """Inference on random noise should not crash."""
        from app import run_detection

        noise = make_random_image(640, 480)
        preds = run_detection(noise, conf_threshold=0.50, slice_size=640)
        assert isinstance(preds, list)

    def test_prediction_format(self):
        """Each prediction should have required keys."""
        from app import run_detection

        img = make_random_image(640, 480)
        preds = run_detection(img, conf_threshold=0.10, slice_size=640)
        required_keys = {"x1", "y1", "x2", "y2", "confidence"}

        for pred in preds:
            assert required_keys.issubset(pred.keys()), (
                f"Prediction missing keys: {required_keys - set(pred.keys())}"
            )
            assert 0 <= pred["confidence"] <= 1.0
            assert pred["x2"] > pred["x1"]
            assert pred["y2"] > pred["y1"]

    @pytest.mark.parametrize(
        "size",
        [(320, 240), (640, 480), (1024, 768)],
        ids=["small", "medium", "large"],
    )
    def test_various_image_sizes(self, size):
        """Inference should work on various image dimensions."""
        from app import run_detection

        w, h = size
        img = make_random_image(w, h)
        preds = run_detection(img, conf_threshold=0.50, slice_size=min(w, 512))
        assert isinstance(preds, list)

    def test_confidence_threshold_filtering(self):
        """Higher confidence threshold should produce fewer/equal detections."""
        from app import run_detection

        img = make_random_image(640, 480)

        preds_low = run_detection(img, conf_threshold=0.10, slice_size=640)
        preds_high = run_detection(img, conf_threshold=0.80, slice_size=640)

        assert len(preds_high) <= len(preds_low)


@pytest.mark.slow
@pytest.mark.skipif(
    not _model_available() or _get_test_image_dir() is None,
    reason="Model or test images not found",
)
class TestRealImages:
    """Integration tests using actual aerial test images."""

    def test_detection_on_real_image(self):
        """Run detection on a real aerial image with known elephants."""
        from app import run_detection

        img_dir = _get_test_image_dir()
        images = sorted(img_dir.glob("*.jpg"))[:1]  # Test first image only

        for img_path in images:
            img = np.array(Image.open(img_path).convert("RGB"))
            preds = run_detection(
                img, conf_threshold=0.30, slice_size=1024, overlap_ratio=0.30
            )
            assert isinstance(preds, list)
            # Real aerial images should produce at least some detections
            # (not asserting count since it depends on the specific image)

    def test_detection_with_ground_truth(self):
        """Compare detections against ground truth labels."""
        img_dir = _get_test_image_dir()
        labels_dir = _get_test_labels_dir()

        if labels_dir is None:
            pytest.skip("Labels directory not found")

        from app import run_detection

        # Pick an image that has ground truth labels
        label_files = sorted(labels_dir.glob("*.txt"))
        if not label_files:
            pytest.skip("No label files found")

        # Use first labelled image
        label_file = label_files[0]
        img_path = img_dir / f"{label_file.stem}.jpg"
        if not img_path.exists():
            pytest.skip(f"Image {img_path.name} not found")

        # Count ground truth objects
        with open(label_file) as f:
            gt_count = sum(1 for line in f if line.strip())

        img = np.array(Image.open(img_path).convert("RGB"))
        preds = run_detection(
            img, conf_threshold=0.25, slice_size=1024, overlap_ratio=0.30
        )

        # We expect at least *some* detections if there are ground truths
        if gt_count > 0:
            assert len(preds) > 0, (
                f"Expected detections for {img_path.name} "
                f"(ground truth has {gt_count} objects)"
            )


# ===========================================================================
# End-to-end pipeline test
# ===========================================================================

@pytest.mark.slow
@pytest.mark.skipif(not _model_available(), reason="Model file (best.pt) not found")
class TestEndToEnd:
    """Full pipeline test: image -> detection -> visualisation -> stats."""

    def test_full_pipeline(self):
        """Test the complete process_image function."""
        from app import process_image

        img = Image.fromarray(make_random_image(640, 480))

        det_img, heatmap_img, stats = process_image(
            image=img,
            conf_threshold=0.30,
            slice_size=640,
            overlap_ratio=0.20,
            iou_threshold=0.40,
        )

        # Stats should always be a non-empty string
        assert isinstance(stats, str)
        assert len(stats) > 0
        assert "Detection Results" in stats

        # Images can be None if no detections, but should be PIL or None
        if det_img is not None:
            assert isinstance(det_img, Image.Image)
        if heatmap_img is not None:
            assert isinstance(heatmap_img, Image.Image)


# ===========================================================================
# Configuration
# ===========================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
