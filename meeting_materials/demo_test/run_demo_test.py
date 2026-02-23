"""
Demo Test Script for Meeting Materials
=======================================

This script runs elephant detection on the demo test images using the trained model.
Designed to be portable and work on different systems (Windows/Mac/Linux).

For AI Agents:
--------------
This is a self-contained demo that:
1. Loads the best trained YOLOv11 model (best.pt in ../models/)
2. Runs SAHI inference on 50 demo images
3. Generates evaluation metrics and visualizations
4. All paths are relative to this script's location

Usage:
------
python run_demo_test.py
"""

import sys
from pathlib import Path
import json
from datetime import datetime

# Add project root to path for imports
script_dir = Path(__file__).parent
project_root = script_dir.parent.parent
sys.path.insert(0, str(project_root / "scripts"))
sys.path.append(str(project_root))

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    from sahi.utils.cv import read_image
    import cv2
    import numpy as np
except ImportError as e:
    print(f"ERROR: Missing required packages: {e}")
    print("\nPlease install:")
    print("  pip install sahi ultralytics opencv-python numpy")
    sys.exit(1)

# ============================================================================
# CONFIGURATION - Adjust these paths if needed
# ============================================================================

# Relative paths from this script's location
MODEL_PATH = script_dir.parent / "models" / "best.pt"
TEST_IMAGES_DIR = script_dir / "images"
TEST_LABELS_DIR = script_dir / "labels"
OUTPUT_DIR = script_dir / "results"

# SAHI Parameters (optimized)
SLICE_SIZE = 1024
OVERLAP_RATIO = 0.30
CONF_THRESHOLD = 0.30
IOU_THRESHOLD = 0.40

# ============================================================================


def load_ground_truth(image_path: Path, labels_dir: Path) -> list:
    """Load YOLO format ground truth labels."""
    label_file = labels_dir / f"{image_path.stem}.txt"
    
    if not label_file.exists():
        return []
    
    # Read image to get dimensions
    image = cv2.imread(str(image_path))
    img_h, img_w = image.shape[:2]
    
    ground_truths = []
    with open(label_file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id, x_center, y_center, width, height = map(float, parts[:5])
                
                # Convert normalized to absolute
                x1 = max(0, min((x_center - width / 2) * img_w, img_w))
                y1 = max(0, min((y_center - height / 2) * img_h, img_h))
                x2 = max(0, min((x_center + width / 2) * img_w, img_w))
                y2 = max(0, min((y_center + height / 2) * img_h, img_h))
                
                ground_truths.append({
                    'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2,
                    'class': int(cls_id)
                })
    
    return ground_truths


def calculate_iou(box1: dict, box2: dict) -> float:
    """Calculate IoU between two boxes."""
    x1 = max(box1['x1'], box2['x1'])
    y1 = max(box1['y1'], box2['y1'])
    x2 = min(box1['x2'], box2['x2'])
    y2 = min(box1['y2'], box2['y2'])
    
    if x2 < x1 or y2 < y1:
        return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    area1 = (box1['x2'] - box1['x1']) * (box1['y2'] - box1['y1'])
    area2 = (box2['x2'] - box2['x1']) * (box2['y2'] - box2['y1'])
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def evaluate_image(image_path: Path, model, labels_dir: Path) -> dict:
    """Run inference and evaluate on single image."""
    # Load image
    image = read_image(str(image_path))
    img_h, img_w = image.shape[:2]
    
    # Run SAHI prediction
    result = get_sliced_prediction(
        image=str(image_path),
        detection_model=model,
        slice_height=SLICE_SIZE,
        slice_width=SLICE_SIZE,
        overlap_height_ratio=OVERLAP_RATIO,
        overlap_width_ratio=OVERLAP_RATIO,
        postprocess_type="NMS",
        postprocess_match_threshold=IOU_THRESHOLD,
        verbose=0
    )
    
    # Extract predictions
    predictions = []
    for obj in result.object_prediction_list:
        bbox = obj.bbox
        predictions.append({
            'x1': bbox.minx, 'y1': bbox.miny,
            'x2': bbox.maxx, 'y2': bbox.maxy,
            'confidence': obj.score.value,
            'class': obj.category.id
        })
    
    # Load ground truth
    ground_truths = load_ground_truth(image_path, labels_dir)
    
    # Match predictions to ground truths
    matched_gt = set()
    matched_pred = set()
    
    for i, pred in enumerate(predictions):
        best_iou = 0
        best_gt_idx = -1
        
        for j, gt in enumerate(ground_truths):
            if j in matched_gt:
                continue
            iou = calculate_iou(pred, gt)
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = j
        
        if best_iou >= 0.5 and best_gt_idx not in matched_gt:
            matched_gt.add(best_gt_idx)
            matched_pred.add(i)
    
    tp = len(matched_pred)
    fp = len(predictions) - tp
    fn = len(ground_truths) - tp
    
    return {
        'image': image_path.name,
        'width': img_w,
        'height': img_h,
        'predictions': len(predictions),
        'ground_truths': len(ground_truths),
        'tp': tp, 'fp': fp, 'fn': fn
    }


def main():
    """Run demo evaluation."""
    print("="*80)
    print("ELEPHANT DETECTION DEMO TEST")
    print("="*80)
    print()
    
    # Verify paths
    print("📁 Configuration:")
    print(f"  Script location: {script_dir}")
    print(f"  Model: {MODEL_PATH}")
    print(f"  Test images: {TEST_IMAGES_DIR}")
    print(f"  Test labels: {TEST_LABELS_DIR}")
    print(f"  Output: {OUTPUT_DIR}")
    print()
    
    # Check if paths exist
    if not MODEL_PATH.exists():
        print(f"❌ ERROR: Model not found at {MODEL_PATH}")
        print("   Make sure best.pt is in the ../models/ folder")
        sys.exit(1)
    
    if not TEST_IMAGES_DIR.exists():
        print(f"❌ ERROR: Test images not found at {TEST_IMAGES_DIR}")
        sys.exit(1)
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("🔧 Loading model...")

    # Detect best available device (MPS for Apple Silicon, CUDA for NVIDIA, or CPU)
    try:
        import torch
        if torch.backends.mps.is_available():
            device = 'mps'  # Apple Silicon (M1/M2/M3)
        elif torch.cuda.is_available():
            device = 'cuda:0'  # NVIDIA GPU
        else:
            device = 'cpu'
    except:
        device = 'cpu'

    print(f"   Device selected: {device}")
    
    model = AutoDetectionModel.from_pretrained(
        model_type='yolov8',
        model_path=str(MODEL_PATH),
        confidence_threshold=CONF_THRESHOLD,
        device=device
    )
    print(f"✅ Model loaded successfully")
    print()
    
    print("📊 SAHI Configuration:")
    print(f"  Slice size: {SLICE_SIZE}×{SLICE_SIZE}")
    print(f"  Overlap ratio: {OVERLAP_RATIO}")
    print(f"  Confidence threshold: {CONF_THRESHOLD}")
    print(f"  IoU threshold: {IOU_THRESHOLD}")
    print()
    
    # Get test images
    image_files = sorted(list(TEST_IMAGES_DIR.glob('*.jpg')))
    print(f"🖼️  Found {len(image_files)} test images")
    print()
    
    # Run evaluation
    print("🚀 Running evaluation...")
    results = []
    total_tp, total_fp, total_fn = 0, 0, 0
    
    for i, img_path in enumerate(image_files, 1):
        print(f"  [{i}/{len(image_files)}] {img_path.name}", end='\r')
        
        result = evaluate_image(img_path, model, TEST_LABELS_DIR)
        results.append(result)
        
        total_tp += result['tp']
        total_fp += result['fp']
        total_fn += result['fn']
    
    print()  # New line after progress
    
    # Calculate overall metrics
    precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Print results
    print()
    print("="*80)
    print("RESULTS")
    print("="*80)
    print(f"Images tested: {len(image_files)}")
    print(f"True Positives:  {total_tp}")
    print(f"False Positives: {total_fp}")
    print(f"False Negatives: {total_fn}")
    print()
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print("="*80)
    
    # Save results
    summary = {
        'timestamp': datetime.now().isoformat(),
        'configuration': {
            'model': str(MODEL_PATH.name),
            'slice_size': SLICE_SIZE,
            'overlap_ratio': OVERLAP_RATIO,
            'conf_threshold': CONF_THRESHOLD,
            'iou_threshold': IOU_THRESHOLD
        },
        'results': {
            'images_tested': len(image_files),
            'true_positives': total_tp,
            'false_positives': total_fp,
            'false_negatives': total_fn,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        },
        'per_image_results': results
    }
    
    output_file = OUTPUT_DIR / "demo_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print()
    print(f"✅ Results saved to: {output_file}")
    print()


if __name__ == "__main__":
    main()
