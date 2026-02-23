# Improving Detection Performance - Reducing False Positives & False Negatives

## Current Performance Issues

**Test Results:**
- **False Positives (FP): 1438** - Model detects non-elephants as elephants (49.77% precision means ~50% of detections are wrong)
- **False Negatives (FN): 1545** - Model misses actual elephants (47.98% recall means ~52% of elephants are missed)

This guide provides actionable strategies to improve both metrics.

---

## 🎯 Strategies to Reduce FALSE POSITIVES (Improve Precision)

### 1. **Increase Confidence Threshold** ⚡ Quick Win
**Current:** 0.25 (very permissive)  
**Try:** 0.35-0.50

```python
# In test_with_sahi.py or inference code
conf_threshold = 0.40  # Increase from 0.25
```

**Why it works:** Filters out low-confidence detections (likely false alarms)  
**Trade-off:** May slightly reduce recall (miss more elephants)

### 2. **Adjust IoU Threshold for NMS**
**Current:** 0.45  
**Try:** 0.30-0.40 (lower = more aggressive duplicate removal)

```python
iou_threshold = 0.35  # Reduce from 0.45
```

**Why it works:** Removes overlapping duplicate detections more aggressively

### 3. **Hard Negative Mining** 📊 Recommended
Retrain with difficult false positive examples:

```python
# scripts/training/hard_negative_mining.py
"""
1. Collect all false positive predictions from test set
2. Add these as background examples to training data
3. Retrain model to learn what NOT to detect
"""
```

**Implementation:**
- Extract FP bounding boxes from test results
- Save as "background" class or empty label files
- Include in next training run

### 4. **Class-Balanced Training**
```yaml
# In training config
cls_pw: 2.0  # Increase classification loss weight (default: 1.0)
```

**Why it works:** Forces model to be more careful about classification

### 5. **Post-Processing Filters**
Add custom filters based on detection characteristics:

```python
def filter_false_positives(detections):
    filtered = []
    for det in detections:
        # Filter by size (elephants have min/max size)
        bbox_area = (det['x2'] - det['x1']) * (det['y2'] - det['y1'])
        if bbox_area < 100 or bbox_area > 50000:  # Adjust based on your data
            continue
        
        # Filter by aspect ratio (elephants are roughly square)
        width = det['x2'] - det['x1']
        height = det['y2'] - det['y1']
        aspect_ratio = width / height
        if aspect_ratio < 0.5 or aspect_ratio > 2.0:  # Too elongated
            continue
        
        filtered.append(det)
    return filtered
```

---

## 🎯 Strategies to Reduce FALSE NEGATIVES (Improve Recall)

### 1. **Decrease Confidence Threshold** ⚡ Quick Win
**Current:** 0.25  
**Try:** 0.15-0.20

```python
conf_threshold = 0.20  # Decrease to catch more elephants
```

**Why it works:** Allows lower-confidence detections (catches hard cases)  
**Trade-off:** Will increase false positives

### 2. **Increase SAHI Overlap Ratio**
**Current:** 0.2 (20%)  
**Try:** 0.3-0.4 (30-40%)

```python
overlap_ratio = 0.35  # Increase from 0.2
```

**Why it works:** More overlap means elephants at slice boundaries are seen multiple times

### 3. **Multi-Scale Testing**
Test with different slice sizes:

```python
# Test with multiple scales and combine results
slice_sizes = [640, 1024, 1280]
all_predictions = []

for slice_size in slice_sizes:
    predictions = run_sahi_inference(image, slice_size=slice_size)
    all_predictions.extend(predictions)

# Merge with NMS
final_predictions = apply_nms(all_predictions)
```

### 4. **Data Augmentation** 📊 Recommended for Retraining
Add augmentations for difficult cases:

```yaml
# In training config (config/data.yaml or training args)
augment: true
mosaic: 1.0
mixup: 0.5
copy_paste: 0.3  # Copy-paste augmentation for small objects

# Add specific augmentations for small/occluded elephants
scale: 0.5  # Scale down to simulate distant elephants
hsv_h: 0.015
hsv_s: 0.7
hsv_v: 0.4
```

### 5. **Focus on Small Object Detection**
Modify architecture for better small object detection:

```python
# When training
model = YOLO('yolo11s.pt')  # Use 'small' variant instead of 'nano'

# Or train with focus on small objects
model.train(
    data='config/data.yaml',
    epochs=100,
    imgsz=1280,  # Larger image size preserves small objects
    batch=8,     # Reduce batch if memory limited
    mosaic=1.0,  # Mosaic augmentation helps with small objects
)
```

### 6. **Ensemble Methods**
Combine predictions from multiple models:

```python
# Train multiple models and average predictions
model1_preds = model1.predict(image)
model2_preds = model2.predict(image)
model3_preds = model3.predict(image)

# Weighted ensemble
combined = weighted_boxes_fusion([model1_preds, model2_preds, model3_preds],
                                 weights=[1, 1, 1],
                                 iou_thr=0.5)
```

---

## 🔄 Balanced Approach - Optimize Both Metrics

### Strategy 1: Two-Stage Detection (Recommended)
**Stage 1:** High recall (catch everything)
- Confidence: 0.15
- Overlap: 0.35

**Stage 2:** Filter with classifier (high precision)
- Train separate classifier to verify detections
- Apply to Stage 1 results

```python
# Pseudo-code
candidates = sahi_detect(image, conf=0.15, overlap=0.35)  # High recall
verified = []
for candidate in candidates:
    crop = image[candidate.bbox]
    is_elephant = classifier.predict(crop)  # Verification step
    if is_elephant > 0.7:
        verified.append(candidate)
```

### Strategy 2: Retrain with Better Data
**Key improvements:**

1. **Clean your training data:**
```python
# Review and fix incorrect labels
# Remove ambiguous examples
# Ensure consistent annotation quality
```

2. **Balance your dataset:**
```python
# Ensure variety:
# - Different lighting conditions
# - Various elephant sizes (close/far)
# - Different backgrounds
# - Partial occlusions
```

3. **Add more training data:**
```python
# Especially for:
# - Small elephants (distant aerial views)
# - Partially visible elephants
# - Elephants in shadows
# - Clustered elephants
```

### Strategy 3: Hyperparameter Optimization
Run systematic search for best thresholds:

```python
# scripts/evaluation/optimize_thresholds.py
import numpy as np
from itertools import product

conf_thresholds = np.arange(0.1, 0.6, 0.05)
iou_thresholds = np.arange(0.3, 0.6, 0.05)

best_f1 = 0
best_params = {}

for conf, iou in product(conf_thresholds, iou_thresholds):
    precision, recall, f1 = evaluate(conf_threshold=conf, iou_threshold=iou)
    
    if f1 > best_f1:
        best_f1 = f1
        best_params = {'conf': conf, 'iou': iou}
        print(f"New best F1: {f1:.3f} at conf={conf}, iou={iou}")

print(f"Optimal parameters: {best_params}")
```

---

## 📋 Action Plan - What to Do Next

### ⚡ **Immediate Actions (No Retraining)**

1. **Optimize thresholds** (1-2 hours)
   - Run grid search for best conf/IoU thresholds
   - Test different SAHI overlap ratios
   - Implement in `scripts/evaluation/optimize_thresholds.py`

2. **Add post-processing filters** (2-3 hours)
   - Size-based filtering
   - Aspect ratio filtering
   - Implement in `scripts/evaluation/post_process_filters.py`

3. **Multi-scale testing** (3-4 hours)
   - Test with slice sizes: [640, 1024, 1280]
   - Combine results
   - Implement in `scripts/evaluation/multiscale_test.py`

**Expected improvement:** 5-10% boost in F1-score

### 📊 **Short-term (With Limited Retraining)**

4. **Hard negative mining** (1-2 days)
   - Extract false positives from test set
   - Create hard negative dataset
   - Retrain for 20-30 epochs
   - Script: `scripts/training/hard_negative_training.py`

5. **Better augmentation** (1 day)
   - Update augmentation config
   - Focus on small object augmentations
   - Retrain with new config

**Expected improvement:** 10-15% boost in F1-score

### 🎯 **Long-term (Full Improvement Cycle)**

6. **Data quality review** (1 week)
   - Review all training labels
   - Fix inconsistencies
   - Add difficult examples

7. **Ensemble approach** (1-2 weeks)
   - Train 3-5 models with different configs
   - Implement weighted box fusion
   - Script: `scripts/evaluation/ensemble_inference.py`

8. **Two-stage detection** (2-3 weeks)
   - Train verification classifier
   - Implement two-stage pipeline
   - Optimize both stages

**Expected improvement:** 20-30% boost in F1-score (reach 60-70% F1)

---

## 🛠️ Implementation Templates

### Template 1: Threshold Optimization Script
```python
# scripts/evaluation/optimize_thresholds.py
"""Find optimal confidence and IoU thresholds"""

from pathlib import Path
import json
import numpy as np
from tqdm import tqdm
from test_with_sahi import SAHITester

def grid_search_thresholds():
    test_images_dir = "AED/test_images"
    test_labels_dir = "AED/labels/test"
    model_path = "runs/train/elephants_yolo11_sliced7/weights/best.pt"
    
    conf_range = np.arange(0.15, 0.55, 0.05)
    iou_range = np.arange(0.30, 0.55, 0.05)
    overlap_range = [0.2, 0.3, 0.4]
    
    results = []
    
    for conf in conf_range:
        for iou in iou_range:
            for overlap in overlap_range:
                print(f"\nTesting: conf={conf:.2f}, iou={iou:.2f}, overlap={overlap:.2f}")
                
                tester = SAHITester(
                    model_path=model_path,
                    test_images_dir=test_images_dir,
                    test_labels_dir=test_labels_dir,
                    output_dir=f"runs/threshold_search/conf{conf:.2f}_iou{iou:.2f}_ovl{overlap:.2f}",
                    conf_threshold=conf,
                    iou_threshold=iou,
                    overlap_ratio=overlap
                )
                
                summary = tester.run_evaluation(save_predictions=False, max_images=50)
                
                results.append({
                    'conf': conf,
                    'iou': iou,
                    'overlap': overlap,
                    'precision': summary['overall_precision'],
                    'recall': summary['overall_recall'],
                    'f1_score': summary['overall_f1_score']
                })
    
    # Find best F1
    best_result = max(results, key=lambda x: x['f1_score'])
    
    print("\n" + "="*60)
    print("BEST PARAMETERS:")
    print("="*60)
    print(f"Confidence Threshold: {best_result['conf']:.2f}")
    print(f"IoU Threshold: {best_result['iou']:.2f}")
    print(f"Overlap Ratio: {best_result['overlap']:.2f}")
    print(f"\nPrecision: {best_result['precision']:.4f}")
    print(f"Recall: {best_result['recall']:.4f}")
    print(f"F1-Score: {best_result['f1_score']:.4f}")
    
    # Save results
    with open("runs/threshold_optimization_results.json", 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == "__main__":
    grid_search_thresholds()
```

### Template 2: Post-Processing Filter
```python
# scripts/evaluation/apply_filters.py
"""Apply post-processing filters to reduce false positives"""

def filter_detections(predictions, image_width, image_height):
    """
    Apply heuristic filters to remove likely false positives
    
    Args:
        predictions: List of detection dicts with bbox coordinates
        image_width: Image width in pixels
        image_height: Image height in pixels
    
    Returns:
        Filtered predictions list
    """
    filtered = []
    
    for pred in predictions:
        x1, y1, x2, y2 = pred['x1'], pred['y1'], pred['x2'], pred['y2']
        
        # Calculate bbox properties
        width = x2 - x1
        height = y2 - y1
        area = width * height
        aspect_ratio = width / height if height > 0 else 0
        
        # Relative size (percentage of image)
        rel_area = area / (image_width * image_height)
        
        # FILTER 1: Size constraints
        # Elephants in aerial imagery typically occupy 0.01% to 5% of image
        if rel_area < 0.0001 or rel_area > 0.05:
            continue
        
        # FILTER 2: Aspect ratio
        # Elephants are roughly circular/square (0.5 to 2.0 ratio)
        if aspect_ratio < 0.4 or aspect_ratio > 2.5:
            continue
        
        # FILTER 3: Minimum absolute size
        # Elephants should be at least 20x20 pixels
        if width < 20 or height < 20:
            continue
        
        # FILTER 4: Edge detections (often false positives)
        margin = 10
        if (x1 < margin or y1 < margin or 
            x2 > image_width - margin or y2 > image_height - margin):
            # Reduce confidence for edge detections
            pred['confidence'] *= 0.8
        
        filtered.append(pred)
    
    return filtered

# Example usage in test_with_sahi.py
# After getting predictions:
# predictions = filter_detections(predictions, img_width, img_height)
```

---

## 📊 Expected Outcomes

| Action | F1 Improvement | Effort | Timeline |
|--------|----------------|--------|----------|
| Threshold optimization | +3-5% | Low | 1-2 hours |
| Post-processing filters | +2-4% | Low | 2-3 hours |
| Multi-scale testing | +3-5% | Medium | 3-4 hours |
| Hard negative mining | +5-8% | Medium | 1-2 days |
| Better augmentation | +5-10% | Medium | 1-2 days |
| Data cleaning | +8-12% | High | 1 week |
| Ensemble methods | +10-15% | High | 1-2 weeks |
| **Combined** | **+20-30%** | **High** | **2-4 weeks** |

**Realistic target:** 65-75% F1-score (from current 48.86%)

---

## 💡 Quick Start

**Start here for fastest results:**

1. Create `scripts/evaluation/optimize_thresholds.py` (use template above)
2. Run: `python scripts/evaluation/optimize_thresholds.py`
3. Apply best thresholds found
4. Test improvement

**Next:**
5. Implement post-processing filters
6. Test multi-scale inference

This should get you ~10% improvement within a day!
