# Demo Test Package - Elephant Detection

This folder contains a **self-contained demo** for testing the trained elephant detection model. It's designed to be portable and work on any system (Windows, Mac, Linux).

---

## 📦 Contents

```
demo_test/
├── run_demo_test.py        # Main test script (runs inference + evaluation)
├── images/                 # 50 sample test images (5472×3648 aerial photos)
├── labels/                 # Ground truth labels in YOLO format
├── results/                # Output folder (created when script runs)
└── README.md              # This file
```

---

## 🎯 What This Demo Does

1. **Loads the trained model** (`best.pt` from `../models/`)
2. **Runs SAHI inference** on 50 test images using optimized parameters
3. **Calculates metrics**: Precision, Recall, F1-Score
4. **Saves results** to `results/demo_results.json`

**No path changes needed** - all paths are relative to script location!

---

## 🚀 Quick Start

### On Windows:
```bash
cd meeting_materials/demo_test
python run_demo_test.py
```

### On Mac/Linux:
```bash
cd meeting_materials/demo_test
python3 run_demo_test.py
```

---

## 📋 Requirements

The script needs these Python packages:
```bash
pip install sahi ultralytics opencv-python numpy
```

**For Mac users:** If you have Apple Silicon (M1/M2/M3), you might need:
```bash
pip install torch torchvision
pip install ultralytics sahi opencv-python numpy
```

---

## 🔧 Configuration

The demo uses **optimized parameters** (found to work best):

| Parameter | Value | Description |
|-----------|-------|-------------|
| Slice Size | 1024×1024 | Matches training configuration |
| Overlap Ratio | 0.30 | 30% overlap between slices |
| Confidence | 0.30 | Minimum detection confidence |
| IoU Threshold | 0.40 | For duplicate removal (NMS) |

These are hardcoded in the script for consistency. To change them, edit `run_demo_test.py` lines 34-37.

---

## 📊 Expected Results

Based on our testing, you should see approximately:

- **Precision:** ~53%
- **Recall:** ~49%
- **F1-Score:** ~51%

These are realistic metrics for challenging aerial elephant detection on high-resolution images.

---

## 🖥️ Running on MacBook

### Path Differences

**Windows paths:**
```
D:\Users\...\elephant_detection-master\meeting_materials\demo_test
```

**Mac paths:**
```
/Users/.../elephant_detection-master/meeting_materials/demo_test
```

**Good news:** This script uses `Path(__file__).parent` which works on both!

### Device Detection

The script automatically detects:
- **CUDA GPU** (if available on Windows/Linux with NVIDIA)
- **CPU** (fallback, works everywhere)
- **MPS** (Apple Silicon - automatically used by PyTorch if available)

### Running on Mac

1. **Open Terminal**
2. **Navigate to project:**
   ```bash
   cd ~/path/to/elephant_detection-master/meeting_materials/demo_test
   ```
3. **Install dependencies:**
   ```bash
   pip3 install sahi ultralytics opencv-python numpy
   ```
4. **Run demo:**
   ```bash
   python3 run_demo_test.py
   ```

---

## 📁 File Structure Details

### Images (`images/`)
- 50 aerial photographs of elephants
- Original resolution: 5472×3648 pixels
- JPEG format
- Representative sample from full test set of 439 images

### Labels (`labels/`)
- YOLO format text files (one per image)
- Format: `class x_center y_center width height` (all normalized 0-1)
- Each line = one elephant bounding box
- Class 0 = elephant

### Model (`../models/best.pt`)
- YOLOv11 trained on sliced images
- Input size: 1024×1024 during training
- Fine-tuned for elephant detection from aerial imagery
- File size: ~19.3 MB

---

## 🤖 For AI Agents

### Context Understanding

This demo package is part of a **supervisor meeting presentation** for an elephant detection research project. Key facts:

1. **Problem:** Detect elephants in high-resolution aerial drone imagery
2. **Challenge:** Images are 5472×3648 (too large for standard object detection)
3. **Solution:** SAHI (Slicing Aided Hyper Inference) - slice images into 1024×1024 patches
4. **Model:** YOLOv11 trained on sliced dataset
5. **Performance:** 51% F1-score after optimization

### Portable Design

- ✅ All paths are **relative to script location**
- ✅ Works on **Windows, Mac, Linux**
- ✅ Automatically detects **GPU/CPU/MPS**
- ✅ **Self-contained** (doesn't depend on project structure)
- ✅ Clear error messages if dependencies missing

### Expected User Actions

1. Copy `meeting_materials/` folder to MacBook
2. Install Python dependencies
3. Run `run_demo_test.py`
4. Review results in `results/demo_results.json`
5. Show results during supervisor meeting

### Troubleshooting Tips

**If model not found:**
- Ensure `best.pt` is in `../models/` relative to this script
- Check file wasn't corrupted during transfer

**If SAHI import fails:**
- Run: `pip install sahi`
- May need to upgrade pip: `pip install --upgrade pip`

**If slow on Mac:**
- First run downloads model weights (one-time)
- Expect ~2-3 seconds per image on modern Mac
- Total runtime: ~2-5 minutes for 50 images

**If different results on Mac:**
- PyTorch versions may differ slightly
- F1-score should be within ±1-2% of Windows results
- If drastically different, check model loaded correctly

---

## 📝 Output Format

Results are saved to `results/demo_results.json`:

```json
{
  "timestamp": "2026-01-28T...",
  "configuration": {
    "model": "best.pt",
    "slice_size": 1024,
    "overlap_ratio": 0.3,
    "conf_threshold": 0.3,
    "iou_threshold": 0.4
  },
  "results": {
    "images_tested": 50,
    "true_positives": 769,
    "false_positives": 678,
    "false_negatives": 798,
    "precision": 0.5316,
    "recall": 0.4907,
    "f1_score": 0.5103
  },
  "per_image_results": [...]
}
```

---

## 🔗 Related Files in Meeting Materials

- **Models:** `../models/best.pt` (trained weights)
- **Training Results:** `../training_visualizations/`
- **Full Test Report:** `../test_evaluation/`
- **XAI Heatmaps:** `../xai_visualizations/`
- **Documentation:** `../documentation/MEETING_SUMMARY.md`

---

## ⚡ Quick Reference

**To run demo:**
```bash
python run_demo_test.py
```

**To check if packages installed:**
```bash
python -c "import sahi, ultralytics, cv2, numpy; print('✅ All packages installed')"
```

**To see results:**
```bash
cat results/demo_results.json
# or on Windows:
type results\demo_results.json
```

---

## 📞 Support

If you encounter issues:

1. **Check error message** - script provides clear diagnostics
2. **Verify dependencies** - `pip list | grep -E "sahi|ultralytics|opencv"`
3. **Check Python version** - Requires Python 3.8+
4. **Read error details** - script explains what's missing

**Common fixes:**
- Missing packages: `pip install sahi ultralytics opencv-python numpy`
- Wrong Python version: Use `python3` instead of `python`
- Permission issues: Run from user directory, not system folders

---

**Last Updated:** January 28, 2026  
**Tested On:** Windows 11, Python 3.13  
**Compatible With:** Windows, macOS, Linux
