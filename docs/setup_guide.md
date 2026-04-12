# Object Detector - Complete Setup Guide

## Overview
This guide walks you through setting up and running the YOLOv8x object detection pipeline with TensorRT acceleration and ByteTrack tracking.

## Prerequisites

### Hardware Requirements
- NVIDIA GPU with CUDA support (RTX 4090 recommended)
- 24GB+ VRAM for training
- 16GB+ VRAM minimum for inference only

### Software Requirements
- Python 3.10
- CUDA 12.1
- cuDNN 8.9
- Git

## Installation Steps

### 1. Create Python Environment
```bash
# Using conda (recommended)
conda create -n objdet python=3.10
conda activate objdet

# Or using venv
python3.10 -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows
```

### 2. Install PyTorch with CUDA
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 3. Install Project Dependencies
```bash
pip install -r requirements.txt
```

### 4. Verify GPU Setup
```bash
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0)}')"
```

Expected output:
```
CUDA: True
GPU: NVIDIA GeForce RTX 4090
```

### 5. Create .env File
```bash
cp .env.example .env
```

The default values should work for most setups. Edit if needed.

## Dataset Preparation

### Option 1: Using Existing Dataset

If you have an existing dataset in YOLO, COCO, or Pascal VOC format:

```bash
# Place your dataset in the data/ directory with this structure:
data/
├── images/
│   ├── train/
│   ├── val/
│   └── test/
└── labels/  (for YOLO format)
    ├── train/
    ├── val/
    └── test/
```

Then run:
```bash
python src/dataset.py
```

This will:
- Detect your dataset format
- Convert to YOLO format if needed
- Validate all annotations
- Generate `data/dataset.yaml`

### Option 2: Starting Fresh

If you don't have a dataset yet, you'll need to:
1. Collect or download images
2. Annotate them using tools like:
   - [Roboflow](https://roboflow.com) (recommended for beginners)
   - [CVAT](https://github.com/opencv/cvat)
   - [LabelImg](https://github.com/heartexlabs/labelImg)
3. Export in YOLO format
4. Place in `data/` directory

## Training the Model

### 1. Run Training
```bash
python src/train.py
```

This will:
- Fine-tune YOLOv8x on your dataset
- Save checkpoints every 10 epochs
- Apply early stopping (patience=10)
- Export best weights to TensorRT engine

**Estimated time on RTX 4090:**
- 5,000 images: ~1.5 hours
- 10,000 images: ~2.5 hours

### 2. Monitor Training
Training progress is shown in the console:
```
Epoch   GPU_mem   box_loss  cls_loss  Instances  Size
1/50    4.2G      2.341     1.823     234        640
...
```

**Good signs:**
- Losses decreasing over time
- GPU memory stable
- mAP@0.5 increasing in validation

**Bad signs:**
- CUDA OOM error → reduce batch size in `src/train.py`
- Loss flat after 5 epochs → check data quality
- Val loss rising → early stopping will activate

### 3. Training Outputs
```
runs/train/yolov8x_custom/
├── weights/
│   ├── best.pt          ← Best model (use this)
│   └── last.pt          ← Final epoch
└── results.png          ← Training curves
```

```
exports/
└── yolov8x_custom.engine  ← TensorRT engine for inference
```

## Evaluation

```bash
python src/eval.py
```

This generates:
- `outputs/eval_metrics.json` - Detailed per-class metrics
- `runs/val/yolov8x_custom/confusion_matrix.png` - Confusion matrix
- `runs/val/yolov8x_custom/PR_curve.png` - Precision-recall curves

**Interpreting Results:**
- mAP@0.5 > 0.80: Excellent, production ready
- mAP@0.5 0.70-0.80: Good, usable
- mAP@0.5 0.60-0.70: Acceptable, could improve
- mAP@0.5 < 0.60: Needs more data or longer training

## Running Inference

### Command Line (Single Video)
```bash
python src/pipeline.py --input path/to/video.mp4
```

Optional arguments:
```bash
python src/pipeline.py \
  --input path/to/video.mp4 \
  --output custom_output_dir/ \
  --confidence 0.4  # Higher = fewer but more confident detections
```

Outputs:
- `outputs/annotated_<filename>.mp4` - Video with bounding boxes
- `outputs/detections_<filename>.csv` - Detection log

### Web Interface
```bash
# Start the server
uvicorn src.api:app --host 0.0.0.0 --port 8000

# Or with auto-reload for development
uvicorn src.api:app --host 0.0.0.0 --port 8000 --reload
```

Then open your browser to: `http://localhost:8000`

**Features:**
- Drag-and-drop video upload
- Real-time processing status
- Download annotated video + CSV
- Clean dark-themed UI

## Performance Tuning

### Inference Speed
On RTX 4090 with TensorRT:
- 1080p video: ~60-80 FPS
- 720p video: ~90-120 FPS
- 4K video: ~30-40 FPS

### Improving Speed
1. **Lower confidence threshold:** Fewer detections = faster
2. **Reduce imgsz:** Edit `src/scaler.py` to use smaller sizes
3. **Skip frames:** Modify pipeline to process every Nth frame

### Improving Accuracy
1. **Collect more data:** 300+ samples per class recommended
2. **Balance dataset:** Ensure all classes have similar sample counts
3. **Higher confidence threshold:** Reduces false positives
4. **Longer training:** Increase epochs in `src/train.py`

## Troubleshooting

### CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solutions:**
1. Reduce batch size in `src/train.py`:
   ```python
   batch=8  # instead of 16
   ```
2. Close other GPU-using programs
3. Use smaller model (YOLOv8l or YOLOv8m)

### TensorRT Engine Not Found
**Error:** `FileNotFoundError: TensorRT engine not found`

**Solution:**
Run training first:
```bash
python src/train.py
```

### Empty Detections
**Problem:** Video processes but no objects detected

**Solutions:**
1. Check confidence threshold (try lowering to 0.1)
2. Verify model was trained on relevant classes
3. Check if input video matches training data quality

### Corrupted Output Video
**Problem:** Output .mp4 won't play

**Cause:** Program crashed before `writer.release()` was called

**Solution:**
- Always use try/finally blocks (already implemented in pipeline.py)
- Reprocess the video

## Project File Structure

```
object-detector/
├── src/                    # Source code
│   ├── dataset.py          # Dataset validation and conversion
│   ├── train.py            # Model training + TensorRT export
│   ├── eval.py             # Model evaluation
│   ├── detector.py         # TensorRT inference wrapper
│   ├── pipeline.py         # Video processing pipeline
│   ├── logger.py           # CSV logging
│   ├── utils.py            # Drawing utilities
│   ├── scaler.py           # Resolution scaling
│   └── api.py              # FastAPI web server
├── frontend/               # Web UI
│   ├── index.html          # Main page
│   └── main.js             # Client-side logic
├── data/                   # Dataset (gitignored)
│   ├── images/
│   ├── labels/
│   └── dataset.yaml        # Auto-generated by dataset.py
├── exports/                # TensorRT engines (gitignored)
├── outputs/                # Generated videos/CSVs (gitignored)
├── runs/                   # Training logs (gitignored)
├── temp/                   # Temporary uploads (gitignored)
├── docs/                   # Documentation
├── .env                    # Local config (gitignored)
├── .env.example            # Config template
├── .gitignore              # Git ignore rules
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

## Next Steps

1. **Prepare your dataset** using `src/dataset.py`
2. **Train the model** using `src/train.py` (~2 hours)
3. **Evaluate performance** using `src/eval.py`
4. **Process videos** using `src/pipeline.py` or the web UI
5. **Deploy** by running the FastAPI server on a remote machine

## Git Workflow (for team development)

```bash
# Initial setup
git init
git add .gitignore README.md requirements.txt .env.example src/ frontend/
git commit -m "feat: initial project structure"
git branch -M main
git remote add origin https://github.com/yourname/object-detector.git
git push -u origin main

# Feature development
git checkout -b feature/my-feature
# ... make changes ...
git add .
git commit -m "feat: add my feature"
git push origin feature/my-feature
# Open PR on GitHub → merge → delete branch
```

## Support

For issues or questions:
1. Check this guide and the main README
2. Review docs/ folder for additional documentation
3. Check Ultralytics documentation: https://docs.ultralytics.com
4. File an issue in your repository

## License

This project uses:
- YOLOv8 (AGPL-3.0)
- ByteTrack (MIT)
- FastAPI (MIT)
- OpenCV (Apache 2.0)

Ensure compliance with these licenses for your use case.
