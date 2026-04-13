# Object Detector

Detects objects in video files using a fine-tuned YOLOv8x model.
Outputs an annotated video with bounding boxes + a CSV of all detections.

Runs fully locally — no cloud APIs, no paid services.

---

## Requirements
- Python 3.10
- NVIDIA GPU (RTX 4090 recommended)
- CUDA 12.1 + cuDNN 8.9
- 24GB+ VRAM for training (16GB minimum for inference)

## Setup
```bash
git clone https://github.com/yourname/object-detector
cd object-detector
python3.10 -m venv venv
source venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On Windows (PowerShell), activate with:
```powershell
venv\Scripts\Activate.ps1
```

For CUDA 12.1 builds of PyTorch:
```bash
pip install torch==2.3.0+cu121 torchvision==0.18.0+cu121 --index-url https://download.pytorch.org/whl/cu121
```

## Usage

### Run on a video file
```bash
python src/pipeline.py --input path/to/video.mp4
```

### Run API service
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### Run Streamlit app
```bash
streamlit run app.py
```

### Using Shell Script
```bash
./run.sh
```

### Train the model
```bash
python src/train.py
```

### Evaluate model performance
```bash
python src/eval.py
```

## Output
- `outputs/annotated_<filename>.mp4` — video with bounding boxes drawn
- `outputs/detections_<filename>.csv` — frame-by-frame detection log

## Architecture
```mermaid
flowchart TD
    A[User] --> B{Entrypoint}

    B --> C1[CLI: src/pipeline.py]
    B --> C2[API: src/api.py]
    B --> C3[Streamlit: app.py]

    %% CLI / API shared pipeline
    C1 --> D1_CLI["OpenCV VideoCapture"]
    C2 --> U1["upload saves temp file"]
    U1 --> U2["Background run_pipeline"]
    U2 --> D1_API["OpenCV VideoCapture"]

    D1_CLI --> E1["Detector.detect\n(src/detector.py)"]
    D1_API --> E1

    E1 --> E2["YOLO engine + ByteTrack tracker"]
    E1 --> F["Post-processing"]
    F --> F1["DetectionLogger\n(src/logger.py)"]
    F --> F2["draw_detection + overlays\n(src/utils.py)"]

    F2 --> G1["VideoWriter annotated MP4"]
    F1 --> G2["CSV detections log"]

    G1 --> H1["outputs/annotated_*.mp4"]
    G2 --> H2["outputs/detections_*.csv"]

    C2 --> S1["/status/{job_id}/"]
    C2 --> S2["/download/{job_id}/video"]
    C2 --> S3["/download/{job_id}/csv"]

    %% Streamlit path
    C3 --> T1["Load YOLO model yolov8x.pt"]
    T1 --> T2["Frame-by-frame inference"]
    T2 --> T3["Draw and filter detections"]
    T3 --> T4["Temp MP4"]

    T4 --> T5{ffmpeg available?}
    T5 -->|yes| T6["H.264 re-encode"]
    T5 -->|no| T7["Use OpenCV MP4"]

    T6 --> T8["Preview and download"]
    T7 --> T8

    %% Training / eval
    subgraph "Training and Eval"
      P1["src/dataset.py prepare + validate"] --> P2["data/dataset.yaml"]
      P2 --> P3["src/train.py fine-tune YOLOv8x"]
      P3 --> P4["runs/train/.../best.pt"]
      P4 --> P5["Export TensorRT engine"]
      P5 --> P6["exports/yolov8x_custom.engine"]
      P4 --> P7["src/eval.py"]
      P7 --> P8["outputs/eval_metrics.json + plots"]
    end
```

## Project Structure
```
object-detector/
├── src/                ← Python source code
│   ├── dataset.py      ← Dataset preparation and validation
│   ├── train.py        ← Model training and TensorRT export
│   ├── eval.py         ← Model evaluation
│   ├── detector.py     ← TensorRT inference engine
│   ├── pipeline.py     ← Video processing pipeline
│   ├── logger.py       ← CSV detection logging
│   ├── utils.py        ← Drawing utilities
│   ├── scaler.py       ← Resolution scaling
│   └── api.py          ← FastAPI web server
├── app.py              ← Streamlit app
├── data/               ← Dataset (not committed to Git)
├── exports/            ← TensorRT engine files (not committed)
├── outputs/            ← Generated videos and CSVs (not committed)
└── docs/               ← Documentation and reference material
```

## Classes detected
person, chair, laptop, mouse, keyboard, monitor, mobile phone, bottle, cup,
book, backpack, handbag, umbrella, dog, cat, bird, bicycle, car, motorbike,
dining table, couch, potted plant, bed
