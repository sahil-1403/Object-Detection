"""
Object detection module with TensorRT acceleration and ByteTrack tracking.

Provides a high-level interface for running GPU-accelerated object detection
on video frames with stable multi-object tracking across frames.
"""

import os
from ultralytics import YOLO
from src.scaler import get_imgsz


class Detector:
    """
    TensorRT-accelerated object detector with ByteTrack tracking.

    This class encapsulates model loading, inference, and tracking state.
    The model is loaded once during initialization and reused for all frames,
    ensuring fast inference without repeated model loading overhead.

    Usage:
        detector = Detector()
        detections = detector.detect(frame)
        # ... process more frames ...
        detector.close()
    """

    def __init__(
        self,
        engine_path: str = 'exports/yolov8x_custom.engine',
        confidence: float = 0.25,
        device: int = 0
    ):
        """
        Initialize the detector with a TensorRT engine.

        Args:
            engine_path: Path to TensorRT .engine file
            confidence: Minimum detection confidence threshold (0.0-1.0)
                       Lower values detect more objects but include more false positives
            device: GPU device ID (0 = first GPU, 1 = second GPU, etc.)

        Raises:
            FileNotFoundError: If engine file doesn't exist
        """
        if not os.path.exists(engine_path):
            raise FileNotFoundError(
                f"TensorRT engine not found at: {engine_path}\n"
                "Run train.py first to train the model and export to TensorRT."
            )

        print(f"Loading TensorRT engine from: {engine_path}")
        # task='detect' must be explicitly specified for .engine files
        self.model = YOLO(engine_path, task='detect')
        self.confidence = confidence
        self.device = device
        self.imgsz = None    # Will be set from first frame
        print("Detector initialized and ready.")

    def detect(self, frame) -> list[dict]:
        """
        Run object detection and tracking on a single frame.

        Args:
            frame: numpy array of shape (H, W, 3) in BGR format (OpenCV standard)

        Returns:
            List of detection dictionaries, each containing:
            {
                'track_id':   int   - Stable ID across frames (-1 if not yet tracked)
                'class_id':   int   - Numeric class index
                'class_name': str   - Human-readable class name (e.g., 'person')
                'confidence': float - Detection confidence (0.0-1.0)
                'x1': int, 'y1': int - Top-left corner of bounding box (pixels)
                'x2': int, 'y2': int - Bottom-right corner of bounding box (pixels)
            }

        Notes:
            - imgsz is calculated once from the first frame and reused for all
              subsequent frames (all frames in a video have the same resolution)
            - ByteTrack tracking maintains consistent track_id values across frames
            - persist=True ensures the tracker remembers objects between frames
        """
        # Calculate optimal inference size from first frame only
        if self.imgsz is None:
            self.imgsz = get_imgsz(frame)
            h, w = frame.shape[:2]
            print(f"Input resolution: {w}x{h} → using imgsz={self.imgsz}")

        # Run inference with ByteTrack tracking
        results = self.model.track(
            frame,
            imgsz=self.imgsz,
            conf=self.confidence,
            device=self.device,
            tracker='bytetrack.yaml',   # Multi-object tracking algorithm
            persist=True,                # Maintain tracking state between frames
            verbose=False                # Suppress per-frame console output
        )

        detections = []

        # Extract detection data from results
        if results[0].boxes is None:
            return detections

        for box in results[0].boxes:
            # Extract track ID (None on first frame or for unconfirmed tracks)
            track_id = int(box.id) if box.id is not None else -1

            # Extract class information
            class_id = int(box.cls)
            class_name = self.model.names[class_id]

            # Extract confidence score
            confidence = round(float(box.conf), 3)

            # Extract bounding box coordinates (xyxy format)
            # x1, y1 = top-left corner
            # x2, y2 = bottom-right corner
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            detections.append({
                'track_id':   track_id,
                'class_id':   class_id,
                'class_name': class_name,
                'confidence': confidence,
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2,
            })

        return detections

    def close(self):
        """
        Release GPU resources.

        Always call this method when done processing to free GPU memory.
        Failure to call close() may cause memory leaks in long-running processes.
        """
        del self.model
        print("Detector closed, GPU memory released.")
