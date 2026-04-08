"""
CSV detection logger for frame-by-frame object tracking data.

Writes detection data to CSV format for post-processing, analysis,
and integration with data analysis tools like pandas or Excel.
"""

import csv
import os


# CSV column headers
FIELDS = [
    'frame_number',
    'timestamp_sec',
    'track_id',
    'class_name',
    'confidence',
    'x1', 'y1', 'x2', 'y2'
]


class DetectionLogger:
    """
    Writes object detection data to a CSV file with automatic flushing.

    Each row represents one detected object in one frame. A frame with 3
    detected objects will generate 3 rows in the CSV.

    CSV Schema:
        frame_number  : int   - Zero-indexed frame count
        timestamp_sec : float - Time in seconds (frame_number / fps)
        track_id      : int   - ByteTrack stable object ID
        class_name    : str   - Object class (e.g., 'person', 'chair')
        confidence    : float - Detection confidence (0.0-1.0)
        x1, y1        : int   - Top-left corner of bounding box
        x2, y2        : int   - Bottom-right corner of bounding box

    Usage:
        logger = DetectionLogger('outputs/detections.csv')
        logger.log(frame_no, fps, detections)
        # ... process more frames ...
        logger.close()
    """

    def __init__(self, csv_path: str):
        """
        Initialize CSV logger and write header row.

        Args:
            csv_path: Path where CSV file will be created

        Notes:
            Parent directory will be created if it doesn't exist.
            newline='' prevents double line breaks on Windows.
        """
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Open file with newline='' to prevent double line breaks on Windows
        self._file = open(csv_path, 'w', newline='')
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()
        print(f"CSV logger initialized: {csv_path}")

    def log(self, frame_no: int, fps: float, detections: list[dict]) -> None:
        """
        Write detections from one frame to CSV.

        Args:
            frame_no: Zero-indexed frame number
            fps: Video frames per second (for timestamp calculation)
            detections: List of detection dicts from Detector.detect()

        Notes:
            Automatically flushes to disk every 100 frames to prevent
            data loss if the program crashes mid-processing.
        """
        timestamp = round(frame_no / fps, 3) if fps > 0 else 0.0

        for d in detections:
            self._writer.writerow({
                'frame_number': frame_no,
                'timestamp_sec': timestamp,
                'track_id':     d['track_id'],
                'class_name':   d['class_name'],
                'confidence':   d['confidence'],
                'x1': d['x1'],
                'y1': d['y1'],
                'x2': d['x2'],
                'y2': d['y2'],
            })

        # Flush to disk every 100 frames to minimize data loss on crash
        if frame_no % 100 == 0:
            self._file.flush()

    def close(self) -> None:
        """
        Flush remaining data and close the CSV file.

        Always call this method when done logging. Use a try/finally block
        to ensure close() is called even if an exception occurs during processing.
        """
        self._file.close()
        print("CSV logger closed.")
