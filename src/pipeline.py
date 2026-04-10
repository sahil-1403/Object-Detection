"""
Main video processing pipeline for object detection.

Orchestrates video reading, object detection, frame annotation,
CSV logging, and output video writing.
"""

import cv2
import time
import argparse
import os

from src.detector import Detector
from src.logger import DetectionLogger
from src.utils import draw_detection, draw_fps, draw_count


def process_video(
    input_path: str,
    output_dir: str = 'outputs',
    confidence: float = 0.25
) -> tuple[str, str]:
    """
    Process a video file with object detection and tracking.

    Processing flow:
        1. Open input video and read properties (resolution, fps, frame count)
        2. Initialize detector, logger, and output video writer
        3. For each frame:
           - Run detection + tracking
           - Log detections to CSV
           - Draw bounding boxes on frame
           - Write annotated frame to output video
        4. Release all resources

    Args:
        input_path: Path to input video file
        output_dir: Directory for output files (created if doesn't exist)
        confidence: Minimum detection confidence threshold (0.0-1.0)

    Returns:
        Tuple of (output_video_path, output_csv_path)

    Raises:
        FileNotFoundError: If input video doesn't exist
        RuntimeError: If video file cannot be opened
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input video not found: {input_path}")

    # Build output file paths
    base_name = os.path.splitext(os.path.basename(input_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    output_video = os.path.join(output_dir, f'annotated_{base_name}.mp4')
    output_csv = os.path.join(output_dir, f'detections_{base_name}.csv')

    # Open input video
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video file: {input_path}")

    # Read video properties
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\n{'='*70}")
    print("VIDEO PROCESSING PIPELINE")
    print(f"{'='*70}")
    print(f"Input:      {input_path}")
    print(f"Resolution: {frame_width}x{frame_height} @ {fps:.1f} FPS")
    print(f"Frames:     {total_frames}")
    print(f"Output:     {output_video}")
    print(f"CSV Log:    {output_csv}")
    print(f"{'='*70}\n")

    # Set up output video writer (MP4 format)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_video, fourcc, fps, (frame_width, frame_height))

    # Initialize detector and logger
    detector = Detector(confidence=confidence)
    logger = DetectionLogger(output_csv)

    frame_no = 0
    prev_time = time.time()

    try:
        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break    # End of video

            # Run detection + tracking
            detections = detector.detect(frame)

            # Log detections to CSV
            logger.log(frame_no, fps, detections)

            # Draw bounding boxes on frame
            for detection in detections:
                draw_detection(frame, detection)

            # Draw performance overlays
            now = time.time()
            live_fps = 1.0 / max(now - prev_time, 1e-6)
            prev_time = now

            draw_fps(frame, live_fps)
            draw_count(frame, len(detections))

            # Write annotated frame to output video
            writer.write(frame)

            frame_no += 1

            # Print progress every 100 frames
            if frame_no % 100 == 0:
                pct = (frame_no / total_frames * 100) if total_frames > 0 else 0
                print(f"  Progress: {frame_no}/{total_frames} ({pct:.1f}%) | "
                      f"FPS: {live_fps:.1f} | Objects: {len(detections)}")

    finally:
        # ALWAYS release resources, even if an error occurred
        cap.release()
        writer.release()
        logger.close()
        detector.close()
        cv2.destroyAllWindows()

    print(f"\n{'='*70}")
    print("PROCESSING COMPLETE")
    print(f"{'='*70}")
    print(f"Output video: {output_video}")
    print(f"CSV log:      {output_csv}")
    print(f"Total frames: {frame_no}")
    print(f"{'='*70}\n")

    return output_video, output_csv


def main():
    """Command-line interface for video processing pipeline."""
    parser = argparse.ArgumentParser(
        description='Object Detection Pipeline - Process videos with YOLOv8x + ByteTrack',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python src/pipeline.py --input videos/sample.mp4
  python src/pipeline.py --input videos/sample.mp4 --output results/
  python src/pipeline.py --input videos/sample.mp4 --confidence 0.4
        """
    )

    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input video file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='outputs',
        help='Output directory for annotated video and CSV (default: outputs/)'
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.25,
        help='Detection confidence threshold 0.0-1.0 (default: 0.25)'
    )

    args = parser.parse_args()

    # Validate confidence range
    if not 0.0 <= args.confidence <= 1.0:
        parser.error("Confidence must be between 0.0 and 1.0")

    # Run pipeline
    process_video(args.input, args.output, args.confidence)


if __name__ == '__main__':
    main()
