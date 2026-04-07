"""
Drawing utilities for video frame annotation.

Provides functions to overlay bounding boxes, labels, and performance
metrics on video frames for visualization.
"""

import cv2


# Fixed color palette for consistent class visualization
# Colors are in BGR format (Blue, Green, Red) as required by OpenCV
PALETTE = [
    (56,  56,  255),    # Red
    (255, 56,  56),     # Blue
    (56,  255, 56),     # Green
    (0,   165, 255),    # Orange
    (255, 0,   180),    # Purple
    (200, 200, 0),      # Teal
    (0,   255, 255),    # Yellow
    (180, 100, 255),    # Pink
]


def get_color(class_id: int) -> tuple:
    """
    Get a consistent color for a given class ID.

    Args:
        class_id: Numeric class identifier

    Returns:
        BGR color tuple
    """
    return PALETTE[class_id % len(PALETTE)]


def draw_detection(frame, detection: dict) -> None:
    """
    Draw a bounding box with label on a frame.

    Modifies the frame in-place by adding a colored bounding box
    and a label showing the class name, track ID, and confidence.

    Args:
        frame: numpy array (H, W, 3) to draw on
        detection: Dict containing detection data with keys:
                  'class_id', 'class_name', 'track_id', 'confidence',
                  'x1', 'y1', 'x2', 'y2'

    Label format: "person #3  0.87"
                   [class] [track_id] [confidence]
    """
    color = get_color(detection['class_id'])
    x1 = detection['x1']
    y1 = detection['y1']
    x2 = detection['x2']
    y2 = detection['y2']

    # Draw bounding box rectangle (2px thickness)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    # Build label text
    label = f"{detection['class_name']} #{detection['track_id']}  {detection['confidence']}"

    # Measure text size for background rectangle
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    thickness = 1
    (text_w, text_h), _ = cv2.getTextSize(label, font, font_scale, thickness)

    # Draw filled rectangle as label background
    cv2.rectangle(
        frame,
        (x1, y1 - text_h - 8),
        (x1 + text_w + 4, y1),
        color,
        -1    # -1 = filled rectangle
    )

    # Draw label text (white on colored background)
    cv2.putText(
        frame,
        label,
        (x1 + 2, y1 - 4),
        font,
        font_scale,
        (255, 255, 255),    # White text
        thickness
    )


def draw_fps(frame, fps: float) -> None:
    """
    Draw FPS counter in the top-left corner.

    Args:
        frame: numpy array to draw on
        fps: Current frames per second
    """
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),    # Green
        2
    )


def draw_count(frame, count: int) -> None:
    """
    Draw object count below the FPS counter.

    Args:
        frame: numpy array to draw on
        count: Number of detected objects in current frame
    """
    cv2.putText(
        frame,
        f"Objects: {count}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),    # Yellow
        2
    )
