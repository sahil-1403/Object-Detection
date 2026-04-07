"""
Resolution scaling utility for optimal YOLO inference.

Determines the appropriate imgsz parameter based on input frame resolution
to balance detection quality and inference speed.
"""

import numpy as np


def get_imgsz(frame: np.ndarray) -> int:
    """
    Calculate optimal inference size based on input frame resolution.

    YOLOv8x was trained at 640px. For low-resolution inputs, we upscale
    during inference to improve small object detection. For high-resolution
    inputs, we use larger imgsz values to preserve detail.

    Args:
        frame: numpy array of shape (H, W, 3)

    Returns:
        imgsz value to pass to model inference (640, 960, or 1280)

    Examples:
        480p video  → imgsz=640  (upscale to improve detection)
        720p video  → imgsz=960  (moderate upscale)
        1080p video → imgsz=1280 (preserve detail)
    """
    h, w = frame.shape[:2]
    longest_side = max(h, w)

    if longest_side <= 480:
        return 640      # Low-res: aggressive upscale helps detect small objects
    elif longest_side <= 720:
        return 960      # Medium-res: moderate upscale
    else:
        return 1280     # High-res: preserve detail, minimal upscale
