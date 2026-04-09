"""
Training script for YOLOv8x object detector.

This script fine-tunes a pretrained YOLOv8x model on a custom dataset
and exports the trained weights to TensorRT format for optimized inference.
"""

from ultralytics import YOLO
import os
import torch


def check_prerequisites():
    """Verify that all requirements for training are met."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available. Check GPU drivers and CUDA installation.")

    if not os.path.exists('data/dataset.yaml'):
        raise FileNotFoundError(
            "data/dataset.yaml not found. Run dataset.py first to prepare the dataset."
        )

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print("Prerequisites OK. Starting training...\n")


def train():
    """
    Fine-tune YOLOv8x on custom dataset.

    Training parameters:
    - epochs=50: Maximum number of training passes through the dataset
    - imgsz=640: Resize all images to 640x640 for training
    - batch=16: Process 16 images simultaneously (optimized for 24GB VRAM)
    - patience=10: Early stopping if no improvement for 10 consecutive epochs
    - augment=True: Apply random augmentations (flips, rotations, crops)
    - cos_lr=True: Cosine learning rate schedule for better convergence
    """
    # Load pretrained YOLOv8x model (trained on COCO dataset)
    model = YOLO('yolov8x.pt')

    print("Starting training with YOLOv8x pretrained weights...\n")

    results = model.train(
        data='data/dataset.yaml',
        epochs=50,
        imgsz=640,
        batch=16,
        device=0,
        optimizer='AdamW',
        lr0=0.001,            # Initial learning rate
        lrf=0.01,             # Final learning rate (1% of initial)
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,      # Gradual warmup for first 3 epochs
        patience=10,          # Early stopping patience
        augment=True,         # Enable data augmentation
        cos_lr=True,          # Cosine learning rate scheduler
        save=True,
        save_period=10,       # Save checkpoint every 10 epochs
        project='runs/train',
        name='yolov8x_custom',
        exist_ok=True,
        verbose=True
    )

    # Extract and display final metrics
    map50 = results.results_dict.get('metrics/mAP50(B)', 0)
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Best mAP@0.5: {map50:.4f}")
    print(f"Best weights: runs/train/yolov8x_custom/weights/best.pt")

    # Interpret results for user
    if map50 >= 0.70:
        verdict = "EXCELLENT — model is ready for production"
    elif map50 >= 0.60:
        verdict = "GOOD — model should work well for most cases"
    elif map50 >= 0.50:
        verdict = "ACCEPTABLE — consider collecting more data or training longer"
    else:
        verdict = "NEEDS IMPROVEMENT — check data quality and class distribution"

    print(f"Result: {verdict}")
    print(f"{'='*60}\n")

    return results


def export_to_tensorrt():
    """
    Convert trained PyTorch weights to TensorRT engine file.

    TensorRT compilation optimizes the model specifically for the target GPU,
    resulting in approximately 3x faster inference with FP16 precision.
    The .engine file is hardware-specific and cannot be transferred to different GPUs.
    """
    best_weights = 'runs/train/yolov8x_custom/weights/best.pt'

    if not os.path.exists(best_weights):
        raise FileNotFoundError(
            f"Trained weights not found at {best_weights}. "
            "Training may not have completed successfully."
        )

    print(f"Exporting to TensorRT (this takes approximately 2 minutes)...")
    print(f"Source: {best_weights}")

    model = YOLO(best_weights)
    model.export(
        format='engine',
        device=0,
        half=True,          # FP16 precision for faster inference
        imgsz=640,
        simplify=True       # Apply graph optimizations
    )

    # Move engine to exports/ directory
    engine_src = best_weights.replace('.pt', '.engine')
    os.makedirs('exports', exist_ok=True)
    engine_dest = 'exports/yolov8x_custom.engine'

    if os.path.exists(engine_dest):
        os.remove(engine_dest)
    os.rename(engine_src, engine_dest)

    print(f"\n{'='*60}")
    print(f"TensorRT engine successfully exported")
    print(f"Location: {engine_dest}")
    print(f"{'='*60}\n")


def main():
    """Main entry point for training pipeline."""
    print("\n" + "="*60)
    print("YOLOV8X TRAINING PIPELINE")
    print("="*60 + "\n")

    # Step 1: Verify prerequisites
    check_prerequisites()

    # Step 2: Train the model
    train()

    # Step 3: Export to TensorRT
    export_to_tensorrt()

    print("Training pipeline complete!")
    print("Next steps:")
    print("  1. Run src/eval.py to evaluate model performance")
    print("  2. Run src/pipeline.py to process videos")


if __name__ == '__main__':
    main()