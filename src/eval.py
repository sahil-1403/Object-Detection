"""
Evaluation script for YOLOv8x object detector.

Measures model accuracy on the held-out test set and generates detailed
per-class performance metrics, confusion matrix, and precision-recall curves.
"""

from ultralytics import YOLO
import json
import os


def evaluate():
    """
    Run model evaluation on test split and generate detailed metrics.

    Uses the trained PyTorch weights (best.pt) rather than TensorRT engine
    because Ultralytics provides richer metric reporting with .pt format.
    """
    best_weights = 'runs/train/yolov8x_custom/weights/best.pt'

    if not os.path.exists(best_weights):
        raise FileNotFoundError(
            f"Trained weights not found at {best_weights}. "
            "Run train.py first to train the model."
        )

    print("="*60)
    print("MODEL EVALUATION")
    print("="*60)
    print(f"\nLoading model from: {best_weights}")
    model = YOLO(best_weights)

    print("Running evaluation on test split...\n")
    metrics = model.val(
        data='data/dataset.yaml',
        split='test',
        device=0,
        imgsz=640,
        verbose=True,
        save_json=True,     # Save COCO-format results for external analysis
        plots=True          # Generate confusion matrix and PR curves
    )

    # Extract overall metrics
    names = model.names
    overall = {
        'mAP50':     round(float(metrics.box.map50), 4),
        'mAP50_95':  round(float(metrics.box.map), 4),
        'precision': round(float(metrics.box.mp), 4),
        'recall':    round(float(metrics.box.mr), 4),
    }

    # Extract per-class metrics
    per_class = {}
    for i, name in names.items():
        ap50 = float(metrics.box.ap50[i]) if i < len(metrics.box.ap50) else 0.0
        per_class[name] = {
            'AP50': round(ap50, 4),
            'status': 'NEEDS_MORE_DATA' if ap50 < 0.50 else 'OK'
        }

    results = {
        'overall': overall,
        'per_class': per_class
    }

    # Save metrics to JSON file
    os.makedirs('outputs', exist_ok=True)
    metrics_file = 'outputs/eval_metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(results, f, indent=2)

    # Print detailed report
    print(f"\n{'='*60}")
    print("EVALUATION RESULTS (test set)")
    print(f"{'='*60}")
    print(f"  mAP@0.5:       {overall['mAP50']:.4f}")
    print(f"  mAP@0.5:0.95:  {overall['mAP50_95']:.4f}")
    print(f"  Precision:     {overall['precision']:.4f}")
    print(f"  Recall:        {overall['recall']:.4f}")

    # Overall verdict based on mAP@0.5
    m = overall['mAP50']
    if m >= 0.80:
        verdict = "EXCELLENT — ready for production"
    elif m >= 0.70:
        verdict = "GOOD — production ready"
    elif m >= 0.60:
        verdict = "ACCEPTABLE — consider collecting more data"
    else:
        verdict = "NEEDS IMPROVEMENT — check data quality and class distribution"

    print(f"\n  Verdict: {verdict}")

    # Per-class breakdown
    print(f"\n  Per-class AP@0.5:")
    print(f"  {'Class':<25} {'AP@0.5':>8}  Status")
    print(f"  {'-'*50}")

    weak_classes = []
    for cls, data in sorted(per_class.items(), key=lambda x: x[1]['AP50'], reverse=True):
        flag = ''
        if data['status'] == 'NEEDS_MORE_DATA':
            flag = '  ← needs more data'
            weak_classes.append(cls)
        print(f"  {cls:<25} {data['AP50']:>8.4f}{flag}")

    # Recommendations
    if weak_classes:
        print(f"\n  WARNING: {len(weak_classes)} classes have AP < 0.50:")
        for cls in weak_classes:
            print(f"    - {cls}")
        print(f"\n  Recommendation:")
        print(f"    Collect 200+ additional labeled samples for each flagged class")
        print(f"    and retrain the model to improve performance.")
    else:
        print(f"\n  All classes exceed the 0.50 AP threshold.")

    print(f"\n  Metrics saved to: {metrics_file}")
    print(f"  Confusion matrix: runs/val/yolov8x_custom/confusion_matrix.png")
    print(f"  PR curves: runs/val/yolov8x_custom/PR_curve.png")
    print(f"{'='*60}\n")

    return results


def main():
    """Main entry point for evaluation."""
    results = evaluate()

    print("Evaluation complete!")
    print("\nNext steps:")
    print("  1. Review confusion matrix to identify common misclassifications")
    print("  2. Review PR curves to understand precision-recall tradeoffs")
    print("  3. If satisfied, proceed to run src/pipeline.py on test videos")


if __name__ == '__main__':
    main()
