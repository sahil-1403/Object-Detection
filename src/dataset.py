"""
Dataset validation and conversion script for YOLOv8x object detection.

This module handles dataset format detection (YOLO/COCO/VOC), conversion to
YOLO format if needed, validation of bounding box annotations, and generation
of the dataset.yaml configuration file required for training.
"""

import os
import yaml
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict


def detect_format(dataset_dir: str) -> str:
    """
    Detect the annotation format of a dataset.

    Args:
        dataset_dir: Path to dataset root directory

    Returns:
        'yolo', 'coco', or 'voc'
    """
    dataset_path = Path(dataset_dir)

    # Check for YOLO format (labels/ directory with .txt files)
    labels_dir = dataset_path / 'labels'
    if labels_dir.exists() and any(labels_dir.rglob('*.txt')):
        return 'yolo'

    # Check for COCO format (annotations.json file)
    if (dataset_path / 'annotations.json').exists():
        return 'coco'

    # Check for Pascal VOC format (.xml files)
    if any(dataset_path.rglob('*.xml')):
        return 'voc'

    raise ValueError("Could not detect dataset format. Expected YOLO, COCO, or VOC structure.")


def convert_voc_to_yolo(dataset_dir: str, class_mapping: dict):
    """
    Convert Pascal VOC XML annotations to YOLO format.

    Args:
        dataset_dir: Path to dataset root
        class_mapping: Dict mapping class names to integer IDs
    """
    dataset_path = Path(dataset_dir)

    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        if not images_dir.exists():
            continue

        labels_dir = dataset_path / 'labels' / split
        labels_dir.mkdir(parents=True, exist_ok=True)

        # Find all XML files
        xml_files = list(images_dir.glob('*.xml'))

        for xml_file in xml_files:
            tree = ET.parse(xml_file)
            root = tree.getroot()

            # Get image dimensions
            size = root.find('size')
            img_width = float(size.find('width').text)
            img_height = float(size.find('height').text)

            # Process each object
            yolo_annotations = []
            for obj in root.findall('object'):
                class_name = obj.find('name').text
                if class_name not in class_mapping:
                    print(f"WARNING: Unknown class '{class_name}' in {xml_file.name}, skipping")
                    continue

                class_id = class_mapping[class_name]

                bbox = obj.find('bndbox')
                xmin = float(bbox.find('xmin').text)
                ymin = float(bbox.find('ymin').text)
                xmax = float(bbox.find('xmax').text)
                ymax = float(bbox.find('ymax').text)

                # Convert to YOLO format (normalized center coordinates + size)
                x_center = (xmin + xmax) / 2 / img_width
                y_center = (ymin + ymax) / 2 / img_height
                width = (xmax - xmin) / img_width
                height = (ymax - ymin) / img_height

                yolo_annotations.append(f"{class_id} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

            # Write YOLO format label file
            label_file = labels_dir / f"{xml_file.stem}.txt"
            with open(label_file, 'w') as f:
                f.write('\n'.join(yolo_annotations))


def convert_to_yolo(dataset_dir: str, format_type: str):
    """
    Convert dataset to YOLO format if needed.

    Args:
        dataset_dir: Path to dataset root
        format_type: Detected format ('yolo', 'coco', or 'voc')
    """
    if format_type == 'yolo':
        print("Format is already YOLO — no conversion needed.")
        return

    if format_type == 'coco':
        print("Converting COCO format to YOLO...")
        from ultralytics.data.converter import convert_coco
        convert_coco(labels_dir=f'{dataset_dir}/coco_annotations/', save_dir=dataset_dir)
        print("COCO conversion complete.")

    elif format_type == 'voc':
        print("Converting Pascal VOC format to YOLO...")
        # Load class mapping
        class_mapping = get_default_class_mapping()
        convert_voc_to_yolo(dataset_dir, class_mapping)
        print("VOC conversion complete.")


def get_default_class_mapping():
    """Return the default class ID to name mapping."""
    return {
        'person': 0, 'chair': 1, 'laptop': 2, 'mouse': 3, 'keyboard': 4,
        'monitor': 5, 'mobile phone': 6, 'bottle': 7, 'cup': 8, 'book': 9,
        'backpack': 10, 'handbag': 11, 'umbrella': 12, 'dog': 13, 'cat': 14,
        'bird': 15, 'bicycle': 16, 'car': 17, 'motorbike': 18, 'dining table': 19,
        'couch': 20, 'potted plant': 21, 'bed': 22
    }


def build_class_list(dataset_dir: str) -> dict:
    """
    Build a class list by scanning all label files.

    Args:
        dataset_dir: Path to dataset root

    Returns:
        Dict mapping class IDs to class names
    """
    dataset_path = Path(dataset_dir)
    labels_dir = dataset_path / 'labels'

    # Scan for all unique class IDs
    found_class_ids = set()
    for label_file in labels_dir.rglob('*.txt'):
        if label_file.stat().st_size == 0:
            continue
        with open(label_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    found_class_ids.add(class_id)

    # Map to default class names
    default_names = {v: k for k, v in get_default_class_mapping().items()}
    class_list = {}
    for class_id in sorted(found_class_ids):
        if class_id in default_names:
            class_list[class_id] = default_names[class_id]
        else:
            class_list[class_id] = f"class_{class_id}"

    return class_list


def validate_dataset(dataset_dir: str) -> tuple[bool, dict]:
    """
    Validate dataset integrity and compute statistics.

    Args:
        dataset_dir: Path to dataset root

    Returns:
        Tuple of (is_valid, stats_dict)
    """
    dataset_path = Path(dataset_dir)
    is_valid = True
    stats = defaultdict(lambda: {'train': 0, 'val': 0, 'test': 0})

    for split in ['train', 'val', 'test']:
        images_dir = dataset_path / 'images' / split
        labels_dir = dataset_path / 'labels' / split

        if not images_dir.exists():
            print(f"WARNING: {split} split not found at {images_dir}")
            continue

        image_files = list(images_dir.glob('*.jpg')) + list(images_dir.glob('*.png'))

        if len(image_files) == 0:
            print(f"ERROR: No images found in {split} split")
            is_valid = False
            continue

        for img_file in image_files:
            label_file = labels_dir / f"{img_file.stem}.txt"

            if not label_file.exists():
                print(f"ERROR: Missing label file for {img_file.name}")
                is_valid = False
                continue

            # Validate label file
            if label_file.stat().st_size == 0:
                print(f"WARN: Empty label file {label_file.name} (image may have no objects)")
                continue

            with open(label_file, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue

                    class_id = int(parts[0])
                    x_center, y_center, width, height = map(float, parts[1:5])

                    # Validate coordinates are in 0-1 range
                    if not (0 <= x_center <= 1 and 0 <= y_center <= 1 and
                            0 <= width <= 1 and 0 <= height <= 1):
                        print(f"ERROR: Invalid bbox coordinates in {label_file.name}: {line.strip()}")
                        is_valid = False

                    # Count class occurrences
                    stats[class_id][split] += 1

    return is_valid, dict(stats)


def generate_dataset_yaml(dataset_dir: str, class_list: dict):
    """
    Generate data/dataset.yaml configuration file for Ultralytics.

    Args:
        dataset_dir: Path to dataset root
        class_list: Dict mapping class IDs to class names
    """
    config = {
        'path': str(Path(dataset_dir).absolute()),
        'train': 'images/train',
        'val': 'images/val',
        'test': 'images/test',
        'nc': len(class_list),
        'names': class_list
    }

    yaml_path = Path(dataset_dir) / 'dataset.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)

    print(f"Generated dataset.yaml at: {yaml_path}")


def print_statistics(class_list: dict, stats: dict):
    """
    Print class distribution table.

    Args:
        class_list: Dict mapping class IDs to class names
        stats: Dict of class statistics
    """
    print("\n" + "="*70)
    print("CLASS DISTRIBUTION")
    print("="*70)
    print(f"{'Class':<25} {'Train':>8} {'Val':>8} {'Test':>8} {'Total':>8}")
    print("-"*70)

    total_train = total_val = total_test = 0

    for class_id in sorted(class_list.keys()):
        class_name = class_list[class_id]
        train = stats.get(class_id, {}).get('train', 0)
        val = stats.get(class_id, {}).get('val', 0)
        test = stats.get(class_id, {}).get('test', 0)
        total = train + val + test

        total_train += train
        total_val += val
        total_test += test

        print(f"{class_name:<25} {train:>8} {val:>8} {test:>8} {total:>8}")

        if train < 300:
            print(f"  └─ WARNING: Only {train} train samples (minimum recommended: 300)")

    print("-"*70)
    print(f"{'TOTAL':<25} {total_train:>8} {total_val:>8} {total_test:>8} {total_train+total_val+total_test:>8}")
    print("="*70 + "\n")


def main():
    """Main entry point for dataset preparation."""
    dataset_dir = 'data'

    # Step 1: Detect format
    print("Detecting dataset format...")
    format_type = detect_format(dataset_dir)
    print(f"Detected format: {format_type.upper()}")

    # Step 2: Convert if needed
    convert_to_yolo(dataset_dir, format_type)

    # Step 3: Build class list
    print("\nBuilding class list...")
    class_list = build_class_list(dataset_dir)
    print(f"Found {len(class_list)} classes")

    # Step 4: Validate dataset
    print("\nValidating dataset...")
    is_valid, stats = validate_dataset(dataset_dir)

    # Step 5: Generate dataset.yaml
    generate_dataset_yaml(dataset_dir, class_list)

    # Step 6: Print statistics
    print_statistics(class_list, stats)

    # Step 7: Summary
    print("\nDATASET SUMMARY:")
    print(f"  Format detected: {format_type.upper()}")
    print(f"  Conversion: {'done' if format_type != 'yolo' else 'skipped'}")
    print(f"  Total classes: {len(class_list)}")

    train_count = sum(s.get('train', 0) for s in stats.values())
    val_count = sum(s.get('val', 0) for s in stats.values())
    test_count = sum(s.get('test', 0) for s in stats.values())

    print(f"  Train samples: {train_count}")
    print(f"  Val samples:   {val_count}")
    print(f"  Test samples:  {test_count}")
    print(f"  Validation: {'PASSED' if is_valid else 'FAILED'}")
    print(f"  dataset.yaml: data/dataset.yaml")


if __name__ == '__main__':
    main()
