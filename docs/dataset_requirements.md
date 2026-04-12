# Dataset Requirements
## Object Detection Project — Send this to your colleague

---

## Summary
We need a labeled image dataset for training a YOLOv8x object detection model.
The model will detect common indoor objects in video footage.

---

## Format Required: YOLO format

Each image must have a paired .txt label file with the same filename.

**Image:** `images/train/frame_001.jpg`
**Label:** `labels/train/frame_001.txt`

Each line in the .txt file = one object in that image:
```
class_id  x_center  y_center  width  height
```

All coordinates are **normalized** (0.0 to 1.0), not pixels:
- `x_center` = horizontal center of box / image width
- `y_center` = vertical center of box / image height
- `width`    = box width / image width
- `height`   = box height / image height

Example (a person in the center of a 1920x1080 image):
```
0 0.500 0.500 0.250 0.700
```

---

## Folder Structure Required

```
dataset/
├── images/
│   ├── train/     ← 70% of total images
│   ├── val/       ← 15% of total images
│   └── test/      ← 15% of total images
├── labels/
│   ├── train/     ← .txt files matching train images
│   ├── val/       ← .txt files matching val images
│   └── test/      ← .txt files matching test images
└── classes.txt    ← one class name per line, in ID order
```

---

## Classes Required (23 classes)

```
ID    Class name
 0    person
 1    chair
 2    laptop
 3    mouse
 4    keyboard
 5    monitor
 6    mobile phone
 7    bottle
 8    cup
 9    book
10    backpack
11    handbag
12    umbrella
13    dog
14    cat
15    bird
16    bicycle
17    car
18    motorbike
19    dining table
20    couch
21    potted plant
22    bed
```

If you have additional custom classes, append them starting at ID 23.

---

## Quantity Requirements

| Requirement | Minimum | Ideal |
|-------------|---------|-------|
| Images per class (train) | 300 | 500 |
| Total images | ~7,000 | ~11,500 |
| Val images | ~1,050 | ~1,725 |
| Test images | ~1,050 | ~1,725 |

---

## Image Requirements

| Property | Requirement |
|----------|-------------|
| Format | .jpg or .png |
| Resolution | 480p to 1080p (mixed is fine) |
| Variety | Different angles, lighting, backgrounds |
| Quality | Clear, not heavily blurred or over-compressed |
| Source | Indoor scenes, offices, homes, public spaces |

**Important:** Variety matters more than quantity.
500 images of the same chair from the same angle = poor training.
200 images of chairs from different angles, lighting, environments = better.

---

## What makes a good annotation?

- **Tight boxes** — box should fit the object boundary closely, no large padding
- **Full objects** — include the entire visible object, even if partially occluded
- **Every instance** — label EVERY object of each class in the image, not just prominent ones
- **Minimum size** — skip objects smaller than 10×10 pixels (too small to be useful)
- **One box per object** — never draw two boxes on the same object

---

## Optional: dataset.yaml

If you can provide this file, it saves us a step:

```yaml
path: dataset/
train: images/train
val: images/val
test: images/test
nc: 23
names:
  0: person
  1: chair
  2: laptop
  # ... etc
```

---

## If using an existing dataset (e.g. from Roboflow, COCO, etc.)

We can work with:
- COCO format (annotations.json) — we'll convert
- Pascal VOC format (.xml files) — we'll convert
- YOLO format (.txt files) — ready to use

Just let us know the format and we'll handle conversion.

---

## Questions?
Contact: [your name/email]
