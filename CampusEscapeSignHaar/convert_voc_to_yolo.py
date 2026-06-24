import os
import random
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

IMG_DIR = Path("dataset/positive/img")
XML_DIR = Path("dataset/positive/processed")

OUT_DIR = Path("yolo_dataset")

for folder in [
    OUT_DIR / "images/train",
    OUT_DIR / "images/val",
    OUT_DIR / "labels/train",
    OUT_DIR / "labels/val",
]:
    folder.mkdir(parents=True, exist_ok=True)

images = sorted(list(IMG_DIR.glob("*.jpg")))
random.shuffle(images)

split_idx = int(len(images) * 0.8)

train_imgs = images[:split_idx]
val_imgs = images[split_idx:]

def convert_box(size, box):
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]

    x = (box[0] + box[1]) / 2.0
    y = (box[2] + box[3]) / 2.0

    w = box[1] - box[0]
    h = box[3] - box[2]

    return (
        x * dw,
        y * dh,
        w * dw,
        h * dh
    )

def process_image(img_path, split):

    xml_path = XML_DIR / (img_path.stem + ".xml")

    if not xml_path.exists():
        print(f"Missing XML: {xml_path}")
        return

    tree = ET.parse(xml_path)
    root = tree.getroot()

    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    label_lines = []

    for obj in root.findall("object"):

        bbox = obj.find("bndbox")

        xmin = float(bbox.find("xmin").text)
        xmax = float(bbox.find("xmax").text)
        ymin = float(bbox.find("ymin").text)
        ymax = float(bbox.find("ymax").text)

        x, y, w, h = convert_box(
            (width, height),
            (xmin, xmax, ymin, ymax)
        )

        label_lines.append(
            f"0 {x} {y} {w} {h}"
        )

    shutil.copy(
        img_path,
        OUT_DIR / f"images/{split}" / img_path.name
    )

    txt_path = (
        OUT_DIR /
        f"labels/{split}" /
        (img_path.stem + ".txt")
    )

    with open(txt_path, "w") as f:
        f.write("\n".join(label_lines))

for img in train_imgs:
    process_image(img, "train")

for img in val_imgs:
    process_image(img, "val")

print("Done.")