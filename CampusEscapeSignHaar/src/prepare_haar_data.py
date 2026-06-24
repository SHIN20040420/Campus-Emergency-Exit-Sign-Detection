import argparse
import os
import xml.etree.ElementTree as ET
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_boxes(xml_path: Path) -> list[tuple[int, int, int, int]]:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    boxes: list[tuple[int, int, int, int]] = []

    for obj in root.findall("object"):
        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = int(float(bndbox.findtext("xmin", "0")))
        ymin = int(float(bndbox.findtext("ymin", "0")))
        xmax = int(float(bndbox.findtext("xmax", "0")))
        ymax = int(float(bndbox.findtext("ymax", "0")))

        width = max(0, xmax - xmin)
        height = max(0, ymax - ymin)
        if width > 0 and height > 0:
            boxes.append((xmin, ymin, width, height))

    return boxes


def find_image_for_xml(xml_path: Path, positive_dir: Path) -> Path | None:
    for extension in IMAGE_EXTENSIONS:
        candidate = positive_dir / f"{xml_path.stem}{extension}"
        if candidate.exists():
            return candidate
    return None


def write_positive_info(positive_dir: Path, output_path: Path) -> int:
    count = 0
    lines: list[str] = []

    for xml_path in sorted(positive_dir.glob("*.xml")):
        image_path = find_image_for_xml(xml_path, positive_dir)
        if image_path is None:
            print(f"Skip XML without image: {xml_path}")
            continue

        boxes = parse_boxes(xml_path)
        if not boxes:
            print(f"Skip XML without valid box: {xml_path}")
            continue

        box_text = " ".join(f"{x} {y} {w} {h}" for x, y, w, h in boxes)
        lines.append(f"{image_path.as_posix()} {len(boxes)} {box_text}")
        count += len(boxes)

    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return count


def write_negative_info(negative_dir: Path, output_path: Path) -> int:
    image_paths = [
        path
        for path in sorted(negative_dir.rglob("*"))
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]

    lines = [path.as_posix() for path in image_paths]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare OpenCV Haar Cascade training lists.")
    parser.add_argument("--dataset", default="dataset", help="Dataset folder containing positive and negative folders.")
    parser.add_argument("--output", default="training", help="Output folder for positives.txt and negatives.txt.")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset)
    positive_dir = dataset_dir / "positive"
    negative_dir = dataset_dir / "negative"
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not positive_dir.exists():
        raise FileNotFoundError(f"Missing positive directory: {positive_dir}")
    if not negative_dir.exists():
        raise FileNotFoundError(f"Missing negative directory: {negative_dir}")

    positive_boxes = write_positive_info(positive_dir, output_dir / "positives.txt")
    negative_images = write_negative_info(negative_dir, output_dir / "negatives.txt")

    print(f"Positive boxes: {positive_boxes}")
    print(f"Negative images: {negative_images}")
    print(f"Wrote: {output_dir / 'positives.txt'}")
    print(f"Wrote: {output_dir / 'negatives.txt'}")


if __name__ == "__main__":
    main()
