import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect emergency exit signs in one image.")
    parser.add_argument("--cascade", required=True, help="Path to trained cascade.xml.")
    parser.add_argument("--image", required=True, help="Input image path.")
    parser.add_argument("--output", default="result.jpg", help="Output image path.")
    parser.add_argument("--scale-factor", type=float, default=1.08)
    parser.add_argument("--min-neighbors", type=int, default=5)
    parser.add_argument("--min-width", type=int, default=24)
    parser.add_argument("--min-height", type=int, default=12)
    args = parser.parse_args()

    cascade = cv2.CascadeClassifier(args.cascade)
    if cascade.empty():
        raise FileNotFoundError(f"Cannot load cascade: {args.cascade}")

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {args.image}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    signs = cascade.detectMultiScale(
        gray,
        scaleFactor=args.scale_factor,
        minNeighbors=args.min_neighbors,
        minSize=(args.min_width, args.min_height),
    )

    for x, y, w, h in signs:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(
            image,
            "escape_sign",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), image)
    print(f"Detections: {len(signs)}")
    print(f"Wrote: {output_path}")


if __name__ == "__main__":
    main()
