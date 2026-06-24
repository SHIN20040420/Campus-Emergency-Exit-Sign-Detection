import argparse

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect campus emergency exit signs from camera.")
    parser.add_argument("--cascade", required=True, help="Path to trained cascade.xml.")
    parser.add_argument("--camera", type=int, default=0, help="Camera index.")
    parser.add_argument("--scale-factor", type=float, default=1.08, help="Scale factor for detectMultiScale.")
    parser.add_argument("--min-neighbors", type=int, default=5, help="Minimum neighbors for detectMultiScale.")
    parser.add_argument("--min-width", type=int, default=24, help="Minimum detection width.")
    parser.add_argument("--min-height", type=int, default=12, help="Minimum detection height.")
    args = parser.parse_args()

    cascade = cv2.CascadeClassifier(args.cascade)
    if cascade.empty():
        raise FileNotFoundError(f"Cannot load cascade: {args.cascade}")

    camera = cv2.VideoCapture(args.camera)
    if not camera.isOpened():
        raise RuntimeError(f"Cannot open camera index: {args.camera}")

    while True:
        ok, frame = camera.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        signs = cascade.detectMultiScale(
            gray,
            scaleFactor=args.scale_factor,
            minNeighbors=args.min_neighbors,
            minSize=(args.min_width, args.min_height),
        )

        for x, y, w, h in signs:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "escape_sign",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Campus Emergency Exit Sign Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
