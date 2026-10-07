import argparse
from pathlib import Path
import cv2


def detect_vehicles(input_path: str, output_path: str) -> None:
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {input_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps > 1 else 25.0

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    subtractor = cv2.createBackgroundSubtractorMOG2(
        history=500, varThreshold=40, detectShadows=True
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        mask = subtractor.apply(frame)
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.dilate(mask, kernel, iterations=2)

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        vehicle_count = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 1200:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            if w < 25 or h < 25:
                continue
            vehicle_count += 1
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                frame,
                "Vehicle",
                (x, max(20, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )

        cv2.putText(
            frame,
            f"Vehicles: {vehicle_count}",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )
        writer.write(frame)

    cap.release()
    writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(description="Traffic video vehicle detector")
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument(
        "--output",
        default="output/detected_traffic.mp4",
        help="Output video path",
    )
    args = parser.parse_args()
    detect_vehicles(args.input, args.output)
    print(f"Saved processed video to {args.output}")


if __name__ == "__main__":
    main()
