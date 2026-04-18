#!/usr/bin/env python3
import cv2
import math
from ultralytics import YOLO

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
FLOOR_DIST_K = 5000.0
MIN_BOX_HEIGHT_PX = 10
YOLO_MODEL_PATH = "best_yolo.pt"  # Change this to 'yolov8n.pt' to test without your custom weights
YOLO_CONFIDENCE = 0.5


def floor_distance(box_bottom_y):
    denom = IMAGE_HEIGHT - box_bottom_y
    if denom < 1:
        return math.inf
    return FLOOR_DIST_K / denom


def main():
    try:
        model = YOLO(YOLO_MODEL_PATH)
        print("YOLO model loaded.")
    except Exception as exc:
        print(f"YOLO load error: {exc}")
        return

    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        display = frame.copy()

        results = model(frame, conf=YOLO_CONFIDENCE, verbose=False)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (y2 - y1) < MIN_BOX_HEIGHT_PX:
                    continue

                label = str(r.names[int(box.cls[0])]).upper()
                conf = float(box.conf[0])
                dist_cm = floor_distance(float(y2))

                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 50, 50), 2)

                dist_txt = f"{dist_cm:.1f}cm" if dist_cm != math.inf else "far"
                cv2.putText(display, f"{label} {conf:.2f} | {dist_txt}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

                print(f"Detected: {label} | Distance: {dist_txt}")

        cv2.imshow("ML Decision & Distance Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()