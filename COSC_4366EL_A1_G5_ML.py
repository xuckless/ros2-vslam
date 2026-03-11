import cv2
import time
from ultralytics import YOLO

"""
Credits: Syed Ali; Roboflow; Gemini
"""


def trackSign_ML(frame, model):
    """
    Detects visual signs using a trained YOLO model and overlays bounding boxes,
    labels, confidence, and estimated distance.
    """
    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            # Extract bounding box coordinates
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # Extract class ID, label name, and confidence score
            cls_id = int(box.cls[0])
            label = r.names[cls_id]
            conf = float(box.conf[0])

            # Distance Estimation (Heuristic based on bounding box height)
            box_height = y2 - y1
            estimated_distance_cm = (800 / box_height) if box_height > 0 else 0

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)

            # Overlay label, confidence, and distance
            info_text = f"{label} {conf:.2f} | Dist: {estimated_distance_cm:.1f}cm"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

    return frame


def main():
    # Load the trained YOLO model (Replace 'best_yolo.pt' with your actual exported weights file)
    try:
        model = YOLO("best_yolo.pt")
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure your trained 'best_yolo.pt' file is in the same directory.")
        return

    # Initialize Camera (Targeting 640x480 for >= 10-15 FPS)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0

    print("YOLO ML Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Process frame
        processed_frame = trackSign_ML(frame, model)

        # Draw FPS counter
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Display output
        cv2.imshow("Visual Sign Detection (ML)", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()