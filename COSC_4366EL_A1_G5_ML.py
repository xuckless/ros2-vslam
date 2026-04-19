#!/usr/bin/env python3
import cv2
import time
import math
from ultralytics import YOLO

"""
Credits: Syed Ali; Roboflow; Gemini
"""

# --- PHYSICAL CONSTANTS ---
IMAGE_HEIGHT = 480
CAMERA_HEIGHT_CM = 18.5  # Height of lens from floor
CAMERA_TILT_DEG = 35.0  # Downward tilt from horizontal
V_FOV = 48.0  # Vertical Field of View of your webcam

# --- FINE TUNING ---
DISTANCE_CORRECTION_FACTOR = .9  # Multiplier (e.g., 0.9 for 10% less)
DISTANCE_OFFSET = 0.0  # Constant addition/subtraction in cm


def estimate_distance_geometry(y_pixel):
    """
    Calculates distance. Inverting the relative_y ensures that
    higher pixel values (bottom of screen) result in shorter distances.
    """
    # Inverted: Bottom of screen (y=480) becomes -1, center 0, top 1
    # This aligns the pixel coordinate system with the downward tilt
    relative_y = ((IMAGE_HEIGHT / 2) - y_pixel) / (IMAGE_HEIGHT / 2)

    # Angle of the specific pixel relative to the camera optical axis
    pixel_angle_offset = relative_y * (V_FOV / 2)

    # Calculate angle relative to the floor
    # As pixel_angle_offset becomes more negative (bottom of screen),
    # the total angle decreases, bringing the distance closer.
    total_angle_deg = (90 - CAMERA_TILT_DEG) + pixel_angle_offset

    # Prevent tangent of 90 or negative angles
    if total_angle_deg >= 89:
        return math.inf

    raw_distance = CAMERA_HEIGHT_CM * math.tan(math.radians(total_angle_deg))

    # Apply fine-tuning
    corrected_distance = (raw_distance * DISTANCE_CORRECTION_FACTOR) + DISTANCE_OFFSET

    return max(0, corrected_distance)


def trackSign_ML(frame, model):
    results = model(frame, conf=0.5, verbose=False)

    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])

            if (y2 - y1) < 10:
                continue

            label = r.names[int(box.cls[0])]
            conf = float(box.conf[0])

            # Calculate distance
            dist_cm = estimate_distance_geometry(float(y2))

            # Visuals
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 50, 50), 2)

            dist_txt = f"{dist_cm:.1f}cm" if dist_cm != math.inf else "Too Far"
            info_text = f"{label.upper()} {conf:.2f} | {dist_txt}"
            cv2.putText(frame, info_text, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

            # Debugging print
            print(f"Bottom Pixel: {y2} | Distance: {dist_txt}")

    return frame


def main():
    try:
        model = YOLO("weights/best.pt")
    except Exception as e:
        print(f"Error: {e}")
        return

    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

    prev_time = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        processed_frame = trackSign_ML(frame, model)

        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Sign Detection (Fixed Inversion)", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()