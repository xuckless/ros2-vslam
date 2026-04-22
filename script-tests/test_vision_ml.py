#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import cv2
import time
import math
from ultralytics import YOLO

# --- PHYSICAL CONSTANTS ---
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
CAMERA_HEIGHT_CM = 20  # Height of lens from floor
CAMERA_TILT_DEG = 36.0  # Downward tilt from horizontal
V_FOV = 48.0  # Vertical Field of View of your webcam

# --- FINE TUNING ---
DISTANCE_CORRECTION_FACTOR = 1
DISTANCE_OFFSET = -5

MIN_BOX_HEIGHT_PX = 10
YOLO_MODEL_PATH = "/home/agilex/limo5_ws/src/cosc_4366el_a2_g5/cosc_4366el_a2_g5/best.pt"
YOLO_CONFIDENCE = 0.5


def estimate_distance_geometry(y_pixel):
    relative_y = ((IMAGE_HEIGHT / 2) - y_pixel) / (IMAGE_HEIGHT / 2)
    pixel_angle_offset = relative_y * (V_FOV / 2)
    total_angle_deg = (90 - CAMERA_TILT_DEG) + pixel_angle_offset

    if total_angle_deg >= 89:
        return math.inf

    raw_distance = CAMERA_HEIGHT_CM * math.tan(math.radians(total_angle_deg))
    corrected_distance = (raw_distance * DISTANCE_CORRECTION_FACTOR) + DISTANCE_OFFSET

    return max(0, corrected_distance)


class LimoVisionNode(Node):
    def __init__(self):
        super().__init__('limo_vision_node')

        try:
            self.model = YOLO(YOLO_MODEL_PATH)
            self.get_logger().info("YOLO model loaded.")
        except Exception as exc:
            self.get_logger().error(f"YOLO load error: {exc}")
            raise exc

        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, IMAGE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)

        if not self.cap.isOpened():
            self.get_logger().error("Failed to open OpenCV camera 0.")

        self.prev_time = 0
        self.timer = self.create_timer(0.033, self.timer_callback)

    def timer_callback(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        curr_time = time.time()
        fps = 1 / (curr_time - self.prev_time) if (curr_time - self.prev_time) > 0 else 0
        self.prev_time = curr_time

        # Ensure correct size for consistent distance calculations
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        display = frame.copy()

        results = self.model(frame, conf=YOLO_CONFIDENCE, verbose=False)
        result = results[0]

        # --- CUSTOM STRICT FILTERING ---
        if len(result.boxes) > 0:
            keep_indices = []
            boxes = result.boxes.xyxy

            img_h, img_w = result.orig_shape
            max_allowed_area = 0.20 * (img_w * img_h)

            for i in range(len(boxes)):
                box_a = boxes[i]

                # 1. Check if box is egregiously big
                box_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                if box_area > max_allowed_area:
                    continue

                    # 2. Check for encapsulation
                is_encapsulating = False
                for j in range(len(boxes)):
                    if i == j:
                        continue
                    box_b = boxes[j]

                    if (box_a[0] <= box_b[0] and box_a[1] <= box_b[1] and
                            box_a[2] >= box_b[2] and box_a[3] >= box_b[3]):
                        is_encapsulating = True
                        break

                if not is_encapsulating:
                    keep_indices.append(i)

            result.boxes = result.boxes[keep_indices]
        # -----------------------------------------------------------

        # Draw retained boxes and calculate geometry-based distance
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            if (y2 - y1) < MIN_BOX_HEIGHT_PX:
                continue

            label = str(result.names[int(box.cls[0])]).upper()
            conf = float(box.conf[0])

            dist_cm = estimate_distance_geometry(float(y2))

            cv2.rectangle(display, (x1, y1), (x2, y2), (255, 50, 50), 2)
            dist_txt = f"{dist_cm:.1f}cm" if dist_cm != math.inf else "Too Far"
            cv2.putText(display, f"{label} {conf:.2f} | {dist_txt}",
                        (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

            self.get_logger().info(f"Detected: {label} | Bottom Pixel: {y2} | Distance: {dist_txt}")

        cv2.putText(display, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("LIMO ML & Geometry Distance", display)
        cv2.waitKey(1)

    def destroy_node(self):
        self.cap.release()
        cv2.destroyAllWindows()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LimoVisionNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()