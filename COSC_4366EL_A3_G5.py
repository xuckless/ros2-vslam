#!/usr/bin/env python3
"""
vision_slam_node.py — AgileX LIMO autonomous navigation node.

Integrates:
  • ROI-based green line following with EMA-smoothed centroid PID  [LimoPathFollower]
  • Geometric YOLO distance estimation + strict box filtering       [LimoVisionNode]
  • Priority-driven, interrupt-capable state machine
  • VSLAM odometry mapping with live top-down visualisation
  • OpenCV direct camera (cv2.VideoCapture(0)) — no ROS image topic

State machine:  FOLLOW_LINE → APPROACHING → TURNING_DYNAMIC
                           ↘ STOPPING
                           ↘ SEARCHING → TURNING_DYNAMIC  (alternate path)
                                       → TURNING_FIXED    (180° retreat)

PID:    Single lateral-offset PID on ROI centroid X, EMA-smoothed.
        Heading angle retained only for horizontal-line rejection and
        SEARCHING alternate-path detection — never fed into PID.
"""

import math
import threading
import json
import time
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from ultralytics import YOLO

# ============================================================
#  CONFIGURATION CONSTANTS
# ============================================================

# -- Odometry --
ODOM_TOPIC_CANDIDATES = [
    '/odom',
    '/limo_base/odom',
    '/ackermann_odom',
    '/wheel/odom',
]

# -- Camera (OpenCV direct, no ROS subscription) ------------
CAMERA_INDEX = 0
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CENTER_X = IMAGE_WIDTH // 2
OVERLAY_ALPHA = 0.5
CAMERA_LOOP_HZ = 30
CAMERA_LOOP_PERIOD = 1.0 / CAMERA_LOOP_HZ

# -- ROI trapezoid (bottom-focused, matches tested tuning) --
ROI_VERTICES = np.array(
    [[(200, 480), (280, 200), (360, 200), (440, 480)]], dtype=np.int32
)

# -- HSV Green Tape -----------------------------------------
HSV_LOWER_GREEN = np.array([35, 40, 40])
HSV_UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 1000
HORIZONTAL_THRESHOLD_DEG = 65  # Contours more horizontal than this are ignored

# -- EMA smoothing ------------------------------------------
EMA_ALPHA = 0.75  # Higher = more responsive, lower = smoother

# -- Geometric Distance Model (tuned on physical robot) -----
CAMERA_HEIGHT_CM = 20.0
CAMERA_TILT_DEG = 36.0
V_FOV = 48.0
DISTANCE_CORRECTION_FACTOR = 1.0
DISTANCE_OFFSET = -5.0
ACTION_DISTANCE_CM = 25.0   # Signs beyond this are ignored
MIN_BOX_HEIGHT_PX = 10

# -- YOLO ---------------------------------------------------
YOLO_MODEL_PATH = (
    "/home/agilex/limo5_ws/src/cosc_4366el_a2_g5/cosc_4366el_a2_g5/best.pt"
)
YOLO_CONFIDENCE = 0.5
MAX_BOX_AREA_RATIO = 0.20   # Boxes > 20% of frame area are rejected as noise

# -- ROS ----------------------------------------------------
PUBLISHER_QUEUE_SIZE = 10
CONTROL_LOOP_HZ = 10
CONTROL_LOOP_PERIOD = 1.0 / CONTROL_LOOP_HZ

# -- Movement & PID (tuned values from LimoPathFollower) ----
FORWARD_SPEED = 0.07        # m/s
TURN_SPEED = 0.30           # rad/s — used during TURNING_DYNAMIC / TURNING_FIXED
SENSITIVITY = 1.0           # Global gain multiplier on PID output
MAX_ANGULAR = 0.20          # rad/s hard clamp on published angular.z

# Lateral centroid PID gains (pixel-space error)
KP = 0.005
KI = 0.0001
KD = 0.001
MAX_WINDUP = 1.0            # Integral anti-windup clamp

SEVERE_LATERAL_OFFSET = 60  # px — speed halved above this to aid recovery

# -- State Machine Timing -----------------------------------
STOP_DURATION_SEC = 3.0
SEARCH_TURN_SPEED = 0.20
SEARCH_TIMEOUT_SEC = 4.0
SEARCH_PATH_ANGLE = 0.40    # rad — heading deviation that signals an alternate path
TURN_DONE_THRESH = 0.08     # rad — yaw error threshold for TURNING_FIXED completion
DYNAMIC_TURN_LATERAL_THRESH = 30   # px — lateral offset to call a dynamic turn done
DYNAMIC_TURN_MIN_DURATION = 0.5    # s  — minimum turn time before alignment is checked
SIGN_COOLDOWN_SEC = 5.0
TURN_DISTANCE_MULTIPLIER = 1.2

# -- Action Priority ----------------------------------------
ACTION_PRIORITY = {
    "NO ENTRY": 4,
    "DEAD END": 4,
    "STOP": 3,
    "RIGHT": 2,
    "LEFT": 2,
    "FORWARD": 1,
}
IDLE_PRIORITY = 0

# -- VSLAM --------------------------------------------------
MAP_SAVE_PATH = "/home/agilex/limo5_ws/vslam_map.json"
MAP_VISUALIZATION_SCALE = 100   # px per metre
MAP_UPDATE_INTERVAL = 0.5       # s between map snapshots


# ============================================================
#  GEOMETRIC DISTANCE ESTIMATION
# ============================================================

def estimate_distance_geometry(y_pixel: float) -> float:
    """
    Floor-projection using camera geometry (height + tilt + vFOV).
    y_pixel is the bottom edge of the bounding box in image coordinates.
    Returns distance in cm; math.inf when the pixel is too close to horizon.
    """
    # Invert so that pixels near the bottom map to shorter distances
    relative_y = ((IMAGE_HEIGHT / 2) - y_pixel) / (IMAGE_HEIGHT / 2)
    pixel_angle_offset = relative_y * (V_FOV / 2)
    total_angle_deg = (90.0 - CAMERA_TILT_DEG) + pixel_angle_offset

    if total_angle_deg >= 89.0:
        return math.inf

    raw_dist = CAMERA_HEIGHT_CM * math.tan(math.radians(total_angle_deg))
    corrected = raw_dist * DISTANCE_CORRECTION_FACTOR + DISTANCE_OFFSET
    return max(0.0, corrected)


# ============================================================
#  VSLAM MAPPER
# ============================================================

class VSLAMMapper:
    """
    Lightweight occupancy-style mapper driven by odometry + YOLO detections.
    Stores the robot trajectory and the global position of every detected sign.
    Saves to JSON on shutdown; renders a top-down OpenCV image at runtime.
    """

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.trajectory: list = []            # [(x, y, yaw, t), ...]
        self.signs: dict = defaultdict(list)  # {label: [(x, y, t), ...]}
        self.lock = threading.Lock()

    # ------------------------------------------------------------------
    def update(self, robot_x: float, robot_y: float, robot_yaw: float,
               detections: list, timestamp: float) -> None:
        with self.lock:
            self.trajectory.append((robot_x, robot_y, robot_yaw, timestamp))
            for det in detections:
                dist_m = det['distance'] / 100.0
                sx = robot_x + dist_m * math.cos(robot_yaw)
                sy = robot_y + dist_m * math.sin(robot_yaw)
                self.signs[det['label']].append((sx, sy, timestamp))

    # ------------------------------------------------------------------
    def save_map(self) -> None:
        with self.lock:
            data = {
                'trajectory': self.trajectory,
                'signs': dict(self.signs),
            }
        try:
            with open(self.save_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as exc:
            print(f"[VSLAMMapper] Save error: {exc}")

    # ------------------------------------------------------------------
    def visualize(self):
        """Returns a white top-down map image, or None if no data yet."""
        with self.lock:
            if not self.trajectory:
                return None
            traj = list(self.trajectory)
            signs_snap = {k: list(v) for k, v in self.signs.items()}

        all_x = [p[0] for p in traj]
        all_y = [p[1] for p in traj]
        for lst in signs_snap.values():
            all_x.extend(s[0] for s in lst)
            all_y.extend(s[1] for s in lst)

        pad = 1.0
        min_x, max_x = min(all_x) - pad, max(all_x) + pad
        min_y, max_y = min(all_y) - pad, max(all_y) + pad

        w = max(400, min(int((max_x - min_x) * MAP_VISUALIZATION_SCALE), 1200))
        h = max(400, min(int((max_y - min_y) * MAP_VISUALIZATION_SCALE), 1200))
        img = np.full((h, w, 3), 255, dtype=np.uint8)

        def to_px(x, y):
            return (
                int((x - min_x) * MAP_VISUALIZATION_SCALE),
                h - int((y - min_y) * MAP_VISUALIZATION_SCALE),
            )

        # Trajectory path
        for i in range(1, len(traj)):
            cv2.line(img,
                     to_px(traj[i - 1][0], traj[i - 1][1]),
                     to_px(traj[i][0],     traj[i][1]),
                     (0, 0, 255), 2)

        # Current robot position + heading arrow
        rx, ry, ryaw = traj[-1][:3]
        rp = to_px(rx, ry)
        cv2.circle(img, rp, 6, (0, 200, 0), -1)
        arrow_end = to_px(
            rx + 0.15 * math.cos(ryaw),
            ry + 0.15 * math.sin(ryaw),
        )
        cv2.arrowedLine(img, rp, arrow_end, (0, 150, 0), 2, tipLength=0.4)

        # Detected signs
        sign_colors = {
            'STOP':     (0, 0, 220),
            'LEFT':     (220, 0, 0),
            'RIGHT':    (0, 200, 200),
            'FORWARD':  (0, 180, 0),
            'NO ENTRY': (140, 0, 140),
            'DEAD END': (80, 80, 80),
        }
        for label, pts in signs_snap.items():
            col = sign_colors.get(label, (128, 128, 128))
            for sx, sy, _ in pts:
                p = to_px(sx, sy)
                cv2.circle(img, p, 9, col, -1)
                cv2.putText(img, label[:3], (p[0] + 11, p[1]),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, col, 1)

        # Legend
        cv2.putText(img, "VSLAM Map", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        cv2.circle(img, (15, 40), 4, (0, 0, 255), -1)
        cv2.putText(img, "Trajectory", (25, 44),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

        return img


# ============================================================
#  VISION PROCESSOR
# ============================================================

class VisionProcessor:
    """
    Per-frame green-line detection and YOLO sign detection.

    Line detection
    ──────────────
    HSV mask → ROI clip → contour filter (area + horizontal rejection)
    Returns centroid lateral offset (px) as the PID error signal.
    Heading angle is also returned but used only by the SEARCHING state —
    it is never fed into the PID controller.
    Horizontal contours are drawn in olive and discarded.

    Sign detection
    ──────────────
    Strict box filtering from LimoVisionNode:
      • Boxes occupying > MAX_BOX_AREA_RATIO of the frame are dropped.
      • Parent boxes that fully contain a smaller detection are dropped.
    Distance is calculated via estimate_distance_geometry().
    """

    def __init__(self, model_path: str, confidence: float):
        self.confidence = confidence
        try:
            self.model = YOLO(model_path)
            print("[VisionProcessor] YOLO model loaded.")
        except Exception as exc:
            print(f"[VisionProcessor] YOLO load error: {exc}")
            self.model = None

    # ------------------------------------------------------------------
    @staticmethod
    def _contour_angle_and_centroid(contour):
        """
        Returns (heading_err_rad, lateral_err_px, is_horizontal).
        heading_err = angle from vertical (used for horizontal filtering +
                      SEARCHING path detection, NOT for PID).
        """
        [vx, vy, _cx, _cy] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)
        if vy > 0:
            vx, vy = -vx, -vy

        heading_err = math.atan2(vx, -vy)
        is_horizontal = abs(math.degrees(heading_err)) > HORIZONTAL_THRESHOLD_DEG

        M = cv2.moments(contour)
        lat_err = (M['m10'] / M['m00'] - IMAGE_CENTER_X) if M['m00'] > 0 else 0.0
        return heading_err, lat_err, is_horizontal

    # ------------------------------------------------------------------
    def process_frame(self, frame: np.ndarray):
        """
        Returns
        -------
        display     : annotated BGR image
        green_mask  : binary mask after ROI clip
        line_ok     : True when a usable vertical line is found
        heading_err : rad from vertical (secondary info — not for PID)
        lat_err_px  : raw centroid X offset in pixels (PID input)
        detections  : list of {label, distance, conf}
        """
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        display = frame.copy()

        # ── Green mask ──────────────────────────────────────────────
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)
        green_mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
        green_mask = cv2.dilate(green_mask, kernel, iterations=4)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        # ── Apply ROI ───────────────────────────────────────────────
        roi_mask = np.zeros(green_mask.shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, ROI_VERTICES, 255)
        green_mask = cv2.bitwise_and(green_mask, roi_mask)

        # ROI outline + centre line on display
        cv2.polylines(display, [ROI_VERTICES], True, (0, 0, 200), 1)
        overlay = cv2.bitwise_and(frame, frame, mask=green_mask)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, display, 1 - OVERLAY_ALPHA, 0, display)
        cv2.line(display, (IMAGE_CENTER_X, 0), (IMAGE_CENTER_X, IMAGE_HEIGHT),
                 (255, 0, 0), 1)

        # ── Contour analysis ────────────────────────────────────────
        line_ok    = False
        heading_err = 0.0
        lat_err_px  = 0.0

        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) < MIN_CONTOUR_AREA:
                continue
            h_err, l_err, is_horiz = self._contour_angle_and_centroid(cnt)
            if is_horiz:
                # Acknowledged but excluded — draw in olive for operator visibility
                cv2.drawContours(display, [cnt], -1, (30, 120, 30), 1)
                cv2.putText(display, "HORIZ (ignored)",
                            (int(cnt[0][0][0]), int(cnt[0][0][1]) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (30, 120, 30), 1)
                continue
            valid_contours.append(cnt)

        if valid_contours:
            best = max(valid_contours, key=cv2.contourArea)
            line_ok = True
            heading_err, lat_err_px, _ = self._contour_angle_and_centroid(best)

            # Draw fit-line
            [vx, vy, x, y] = cv2.fitLine(best, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            if vy > 0:
                vx, vy = -vx, -vy
            cv2.line(display,
                     (int(x - vx * 1000), int(y - vy * 1000)),
                     (int(x + vx * 1000), int(y + vy * 1000)),
                     (0, 0, 255), 2)

            # Draw centroid + lateral offset indicator
            M = cv2.moments(best)
            if M['m00'] > 0:
                cx_px = int(M['m10'] / M['m00'])
                cy_px = int(M['m01'] / M['m00'])
                cv2.circle(display, (cx_px, cy_px), 6, (0, 255, 255), -1)
                cv2.line(display, (IMAGE_CENTER_X, cy_px), (cx_px, cy_px),
                         (0, 255, 255), 2)

        # ── YOLO detections with strict box filtering ────────────────
        detections = []
        if self.model:
            results = self.model(frame, conf=self.confidence, verbose=False)
            result  = results[0]

            keep_indices = []
            if len(result.boxes) > 0:
                boxes    = result.boxes.xyxy
                img_h, img_w = result.orig_shape
                max_area = MAX_BOX_AREA_RATIO * img_w * img_h

                for i in range(len(boxes)):
                    bx = boxes[i]
                    # Reject oversized boxes
                    if (bx[2] - bx[0]) * (bx[3] - bx[1]) > max_area:
                        continue
                    # Reject boxes that fully contain a smaller sibling
                    encapsulates = any(
                        j != i
                        and bx[0] <= boxes[j][0] and bx[1] <= boxes[j][1]
                        and bx[2] >= boxes[j][2] and bx[3] >= boxes[j][3]
                        for j in range(len(boxes))
                    )
                    if not encapsulates:
                        keep_indices.append(i)

                result.boxes = result.boxes[keep_indices]

            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                if (y2 - y1) < MIN_BOX_HEIGHT_PX:
                    continue

                label    = str(result.names[int(box.cls[0])]).upper()
                conf     = float(box.conf[0])
                dist_cm  = estimate_distance_geometry(float(y2))

                detections.append({'label': label, 'distance': dist_cm, 'conf': conf})

                cv2.rectangle(display, (x1, y1), (x2, y2), (255, 50, 50), 2)
                dist_txt = f"{dist_cm:.1f}cm" if dist_cm != math.inf else "Too Far"
                cv2.putText(display, f"{label} {conf:.2f} | {dist_txt}",
                            (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 50, 50), 2)

        # ── HUD ─────────────────────────────────────────────────────
        hud_col = (0, 255, 0) if line_ok else (0, 0, 255)
        cv2.putText(display,
                    f"L_err: {lat_err_px:+.1f}px  |  H_err: {heading_err:+.2f}rad (info)",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        cv2.putText(display,
                    f"line: {'TRACKING' if line_ok else 'LOST / HORIZ IGNORED'}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        return display, green_mask, line_ok, heading_err, lat_err_px, detections


# ============================================================
#  MAIN ROS2 NODE
# ============================================================

class VisionSlamNode(Node):

    def __init__(self):
        super().__init__('vision_slam_node')

        # ── Subsystems ──────────────────────────────────────────────
        self._vis_lock = threading.Lock()
        self.vision    = VisionProcessor(YOLO_MODEL_PATH, YOLO_CONFIDENCE)
        self.mapper    = VSLAMMapper(MAP_SAVE_PATH)

        # ── OpenCV camera (direct — no ROS image subscription) ──────
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  IMAGE_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, IMAGE_HEIGHT)
        if not self.cap.isOpened():
            self.get_logger().error("Failed to open camera index 0.")
        self._prev_cam_time = time.time()

        # ── Odometry subscriptions (tries all candidate topics) ─────
        self._odom_subs      = []
        self.active_odom_topic = None
        for topic in ODOM_TOPIC_CANDIDATES:
            sub = self.create_subscription(
                Odometry, topic,
                self._make_odom_cb(topic),
                qos_profile=qos_profile_sensor_data)
            self._odom_subs.append(sub)

        # ── ROS publisher ────────────────────────────────────────────
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', PUBLISHER_QUEUE_SIZE)

        # ── Odometry state ───────────────────────────────────────────
        self.odom_ok  = False
        self.init_x   = self.init_y = self.init_yaw = 0.0
        self.robot_x  = self.robot_y = self.robot_yaw = 0.0

        # ── Shared vision state (written by _camera_loop) ────────────
        self._line_ok     = False
        self._heading_err = 0.0   # rad — secondary info for SEARCHING, not PID
        self._lat_err_px  = 0.0   # px  — primary PID input
        self._detections  = []

        # ── Navigation state machine ─────────────────────────────────
        self.nav_state        = "FOLLOW_LINE"
        self.active_priority  = IDLE_PRIORITY
        self.state_entry_time = 0.0
        self.target_yaw       = 0.0   # TURNING_FIXED only
        self.cooldown_until   = 0.0

        # Pending turn
        self.pending_turn_direction = ""  # "LEFT" or "RIGHT"
        self.pending_turn_priority  = 0
        self.target_travel_dist     = 0.0
        self.action_start_x         = 0.0
        self.action_start_y         = 0.0
        self.search_entry_yaw       = 0.0

        # ── PID state (single lateral channel + EMA) ─────────────────
        self.integral     = 0.0
        self.prev_error   = 0.0
        self.smooth_error = 0.0
        self.last_pid_time = 0.0

        # ── VSLAM ───────────────────────────────────────────────────
        self.last_map_update = 0.0

        # ── Dispatch tables ──────────────────────────────────────────
        self._action_dispatch = {
            "STOP":     self._on_stop,
            "FORWARD":  self._on_forward,
            "RIGHT":    self._on_right,
            "LEFT":     self._on_left,
            "NO ENTRY": self._on_dead_end,
            "DEAD END": self._on_dead_end,
        }
        self._state_dispatch = {
            "FOLLOW_LINE":     self._tick_follow_line,
            "APPROACHING":     self._tick_approaching,
            "STOPPING":        self._tick_stopping,
            "TURNING_DYNAMIC": self._tick_turning_dynamic,
            "SEARCHING":       self._tick_searching,
        }

        # ── Timers ───────────────────────────────────────────────────
        self.camera_timer  = self.create_timer(CAMERA_LOOP_PERIOD,  self._camera_loop)
        self.control_timer = self.create_timer(CONTROL_LOOP_PERIOD, self._control_loop)

        self.get_logger().info(
            "VisionSlamNode ready — camera: OpenCV(0), VSLAM: active.")

    # ================================================================
    #  HELPERS
    # ================================================================

    def _reset_pid(self) -> None:
        """Clear all PID memory on state transitions into FOLLOW_LINE."""
        self.integral      = 0.0
        self.prev_error    = 0.0
        self.smooth_error  = 0.0
        self.last_pid_time = 0.0

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _yaw_diff(self, target: float, current: float) -> float:
        return math.atan2(math.sin(target - current), math.cos(target - current))

    def _enter_state(self, state: str, priority: int = IDLE_PRIORITY) -> None:
        if state == "FOLLOW_LINE":
            self._reset_pid()
        self.nav_state        = state
        self.active_priority  = priority
        self.state_entry_time = self._now()
        self.get_logger().info(f"  → State: {state}  (priority {priority})")

    # ================================================================
    #  ODOMETRY CALLBACK
    # ================================================================

    def _make_odom_cb(self, topic: str):
        def cb(msg):
            if self.active_odom_topic is None:
                self.active_odom_topic = topic
                self.get_logger().info(f"Odometry latched → {topic}")
            if topic != self.active_odom_topic:
                return
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            q  = msg.pose.pose.orientation
            raw_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            if not self.odom_ok:
                self.init_x, self.init_y, self.init_yaw = px, py, raw_yaw
                self.odom_ok = True
            dx, dy = px - self.init_x, py - self.init_y
            c, s = math.cos(-self.init_yaw), math.sin(-self.init_yaw)
            self.robot_x   = dx * c - dy * s
            self.robot_y   = dx * s + dy * c
            self.robot_yaw = math.atan2(
                math.sin(raw_yaw - self.init_yaw),
                math.cos(raw_yaw - self.init_yaw),
            )
        return cb

    # ================================================================
    #  CAMERA LOOP  (30 Hz)
    # ================================================================

    def _camera_loop(self) -> None:
        """Read one frame, run vision processing, update shared state."""
        ret, frame = self.cap.read()
        if not ret:
            return

        now = time.time()
        fps = 1.0 / max(now - self._prev_cam_time, 1e-6)
        self._prev_cam_time = now

        display, mask, line_ok, h_err, l_err, dets = \
            self.vision.process_frame(frame)

        # Extra HUD overlays
        cv2.putText(display, f"FPS: {int(fps)}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
        cv2.putText(display, f"State: {self.nav_state}",
                    (10, 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 0), 1)

        with self._vis_lock:
            self._line_ok     = line_ok
            self._heading_err = h_err
            self._lat_err_px  = l_err
            self._detections  = dets

        cv2.imshow("LIMO — Vision + YOLO", display)
        cv2.imshow("Green Mask (ROI)", mask)

        map_img = self.mapper.visualize()
        if map_img is not None:
            cv2.imshow("VSLAM Map", map_img)

        cv2.waitKey(1)

    # ================================================================
    #  ACTION HANDLERS
    # ================================================================

    def _on_stop(self, _t, _dist_cm):
        self._enter_state("STOPPING", ACTION_PRIORITY["STOP"])

    def _on_forward(self, _t, _dist_cm):
        if self.nav_state != "FOLLOW_LINE":
            self._enter_state("FOLLOW_LINE", ACTION_PRIORITY["FORWARD"])

    def _on_right(self, _t, dist_cm):
        self._schedule_turn("RIGHT", ACTION_PRIORITY["RIGHT"], dist_cm)

    def _on_left(self, _t, dist_cm):
        self._schedule_turn("LEFT", ACTION_PRIORITY["LEFT"], dist_cm)

    def _on_dead_end(self, _t, _dist_cm):
        self.search_entry_yaw = self.robot_yaw
        self._enter_state("SEARCHING", ACTION_PRIORITY["DEAD END"])

    # ================================================================
    #  TURN SCHEDULING
    # ================================================================

    def _schedule_turn(self, direction: str, priority: int, dist_cm: float) -> None:
        """Continue line-following until dist_cm travelled, then turn."""
        self.pending_turn_direction = direction
        self.pending_turn_priority  = priority
        self.target_travel_dist     = (dist_cm / 100.0) * TURN_DISTANCE_MULTIPLIER
        self.action_start_x         = self.robot_x
        self.action_start_y         = self.robot_y
        self._enter_state("APPROACHING", priority)

    def _command_dynamic_turn(self, direction: str, priority: int) -> None:
        """Begin a dynamic turn; exits when the line is re-acquired."""
        self.pending_turn_direction = direction
        self._enter_state("TURNING_DYNAMIC", priority)

    # ================================================================
    #  STATE TICK HANDLERS
    # ================================================================

    def _tick_follow_line(self, twist: Twist, t: float,
                          line_ok: bool, heading_err: float, lat_err_px: float):
        """
        EMA-smoothed centroid lateral PID from LimoPathFollower.
        heading_err is available in the signature but intentionally unused.
        """
        if not line_ok:
            self._reset_pid()
            return                  # safety stop — publish zero twist

        # Time delta
        if self.last_pid_time == 0.0:
            dt = CONTROL_LOOP_PERIOD
        else:
            dt = max(t - self.last_pid_time, 1e-4)
        self.last_pid_time = t

        # EMA smoothing
        self.smooth_error = (EMA_ALPHA * lat_err_px
                             + (1.0 - EMA_ALPHA) * self.smooth_error)
        error = self.smooth_error

        # PID
        self.integral = float(np.clip(
            self.integral + error * dt, -MAX_WINDUP, MAX_WINDUP))
        derivative  = (error - self.prev_error) / dt
        self.prev_error = error

        pid_out  = KP * error + KI * self.integral + KD * derivative
        yaw_cmd  = float(np.clip(-pid_out * SENSITIVITY, -MAX_ANGULAR, MAX_ANGULAR))

        # Speed scaling on severe offset
        speed = FORWARD_SPEED * (0.5 if abs(lat_err_px) > SEVERE_LATERAL_OFFSET
                                  else 1.0)

        twist.linear.x  = speed
        twist.angular.z = yaw_cmd

    # ------------------------------------------------------------------
    def _tick_approaching(self, twist: Twist, t: float,
                          line_ok: bool, heading_err: float, lat_err_px: float):
        traveled = math.hypot(
            self.robot_x - self.action_start_x,
            self.robot_y - self.action_start_y,
        )
        if traveled >= self.target_travel_dist:
            self.get_logger().info(
                f"Approach done ({traveled:.2f} m) — starting dynamic turn.")
            self._command_dynamic_turn(
                self.pending_turn_direction, self.pending_turn_priority)
            return
        self._tick_follow_line(twist, t, line_ok, heading_err, lat_err_px)

    # ------------------------------------------------------------------
    def _tick_stopping(self, twist: Twist, t: float,
                       line_ok: bool, heading_err: float, lat_err_px: float):
        if t - self.state_entry_time >= STOP_DURATION_SEC:
            self.get_logger().info("STOP complete — resuming line follow.")
            self._enter_state("FOLLOW_LINE")
        # twist stays at zero while waiting

    # ------------------------------------------------------------------
    def _tick_turning_dynamic(self, twist: Twist, t: float,
                              line_ok: bool, heading_err: float, lat_err_px: float):
        """
        Turns in the commanded direction until:
          • The line is visible again (line_ok), AND
          • The lateral offset is within DYNAMIC_TURN_LATERAL_THRESH, AND
          • At least DYNAMIC_TURN_MIN_DURATION seconds have elapsed
            (prevents premature exit before the robot has actually turned).
        """
        elapsed = t - self.state_entry_time
        if (line_ok
                and elapsed >= DYNAMIC_TURN_MIN_DURATION
                and abs(lat_err_px) < DYNAMIC_TURN_LATERAL_THRESH):
            self.get_logger().info(
                f"Dynamic turn aligned — l_err={lat_err_px:.1f}px, "
                f"elapsed={elapsed:.2f}s")
            self._enter_state("FOLLOW_LINE")
            return

        twist.angular.z = (TURN_SPEED
                           if self.pending_turn_direction == "LEFT"
                           else -TURN_SPEED)

    # ------------------------------------------------------------------
    def _tick_searching(self, twist: Twist, t: float,
                        line_ok: bool, heading_err: float, lat_err_px: float):
        """
        Sweeps left then right looking for an alternate path.
        heading_err (from fitLine — NOT the PID error) detects branch lines.
        Times out into a 180° odometry-guided retreat.
        """
        elapsed = t - self.state_entry_time

        # Alternate path found
        if line_ok and abs(heading_err) > SEARCH_PATH_ANGLE:
            direction = "LEFT" if heading_err > 0 else "RIGHT"
            self.get_logger().info(
                f"Alternate path found ({direction}, h_err={heading_err:.2f}rad).")
            self._command_dynamic_turn(direction, ACTION_PRIORITY["DEAD END"])
            return

        # Timeout — 180° fixed retreat
        if elapsed >= SEARCH_TIMEOUT_SEC:
            self.get_logger().info("Search timeout — 180° retreat.")
            self.target_yaw = math.atan2(
                math.sin(self.search_entry_yaw + math.pi),
                math.cos(self.search_entry_yaw + math.pi),
            )
            # TURNING_FIXED handled directly in _control_loop (no line args needed)
            self.nav_state        = "TURNING_FIXED"
            self.active_priority  = ACTION_PRIORITY["DEAD END"]
            self.state_entry_time = t
            return

        twist.angular.z = (SEARCH_TURN_SPEED
                           if elapsed < SEARCH_TIMEOUT_SEC / 2.0
                           else -SEARCH_TURN_SPEED)

    # ------------------------------------------------------------------
    def _tick_turning_fixed(self, twist: Twist) -> None:
        """Odometry-guided fixed-yaw turn — used only for 180° retreat."""
        diff = self._yaw_diff(self.target_yaw, self.robot_yaw)
        if abs(diff) < TURN_DONE_THRESH:
            self.get_logger().info("180° retreat done — resuming line follow.")
            self._enter_state("FOLLOW_LINE")
        else:
            twist.angular.z = TURN_SPEED if diff > 0 else -TURN_SPEED

    # ================================================================
    #  DETECTION PROCESSING
    # ================================================================

    def _process_detections(self, t: float, detections: list) -> None:
        valid_bypasses = ["LEFT", "RIGHT", "FORWARD"]
        in_cooldown    = t <= self.cooldown_until
        is_searching   = self.nav_state == "SEARCHING"

        if in_cooldown:
            if not (is_searching and
                    any(d['label'] in valid_bypasses for d in detections)):
                return

        best_label    = None
        best_priority = -1
        best_dist     = math.inf

        for det in detections:
            label    = det['label']
            is_escape = is_searching and label in valid_bypasses

            # Distance gate (escape signs bypass it)
            if not is_escape and det['distance'] > ACTION_DISTANCE_CM:
                continue

            priority = ACTION_PRIORITY.get(label, 0)

            if in_cooldown and not is_escape:
                continue

            if priority < self.active_priority:
                continue

            if (priority > best_priority or
                    (priority == best_priority and det['distance'] < best_dist)):
                best_label    = label
                best_priority = priority
                best_dist     = det['distance']

        if best_label and best_label in self._action_dispatch:
            self.get_logger().info(
                f"Sign accepted → {best_label}  "
                f"dist={best_dist:.1f}cm  pri={best_priority}")
            self.cooldown_until = t + SIGN_COOLDOWN_SEC
            self._action_dispatch[best_label](t, best_dist)

    # ================================================================
    #  CONTROL LOOP  (10 Hz)
    # ================================================================

    def _control_loop(self) -> None:
        twist = Twist()
        t     = self._now()

        if not self.odom_ok:
            self.cmd_pub.publish(twist)
            return

        with self._vis_lock:
            line_ok = self._line_ok
            h_err   = self._heading_err
            l_err   = self._lat_err_px
            dets    = list(self._detections)

        # VSLAM map update
        if t - self.last_map_update >= MAP_UPDATE_INTERVAL:
            self.mapper.update(
                self.robot_x, self.robot_y, self.robot_yaw, dets, t)
            self.last_map_update = t

        self._process_detections(t, dets)

        # TURNING_FIXED has no line/heading args — handled separately
        if self.nav_state == "TURNING_FIXED":
            self._tick_turning_fixed(twist)
        else:
            handler = self._state_dispatch.get(self.nav_state)
            if handler:
                handler(twist, t, line_ok, h_err, l_err)

        self.cmd_pub.publish(twist)

    # ================================================================
    #  SHUTDOWN
    # ================================================================

    def shutdown(self) -> None:
        self.get_logger().info("Saving VSLAM map…")
        self.mapper.save_map()
        self.get_logger().info(f"Map saved → {MAP_SAVE_PATH}")
        self.cap.release()


# ============================================================
#  ENTRY POINT
# ============================================================

def main(args=None):
    rclpy.init(args=args)
    node = VisionSlamNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info("Keyboard interrupt — shutting down.")
    finally:
        node.shutdown()
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        cv2.destroyAllWindows()
        rclpy.shutdown()


if __name__ == '__main__':
    main()