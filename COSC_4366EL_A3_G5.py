#!/usr/bin/env python3
"""
vision_slam_node.py

ROS2 autonomous navigation node for the AgileX LIMO robot.
Combines dual-PID green-line following (Heading + Lateral offset)
with YOLO sign detection and a priority-driven, interrupt-capable state machine.

Distance Estimation : Geometric model based on camera height and tilt angle
Angle Estimation    : Dynamic full-frame extraction. Evaluates both heading angle
                      and lateral pixel offset independently for robust recentering.

FIXES:
- Horizontal lines are detected but excluded from control calculations
- New geometric distance estimation (20-25cm action threshold)
- Dynamic turn alignment to vertical line heading instead of fixed 90°
- VSLAM mapping system with odometry-based environment mapping
"""

import math
import threading
import json
from collections import defaultdict

import cv2
import numpy as np
import rclpy
from cv_bridge import CvBridge
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from rclpy.qos import (DurabilityPolicy, HistoryPolicy, QoSProfile,
                       ReliabilityPolicy, qos_profile_sensor_data)
from sensor_msgs.msg import Image
from ultralytics import YOLO

# ============================================================
#  CONFIGURATION CONSTANTS
# ============================================================

# -- Odometry topic candidates -------
ODOM_TOPIC_CANDIDATES = [
    '/odom',
    '/limo_base/odom',
    '/ackermann_odom',
    '/wheel/odom',
]

# -- Camera --------------------------------------------------
CAMERA_TOPIC = '/camera/color/image_raw'
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CENTER_X = IMAGE_WIDTH // 2
OVERLAY_ALPHA = 0.5

# -- Geometric Distance Model (NEW) --------------------------
CAMERA_HEIGHT_CM = 18.5  # Height of lens from floor
CAMERA_TILT_DEG = 35.0  # Downward tilt from horizontal
V_FOV = 48.0  # Vertical Field of View
DISTANCE_CORRECTION_FACTOR = 0.9  # Fine-tuning multiplier
DISTANCE_OFFSET = 0.0  # Constant offset in cm
ACTION_DISTANCE_CM = 25.0  # Signs beyond this are ignored (was 60, now 20-25cm)
MIN_BOX_HEIGHT_PX = 10

# -- HSV Green Tape Segmentation -----------------------------
HSV_LOWER_GREEN = np.array([35, 40, 40])
HSV_UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 1000
HORIZONTAL_THRESHOLD_DEG = 65  # Lines more horizontal than this are flagged

# -- YOLO ----------------------------------------------------
YOLO_MODEL_PATH = "best_yolo.pt"
YOLO_CONFIDENCE = 0.5

# -- ROS -----------------------------------------------------
QOS_DEPTH = 10
PUBLISHER_QUEUE_SIZE = 10
CONTROL_LOOP_HZ = 10
CONTROL_LOOP_PERIOD = 1.0 / CONTROL_LOOP_HZ

# -- Movement & Dual-PID Control -----------------------------
FORWARD_SPEED = 0.15  # m/s base speed
TURN_SPEED = 0.30  # rad/s for dynamic alignment turns

# Heading PID Gains (Radians)
KP_HEADING = 1.5
KI_HEADING = 0.05
KD_HEADING = 0.20

# Lateral PID Gains (Pixels)
KP_LATERAL = 0.004
KI_LATERAL = 0.0001
KD_LATERAL = 0.001

MAX_INTEGRAL_WINDUP = 1.0
MAX_ANGULAR_CORRECTION = 0.40  # rad/s clamp on final output
MAX_HEADING_TO_DRIVE = 0.25  # rad: drive forward only below this error
SEVERE_LATERAL_OFFSET = 60  # px: if offset > this, slow down to recover

# -- State Machine Timing & Distances ------------------------
STOP_DURATION_SEC = 3.0
SEARCH_TURN_SPEED = 0.20
SEARCH_TIMEOUT_SEC = 4.0
SEARCH_PATH_ANGLE = 0.40
TURN_DONE_THRESH = 0.08  # Threshold for fixed turns
DYNAMIC_TURN_ALIGNMENT_THRESH = 0.15  # rad: heading error to consider aligned
DYNAMIC_TURN_LATERAL_THRESH = 30  # px: lateral error to consider centered
SIGN_COOLDOWN_SEC = 5.0

# Multiplier to delay the exact point of turning
TURN_DISTANCE_MULTIPLIER = 1.2

# -- Action Priority -----------------------------------------
ACTION_PRIORITY = {
    "NO ENTRY": 4,
    "DEAD END": 4,
    "STOP": 3,
    "RIGHT": 2,
    "LEFT": 2,
    "FORWARD": 1,
}
IDLE_PRIORITY = 0

# -- VSLAM Mapping -------------------------------------------
MAP_SAVE_PATH = "/home/claude/vslam_map.json"
MAP_VISUALIZATION_SCALE = 100  # pixels per meter for visualization
MAP_UPDATE_INTERVAL = 0.5  # seconds between map updates


# ============================================================
#  GEOMETRIC DISTANCE ESTIMATION (NEW)
# ============================================================

def estimate_distance_geometry(y_pixel):
    """
    Calculates distance using geometric projection from camera parameters.
    Inverting the relative_y ensures that higher pixel values (bottom of screen)
    result in shorter distances.
    """
    # Inverted: Bottom of screen (y=480) becomes -1, center 0, top 1
    relative_y = ((IMAGE_HEIGHT / 2) - y_pixel) / (IMAGE_HEIGHT / 2)

    # Angle of the specific pixel relative to the camera optical axis
    pixel_angle_offset = relative_y * (V_FOV / 2)

    # Calculate angle relative to the floor
    total_angle_deg = (90 - CAMERA_TILT_DEG) + pixel_angle_offset

    # Prevent tangent of 90 or negative angles
    if total_angle_deg >= 89:
        return math.inf

    raw_distance = CAMERA_HEIGHT_CM * math.tan(math.radians(total_angle_deg))

    # Apply fine-tuning
    corrected_distance = (raw_distance * DISTANCE_CORRECTION_FACTOR) + DISTANCE_OFFSET

    return max(0, corrected_distance)


# ============================================================
#  VSLAM MAPPER
# ============================================================

class VSLAMMapper:
    """
    Maintains a map of the environment based on odometry and detections.
    Stores sign locations, path trajectory, and generates visualizations.
    """

    def __init__(self, save_path: str):
        self.save_path = save_path
        self.trajectory = []  # List of (x, y, yaw, timestamp) tuples
        self.signs = defaultdict(list)  # {sign_label: [(x, y, timestamp), ...]}
        self.last_update_time = 0.0
        self.lock = threading.Lock()

    def update(self, robot_x: float, robot_y: float, robot_yaw: float,
               detections: list, timestamp: float):
        """
        Update the map with current robot pose and detected signs.
        """
        with self.lock:
            # Update trajectory
            self.trajectory.append((robot_x, robot_y, robot_yaw, timestamp))

            # Add detected signs with global coordinates
            for det in detections:
                label = det['label']
                dist_m = det['distance'] / 100.0  # Convert cm to meters

                # Calculate global position of sign
                sign_x = robot_x + dist_m * math.cos(robot_yaw)
                sign_y = robot_y + dist_m * math.sin(robot_yaw)

                self.signs[label].append((sign_x, sign_y, timestamp))

    def save_map(self):
        """Save map data to JSON file"""
        with self.lock:
            data = {
                'trajectory': self.trajectory,
                'signs': {k: v for k, v in self.signs.items()},
            }
            try:
                with open(self.save_path, 'w') as f:
                    json.dump(data, f, indent=2)
            except Exception as e:
                print(f"[VSLAMMapper] Save error: {e}")

    def visualize(self):
        """
        Generate a top-down visualization of the map.
        Returns an OpenCV image.
        """
        with self.lock:
            if not self.trajectory:
                return None

            # Calculate map bounds
            all_x = [p[0] for p in self.trajectory]
            all_y = [p[1] for p in self.trajectory]

            for sign_list in self.signs.values():
                all_x.extend([s[0] for s in sign_list])
                all_y.extend([s[1] for s in sign_list])

            min_x, max_x = min(all_x), max(all_x)
            min_y, max_y = min(all_y), max(all_y)

            # Add padding
            padding = 1.0  # meters
            min_x -= padding
            max_x += padding
            min_y -= padding
            max_y += padding

            # Create image
            width = int((max_x - min_x) * MAP_VISUALIZATION_SCALE)
            height = int((max_y - min_y) * MAP_VISUALIZATION_SCALE)
            width = max(400, min(width, 1200))
            height = max(400, min(height, 1200))

            img = np.ones((height, width, 3), dtype=np.uint8) * 255

            def to_px(x, y):
                px_x = int((x - min_x) * MAP_VISUALIZATION_SCALE)
                px_y = height - int((y - min_y) * MAP_VISUALIZATION_SCALE)
                return (px_x, px_y)

            # Draw trajectory
            for i in range(1, len(self.trajectory)):
                p1 = to_px(self.trajectory[i - 1][0], self.trajectory[i - 1][1])
                p2 = to_px(self.trajectory[i][0], self.trajectory[i][1])
                cv2.line(img, p1, p2, (0, 0, 255), 2)

            # Draw robot current position
            if self.trajectory:
                curr = to_px(self.trajectory[-1][0], self.trajectory[-1][1])
                cv2.circle(img, curr, 5, (0, 255, 0), -1)

            # Draw signs
            sign_colors = {
                'STOP': (0, 0, 255),
                'LEFT': (255, 0, 0),
                'RIGHT': (0, 255, 255),
                'FORWARD': (0, 255, 0),
                'NO ENTRY': (128, 0, 128),
                'DEAD END': (64, 64, 64),
            }

            for label, sign_list in self.signs.items():
                color = sign_colors.get(label, (128, 128, 128))
                for (sx, sy, _) in sign_list:
                    pos = to_px(sx, sy)
                    cv2.circle(img, pos, 8, color, -1)
                    cv2.putText(img, label[:3], (pos[0] + 10, pos[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Add legend
            legend_y = 20
            cv2.putText(img, "VSLAM Map", (10, legend_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            legend_y += 25
            cv2.circle(img, (15, legend_y), 3, (0, 0, 255), -1)
            cv2.putText(img, "Trajectory", (25, legend_y + 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            return img


# ============================================================
#  VISION PROCESSOR
# ============================================================

class VisionProcessor:

    def __init__(self, model_path: str, confidence: float):
        self.confidence = confidence
        try:
            self.model = YOLO(model_path)
            print("[VisionProcessor] YOLO model loaded.")
        except Exception as exc:
            print(f"[VisionProcessor] YOLO load error: {exc}")
            self.model = None

    @staticmethod
    def extract_errors(contour: np.ndarray):
        """
        Extract heading error and lateral offset from a contour.
        Returns also if the line is horizontal (should be ignored).
        """
        [vx, vy, cx, cy] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        vx, vy = float(vx), float(vy)
        if vy > 0:
            vx, vy = -vx, -vy

        heading_err = math.atan2(vx, -vy)
        angle_from_vertical = abs(math.degrees(heading_err))

        # Check if line is too horizontal
        is_horizontal = angle_from_vertical > HORIZONTAL_THRESHOLD_DEG

        M = cv2.moments(contour)
        if M['m00'] > 0:
            centroid_x = M['m10'] / M['m00']
            lateral_err_px = centroid_x - IMAGE_CENTER_X
        else:
            lateral_err_px = 0.0

        return heading_err, lateral_err_px, is_horizontal

    def process_frame(self, frame: np.ndarray):
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        display = frame.copy()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        kernel = np.ones((5, 5), np.uint8)

        green_mask = cv2.inRange(hsv, HSV_LOWER_GREEN, HSV_UPPER_GREEN)
        green_mask = cv2.dilate(green_mask, kernel, iterations=4)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        overlay = cv2.bitwise_and(frame, frame, mask=green_mask)
        cv2.addWeighted(overlay, OVERLAY_ALPHA, display, 1 - OVERLAY_ALPHA, 0, display)

        cv2.line(display, (IMAGE_CENTER_X, 0), (IMAGE_CENTER_X, IMAGE_HEIGHT), (255, 0, 0), 1)

        line_ok = False
        heading_err = 0.0
        lat_err_px = 0.0

        contours, _ = cv2.findContours(
            green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        valid_contours = []
        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                h_err, l_err, is_horiz = self.extract_errors(cnt)

                # FIX #1: Acknowledge horizontal lines but don't use them
                if is_horiz:
                    # Draw in different color to show it's detected but ignored
                    cv2.drawContours(display, [cnt], -1, (128, 128, 0), 1)
                    continue

                valid_contours.append(cnt)

        if valid_contours:
            best = max(valid_contours, key=cv2.contourArea)
            line_ok = True
            heading_err, lat_err_px, _ = self.extract_errors(best)

            [vx, vy, x, y] = cv2.fitLine(best, cv2.DIST_L2, 0, 0.01, 0.01)
            vx, vy = float(vx), float(vy)
            if vy > 0:
                vx, vy = -vx, -vy
            p1 = (int(x - vx * 1000), int(y - vy * 1000))
            p2 = (int(x + vx * 1000), int(y + vy * 1000))
            cv2.line(display, p1, p2, (0, 0, 255), 2)

            M = cv2.moments(best)
            if M['m00'] > 0:
                cx_px = int(M['m10'] / M['m00'])
                cy_px = int(M['m01'] / M['m00'])
                cv2.circle(display, (cx_px, cy_px), 6, (0, 255, 255), -1)
                cv2.line(display,
                         (IMAGE_CENTER_X, cy_px), (cx_px, cy_px),
                         (0, 255, 255), 2)

        detections = []
        if self.model:
            results = self.model(frame, conf=self.confidence, verbose=False)
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    if (y2 - y1) < MIN_BOX_HEIGHT_PX:
                        continue

                    label = str(r.names[int(box.cls[0])]).upper()
                    conf = float(box.conf[0])

                    # FIX #2: Use geometric distance estimation
                    dist_cm = estimate_distance_geometry(float(y2))

                    detections.append({'label': label, 'distance': dist_cm, 'conf': conf})

                    cv2.rectangle(display, (x1, y1), (x2, y2), (255, 50, 50), 2)
                    dist_txt = f"{dist_cm:.1f}cm" if dist_cm != math.inf else "Too Far"
                    cv2.putText(display,
                                f"{label} {conf:.2f} | {dist_txt}",
                                (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 50, 50), 2)

        hud_col = (0, 255, 0) if line_ok else (0, 0, 255)
        cv2.putText(display, f"H_err: {heading_err:+.2f} rad | L_err: {lat_err_px:+.1f} px",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display,
                    f"line: {'YES' if line_ok else 'NO / HORIZONTAL IGNORED'}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        return display, green_mask, line_ok, heading_err, lat_err_px, detections


# ============================================================
#  MAIN ROS2 NODE
# ============================================================

class VisionSlamNode(Node):

    def __init__(self):
        super().__init__('vision_slam_node')

        self._vis_lock = threading.Lock()
        self.bridge = CvBridge()
        self.vision = VisionProcessor(YOLO_MODEL_PATH, YOLO_CONFIDENCE)
        self.mapper = VSLAMMapper(MAP_SAVE_PATH)

        cam_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=QOS_DEPTH,
        )
        self.image_sub = self.create_subscription(
            Image, CAMERA_TOPIC, self._image_cb, qos_profile=cam_qos)

        self._odom_subs = []
        self.active_odom_topic = None
        for topic in ODOM_TOPIC_CANDIDATES:
            sub = self.create_subscription(
                Odometry, topic,
                self._make_odom_cb(topic),
                qos_profile=qos_profile_sensor_data)
            self._odom_subs.append(sub)

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', PUBLISHER_QUEUE_SIZE)

        self.odom_ok = False
        self.init_x = self.init_y = 0.0
        self.init_yaw = 0.0
        self.robot_x = self.robot_y = 0.0
        self.robot_yaw = 0.0

        self._line_ok = False
        self._heading_err = 0.0
        self._lat_err_px = 0.0
        self._detections = []

        self.nav_state = "FOLLOW_LINE"
        self.active_priority = IDLE_PRIORITY
        self.state_entry_time = 0.0
        self.target_yaw = 0.0
        self.cooldown_until = 0.0

        # Delayed turn state tracking
        self.pending_turn_direction = ""  # "LEFT" or "RIGHT"
        self.pending_turn_priority = 0
        self.target_travel_dist = 0.0
        self.action_start_x = 0.0
        self.action_start_y = 0.0
        self.search_entry_yaw = 0.0

        # PID Memory Variables
        self.integral_heading = 0.0
        self.integral_lateral = 0.0
        self.prev_heading_err = 0.0
        self.prev_lat_err_px = 0.0
        self.last_pid_time = 0.0

        # Map update tracking
        self.last_map_update = 0.0

        self._action_dispatch = {
            "STOP": self._on_stop,
            "FORWARD": self._on_forward,
            "RIGHT": self._on_right,
            "LEFT": self._on_left,
            "NO ENTRY": self._on_dead_end,
            "DEAD END": self._on_dead_end,
        }
        self._state_dispatch = {
            "FOLLOW_LINE": self._tick_follow_line,
            "APPROACHING": self._tick_approaching,
            "STOPPING": self._tick_stopping,
            "TURNING_DYNAMIC": self._tick_turning_dynamic,
            "SEARCHING": self._tick_searching,
        }

        self.control_timer = self.create_timer(CONTROL_LOOP_PERIOD, self._control_loop)
        self.get_logger().info("VisionSlamNode ready with VSLAM mapping.")

    def _reset_pid(self):
        """Clear memory to prevent stale integrals/derivatives after turning/stopping"""
        self.integral_heading = 0.0
        self.integral_lateral = 0.0
        self.prev_heading_err = 0.0
        self.prev_lat_err_px = 0.0
        self.last_pid_time = 0.0

    def _make_odom_cb(self, topic: str):
        def cb(msg):
            if self.active_odom_topic is None:
                self.active_odom_topic = topic
                self.get_logger().info(f"Odometry latched → {topic}")
            if topic != self.active_odom_topic:
                return
            px = msg.pose.pose.position.x
            py = msg.pose.pose.position.y
            q = msg.pose.pose.orientation
            raw_yaw = math.atan2(
                2.0 * (q.w * q.z + q.x * q.y),
                1.0 - 2.0 * (q.y * q.y + q.z * q.z),
            )
            if not self.odom_ok:
                self.init_x, self.init_y, self.init_yaw = px, py, raw_yaw
                self.odom_ok = True
            dx, dy = px - self.init_x, py - self.init_y
            c, s = math.cos(-self.init_yaw), math.sin(-self.init_yaw)
            self.robot_x = dx * c - dy * s
            self.robot_y = dx * s + dy * c
            self.robot_yaw = math.atan2(
                math.sin(raw_yaw - self.init_yaw),
                math.cos(raw_yaw - self.init_yaw),
            )

        return cb

    def _image_cb(self, ros_img):
        try:
            frame = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f"cv_bridge: {exc}")
            return

        display, mask, line_ok, h_err, l_err, dets = self.vision.process_frame(frame)

        with self._vis_lock:
            self._line_ok = line_ok
            self._heading_err = h_err
            self._lat_err_px = l_err
            self._detections = dets

        cv2.imshow("Green Mask (full frame)", mask)
        cv2.imshow("Vision + YOLO", display)

        # Show VSLAM map
        map_img = self.mapper.visualize()
        if map_img is not None:
            cv2.imshow("VSLAM Map", map_img)

        cv2.waitKey(1)

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds / 1e9

    def _yaw_diff(self, target: float, current: float) -> float:
        return math.atan2(
            math.sin(target - current),
            math.cos(target - current),
        )

    def _enter_state(self, state: str, priority: int = IDLE_PRIORITY):
        if state == "FOLLOW_LINE":
            self._reset_pid()

        self.nav_state = state
        self.active_priority = priority
        self.state_entry_time = self._now()
        self.get_logger().info(f"  → State: {state}  (priority {priority})")

    def _schedule_turn(self, direction: str, priority: int, dist_cm: float):
        """FIX #4: Schedule a dynamic turn instead of fixed angle"""
        self.pending_turn_direction = direction
        self.pending_turn_priority = priority
        self.target_travel_dist = (dist_cm / 100.0) * TURN_DISTANCE_MULTIPLIER
        self.action_start_x = self.robot_x
        self.action_start_y = self.robot_y
        self._enter_state("APPROACHING", priority)

    def _command_dynamic_turn(self, direction: str, priority: int):
        """FIX #4: Start a dynamic turn that aligns to the line"""
        self.pending_turn_direction = direction
        self._enter_state("TURNING_DYNAMIC", priority)

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
        self._enter_state("SEARCHING", ACTION_PRIORITY["FORWARD"])

    def _tick_follow_line(self, twist: Twist, t: float,
                          line_ok: bool, heading_err: float, lat_err_px: float):
        if not line_ok:
            self._reset_pid()
            return

        # Time delta calculation for I and D terms
        if self.last_pid_time == 0.0:
            self.last_pid_time = t
            dt = CONTROL_LOOP_PERIOD
        else:
            dt = t - self.last_pid_time
            if dt <= 0.0:
                dt = CONTROL_LOOP_PERIOD
        self.last_pid_time = t

        # Speed braking Failsafe
        speed_multiplier = 1.0
        if abs(lat_err_px) > SEVERE_LATERAL_OFFSET:
            speed_multiplier = 0.5

        # === HEADING PID ===
        self.integral_heading += heading_err * dt
        self.integral_heading = max(-MAX_INTEGRAL_WINDUP, min(MAX_INTEGRAL_WINDUP, self.integral_heading))
        derivative_heading = (heading_err - self.prev_heading_err) / dt

        pid_heading = (KP_HEADING * heading_err) + (KI_HEADING * self.integral_heading) + (
                KD_HEADING * derivative_heading)

        # === LATERAL PID ===
        self.integral_lateral += lat_err_px * dt
        self.integral_lateral = max(-MAX_INTEGRAL_WINDUP, min(MAX_INTEGRAL_WINDUP, self.integral_lateral))
        derivative_lateral = (lat_err_px - self.prev_lat_err_px) / dt

        pid_lateral = (KP_LATERAL * lat_err_px) + (KI_LATERAL * self.integral_lateral) + (
                KD_LATERAL * derivative_lateral)

        # Save states for next loop
        self.prev_heading_err = heading_err
        self.prev_lat_err_px = lat_err_px

        # Combine, applying negative polarity to correct towards zero
        yaw_cmd = -pid_heading - pid_lateral
        yaw_cmd = max(-MAX_ANGULAR_CORRECTION, min(MAX_ANGULAR_CORRECTION, yaw_cmd))

        if abs(heading_err) <= MAX_HEADING_TO_DRIVE:
            twist.linear.x = FORWARD_SPEED * speed_multiplier
            twist.angular.z = yaw_cmd
        else:
            twist.angular.z = yaw_cmd

    def _tick_approaching(self, twist: Twist, t: float,
                          line_ok: bool, heading_err: float, lat_err_px: float):
        traveled = math.hypot(self.robot_x - self.action_start_x, self.robot_y - self.action_start_y)

        if traveled >= self.target_travel_dist:
            self.get_logger().info(f"Approach complete ({traveled:.2f}m) — executing dynamic turn.")
            self._command_dynamic_turn(self.pending_turn_direction, self.pending_turn_priority)
            return

        self._tick_follow_line(twist, t, line_ok, heading_err, lat_err_px)

    def _tick_stopping(self, twist: Twist, t: float,
                       line_ok: bool, heading_err: float, lat_err_px: float):
        if t - self.state_entry_time >= STOP_DURATION_SEC:
            self.get_logger().info("STOP complete — resuming.")
            self._enter_state("FOLLOW_LINE")

    def _tick_turning_dynamic(self, twist: Twist, t: float,
                              line_ok: bool, heading_err: float, lat_err_px: float):
        """
        FIX #4: Dynamic turn that continues until aligned with a vertical line.
        Uses both heading error and lateral offset to determine completion.
        """
        if line_ok:
            # Check if we're aligned
            heading_aligned = abs(heading_err) < DYNAMIC_TURN_ALIGNMENT_THRESH
            lateral_aligned = abs(lat_err_px) < DYNAMIC_TURN_LATERAL_THRESH

            if heading_aligned and lateral_aligned:
                self.get_logger().info(
                    f"Dynamic turn complete — aligned (h_err={heading_err:.3f}, l_err={lat_err_px:.1f})")
                self._enter_state("FOLLOW_LINE")
                return

        # Continue turning in the specified direction
        turn_speed = TURN_SPEED if self.pending_turn_direction == "LEFT" else -TURN_SPEED
        twist.angular.z = turn_speed

    def _tick_searching(self, twist: Twist, t: float,
                        line_ok: bool, heading_err: float, lat_err_px: float):
        elapsed = t - self.state_entry_time

        if line_ok and abs(heading_err) > SEARCH_PATH_ANGLE:
            direction = "LEFT" if heading_err > 0 else "RIGHT"
            self.get_logger().info(
                f"Alternate path found ({direction}, err={heading_err:.2f} rad) — turning.")
            self._command_dynamic_turn(direction, ACTION_PRIORITY["DEAD END"])
            return

        if elapsed >= SEARCH_TIMEOUT_SEC:
            self.get_logger().info("Search timeout — Exact 180° retreat from original heading.")
            escape_yaw = math.atan2(
                math.sin(self.search_entry_yaw + math.pi),
                math.cos(self.search_entry_yaw + math.pi),
            )
            self.target_yaw = escape_yaw
            # Use old fixed turning for 180° retreat
            self.nav_state = "TURNING_FIXED"
            self.active_priority = ACTION_PRIORITY["DEAD END"]
            return

        twist.angular.z = (SEARCH_TURN_SPEED
                           if elapsed < SEARCH_TIMEOUT_SEC / 2.0
                           else -SEARCH_TURN_SPEED)

    def _tick_turning_fixed(self, twist: Twist):
        """Legacy fixed-angle turn for 180° retreat only"""
        diff = self._yaw_diff(self.target_yaw, self.robot_yaw)
        if abs(diff) < TURN_DONE_THRESH:
            self.get_logger().info("Fixed turn complete — resuming line follow.")
            self._enter_state("FOLLOW_LINE")
        else:
            twist.angular.z = TURN_SPEED if diff > 0 else -TURN_SPEED

    def _process_detections(self, t: float, detections: list):
        """FIX #2: Only process signs within ACTION_DISTANCE_CM (20-25cm)"""
        valid_bypasses = ["LEFT", "RIGHT", "FORWARD"]
        if t <= self.cooldown_until:
            if not (self.nav_state == "SEARCHING" and any(d['label'] in valid_bypasses for d in detections)):
                return

        best_label = None
        best_priority = -1
        best_dist = math.inf

        for det in detections:
            label = det['label']
            is_escape_sign = (self.nav_state == "SEARCHING" and label in valid_bypasses)

            # FIX #2: Ignore signs beyond ACTION_DISTANCE_CM unless escaping
            if not is_escape_sign and det['distance'] > ACTION_DISTANCE_CM:
                continue

            priority = ACTION_PRIORITY.get(label, 0)

            if t <= self.cooldown_until and not is_escape_sign:
                continue

            if priority < self.active_priority:
                continue

            if (priority > best_priority or
                    (priority == best_priority and det['distance'] < best_dist)):
                best_label = label
                best_priority = priority
                best_dist = det['distance']

        if best_label and best_label in self._action_dispatch:
            self.get_logger().info(
                f"Sign accepted → {best_label}  "
                f"dist={best_dist:.1f}cm  priority={best_priority}")
            self.cooldown_until = t + SIGN_COOLDOWN_SEC
            self._action_dispatch[best_label](t, best_dist)

    def _update_map(self, t: float, detections: list):
        """Update VSLAM map periodically"""
        if t - self.last_map_update >= MAP_UPDATE_INTERVAL:
            self.mapper.update(self.robot_x, self.robot_y, self.robot_yaw,
                               detections, t)
            self.last_map_update = t

    def _control_loop(self):
        twist = Twist()
        t = self._now()

        if not self.odom_ok:
            self.cmd_pub.publish(twist)
            return

        with self._vis_lock:
            line_ok = self._line_ok
            h_err = self._heading_err
            l_err = self._lat_err_px
            dets = list(self._detections)

        # Update VSLAM map
        self._update_map(t, dets)

        self._process_detections(t, dets)

        # Handle fixed turning (only for 180° retreat)
        if self.nav_state == "TURNING_FIXED":
            self._tick_turning_fixed(twist)
        else:
            handler = self._state_dispatch.get(self.nav_state)
            if handler:
                handler(twist, t, line_ok, h_err, l_err)

        self.cmd_pub.publish(twist)

    def shutdown(self):
        """Save map on shutdown"""
        self.get_logger().info("Saving VSLAM map...")
        self.mapper.save_map()
        self.get_logger().info(f"Map saved to {MAP_SAVE_PATH}")


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