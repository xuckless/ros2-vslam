import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from rclpy.qos import qos_profile_sensor_data          # pre-built permissive sensor profile
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge
import cv2
import numpy as np
import math


# ---------------------------------------------------------------------------
# Candidate odometry topic names for AgileX LIMO.
# The node subscribes to ALL of them simultaneously; whichever one actually
# publishes will win.  Only the first received message activates odometry.
# ---------------------------------------------------------------------------
ODOM_TOPIC_CANDIDATES = [
    '/odom',
    '/limo_base/odom',
    '/ackermann_odom',
    '/wheel/odom'
]


class LimoNavigationNode(Node):
    """
    ROS2 Node for LIMO robot navigation using computer vision and odometry.
    Implements a state-machine to follow green paths while avoiding orange/white obstacles.
    """

    def __init__(self):
        super().__init__('limo_navigation_node')
        self.bridge = CvBridge()

        # ── Camera QoS (BEST_EFFORT / VOLATILE — required for real-time video) ──
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        # ── Subscribers ──────────────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image, '/camera/color/image_raw',
            self.image_callback, qos_profile=camera_qos)

        # Subscribe to every candidate odom topic with the pre-built sensor
        # profile (BEST_EFFORT / VOLATILE / KEEP_LAST-5).  This is compatible
        # with both RELIABLE and BEST_EFFORT publishers, eliminating silent
        # QoS-mismatch drops that kept the robot frozen at (0, 0).
        self._odom_subs = []
        for topic in ODOM_TOPIC_CANDIDATES:
            sub = self.create_subscription(
                Odometry, topic,
                self._make_odom_callback(topic),
                qos_profile=qos_profile_sensor_data)
            self._odom_subs.append(sub)
            self.get_logger().info(f'  Listening for odometry on: {topic}')

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Navigation / State ────────────────────────────────────────────────────
        self.current_twist = Twist()
        self.is_path_green = False
        self.orange_detected = False
        self.white_detected = False

        # ── Localization ──────────────────────────────────────────────────────────
        self.robot_x = 0.0
        self.robot_y = 0.0
        self.robot_yaw = 0.0
        self.trajectory = []
        self.cone_positions = []

        # Only positions confirmed as green/safe are stored here.
        # Unsafe positions are intentionally never appended → never drawn.
        self.green_path_positions = []

        # Odom health tracking
        self.odom_received = False
        self.active_odom_topic = None

        # ── Map settings ──────────────────────────────────────────────────────────
        self.map_size = 600
        self.map_resolution = 100   # pixels per metre
        self.map_offset_x = self.map_size // 2
        self.map_offset_y = self.map_size // 2

        # ── State-machine ─────────────────────────────────────────────────────────
        self.nav_state = "EVALUATE"
        self.state_start_time = self.get_clock().now()
        self.turn_speed = 0.5                                   # rad/s
        self.turn_duration = (60.0 * np.pi / 180.0) / self.turn_speed
        self.eval_duration = 1.2

        # ── Timers ────────────────────────────────────────────────────────────────
        # Main control loop at 10 Hz
        self.timer = self.create_timer(0.1, self.update_state)

        # Odom watchdog — fires once after 3 s; warns loudly if still no odom
        self.odom_watchdog = self.create_timer(3.0, self._odom_watchdog_cb)

        self.get_logger().info('LimoNavigationNode started.')

    # ── Odom callback factory ─────────────────────────────────────────────────

    def _make_odom_callback(self, topic_name: str):
        """Returns a closure that tags each message with its source topic."""
        def cb(msg):
            self._odom_handler(msg, topic_name)
        return cb

    def _odom_handler(self, msg, topic_name: str):
        """Unified odometry handler for all candidate topics."""
        # Latch onto the first topic that delivers a message; ignore the rest.
        if self.active_odom_topic is None:
            self.active_odom_topic = topic_name
            self.get_logger().info(
                f'✓ Odometry active on: {topic_name}  (other candidates ignored)')

        if topic_name != self.active_odom_topic:
            return   # discard messages from the unused topics

        self.odom_received = True
        self.robot_x = msg.pose.pose.position.x
        self.robot_y = msg.pose.pose.position.y

        # Quaternion → yaw
        q = msg.pose.pose.orientation
        siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
        cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        self.robot_yaw = math.atan2(siny_cosp, cosy_cosp)

        # Full trajectory (internal reference, not drawn on map)
        if not self.trajectory or math.hypot(
                self.robot_x - self.trajectory[-1][0],
                self.robot_y - self.trajectory[-1][1]) > 0.05:
            self.trajectory.append((self.robot_x, self.robot_y))

        # Green-path waypoints — only when the camera confirms safe ground ahead
        if self.is_path_green:
            if not self.green_path_positions or math.hypot(
                    self.robot_x - self.green_path_positions[-1][0],
                    self.robot_y - self.green_path_positions[-1][1]) > 0.04:
                self.green_path_positions.append((self.robot_x, self.robot_y))

    # ── Odom watchdog ─────────────────────────────────────────────────────────

    def _odom_watchdog_cb(self):
        """Fires 3 s after startup; complains if no odom message has arrived."""
        self.odom_watchdog.cancel()   # one-shot
        if not self.odom_received:
            self.get_logger().error(
                '✗ No odometry received after 3 s! '
                'Run:  ros2 topic list | grep odom  '
                'to find the correct topic, then update ODOM_TOPIC_CANDIDATES.')
        else:
            self.get_logger().info('Odom watchdog: all good.')

    # ── State machine ─────────────────────────────────────────────────────────

    def update_state(self):
        """Main state machine logic executed at every timer tick (10 Hz)."""
        now = self.get_clock().now()
        elapsed = (now - self.state_start_time).nanoseconds / 1e9

        if self.nav_state == "EVALUATE":
            self.current_twist.linear.x = 0.0
            self.current_twist.angular.z = 0.0
            if elapsed > self.eval_duration:
                if self.orange_detected or self.white_detected or not self.is_path_green:
                    self.change_state("TURN_60")
                else:
                    self.change_state("FORWARD")

        elif self.nav_state == "FORWARD":
            if self.orange_detected or self.white_detected or not self.is_path_green:
                self.change_state("EVALUATE")
            else:
                self.current_twist.linear.x = 0.15
                self.current_twist.angular.z = 0.0

        elif self.nav_state == "TURN_60":
            if elapsed > self.turn_duration:
                self.change_state("EVALUATE")
            else:
                self.current_twist.linear.x = 0.0
                self.current_twist.angular.z = -self.turn_speed

        self.cmd_pub.publish(self.current_twist)
        self.draw_localization_map()

    def change_state(self, new_state):
        self.nav_state = new_state
        self.state_start_time = self.get_clock().now()
        self.get_logger().info(f'State → {self.nav_state}')

    def stop(self):
        self.cmd_pub.publish(Twist())

    # ── Image processing ──────────────────────────────────────────────────────

    def image_callback(self, ros_img):
        """Performs image processing to detect green path and obstacles."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return

        cv_image = cv2.resize(cv_image, (320, 240))
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # HSV colour thresholds
        lower_green  = np.array([35,  40,  40]);  upper_green  = np.array([85,  255, 255])
        lower_orange = np.array([0,  100, 100]);  upper_orange = np.array([20,  255, 255])
        lower_white  = np.array([0,    0, 180]);  upper_white  = np.array([180,  40, 255])

        kernel = np.ones((5, 5), np.uint8)
        green_mask  = cv2.morphologyEx(cv2.inRange(hsv, lower_green,  upper_green),  cv2.MORPH_OPEN, kernel)
        orange_mask = cv2.morphologyEx(cv2.inRange(hsv, lower_orange, upper_orange), cv2.MORPH_OPEN, kernel)
        white_mask  = cv2.morphologyEx(cv2.inRange(hsv, lower_white,  upper_white),  cv2.MORPH_OPEN, kernel)

        # Trapezoidal ROI in front of robot
        h, w = green_mask.shape
        top_offset = int(h * 0.3)
        pts = np.array([[0, h], [w, h], [w // 2 + 20, top_offset], [w // 2 - 20, top_offset]], np.int32)
        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts], 255)

        orange_roi = cv2.bitwise_and(orange_mask, orange_mask, mask=roi_mask)
        green_roi  = cv2.bitwise_and(green_mask,  green_mask,  mask=roi_mask)
        white_roi  = cv2.bitwise_and(white_mask,  white_mask,  mask=roi_mask)

        # Orange cone detection
        M_orange = cv2.moments(orange_roi)
        self.orange_detected = M_orange['m00'] > 50000
        if self.orange_detected:
            cone_x = self.robot_x + 0.5 * math.cos(self.robot_yaw)
            cone_y = self.robot_y + 0.5 * math.sin(self.robot_yaw)
            if not any(math.hypot(cone_x - cx, cone_y - cy) < 0.3 for cx, cy in self.cone_positions):
                self.cone_positions.append((cone_x, cone_y))

        # White obstacle detection
        self.white_detected = cv2.moments(white_roi)['m00'] > 50000

        # Green path ratio in bottom strip of ROI
        bottom_green = green_roi[h - 60:h, :]
        bottom_mask  = roi_mask[h - 60:h, :]
        roi_area = cv2.countNonZero(bottom_mask)
        self.is_path_green = (cv2.countNonZero(bottom_green) / roi_area >= 0.5) if roi_area > 0 else False

        # Camera view HUD
        cv2.polylines(cv_image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(cv_image, f'State: {self.nav_state}', (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('LIMO Camera Vision', cv_image)
        cv2.waitKey(1)

    # ── Map rendering ─────────────────────────────────────────────────────────

    def world_to_map(self, x, y):
        mx = int(x * self.map_resolution) + self.map_offset_x
        my = int(-y * self.map_resolution) + self.map_offset_y
        return mx, my

    def draw_localization_map(self):
        """
        Top-down 2D map.  Layers (bottom → top):
          1  Green blobs      — safe-path corridor (green_path_positions only)
          2  Green polyline   — centre-line of safe corridor
          3  Orange triangles — detected cone positions
          4  Blue circle      — robot body
          5  Red line         — robot heading
          6  HUD text         — odom status, state, counters
          7  Legend
        """
        map_img = np.zeros((self.map_size, self.map_size, 3), dtype=np.uint8)

        # ── Layer 1 & 2: Safe green path ──────────────────────────────────────
        if self.green_path_positions:
            for gx, gy in self.green_path_positions:
                px, py = self.world_to_map(gx, gy)
                cv2.circle(map_img, (px, py), 10, (0, 160, 50), -1)

            if len(self.green_path_positions) > 1:
                safe_pts = np.array(
                    [self.world_to_map(gx, gy) for gx, gy in self.green_path_positions],
                    dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(map_img, [safe_pts], isClosed=False,
                              color=(0, 255, 100), thickness=2)

        # ── Layer 3: Orange cone triangles ────────────────────────────────────
        for cx, cy in self.cone_positions:
            px, py = self.world_to_map(cx, cy)
            tri = np.array([(px, py - 11), (px - 10, py + 9), (px + 10, py + 9)], dtype=np.int32)
            cv2.drawContours(map_img, [tri], 0, (0, 165, 255), -1)
            cv2.drawContours(map_img, [tri], 0, (255, 255, 255), 1)

        # ── Layer 4: Robot body ───────────────────────────────────────────────
        rx, ry = self.world_to_map(self.robot_x, self.robot_y)
        cv2.circle(map_img, (rx, ry), 7, (200, 80, 0), -1)

        # ── Layer 5: Heading line ─────────────────────────────────────────────
        hx = int(rx + 18 * math.cos(-self.robot_yaw))
        hy = int(ry + 18 * math.sin(-self.robot_yaw))
        cv2.line(map_img, (rx, ry), (hx, hy), (0, 0, 220), 2)

        # ── Layer 6: HUD ──────────────────────────────────────────────────────
        # Odom status indicator (green = ok, red = not connected)
        if self.odom_received:
            odom_color = (0, 220, 60)
            odom_label = f'ODOM OK  [{self.active_odom_topic}]'
        else:
            odom_color = (0, 0, 220)
            odom_label = 'ODOM -- NO DATA --'

        cv2.circle(map_img, (12, 15), 6, odom_color, -1)
        cv2.putText(map_img, odom_label, (24, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, odom_color, 1)

        cv2.putText(map_img, f'X: {self.robot_x:.3f} m   Y: {self.robot_y:.3f} m',
                    (10, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(map_img, f'State: {self.nav_state}',
                    (10, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        cv2.putText(map_img, f'Safe pts: {len(self.green_path_positions)}   Cones: {len(self.cone_positions)}',
                    (10, 83), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (160, 160, 160), 1)

        # ── Layer 7: Legend ───────────────────────────────────────────────────
        ly = self.map_size - 65
        cv2.circle(map_img, (16, ly), 8, (0, 160, 50), -1)
        cv2.putText(map_img, 'Safe path', (30, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        ly += 22
        tri = np.array([(16, ly - 9), (8, ly + 7), (24, ly + 7)], dtype=np.int32)
        cv2.drawContours(map_img, [tri], 0, (0, 165, 255), -1)
        cv2.putText(map_img, 'Cone', (30, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        ly += 20
        cv2.circle(map_img, (16, ly), 6, (200, 80, 0), -1)
        cv2.putText(map_img, 'Robot', (30, ly + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200, 200, 200), 1)

        cv2.imshow('Localization Map', map_img)
        cv2.waitKey(1)

    # ── Shutdown ──────────────────────────────────────────────────────────────

    def destroy_node(self):
        try:
            self.stop()
            cv2.destroyAllWindows()
        finally:
            super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    node = LimoNavigationNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Shutting down.')
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()