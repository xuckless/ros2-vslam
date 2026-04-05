import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class LimoNavigationNode(Node):
    def __init__(self):
        super().__init__('limo_navigation_node')
        self.bridge = CvBridge()

        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.image_sub = self.create_subscription(
            Image,
            '/camera/color/image_raw',
            self.image_callback,
            qos_profile=qos_profile
        )

        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self.current_twist = Twist()
        self.is_path_green = False
        self.orange_detected = False
        self.white_detected = False

        self.nav_state = "EVALUATE"
        self.state_start_time = self.get_clock().now()

        self.turn_speed = 0.5  # rad/s
        self.turn_duration = (60.0 * np.pi / 180.0) / self.turn_speed
        self.eval_duration = 1.2

        self.timer = self.create_timer(0.1, self.update_state)
        self.get_logger().info('Running LIMO')

    def update_state(self):
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
                self.current_twist.angular.z = -self.turn_speed  # CLOCKWISE ONLY

        self.cmd_pub.publish(self.current_twist)

    def change_state(self, new_state):
        self.nav_state = new_state
        self.state_start_time = self.get_clock().now()
        self.get_logger().info(f"State Changed: {self.nav_state}")

    def stop(self):
        self.cmd_pub.publish(Twist())

    def image_callback(self, ros_img):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(ros_img, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        cv_image = cv2.resize(cv_image, (320, 240))
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Color thresholds
        lower_green = np.array([35, 40, 40])
        upper_green = np.array([85, 255, 255])

        lower_orange = np.array([0, 100, 100])
        upper_orange = np.array([20, 255, 255])

        lower_white = np.array([0, 0, 180])
        upper_white = np.array([180, 40, 255])

        # Masks
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        kernel = np.ones((5, 5), np.uint8)

        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)

        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_CLOSE, kernel)
        orange_mask = cv2.morphologyEx(orange_mask, cv2.MORPH_OPEN, kernel)

        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)

        h, w = green_mask.shape

        # --- Trapezoid ROI Setup ---
        # Ignore top 40%, covering only the bottom 60% of the image.
        top_offset = int(h * 0.3)

        pts = np.array([
            [0, h],
            [w, h],
            [w // 2 + 20, top_offset],
            [w // 2 - 20, top_offset]
        ], np.int32)

        roi_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(roi_mask, [pts], 255)

        # Apply ROI mask
        orange_roi = cv2.bitwise_and(orange_mask, orange_mask, mask=roi_mask)
        green_roi = cv2.bitwise_and(green_mask, green_mask, mask=roi_mask)
        white_roi = cv2.bitwise_and(white_mask, white_mask, mask=roi_mask)

        # Evaluate Obstacles inside ROI ONLY
        M_orange = cv2.moments(orange_roi)
        self.orange_detected = M_orange['m00'] > 50000

        M_white = cv2.moments(white_roi)
        self.white_detected = M_white['m00'] > 50000

        # Evaluate Green Path inside ROI ONLY
        bottom_green_roi = green_roi[h - 60:h, :]
        bottom_roi_mask = roi_mask[h - 60:h, :]
        roi_area = cv2.countNonZero(bottom_roi_mask)

        if roi_area > 0:
            green_ratio = cv2.countNonZero(bottom_green_roi) / roi_area
            self.is_path_green = (green_ratio >= 0.5)
        else:
            self.is_path_green = False

        # Visual Overlays
        cv2.polylines(cv_image, [pts], isClosed=True, color=(0, 255, 255), thickness=2)
        cv2.putText(cv_image, f"State: {self.nav_state}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        cv2.imshow("LIMO Camera Vision", cv_image)
        cv2.imshow("Masks: Green ROI", green_roi)
        cv2.imshow("Masks: Orange ROI", orange_roi)
        cv2.imshow("Masks: White ROI", white_roi)
        cv2.waitKey(1)

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
        node.get_logger().info("Shutting down (KeyboardInterrupt).")
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()