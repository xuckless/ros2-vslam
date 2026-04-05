import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge
import cv2
import numpy as np


class LimoNavigationNode(Node):
    def __init__(self):
        super().__init__('limo_navigation_node')
        self.bridge = CvBridge()

        # Subscribe to camera
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            10
        )

        # Publisher for navigation commands
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.get_logger().info('Vision Navigation Node Started (Tank Turn Mode)')

    def image_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Color Segmentation thresholds
        lower_green = np.array([40, 50, 50])
        upper_green = np.array([80, 255, 255])

        lower_orange = np.array([5, 150, 150])
        upper_orange = np.array([15, 255, 255])

        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)

        h, w = green_mask.shape
        twist = Twist()

        # 1. Obstacle Avoidance (Orange Cones)
        M_orange = cv2.moments(orange_mask)

        if M_orange['m00'] > 500000:
            cx = int(M_orange['m10'] / M_orange['m00'])
            twist.linear.x = 0.0  # Stop forward movement for a standing turn

            if cx < w / 2:
                twist.angular.z = -0.5  # Tank turn right
            else:
                twist.angular.z = 0.5  # Tank turn left

        else:
            # 2. Boundary Detection (Green Carpet)
            bottom_green = green_mask[h - 100:h, :]
            green_ratio = cv2.countNonZero(bottom_green) / (100 * w)

            if green_ratio < 0.6:
                # Out of bounds: Standing 360-style turn to recover
                twist.linear.x = 0.0
                twist.angular.z = 0.8
            else:
                # Safe path: Move straight
                twist.linear.x = 0.15
                twist.angular.z = 0.0

        self.cmd_pub.publish(twist)


def main(args=None):
    rclpy.init(args=args)
    node = LimoNavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()