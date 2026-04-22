#!/usr/bin/env python3
import sys
import os
import math
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry

# Force system NumPy for compatibility if needed
sys.path.insert(0, '/usr/lib/python3/dist-packages')

# Configuration
LOCAL_DIR = os.path.expanduser("~/Desktop/limo_caps")
TARGET_DIST = 0.5
NUM_PICS = 15
SPEED = 0.1
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480


class LimoCaptureNode(Node):
    def __init__(self):
        super().__init__('limo_capture_node')

        # Open camera hardware directly (matches demo script)
        # Note: Ensure any ROS camera driver nodes are stopped to avoid device conflict
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            self.get_logger().error("Could not open camera device /dev/video0.")

        # Create Directory on Desktop
        if not os.path.exists(LOCAL_DIR):
            os.makedirs(LOCAL_DIR, exist_ok=True)
            self.get_logger().info(f"Directory CREATED at: {LOCAL_DIR}")

        # ROS interfaces
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        self.odom_sub = self.create_subscription(Odometry, '/wheel/odom', self.odom_cb, 10)

        # Timer drives both the movement logic and camera reading
        self.timer = self.create_timer(0.1, self.control_loop)

        self.start_x = None
        self.start_y = None
        self.dist_driven = 0.0
        self.pics_taken = 0
        self.done = False

        self.get_logger().info("Node started. Driving 50cm and capturing from hardware camera...")

    def odom_cb(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        if self.start_x is None:
            self.start_x = x
            self.start_y = y
        self.dist_driven = math.hypot(x - self.start_x, y - self.start_y)

    def control_loop(self):
        if self.done or self.start_x is None:
            return

        # Read frame directly from hardware
        ret, frame = self.cap.read()
        if not ret:
            return

        # Process and show frame
        frame = cv2.resize(frame, (IMAGE_WIDTH, IMAGE_HEIGHT))
        cv2.imshow("LIMO Live Feed", frame)
        cv2.waitKey(1)

        # Move Robot
        twist = Twist()
        twist.linear.x = SPEED
        self.cmd_pub.publish(twist)

        # Capture Logic
        interval = TARGET_DIST / NUM_PICS
        if self.dist_driven >= self.pics_taken * interval and self.pics_taken < NUM_PICS:
            filename = os.path.join(LOCAL_DIR, f"cap_{self.pics_taken:02d}.jpg")
            cv2.imwrite(filename, frame)
            self.get_logger().info(f"Image {self.pics_taken + 1}/{NUM_PICS} saved: {filename}")
            self.pics_taken += 1

        # Check Finish
        if self.dist_driven >= TARGET_DIST:
            self.get_logger().info(f"Target of {TARGET_DIST}m reached. Stopping.")
            self.done = True
            self.cmd_pub.publish(Twist())  # Stop
            self.cleanup()
            rclpy.shutdown()

    def cleanup(self):
        if hasattr(self, 'cap') and self.cap.isOpened():
            self.cap.release()
        cv2.destroyAllWindows()


def main(args=None):
    rclpy.init(args=args)
    node = LimoCaptureNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.cleanup()
        if rclpy.ok():
            node.destroy_node()
            rclpy.shutdown()


if __name__ == '__main__':
    main()