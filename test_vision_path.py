#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import cv2
import numpy as np
import time


class LimoPathFollower(Node):
    def __init__(self):
        super().__init__('limo_path_follower')

        # --- TWEAKABLE PARAMETERS ---
        self.FORWARD_SPEED = 0.07  # Constant linear velocity (m/s)
        self.SENSITIVITY = 1.0  # Overall gain multiplier for angular response
        self.MAX_ANGULAR = 0.20  # Hard limit for rotation speed (rad/s)

        # PID Constants (Tuned for pixel-space error)
        self.KP = 0.005  # Proportional: Immediate response to error
        self.KI = 0.0001  # Integral: Corrects long-term steady-state offset
        self.KD = 0.001  # Derivative: Dampens oscillations/overshoot

        # Image Processing Settings
        self.IMAGE_CENTER_X = 320  # Target X-coordinate (Image Width / 2)
        self.ROI_VERTICES = np.array([[(200, 480), (280, 200), (360, 200), (440, 480)]], dtype=np.int32)
        self.HSV_LOWER_GREEN = np.array([35, 40, 40])
        self.HSV_UPPER_GREEN = np.array([85, 255, 255])
        self.MIN_CONTOUR_AREA = 1000  # Noise floor for green detection

        # Filters
        self.ALPHA = 0.75  # EMA Smoothing (0.0 to 1.0; lower is smoother)
        self.MAX_WINDUP = 1.0  # Caps the integral term to prevent runaway
        # ----------------------------

        # ROS2 Publisher
        self.publisher = self.create_publisher(Twist, '/cmd_vel', 10)

        # Hardware Camera
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        # State Memory
        self.integral = 0.0
        self.prev_error = 0.0
        self.smooth_error = 0.0
        self.last_time = time.time()

        self.create_timer(0.1, self.process_loop)

    def get_centroid(self, contour):
        M = cv2.moments(contour)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            return cx, cy
        return None

    def process_loop(self):
        ret, frame = self.cap.read()
        if not ret: return

        # 1. Image Pre-processing
        display = frame.copy()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.HSV_LOWER_GREEN, self.HSV_UPPER_GREEN)

        # 2. Apply ROI
        roi_mask = np.zeros(mask.shape, dtype=np.uint8)
        cv2.fillPoly(roi_mask, self.ROI_VERTICES, 255)
        mask = cv2.bitwise_and(mask, roi_mask)

        # 3. Path Detection
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = [c for c in contours if cv2.contourArea(c) > self.MIN_CONTOUR_AREA]

        twist = Twist()
        line_detected = False
        curr_error = 0.0

        if valid_contours:
            line_detected = True
            best_cnt = max(valid_contours, key=cv2.contourArea)
            centroid = self.get_centroid(best_cnt)

            if centroid:
                cx, cy = centroid
                raw_error = cx - self.IMAGE_CENTER_X

                # Jitter Reduction (EMA Filter)
                self.smooth_error = (self.ALPHA * raw_error) + (1.0 - self.ALPHA) * self.smooth_error
                curr_error = self.smooth_error

                # PID Logic
                curr_time = time.time()
                dt = max(curr_time - self.last_time, 0.001)
                self.last_time = curr_time

                self.integral = np.clip(self.integral + (curr_error * dt), -self.MAX_WINDUP, self.MAX_WINDUP)
                derivative = (curr_error - self.prev_error) / dt

                pid_out = (self.KP * curr_error) + (self.KI * self.integral) + (self.KD * derivative)
                self.prev_error = curr_error

                # Apply Sensitivity and Saturation
                yaw_cmd = -pid_out * self.SENSITIVITY

                twist.linear.x = self.FORWARD_SPEED
                twist.angular.z = np.clip(yaw_cmd, -self.MAX_ANGULAR, self.MAX_ANGULAR)

                # Visuals
                cv2.circle(display, (cx, cy), 8, (0, 255, 255), -1)
                cv2.line(display, (self.IMAGE_CENTER_X, cy), (cx, cy), (0, 255, 255), 2)

        if not line_detected:
            # Safety stop
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            self.integral = 0.0

        self.publisher.publish(twist)

        # HUD
        cv2.polylines(display, [self.ROI_VERTICES], True, (0, 0, 255), 2)
        status_color = (0, 255, 0) if line_detected else (0, 0, 255)
        cv2.putText(display, f"Error: {curr_error:.1f} px", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(display, f"Sensitivity: {self.SENSITIVITY}x", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 165, 0), 1)
        cv2.putText(display, f"Status: {'TRACKING' if line_detected else 'LOST'}", (10, 75), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, status_color, 2)

        cv2.imshow("LIMO Path Follower", display)
        cv2.waitKey(1)


def main():
    rclpy.init()
    node = LimoPathFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cap.release()
        cv2.destroyAllWindows()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()