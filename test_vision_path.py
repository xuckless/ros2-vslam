#!/usr/bin/env python3
import cv2
import numpy as np
import math
import time

# -- Camera Settings -----------------------------------------
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CENTER_X = IMAGE_WIDTH // 2
OVERLAY_ALPHA = 0.5

# -- HSV Green Tape Segmentation -----------------------------
HSV_LOWER_GREEN = np.array([35, 40, 40])
HSV_UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 1000
HORIZONTAL_THRESHOLD_DEG = 65

# -- Movement & Dual-PID Control -----------------------------
FORWARD_SPEED = 0.15
CONTROL_LOOP_PERIOD = 0.1  # Based on 10Hz in main script

KP_HEADING = 1.5
KI_HEADING = 0.05
KD_HEADING = 0.20

KP_LATERAL = 0.004
KI_LATERAL = 0.0001
KD_LATERAL = 0.001

MAX_INTEGRAL_WINDUP = 1.0
MAX_ANGULAR_CORRECTION = 0.40
MAX_HEADING_TO_DRIVE = 0.25
SEVERE_LATERAL_OFFSET = 60


def extract_errors(contour):
    [vx, vy, cx, cy] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    if vy > 0:
        vx, vy = -vx, -vy

    heading_err = math.atan2(vx, -vy)
    angle_from_vertical = abs(math.degrees(heading_err))
    is_horizontal = angle_from_vertical > HORIZONTAL_THRESHOLD_DEG

    M = cv2.moments(contour)
    if M['m00'] > 0:
        centroid_x = M['m10'] / M['m00']
        lateral_err_px = centroid_x - IMAGE_CENTER_X
    else:
        lateral_err_px = 0.0

    return heading_err, lateral_err_px, is_horizontal


def main():
    cap = cv2.VideoCapture(0)

    # PID Memory Variables
    integral_heading = 0.0
    integral_lateral = 0.0
    prev_heading_err = 0.0
    prev_lat_err_px = 0.0
    last_time = time.time()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

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

        contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid_contours = []

        for cnt in contours:
            if cv2.contourArea(cnt) > MIN_CONTOUR_AREA:
                h_err, l_err, is_horiz = extract_errors(cnt)

                if is_horiz:
                    cv2.drawContours(display, [cnt], -1, (128, 128, 0), 1)
                    continue

                valid_contours.append(cnt)

        if valid_contours:
            best = max(valid_contours, key=cv2.contourArea)
            line_ok = True
            heading_err, lat_err_px, _ = extract_errors(best)

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
                cv2.line(display, (IMAGE_CENTER_X, cy_px), (cx_px, cy_px), (0, 255, 255), 2)

        # -- Dual-PID Movement Logic Extraction --
        linear_x = 0.0
        angular_z = 0.0

        current_time = time.time()
        dt = current_time - last_time
        if dt <= 0.0:
            dt = CONTROL_LOOP_PERIOD
        last_time = current_time

        if line_ok:
            speed_multiplier = 1.0
            if abs(lat_err_px) > SEVERE_LATERAL_OFFSET:
                speed_multiplier = 0.5

            # === HEADING PID ===
            integral_heading += heading_err * dt
            integral_heading = max(-MAX_INTEGRAL_WINDUP, min(MAX_INTEGRAL_WINDUP, integral_heading))
            derivative_heading = (heading_err - prev_heading_err) / dt
            pid_heading = (KP_HEADING * heading_err) + (KI_HEADING * integral_heading) + (
                        KD_HEADING * derivative_heading)

            # === LATERAL PID ===
            integral_lateral += lat_err_px * dt
            integral_lateral = max(-MAX_INTEGRAL_WINDUP, min(MAX_INTEGRAL_WINDUP, integral_lateral))
            derivative_lateral = (lat_err_px - prev_lat_err_px) / dt
            pid_lateral = (KP_LATERAL * lat_err_px) + (KI_LATERAL * integral_lateral) + (
                        KD_LATERAL * derivative_lateral)

            prev_heading_err = heading_err
            prev_lat_err_px = lat_err_px

            yaw_cmd = -pid_heading - pid_lateral
            yaw_cmd = max(-MAX_ANGULAR_CORRECTION, min(MAX_ANGULAR_CORRECTION, yaw_cmd))

            if abs(heading_err) <= MAX_HEADING_TO_DRIVE:
                linear_x = FORWARD_SPEED * speed_multiplier
                angular_z = yaw_cmd
            else:
                angular_z = yaw_cmd

        else:
            # Reset memory on line loss
            integral_heading = 0.0
            integral_lateral = 0.0
            prev_heading_err = 0.0
            prev_lat_err_px = 0.0

        # -- HUD Display --
        hud_col = (0, 255, 0) if line_ok else (0, 0, 255)
        cv2.putText(display, f"H_err: {heading_err:+.2f} rad | L_err: {lat_err_px:+.1f} px",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"line: {'YES' if line_ok else 'NO / HORIZONTAL IGNORED'}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)
        cv2.putText(display, f"CMD_VEL -> linear.x: {linear_x:.2f} | angular.z: {angular_z:.2f}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("PID Movement Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()