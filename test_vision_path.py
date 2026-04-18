#!/usr/bin/env python3
import cv2
import numpy as np
import math

IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
IMAGE_CENTER_X = IMAGE_WIDTH // 2
OVERLAY_ALPHA = 0.5

HSV_LOWER_GREEN = np.array([35, 40, 40])
HSV_UPPER_GREEN = np.array([85, 255, 255])
MIN_CONTOUR_AREA = 1000
HORIZONTAL_THRESHOLD_DEG = 65


def extract_errors(contour):
    [vx, vy, cx, cy] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy = float(vx), float(vy)
    if vy > 0:
        vx, vy = -vx, -vy
    heading_err = math.atan2(vx, -vy)

    M = cv2.moments(contour)
    if M['m00'] > 0:
        centroid_x = M['m10'] / M['m00']
        lateral_err_px = centroid_x - IMAGE_CENTER_X
    else:
        lateral_err_px = 0.0

    return heading_err, lateral_err_px


def main():
    cap = cv2.VideoCapture(0)

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
                [vx, vy, cx, cy] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
                vx, vy = float(vx), float(vy)
                if vy > 0:
                    vx, vy = -vx, -vy

                angle_from_vertical = abs(math.degrees(math.atan2(vx, -vy)))
                if angle_from_vertical > HORIZONTAL_THRESHOLD_DEG:
                    continue
                valid_contours.append(cnt)

        if valid_contours:
            best = max(valid_contours, key=cv2.contourArea)
            line_ok = True
            heading_err, lat_err_px = extract_errors(best)

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

        hud_col = (0, 255, 0) if line_ok else (0, 0, 255)
        cv2.putText(display, f"H_err: {heading_err:+.2f} rad | L_err: {lat_err_px:+.1f} px",
                    (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(display, f"line: {'YES' if line_ok else 'NO / HORIZONTAL IGNORED'}",
                    (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5, hud_col, 1)

        cv2.imshow("Green Mask", green_mask)
        cv2.imshow("Path Alignment Test", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()