import cv2
import time

"""
Credits: Syed Ali; Roboflow; Gemini 
"""

# Dictionary mapping AprilTag IDs to navigation actions
SIGN_MAP = {
    0: "TURN LEFT",
    1: "TURN RIGHT",
    2: "GO FORWARD",
    3: "STOP",
    4: "NO ENTRY",
    5: "DEAD END"
}


def trackSign_CV(frame, detector):
    """
    Detects AprilTag 36h11 markers, draws corners, and overlays navigation actions.
    """
    # Detect markers
    corners, ids, rejected = detector.detectMarkers(frame)

    if ids is not None:
        # Draw the bounding box around detected tags
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)

        for i in range(len(ids)):
            tag_id = ids[i][0]
            action = SIGN_MAP.get(tag_id, "UNKNOWN TAG")

            # Get the top-left corner of the current tag for text placement
            curr_corners = corners[i][0]
            top_left = (int(curr_corners[0][0]), int(curr_corners[0][1]) - 10)

            # Overlay the decided action
            cv2.putText(frame, f"ID: {tag_id} -> {action}", top_left,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame


def main():
    # Setup AprilTag 36h11 Detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36h11)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)

    # Initialize Camera (Targeting 640x480 for >= 15 FPS)
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    prev_time = 0

    print("AprilTag CV Detection. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame.")
            break

        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if (curr_time - prev_time) > 0 else 0
        prev_time = curr_time

        # Process frame
        processed_frame = trackSign_CV(frame, detector)

        # Draw FPS counter
        cv2.putText(processed_frame, f"FPS: {int(fps)}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Display output
        cv2.imshow("Camera", processed_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()