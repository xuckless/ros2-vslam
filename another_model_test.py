import os
import cv2
from ultralytics import YOLO

if __name__ == '__main__':
    # 1. Load your exact trained weights
    model = YOLO("weights/best.pt")

    # 2. Point this to the directory containing your images
    image_dir = "/Users/ali/Downloads/limo_caps/forward-right-left"

    # 3. Loop through all valid image files in the directory
    for filename in os.listdir(image_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(image_dir, filename)

            # Run inference with the exact size you trained on
            results = model(image_path, conf=0.50, imgsz=640)
            result = results[0]

            # --- CUSTOM STRICT FILTERING ---
            if len(result.boxes) > 0:
                keep_indices = []
                boxes = result.boxes.xyxy

                # Calculate max allowed area dynamically (30% of original image resolution)
                img_h, img_w = result.orig_shape
                max_allowed_area = 0.200000 * (img_w * img_h)

                for i in range(len(boxes)):
                    box_a = boxes[i]

                    # 1. Check if box is egregiously big
                    box_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
                    if box_area > max_allowed_area:
                        continue  # Skip adding this box to the final output

                    # 2. Check for encapsulation
                    is_encapsulating = False
                    for j in range(len(boxes)):
                        if i == j:
                            continue
                        box_b = boxes[j]

                        # Check if Box A completely surrounds Box B
                        if (box_a[0] <= box_b[0] and box_a[1] <= box_b[1] and
                                box_a[2] >= box_b[2] and box_a[3] >= box_b[3]):
                            is_encapsulating = True
                            break  # Flag the giant box for removal

                    if not is_encapsulating:
                        keep_indices.append(i)

                # Apply the filter to keep only valid boxes
                result.boxes = result.boxes[keep_indices]
            # -----------------------------------------------------------

            # Display the strictly filtered result
            annotated_frame = result.plot()
            cv2.imshow("True Local Inference", annotated_frame)

            # Wait for user to press a key before moving to the next image
            key = cv2.waitKey(0)
            if key == 27 or key == ord('q'):
                break

    # Clean up when done
    cv2.destroyAllWindows()