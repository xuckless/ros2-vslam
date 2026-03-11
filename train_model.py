from ultralytics import YOLO

if __name__ == '__main__':
    # Load the YOLOv8 nano model
    model = YOLO("yolov8n.pt")

    # Train the model pointing to the data.yaml inside your specific folder
    results = model.train(data="robotics-vision-2/data.yaml", epochs=50, imgsz=640)