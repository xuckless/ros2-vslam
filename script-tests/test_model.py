from ultralytics import YOLO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os

model = YOLO('weights/best.pt')

metrics = model.val(data='test-set/data.yaml', split='test')

save_dir = metrics.save_dir

plots_to_display = {
    'confusion_matrix.png': 'Confusion Matrix',
    'F1_curve.png': 'F1 Confidence Curve',
    'val_batch0_pred.jpg': 'Sample Test Predictions'
}

for filename, title in plots_to_display.items():
    filepath = os.path.join(save_dir, filename)
    if os.path.exists(filepath):
        img = mpimg.imread(filepath)
        plt.figure(figsize=(12, 8))
        plt.imshow(img)
        plt.title(title)
        plt.axis('off')
        plt.show()
    else:
        print(f"File not found: {filepath}")

print(f"Mean Average Precision (mAP@50-95): {metrics.box.map}")
print(f"Mean Average Precision (mAP@50): {metrics.box.map50}")