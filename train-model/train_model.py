from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO("yolov8n.pt")

    model.train(
        data='robotics-vision-4/data.yaml',
        epochs=100,  # ← Increased: controlled environment converges well
        imgsz=640,
        batch=16,
        device='mps',
        name='v4_lab_optimized',

        # --- MINIMAL LIGHTING AUGMENTATION (Lab has consistent overhead LEDs) ---
        hsv_h=0.01,  # Very slight hue variation (LED color temp changes)
        hsv_s=0.3,  # Moderate saturation (handles minor brightness changes)
        hsv_v=0.3,  # Moderate brightness (LED dimming, shadows from robot)

        # --- CONSERVATIVE GEOMETRIC AUGMENTATION ---
        degrees=5.0,  # Small rotation (robot turning, minor camera shake)
        translate=0.05,  # Minimal translation (camera isn't moving wildly)
        scale=0.3,  # Moderate scaling (sign appears closer/farther as robot moves)
        shear=0.0,  # DISABLED: Camera is perpendicular to floor
        perspective=0.0,  # DISABLED: No wide-angle distortion in your setup

        # --- DIRECTIONAL INTEGRITY (CRITICAL - DO NOT CHANGE) ---
        flipud=0.0,  # ✓ No upside-down
        fliplr=0.0,  # ✓ No left-right reversal

        # --- SMALL OBJECT OPTIMIZATIONS (Your signs are tiny!) ---
        mosaic=1.0,  # ✓ KEEP: Crucial for small objects
        mixup=0.0,  # DISABLED: Blurring hurts small sign details
        copy_paste=0.3,  # INCREASED: Paste more sign instances for variety

        # --- OCCLUSION & NOISE ---
        erasing=0.1,  # REDUCED: Small signs can't afford to lose pixels

        # --- TRAINING STABILITY ---
        close_mosaic=10,  # Disable mosaic in last 10 epochs
        patience=15,  # Early stopping
        save_period=5,

        # --- OPTIMIZER TUNING ---
        optimizer='AdamW',  # Better for small datasets
        lr0=0.001,  # Slightly lower learning rate
        lrf=0.01,  # Final LR = 1% of initial
        momentum=0.937,
        weight_decay=0.0005,
        warmup_epochs=3,

        # --- LOSS FUNCTION TUNING ---
        box=7.5,  # Default is 7.5
        cls=0.5,  # Default is 0.5
        dfl=1.5,  # Default is 1.5
    )