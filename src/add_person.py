from ultralytics import YOLO
import cv2
import os

# 1. Path (Train/Test/Valid)
image_dir = 'datasets/mix2/test/images'
label_dir = 'datasets/mix2/test/labels'

# 2. YOLOv8 pre-trained model load 
model = YOLO('yolov8n.pt') 

# 3. Detection Class: person only (class 0 in COCO)
TARGET_CLASS_ID = 0
NEW_CLASS_ID = 3  # Class ID 

# 4. Detect for all images
for name in os.listdir(image_dir):
    if not name.lower().endswith(('.jpg', '.jpeg', '.png')):
        continue

    img_path = os.path.join(image_dir, name)
    label_path = os.path.join(label_dir, name.replace('.jpg', '.txt').replace('.png', '.txt'))

    results = model(img_path, verbose=False)
    boxes = results[0].boxes

    # 5. load original labels
    if os.path.exists(label_path):
        with open(label_path, 'r') as f:
            original_labels = f.read().strip().split('\n')
    else:
        original_labels = []

    new_labels = []

    # 6. Add box(detected person)
    for box in boxes:
        cls = int(box.cls[0])
        if cls != TARGET_CLASS_ID:
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        # Transform box to YOLO type
        x_center = (x1 + x2) / 2 / w
        y_center = (y1 + y2) / 2 / h
        width = (x2 - x1) / w
        height = (y2 - y1) / h

        new_labels.append(f"{NEW_CLASS_ID} {x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}")

    # 7. Save new person label
    combined = original_labels + new_labels
    with open(label_path, 'w') as f:
        f.write('\n'.join(combined))
