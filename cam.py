from ultralytics import YOLO
import cv2
import numpy as np

# calculate boxes are overlapped
def box_region(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # Compute overlapping Area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)    # To prevent division by zero, a very small number is added

# calculate PPE is on the person
def overlapping_degree(object_box, person_box, threshold=0.3):
    xA = max(object_box[0], person_box[0])
    yA = max(object_box[1], person_box[1])
    xB = min(object_box[2], person_box[2])
    yB = min(object_box[3], person_box[3])

    # Compute overlapping Area
    interArea = max(0, xB - xA) * max(0, yB - yA)
    objArea = (object_box[2] - object_box[0]) * (object_box[3] - object_box[1])

    # if overlapping_percent > threshold, it is considered overlapping
    overlapping_percent = (interArea / (objArea + 1e-6))  # To prevent division by zero, a very small number is added
    return overlapping_percent > threshold

def cam_ppe_check():
    model = YOLO('runs/detect/train9/weights/best.pt')  # pretrained model path

    # Class index(data.yaml)
    CLASS_GLOVE = 0
    CLASS_MASK = 1
    CLASS_COAT = 2
    CLASS_PERSON = 3

    # opne Camera
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("fail to open Camera")
        return

    while True:
        # Read frame at cap
        ret, frame = cap.read()

        # if can't read frame, break
        if not ret:
            break
        
        # Detect using pretrained model
        results = model.predict(source=frame, save=False, conf=0.3)


        for r in results:
            
            # Made images with bounding box and labels
            dst = frame.copy()
            boxes = r.boxes
            class_ids = boxes.cls.cpu().numpy() # Class ID for each boxes
            xyxy = boxes.xyxy.cpu().numpy() # Get coordinates [x1, y1, x2, y2]

            person_boxes = []
            other_objects = []

            # Repeat for detected boxes
            for cls_id, box in zip(class_ids, xyxy):
                cls_id = int(cls_id)
                x1, y1, x2, y2 = map(int, box)
                if cls_id == CLASS_PERSON:
                    person_boxes.append(box)
                else:
                    other_objects.append((cls_id, box))
                    x1, y1, x2, y2 = map(int, box)
                    label = "Glove" if cls_id == CLASS_GLOVE else \
                            "Mask" if cls_id == CLASS_MASK else \
                            "Coat" if cls_id == CLASS_COAT else f"Class {cls_id}"
                    cv2.rectangle(dst, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(dst, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Repeat for detected person
            for i, person_box in enumerate(person_boxes):

                # Sepearate coordinates
                px1, py1, px2, py2 = person_box

                # PPE variables
                gloves = 0
                masks = 0
                coats = 0

                # Repeat for detected PPE
                for cls_id, obj_box in other_objects:

                    # Calculate the degree of overlap
                    iou = box_region(person_box, obj_box)
                    overlap = overlapping_degree(obj_box, person_box, threshold=0.3)
                    
                    # If overlap, judge PPE worn by a person
                    if iou > 0.1 or overlap:
                        if cls_id == CLASS_GLOVE:
                            gloves += 1
                        elif cls_id == CLASS_MASK:
                            masks += 1
                        elif cls_id == CLASS_COAT:
                            coats += 1

                missing_items = []
                # If there are any missing items, append it to messing_items
                if gloves < 2:
                    missing_items.append(f"Gloves({gloves}/2)")
                if masks < 1:
                    missing_items.append(f"Mask({masks}/1)")
                if coats < 1:
                    missing_items.append(f"Coat({coats}/1)")

                # If there are missing items, display warning messages
                if missing_items:
                    warning_message = f"Person {i+1}: ({', '.join(missing_items)})"
                    print(warning_message)
                    # Draw Red boxe on the Person boxe
                    cv2.rectangle(dst, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                    # location of text
                    text_x = int(px1)
                    text_y = max(30, int(py1) - 10)
                    # Output warning text to image
                    cv2.putText(dst, warning_message, (text_x, text_y),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
           
            # Display result image
            cv2.imshow("PPE Check (CAM)", dst)

        # if push 'q', quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    cam_ppe_check()




