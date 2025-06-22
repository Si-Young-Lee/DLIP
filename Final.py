from ultralytics import YOLO
import cv2 as cv
import numpy as np
import os
from datetime import datetime
import tensorflow as tf
import time
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import threading

# Config
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10

# EDGES = {
#     (5, 7), (7, 9),
#     (6, 8), (8, 10),
#     (5, 6)
# }

CONFIDENCE_THRESHOLD = 0.2
WARNING_HOLD_FRAMES = 10
MAX_TRACKED = 6

# Directory setting
os.makedirs("warning_logs", exist_ok=True)

# Load Models
model = tf.saved_model.load("saved_model")
inference_func = model.signatures["serving_default"]


# Function to send an email alert when a warning is triggered
def send_email_alert(ppe_status: str, person_id: int, timestamp: str):
    SENDER_EMAIL = "danielyuha@gmail.com"
    APP_PASSWORD = "hchv zbsy hekj kuuz"  # App password for Gmail
    RECEIVER_EMAIL = "danielyuha@gmail.com"

    try:
        msg = MIMEMultipart()
        msg["Subject"] = "DLIP Alert - PPE Violation Detected"
        msg["From"] = SENDER_EMAIL
        msg["To"] = RECEIVER_EMAIL

        body = f"""
        [DLIP Real-Time Safety Monitoring System]

        Time: {timestamp}
        Person: Person {person_id + 1}
        Detected: {ppe_status}

        ※ This alert was sent automatically by the deep learning-based surveillance system.
        """

        msg.attach(MIMEText(body, "plain"))

        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
            smtp.login(SENDER_EMAIL, APP_PASSWORD)
            smtp.send_message(msg)
            print(f"[Email Sent] To: {RECEIVER_EMAIL} / Content: {ppe_status}")

    except Exception as e:
        print(f"[Email Failed] {e}")


# Resize and pad the input image to match the model input format
def preprocess_frame(frame):
    img = tf.image.resize_with_pad(tf.convert_to_tensor(frame), 256, 256)
    input_tensor = tf.expand_dims(tf.cast(img, dtype=tf.int32), axis=0)
    return input_tensor

# Detect people from the current frame using MoveNet
def detect_people(frame, original_shape):
    input_tensor = preprocess_frame(frame)
    output = inference_func(input=input_tensor)
    people = output["output_0"].numpy()[0]
    h, w, _ = original_shape
    keypoints_list = []

    for person in people:
        score = person[55]
        if score < 0.2:
            continue
        keypoints = person[:51].reshape((17, 3))
        shaped = np.multiply(keypoints, [h, w, 1])
        keypoints_list.append(shaped)

    return keypoints_list

# Calculate average Euclidean distance between two poses
def pose_similarity(pose1, pose2):
    dists = []
    for k1, k2 in zip(pose1, pose2):
        if k1[2] > CONFIDENCE_THRESHOLD and k2[2] > CONFIDENCE_THRESHOLD:
            dists.append(np.linalg.norm(k1[:2] - k2[:2]))
    return np.mean(dists) if dists else float('inf')

# Match current poses with previously tracked IDs
def match_poses_with_ids(current_poses, tracked_poses, threshold=40):
    id_map = [-1] * len(current_poses)
    used = [False] * MAX_TRACKED

    for i, curr_pose in enumerate(current_poses):
        best_id = -1
        min_sim = float('inf')
        for j in range(MAX_TRACKED):
            if tracked_poses[j] is None or used[j]:
                continue
            sim = pose_similarity(curr_pose, tracked_poses[j])
            if sim < threshold and sim < min_sim:
                min_sim = sim
                best_id = j
        if best_id != -1:
            id_map[i] = best_id
            used[best_id] = True

    # Register new person
    for i, pose in enumerate(current_poses):
        if id_map[i] == -1:
            for j in range(MAX_TRACKED):
                if not used[j] and tracked_poses[j] is None:
                    id_map[i] = j
                    used[j] = True
                    tracked_poses[j] = pose
                    break

    return id_map

# Determine whether either arm is raised above the shoulder
def is_one_arm_raised(shaped):
    l_sh_y, _, l_sh_c = shaped[LEFT_SHOULDER]
    r_sh_y, _, r_sh_c = shaped[RIGHT_SHOULDER]
    l_wr_y, _, l_wr_c = shaped[LEFT_WRIST]
    r_wr_y, _, r_wr_c = shaped[RIGHT_WRIST]

    if l_sh_c < CONFIDENCE_THRESHOLD or r_sh_c < CONFIDENCE_THRESHOLD:
        return False
    if l_wr_c < CONFIDENCE_THRESHOLD or r_wr_c < CONFIDENCE_THRESHOLD:
        return False

    left_up = l_wr_y < l_sh_y - 30
    right_up = r_wr_y < r_sh_y - 30
    return left_up or right_up


# def draw_skeleton(frame, shaped):
#     for p1, p2 in EDGES:
#         y1, x1, c1 = shaped[p1]
#         y2, x2, c2 = shaped[p2]
#         if c1 > CONFIDENCE_THRESHOLD and c2 > CONFIDENCE_THRESHOLD:
#             cv.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
#             cv.circle(frame, (int(x1), int(y1)), 4, (0, 255, 0), -1)
#             cv.circle(frame, (int(x2), int(y2)), 4, (0, 255, 0), -1)


    
def draw_warning_overlay(frame):
    h, w, _ = frame.shape

    # Empty mask
    mask = np.zeros((h, w, 3), dtype=np.uint8)

    # Define Boundary region
    border_thickness = 50  

    # Create borders in each direction for the red gradient
    for i in range(border_thickness):
        alpha = (1 - i / border_thickness) ** 2 
        intensity = int(255 * alpha)

        color = (0, 0, intensity)

        # upside borders
        cv.rectangle(mask, (i, i), (w - i, i + 1), color, -1)
        # lowerside borders
        cv.rectangle(mask, (i, h - i - 1), (w - i, h - i), color, -1)
        # leftside borders
        cv.rectangle(mask, (i, i), (i + 1, h - i), color, -1)
        # rightside borders
        cv.rectangle(mask, (w - i - 1, i), (w - i, h - i), color, -1)

    # Mask + Frame
    frame[:] = cv.addWeighted(frame, 1.0, mask, 0.5, 0)

def save_image_async(img, filename):
    cv.imwrite(filename, img)



# Save warning log, image, and send an email alert
def log_warning(person_id, frame):
    now = datetime.now()
    timestamp = now.strftime("date_%Y-%m-%d_time_%Hh_%Mm_%Ss")
    readable_time = now.strftime("%Y-%m-%d %H:%M:%S")

    img_copy = frame.copy()
    y, x, _ = prev_poses[person_id][0]
    cv.putText(img_copy, f"WARNING {person_id+1}", (int(x)-50, int(y)-50), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
    cv.rectangle(img_copy, (int(x)-60, int(y)-70), (int(x)+100, int(y)-30), (0,0,255), 2)

    # filename = f"warning_logs/warning_{timestamp}_person{person_id+1}.jpg"
    # cv.imwrite(filename, frame)
    filename = f"warning_logs/warning_{timestamp}_person{person_id+1}.jpg"

    threading.Thread(target=save_image_async, args=(img_copy, filename)).start()

    with open("warning_logs/warning_log.txt", "a") as f:
        f.write(f"[{readable_time}] Person {person_id+1} WARNING → {filename}\n")

    # send_email_alert("Arm raise detected", person_id, readable_time)
    threading.Thread(target=send_email_alert, args=("Arm raise detected", person_id, readable_time)).start()
    print(f"[LOG] Warning for Person {person_id+1} at {readable_time}")


# Initialize tracking state
warning_counters = [0] * MAX_TRACKED
log_saved = [False] * MAX_TRACKED
prev_poses = [None] * MAX_TRACKED

# Calculate IOU
def box_region(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return interArea / float(boxAArea + boxBArea - interArea + 1e-6)

# Calcualte overlapping degree
def is_overlapping(obj_box, person_box, threshold=0.3):
    xA = max(obj_box[0], person_box[0])
    yA = max(obj_box[1], person_box[1])
    xB = min(obj_box[2], person_box[2])
    yB = min(obj_box[3], person_box[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    objArea = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])
    return (interArea / (objArea + 1e-6)) > threshold

def cam_ppe_check():
    frame_count = 0 
    model = YOLO('runs/detect/train9/weights/best.pt')  # 학습된 모델 경로

    # Index of class (data.yaml)
    CLASS_GLOVE = 0
    CLASS_MASK = 1
    CLASS_COAT = 2
    CLASS_PERSON = 3

    cap = cv.VideoCapture(0)
    if not cap.isOpened():
        print("fail to open camera")
        return

    while True:
        start_time = time.time()
        frame_count += 1
        ret, frame = cap.read()
        if not ret:
            break

        results = model.predict(source=frame, save=False, conf=0.3)

        for r in results:
            dst = frame.copy()
            boxes = r.boxes
            class_ids = boxes.cls.cpu().numpy()
            xyxy = boxes.xyxy.cpu().numpy()

            person_boxes = []
            other_objects = []

            for cls_id, box in zip(class_ids, xyxy):
                if int(cls_id) == CLASS_PERSON:
                    person_boxes.append(box)
                else:
                    other_objects.append((int(cls_id), box))
                    x1, y1, x2, y2 = map(int, box)
                    label = "Glove" if cls_id == CLASS_GLOVE else \
                            "Mask" if cls_id == CLASS_MASK else \
                            "Coat" if cls_id == CLASS_COAT else f"Class {cls_id}"
                    cv.rectangle(dst, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv.putText(dst, label, (x1, y1 - 10),
                                cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            for i, person_box in enumerate(person_boxes):
                px1, py1, px2, py2 = person_box
                gloves = 0
                masks = 0
                coats = 0

                for cls_id, obj_box in other_objects:
                    iou = box_region(person_box, obj_box)
                    overlap = is_overlapping(obj_box, person_box, threshold=0.3)

                    if iou > 0.1 or overlap:
                        if cls_id == CLASS_GLOVE:
                            gloves += 1
                        elif cls_id == CLASS_MASK:
                            masks += 1
                        elif cls_id == CLASS_COAT:
                            coats += 1

                missing_items = []
                if gloves < 2:
                    missing_items.append(f"Gloves({gloves}/2)")
                if masks < 1:
                    missing_items.append(f"Mask({masks}/1)")
                if coats < 1:
                    missing_items.append(f"Coat({coats}/1)")

                if missing_items:
                    warning_msg = f"Person {i+1}:({', '.join(missing_items)})"
                    print(warning_msg)
                    # cv.rectangle(dst, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                    text_x = int(px1)
                    text_y = max(30, int(py1) - 10)
                    cv.putText(dst, warning_msg, (text_x, text_y),
                                cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # cv.imshow("PPE Check (CAM)", dst)

        

        keypoints_list = detect_people(dst, dst.shape)
        id_map = match_poses_with_ids(keypoints_list, prev_poses)

        warning_triggered = any(counter >= WARNING_HOLD_FRAMES for counter in warning_counters)

        for i, shaped in enumerate(keypoints_list):
            person_id = id_map[i]
            if person_id == -1:
                continue

            prev_poses[person_id] = shaped
            # draw_skeleton(dst, shaped)

            if is_one_arm_raised(shaped):
                warning_counters[person_id] += 1
            else:
                warning_counters[person_id] = max(0, warning_counters[person_id] - 2)

            if warning_counters[person_id] >= WARNING_HOLD_FRAMES:
                if not log_saved[person_id]:
                    log_warning(person_id, dst)
                    log_saved[person_id] = True
                   
            else:
                log_saved[person_id] = False

        if warning_triggered and (frame_count % 20 < 10):
            draw_warning_overlay(dst) 
            # draw_body_warning_text(dst, person_id, shaped)

        cv.imshow("MultiPose Warning", dst)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
            # Calculate FPS
        fps = 1.0 / (time.time() - start_time + 1e-6)

        # Output right side
        h, w, _ = dst.shape
        fps_text = f"FPS: {fps:.2f}"
        text_size = cv.getTextSize(fps_text, cv.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]

        text_x = w - text_size[0] - 10  # Right setting
        text_y = h - 10                 # Space

        cv.putText(dst, fps_text, (text_x, text_y), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)


        # Final output
        cv.namedWindow("DLIP System", cv.WINDOW_NORMAL)
        cv.setWindowProperty("DLIP System", cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN)

        cv.imshow("DLIP System", dst)

        
    

    cap.release()
    cv.destroyAllWindows()

if __name__ == '__main__':
    cam_ppe_check()