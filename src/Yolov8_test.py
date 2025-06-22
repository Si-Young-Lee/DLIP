from ultralytics import YOLO
import cv2
import numpy as np

# IoU 계산 함수
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

# obj_box가 person_box와 얼마나 겹치는지를 계산
def is_overlapping(obj_box, person_box, threshold=0.3):
    xA = max(obj_box[0], person_box[0])
    yA = max(obj_box[1], person_box[1])
    xB = min(obj_box[2], person_box[2])
    yB = min(obj_box[3], person_box[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    objArea = (obj_box[2] - obj_box[0]) * (obj_box[3] - obj_box[1])

    return (interArea / (objArea + 1e-6)) > threshold

def test():
    # 모델 로드
    model = YOLO('runs/detect/train8/weights/best.pt')

    # 테스트 이미지 로드
    src = cv2.imread("datasets/mix2/test/images/016_0349_005714_crop_margin_0-1_jpg.rf.266404f551ad2d99fdb2f6470168d5df.jpg")

    # 클래스 번호 (data.yaml 순서 기준)
    CLASS_GLOVE = 0
    CLASS_MASK = 1
    CLASS_COAT = 2
    CLASS_PERSON = 3

    results = model.predict(source=src, save=False)

    for r in results:
        dst = r.plot()
        cv2.imshow("result plot", dst)

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

        for i, person_box in enumerate(person_boxes):
            px1, py1, px2, py2 = person_box
            gloves = 0
            masks = 0
            coats = 0

            for cls_id, obj_box in other_objects:
                iou = compute_iou(person_box, obj_box)
                overlap = is_overlapping(obj_box, person_box, threshold=0.3)

                if iou > 0.1 or overlap:
                    if cls_id == CLASS_GLOVE:
                        gloves += 1
                    elif cls_id == CLASS_MASK:
                        masks += 1
                    elif cls_id == CLASS_COAT:
                        coats += 1

            # PPE 누락 체크
            missing_items = []
            if gloves < 2:
                missing_items.append(f"Gloves({gloves}/2)")
            if masks < 1:
                missing_items.append(f"Mask({masks}/1)")
            if coats < 1:
                missing_items.append(f"Coat({coats}/1)")

            if missing_items:
                warning_msg = f"Person {i+1}: PPE Missing ({', '.join(missing_items)})"
                print(warning_msg)
                cv2.rectangle(dst, (int(px1), int(py1)), (int(px2), int(py2)), (0, 0, 255), 2)
                text_x = int(px1)
                text_y = max(30, int(py1) - 10)  # 최소 40픽셀은 확보
                cv2.putText(dst, warning_msg, (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        dst = cv2.resize(dst, (500, 700))           
        cv2.imshow("PPE Check", dst)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    test()
