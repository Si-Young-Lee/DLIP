import os
import shutil

# 원본 데이터셋 폴더 리스트
dataset_dirs = [
    "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/Gloves_data",
    "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/mask_data",
    "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/mix"
]

# 병합 대상 경로
merged_root = "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/merged"

# split 목록
splits = ['train', 'valid', 'test']

# 병합 폴더 생성 (images/split, labels/split)
for split in splits:
    os.makedirs(os.path.join(merged_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(merged_root, 'labels', split), exist_ok=True)

# split별 병합 수행
for split in splits:
    img_idx = 0  # 파일 이름 중복 방지를 위한 번호
    for dataset in dataset_dirs:
        img_dir = os.path.join(dataset, split, "images")
        lbl_dir = os.path.join(dataset, split, "labels")

        if not os.path.exists(img_dir):
            continue  # 없는 경우 생략

        for fname in os.listdir(img_dir):
            name, ext = os.path.splitext(fname)
            src_img = os.path.join(img_dir, fname)
            src_lbl = os.path.join(lbl_dir, name + ".txt")

            dst_img = os.path.join(merged_root, "images", split, f"{split}_{img_idx:05d}{ext}")
            dst_lbl = os.path.join(merged_root, "labels", split, f"{split}_{img_idx:05d}.txt")

            shutil.copyfile(src_img, dst_img)
            shutil.copyfile(src_lbl, dst_lbl)
            img_idx += 1

    print(f"[{split}] 병합 완료 → 총 {img_idx}개 이미지 복사됨")

print("\n✅ 모든 train/valid/test 병합 완료! data.yaml 그대로 사용 가능")

