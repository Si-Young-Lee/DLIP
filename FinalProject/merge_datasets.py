import os
import shutil

# original folder list
dataset_dirs = [
    "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/Gloves_data",
    "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/mask_data",
    "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/mix"
]

# merge directory
merged_root = "C:/Users/USER/source/repos/DLIP/LAB/DLIP_Final_Project/datasets/merged"

# split 
splits = ['train', 'valid', 'test']

# Make Merge folder (images/split, labels/split)
for split in splits:
    os.makedirs(os.path.join(merged_root, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(merged_root, 'labels', split), exist_ok=True)

# Merging
for split in splits:
    img_idx = 0  
    for dataset in dataset_dirs:
        img_dir = os.path.join(dataset, split, "images")
        lbl_dir = os.path.join(dataset, split, "labels")

        if not os.path.exists(img_dir):
            continue  

        for fname in os.listdir(img_dir):
            name, ext = os.path.splitext(fname)
            src_img = os.path.join(img_dir, fname)
            src_lbl = os.path.join(lbl_dir, name + ".txt")

            dst_img = os.path.join(merged_root, "images", split, f"{split}_{img_idx:05d}{ext}")
            dst_lbl = os.path.join(merged_root, "labels", split, f"{split}_{img_idx:05d}.txt")

            shutil.copyfile(src_img, dst_img)
            shutil.copyfile(src_lbl, dst_lbl)
            img_idx += 1

    print(f"[{split}] merging Complete →  {img_idx} images")

print("\n✅ All train/valid/test merging Complete! data.yaml")

