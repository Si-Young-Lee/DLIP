import os

label_dir = './datasets/mix2/valid/labels'  # path(valid/test/train)
merge_ids = ['5', '6']    # Class IDs to merge
target_id = '5'           # Merged class ID

for name in os.listdir(label_dir):
    if not name.endswith('.txt'):  # if not txt file, ignore
        continue
    path = os.path.join(label_dir, name)
    with open(path, 'r') as f:  # Open files in reading modes
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()    # Split by space
        if parts[0] in merge_ids:
            parts[0] = target_id
        new_lines.append(' '.join(parts))
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))