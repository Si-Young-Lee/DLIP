import os

# 1. Path
label_dir = './datasets/mix2/valid/labels'

# 2. remapping class: exist ID → new ID
remap_dict = {
    '5': '2',
}

# 3. 파일 반복
for name in os.listdir(label_dir):
    if not name.endswith('.txt'): # Open files in reading modes
        continue

    path = os.path.join(label_dir, name)
    with open(path, 'r') as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        # Remapping class if class in remap_dict
        parts[0] = remap_dict.get(parts[0], parts[0])
        new_lines.append(' '.join(parts))

    # 4. overwrite
    with open(path, 'w') as f:
        f.write('\n'.join(new_lines))

