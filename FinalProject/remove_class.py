import os

label_dir = './datasets/mix2/valid/labels'  # valid/test/train path
remove_ids = ['0', '3', '4', '9', '10', '11', '12', '13']   # Remove ID number 

for name in os.listdir(label_dir):
    if not name.endswith('.txt'):
        continue

    path = os.path.join(label_dir, name)
    with open(path, 'r') as f:  # Open files in reading modes
        lines = f.readlines()

    new_lines = []
    for line in lines:
        if line.strip() == "":
            continue 
        parts = line.strip().split()    # Split by space
        if parts[0] not in remove_ids:  # if class number not in remove_ids, append it to new lines
            new_lines.append(line)

    # save result
    with open(path, 'w') as f:
        f.writelines(new_lines)

