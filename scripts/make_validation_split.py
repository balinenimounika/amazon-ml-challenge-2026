import os
import random

def make_split(train_gt_path, out_train_path, out_val_path, val_ratio=0.2, seed=42):
    random.seed(seed)
    
    with open(train_gt_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        lines = f.readlines()
        
    random.shuffle(lines)
    
    val_size = int(len(lines) * val_ratio)
    val_lines = lines[:val_size]
    train_lines = lines[val_size:]
    
    with open(out_val_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(val_lines)
        
    with open(out_train_path, 'w', encoding='utf-8') as f:
        f.write(header)
        f.writelines(train_lines)

if __name__ == "__main__":
    base_dir = "dataset/train"
    train_gt_path = os.path.join(base_dir, "train_ground_truth.tsv")
    out_train_path = os.path.join(base_dir, "dev_train_ground_truth.tsv")
    out_val_path = os.path.join(base_dir, "dev_val_ground_truth.tsv")
    
    make_split(train_gt_path, out_train_path, out_val_path)
    print(f"Split complete. Saved to {out_train_path} and {out_val_path}")
