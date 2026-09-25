import os
import subprocess
import sys

def create_dummy_baseline(val_gt_path, dummy_pred_path):
    with open(val_gt_path, 'r', encoding='utf-8') as f_in, open(dummy_pred_path, 'w', encoding='utf-8') as f_out:
        header = f_in.readline()
        f_out.write(header)
        for line in f_in:
            parts = line.strip('\n').split('\t')
            if not parts or parts == ['']:
                continue
            s1_id = parts[0]
            # Predict empty matches for everyone
            f_out.write(f"{s1_id}\t\n")

if __name__ == "__main__":
    val_gt_path = "dataset/train/dev_val_ground_truth.tsv"
    dummy_pred_path = "dataset/train/dummy_predictions.tsv"
    
    if not os.path.exists(val_gt_path):
        print(f"Error: {val_gt_path} not found. Did you run make_validation_split.py first?")
        sys.exit(1)
        
    print(f"Creating dummy baseline predicting empty matches for all entities...")
    create_dummy_baseline(val_gt_path, dummy_pred_path)
    print(f"Dummy baseline created at {dummy_pred_path}")
    
    print("\nRunning scorer.py on dummy baseline...")
    subprocess.run([sys.executable, "scorer.py", "--truth", val_gt_path, "--pred", dummy_pred_path])
