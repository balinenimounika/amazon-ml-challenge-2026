import sys
import pandas as pd

def check_singletons(val_gt_path):
    print(f"Reading {val_gt_path}...")
    
    # We will use standard file reading to be robust against NaN interpretations
    total_rows = 0
    singleton_count = 0
    
    with open(val_gt_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            line = line.strip('\n')
            if not line:
                continue
            
            parts = line.split('\t')
            total_rows += 1
            
            # Singleton criteria:
            # - Missing column entirely (len < 2)
            # - Column exists but is empty after strip
            # - Column exists and equals literal "NaN" or "nan"
            if len(parts) < 2:
                singleton_count += 1
            else:
                matched_val = parts[1].strip()
                if matched_val == '' or matched_val.lower() == 'nan':
                    singleton_count += 1
                    
    if total_rows == 0:
        print("No rows found!")
        return
        
    singleton_percentage = (singleton_count / total_rows) * 100
    
    print("-" * 30)
    print(f"Total Rows:      {total_rows}")
    print(f"Singleton Count: {singleton_count}")
    print(f"Singleton %:     {singleton_percentage:.2f}%")
    print("-" * 30)
    
if __name__ == "__main__":
    val_gt_path = "dataset/train/dev_val_ground_truth.tsv"
    check_singletons(val_gt_path)
