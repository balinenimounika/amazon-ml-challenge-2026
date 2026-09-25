import os
import string
import subprocess
import sys

def normalize(text):
    if not text:
        return ""
    text = text.lower().strip()
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

def build_index(filepath):
    """Build a lookup dictionary matching (normalized_name, country) to entity IDs."""
    index = {}
    with open(filepath, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            # Expecting at least entity_id, business_name, business_address, country
            if len(parts) >= 4:
                eid = parts[0]
                name = normalize(parts[1])
                country = parts[3].strip()
                if name: # skip entirely empty names
                    key = (name, country)
                    if key not in index:
                        index[key] = set()
                    index[key].add(eid)
    return index

def main():
    s1_path = os.path.join('dataset', 'train', 'train_source1.tsv')
    s2_path = os.path.join('dataset', 'train', 'train_source2.tsv')
    s3_path = os.path.join('dataset', 'train', 'train_source3.tsv')
    out_path = os.path.join('outputs', 'baseline_matching_results.tsv')
    
    print("Building indices for Source 2 and Source 3...")
    idx2 = build_index(s2_path)
    idx3 = build_index(s3_path)
    
    print("Processing first 2000 rows of Source 1...")
    results = {}
    
    with open(s1_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        count = 0
        for line in f:
            if count >= 2000:
                break
            parts = line.strip('\n').split('\t')
            if len(parts) >= 4:
                s1_id = parts[0]
                name = normalize(parts[1])
                country = parts[3].strip()
                
                matched = set()
                if name:
                    key = (name, country)
                    if key in idx2:
                        matched.update(idx2[key])
                    if key in idx3:
                        matched.update(idx3[key])
                    
                results[s1_id] = matched
            count += 1
            
    print(f"Writing results to {out_path}...")
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id, matches in results.items():
            f.write(f"{s1_id}\t{','.join(sorted(matches))}\n")
            
    print("\n" + "="*50)
    print("RUNNING POST-GENERATION CHECKS")
    print("="*50 + "\n")
    
    # 1. Recall check
    print("--- 1. check_candidate_recall.py ---")
    subprocess.run([sys.executable, os.path.join('scripts', 'check_candidate_recall.py'), '--candidates', out_path])
    
    # 2. Scorer
    print("\n--- 2. scorer.py ---")
    subprocess.run([sys.executable, os.path.join('scripts', 'scorer.py'), 
                    '--truth', os.path.join('dataset', 'train', 'dev_val_ground_truth.tsv'), 
                    '--pred', out_path])
    
    # 3. Validator
    print("\n--- 3. validate_submission.py ---")
    subprocess.run([sys.executable, os.path.join('utils', 'validate_submission.py'), 
                    '--matching', out_path, 
                    '--test-dir', os.path.join('dataset', 'test')])

if __name__ == "__main__":
    main()
