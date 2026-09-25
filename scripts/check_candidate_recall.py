import sys
import argparse

def load_ground_truth(file_path):
    """Load ground truth into a dictionary {s1_id: set(matched_ids)}."""
    true_labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if not parts or parts == ['']:
                continue
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                matched_val = parts[1].strip()
                if matched_val.lower() == 'nan':
                    true_labels[s1_id] = set()
                else:
                    true_labels[s1_id] = set(matched_val.split(','))
            else:
                true_labels[s1_id] = set()
    return true_labels

def check_candidate_recall(val_gt_path, candidates_path):
    print(f"Loading validation ground truth from {val_gt_path}...")
    true_labels = load_ground_truth(val_gt_path)
    total_entities = len(true_labels)
    
    if total_entities == 0:
        print("No validation entities found.")
        return
        
    print(f"Loading candidates from {candidates_path}...")
    candidates = {}
    with open(candidates_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if not parts or parts == ['']:
                continue
            s1_id = parts[0]
            # Only store candidates for entities in our validation set to save memory
            if s1_id in true_labels:
                if len(parts) > 1 and parts[1].strip():
                    cand_val = parts[1].strip()
                    if cand_val.lower() != 'nan':
                        candidates[s1_id] = set(cand_val.split(','))
                    else:
                        candidates[s1_id] = set()
                else:
                    candidates[s1_id] = set()
                    
    print("\nCalculating recall metrics...")
    
    total_recall = 0.0
    total_candidates = 0
    zero_recall_count = 0
    
    for s1_id, true_set in true_labels.items():
        cand_set = candidates.get(s1_id, set())
        total_candidates += len(cand_set)
        
        if not true_set:
            # Singleton entity (no true matches) -> always 1.0 recall
            recall = 1.0
        else:
            intersection = true_set.intersection(cand_set)
            recall = len(intersection) / len(true_set)
            
        if recall == 0.0:
            zero_recall_count += 1
            
        total_recall += recall
        
    macro_recall = total_recall / total_entities
    avg_candidates = total_candidates / total_entities
    zero_recall_pct = (zero_recall_count / total_entities) * 100
    
    print("-" * 50)
    print(f"Total Validation Entities: {total_entities}")
    print(f"Macro-Average Recall:      {macro_recall:.4f} ({macro_recall * 100:.2f}%)")
    print(f"Avg Candidates/Entity:     {avg_candidates:.2f}")
    print(f"Entities with 0% Recall:   {zero_recall_count} ({zero_recall_pct:.2f}%)")
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Check Blocking Candidate Recall")
    parser.add_argument("--candidates", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--truth", default="dataset/train/dev_val_ground_truth.tsv", 
                        help="Path to validation ground truth TSV")
    
    args = parser.parse_args()
    check_candidate_recall(args.truth, args.candidates)
