import sys
import argparse

def compute_f05(true_set, pred_set):
    if not true_set:
        return 1.0 if not pred_set else 0.0
    
    if not pred_set:
        return 0.0
        
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0
        
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    f05 = (1.25 * precision * recall) / (0.25 * precision + recall)
    return f05

def load_labels(file_path):
    labels = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if not parts or parts == ['']:
                continue
            s1_id = parts[0]
            if len(parts) > 1 and parts[1].strip():
                matched = set(parts[1].split(','))
            else:
                matched = set()
            labels[s1_id] = matched
    return labels

def score(truth_path, pred_path):
    print(f"Loading ground truth from {truth_path}...")
    true_labels = load_labels(truth_path)
    
    print(f"Scoring predictions from {pred_path}...")
    total_score = 0.0
    count = 0
    missing_preds = 0
    
    with open(pred_path, 'r', encoding='utf-8') as f:
        header = f.readline()
        for line in f:
            parts = line.strip('\n').split('\t')
            if not parts or parts == ['']:
                continue
            s1_id = parts[0]
            if s1_id not in true_labels:
                continue # ignore predictions for things not in ground truth
            
            if len(parts) > 1 and parts[1].strip():
                pred_set = set(parts[1].split(','))
            else:
                pred_set = set()
                
            f05 = compute_f05(true_labels[s1_id], pred_set)
            total_score += f05
            count += 1
            
            # mark as seen by removing from true_labels
            del true_labels[s1_id]
            
    # For any remaining true_labels that were not in predictions, they count as empty predictions
    for s1_id, true_set in true_labels.items():
        missing_preds += 1
        f05 = compute_f05(true_set, set())
        total_score += f05
        count += 1
        
    if count == 0:
        print("No matching entities found to score.")
        return 0.0
        
    macro_f05 = total_score / count
    print(f"Scored {count} entities (Missing predictions for {missing_preds} entities)")
    print(f"Macro F0.5 Score: {macro_f05:.6f}")
    return macro_f05

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute Macro F0.5 Score")
    parser.add_argument("--truth", required=True, help="Path to ground truth TSV")
    parser.add_argument("--pred", required=True, help="Path to predictions TSV")
    
    args = parser.parse_args()
    score(args.truth, args.pred)
