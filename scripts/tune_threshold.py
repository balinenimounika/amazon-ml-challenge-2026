import os
import argparse
import pandas as pd
import numpy as np

def compute_f05_score(true_matches_str, pred_matches_str):
    # Handle NaN and empty strings
    if pd.isna(true_matches_str) or str(true_matches_str).strip() == '':
        true_set = set()
    else:
        true_set = set(str(true_matches_str).split(','))
        
    if pd.isna(pred_matches_str) or str(pred_matches_str).strip() == '':
        pred_set = set()
    else:
        pred_set = set(str(pred_matches_str).split(','))
        
    if not true_set:
        return 1.0 if not pred_set else 0.0
        
    if not pred_set:
        return 0.0
        
    tp = len(true_set.intersection(pred_set))
    if tp == 0:
        return 0.0
        
    precision = tp / len(pred_set)
    recall = tp / len(true_set)
    
    return (1.25 * precision * recall) / ((0.25 * precision) + recall)

def tune_threshold(gt_path, preds_path):
    print(f"Loading ground truth from {gt_path}...")
    # Read ground truth
    df_gt = pd.read_csv(gt_path, sep='\t', dtype=str)
    # Ensure matched_entity_ids exists and NaNs are empty strings
    if 'matched_entity_ids' not in df_gt.columns:
        df_gt['matched_entity_ids'] = ''
    df_gt['matched_entity_ids'] = df_gt['matched_entity_ids'].fillna('')
    
    print(f"Loading predictions from {preds_path}...")
    # Read predictions
    df_preds = pd.read_csv(preds_path, sep='\t')
    # Expected columns: source1_entity_id, candidate_entity_id, match_probability
    
    thresholds = np.arange(0.50, 0.96, 0.05)
    best_thresh = None
    best_score = -1
    
    print("\nSweeping thresholds:")
    print("-" * 40)
    print("Threshold | Macro F0.5 Score")
    print("-" * 40)
    
    for thresh in thresholds:
        # Filter predictions
        mask = df_preds['match_probability'] >= thresh
        df_filtered = df_preds[mask]
        
        # Group by S1 ID
        grouped = df_filtered.groupby('source1_entity_id')['candidate_entity_id'].apply(
            lambda x: ','.join(x.astype(str))
        ).reset_index()
        grouped.rename(columns={'candidate_entity_id': 'pred_matched_entity_ids'}, inplace=True)
        
        # Merge with ground truth to ensure ALL S1 entities are present
        df_eval = pd.merge(df_gt[['source1_entity_id', 'matched_entity_ids']], 
                           grouped, 
                           on='source1_entity_id', 
                           how='left')
        df_eval['pred_matched_entity_ids'] = df_eval['pred_matched_entity_ids'].fillna('')
        
        # Compute scores
        # We use a vectorized approach or apply
        scores = df_eval.apply(
            lambda row: compute_f05_score(row['matched_entity_ids'], row['pred_matched_entity_ids']),
            axis=1
        )
        
        macro_f05 = scores.mean()
        print(f"  {thresh:.2f}    | {macro_f05:.6f}")
        
        if macro_f05 > best_score:
            best_score = macro_f05
            best_thresh = thresh
            
    print("-" * 40)
    print(f"\nOptimal Threshold: {best_thresh:.2f}")
    print(f"Maximum F0.5 Score: {best_score:.6f}")
    print("\nUse this threshold to generate the final matching_results.tsv for validation/submission.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Tune probability threshold for matching predictions.")
    # Defaulting to the true dataset location despite the prompt's minor typo, to obey strict repo rules.
    parser.add_argument("--truth", default="dataset/train/dev_val_ground_truth.tsv", 
                        help="Path to validation ground truth TSV")
    parser.add_argument("--preds", required=False, default="outputs/mock_role2_predictions.tsv", 
                        help="Path to Role 2 output TSV with probabilities")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.truth):
        print(f"Error: Ground truth file not found at {args.truth}")
    elif not os.path.exists(args.preds):
        print(f"Note: Predictions file {args.preds} not found. Ready for when Role 2 finishes!")
    else:
        tune_threshold(args.truth, args.preds)
