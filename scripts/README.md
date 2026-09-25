# Validation, Evaluation & Baseline Pipeline (Role 3)

Welcome to the evaluation pipeline! This directory contains all scripts related to validating, scoring, and exploring the data.

## Important: The Validation Split
Inside `dataset/train/`, you will find two files:
- `dev_train_ground_truth.tsv` (80% of labels)
- `dev_val_ground_truth.tsv` (20% of labels)

**DO NOT train on `dev_val_ground_truth.tsv`.** This is our local validation set. Training on this will ruin our ability to measure whether the model is actually generalizing to unseen data.

## Expected Workflow

The standard pipeline for making a submission should always flow like this:

1. **Blocking (Role 1)**: Generates candidates based on rules/heuristics.
   - Outputs: `outputs/candidate_pairs.tsv`
   - *Check:* Run `python scripts/check_candidate_recall.py --candidates outputs/candidate_pairs.tsv` to ensure the true matches weren't dropped.
2. **Matching (Role 2)**: Trains on candidates and generates final predicted matches.
   - Outputs: `outputs/matching_results.tsv`
   - *Check:* Run `python scripts/scorer.py --truth dataset/train/dev_val_ground_truth.tsv --pred outputs/matching_results.tsv` to get our Local F0.5 Score.
3. **Submission (Role 3)**: Validates and pushes to the leaderboard.
   - *Check:* Run `python utils/validate_submission.py --matching outputs/matching_results.tsv --test-dir dataset/test`
   - After receiving a score on the leaderboard, **update `outputs/submission_tracker.md`**.

## Script Quick Reference

| Script | Purpose | Example Command |
| :--- | :--- | :--- |
| `make_validation_split.py` | (Run once) Creates the 80/20 train/val split. | `python scripts/make_validation_split.py` |
| `scorer.py` | Computes our official F0.5 score locally against the validation set. | `python scripts/scorer.py --truth dataset/train/dev_val_ground_truth.tsv --pred outputs/my_predictions.tsv` |
| `check_candidate_recall.py` | Measures what % of true matches were successfully found by the blocking stage. | `python scripts/check_candidate_recall.py --candidates outputs/my_candidates.tsv` |
| `check_singletons.py` | Verifies the exact percentage of "no-match" entities in our validation split. | `python scripts/check_singletons.py` |
| `explore_singletons.py` | Prints 20 random true "no-match" entities for visual inspection. | `python scripts/explore_singletons.py` |
| `baseline_exact_match.py` | A simple exact-match baseline (Name + Country) on a 2000-row sample to test the pipeline. | `python scripts/baseline_exact_match.py` |
| `explore_test_set.py` | Read-only analysis script exploring country distributions and edge cases in the test set. | `python scripts/explore_test_set.py` |

## Repository Hygiene Rules
- **Python scripts** only go in `scripts/`.
- **Output TSVs/Logs** only go in `outputs/`.
- **DO NOT** create loose files in the repo root.
- **DO NOT** write or modify any files in `dataset/`. Treat raw data as strictly read-only!
