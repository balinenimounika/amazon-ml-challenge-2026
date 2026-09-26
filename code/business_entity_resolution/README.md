# Business Entity Resolution Pipeline

## Overview
This package contains the high-recall blocking and candidate generation engine for the Amazon ML Challenge 2026.

## Setup
Install dependencies:
```bash
pip install -r requirements.txt
```

## Running the Blocking Pipeline
To run candidate generation on the test dataset:
```bash
python src/blocking_pipeline.py --data-dir student_resource/dataset/test
```

To run the empirical recall benchmark on training ground truth:
```bash
python src/blocking_pipeline.py --benchmark
```

## Output Artifacts
- `candidate_pairs.tsv`: Official competition format (`source1_entity_id`, `candidate_entity_ids`)
- `candidate_pairs_pairwise.tsv`: Pairwise format (`source1_id`, `source2_id`)
