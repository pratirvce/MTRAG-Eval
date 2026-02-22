# Data Splitting Script

The `split_data.py` script splits the MT-RAG retrieval data into train, validation, and test sets for each domain.

## Usage

### Basic Usage (Default 70/15/15 split)

```bash
python split_data.py
```

This will:
- Process all 4 domains (clapnq, fiqa, govt, cloud)
- Use default split ratios: 70% train, 15% val, 15% test
- Output to `./data_splits/` directory
- Use random seed 42 for reproducibility

### Custom Split Ratios

```bash
python split_data.py --train_ratio 0.8 --val_ratio 0.1 --test_ratio 0.1
```

### Custom Output Directory

```bash
python split_data.py --output_root ./my_splits
```

### Process Specific Domains

```bash
python split_data.py --domains clapnq fiqa
```

### Custom Random Seed

```bash
python split_data.py --seed 123
```

## Output Structure

The script creates the following directory structure:

```
data_splits/
└── retrieval_tasks/
    ├── clapnq/
    │   ├── train/
    │   │   ├── clapnq_questions.jsonl
    │   │   └── qrels/
    │   │       └── dev.tsv
    │   ├── val/
    │   │   ├── clapnq_questions.jsonl
    │   │   └── qrels/
    │   │       └── dev.tsv
    │   └── test/
    │       ├── clapnq_questions.jsonl
    │       └── qrels/
    │           └── dev.tsv
    ├── fiqa/
    │   └── ... (same structure)
    ├── govt/
    │   └── ... (same structure)
    └── cloud/
        └── ... (same structure)
```

## Features

- **Maintains BEIR format**: Output files are in the same format as input files
- **No data leakage**: Splits are done by query ID, ensuring queries don't appear in multiple splits
- **Reproducible**: Uses random seed for consistent splits across runs
- **Preserves relationships**: Maintains the relationship between queries and their qrels
- **Logging**: Provides detailed logging of the splitting process

## Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--data_root` | `.` | Root directory containing the data |
| `--output_root` | `./data_splits` | Root directory for output splits |
| `--train_ratio` | `0.7` | Ratio for training set |
| `--val_ratio` | `0.15` | Ratio for validation set |
| `--test_ratio` | `0.15` | Ratio for test set |
| `--seed` | `42` | Random seed for reproducibility |
| `--domains` | `clapnq fiqa govt cloud` | Domains to process |

## Example Output

```
2025-11-19 21:23:35 - INFO - Splitting data for domain: clapnq
2025-11-19 21:23:35 - INFO - Loaded 208 queries
2025-11-19 21:23:35 - INFO - Loaded qrels for 208 queries
2025-11-19 21:23:35 - INFO - Found 208 queries with both query text and qrels
2025-11-19 21:23:35 - INFO - Split sizes: Train=145, Val=31, Test=32
```

## Notes

- The script only processes queries that have both query text and qrels
- Split ratios must sum to 1.0
- The test set gets any remainder after rounding, ensuring all queries are used
- The original data files are not modified

