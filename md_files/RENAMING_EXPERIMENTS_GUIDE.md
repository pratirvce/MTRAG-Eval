# Guide to Renaming Experiments

## Overview

This guide explains how to safely rename experiments from names like `phase1_baseline` or `tier1_contrastive_learning` to more standard journal-style names like `baseline_bge_finetuned` or `contrastive_learning_finetuning`.

## Impact Analysis

### ✅ What WILL be affected:

1. **Directory Names**: `experiments/retrieval/{old_name}/` → `experiments/retrieval/{new_name}/`
2. **Resume Functionality**: `stopped_experiments_resume_info.json` uses experiment names as keys
3. **Status Tracking**: `experiment_status.json` uses experiment names as keys
4. **Config Files**: `config.json` files may contain experiment names
5. **Path References**: JSON files contain paths that include experiment names

### ✅ What will NOT be affected:

1. **Results Files**: `results.json` files don't contain experiment names - they're safe!
2. **Model Checkpoints**: Checkpoints don't depend on experiment names
3. **Training Logs**: Logs are just text files - renaming directory doesn't affect them
4. **Actual Experiment Data**: All data, models, and results remain intact

## Safe Renaming Process

### Step 1: Preview Changes (Dry Run)

First, see what would be renamed without actually doing it:

```bash
python3 rename_experiments.py --dry-run
```

This will show:
- Which directories would be renamed
- Which JSON files would be updated
- Which config files would be updated
- A summary of changes

### Step 2: Rename All Experiments

If the dry run looks good, rename everything:

```bash
python3 rename_experiments.py
```

This will:
1. Rename all experiment directories
2. Update all JSON tracking files
3. Update config.json files in each experiment directory
4. Create `experiment_name_mapping.json` for reference

### Step 3: Rename a Single Experiment

To rename just one experiment:

```bash
python3 rename_experiments.py --experiment "phase1_baseline:baseline_bge_finetuned"
```

### Step 4: Custom Mapping File

Create your own mapping file `my_mapping.json`:

```json
{
  "phase1_baseline": "baseline_bge_finetuned",
  "tier1_contrastive_learning": "contrastive_learning_finetuning"
}
```

Then use it:

```bash
python3 rename_experiments.py --mapping-file my_mapping.json
```

## Resuming After Renaming

**Good News**: After renaming, resume functionality will work correctly because:

1. The rename script updates `stopped_experiments_resume_info.json` with new names
2. The rename script updates all path references in JSON files
3. The `resume_from_checkpoint.py` script will use the new names

To resume experiments after renaming:

```bash
python3 resume_from_checkpoint.py --list  # Check what's available
python3 resume_from_checkpoint.py        # Resume all
```

## What Gets Updated

The rename script automatically updates:

1. **Experiment Directories**: 
   - `experiments/retrieval/phase1_baseline/` → `experiments/retrieval/baseline_bge_finetuned/`

2. **Status JSON Files**:
   - `experiment_status.json`
   - `stopped_experiments_resume_info.json`
   - `fixed_experiments_status.json`
   - `tier1_experiments_status.json`
   - `auto_experiments_status.json`
   - `auto_fixed_experiments_status.json`

3. **Config Files**:
   - `experiments/retrieval/{new_name}/config.json`

4. **Path References**: All paths in JSON files that reference experiment names

## Example Mapping

The script includes pre-defined mappings for common patterns:

| Old Name | New Name |
|----------|----------|
| `phase1_baseline` | `baseline_bge_finetuned` |
| `phase4_domain_specific_clapnq` | `domain_specific_finetuning_clapnq` |
| `tier1_contrastive_learning` | `contrastive_learning_finetuning` |
| `best_paper_large_model_finetuning` | `large_model_finetuning` |
| `phase5_query_expansion_govt` | `query_expansion_govt` |

## Verification

After renaming, verify everything worked:

1. **Check directories**:
   ```bash
   ls experiments/retrieval/ | head -20
   ```

2. **Check resume info**:
   ```bash
   python3 resume_from_checkpoint.py --list
   ```

3. **Check status file**:
   ```bash
   python3 -c "import json; print(json.load(open('experiment_status.json')).keys())"
   ```

## Rollback

If you need to rollback, the script creates `experiment_name_mapping.json` with the reverse mapping:

```bash
python3 rename_experiments.py --mapping-file experiment_name_mapping.json
```

(You may need to reverse the mapping manually if needed)

## Important Notes

1. **Results are Safe**: `results.json` files don't contain experiment names, so your scores are safe!
2. **Backup First**: Consider backing up your `experiments/retrieval/` directory before bulk renaming
3. **Stop Running Experiments**: Make sure no experiments are running during rename
4. **Check After Rename**: Verify the rename worked by checking a few experiments manually

## Troubleshooting

### "Directory already exists" error
- The new name is already taken
- Either delete the conflicting directory or choose a different name

### Resume not working after rename
- Check `stopped_experiments_resume_info.json` was updated
- Verify the experiment directory was renamed correctly
- Try running the rename script again

### Missing experiments in status file
- Some experiments might not exist yet
- Check which ones exist: `ls experiments/retrieval/`
- The script only renames existing directories

## Questions?

- Results files are NOT affected by renaming
- Resume functionality WILL work after renaming (if you use the script)
- Starting new experiments with new names is fine
- The script handles all the JSON updates automatically

