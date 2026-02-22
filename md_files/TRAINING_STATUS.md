# Training and Evaluation Status

## Completed Steps

### 1. ✅ Data Splitting
- Created `split_data.py` script to split data into train/val/test sets
- Successfully split all 4 domains (clapnq, fiqa, govt, cloud) with 70/15/15 ratio
- Output location: `data_splits/retrieval_tasks/{domain}/{split}/`

**Split Summary:**
- **Total queries:** 777
- **Train:** 541 queries (70%)
- **Val:** 116 queries (15%)
- **Test:** 120 queries (15%)

### 2. ✅ Script Updates
- Updated `train_finetuned_bge.py` to use train split from `data_splits/`
- Updated `evaluate_finetuned_bge.py` to use test split from `data_splits/`
- Both scripts fall back to original data location if splits don't exist

### 3. ✅ Data Preparation
- Extracted all corpora files (clapnq, fiqa, govt, cloud) from zip files
- All corpus files are ready in `corpora/passage_level/`

### 4. 🔄 Training in Progress
- **Status:** Training started
- **Model:** BAAI/bge-base-en-v1.5
- **Training examples:** 1,467 (from train split)
- **Epochs:** 1
- **Batch size:** 16
- **Output:** `./bge-finetuned-all-domains`
- **Log file:** `training.log`

**Training Data Breakdown:**
- clapnq: 401 examples
- fiqa: 373 examples
- govt: 352 examples
- cloud: 339 examples
- custom: 2 examples (placeholder)

### 5. ⏳ Evaluation (Pending)
- Evaluation script is ready and will use test split
- Will run after training completes
- Will evaluate on all 4 domains using test set (120 queries)

## Next Steps

1. **Monitor Training:**
   ```bash
   tail -f training.log
   ```

2. **After Training Completes:**
   ```bash
   python evaluate_finetuned_bge.py
   ```

3. **Check Training Results:**
   - Model will be saved to: `./bge-finetuned-all-domains/`
   - Checkpoints: `./bge-finetuned-all-domains-checkpoints/`

## Files Created/Modified

- ✅ `split_data.py` - Data splitting script
- ✅ `SPLIT_DATA_README.md` - Documentation for splitting
- ✅ `train_finetuned_bge.py` - Updated to use train split
- ✅ `evaluate_finetuned_bge.py` - Updated to use test split
- ✅ `data_splits/` - Directory with train/val/test splits
- 🔄 `training.log` - Training progress log (updating)
- 🔄 `bge-finetuned-all-domains/` - Model output (in progress)

## Notes

- Training is running in the background
- GPU is being used (CUDA available)
- Training may take 10-30 minutes depending on GPU
- Evaluation will automatically use the test split when ready

