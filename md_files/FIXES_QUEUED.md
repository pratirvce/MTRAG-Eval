# Fixes Applied and Queued

**Date:** 2025-12-18

---

## ✅ Fixed: Hierarchical Routing Error

### Issue
- **Error:** "TypeError: Unable to extract query/object scores"
- **Root Cause:** Code from line 201-257 was outside the `for qid, query_text in batch_queries.items():` loop, causing variables to be undefined
- **Result:** Experiment failed during evaluation

### Fix Applied
1. **Fixed indentation** - Moved all code (lines 201-257) inside the for loop
2. **Fixed results format** - Ensured all scores are converted to floats explicitly
3. **Fixed fallback** - Properly handle case when no doc_texts are available

### Files Modified
- `train_hierarchical_routing.py` - Fixed indentation and result format

### Status
✅ **Fix applied** - Ready to re-run when needed

---

## 📋 Queued: Stop Old Experiment Versions

### Script Created
- **File:** `stop_old_experiments.sh`
- **Purpose:** Stop old experiment versions after fixed versions complete

### Old Versions to Stop
1. `best_paper_temporal_memory` (old) → `best_paper_temporal_memory_fixed` (new)
2. `best_paper_rl_adaptive_retrieval` (old) → `best_paper_rl_adaptive_retrieval_fixed` (new)

### How It Works
- Checks if fixed version has `results.json` (completed)
- If yes, stops the old version process
- If no, keeps old version running

### Usage
```bash
# Run after fixed experiments complete
./stop_old_experiments.sh
```

### Status
✅ **Script created and queued** - Run after fixed experiments complete

---

## 🎯 Next Steps

1. **Re-run Hierarchical Routing Fixed** (when ready)
   - The fix is applied, just needs to be re-run
   - Command: `venv/bin/python3 train_hierarchical_routing_tier1.py --experiment_name best_paper_hierarchical_routing_fixed --gpu 1 --output_dir experiments/retrieval/best_paper_hierarchical_routing_fixed --no-resume`

2. **Run stop_old_experiments.sh** (after fixed experiments complete)
   - Will automatically stop old versions when fixed versions have results

---

**All fixes applied and queued!** ✅

