# Critical Bug Fix: Evaluation Using Base Model Instead of Trained Model

**Date:** 2025-12-20  
**Issue:** All experiments scoring exactly 0.1796 nDCG@10 were evaluating with the base model instead of the trained model.

---

## Root Cause

All affected experiments had the same bug in their evaluation code:

```python
# WRONG - Using base model for evaluation
retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
```

This meant that even though the models were being trained correctly, the evaluation was using the original untrained base model, resulting in identical baseline scores (0.1796 nDCG@10, 0.2420 Recall@10).

---

## Fixed Experiments

The following 10 experiments have been fixed:

1. ✅ `tier1_curriculum_contrastive`
2. ✅ `tier1_causal_inference`
3. ✅ `tier1_uncertainty_aware`
4. ✅ `tier1_mixture_experts`
5. ✅ `tier1_multi_turn_state_tracking`
6. ✅ `tier1_learned_indices`
7. ✅ `tier1_differentiable_retrieval`
8. ✅ `tier1_foundation_distillation`
9. ✅ `tier1_graph_enhanced_reranking`
10. ✅ `best_paper_rl_adaptive_retrieval_fixed` (uses same code)

---

## Fix Applied

For each experiment, the following changes were made:

### 1. Save Trained Model After Training

Added model saving after each domain's training:

```python
# Save trained model after each domain
trained_model_path = checkpoint_dir / f"trained_model_{domain}"
model.save(str(trained_model_path))
logging.info(f"Saved trained model for {domain} to {trained_model_path}")
```

### 2. Load Trained Model for Evaluation

Changed evaluation to use the trained model:

```python
# Evaluation using the trained model
logging.info("\nEvaluating trained model...")
# Use the trained model from the last domain
if len(domains) > 0:
    last_domain = list(domains)[-1] if isinstance(domains, (list, tuple)) else domains[-1]
    trained_model_path = checkpoint_dir / f"trained_model_{last_domain}"
    if trained_model_path.exists():
        logging.info(f"Loading trained model from {trained_model_path}")
        retriever = DenseRetrievalExactSearch(SentenceBERT(str(trained_model_path), device=device), batch_size=128)
    else:
        logging.warning(f"Trained model not found at {trained_model_path}, using base model")
        retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
else:
    logging.warning("No domains to evaluate, using base model")
    retriever = DenseRetrievalExactSearch(SentenceBERT(base_model, device=device), batch_size=128)
```

---

## Files Modified

1. `train_curriculum_contrastive.py`
2. `train_causal_inference.py`
3. `train_uncertainty_aware.py`
4. `train_mixture_experts.py`
5. `train_state_tracking.py`
6. `train_learned_indices.py`
7. `train_differentiable_retrieval.py`
8. `train_foundation_distillation.py`
9. `train_graph_enhanced_reranking.py`

---

## Next Steps

1. **Re-run all fixed experiments** with `--no-resume` flag to ensure fresh training
2. **Verify training logs** show loss decreasing
3. **Check that trained models are saved** in experiment directories
4. **Compare new scores** with baseline (0.1796) to confirm improvement

---

## Expected Impact

After re-running these experiments, we expect:

- **Significant score improvements** (from 0.1796 to potentially 0.30-0.45 nDCG@10)
- **Different scores** for each experiment (they were all identical before)
- **Proper evaluation** of the trained models' performance

---

## Verification Commands

To verify the fix worked:

```bash
# Check if trained models are being saved
ls -la experiments/retrieval/tier1_curriculum_contrastive/trained_model_*

# Re-run an experiment with --no-resume
python3 train_curriculum_contrastive_tier1.py \
    --experiment_name tier1_curriculum_contrastive_fixed \
    --gpu 0 \
    --output_dir experiments/retrieval \
    --no-resume

# Check training logs for model saving
tail -50 experiments/retrieval/tier1_curriculum_contrastive_fixed/training.log | grep "Saved trained model"
```

---

## Notes

- The fix uses the last domain's trained model for evaluation across all domains
- For better results, consider using domain-specific models for each domain's evaluation
- Model saving happens after each domain's training completes
- If a trained model is not found, the code falls back to the base model with a warning

