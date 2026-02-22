# Failed Experiments - Needs Fixes

**Generated:** 2025-12-19  
**Total Failed Experiments:** 16

## Summary Table

| Experiment Name | Error Type | Status | Priority | Fix Needed |
|----------------|------------|--------|----------|------------|
| best_paper_adversarial_curriculum | Repo id validation error | ❌ Failed | High | Fix model name/save path |
| best_paper_hierarchical_routing | Unable to extract query/object scores | ❌ Failed | High | Already fixed (see _fixed version) |
| best_paper_llm_distillation | Repo id validation error | ❌ Failed | Medium | Fix model name/save path |
| best_paper_rl_adaptive_retrieval | Index out of bounds | ❌ Failed | High | Fix indexing bug |
| phase2_cosine_loss | Training configuration error | ❌ Failed | Low | Fix training setup |
| tier1_adversarial_curriculum | Import error (beir) | ❌ Failed | Medium | Fix imports |
| tier1_cross_attention_query_document | Device/model loading error | ❌ Failed | Medium | Fix device handling |
| tier1_cross_attention_query_document_fixed | ModuleNotFoundError: beir | ❌ Failed | High | Install beir or fix imports |
| tier1_cross_encoder_large | Device/model loading error | ❌ Failed | Medium | Fix device handling |
| tier1_enhanced_contrastive_hardnegatives | 'last_hidden_state' KeyError | ❌ Failed | High | Fixed in _fixed version |
| tier1_enhanced_contrastive_hardnegatives_fixed | Tensor size mismatch | ❌ Failed | High | Fix tensor alignment |
| tier1_graph_aware_retrieval | ModuleNotFoundError: beir | ❌ Failed | Medium | Install beir or fix imports |
| tier1_learning_to_rank_listwise | Evaluation error | ❌ Failed | High | Fixed in _fixed version |
| tier1_learning_to_rank_listwise_fixed | Evaluator retriever not set | ❌ Failed | High | Fix evaluator initialization |
| tier1_multitask_retrieval | Import error (beir) | ❌ Failed | Medium | Fix imports |
| tier1_pseudo_relevance_feedback | ModuleNotFoundError: beir | ❌ Failed | Medium | Install beir or fix imports |

---

## Detailed Error Information

### 1. best_paper_adversarial_curriculum
- **Script**: `train_adversarial_curriculum_tier1.py`
- **Error**: `Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96: 'sentence-transformers/SentenceTransformer(...)'`
- **Root Cause**: Model save path or name contains invalid characters or is too long
- **Fix Needed**: 
  - Check model save path in script
  - Ensure model name is valid (alphanumeric, '-', '_', '.')
  - Limit model name to 96 characters

### 2. best_paper_hierarchical_routing
- **Script**: `train_hierarchical_routing_tier1.py`
- **Error**: `Unable to extract query/object scores.`
- **Root Cause**: Evaluation function has issues extracting scores
- **Status**: ✅ **FIXED** - See `best_paper_hierarchical_routing_fixed` (indentation bug fixed)
- **Fix Applied**: Fixed indentation in `run_hierarchical_routing_evaluation`

### 3. best_paper_llm_distillation
- **Script**: `train_llm_distillation_tier1.py`
- **Error**: `Repo id must use alphanumeric chars, '-', '_' or '.'. The name cannot start or end with '-' or '.' and the maximum length is 96`
- **Root Cause**: Model save path or name contains invalid characters
- **Fix Needed**: Validate and fix model save path/name

### 4. best_paper_rl_adaptive_retrieval
- **Script**: `train_rl_adaptive_tier1.py`
- **Error**: `index 1 is out of bounds for dimension 0 with size 1`
- **Root Cause**: Array/tensor indexing error - trying to access index 1 when array only has 1 element
- **Fix Needed**: 
  - Check array/tensor dimensions before indexing
  - Add bounds checking
  - Fix logic that assumes multiple elements

### 5. phase2_cosine_loss
- **Script**: `train_improved_bge.py`
- **Error**: Training configuration error in `fit()` method
- **Root Cause**: Issue with training objectives or checkpoint path configuration
- **Fix Needed**: Review training configuration parameters

### 6. tier1_adversarial_curriculum
- **Script**: `train_adversarial_curriculum_tier1.py`
- **Error**: `ModuleNotFoundError: No module named 'beir'` or import error
- **Root Cause**: Missing beir module or incorrect import path
- **Fix Needed**: 
  - Install beir: `pip install beir`
  - Or fix import statements if beir is installed but path is wrong

### 7. tier1_cross_attention_query_document
- **Script**: `train_cross_attention_tier1.py`
- **Error**: Device/model loading error in SentenceTransformer initialization
- **Root Cause**: Issue with device assignment or model loading
- **Fix Needed**: 
  - Check device handling
  - Ensure CUDA is properly initialized
  - Fix model loading sequence

### 8. tier1_cross_attention_query_document_fixed
- **Script**: `train_cross_attention_tier1.py`
- **Error**: `ModuleNotFoundError: No module named 'beir'`
- **Root Cause**: Missing beir module
- **Fix Needed**: 
  - Install beir: `pip install beir`
  - Or ensure virtual environment is activated

### 9. tier1_cross_encoder_large
- **Script**: `train_cross_encoder_large_tier1.py`
- **Error**: Device/model loading error in CrossEncoder initialization
- **Root Cause**: Issue with device assignment during model initialization
- **Fix Needed**: 
  - Check device handling in CrossEncoder initialization
  - Ensure proper CUDA device assignment

### 10. tier1_enhanced_contrastive_hardnegatives
- **Script**: `train_enhanced_contrastive_tier1.py`
- **Error**: `'last_hidden_state'` KeyError
- **Root Cause**: Model output structure doesn't match expected format
- **Status**: ✅ **FIXED** - See `tier1_enhanced_contrastive_hardnegatives_fixed`
- **Note**: Fixed version has dtype issues that also need addressing

### 11. tier1_enhanced_contrastive_hardnegatives_fixed
- **Script**: `train_enhanced_contrastive_tier1.py`
- **Error**: `The size of tensor a (121) must match the size of tensor b (282) at non-singleton dimension 1`
- **Previous Error**: `linalg.vector_norm: Expected a floating point or complex tensor as input. Got Long`
- **Root Cause**: 
  1. Dtype issue: tensors need to be float before normalization (partially fixed)
  2. Tensor size mismatch: query and document embeddings have different sizes
- **Fix Needed**: 
  - ✅ Dtype conversion before F.normalize() - **APPLIED**
  - ⚠️ Fix tensor size alignment (padding/truncation or proper handling of variable-length sequences)
  - Check embedding extraction logic

### 12. tier1_graph_aware_retrieval
- **Script**: `train_graph_aware_tier1.py`
- **Error**: `ModuleNotFoundError: No module named 'beir'`
- **Root Cause**: Missing beir module
- **Fix Needed**: 
  - Install beir: `pip install beir`
  - Or ensure virtual environment is activated

### 13. tier1_learning_to_rank_listwise
- **Script**: `train_learning_to_rank_tier1.py`
- **Error**: Evaluation error in `run_learning_to_rank_evaluation`
- **Root Cause**: Issue with evaluation setup or retriever initialization
- **Status**: ✅ **FIXED** - See `tier1_learning_to_rank_listwise_fixed`
- **Note**: Fixed version still has evaluator issues

### 14. tier1_learning_to_rank_listwise_fixed
- **Script**: `train_learning_to_rank_tier1.py`
- **Error**: `Evaluator retriever is not set!`
- **Root Cause**: Evaluator object is being re-initialized with None retriever
- **Fix Applied**: Removed line that re-initialized evaluator with None
- **Status**: ⚠️ **PARTIALLY FIXED** - May need additional verification
- **Fix Needed**: 
  - Verify evaluator is properly initialized before use
  - Ensure retriever is set correctly

### 15. tier1_multitask_retrieval
- **Script**: `train_multitask_retrieval_tier1.py`
- **Error**: `ModuleNotFoundError: No module named 'beir'`
- **Root Cause**: Missing beir module
- **Fix Needed**: 
  - Install beir: `pip install beir`
  - Or ensure virtual environment is activated

### 16. tier1_pseudo_relevance_feedback
- **Script**: `train_pseudo_relevance_feedback_tier1.py`
- **Error**: `ModuleNotFoundError: No module named 'beir'`
- **Root Cause**: Missing beir module
- **Fix Needed**: 
  - Install beir: `pip install beir`
  - Or ensure virtual environment is activated

---

## Common Issues and Solutions

### Issue 1: Missing `beir` Module
**Affected Experiments:** 6 experiments
- tier1_adversarial_curriculum
- tier1_cross_attention_query_document_fixed
- tier1_graph_aware_retrieval
- tier1_multitask_retrieval
- tier1_pseudo_relevance_feedback

**Solution:**
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
source venv/bin/activate
pip install beir
```

### Issue 2: Model Save Path Validation Error
**Affected Experiments:** 2 experiments
- best_paper_adversarial_curriculum
- best_paper_llm_distillation

**Solution:**
- Check model save paths in scripts
- Ensure paths use only alphanumeric, '-', '_', '.' characters
- Limit path length to 96 characters

### Issue 3: Device/Model Loading Errors
**Affected Experiments:** 2 experiments
- tier1_cross_attention_query_document
- tier1_cross_encoder_large

**Solution:**
- Verify CUDA availability before model loading
- Check device assignment logic
- Ensure proper model initialization sequence

### Issue 4: Tensor Size Mismatch
**Affected Experiments:** 1 experiment
- tier1_enhanced_contrastive_hardnegatives_fixed

**Solution:**
- Add padding/truncation for variable-length sequences
- Check embedding extraction logic
- Ensure consistent tensor dimensions

---

## Priority Fixes

### High Priority (Critical)
1. ✅ **best_paper_hierarchical_routing** - Fixed (see _fixed version)
2. ⚠️ **tier1_enhanced_contrastive_hardnegatives_fixed** - Tensor size mismatch needs fix
3. ⚠️ **tier1_learning_to_rank_listwise_fixed** - Evaluator initialization needs verification
4. **best_paper_rl_adaptive_retrieval** - Index out of bounds bug
5. **tier1_cross_attention_query_document_fixed** - Missing beir module

### Medium Priority
1. **tier1_graph_aware_retrieval** - Missing beir module
2. **tier1_multitask_retrieval** - Missing beir module
3. **tier1_pseudo_relevance_feedback** - Missing beir module
4. **best_paper_adversarial_curriculum** - Model save path issue
5. **best_paper_llm_distillation** - Model save path issue

### Low Priority
1. **phase2_cosine_loss** - Training configuration issue
2. **tier1_cross_attention_query_document** - Device handling (non-fixed version)
3. **tier1_cross_encoder_large** - Device handling

---

## Next Steps

1. **Install beir module** for all affected experiments
2. **Fix tensor size mismatch** in tier1_enhanced_contrastive_hardnegatives_fixed
3. **Verify evaluator initialization** in tier1_learning_to_rank_listwise_fixed
4. **Fix indexing bug** in best_paper_rl_adaptive_retrieval
5. **Validate model save paths** in best_paper_adversarial_curriculum and best_paper_llm_distillation

---

## Notes

- Some experiments have "_fixed" versions that address original issues but may have new problems
- The auto experiment runner will automatically retry failed experiments when GPUs become available
- Check `auto_fixed_experiments_status.json` for current status of all experiments
- Monitor `logs/auto_fixed_experiments.log` for auto-runner activity

