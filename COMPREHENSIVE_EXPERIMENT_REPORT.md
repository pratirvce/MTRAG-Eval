# Comprehensive Experiment Report

**Generated:** 2026-01-06

This document provides a complete overview of all experiments:
- Experiments that were run and completed (with results)
- Experiments that were started but stopped/paused  
- Experiments that failed
- Experiments that were never run

---

## Summary Statistics

Based on available data:
- **Total Experiments Found:** ~200+ (across all directories and status files)
- **Experiments with Results:** 109 (from `EXPERIMENT_RESULTS_SUMMARY_LATEST.md`)
- **Stopped Experiments:** 165 (from `stopped_experiments_resume_info.json`)
- **Status File Experiments:** 20 (from `experiment_status.json`)

### Status Breakdown

| Status | Count | Notes |
|--------|-------|-------|
| Completed (with results) | 109 | Have results.json files |
| Stopped/Paused | 165 | Recorded in resume file |
| Failed | Unknown | Some in experiment_status.json |
| Never Run | Unknown | Referenced but not executed |

---

## ✅ Completed Experiments (With Results)

**Total:** 109 experiments with complete results

### Top 20 Experiments by nDCG@10

| Rank | Experiment Name | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | Category |
|------|----------------|---------|-----------|--------|----------|----------|
| 1 | **large_model_finetuning** | 0.5101 | 0.6221 | 0.4538 | 0.4891 | Large Model Fine-tuning |
| 2 | **domain_specific_finetuning_clapnq** | 0.4981 | 0.6016 | 0.4399 | 0.4529 | Domain-Specific Fine-tuning |
| 3 | **domain_specific_finetuning_govt** | 0.4628 | 0.5511 | 0.4210 | 0.4435 | Domain-Specific Fine-tuning |
| 4 | **baseline_bge_finetuned** | 0.4576 | 0.5331 | 0.4092 | 0.4240 | Baseline |
| 5 | **contrastive_learning_finetuning** | 0.4576 | 0.5331 | 0.4092 | 0.4240 | Contrastive Learning |
| 6 | **ensemble_best_performing** | 0.4576 | 0.5331 | 0.4092 | 0.4240 | Ensemble Methods |
| 7 | **query_expansion_govt** | 0.4515 | 0.5317 | 0.4121 | 0.4436 | Query Expansion |
| 8 | **meta_learning_adaptation** | 0.4494 | 0.5644 | 0.3907 | 0.4285 | Meta-Learning |
| 9 | **adversarial_curriculum_learning** | 0.4442 | 0.5561 | 0.3906 | 0.4304 | Adversarial Training |
| 10 | **ensemble_domain_specific_models** | 0.4434 | 0.5355 | 0.3990 | 0.4324 | Ensemble Methods |
| 11 | **ensemble_weighted_fusion** | 0.4370 | 0.5284 | 0.3898 | 0.4154 | Ensemble Methods |
| 12 | **query_expansion_clapnq** | 0.4186 | 0.4999 | 0.3773 | 0.3999 | Query Expansion |
| 13 | **domain_specific_finetuning_cloud** | 0.4104 | 0.5293 | 0.3643 | 0.4293 | Domain-Specific Fine-tuning |
| 14 | **data_augmentation_retrieval** | 0.4098 | 0.5099 | 0.3588 | 0.3868 | Data Augmentation |
| 15 | **query_expansion_multi_domain** | 0.4098 | 0.5099 | 0.3588 | 0.3868 | Query Expansion |
| 16 | **domain_specific_finetuning_fiqa** | 0.4026 | 0.5119 | 0.3500 | 0.3869 | Domain-Specific Fine-tuning |
| 17 | **llm_query_expansion_gpt4** | 0.3787 | 0.4879 | 0.3270 | 0.3614 | LLM-Based Methods |
| 18 | **baseline_5_epochs** | 0.3671 | 0.4693 | 0.3176 | 0.3522 | Baseline |
| 19 | **conversation_aware_attention** | 0.3657 | 0.4642 | 0.3188 | 0.3528 | Conversation-Aware |
| 20 | **multistage_retrieval_2stage_finetuned** | 0.3339 | 0.4223 | 0.2955 | 0.3328 | Multi-Stage Retrieval |

**Note:** For complete detailed results, see `EXPERIMENT_RESULTS_SUMMARY_LATEST.md`

---

## ⏸️ Stopped/Paused Experiments

**Total:** 165 experiments stopped/paused (can be resumed from checkpoints)

Sample of stopped experiments:
- synthetic_data_augmentation
- cross_domain_transfer_learning  
- semantic_drift_adaptation
- temporal_memory_retrieval
- graph_aware_retrieval
- llm_distillation
- differentiable_end_to_end
- contrastive_learning_finetuning (has checkpoint)
- graph_enhanced_reranking_fixed
- cross_attention_rerun (has checkpoint)
- ... and 155 more

**Full list:** See `stopped_experiments_resume_info.json`

**To resume:** Use `resume_from_checkpoint.py` script

---

## ⚠️ Failed/Incomplete Experiments

From `experiment_status.json`:
- domain_specific_clapnq_hard_negatives (status: failed, error: "Process terminated unexpectedly")
- domain_specific_govt_hard_negatives (status: failed, error: "Process terminated unexpectedly")

Additional experiments may have failed but were not recorded in status files.

---

## 📋 Pending Experiments (Not Yet Run)

From `experiment_status.json` (status: "pending"):
- reranking_govt
- reranking_cloud
- reranking_multi_domain
- query_expansion_clapnq
- query_expansion_govt
- query_expansion_multi_domain
- hybrid_clapnq_alpha_0.3
- hybrid_clapnq_alpha_0.5
- hybrid_clapnq_alpha_0.7
- hybrid_govt_alpha_0.3
- hybrid_govt_alpha_0.5
- hybrid_govt_alpha_0.7
- hybrid_multi_alpha_0.3
- hybrid_multi_alpha_0.5
- hybrid_multi_alpha_0.7

Additional pending experiments may be listed in `auto_start_fixed_experiments.py`.

---

## 📊 Category Breakdown

Based on completed experiments:

| Category | Completed | Best nDCG@10 | Best Experiment |
|----------|-----------|--------------|-----------------|
| Large Model Fine-tuning | 1 | 0.5101 | large_model_finetuning |
| Domain-Specific Fine-tuning | 4 | 0.4981 | domain_specific_finetuning_clapnq |
| Baseline | 3 | 0.4576 | baseline_bge_finetuned |
| Contrastive Learning | Multiple | 0.4576 | contrastive_learning_finetuning |
| Ensemble Methods | Multiple | 0.4576 | ensemble_best_performing |
| Query Expansion | Multiple | 0.4515 | query_expansion_govt |
| Meta-Learning | Multiple | 0.4494 | meta_learning_adaptation |
| Adversarial Training | Multiple | 0.4442 | adversarial_curriculum_learning |

**Note:** See `EXPERIMENT_RESULTS_SUMMARY_LATEST.md` for complete category breakdown with all metrics.

---

## 📁 Data Sources

This report is compiled from:
1. **Results Files:** `experiments/retrieval/*/results.json` (109 files)
2. **Status Files:** 
   - `experiment_status.json` (20 experiments)
   - `stopped_experiments_resume_info.json` (165 stopped experiments)
   - `fixed_experiments_status.json`
   - `tier1_experiments_status.json`
   - `auto_fixed_experiments_status.json`
3. **Experiment Directories:** `experiments/retrieval/` (159 directories)
4. **Existing Summaries:** `EXPERIMENT_RESULTS_SUMMARY_LATEST.md`

---

## 🔄 To Generate Full Detailed Report

Run the comprehensive report generator:

```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
python3 generate_comprehensive_experiment_report.py
```

This will create `COMPREHENSIVE_EXPERIMENT_REPORT.md` with:
- Complete listing of all experiments
- Detailed status for each
- Full results breakdown by category
- Alphabetical index of all experiments

---

## 📝 Notes

- **Experiment Naming:** Experiments have been renamed to journal-friendly names (see `experiment_name_mapping.json`)
- **Resume Functionality:** Stopped experiments can be resumed using `resume_from_checkpoint.py`
- **Status Tracking:** Multiple status files are maintained for different experiment sets
- **Results Format:** All results follow BEIR evaluation metrics (Recall@K, nDCG@K)

---

**Report Status:** This is a summary report. For the full detailed report with all experiments, please run `generate_comprehensive_experiment_report.py`.

