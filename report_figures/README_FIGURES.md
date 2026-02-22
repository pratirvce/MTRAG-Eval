# Report Figures Required

The following figures should be created for the LaTeX report:

## Existing Figures (should already exist):
1. `system_architecture.png` - Multi-Turn RAG Retrieval System Architecture
2. `fine_tuning_pipeline.png` - Fine-tuning Pipeline
3. `experimental_setup.png` - Four-Phase Experimental Approach (update from 3-phase)
4. `data_flow.png` - Data Flow in Retrieval System
5. `overall_comparison.png` - Overall Performance Comparison
6. `phase_progression.png` - Performance Progression Across Phases
7. `domain_performance.png` - Domain-specific Performance Breakdown
8. `configuration_impact.png` - Training Configuration Impact Analysis
9. `improvement_percentage.png` - Percentage Improvement over Baseline

## New Figures Needed for Phase 4:

### 1. `phase4_domain_comparison.png`
**Description**: Bar chart comparing domain-specific models vs phase2_augmentation
- X-axis: Domains (CLAPNQ, FIQA, GOVT, CLOUD)
- Y-axis: Recall@10 or nDCG@10
- Two bars per domain: phase2_augmentation vs phase4_domain_specific
- Show percentage improvement labels

### 2. `phase4_results.png`
**Description**: Multi-metric visualization of Phase 4 domain-specific results
- Show all 4 domains
- Include R@5, R@10, nDCG@5, nDCG@10 for each domain
- Could be grouped bar chart or radar chart

### 3. `transfer_learning_approach.png`
**Description**: Diagram showing two-stage transfer learning
- Stage 1: Multi-domain pre-training (phase1_epochs5)
- Stage 2: Domain-specific fine-tuning (4 separate models)
- Show arrows/flow from Stage 1 to Stage 2
- Label each domain-specific model

### 4. `all_phases_comparison.png`
**Description**: Line chart or bar chart showing progression across all phases
- X-axis: Experiments (grouped by phase)
- Y-axis: Recall@10 or nDCG@10
- Include Phase 1, 2, 3, and Phase 4 average
- Highlight best performers

### 5. `domain_improvements.png`
**Description**: Bar chart showing percentage improvement per domain
- X-axis: Domains
- Y-axis: Percentage improvement
- Show both Recall@10 and nDCG@10 improvements
- Use different colors for each metric

## Data for Creating Figures:

### Phase 4 Domain-Specific Results:
- CLAPNQ: R@10=0.6016, nDCG@10=0.4981
- FIQA: R@10=0.5119, nDCG@10=0.4026
- GOVT: R@10=0.5511, nDCG@10=0.4628
- CLOUD: R@10=0.5293, nDCG@10=0.4104

### Phase 2 Augmentation (for comparison):
- CLAPNQ: R@10=0.5816, nDCG@10=0.4786
- FIQA: R@10=0.4911, nDCG@10=0.4024
- GOVT: R@10=0.5072, nDCG@10=0.4041
- CLOUD: R@10=0.4598, nDCG@10=0.3543

### Improvements:
- CLAPNQ: +3.4% R@10, +4.1% nDCG@10
- FIQA: +4.2% R@10, +0.0% nDCG@10
- GOVT: +8.7% R@10, +14.5% nDCG@10
- CLOUD: +15.1% R@10, +15.8% nDCG@10
