# Retrieval Experiments Report

## Overview
This directory contains a comprehensive LaTeX report documenting all retrieval experiments conducted on the MT-RAG benchmark.

## Files

### Main Report
- **`retrieval_experiments_report.tex`**: Complete LaTeX source for the report
  - Includes all sections: Introduction, Methodology, Results, Analysis, Conclusions
  - References all visualizations
  - Contains detailed tables and findings

### Visualizations
All visualizations are in the `report_figures/` directory:
- **`overall_comparison.png/pdf`**: Comparison of Recall@10 and nDCG@10 across all experiments
- **`improvement_over_baseline.png/pdf`**: Percentage improvement over baseline
- **`phase_progression.png/pdf`**: Performance progression across phases
- **`domain_performance.png/pdf`**: Domain-specific performance breakdown
- **`performance_heatmap.png/pdf`**: Comprehensive heatmap of all metrics

### Generation Script
- **`generate_report_visualizations.py`**: Python script that generates all visualizations from the backup data

## Compiling the Report

### Prerequisites
Install LaTeX distribution (e.g., TeX Live):
```bash
# Ubuntu/Debian
sudo apt-get install texlive-full

# macOS
brew install --cask mactex

# Or use online LaTeX editors like Overleaf
```

### Compilation
```bash
cd /home/prevanka/prati/su-mt-rag/mt-rag-benchmark
pdflatex retrieval_experiments_report.tex
pdflatex retrieval_experiments_report.tex  # Run twice for references
```

### Online Compilation
Alternatively, upload to [Overleaf](https://www.overleaf.com/) for online compilation.

## Report Contents

1. **Introduction**: Background, objectives, and dataset description
2. **Methodology**: Base model, evaluation metrics, baseline
3. **Experimental Setup**: Detailed description of all three phases
4. **Results**: Comprehensive results with tables and figures
5. **Detailed Analysis**: Phase-by-phase analysis and key findings
6. **Key Findings**: Best configuration and performance gains
7. **Challenges and Limitations**: Infrastructure constraints and future improvements
8. **Conclusions**: Summary of achievements
9. **Recommendations**: Next steps and future work
10. **Appendix**: Experiment configurations and data sources

## Key Results Summary

### Best Model: phase2_augmentation
- **Recall@10**: 0.5099 (34.2% improvement over baseline)
- **nDCG@10**: 0.4098 (36.6% improvement over baseline)
- **Recall@5**: 0.3868 (28.9% improvement over baseline)
- **nDCG@5**: 0.3588 (32.9% improvement over baseline)

### Best Configuration
- Base Model: BAAI/bge-base-en-v1.5
- Epochs: 3
- Batch Size: 32
- Learning Rate: 2e-5
- Data Augmentation: Enabled
- Validation: Enabled with best model saving

## Data Source

All experimental data is from:
- **Backup folder**: `backup-retrieval-1126/retrieval/`
- **Summary file**: `backup-retrieval-1126/retrieval/experiment_summary.json`
- **Individual results**: `backup-retrieval-1126/retrieval/[experiment_name]/results.json`

## Regenerating Visualizations

To regenerate visualizations:
```bash
python generate_report_visualizations.py
```

This will create/update all figures in `report_figures/` directory.

## Notes

- The report references data from the backup folder created on 2025-11-26
- All experiments were conducted systematically across three phases
- Phase 3 experiments (hybrid, reranking) had infrastructure limitations
- The report recommends integrating reranking with the fine-tuned phase2_augmentation model

