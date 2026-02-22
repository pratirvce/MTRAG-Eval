#!/usr/bin/env python3
"""
Generate LaTeX tables from experiment results for the paper.
"""

import json
import pathlib
from collections import defaultdict
from typing import Dict, List

RESULTS_DIR = pathlib.Path("experiments/retrieval")

def load_all_results():
    """Load all experiment results."""
    results = {}
    for exp_dir in RESULTS_DIR.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        results[exp_dir.name] = data
                except Exception as e:
                    print(f"Warning: Could not load {results_file}: {e}")
    return results

def extract_scores(exp_data: Dict) -> Dict:
    """Extract scores from experiment data."""
    scores = {}
    if 'average' in exp_data:
        avg = exp_data['average']
        scores = {
            'Recall@1': avg.get('Recall@1', 0),
            'Recall@3': avg.get('Recall@3', 0),
            'Recall@5': avg.get('Recall@5', 0),
            'Recall@10': avg.get('Recall@10', 0),
            'nDCG@1': avg.get('nDCG@1', 0),
            'nDCG@3': avg.get('nDCG@3', 0),
            'nDCG@5': avg.get('nDCG@5', 0),
            'nDCG@10': avg.get('nDCG@10', 0),
        }
    elif 'averages' in exp_data:
        avg = exp_data['averages']
        scores = {
            'Recall@1': avg.get('Recall@1', 0),
            'Recall@3': avg.get('Recall@3', 0),
            'Recall@5': avg.get('Recall@5', 0),
            'Recall@10': avg.get('Recall@10', 0),
            'nDCG@1': avg.get('nDCG@1', 0),
            'nDCG@3': avg.get('nDCG@3', 0),
            'nDCG@5': avg.get('nDCG@5', 0),
            'nDCG@10': avg.get('nDCG@10', 0),
        }
    return scores

def categorize_experiment(name: str) -> str:
    """Categorize experiment by name prefix."""
    if name.startswith('best_paper_'):
        return 'Best Paper'
    elif name.startswith('tier1_'):
        return 'Tier 1'
    elif name.startswith('phase1_'):
        return 'Phase 1'
    elif name.startswith('phase2_'):
        return 'Phase 2'
    elif name.startswith('phase3_'):
        return 'Phase 3'
    elif name.startswith('phase4_'):
        return 'Phase 4'
    elif name.startswith('phase5_'):
        return 'Phase 5'
    elif name.startswith('phase6_'):
        return 'Phase 6'
    elif name.startswith('phase7_'):
        return 'Phase 7'
    elif name.startswith('task_a_'):
        return 'Task A'
    else:
        return 'Other'

def generate_top_methods_table(results: Dict, top_n: int = 20) -> str:
    """Generate LaTeX table for top N methods."""
    exp_scores = []
    for exp_name, exp_data in results.items():
        scores = extract_scores(exp_data)
        if scores and scores.get('nDCG@10', 0) > 0:
            exp_scores.append({
                'name': exp_name,
                'category': categorize_experiment(exp_name),
                **scores
            })
    
    sorted_by_ndcg = sorted(exp_scores, key=lambda x: x['nDCG@10'], reverse=True)
    
    lines = []
    lines.append("\\begin{table*}[h]")
    lines.append("\\centering")
    lines.append(f"\\caption{{Top {top_n} Methods by nDCG@10}}")
    lines.append("\\label{tab:top_methods_complete}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Rank} & \\textbf{Method} & \\textbf{nDCG@10} & \\textbf{R@10} & \\textbf{nDCG@5} & \\textbf{R@5} \\\\")
    lines.append("\\midrule")
    
    for rank, exp in enumerate(sorted_by_ndcg[:top_n], 1):
        name = exp['name'].replace('_', '\\_')
        # Shorten long names
        if len(name) > 50:
            name = name[:47] + "..."
        lines.append(
            "{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                rank, name, exp['nDCG@10'], exp['Recall@10'], 
                exp['nDCG@5'], exp['Recall@5']
            )
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table*}")
    
    return "\n".join(lines)

def generate_category_table(results: Dict) -> str:
    """Generate LaTeX table summarizing by category."""
    by_category = defaultdict(list)
    for exp_name, exp_data in results.items():
        scores = extract_scores(exp_data)
        if scores and scores.get('nDCG@10', 0) > 0:
            category = categorize_experiment(exp_name)
            by_category[category].append(scores)
    
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\caption{Performance Summary by Category}")
    lines.append("\\label{tab:category_summary}")
    lines.append("\\small")
    lines.append("\\begin{tabular}{lccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Category} & \\textbf{\\# Exps} & \\textbf{Avg nDCG@10} & \\textbf{Best nDCG@10} & \\textbf{Avg R@10} & \\textbf{Best R@10} \\\\")
    lines.append("\\midrule")
    
    for category in sorted(by_category.keys()):
        exps = by_category[category]
        ndcg_scores = [e['nDCG@10'] for e in exps]
        recall_scores = [e['Recall@10'] for e in exps]
        avg_ndcg = sum(ndcg_scores) / len(ndcg_scores) if ndcg_scores else 0
        best_ndcg = max(ndcg_scores) if ndcg_scores else 0
        avg_recall = sum(recall_scores) / len(recall_scores) if recall_scores else 0
        best_recall = max(recall_scores) if recall_scores else 0
        
        lines.append(
            "{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                category, len(exps), avg_ndcg, best_ndcg, avg_recall, best_recall
            )
        )
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

def generate_all_experiments_table(results: Dict) -> str:
    """Generate comprehensive table of all experiments."""
    exp_scores = []
    for exp_name, exp_data in results.items():
        scores = extract_scores(exp_data)
        if scores and scores.get('nDCG@10', 0) > 0:
            exp_scores.append({
                'name': exp_name,
                'category': categorize_experiment(exp_name),
                **scores
            })
    
    # Sort by category, then by nDCG@10
    exp_scores.sort(key=lambda x: (x['category'], -x['nDCG@10']))
    
    lines = []
    lines.append("\\begin{longtable}{lccccc}")
    lines.append(f"\\caption{{Complete Experimental Results (All {len(exp_scores)} Experiments)}}")
    lines.append("\\label{tab:all_experiments} \\\\")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Category} & \\textbf{nDCG@10} & \\textbf{R@10} & \\textbf{nDCG@5} & \\textbf{R@5} \\\\")
    lines.append("\\midrule")
    lines.append("\\endfirsthead")
    lines.append("\\multicolumn{6}{c}{{Continued from previous page}} \\\\")
    lines.append("\\toprule")
    lines.append("\\textbf{Method} & \\textbf{Category} & \\textbf{nDCG@10} & \\textbf{R@10} & \\textbf{nDCG@5} & \\textbf{R@5} \\\\")
    lines.append("\\midrule")
    lines.append("\\endhead")
    lines.append("\\bottomrule")
    lines.append("\\multicolumn{6}{r}{{Continued on next page}} \\\\")
    lines.append("\\endfoot")
    lines.append("\\bottomrule")
    lines.append("\\endlastfoot")
    
    for exp in exp_scores:
        name = exp['name'].replace('_', '\\_')
        if len(name) > 40:
            name = name[:37] + "..."
        lines.append(
            "{} & {} & {:.4f} & {:.4f} & {:.4f} & {:.4f} \\\\".format(
                name, exp['category'], exp['nDCG@10'], exp['Recall@10'],
                exp['nDCG@5'], exp['Recall@5']
            )
        )
    
    lines.append("\\end{longtable}")
    
    return "\n".join(lines)

def main():
    """Main function."""
    print("Loading experiment results...")
    results = load_all_results()
    
    print(f"Found {len(results)} experiments")
    
    print("Generating LaTeX tables...")
    
    # Generate tables
    category_table = generate_category_table(results)
    top_methods_table = generate_top_methods_table(results, top_n=20)
    all_experiments_table = generate_all_experiments_table(results)
    
    # Write to file
    output_file = "latex_tables.tex"
    with open(output_file, 'w') as f:
        f.write("% Category Summary Table\n")
        f.write(category_table)
        f.write("\n\n")
        f.write("% Top 20 Methods Table\n")
        f.write(top_methods_table)
        f.write("\n\n")
        f.write("% All Experiments Table\n")
        f.write(all_experiments_table)
    
    print(f"\n✅ LaTeX tables written to: {output_file}")

if __name__ == "__main__":
    main()

