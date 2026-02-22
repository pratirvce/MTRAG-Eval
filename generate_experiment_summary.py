#!/usr/bin/env python3
"""
Generate a comprehensive summary of all experiment results with scores.
"""

import json
import pathlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional

RESULTS_DIR = pathlib.Path("experiments/retrieval")

def load_all_results():
    """Load all experiment results from results.json files."""
    results = {}
    
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return results
    
    # Find all results.json files
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
    """Extract scores from experiment data (handles different formats)."""
    scores = {}
    
    # Check if it has 'average' key
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
    # Check if it has 'averages' key
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
        return 'Best Paper Methods'
    elif name.startswith('tier1_'):
        return 'Tier 1 Experiments'
    elif name.startswith('phase1_'):
        return 'Phase 1 (Baseline)'
    elif name.startswith('phase2_'):
        return 'Phase 2 Experiments'
    elif name.startswith('phase3_'):
        return 'Phase 3 Experiments'
    elif name.startswith('phase4_'):
        return 'Phase 4 Experiments'
    elif name.startswith('phase5_'):
        return 'Phase 5 Experiments'
    elif name.startswith('phase6_'):
        return 'Phase 6 Experiments'
    elif name.startswith('phase7_'):
        return 'Phase 7 Experiments'
    elif name.startswith('task_a_'):
        return 'Task A Experiments'
    else:
        return 'Other Experiments'

def generate_summary(results: Dict) -> str:
    """Generate markdown summary."""
    lines = []
    lines.append("# Experiment Results Summary")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    
    # Extract scores for all experiments
    exp_scores = []
    for exp_name, exp_data in results.items():
        scores = extract_scores(exp_data)
        if scores and scores.get('nDCG@10', 0) > 0:  # Only include if has valid scores
            exp_scores.append({
                'name': exp_name,
                'category': categorize_experiment(exp_name),
                **scores
            })
    
    # Overall statistics
    if exp_scores:
        ndcg10_scores = [e['nDCG@10'] for e in exp_scores]
        recall10_scores = [e['Recall@10'] for e in exp_scores]
        
        lines.append("## Overview")
        lines.append("")
        lines.append(f"- **Total Experiments:** {len(exp_scores)}")
        lines.append(f"- **Average nDCG@10:** {sum(ndcg10_scores) / len(ndcg10_scores):.4f}")
        lines.append(f"- **Best nDCG@10:** {max(ndcg10_scores):.4f}")
        lines.append(f"- **Worst nDCG@10:** {min(ndcg10_scores):.4f}")
        lines.append(f"- **Average Recall@10:** {sum(recall10_scores) / len(recall10_scores):.4f}")
        lines.append(f"- **Best Recall@10:** {max(recall10_scores):.4f}")
        lines.append(f"- **Worst Recall@10:** {min(recall10_scores):.4f}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Top 20 experiments by nDCG@10
    if exp_scores:
        sorted_by_ndcg = sorted(exp_scores, key=lambda x: x['nDCG@10'], reverse=True)
        
        lines.append("## Top 20 Experiments by nDCG@10")
        lines.append("")
        lines.append("| Rank | Experiment Name | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | Category |")
        lines.append("|------|----------------|---------|-----------|--------|----------|----------|")
        
        for rank, exp in enumerate(sorted_by_ndcg[:20], 1):
            name = exp['name']
            if len(name) > 40:
                name = name[:37] + "..."
            lines.append(
                f"| {rank} | **{name}** | {exp['nDCG@10']:.4f} | {exp['Recall@10']:.4f} | "
                f"{exp['nDCG@5']:.4f} | {exp['Recall@5']:.4f} | {exp['category']} |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Group by category
    by_category = defaultdict(list)
    for exp in exp_scores:
        by_category[exp['category']].append(exp)
    
    # Sort categories by best score
    category_order = sorted(
        by_category.items(),
        key=lambda x: max(e['nDCG@10'] for e in x[1]) if x[1] else 0,
        reverse=True
    )
    
    for category, exps in category_order:
        if not exps:
            continue
            
        # Sort experiments in category by nDCG@10
        exps_sorted = sorted(exps, key=lambda x: x['nDCG@10'], reverse=True)
        
        # Category statistics
        cat_ndcg = [e['nDCG@10'] for e in exps]
        cat_recall = [e['Recall@10'] for e in exps]
        cat_avg_ndcg = sum(cat_ndcg) / len(cat_ndcg) if cat_ndcg else 0
        cat_avg_recall = sum(cat_recall) / len(cat_recall) if cat_recall else 0
        cat_best_ndcg = max(cat_ndcg) if cat_ndcg else 0
        cat_best_recall = max(cat_recall) if cat_recall else 0
        
        lines.append(f"## {category} ({len(exps)} experiments)")
        lines.append("")
        lines.append(f"**Category Average:** nDCG@10: {cat_avg_ndcg:.4f}, Recall@10: {cat_avg_recall:.4f}  ")
        lines.append(f"**Category Best:** nDCG@10: {cat_best_ndcg:.4f}, Recall@10: {cat_best_recall:.4f}")
        lines.append("")
        lines.append("| Experiment Name | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | nDCG@3 | Recall@3 |")
        lines.append("|----------------|---------|-----------|--------|----------|--------|----------|")
        
        for exp in exps_sorted:
            name = exp['name']
            # Remove category prefix for readability
            if name.startswith('best_paper_'):
                display_name = name[11:]
            elif name.startswith('tier1_'):
                display_name = name[6:]
            elif name.startswith('phase') and '_' in name:
                display_name = name
            else:
                display_name = name
            
            if len(display_name) > 40:
                display_name = display_name[:37] + "..."
            
            lines.append(
                f"| **{display_name}** | {exp['nDCG@10']:.4f} | {exp['Recall@10']:.4f} | "
                f"{exp['nDCG@5']:.4f} | {exp['Recall@5']:.4f} | "
                f"{exp['nDCG@3']:.4f} | {exp['Recall@3']:.4f} |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Baseline comparison
    baseline_exp = None
    for exp in exp_scores:
        if exp['name'] == 'phase1_baseline':
            baseline_exp = exp
            break
    
    if baseline_exp:
        lines.append("## Baseline Comparison")
        lines.append("")
        lines.append(f"**Baseline (phase1_baseline):**")
        lines.append(f"- nDCG@10: {baseline_exp['nDCG@10']:.4f}")
        lines.append(f"- Recall@10: {baseline_exp['Recall@10']:.4f}")
        lines.append("")
        lines.append("Experiments that outperform baseline:")
        lines.append("")
        lines.append("| Experiment | nDCG@10 | Recall@10 | Improvement (nDCG@10) |")
        lines.append("|------------|---------|-----------|----------------------|")
        
        improvements = [
            e for e in exp_scores
            if e['nDCG@10'] > baseline_exp['nDCG@10']
        ]
        improvements_sorted = sorted(improvements, key=lambda x: x['nDCG@10'], reverse=True)
        
        for exp in improvements_sorted[:30]:  # Top 30 improvements
            name = exp['name']
            if len(name) > 35:
                name = name[:32] + "..."
            improvement = exp['nDCG@10'] - baseline_exp['nDCG@10']
            lines.append(
                f"| {name} | {exp['nDCG@10']:.4f} | {exp['Recall@10']:.4f} | "
                f"+{improvement:.4f} |"
            )
        
        lines.append("")
    
    return "\n".join(lines)

def main():
    """Main function."""
    print("Loading experiment results...")
    results = load_all_results()
    
    print(f"Found {len(results)} experiments with results")
    
    print("Generating summary...")
    summary = generate_summary(results)
    
    # Write to file
    output_file = "EXPERIMENT_RESULTS_SUMMARY_LATEST.md"
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"\n✅ Summary written to: {output_file}")
    print(f"\nTotal experiments with results: {len(results)}")
    
    # Print summary to console
    print("\n" + "="*80)
    print("QUICK SUMMARY")
    print("="*80)
    
    exp_scores = []
    for exp_name, exp_data in results.items():
        scores = extract_scores(exp_data)
        if scores and scores.get('nDCG@10', 0) > 0:
            exp_scores.append({
                'name': exp_name,
                **scores
            })
    
    if exp_scores:
        sorted_by_ndcg = sorted(exp_scores, key=lambda x: x['nDCG@10'], reverse=True)
        
        print(f"\nTop 10 by nDCG@10:")
        print(f"{'Rank':<6} {'Experiment':<50} {'nDCG@10':<10} {'Recall@10':<10}")
        print("-" * 80)
        for rank, exp in enumerate(sorted_by_ndcg[:10], 1):
            name = exp['name']
            if len(name) > 48:
                name = name[:45] + "..."
            print(f"{rank:<6} {name:<50} {exp['nDCG@10']:<10.4f} {exp['Recall@10']:<10.4f}")

if __name__ == "__main__":
    main()

