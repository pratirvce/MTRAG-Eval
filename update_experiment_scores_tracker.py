#!/usr/bin/env python3
"""
Script to update EXPERIMENT_SCORES_TRACKER.md with current experiment results
Run this periodically to keep the tracker up to date.
"""

import json
import pathlib
from datetime import datetime
from collections import defaultdict

def load_experiment_data():
    """Load all experiment data from results files and status file"""
    
    # Load status file
    status_file = pathlib.Path("auto_fixed_experiments_status.json")
    status_data = {}
    if status_file.exists():
        with open(status_file, 'r') as f:
            status_data = json.load(f)
    
    # Find all experiments
    experiments_dir = pathlib.Path("experiments/retrieval")
    all_experiments = {}
    
    if not experiments_dir.exists():
        return all_experiments
    
    for exp_dir in experiments_dir.iterdir():
        if exp_dir.is_dir():
            exp_name = exp_dir.name
            results_file = exp_dir / "results.json"
            checkpoint_file = exp_dir / "checkpoint.json"
            
            exp_info = {
                'name': exp_name,
                'has_results': results_file.exists(),
                'has_checkpoint': checkpoint_file.exists(),
                'results': None,
                'status': 'unknown',
                'last_updated': None
            }
            
            # Load results if available
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        exp_info['results'] = json.load(f)
                    # Get file modification time
                    exp_info['last_updated'] = datetime.fromtimestamp(
                        results_file.stat().st_mtime
                    ).strftime('%Y-%m-%d %H:%M:%S')
                except Exception as e:
                    print(f"Warning: Could not load results for {exp_name}: {e}")
            
            # Get status from status file
            if exp_name in status_data.get('experiment_statuses', {}):
                exp_info['status'] = status_data['experiment_statuses'][exp_name].get('status', 'unknown')
            elif exp_info['has_results']:
                exp_info['status'] = 'completed'
            elif exp_info['has_checkpoint']:
                exp_info['status'] = 'in_progress'
            else:
                exp_info['status'] = 'pending'
            
            all_experiments[exp_name] = exp_info
    
    return all_experiments

def extract_scores(exp_info):
    """Extract scores from experiment results"""
    if not exp_info['results']:
        return None
    
    results = exp_info['results']
    avg_scores = results.get('average', {})
    domains = results.get('domains', {})
    
    return {
        'nDCG@10': avg_scores.get('nDCG@10'),
        'Recall@10': avg_scores.get('Recall@10'),
        'nDCG@5': avg_scores.get('nDCG@5'),
        'Recall@5': avg_scores.get('Recall@5'),
        'nDCG@1': avg_scores.get('nDCG@1'),
        'Recall@1': avg_scores.get('Recall@1'),
        'domains': list(domains.keys()) if domains else []
    }

def format_score(score):
    """Format score for display"""
    if score is None:
        return "-"
    return f"{score:.4f}"

def categorize_experiment(name):
    """Categorize experiment by name"""
    if name.startswith('task_a_'):
        return 'Task A'
    elif name.startswith('tier1_'):
        return 'Tier 1'
    elif name.startswith('tier2_'):
        return 'Tier 2'
    elif name.startswith('best_paper_'):
        return 'Best Paper'
    elif name.startswith('phase'):
        return 'Phase'
    else:
        return 'Other'

def generate_report():
    """Generate the updated report"""
    
    all_experiments = load_experiment_data()
    
    # Extract experiments with scores
    experiments_with_scores = []
    experiments_by_category = defaultdict(list)
    
    for exp_name, exp_info in all_experiments.items():
        scores = extract_scores(exp_info)
        category = categorize_experiment(exp_name)
        
        if scores:
            exp_data = {
                'name': exp_name,
                'status': exp_info['status'],
                'category': category,
                'last_updated': exp_info.get('last_updated', '-'),
                **scores
            }
        else:
            exp_data = {
                'name': exp_name,
                'status': exp_info['status'],
                'category': category,
                'last_updated': exp_info.get('last_updated', '-'),
                'nDCG@10': None, 'Recall@10': None, 'nDCG@5': None,
                'Recall@5': None, 'nDCG@1': None, 'Recall@1': None,
                'domains': []
            }
        
        experiments_with_scores.append(exp_data)
        experiments_by_category[category].append(exp_data)
    
    # Sort by nDCG@10 (descending, None last)
    experiments_with_scores.sort(
        key=lambda x: x['nDCG@10'] if x['nDCG@10'] is not None else -1,
        reverse=True
    )
    
    # Calculate stats
    completed = [e for e in experiments_with_scores if e['status'] == 'completed']
    running = [e for e in experiments_with_scores if e['status'] == 'running']
    failed = [e for e in experiments_with_scores if e['status'] == 'failed']
    pending = [e for e in experiments_with_scores if e['status'] == 'pending']
    
    completed_with_scores = [e for e in completed if e['nDCG@10'] is not None]
    
    avg_ndcg10 = sum(e['nDCG@10'] for e in completed_with_scores) / len(completed_with_scores) if completed_with_scores else 0
    avg_recall10 = sum(e['Recall@10'] for e in completed_with_scores) / len(completed_with_scores) if completed_with_scores else 0
    
    best_ndcg10 = max((e['nDCG@10'] for e in completed_with_scores), default=None)
    best_recall10 = max((e['Recall@10'] for e in completed_with_scores), default=None)
    
    # Generate report
    report = f"""# Experiment Scores Tracker

**Last Updated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Total Experiments:** {len(all_experiments)}  
**Best nDCG@10:** {format_score(best_ndcg10) if best_ndcg10 else 'TBD'}

This document tracks all experiment scores since cloning the repository. It is automatically updated as experiments complete.

---

## Quick Stats

| Metric | Value |
|--------|-------|
| **Total Completed** | {len(completed)} |
| **Total Running** | {len(running)} |
| **Total Failed** | {len(failed)} |
| **Total Pending** | {len(pending)} |
| **Best nDCG@10** | {format_score(best_ndcg10) if best_ndcg10 else 'TBD'} |
| **Average nDCG@10** | {format_score(avg_ndcg10) if completed_with_scores else 'TBD'} |
| **Best Recall@10** | {format_score(best_recall10) if best_recall10 else 'TBD'} |
| **Average Recall@10** | {format_score(avg_recall10) if completed_with_scores else 'TBD'} |

---

## All Experiments (Sorted by nDCG@10)

| Rank | Experiment Name | Status | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | nDCG@1 | Recall@1 | Domains | Last Updated |
|------|----------------|--------|---------|-----------|--------|----------|--------|----------|---------|--------------|
"""
    
    # Add rows for experiments with scores
    rank = 1
    for exp in experiments_with_scores:
        if exp['nDCG@10'] is not None:
            domains_str = ', '.join(exp['domains']) if exp['domains'] else '-'
            report += f"| {rank} | {exp['name']} | {exp['status']} | {format_score(exp['nDCG@10'])} | {format_score(exp['Recall@10'])} | {format_score(exp['nDCG@5'])} | {format_score(exp['Recall@5'])} | {format_score(exp['nDCG@1'])} | {format_score(exp['Recall@1'])} | {domains_str} | {exp['last_updated']} |\n"
            rank += 1
    
    # Add rows for experiments without scores
    for exp in experiments_with_scores:
        if exp['nDCG@10'] is None:
            domains_str = ', '.join(exp['domains']) if exp['domains'] else '-'
            report += f"| - | {exp['name']} | {exp['status']} | - | - | - | - | - | - | {domains_str} | {exp['last_updated']} |\n"
    
    # Add category sections
    report += "\n---\n\n## Experiments by Category\n\n"
    
    # Task A Experiments
    task_a_exps = [e for e in experiments_with_scores if e['category'] == 'Task A']
    if task_a_exps:
        report += "### Task A Experiments (Ultra High Performance)\n\n"
        report += "| Experiment Name | Status | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | Last Updated |\n"
        report += "|----------------|--------|---------|-----------|--------|----------|--------------|\n"
        for exp in sorted(task_a_exps, key=lambda x: x['nDCG@10'] if x['nDCG@10'] is not None else -1, reverse=True):
            report += f"| {exp['name']} | {exp['status']} | {format_score(exp['nDCG@10'])} | {format_score(exp['Recall@10'])} | {format_score(exp['nDCG@5'])} | {format_score(exp['Recall@5'])} | {exp['last_updated']} |\n"
        report += "\n"
    
    # Tier 1 Experiments
    tier1_exps = [e for e in experiments_with_scores if e['category'] == 'Tier 1']
    if tier1_exps:
        report += "### Tier 1 Experiments\n\n"
        report += "| Experiment Name | Status | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | Last Updated |\n"
        report += "|----------------|--------|---------|-----------|--------|----------|--------------|\n"
        for exp in sorted(tier1_exps, key=lambda x: x['nDCG@10'] if x['nDCG@10'] is not None else -1, reverse=True):
            report += f"| {exp['name']} | {exp['status']} | {format_score(exp['nDCG@10'])} | {format_score(exp['Recall@10'])} | {format_score(exp['nDCG@5'])} | {format_score(exp['Recall@5'])} | {exp['last_updated']} |\n"
        report += "\n"
    
    # Best Paper Methods
    best_paper_exps = [e for e in experiments_with_scores if e['category'] == 'Best Paper']
    if best_paper_exps:
        report += "### Best Paper Methods\n\n"
        report += "| Experiment Name | Status | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | Last Updated |\n"
        report += "|----------------|--------|---------|-----------|--------|----------|--------------|\n"
        for exp in sorted(best_paper_exps, key=lambda x: x['nDCG@10'] if x['nDCG@10'] is not None else -1, reverse=True):
            report += f"| {exp['name']} | {exp['status']} | {format_score(exp['nDCG@10'])} | {format_score(exp['Recall@10'])} | {format_score(exp['nDCG@5'])} | {format_score(exp['Recall@5'])} | {exp['last_updated']} |\n"
        report += "\n"
    
    # Phase Experiments
    phase_exps = [e for e in experiments_with_scores if e['category'] == 'Phase']
    if phase_exps:
        report += "### Phase Experiments\n\n"
        report += "| Experiment Name | Status | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | Last Updated |\n"
        report += "|----------------|--------|---------|-----------|--------|----------|--------------|\n"
        for exp in sorted(phase_exps, key=lambda x: x['nDCG@10'] if x['nDCG@10'] is not None else -1, reverse=True):
            report += f"| {exp['name']} | {exp['status']} | {format_score(exp['nDCG@10'])} | {format_score(exp['Recall@10'])} | {format_score(exp['nDCG@5'])} | {format_score(exp['Recall@5'])} | {exp['last_updated']} |\n"
        report += "\n"
    
    # Add footer
    report += """---

## Status Legend

- **Completed**: Experiment finished successfully with results
- **Running**: Experiment is currently executing
- **Failed**: Experiment encountered an error
- **Pending**: Experiment not yet started
- **In Progress**: Experiment has checkpoint but not completed

---

## Update Instructions

This report can be updated by running:
```bash
python3 update_experiment_scores_tracker.py
```

Or manually update the tables above with new scores as experiments complete.

---

## Notes

- Scores are extracted from `experiments/retrieval/{experiment_name}/results.json`
- Status is tracked in `auto_fixed_experiments_status.json`
- nDCG@10 is the primary ranking metric
- All scores are averaged across domains (clapnq, fiqa, govt, cloud)
- Last Updated timestamp is based on results.json file modification time
"""
    
    return report

if __name__ == "__main__":
    report = generate_report()
    
    # Write to file
    output_file = pathlib.Path("EXPERIMENT_SCORES_TRACKER.md")
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Updated {output_file}")
    print(f"   Total experiments: {len(load_experiment_data())}")

