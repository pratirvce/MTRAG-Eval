#!/usr/bin/env python3
"""
Generate a comprehensive document with all experiment details:
- Experiments that were run (with results)
- Experiments that were not run
- Experiments that failed
- Detailed results for each
"""

import json
import pathlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set

RESULTS_DIR = pathlib.Path("experiments/retrieval")
STATUS_FILES = [
    "experiment_status.json",
    "stopped_experiments_resume_info.json",
    "fixed_experiments_status.json",
    "tier1_experiments_status.json",
    "auto_fixed_experiments_status.json",
]

def load_all_results() -> Dict[str, Dict]:
    """Load all experiment results from results.json files."""
    results = {}
    
    if not RESULTS_DIR.exists():
        return results
    
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

def load_status_files() -> Dict[str, Dict]:
    """Load all status files to track experiment states."""
    all_status = {}
    
    for status_file in STATUS_FILES:
        filepath = pathlib.Path(status_file)
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_status[status_file] = data
            except Exception as e:
                print(f"Warning: Could not load {status_file}: {e}")
    
    return all_status

def get_all_experiment_directories() -> Set[str]:
    """Get all experiment directory names."""
    if not RESULTS_DIR.exists():
        return set()
    
    return {d.name for d in RESULTS_DIR.iterdir() if d.is_dir()}

def extract_scores(exp_data: Dict) -> Optional[Dict]:
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
    
    return scores if scores else None

def get_experiment_status(exp_name: str, status_files: Dict, all_dirs: Set[str], results: Dict) -> Dict:
    """Determine experiment status."""
    status_info = {
        'has_directory': exp_name in all_dirs,
        'has_results': exp_name in results,
        'status': 'unknown',
        'gpu': None,
        'checkpoint': None,
        'stopped_at': None,
        'error': None,
    }
    
    # Check stopped experiments
    stopped_file = status_files.get('stopped_experiments_resume_info.json', {})
    if 'stopped_experiments' in stopped_file:
        for exp in stopped_file['stopped_experiments']:
            if exp.get('experiment_name') == exp_name:
                status_info['status'] = 'stopped'
                status_info['gpu'] = exp.get('gpu')
                status_info['checkpoint'] = exp.get('checkpoint_dir')
                status_info['stopped_at'] = exp.get('stopped_at')
                break
    
    # Check other status files
    for status_file, data in status_files.items():
        if status_file == 'stopped_experiments_resume_info.json':
            continue
        
        if isinstance(data, dict):
            if exp_name in data:
                exp_status = data[exp_name]
                if isinstance(exp_status, dict):
                    status_info['status'] = exp_status.get('status', status_info['status'])
                    status_info['gpu'] = exp_status.get('gpu', status_info['gpu'])
                    if 'error' in exp_status:
                        status_info['error'] = exp_status['error']
    
    # Determine final status
    if status_info['has_results']:
        if status_info['status'] == 'stopped':
            status_info['status'] = 'completed_after_resume'
        elif status_info['status'] == 'unknown':
            status_info['status'] = 'completed'
    elif status_info['has_directory']:
        if status_info['status'] == 'stopped':
            pass  # Keep as stopped
        elif status_info['status'] == 'unknown':
            status_info['status'] = 'incomplete'
    
    return status_info

def categorize_experiment(name: str) -> str:
    """Categorize experiment by name."""
    name_lower = name.lower()
    
    if 'baseline' in name_lower or name.startswith('baseline_'):
        return 'Baseline'
    elif name.startswith('domain_specific') or 'domain_specific' in name:
        return 'Domain-Specific Fine-tuning'
    elif name.startswith('contrastive') or 'contrastive' in name:
        return 'Contrastive Learning'
    elif name.startswith('meta_learning') or 'meta_learning' in name:
        return 'Meta-Learning'
    elif name.startswith('ensemble') or 'ensemble' in name:
        return 'Ensemble Methods'
    elif name.startswith('query_expansion') or 'query_expansion' in name:
        return 'Query Expansion'
    elif name.startswith('reranking') or 'reranking' in name:
        return 'Reranking'
    elif name.startswith('hybrid') or 'hybrid' in name:
        return 'Hybrid Methods'
    elif name.startswith('multistage') or 'multistage' in name:
        return 'Multi-Stage Retrieval'
    elif name.startswith('learning_to_rank') or 'learning_to_rank' in name:
        return 'Learning-to-Rank'
    elif name.startswith('large_model') or 'large_model' in name:
        return 'Large Model Fine-tuning'
    elif 'hard_negatives' in name or 'hard_neg' in name:
        return 'Hard Negatives Mining'
    elif 'curriculum' in name:
        return 'Curriculum Learning'
    elif 'adversarial' in name:
        return 'Adversarial Training'
    elif 'graph' in name:
        return 'Graph-Based Methods'
    elif 'temporal' in name or 'memory' in name:
        return 'Temporal/Memory Methods'
    elif 'reinforcement' in name or 'rl' in name:
        return 'Reinforcement Learning'
    elif 'distillation' in name:
        return 'Knowledge Distillation'
    elif 'cross_encoder' in name:
        return 'Cross-Encoder'
    elif 'conversation' in name or 'multi_turn' in name:
        return 'Conversation-Aware'
    elif 'llm' in name_lower:
        return 'LLM-Based Methods'
    else:
        return 'Other Methods'

def generate_comprehensive_report() -> str:
    """Generate comprehensive experiment report."""
    lines = []
    
    # Header
    lines.append("# Comprehensive Experiment Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This document provides a complete overview of all experiments:")
    lines.append("- Experiments that were run and completed (with results)")
    lines.append("- Experiments that were started but stopped/paused")
    lines.append("- Experiments that failed")
    lines.append("- Experiments that were never run")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Load data
    print("Loading experiment data...")
    results = load_all_results()
    status_files = load_status_files()
    all_dirs = get_all_experiment_directories()
    
    # Get all unique experiment names
    all_experiments = set()
    all_experiments.update(results.keys())
    all_experiments.update(all_dirs)
    
    # Add experiments from status files
    for status_file, data in status_files.items():
        if isinstance(data, dict):
            if 'stopped_experiments' in data:
                for exp in data['stopped_experiments']:
                    all_experiments.add(exp.get('experiment_name'))
            else:
                all_experiments.update(data.keys())
    
    all_experiments = sorted(all_experiments)
    
    print(f"Found {len(all_experiments)} total experiments")
    print(f"  - {len(results)} with results")
    print(f"  - {len(all_dirs)} with directories")
    
    # Categorize experiments
    experiments_by_status = defaultdict(list)
    experiments_by_category = defaultdict(list)
    
    for exp_name in all_experiments:
        status_info = get_experiment_status(exp_name, status_files, all_dirs, results)
        category = categorize_experiment(exp_name)
        
        exp_data = {
            'name': exp_name,
            'category': category,
            'status_info': status_info,
            'scores': extract_scores(results.get(exp_name, {})) if exp_name in results else None,
        }
        
        experiments_by_status[status_info['status']].append(exp_data)
        experiments_by_category[category].append(exp_data)
    
    # Summary statistics
    lines.append("## Summary Statistics")
    lines.append("")
    lines.append(f"- **Total Experiments:** {len(all_experiments)}")
    lines.append(f"- **Experiments with Results:** {len(results)}")
    lines.append(f"- **Experiments with Directories:** {len(all_dirs)}")
    lines.append("")
    lines.append("### Status Breakdown")
    lines.append("")
    lines.append("| Status | Count |")
    lines.append("|--------|-------|")
    for status in sorted(experiments_by_status.keys()):
        count = len(experiments_by_status[status])
        lines.append(f"| {status} | {count} |")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Experiments with results (completed)
    completed = experiments_by_status.get('completed', []) + experiments_by_status.get('completed_after_resume', [])
    if completed:
        lines.append("## ✅ Completed Experiments (With Results)")
        lines.append("")
        lines.append(f"**Total:** {len(completed)} experiments")
        lines.append("")
        
        # Sort by nDCG@10
        completed_with_scores = [e for e in completed if e['scores']]
        completed_with_scores.sort(key=lambda x: x['scores']['nDCG@10'] if x['scores'] else 0, reverse=True)
        
        lines.append("### Top 20 Completed Experiments by nDCG@10")
        lines.append("")
        lines.append("| Rank | Experiment Name | Category | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 |")
        lines.append("|------|----------------|----------|---------|-----------|--------|----------|")
        
        for rank, exp in enumerate(completed_with_scores[:20], 1):
            name = exp['name']
            if len(name) > 40:
                name = name[:37] + "..."
            scores = exp['scores']
            lines.append(
                f"| {rank} | **{name}** | {exp['category']} | "
                f"{scores['nDCG@10']:.4f} | {scores['Recall@10']:.4f} | "
                f"{scores['nDCG@5']:.4f} | {scores['Recall@5']:.4f} |"
            )
        
        lines.append("")
        lines.append("### All Completed Experiments (Detailed)")
        lines.append("")
        
        # Group by category
        completed_by_category = defaultdict(list)
        for exp in completed_with_scores:
            completed_by_category[exp['category']].append(exp)
        
        for category in sorted(completed_by_category.keys()):
            exps = completed_by_category[category]
            exps.sort(key=lambda x: x['scores']['nDCG@10'] if x['scores'] else 0, reverse=True)
            
            lines.append(f"#### {category} ({len(exps)} experiments)")
            lines.append("")
            lines.append("| Experiment Name | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 | nDCG@3 | Recall@3 | nDCG@1 | Recall@1 |")
            lines.append("|----------------|---------|-----------|--------|----------|--------|----------|--------|----------|")
            
            for exp in exps:
                name = exp['name']
                if len(name) > 50:
                    name = name[:47] + "..."
                scores = exp['scores']
                lines.append(
                    f"| {name} | {scores['nDCG@10']:.4f} | {scores['Recall@10']:.4f} | "
                    f"{scores['nDCG@5']:.4f} | {scores['Recall@5']:.4f} | "
                    f"{scores['nDCG@3']:.4f} | {scores['Recall@3']:.4f} | "
                    f"{scores['nDCG@1']:.4f} | {scores['Recall@1']:.4f} |"
                )
            
            lines.append("")
        
        lines.append("---")
        lines.append("")
    
    # Stopped experiments
    stopped = experiments_by_status.get('stopped', [])
    if stopped:
        lines.append("## ⏸️ Stopped/Paused Experiments")
        lines.append("")
        lines.append(f"**Total:** {len(stopped)} experiments")
        lines.append("")
        lines.append("| Experiment Name | Category | GPU | Checkpoint | Stopped At | Has Results |")
        lines.append("|----------------|----------|-----|------------|------------|-------------|")
        
        for exp in sorted(stopped, key=lambda x: x['name']):
            name = exp['name']
            if len(name) > 40:
                name = name[:37] + "..."
            status = exp['status_info']
            has_results = "✅" if exp['scores'] else "❌"
            checkpoint = status.get('checkpoint', 'N/A')
            if checkpoint and len(str(checkpoint)) > 30:
                checkpoint = str(checkpoint)[:27] + "..."
            stopped_at = status.get('stopped_at', 'N/A')
            if stopped_at and len(str(stopped_at)) > 20:
                stopped_at = str(stopped_at)[:17] + "..."
            
            lines.append(
                f"| {name} | {exp['category']} | {status.get('gpu', 'N/A')} | "
                f"{checkpoint} | {stopped_at} | {has_results} |"
            )
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Incomplete experiments (directory exists but no results)
    incomplete = experiments_by_status.get('incomplete', [])
    if incomplete:
        lines.append("## ⚠️ Incomplete Experiments")
        lines.append("")
        lines.append(f"**Total:** {len(incomplete)} experiments")
        lines.append("")
        lines.append("These experiments have directories but no results.json files.")
        lines.append("")
        lines.append("| Experiment Name | Category | Status |")
        lines.append("|----------------|----------|-------|")
        
        for exp in sorted(incomplete, key=lambda x: x['name']):
            name = exp['name']
            if len(name) > 50:
                name = name[:47] + "..."
            lines.append(f"| {name} | {exp['category']} | {exp['status_info']['status']} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Experiments that were never run
    never_run = experiments_by_status.get('unknown', [])
    if never_run:
        lines.append("## ❌ Experiments Never Run")
        lines.append("")
        lines.append(f"**Total:** {len(never_run)} experiments")
        lines.append("")
        lines.append("These experiments are referenced in status files but were never executed.")
        lines.append("")
        lines.append("| Experiment Name | Category |")
        lines.append("|----------------|----------|")
        
        for exp in sorted(never_run, key=lambda x: x['name']):
            name = exp['name']
            if len(name) > 50:
                name = name[:47] + "..."
            lines.append(f"| {name} | {exp['category']} |")
        
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Category summary
    lines.append("## Category Summary")
    lines.append("")
    lines.append("| Category | Total | Completed | Stopped | Incomplete | Never Run |")
    lines.append("|----------|-------|-----------|---------|------------|----------|")
    
    for category in sorted(experiments_by_category.keys()):
        exps = experiments_by_category[category]
        total = len(exps)
        completed_count = sum(1 for e in exps if e['scores'])
        stopped_count = sum(1 for e in exps if e['status_info']['status'] == 'stopped')
        incomplete_count = sum(1 for e in exps if e['status_info']['status'] == 'incomplete')
        never_run_count = sum(1 for e in exps if e['status_info']['status'] == 'unknown')
        
        lines.append(
            f"| {category} | {total} | {completed_count} | {stopped_count} | "
            f"{incomplete_count} | {never_run_count} |"
        )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Appendix: All experiments (alphabetical)
    lines.append("## Appendix: All Experiments (Alphabetical)")
    lines.append("")
    lines.append("| Experiment Name | Category | Status | Has Results | nDCG@10 |")
    lines.append("|----------------|----------|--------|-------------|---------|")
    
    for exp in sorted(all_experiments):
        exp_data = next(
            (e for e in sum(experiments_by_status.values(), []) if e['name'] == exp),
            None
        )
        if exp_data:
            name = exp_data['name']
            if len(name) > 45:
                name = name[:42] + "..."
            status = exp_data['status_info']['status']
            has_results = "✅" if exp_data['scores'] else "❌"
            ndcg = f"{exp_data['scores']['nDCG@10']:.4f}" if exp_data['scores'] else "N/A"
            
            lines.append(
                f"| {name} | {exp_data['category']} | {status} | {has_results} | {ndcg} |"
            )
    
    return "\n".join(lines)

def main():
    """Main function."""
    print("=" * 80)
    print("Generating Comprehensive Experiment Report")
    print("=" * 80)
    print()
    
    report = generate_comprehensive_report()
    
    # Write to file
    output_file = "COMPREHENSIVE_EXPERIMENT_REPORT.md"
    with open(output_file, 'w') as f:
        f.write(report)
    
    print(f"✅ Report written to: {output_file}")
    print()
    print("Report includes:")
    print("  - All experiments (run and not run)")
    print("  - Detailed results for completed experiments")
    print("  - Status information for all experiments")
    print("  - Category breakdowns")
    print("  - Complete alphabetical listing")

if __name__ == "__main__":
    main()

