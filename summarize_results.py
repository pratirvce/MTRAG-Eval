#!/usr/bin/env python3
"""Summarize all experiment results."""
import json
import pathlib
from datetime import datetime

RESULTS_DIR = pathlib.Path("experiments/retrieval")

def load_results():
    """Load all experiment results."""
    results = {}
    
    if not RESULTS_DIR.exists():
        print("No results directory found.")
        return results
    
    # Load summary if exists
    summary_file = RESULTS_DIR / "experiment_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            return json.load(f)
    
    # Load individual results
    for exp_dir in RESULTS_DIR.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results[exp_dir.name] = json.load(f)
    
    return results

def print_summary(results):
    """Print formatted summary."""
    print("\n" + "="*80)
    print("RETRIEVAL EXPERIMENT RESULTS SUMMARY")
    print("="*80)
    
    baseline = {
        'Recall@5': 0.30,
        'Recall@10': 0.38,
        'nDCG@5': 0.27,
        'nDCG@10': 0.30
    }
    
    print(f"\n{'Experiment':<30} | {'R@10':<8} | {'nDCG@10':<8} | {'vs Baseline':<12}")
    print("-" * 80)
    
    for exp_name, exp_data in sorted(results.items()):
        if isinstance(exp_data, dict) and 'averages' in exp_data:
            avg = exp_data['averages']
            r10 = avg.get('Recall@10', 0)
            ndcg10 = avg.get('nDCG@10', 0)
            vs_baseline = r10 - baseline['Recall@10']
            sign = "+" if vs_baseline >= 0 else ""
            print(f"{exp_name:<30} | {r10:<8.4f} | {ndcg10:<8.4f} | {sign}{vs_baseline:>+11.4f}")
    
    print("\n" + "="*80)
    print("Baseline (Pre-trained BGE): R@10=0.3800, nDCG@10=0.3000")
    print("="*80 + "\n")

if __name__ == "__main__":
    results = load_results()
    
    if isinstance(results, dict) and 'results' in results:
        # New format with phases
        print("\nResults by Phase:")
        for phase, phase_results in results['results'].items():
            print(f"\n{phase.upper()}:")
            phase_dict = {exp['experiment']: exp.get('results', {}) for exp in phase_results}
            print_summary(phase_dict)
    else:
        # Old format
        print_summary(results)

