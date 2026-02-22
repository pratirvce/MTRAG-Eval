#!/usr/bin/env python3
"""
Generate comprehensive experiment status table
Shows all experiments: completed, running, pending, with results
"""

import json
import pathlib
import subprocess
from datetime import datetime
from typing import Dict, List, Optional, Tuple

EXPERIMENTS_DIR = pathlib.Path("experiments/retrieval")

def get_experiment_results(exp_dir: pathlib.Path) -> Optional[Dict]:
    """Get results from results.json if it exists"""
    results_file = exp_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                return json.load(f)
        except:
            pass
    return None

def get_experiment_status(exp_dir: pathlib.Path, exp_name: str) -> Tuple[str, Optional[Dict]]:
    """Determine experiment status and get results"""
    # Check if running
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if exp_name in line and 'python3' in line and 'train' in line:
                return "🔄 Running", None
    except:
        pass
    
    # Check for results
    results = get_experiment_results(exp_dir)
    if results:
        if 'average' in results:
            avg = results['average']
            if avg.get('nDCG@10', 0) > 0 or avg.get('Recall@10', 0) > 0:
                return "✅ Complete", results
            else:
                return "❌ Failed (zero results)", results
        else:
            return "✅ Complete", results
    
    # Check for checkpoint (in progress)
    checkpoint_file = exp_dir / "checkpoints" / "checkpoint.json"
    if checkpoint_file.exists():
        return "⏸️  Paused/Incomplete", None
    
    # Check if directory exists but no results
    if exp_dir.exists():
        return "⏳ Started (no results yet)", None
    
    return "⏳ Pending", None

def get_running_experiments() -> List[str]:
    """Get list of currently running experiments"""
    running = []
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        for line in result.stdout.split('\n'):
            if 'train_' in line and 'tier1.py' in line and 'experiment_name' in line:
                # Extract experiment name
                parts = line.split('--experiment_name')
                if len(parts) > 1:
                    exp_name = parts[1].split()[0] if parts[1].split() else None
                    if exp_name:
                        running.append(exp_name)
    except:
        pass
    return running

def format_results(results: Dict) -> str:
    """Format results for display"""
    if not results:
        return "N/A"
    
    if 'average' in results:
        avg = results['average']
        recall = avg.get('Recall@10', 0)
        ndcg = avg.get('nDCG@10', 0)
        return f"R@10: {recall:.4f}, nDCG@10: {ndcg:.4f}"
    
    # Try to extract from domain results
    domains = ['clapnq', 'fiqa', 'govt', 'cloud']
    recalls = []
    ndcgs = []
    for domain in domains:
        if domain in results:
            domain_results = results[domain]
            if isinstance(domain_results, dict):
                if 'Recall@10' in domain_results:
                    recalls.append(domain_results['Recall@10'])
                if 'nDCG@10' in domain_results:
                    ndcgs.append(domain_results['nDCG@10'])
    
    if recalls and ndcgs:
        avg_recall = sum(recalls) / len(recalls)
        avg_ndcg = sum(ndcgs) / len(ndcgs)
        return f"R@10: {avg_recall:.4f}, nDCG@10: {avg_ndcg:.4f}"
    
    return "Results available (check file)"

def get_experiment_category(exp_name: str) -> str:
    """Categorize experiment"""
    if exp_name.startswith('best_paper_'):
        return "Best Paper"
    elif exp_name.startswith('tier1_'):
        return "Tier 1"
    elif exp_name.startswith('phase'):
        return f"Phase {exp_name.split('_')[0].replace('phase', '')}"
    else:
        return "Other"

def main():
    print("=" * 120)
    print("COMPREHENSIVE EXPERIMENT STATUS TABLE")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)
    print()
    
    # Get all experiment directories
    if not EXPERIMENTS_DIR.exists():
        print("❌ Experiments directory not found!")
        return
    
    experiment_dirs = sorted([d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()])
    
    # Get running experiments
    running_experiments = get_running_experiments()
    
    # Categorize experiments
    experiments_by_category = {}
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        category = get_experiment_category(exp_name)
        if category not in experiments_by_category:
            experiments_by_category[category] = []
        experiments_by_category[category].append(exp_dir)
    
    # Print table
    print(f"{'Experiment Name':<50} {'Category':<15} {'Status':<25} {'Results':<40}")
    print("-" * 130)
    
    total_completed = 0
    total_running = 0
    total_pending = 0
    total_failed = 0
    
    for category in sorted(experiments_by_category.keys()):
        for exp_dir in sorted(experiments_by_category[category], key=lambda x: x.name):
            exp_name = exp_dir.name
            status, results = get_experiment_status(exp_dir, exp_name)
            
            # Check if actually running
            if exp_name in running_experiments:
                status = "🔄 Running"
            
            results_str = format_results(results) if results else "N/A"
            
            # Truncate long names
            display_name = exp_name[:48] + ".." if len(exp_name) > 50 else exp_name
            
            print(f"{display_name:<50} {category:<15} {status:<25} {results_str:<40}")
            
            # Count
            if "Complete" in status:
                total_completed += 1
            elif "Running" in status:
                total_running += 1
            elif "Failed" in status:
                total_failed += 1
            else:
                total_pending += 1
    
    print("-" * 130)
    print()
    print("SUMMARY:")
    print(f"  ✅ Completed: {total_completed}")
    print(f"  🔄 Running: {total_running}")
    print(f"  ⏳ Pending: {total_pending}")
    print(f"  ❌ Failed: {total_failed}")
    print(f"  📊 Total: {total_completed + total_running + total_pending + total_failed}")
    print()
    
    # Show running experiments details
    if running_experiments:
        print("CURRENTLY RUNNING EXPERIMENTS:")
        print("-" * 130)
        for exp_name in running_experiments:
            exp_dir = EXPERIMENTS_DIR / exp_name
            if exp_dir.exists():
                status, _ = get_experiment_status(exp_dir, exp_name)
                print(f"  • {exp_name:<50} {status}")
        print()

if __name__ == "__main__":
    main()

