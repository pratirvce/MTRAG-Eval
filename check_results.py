#!/usr/bin/env python3
"""
Quick script to check experiment results
"""

import json
import pathlib
import sys
from typing import Dict, Optional
from experiment_runner import ExperimentRunner

def format_results(results: Dict) -> str:
    """Format results for display"""
    if not results:
        return "No results available"
    
    lines = []
    
    # Show average if available
    if 'average' in results:
        avg = results['average']
        lines.append("Average Results:")
        lines.append(f"  Recall@1:  {avg.get('Recall@1', 0):.4f}")
        lines.append(f"  Recall@3:  {avg.get('Recall@3', 0):.4f}")
        lines.append(f"  Recall@5:  {avg.get('Recall@5', 0):.4f}")
        lines.append(f"  Recall@10: {avg.get('Recall@10', 0):.4f}")
        lines.append(f"  nDCG@1:   {avg.get('nDCG@1', 0):.4f}")
        lines.append(f"  nDCG@3:   {avg.get('nDCG@3', 0):.4f}")
        lines.append(f"  nDCG@5:   {avg.get('nDCG@5', 0):.4f}")
        lines.append(f"  nDCG@10:  {avg.get('nDCG@10', 0):.4f}")
        lines.append("")
    
    # Show per-domain results
    domains = ['clapnq', 'fiqa', 'govt', 'cloud']
    for domain in domains:
        if domain in results:
            domain_results = results[domain]
            lines.append(f"{domain.upper()}:")
            lines.append(f"  Recall@10: {domain_results.get('Recall@10', 0):.4f}")
            lines.append(f"  nDCG@10:   {domain_results.get('nDCG@10', 0):.4f}")
    
    return "\n".join(lines)

def check_experiment_results(experiment_name: Optional[str] = None):
    """Check results for one or all experiments"""
    runner = ExperimentRunner()
    runner.update_status()
    status = runner.get_status_summary()
    
    if experiment_name:
        # Check specific experiment
        if experiment_name not in status['experiments']:
            print(f"❌ Experiment '{experiment_name}' not found")
            return
        
        exp_info = status['experiments'][experiment_name]
        results_file = pathlib.Path("experiments/retrieval") / experiment_name / "results.json"
        
        print(f"\n{'='*60}")
        print(f"Experiment: {experiment_name}")
        print(f"Status: {exp_info['status']}")
        print(f"{'='*60}\n")
        
        if results_file.exists():
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(format_results(results))
        else:
            print("⏳ Results not yet available")
            if exp_info['status'] == 'running':
                print("   Experiment is still running...")
            elif exp_info['status'] == 'failed':
                print("   Experiment failed. Check logs:")
                print(f"   experiments/retrieval/{experiment_name}/error.log")
    else:
        # Check all completed experiments
        print(f"\n{'='*60}")
        print("EXPERIMENT RESULTS SUMMARY")
        print(f"{'='*60}\n")
        
        completed = [
            name for name, info in status['experiments'].items()
            if info['status'] == 'completed'
        ]
        
        if not completed:
            print("No completed experiments yet.")
            return
        
        for exp_name in sorted(completed):
            results_file = pathlib.Path("experiments/retrieval") / exp_name / "results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    results = json.load(f)
                
                avg = results.get('average', {})
                recall_10 = avg.get('Recall@10', 0)
                ndcg_10 = avg.get('nDCG@10', 0)
                
                print(f"✅ {exp_name}")
                print(f"   Recall@10: {recall_10:.4f} | nDCG@10: {ndcg_10:.4f}")
                print()
            else:
                print(f"⏳ {exp_name} (completed but results not available)")
                print()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Check Experiment Results')
    parser.add_argument('experiment', nargs='?', help='Experiment name (optional, checks all if not provided)')
    
    args = parser.parse_args()
    
    check_experiment_results(args.experiment)

if __name__ == "__main__":
    main()

