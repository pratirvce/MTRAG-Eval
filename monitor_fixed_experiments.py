#!/usr/bin/env python3
"""
Automated monitoring for fixed cross-attention experiments.
Monitors progress, detects completion, and reports results.
"""

import os
import sys
import time
import json
import subprocess
import pathlib
from datetime import datetime
from typing import Dict, Optional, Tuple

# Experiments to monitor
MONITORED_EXPERIMENTS = [
    "phase8_cross_attention_query_document_fixed",
    "tier1_cross_attention_rerun_fixed"
]

EXPERIMENTS_DIR = pathlib.Path(__file__).parent / "experiments" / "retrieval"
DEFAULT_CHECK_INTERVAL = 300  # Check every 5 minutes
STATUS_FILE = pathlib.Path(__file__).parent / "fixed_experiments_status.json"

def get_process_info(experiment_name: str) -> Optional[Dict]:
    """Get process information for an experiment."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        
        for line in result.stdout.split('\n'):
            if experiment_name in line and 'train_cross_attention' in line:
                parts = line.split()
                if len(parts) >= 11:
                    return {
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'status': 'running'
                    }
        return None
    except Exception as e:
        print(f"Error getting process info: {e}")
        return None

def get_experiment_progress(experiment_name: str) -> Dict:
    """Get current progress of an experiment."""
    exp_dir = EXPERIMENTS_DIR / experiment_name
    
    status = {
        'name': experiment_name,
        'is_running': False,
        'progress': {},
        'last_update': None,
        'log_tail': []
    }
    
    # Check if process is running
    proc_info = get_process_info(experiment_name)
    if proc_info:
        status['is_running'] = True
        status['process'] = proc_info
    
    # Check log file
    log_file = exp_dir / "training.log"
    if log_file.exists():
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                status['log_tail'] = lines[-10:] if len(lines) > 10 else lines
                
                # Extract progress information
                for line in reversed(lines[-50:]):
                    if "Domain" in line and "Results:" in line:
                        # Extract domain and results
                        if "clapnq" in line.lower() or "fiqa" in line.lower() or "govt" in line.lower() or "cloud" in line.lower():
                            status['last_update'] = line.strip()
                            break
                    elif "Processed" in line and "/" in line:
                        # Extract query progress
                        status['last_update'] = line.strip()
                        break
                    elif "Batches:" in line and "%" in line:
                        # Extract batch progress
                        status['last_update'] = line.strip()
                        break
        except Exception as e:
            status['error'] = f"Error reading log: {e}"
    
    # Check checkpoint
    checkpoint_file = exp_dir / "checkpoints" / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
                status['checkpoint'] = checkpoint_data
                
                # Calculate completion
                completed_domains = []
                for domain, domain_data in checkpoint_data.items():
                    if isinstance(domain_data, dict) and 'results' in domain_data:
                        results = domain_data['results']
                        if results.get('Recall@10', 0) > 0 or results.get('nDCG@10', 0) > 0:
                            completed_domains.append(domain)
                            status['progress'][domain] = {
                                'status': 'complete',
                                'recall@10': results.get('Recall@10', 0),
                                'ndcg@10': results.get('nDCG@10', 0)
                            }
                        else:
                            status['progress'][domain] = {'status': 'in_progress'}
                    else:
                        status['progress'][domain] = {'status': 'pending'}
                
                status['completion'] = len(completed_domains) / 4 * 100 if completed_domains else 0
        except Exception as e:
            status['checkpoint_error'] = str(e)
    
    # Check results file
    results_file = exp_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
                status['results'] = results_data
                status['is_complete'] = True
                
                # Extract average scores
                if 'average' in results_data:
                    avg = results_data['average']
                    status['final_scores'] = {
                        'recall@10': avg.get('Recall@10', 0),
                        'ndcg@10': avg.get('nDCG@10', 0)
                    }
        except Exception as e:
            status['results_error'] = str(e)
    
    return status

def format_status_report(statuses: Dict[str, Dict]) -> str:
    """Format status report as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append(f"Fixed Cross-Attention Experiments Monitoring")
    lines.append(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("=" * 80)
    lines.append("")
    
    for exp_name, status in statuses.items():
        lines.append(f"📊 {exp_name}")
        lines.append("-" * 80)
        
        if status.get('is_running'):
            proc = status.get('process', {})
            lines.append(f"  Status: 🟢 RUNNING")
            lines.append(f"  PID: {proc.get('pid', 'N/A')}")
            lines.append(f"  CPU: {proc.get('cpu', 'N/A')}%")
            lines.append(f"  Memory: {proc.get('mem', 'N/A')}%")
        elif status.get('is_complete'):
            lines.append(f"  Status: ✅ COMPLETE")
            scores = status.get('final_scores', {})
            if scores:
                lines.append(f"  Final Recall@10: {scores.get('recall@10', 0):.4f}")
                lines.append(f"  Final nDCG@10: {scores.get('ndcg@10', 0):.4f}")
        else:
            lines.append(f"  Status: ⏸️  STOPPED/UNKNOWN")
        
        completion = status.get('completion', 0)
        if completion > 0:
            lines.append(f"  Progress: {completion:.1f}%")
        
        # Domain progress
        progress = status.get('progress', {})
        if progress:
            lines.append(f"  Domain Progress:")
            for domain, domain_status in progress.items():
                if domain_status.get('status') == 'complete':
                    lines.append(f"    {domain}: ✅ Complete (R@10={domain_status.get('recall@10', 0):.4f}, nDCG@10={domain_status.get('ndcg@10', 0):.4f})")
                elif domain_status.get('status') == 'in_progress':
                    lines.append(f"    {domain}: 🔄 In Progress")
                else:
                    lines.append(f"    {domain}: ⏳ Pending")
        
        # Last update
        if status.get('last_update'):
            lines.append(f"  Last Activity: {status['last_update'][:100]}")
        
        lines.append("")
    
    lines.append("=" * 80)
    return "\n".join(lines)

def check_for_completion(statuses: Dict[str, Dict]) -> Tuple[bool, list]:
    """Check if any experiments have completed."""
    completed = []
    for exp_name, status in statuses.items():
        if status.get('is_complete') and not status.get('is_running'):
            completed.append(exp_name)
    return len(completed) > 0, completed

def check_for_errors(statuses: Dict[str, Dict]) -> Tuple[bool, list]:
    """Check if any experiments have errors."""
    errors = []
    for exp_name, status in statuses.items():
        if 'error' in status or 'results_error' in status or 'checkpoint_error' in status:
            errors.append((exp_name, status.get('error') or status.get('results_error') or status.get('checkpoint_error')))
    return len(errors) > 0, errors

def monitor_once() -> Dict[str, Dict]:
    """Run one monitoring check."""
    statuses = {}
    
    for exp_name in MONITORED_EXPERIMENTS:
        status = get_experiment_progress(exp_name)
        statuses[exp_name] = status
    
    # Save status
    status_data = {
        'timestamp': datetime.now().isoformat(),
        'experiments': statuses
    }
    
    with open(STATUS_FILE, 'w') as f:
        json.dump(status_data, f, indent=2)
    
    return statuses

def monitor_continuous(check_interval: int = DEFAULT_CHECK_INTERVAL):
    """Run continuous monitoring."""
    print("🔍 Starting automated monitoring for fixed cross-attention experiments...")
    print(f"   Monitoring: {', '.join(MONITORED_EXPERIMENTS)}")
    print(f"   Check interval: {check_interval} seconds ({check_interval // 60} minutes)")
    print(f"   Status file: {STATUS_FILE}")
    print("\nPress Ctrl+C to stop\n")
    
    try:
        iteration = 0
        while True:
            iteration += 1
            print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Check #{iteration}")
            print("-" * 80)
            
            statuses = monitor_once()
            report = format_status_report(statuses)
            print(report)
            
            # Check for completions
            has_completed, completed = check_for_completion(statuses)
            if has_completed:
                print(f"\n🎉 COMPLETED EXPERIMENTS DETECTED:")
                for exp_name in completed:
                    status = statuses[exp_name]
                    scores = status.get('final_scores', {})
                    print(f"  ✅ {exp_name}")
                    if scores:
                        print(f"     Recall@10: {scores.get('recall@10', 0):.4f}")
                        print(f"     nDCG@10: {scores.get('ndcg@10', 0):.4f}")
            
            # Check for errors
            has_errors, errors = check_for_errors(statuses)
            if has_errors:
                print(f"\n⚠️  ERRORS DETECTED:")
                for exp_name, error in errors:
                    print(f"  ❌ {exp_name}: {error}")
            
            # Check if all are complete
            all_complete = all(
                statuses[exp].get('is_complete', False) 
                for exp in MONITORED_EXPERIMENTS
            )
            
            if all_complete:
                print("\n" + "=" * 80)
                print("✅ ALL EXPERIMENTS COMPLETED!")
                print("=" * 80)
                print("\nFinal Results Summary:")
                for exp_name in MONITORED_EXPERIMENTS:
                    status = statuses[exp_name]
                    scores = status.get('final_scores', {})
                    print(f"\n  {exp_name}:")
                    if scores:
                        print(f"    Recall@10: {scores.get('recall@10', 0):.4f}")
                        print(f"    nDCG@10: {scores.get('ndcg@10', 0):.4f}")
                print("\nMonitoring will continue. Press Ctrl+C to stop.")
            
            print(f"\n⏳ Next check in {check_interval // 60} minutes...")
            time.sleep(check_interval)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")
        import traceback
        traceback.print_exc()

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Monitor fixed cross-attention experiments')
    parser.add_argument('--once', action='store_true', help='Run check once and exit')
    parser.add_argument('--interval', type=int, default=DEFAULT_CHECK_INTERVAL, 
                       help=f'Check interval in seconds (default: {DEFAULT_CHECK_INTERVAL})')
    
    args = parser.parse_args()
    
    if args.once:
        statuses = monitor_once()
        report = format_status_report(statuses)
        print(report)
        
        # Save report to file
        report_file = pathlib.Path(__file__).parent / "fixed_experiments_status.txt"
        with open(report_file, 'w') as f:
            f.write(report)
        print(f"\n✅ Status report saved to: {report_file}")
    else:
        monitor_continuous(check_interval=args.interval)

if __name__ == "__main__":
    main()

