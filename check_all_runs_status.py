#!/usr/bin/env python3
"""
Script to check the status of all experiment runs.
Shows which runs are complete, in progress, pending, or have errors.
"""

import json
import pathlib
import subprocess
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

EXPERIMENTS_DIR = pathlib.Path(__file__).parent / "experiments" / "retrieval"

def get_running_processes() -> Dict[str, Dict]:
    """Get information about currently running training/evaluation processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        processes = {}
        for line in result.stdout.split('\n'):
            if 'train_' in line or 'evaluate_' in line:
                parts = line.split()
                if len(parts) > 11:
                    pid = parts[1]
                    cpu = parts[2]
                    mem = parts[3]
                    cmd = ' '.join(parts[10:])
                    
                    # Try to extract experiment name from command
                    exp_name_found = None
                    
                    # Method 1: Check for --experiment_name flag
                    if '--experiment_name' in cmd:
                        idx = cmd.find('--experiment_name')
                        rest = cmd[idx+len('--experiment_name'):].strip()
                        exp_name_found = rest.split()[0] if rest.split() else None
                    
                    # Method 2: Extract from --output_dir path
                    if not exp_name_found and '--output_dir' in cmd:
                        idx = cmd.find('--output_dir')
                        rest = cmd[idx+len('--output_dir'):].strip()
                        if rest.split():
                            output_path = rest.split()[0]
                            if 'experiments/retrieval/' in output_path:
                                exp_name_found = output_path.split('experiments/retrieval/')[-1].rstrip('/')
                    
                    # Method 3: Extract from path in command
                    if not exp_name_found and 'experiments/retrieval/' in cmd:
                        parts_cmd = cmd.split()
                        for part in parts_cmd:
                            if 'experiments/retrieval/' in part:
                                exp_name_found = part.split('experiments/retrieval/')[-1].rstrip('/').rstrip('"').rstrip("'")
                                # Remove any trailing flags or parameters
                                exp_name_found = exp_name_found.split()[0] if ' ' in exp_name_found else exp_name_found
                                break
                    
                    if exp_name_found:
                        processes[exp_name_found] = {
                            'pid': pid,
                            'cpu': cpu,
                            'mem': mem,
                            'command': cmd[:100] + '...' if len(cmd) > 100 else cmd
                        }
        return processes
    except Exception as e:
        print(f"Error getting processes: {e}")
        return {}

def check_checkpoint(exp_dir: pathlib.Path) -> Optional[Dict]:
    """Check checkpoint file to see progress."""
    checkpoint_file = exp_dir / "checkpoints" / "checkpoint.json"
    if checkpoint_file.exists():
        try:
            with open(checkpoint_file, 'r') as f:
                checkpoint_data = json.load(f)
            return checkpoint_data
        except Exception:
            return None
    return None

def check_results(exp_dir: pathlib.Path) -> Optional[Dict]:
    """Check results file."""
    results_file = exp_dir / "results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            return results_data
        except Exception:
            return None
    return None

def get_domains_status(checkpoint_data: Dict) -> Dict[str, str]:
    """Determine status of each domain from checkpoint."""
    if not checkpoint_data:
        return {}
    
    domains_status = {}
    all_domains = {'clapnq', 'fiqa', 'govt', 'cloud'}
    
    for domain in all_domains:
        if domain in checkpoint_data:
            domain_info = checkpoint_data[domain]
            results = domain_info.get('results', {})
            # Check if domain has non-zero results (completed)
            if results.get('Recall@10', 0) > 0 or results.get('nDCG@10', 0) > 0:
                domains_status[domain] = '✅ Complete'
            else:
                domains_status[domain] = '🔄 In Progress'
        else:
            domains_status[domain] = '⏳ Pending'
    
    return domains_status

def get_experiment_status(exp_dir: pathlib.Path, exp_name: str, running_processes: Dict) -> Dict:
    """Get comprehensive status for an experiment."""
    checkpoint_data = check_checkpoint(exp_dir)
    results_data = check_results(exp_dir)
    
    status = {
        'name': exp_name,
        'has_checkpoint': checkpoint_data is not None,
        'has_results': results_data is not None,
        'is_running': exp_name in running_processes,
        'domains_status': {},
        'completion': 0,
        'last_update': None
    }
    
    if checkpoint_data:
        domains_status = get_domains_status(checkpoint_data)
        status['domains_status'] = domains_status
        
        # Calculate completion percentage
        completed = sum(1 for s in domains_status.values() if '✅' in s)
        status['completion'] = int((completed / len(domains_status)) * 100) if domains_status else 0
        
        # Get last update timestamp
        timestamps = []
        for domain_info in checkpoint_data.values():
            if isinstance(domain_info, dict) and 'timestamp' in domain_info:
                timestamps.append(domain_info['timestamp'])
        if timestamps:
            status['last_update'] = max(timestamps)
    
    if results_data:
        # If results exist, check if all domains are complete
        if 'domains' in results_data:
            all_complete = all(
                results_data['domains'][d].get('Recall@10', 0) > 0 or 
                results_data['domains'][d].get('nDCG@10', 0) > 0
                for d in ['clapnq', 'fiqa', 'govt', 'cloud']
                if d in results_data['domains']
            )
            if all_complete:
                status['completion'] = 100
    
    if exp_name in running_processes:
        status['process_info'] = running_processes[exp_name]
    
    return status

def format_status_table(statuses: List[Dict]) -> str:
    """Format statuses into a readable table."""
    lines = []
    lines.append("=" * 120)
    lines.append(f"{'Experiment Name':<50} {'Status':<15} {'Progress':<10} {'Last Update':<20}")
    lines.append("=" * 120)
    
    for status in sorted(statuses, key=lambda x: x['name']):
        name = status['name'][:48]
        
        # Determine overall status
        if status['completion'] == 100:
            overall_status = "✅ Complete"
        elif status['is_running']:
            overall_status = "🔄 Running"
        elif status['has_checkpoint'] and status['completion'] > 0:
            overall_status = f"🔄 {status['completion']}%"
        elif status['has_checkpoint']:
            overall_status = "⏸️  Started"
        else:
            overall_status = "⏳ Pending"
        
        # Format progress
        progress = f"{status['completion']}%"
        
        # Format last update
        last_update = status['last_update'] or "N/A"
        if last_update != "N/A":
            try:
                dt = datetime.fromisoformat(last_update.replace('Z', '+00:00'))
                last_update = dt.strftime("%Y-%m-%d %H:%M")
            except:
                pass
        
        lines.append(f"{name:<50} {overall_status:<15} {progress:<10} {last_update:<20}")
    
    lines.append("=" * 120)
    return "\n".join(lines)

def get_detailed_status(status: Dict) -> str:
    """Get detailed status for an experiment."""
    lines = []
    lines.append(f"\n{'='*80}")
    lines.append(f"Experiment: {status['name']}")
    lines.append(f"{'='*80}")
    
    if status['is_running']:
        proc = status['process_info']
        lines.append(f"🔄 Status: RUNNING")
        lines.append(f"   PID: {proc['pid']}")
        lines.append(f"   CPU: {proc['cpu']}%")
        lines.append(f"   Memory: {proc['mem']}%")
    
    lines.append(f"Progress: {status['completion']}%")
    lines.append(f"Checkpoint: {'✅' if status['has_checkpoint'] else '❌'}")
    lines.append(f"Results: {'✅' if status['has_results'] else '❌'}")
    
    if status['domains_status']:
        lines.append("\nDomain Status:")
        for domain, domain_status in status['domains_status'].items():
            lines.append(f"  {domain}: {domain_status}")
    
    if status['last_update']:
        lines.append(f"\nLast Update: {status['last_update']}")
    
    return "\n".join(lines)

def main():
    """Main function to check all runs status."""
    import sys
    
    show_details = '--details' in sys.argv or '-d' in sys.argv
    
    print("🔍 Checking status of all experiment runs...")
    print()
    
    # Get running processes
    running_processes = get_running_processes()
    
    # Find all experiment directories
    if not EXPERIMENTS_DIR.exists():
        print(f"❌ Experiments directory not found: {EXPERIMENTS_DIR}")
        return
    
    experiment_dirs = [d for d in EXPERIMENTS_DIR.iterdir() if d.is_dir()]
    
    print(f"Found {len(experiment_dirs)} experiment directories")
    print()
    
    # Get status for each experiment
    all_statuses = []
    for exp_dir in experiment_dirs:
        exp_name = exp_dir.name
        status = get_experiment_status(exp_dir, exp_name, running_processes)
        all_statuses.append(status)
    
    # Print summary table
    print(format_status_table(all_statuses))
    
    # Summary statistics
    print()
    print("📊 Summary Statistics:")
    print(f"  Total experiments: {len(all_statuses)}")
    print(f"  ✅ Complete: {sum(1 for s in all_statuses if s['completion'] == 100)}")
    print(f"  🔄 Running: {sum(1 for s in all_statuses if s['is_running'])}")
    print(f"  🔄 In Progress: {sum(1 for s in all_statuses if s['has_checkpoint'] and s['completion'] < 100 and not s['is_running'])}")
    print(f"  ⏳ Pending: {sum(1 for s in all_statuses if not s['has_checkpoint'] and not s['is_running'])}")
    
    # Show currently running experiments
    running = [s for s in all_statuses if s['is_running']]
    if running:
        print()
        print("🔄 Currently Running Experiments:")
        for status in running:
            print(f"  - {status['name']} (PID: {status['process_info']['pid']}, CPU: {status['process_info']['cpu']}%, MEM: {status['process_info']['mem']}%)")
    
    # Show in-progress experiments with details
    in_progress = [s for s in all_statuses if s['has_checkpoint'] and s['completion'] < 100 and s['completion'] > 0 and not s['is_running']]
    if in_progress:
        print()
        print("🔄 In Progress Experiments (with partial completion):")
        for status in in_progress:
            print(f"\n  📌 {status['name']} ({status['completion']}% complete)")
            if status['domains_status']:
                for domain, domain_status in status['domains_status'].items():
                    print(f"      {domain}: {domain_status}")
    
    # Show detailed status for specific experiments if requested
    if show_details:
        print("\n" + "="*80)
        print("📋 Detailed Status for All Experiments:")
        print("="*80)
        for status in sorted(all_statuses, key=lambda x: (x['completion'] != 100, x['is_running'], x['name'])):
            print(get_detailed_status(status))

if __name__ == "__main__":
    main()

