#!/usr/bin/env python3
"""
Convenient script to manage experiments
Usage:
    python manage_experiments.py start          # Start all pending experiments
    python manage_experiments.py status         # Show status
    python manage_experiments.py pause <name>    # Pause an experiment
    python manage_experiments.py resume <name>   # Resume an experiment
    python manage_experiments.py stop <name>     # Stop an experiment
    python manage_experiments.py update         # Update status
"""

import sys
import subprocess
import json
import pathlib
from experiment_runner import ExperimentRunner

def print_status():
    """Print formatted status"""
    runner = ExperimentRunner()
    runner.update_status()
    summary = runner.get_status_summary()
    
    print("\n" + "="*60)
    print("EXPERIMENT STATUS")
    print("="*60)
    print(f"Total: {summary['total']}")
    print(f"  Pending: {summary['pending']}")
    print(f"  Running: {summary['running']}")
    print(f"  Paused: {summary['paused']}")
    print(f"  Completed: {summary['completed']}")
    print(f"  Failed: {summary['failed']}")
    print("\nExperiments:")
    
    for name, info in summary['experiments'].items():
        status_icon = {
            'pending': '⏳',
            'running': '🟢',
            'paused': '⏸️',
            'completed': '✅',
            'failed': '❌'
        }.get(info['status'], '❓')
        
        print(f"  {status_icon} {name}")
        print(f"      Status: {info['status']}")
        if info['gpu_id'] is not None:
            print(f"      GPU: {info['gpu_id']}")
        if info['pid'] is not None:
            print(f"      PID: {info['pid']}")
        if info['start_time']:
            print(f"      Started: {info['start_time']}")
        print()

def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)
    
    action = sys.argv[1].lower()
    runner = ExperimentRunner()
    
    if action == "start":
        print("Starting all pending experiments...")
        runner.update_status()
        runner.run_parallel()
        print_status()
    
    elif action == "status":
        print_status()
    
    elif action == "pause":
        if len(sys.argv) < 3:
            print("Usage: python manage_experiments.py pause <experiment_name>")
            sys.exit(1)
        exp_name = sys.argv[2]
        if runner.pause_experiment(exp_name):
            print(f"✅ Paused: {exp_name}")
        else:
            print(f"❌ Failed to pause: {exp_name}")
    
    elif action == "resume":
        if len(sys.argv) < 3:
            print("Usage: python manage_experiments.py resume <experiment_name>")
            sys.exit(1)
        exp_name = sys.argv[2]
        if runner.resume_experiment(exp_name):
            print(f"✅ Resumed: {exp_name}")
        else:
            print(f"❌ Failed to resume: {exp_name}")
    
    elif action == "stop":
        if len(sys.argv) < 3:
            print("Usage: python manage_experiments.py stop <experiment_name>")
            sys.exit(1)
        exp_name = sys.argv[2]
        if runner.stop_experiment(exp_name):
            print(f"✅ Stopped: {exp_name}")
        else:
            print(f"❌ Failed to stop: {exp_name}")
    
    elif action == "update":
        print("Updating experiment status...")
        runner.update_status()
        runner.save_status()
        print("✅ Status updated")
    
    elif action == "resume_all":
        print("Resuming all paused experiments...")
        runner.resume_all()
        print_status()
    
    elif action == "pause_all":
        print("Pausing all running experiments...")
        runner.pause_all()
        print_status()
    
    else:
        print(f"Unknown action: {action}")
        print(__doc__)
        sys.exit(1)

if __name__ == "__main__":
    main()

