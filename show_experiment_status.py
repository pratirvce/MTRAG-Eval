#!/usr/bin/env python3
"""
Quick status display for running experiments.
"""

import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

def get_running_processes():
    """Get list of currently running experiment processes."""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            check=True
        )
        
        running_experiments = []
        for line in result.stdout.split('\n'):
            if 'train_' in line and 'tier1' in line and 'python' in line:
                parts = line.split()
                if len(parts) >= 13:
                    # Extract experiment name from command
                    cmd = ' '.join(parts[10:])
                    exp_name = None
                    gpu = None
                    
                    # Parse arguments
                    for i, part in enumerate(parts):
                        if part == '--experiment_name' and i + 1 < len(parts):
                            exp_name = parts[i + 1]
                        elif part == '--gpu' and i + 1 < len(parts):
                            gpu = parts[i + 1]
                    
                    # Fallback: try to extract from script name
                    if not exp_name:
                        for part in parts:
                            if 'train_' in part and '.py' in part:
                                script_name = part.split('/')[-1]  # Get just filename
                                exp_name = script_name.replace('train_', '').replace('_tier1.py', '')
                                break
                    
                    running_experiments.append({
                        'pid': parts[1],
                        'cpu': parts[2],
                        'mem': parts[3],
                        'gpu': gpu,
                        'cmd': cmd[:100] if len(cmd) > 100 else cmd,
                        'name': exp_name or 'unknown'
                    })
        
        return running_experiments
    except Exception as e:
        print(f"Error getting processes: {e}", file=sys.stderr)
        return []

def load_status_json(filepath):
    """Load status JSON file."""
    path = Path(filepath)
    if path.exists():
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {filepath}: {e}", file=sys.stderr)
    return None

def format_duration(start_time_str):
    """Format duration from start time."""
    try:
        from datetime import datetime, timezone
        start = datetime.fromisoformat(start_time_str.replace('Z', '+00:00'))
        if start.tzinfo is None:
            start = start.replace(tzinfo=timezone.utc)
        now = datetime.now(timezone.utc)
        delta = now - start
        hours = delta.total_seconds() / 3600
        if hours < 1:
            return f"{int(delta.total_seconds() / 60)}m"
        elif hours < 24:
            return f"{hours:.1f}h"
        else:
            return f"{hours/24:.1f}d"
    except:
        return "unknown"

def get_gpu_status():
    """Get GPU utilization status."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True
        )
        
        gpus = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 5:
                    gpus.append({
                        'index': parts[0],
                        'name': parts[1],
                        'util': parts[2],
                        'mem_used': parts[3],
                        'mem_total': parts[4]
                    })
        return gpus
    except Exception as e:
        print(f"Error getting GPU status: {e}", file=sys.stderr)
        return []

def main():
    print("=" * 80)
    print(f"EXPERIMENT STATUS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Get running processes
    running = get_running_processes()
    
    # Load status files
    fixed_status = load_status_json("auto_fixed_experiments_status.json")
    if not fixed_status:
        fixed_status = load_status_json("fixed_experiments_status.json")
    
    # Get GPU status
    gpus = get_gpu_status()
    
    # Show running experiments
    print("🔄 CURRENTLY RUNNING EXPERIMENTS")
    print("-" * 80)
    
    if running:
        # Match running processes with status file data
        for exp in running:
            exp_name = exp['name']
            print(f"📊 {exp_name}")
            print(f"   PID: {exp['pid']} | GPU: {exp['gpu'] or 'N/A'} | CPU: {exp['cpu']}% | Memory: {exp['mem']}%")
            
            # Check status file for more info
            if fixed_status and 'experiment_statuses' in fixed_status:
                # Try exact match first, then with/without _fixed suffix
                exp_status = (fixed_status['experiment_statuses'].get(exp_name) or 
                            fixed_status['experiment_statuses'].get(exp_name.replace('_fixed', '')) or
                            fixed_status['experiment_statuses'].get(exp_name + '_fixed'))
                
                if exp_status:
                    start_time = exp_status.get('start_time', '')
                    if start_time:
                        duration = format_duration(start_time)
                        print(f"   Started: {start_time[:19]} | Runtime: {duration}")
                    status = exp_status.get('status', 'unknown')
                    if status == 'running':
                        print(f"   Status: 🟢 RUNNING")
                    elif status == 'completed':
                        print(f"   Status: ✅ COMPLETED")
                    elif status == 'failed':
                        error = exp_status.get('error', '')
                        error_short = (error[:70] + '...') if len(error) > 70 else error
                        print(f"   Status: ❌ FAILED")
                        print(f"   Error: {error_short}")
            print()
    else:
        print("   No experiments currently running")
        print()
    
    # Show GPU status
    print("🖥️  GPU STATUS")
    print("-" * 80)
    if gpus:
        print(f"{'GPU':<4} {'Name':<20} {'Util':<6} {'Memory':<15} {'Usage'}")
        print("-" * 80)
        for gpu in gpus:
            mem_used = int(gpu['mem_used'])
            mem_total = int(gpu['mem_total'])
            mem_percent = (mem_used / mem_total * 100) if mem_total > 0 else 0
            bar_length = int(mem_percent / 5)
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"{gpu['index']:<4} {gpu['name'][:18]:<20} {gpu['util']:<6} "
                  f"{mem_used:>6}MB/{mem_total:>6}MB  [{bar}] {mem_percent:.1f}%")
        print()
    else:
        print("   GPU status unavailable")
        print()
    
    # Show status from JSON files
    if fixed_status and 'experiment_statuses' in fixed_status:
        print("📋 DETAILED STATUS FROM STATUS FILES")
        print("-" * 80)
        
        # Group by status
        by_status = {'running': [], 'completed': [], 'failed': []}
        for exp_name, exp_data in fixed_status['experiment_statuses'].items():
            status = exp_data.get('status', 'unknown')
            if status in by_status:
                by_status[status].append((exp_name, exp_data))
        
        # Show running
        if by_status['running']:
            print(f"\n🟢 RUNNING ({len(by_status['running'])}):")
            for exp_name, exp_data in sorted(by_status['running'], key=lambda x: x[1].get('start_time', '')):
                gpu = exp_data.get('gpu', 'N/A')
                pid = exp_data.get('pid', 'N/A')
                start = exp_data.get('start_time', '')[:19] if exp_data.get('start_time') else 'unknown'
                duration = format_duration(exp_data['start_time']) if exp_data.get('start_time') else 'unknown'
                print(f"   • {exp_name}")
                print(f"     GPU: {gpu} | PID: {pid} | Started: {start} | Runtime: {duration}")
        
        # Show completed
        if by_status['completed']:
            print(f"\n✅ COMPLETED ({len(by_status['completed'])}):")
            for exp_name, exp_data in sorted(by_status['completed'], key=lambda x: x[1].get('start_time', ''), reverse=True)[:5]:
                gpu = exp_data.get('gpu', 'N/A')
                print(f"   • {exp_name} (GPU {gpu})")
            if len(by_status['completed']) > 5:
                print(f"   ... and {len(by_status['completed']) - 5} more")
        
        # Show failed
        if by_status['failed']:
            print(f"\n❌ FAILED ({len(by_status['failed'])}):")
            for exp_name, exp_data in sorted(by_status['failed'], key=lambda x: x[1].get('start_time', ''), reverse=True)[:5]:
                gpu = exp_data.get('gpu', 'N/A')
                error = exp_data.get('error', 'Unknown error')
                error_short = error[:60] + '...' if len(error) > 60 else error
                print(f"   • {exp_name} (GPU {gpu})")
                print(f"     Error: {error_short}")
            if len(by_status['failed']) > 5:
                print(f"   ... and {len(by_status['failed']) - 5} more")
        
        print()
    
    print("=" * 80)
    print(f"Summary: {len(running)} experiment(s) currently running")
    print("=" * 80)

if __name__ == "__main__":
    main()

