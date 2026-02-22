#!/usr/bin/env python3
"""
Show failed experiments and their errors.
"""

import json
import sys
from pathlib import Path
from datetime import datetime

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

def main():
    print("=" * 80)
    print("FAILED EXPERIMENTS REPORT")
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    print()
    
    # Load status files
    fixed_status = load_status_json("auto_fixed_experiments_status.json")
    if not fixed_status:
        fixed_status = load_status_json("fixed_experiments_status.json")
    
    if not fixed_status:
        print("❌ Could not load status files")
        return
    
    # Get experiment_statuses
    if 'experiment_statuses' in fixed_status:
        statuses = fixed_status['experiment_statuses']
    elif 'experiments' in fixed_status:
        statuses = fixed_status['experiments']
    else:
        print("❌ Status file format not recognized")
        return
    
    # Filter failed experiments
    failed = []
    for exp_name, exp_data in statuses.items():
        if exp_data.get('status') == 'failed':
            failed.append((exp_name, exp_data))
    
    if not failed:
        print("✅ No failed experiments found")
        return
    
    # Sort by start time (most recent first)
    failed.sort(key=lambda x: x[1].get('start_time', ''), reverse=True)
    
    print(f"Found {len(failed)} failed experiment(s):\n")
    
    for i, (exp_name, exp_data) in enumerate(failed, 1):
        print(f"{i}. ❌ {exp_name}")
        print("-" * 80)
        
        # Basic info
        gpu = exp_data.get('gpu', 'N/A')
        pid = exp_data.get('pid', 'N/A')
        start_time = exp_data.get('start_time', '')
        retry_count = exp_data.get('retry_count', 0)
        
        print(f"   GPU: {gpu}")
        print(f"   PID: {pid}")
        
        if start_time:
            print(f"   Started: {start_time[:19]}")
            duration = format_duration(start_time)
            print(f"   Runtime before failure: {duration}")
        
        if retry_count > 0:
            print(f"   Retry attempts: {retry_count}")
        
        # Error message
        error = exp_data.get('error')
        if error:
            print(f"\n   Error Message:")
            # Wrap long error messages
            error_lines = error.split('\n')
            if len(error_lines) == 1 and len(error) > 100:
                # Try to split long single-line errors
                words = error.split()
                line = ""
                for word in words:
                    if len(line) + len(word) + 1 > 100:
                        print(f"      {line}")
                        line = word
                    else:
                        line = line + " " + word if line else word
                if line:
                    print(f"      {line}")
            else:
                for line in error_lines[:20]:  # Limit to first 20 lines
                    if line.strip():
                        print(f"      {line}")
                if len(error_lines) > 20:
                    print(f"      ... ({len(error_lines) - 20} more lines)")
        else:
            print(f"\n   Error: (No error message available)")
        
        print()
    
    # Summary by error type
    print("=" * 80)
    print("ERROR SUMMARY")
    print("=" * 80)
    print()
    
    error_types = {}
    for exp_name, exp_data in failed:
        error = exp_data.get('error', 'Unknown error')
        # Categorize error
        error_lower = error.lower()
        if 'cuda out of memory' in error_lower or 'out of memory' in error_lower:
            category = "CUDA Out of Memory"
        elif 'process died' in error_lower or 'died without completing' in error_lower:
            category = "Process Died"
        elif 'does not require grad' in error_lower or 'grad_fn' in error_lower:
            category = "Gradient/Backward Error"
        elif 'not set' in error_lower or 'missing' in error_lower:
            category = "Configuration Error"
        elif 'accelerate' in error_lower or 'requires' in error_lower:
            category = "Dependency Error"
        elif 'index' in error_lower or 'dimension' in error_lower:
            category = "Index/Dimension Error"
        else:
            category = "Other"
        
        if category not in error_types:
            error_types[category] = []
        error_types[category].append(exp_name)
    
    for category, exps in sorted(error_types.items()):
        print(f"{category}: {len(exps)} experiment(s)")
        for exp in exps:
            print(f"  • {exp}")
        print()
    
    print("=" * 80)

if __name__ == "__main__":
    main()

