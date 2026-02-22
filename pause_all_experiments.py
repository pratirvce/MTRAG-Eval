#!/usr/bin/env python3
"""
Safely pause all running experiments using SIGSTOP
This allows experiments to be resumed later with SIGCONT
"""

import os
import sys
import json
import pathlib
import signal
import psutil
import logging
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def load_status_json(filepath):
    """Load status JSON file"""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {filepath}: {e}")
        return None

def save_status_json(filepath, data):
    """Save status JSON file"""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Error saving JSON to {filepath}: {e}")
        return False

def is_process_running(pid):
    """Check if process is running"""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def pause_process(pid):
    """Pause a process using SIGSTOP (can be resumed with SIGCONT)"""
    try:
        process = psutil.Process(pid)
        if process.status() == psutil.STATUS_STOPPED:
            logger.info(f"  Process {pid} is already paused")
            return True
        process.suspend()  # Send SIGSTOP
        logger.info(f"  ✅ Paused process {pid} (SIGSTOP sent)")
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"  ⚠️  Could not pause process {pid}: {e}")
        return False

def main():
    logger.info("="*80)
    logger.info("PAUSING ALL RUNNING EXPERIMENTS")
    logger.info("="*80)
    logger.info("")
    
    # Load status files
    fixed_status_file = pathlib.Path("fixed_experiments_status.json")
    auto_fixed_status_file = pathlib.Path("auto_fixed_experiments_status.json")
    
    fixed_status = load_status_json(fixed_status_file) if fixed_status_file.exists() else None
    auto_fixed_status = load_status_json(auto_fixed_status_file) if auto_fixed_status_file.exists() else None
    
    # Consolidate all experiment statuses
    all_experiments = {}
    if fixed_status:
        if 'experiment_statuses' in fixed_status:
            all_experiments.update(fixed_status['experiment_statuses'])
        elif 'experiments' in fixed_status:
            all_experiments.update(fixed_status['experiments'])
    
    if auto_fixed_status:
        if 'experiment_statuses' in auto_fixed_status:
            all_experiments.update(auto_fixed_status['experiment_statuses'])
        elif 'experiments' in auto_fixed_status:
            all_experiments.update(auto_fixed_status['experiments'])
    
    # Find all running experiments from status files
    running_experiments = []
    for exp_name, exp_data in all_experiments.items():
        if exp_data.get('status') == 'running' and exp_data.get('pid'):
            running_experiments.append((exp_name, exp_data))
    
    # Also find any training processes that might not be in status files
    # This ensures we catch all running experiments
    import subprocess
    import re
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            if 'python' in line and 'train_' in line and 'tier1' in line:
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        # Check if this PID is already in our list
                        already_tracked = any(exp[1].get('pid') == pid for exp in running_experiments)
                        
                        if not already_tracked:
                            # Extract experiment name and GPU from command
                            cmd = ' '.join(parts[10:])
                            exp_match = None
                            gpu_match = None
                            
                            if '--experiment_name' in cmd:
                                match = re.search(r'--experiment_name\s+(\S+)', cmd)
                                if match:
                                    exp_match = match.group(1)
                            
                            if '--gpu' in cmd:
                                match = re.search(r'--gpu\s+(\d+)', cmd)
                                if match:
                                    gpu_match = match.group(1)
                            
                            if exp_match or pid:
                                exp_name = exp_match or f"experiment_pid_{pid}"
                                logger.info(f"Found running experiment not in status files: {exp_name} (PID: {pid})")
                                running_experiments.append((exp_name, {
                                    'pid': pid, 
                                    'status': 'running', 
                                    'gpu': gpu_match or 'unknown',
                                    'start_time': None
                                }))
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.warning(f"Could not scan for additional processes: {e}")
    
    if not running_experiments:
        logger.info("No running experiments found to pause.")
        return
    
    logger.info(f"Found {len(running_experiments)} running experiment(s) to pause:")
    logger.info("")
    
    paused_count = 0
    failed_pauses = []
    
    # Pause each experiment
    for exp_name, exp_data in running_experiments:
        pid = exp_data.get('pid')
        gpu = exp_data.get('gpu', 'N/A')
        start_time = exp_data.get('start_time', 'N/A')[:19] if exp_data.get('start_time') else 'N/A'
        
        logger.info(f"📊 {exp_name}")
        logger.info(f"   PID: {pid} | GPU: {gpu} | Started: {start_time}")
        
        if not pid:
            logger.warning(f"   ⚠️  No PID found, skipping")
            failed_pauses.append(exp_name)
            continue
        
        # Check if process is still running
        if not is_process_running(pid):
            logger.warning(f"   ⚠️  Process {pid} is not running (may have completed or crashed)")
            failed_pauses.append(exp_name)
            continue
        
        # Pause the process
        if pause_process(pid):
            paused_count += 1
            # Update status to paused
            exp_data['status'] = 'paused'
            exp_data['paused_time'] = datetime.now().isoformat()
            logger.info(f"   ✅ Successfully paused")
        else:
            failed_pauses.append(exp_name)
        logger.info("")
    
    # Save updated status files
    logger.info("="*80)
    logger.info("Saving updated status files...")
    
    if fixed_status:
        if 'experiment_statuses' in fixed_status:
            fixed_status['experiment_statuses'] = {k: v for k, v in all_experiments.items() 
                                                   if k in fixed_status.get('experiment_statuses', {})}
        elif 'experiments' in fixed_status:
            fixed_status['experiments'] = {k: v for k, v in all_experiments.items() 
                                          if k in fixed_status.get('experiments', {})}
        save_status_json(fixed_status_file, fixed_status)
    
    if auto_fixed_status:
        if 'experiment_statuses' in auto_fixed_status:
            auto_fixed_status['experiment_statuses'] = {k: v for k, v in all_experiments.items() 
                                                       if k in auto_fixed_status.get('experiment_statuses', {})}
        elif 'experiments' in auto_fixed_status:
            auto_fixed_status['experiments'] = {k: v for k, v in all_experiments.items() 
                                               if k in auto_fixed_status.get('experiments', {})}
        save_status_json(auto_fixed_status_file, auto_fixed_status)
    
    # Summary
    logger.info("")
    logger.info("="*80)
    logger.info("PAUSE SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully paused: {paused_count} experiment(s)")
    if failed_pauses:
        logger.info(f"⚠️  Failed to pause: {len(failed_pauses)} experiment(s)")
        for exp_name in failed_pauses:
            logger.info(f"   - {exp_name}")
    logger.info("")
    logger.info("💡 To resume experiments later, run:")
    logger.info("   python3 resume_all_experiments.py")
    logger.info("")
    logger.info("💡 Note: Processes are paused with SIGSTOP and can be resumed with SIGCONT")
    logger.info("   All processes will continue from where they paused when resumed.")
    logger.info("="*80)

if __name__ == "__main__":
    main()

