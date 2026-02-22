#!/usr/bin/env python3
"""
Resume all paused experiments using SIGCONT
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

def resume_process(pid):
    """Resume a paused process using SIGCONT"""
    try:
        process = psutil.Process(pid)
        status = process.status()
        if status == psutil.STATUS_STOPPED:
            process.resume()  # Send SIGCONT
            logger.info(f"  ✅ Resumed process {pid} (SIGCONT sent)")
            return True
        elif status in [psutil.STATUS_RUNNING, psutil.STATUS_SLEEPING]:
            logger.info(f"  ℹ️  Process {pid} is already running (status: {status})")
            return True
        else:
            logger.warning(f"  ⚠️  Process {pid} has unexpected status: {status}")
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"  ⚠️  Could not resume process {pid}: {e}")
        return False

def main():
    logger.info("="*80)
    logger.info("RESUMING ALL PAUSED EXPERIMENTS")
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
    
    # Find all paused experiments
    paused_experiments = []
    for exp_name, exp_data in all_experiments.items():
        if exp_data.get('status') == 'paused' and exp_data.get('pid'):
            paused_experiments.append((exp_name, exp_data))
    
    if not paused_experiments:
        logger.info("No paused experiments found to resume.")
        return
    
    logger.info(f"Found {len(paused_experiments)} paused experiment(s) to resume:")
    logger.info("")
    
    resumed_count = 0
    failed_resumes = []
    
    # Resume each experiment
    for exp_name, exp_data in paused_experiments:
        pid = exp_data.get('pid')
        gpu = exp_data.get('gpu', 'N/A')
        paused_time = exp_data.get('paused_time', 'N/A')[:19] if exp_data.get('paused_time') else 'N/A'
        
        logger.info(f"📊 {exp_name}")
        logger.info(f"   PID: {pid} | GPU: {gpu} | Paused: {paused_time}")
        
        if not pid:
            logger.warning(f"   ⚠️  No PID found, skipping")
            failed_resumes.append(exp_name)
            continue
        
        # Check if process still exists
        if not is_process_running(pid):
            logger.warning(f"   ⚠️  Process {pid} no longer exists (may have been killed)")
            failed_resumes.append(exp_name)
            continue
        
        # Resume the process
        if resume_process(pid):
            resumed_count += 1
            # Update status to running
            exp_data['status'] = 'running'
            if 'paused_time' in exp_data:
                exp_data['resumed_time'] = datetime.now().isoformat()
            logger.info(f"   ✅ Successfully resumed")
        else:
            failed_resumes.append(exp_name)
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
    logger.info("RESUME SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Successfully resumed: {resumed_count} experiment(s)")
    if failed_resumes:
        logger.info(f"⚠️  Failed to resume: {len(failed_resumes)} experiment(s)")
        for exp_name in failed_resumes:
            logger.info(f"   - {exp_name}")
    logger.info("")
    logger.info("💡 Experiments have been resumed and will continue from where they paused.")
    logger.info("="*80)

if __name__ == "__main__":
    main()

