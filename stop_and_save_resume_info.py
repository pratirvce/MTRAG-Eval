#!/usr/bin/env python3
"""
Stop all paused experiments, release GPU memory, and save resume information
This allows experiments to be resumed from their last checkpoint later
"""

import os
import sys
import json
import pathlib
import signal
import psutil
import logging
import subprocess
import re
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

RESUME_INFO_FILE = pathlib.Path("stopped_experiments_resume_info.json")

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

def save_resume_info(resume_data):
    """Save resume information to JSON file"""
    try:
        with RESUME_INFO_FILE.open('w') as f:
            json.dump(resume_data, f, indent=2)
        logger.info(f"✅ Saved resume information to {RESUME_INFO_FILE}")
        return True
    except Exception as e:
        logger.error(f"Error saving resume info: {e}")
        return False

def find_training_processes():
    """Find all training processes"""
    processes = []
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            if 'python' in line and ('train_' in line or 'tier1' in line or 'task_a' in line):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        state = parts[7] if len(parts) > 7 else '?'
                        cmd = ' '.join(parts[10:])
                        
                        # Extract experiment name, GPU, output_dir, and script path
                        exp_match = None
                        gpu_match = None
                        output_dir_match = None
                        script_path_match = None
                        
                        if '--experiment_name' in cmd:
                            match = re.search(r'--experiment_name\s+(\S+)', cmd)
                            if match:
                                exp_match = match.group(1)
                        
                        if '--gpu' in cmd:
                            match = re.search(r'--gpu\s+(\d+)', cmd)
                            if match:
                                gpu_match = match.group(1)
                        
                        if '--output_dir' in cmd:
                            match = re.search(r'--output_dir\s+(\S+)', cmd)
                            if match:
                                output_dir_match = match.group(1)
                        
                        # Extract script path (first argument after python/python3)
                        script_match = re.search(r'python\d?\s+([^\s]+train[^\s]+\.py)', cmd)
                        if script_match:
                            script_path_match = script_match.group(1)
                        
                        if exp_match or pid:
                            exp_name = exp_match or f"pid_{pid}"
                            processes.append({
                                'pid': pid,
                                'state': state,
                                'cmd': cmd,
                                'exp_name': exp_name,
                                'gpu': gpu_match or 'unknown',
                                'output_dir': output_dir_match,
                                'script_path': script_path_match
                            })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.warning(f"Could not scan processes: {e}")
    
    return processes

def find_checkpoint_dir(output_dir):
    """Find the latest checkpoint directory in the output directory"""
    if not output_dir:
        return None
    
    output_path = pathlib.Path(output_dir)
    if not output_path.exists():
        return None
    
    # Look for checkpoint directories
    checkpoint_dirs = []
    for item in output_path.iterdir():
        if item.is_dir() and 'checkpoint' in item.name.lower():
            checkpoint_dirs.append(item)
    
    if checkpoint_dirs:
        # Sort by modification time, return most recent
        checkpoint_dirs.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        return str(checkpoint_dirs[0])
    
    # Check for model files directly in output_dir (final model)
    model_files = list(output_path.glob("*.pt")) + list(output_path.glob("pytorch_model.bin"))
    if model_files:
        return str(output_path)
    
    return None

def stop_process(pid, exp_name):
    """Stop a process using SIGTERM first, then SIGKILL if needed"""
    try:
        process = psutil.Process(pid)
        status = process.status()
        
        if status == psutil.STATUS_ZOMBIE:
            logger.warning(f"  Process {pid} is a zombie, cannot stop")
            return False
        
        # If process is paused (stopped), resume it first so it can respond to signals
        if status == psutil.STATUS_STOPPED:
            logger.info(f"  Process {pid} is paused, resuming before stopping...")
            try:
                process.resume()  # Send SIGCONT
                import time
                time.sleep(0.5)  # Give it a moment to resume
            except Exception as e:
                logger.warning(f"  Could not resume process {pid}: {e}")
        
        # Try graceful termination first
        try:
            process.terminate()  # Send SIGTERM
            logger.info(f"  ⏳ Sent SIGTERM to process {pid}, waiting for graceful shutdown...")
            
            # Wait up to 5 seconds for graceful shutdown
            try:
                process.wait(timeout=5)
                logger.info(f"  ✅ Process {pid} stopped gracefully")
                return True
            except psutil.TimeoutExpired:
                # Force kill if it didn't stop
                logger.warning(f"  ⚠️  Process didn't stop gracefully, sending SIGKILL...")
                try:
                    process.kill()  # Send SIGKILL
                    # Don't wait for kill, just check if it's gone
                    import time
                    time.sleep(0.5)
                    if not process.is_running():
                        logger.info(f"  ✅ Process {pid} force-killed")
                        return True
                    else:
                        logger.warning(f"  ⚠️  Process {pid} still running after SIGKILL")
                        return False
                except psutil.NoSuchProcess:
                    logger.info(f"  ✅ Process {pid} already terminated")
                    return True
        except psutil.NoSuchProcess:
            logger.info(f"  ✅ Process {pid} already terminated")
            return True
            
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"  ⚠️  Could not stop process {pid}: {e}")
        return False

def get_gpu_status():
    """Get GPU utilization and memory status"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', 
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=10
        )
        return result.stdout.strip()
    except Exception as e:
        logger.warning(f"Could not get GPU status: {e}")
        return None

def main():
    logger.info("="*80)
    logger.info("STOPPING ALL EXPERIMENTS AND SAVING RESUME INFORMATION")
    logger.info("="*80)
    logger.info("")
    logger.info("⚠️  This will STOP all paused/running experiments")
    logger.info("   GPU memory will be released")
    logger.info("   Resume information will be saved for later continuation")
    logger.info("")
    
    # Step 1: Find all training processes
    logger.info("Step 1: Finding all training processes...")
    all_processes = find_training_processes()
    
    if not all_processes:
        logger.info("No training processes found.")
        return
    
    logger.info(f"Found {len(all_processes)} training process(es)")
    logger.info("")
    
    # Step 2: Load existing resume info
    resume_info = {
        'timestamp': datetime.now().isoformat(),
        'stopped_experiments': []
    }
    
    if RESUME_INFO_FILE.exists():
        existing_info = load_status_json(RESUME_INFO_FILE)
        if existing_info and 'stopped_experiments' in existing_info:
            resume_info['stopped_experiments'] = existing_info['stopped_experiments']
    
    # Step 3: Stop processes and save resume info
    logger.info("Step 2: Stopping processes and saving resume information...")
    logger.info("")
    
    stopped_count = 0
    failed_stops = []
    
    for proc in all_processes:
        pid = proc['pid']
        exp_name = proc['exp_name']
        gpu = proc['gpu']
        output_dir = proc.get('output_dir')
        script_path = proc.get('script_path')
        
        logger.info(f"📊 {exp_name}")
        logger.info(f"   PID: {pid} | GPU: {gpu} | State: {proc['state']}")
        logger.info(f"   Output Dir: {output_dir or 'N/A'}")
        logger.info(f"   Script: {script_path or 'N/A'}")
        
        # Find checkpoint directory
        checkpoint_dir = find_checkpoint_dir(output_dir) if output_dir else None
        if checkpoint_dir:
            logger.info(f"   Latest Checkpoint: {checkpoint_dir}")
        
        # Stop the process
        if stop_process(pid, exp_name):
            stopped_count += 1
            
            # Save resume information
            resume_data = {
                'experiment_name': exp_name,
                'pid': pid,
                'gpu': gpu,
                'script_path': script_path,
                'output_dir': output_dir,
                'checkpoint_dir': checkpoint_dir,
                'command_template': proc['cmd'],
                'stopped_at': datetime.now().isoformat(),
                'status': 'stopped'
            }
            resume_info['stopped_experiments'].append(resume_data)
            logger.info(f"   ✅ Stopped and saved resume info")
        else:
            failed_stops.append(exp_name)
            logger.info(f"   ⚠️  Failed to stop")
        logger.info("")
    
    # Step 4: Save resume information
    logger.info("Step 3: Saving resume information...")
    save_resume_info(resume_info)
    logger.info("")
    
    # Step 5: Wait a moment and check GPU status
    logger.info("Step 4: Waiting for GPU memory to be released...")
    import time
    time.sleep(3)
    
    logger.info("="*80)
    logger.info("GPU STATUS AFTER STOPPING")
    logger.info("="*80)
    gpu_status = get_gpu_status()
    if gpu_status:
        logger.info("")
        logger.info("GPU | Model | Utilization | Memory Used / Total | Usage %")
        logger.info("-" * 80)
        total_mem_used = 0
        total_mem_free = 0
        for line in gpu_status.split('\n'):
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 5:
                    idx = parts[0]
                    name = parts[1][:25]
                    util = parts[2]
                    mem_used = int(parts[3])
                    mem_total = int(parts[4])
                    mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                    mem_free = mem_total - mem_used
                    total_mem_used += mem_used
                    total_mem_free += mem_free
                    logger.info(f"GPU {idx} | {name:<25} | {util:>3}% | {mem_used:>6,} MB / {mem_total:>6,} MB | {mem_pct:>5.1f}%")
        logger.info("-" * 80)
        logger.info(f"Total: {total_mem_used:,} MB used / {total_mem_free:,} MB free")
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Stopped: {stopped_count} process(es)")
    if failed_stops:
        logger.info(f"⚠️  Failed to stop: {len(failed_stops)} process(es)")
        for exp_name in failed_stops:
            logger.info(f"   - {exp_name}")
    logger.info(f"💾 Resume information saved to: {RESUME_INFO_FILE}")
    logger.info("")
    logger.info("💡 To resume experiments later, run:")
    logger.info("   python3 resume_from_checkpoint.py")
    logger.info("")
    logger.info("💡 Or manually restart with --resume flag:")
    logger.info("   python3 <script> --experiment_name <name> --gpu <gpu> --output_dir <dir> --resume")
    logger.info("="*80)

if __name__ == "__main__":
    main()

