#!/usr/bin/env python3
"""
Pause all running experiments and clean up zombie/orphaned processes
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

def is_process_running(pid):
    """Check if process exists and is running"""
    try:
        process = psutil.Process(pid)
        return process.is_running()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return False

def get_process_state(pid):
    """Get process state"""
    try:
        process = psutil.Process(pid)
        return process.status()
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return None

def pause_process(pid):
    """Pause a process using SIGSTOP"""
    try:
        process = psutil.Process(pid)
        status = process.status()
        if status == psutil.STATUS_STOPPED:
            logger.info(f"  Process {pid} is already paused")
            return True
        elif status == psutil.STATUS_ZOMBIE:
            logger.warning(f"  Process {pid} is a zombie, cannot pause")
            return False
        process.suspend()  # Send SIGSTOP
        logger.info(f"  ✅ Paused process {pid} (SIGSTOP sent)")
        return True
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.warning(f"  ⚠️  Could not pause process {pid}: {e}")
        return False

def kill_zombie_process(pid):
    """Kill a zombie process by sending SIGKILL to parent"""
    try:
        process = psutil.Process(pid)
        parent = process.parent()
        if parent:
            logger.info(f"  Attempting to reap zombie {pid} by signaling parent {parent.pid}")
            try:
                parent.send_signal(signal.SIGCHLD)
                return True
            except:
                pass
        # If no parent or parent failed, try direct kill (may not work for zombies)
        try:
            os.kill(pid, signal.SIGKILL)
            logger.info(f"  ✅ Sent SIGKILL to zombie process {pid}")
            return True
        except:
            logger.warning(f"  ⚠️  Could not kill zombie {pid}")
            return False
    except (psutil.NoSuchProcess, psutil.AccessDenied) as e:
        logger.info(f"  Process {pid} no longer exists: {e}")
        return True  # Already gone

def find_training_processes():
    """Find all training processes"""
    processes = []
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            if 'python' in line and ('train_' in line or 'tier1' in line):
                parts = line.split()
                if len(parts) > 1:
                    try:
                        pid = int(parts[1])
                        state = parts[7] if len(parts) > 7 else '?'
                        cmd = ' '.join(parts[10:])
                        
                        # Extract experiment name and GPU
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
                        
                        exp_name = exp_match or f"pid_{pid}"
                        processes.append({
                            'pid': pid,
                            'state': state,
                            'cmd': cmd,
                            'exp_name': exp_name,
                            'gpu': gpu_match or 'unknown'
                        })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.warning(f"Could not scan processes: {e}")
    
    return processes

def find_zombie_processes():
    """Find zombie processes"""
    zombies = []
    try:
        result = subprocess.run(['ps', 'aux'], capture_output=True, text=True, timeout=10)
        for line in result.stdout.split('\n'):
            parts = line.split()
            if len(parts) > 7:
                state = parts[7]
                if state == 'Z':  # Zombie state
                    try:
                        pid = int(parts[1])
                        cmd = ' '.join(parts[10:]) if len(parts) > 10 else ''
                        zombies.append({
                            'pid': pid,
                            'cmd': cmd
                        })
                    except (ValueError, IndexError):
                        continue
    except Exception as e:
        logger.warning(f"Could not scan for zombies: {e}")
    
    return zombies

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
    logger.info("PAUSING ALL EXPERIMENTS AND CLEANING UP ZOMBIES")
    logger.info("="*80)
    logger.info("")
    
    # Step 1: Find all training processes
    logger.info("Step 1: Finding all training processes...")
    all_processes = find_training_processes()
    
    if not all_processes:
        logger.info("No training processes found.")
    else:
        logger.info(f"Found {len(all_processes)} training process(es)")
        logger.info("")
    
    # Step 2: Categorize processes
    running_processes = []
    paused_processes = []
    zombie_processes = []
    dead_processes = []
    
    for proc in all_processes:
        pid = proc['pid']
        state = proc['state']
        psutil_state = get_process_state(pid)
        
        if psutil_state is None:
            dead_processes.append(proc)
        elif psutil_state == psutil.STATUS_ZOMBIE:
            zombie_processes.append(proc)
        elif psutil_state == psutil.STATUS_STOPPED:
            paused_processes.append(proc)
        else:
            running_processes.append(proc)
    
    # Step 3: Pause running processes
    if running_processes:
        logger.info("Step 2: Pausing running processes...")
        logger.info("")
        for proc in running_processes:
            logger.info(f"📊 {proc['exp_name']} (PID: {proc['pid']}, GPU: {proc['gpu']}, State: {proc['state']})")
            if pause_process(proc['pid']):
                logger.info(f"   ✅ Successfully paused")
            logger.info("")
    
    # Step 4: Find and clean up zombie processes
    logger.info("Step 3: Finding zombie processes...")
    all_zombies = find_zombie_processes()
    
    # Combine zombies from process scan and dedicated zombie scan
    zombie_pids = set()
    for proc in zombie_processes:
        zombie_pids.add(proc['pid'])
    for z in all_zombies:
        zombie_pids.add(z['pid'])
    
    if zombie_pids:
        logger.info(f"Found {len(zombie_pids)} zombie process(es):")
        logger.info("")
        cleaned = 0
        for pid in zombie_pids:
            logger.info(f"🧟 Zombie PID: {pid}")
            if kill_zombie_process(pid):
                cleaned += 1
            logger.info("")
        logger.info(f"Attempted to clean {cleaned} zombie process(es)")
        logger.info("")
    else:
        logger.info("No zombie processes found.")
        logger.info("")
    
    # Step 5: Report dead/orphaned processes
    if dead_processes:
        logger.info(f"Step 4: Found {len(dead_processes)} dead/orphaned process(es) (no longer exist):")
        for proc in dead_processes:
            logger.info(f"   - PID {proc['pid']}: {proc['exp_name']}")
        logger.info("")
    
    # Step 6: Show GPU status
    logger.info("="*80)
    logger.info("GPU STATUS")
    logger.info("="*80)
    gpu_status = get_gpu_status()
    if gpu_status:
        gpu_lines = gpu_status.split('\n')
        logger.info("")
        logger.info("GPU | Model | Utilization | Memory Used / Total | Usage %")
        logger.info("-" * 80)
        for line in gpu_lines:
            if line.strip():
                parts = line.split(', ')
                if len(parts) >= 5:
                    idx = parts[0]
                    name = parts[1]
                    util = parts[2]
                    mem_used = int(parts[3])
                    mem_total = int(parts[4])
                    mem_pct = (mem_used / mem_total * 100) if mem_total > 0 else 0
                    logger.info(f"GPU {idx} | {name} | {util}% | {mem_used:,} MB / {mem_total:,} MB | {mem_pct:.1f}%")
        logger.info("")
    else:
        logger.info("Could not retrieve GPU status")
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info(f"✅ Paused: {len(running_processes)} running process(es)")
    logger.info(f"⏸️  Already paused: {len(paused_processes)} process(es)")
    logger.info(f"🧟 Cleaned zombies: {len(zombie_pids)} process(es)")
    logger.info(f"💀 Dead/orphaned: {len(dead_processes)} process(es)")
    logger.info("")
    logger.info("💡 Note: GPU memory remains allocated for paused processes")
    logger.info("   This is expected - processes will resume from where they paused")
    logger.info("="*80)

if __name__ == "__main__":
    main()

