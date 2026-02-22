#!/usr/bin/env python3
"""
Resume stopped experiments from their last checkpoint
Reads resume information from stopped_experiments_resume_info.json
"""

import os
import sys
import json
import pathlib
import subprocess
import logging
from datetime import datetime

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

RESUME_INFO_FILE = pathlib.Path("stopped_experiments_resume_info.json")

def load_resume_info():
    """Load resume information from JSON file"""
    try:
        if not RESUME_INFO_FILE.exists():
            logger.error(f"Resume info file not found: {RESUME_INFO_FILE}")
            logger.info("No stopped experiments to resume. Run stop_and_save_resume_info.py first.")
            return None
        
        with RESUME_INFO_FILE.open('r') as f:
            return json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"Error decoding JSON from {RESUME_INFO_FILE}: {e}")
        return None

def build_resume_command(resume_data, gpu_filter=None):
    """Build command to resume an experiment"""
    script_path = resume_data.get('script_path')
    exp_name = resume_data.get('experiment_name')
    gpu = resume_data.get('gpu', '0')
    output_dir = resume_data.get('output_dir')
    
    # Filter by GPU if specified
    if gpu_filter is not None and str(gpu) != str(gpu_filter):
        return None
    
    if not script_path or not exp_name:
        logger.warning(f"Missing script_path or experiment_name for {exp_name}")
        return None
    
    # Check if script exists
    script_path_obj = pathlib.Path(script_path)
    if not script_path_obj.exists():
        logger.warning(f"Script not found: {script_path}")
        return None
    
    # Build command with --resume flag
    cmd_parts = [
        sys.executable,  # python3
        str(script_path),
        '--experiment_name', exp_name,
        '--gpu', str(gpu),
    ]
    
    if output_dir:
        cmd_parts.extend(['--output_dir', output_dir])
    
    # Add --resume flag (most scripts support this)
    cmd_parts.append('--resume')
    
    return cmd_parts

def resume_experiment(resume_data, dry_run=False):
    """Resume a single experiment"""
    exp_name = resume_data.get('experiment_name')
    gpu = resume_data.get('gpu', 'unknown')
    checkpoint_dir = resume_data.get('checkpoint_dir')
    
    logger.info(f"📊 {exp_name} (GPU: {gpu})")
    if checkpoint_dir:
        logger.info(f"   Checkpoint: {checkpoint_dir}")
    
    cmd_parts = build_resume_command(resume_data)
    if not cmd_parts:
        logger.warning(f"   ⚠️  Could not build command, skipping")
        return False
    
    cmd_str = ' '.join(cmd_parts)
    logger.info(f"   Command: {cmd_str}")
    
    if dry_run:
        logger.info(f"   [DRY RUN] Would resume this experiment")
        return True
    
    # Run in background
    try:
        process = subprocess.Popen(
            cmd_parts,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            start_new_session=True
        )
        logger.info(f"   ✅ Started process (PID: {process.pid})")
        return True
    except Exception as e:
        logger.error(f"   ❌ Failed to start: {e}")
        return False

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Resume stopped experiments from checkpoints')
    parser.add_argument('--gpu', type=int, help='Only resume experiments on this GPU')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be resumed without actually resuming')
    parser.add_argument('--list', action='store_true', help='List all stopped experiments and exit')
    args = parser.parse_args()
    
    logger.info("="*80)
    logger.info("RESUME EXPERIMENTS FROM CHECKPOINTS")
    logger.info("="*80)
    logger.info("")
    
    # Load resume info
    resume_info = load_resume_info()
    if not resume_info:
        return
    
    stopped_experiments = resume_info.get('stopped_experiments', [])
    if not stopped_experiments:
        logger.info("No stopped experiments found to resume.")
        return
    
    logger.info(f"Found {len(stopped_experiments)} stopped experiment(s)")
    
    # Filter by GPU if specified
    if args.gpu is not None:
        stopped_experiments = [exp for exp in stopped_experiments if str(exp.get('gpu')) == str(args.gpu)]
        logger.info(f"Filtered to {len(stopped_experiments)} experiment(s) on GPU {args.gpu}")
    
    logger.info("")
    
    if args.list:
        logger.info("Stopped experiments:")
        for exp in stopped_experiments:
            logger.info(f"  - {exp.get('experiment_name')} (GPU: {exp.get('gpu')}, Checkpoint: {exp.get('checkpoint_dir', 'N/A')})")
        return
    
    # Resume experiments
    if args.dry_run:
        logger.info("DRY RUN MODE - No experiments will actually be resumed")
        logger.info("")
    
    resumed_count = 0
    failed_count = 0
    
    for exp_data in stopped_experiments:
        if resume_experiment(exp_data, dry_run=args.dry_run):
            resumed_count += 1
        else:
            failed_count += 1
        logger.info("")
    
    # Summary
    logger.info("="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    if args.dry_run:
        logger.info(f"Would resume: {resumed_count} experiment(s)")
    else:
        logger.info(f"✅ Resumed: {resumed_count} experiment(s)")
        if failed_count > 0:
            logger.info(f"⚠️  Failed: {failed_count} experiment(s)")
    logger.info("="*80)

if __name__ == "__main__":
    main()

