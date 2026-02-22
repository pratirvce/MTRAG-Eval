"""
Experiment Manager for Phase 6 ACL Experiments
- Parallel execution on all GPUs
- Pause/Resume capability
- Checkpoint management
- Status monitoring
"""

import json
import subprocess
import time
import signal
import sys
import os
import argparse
import pathlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import psutil

class ExperimentManager:
    def __init__(self, config_file: str):
        self.config_file = config_file
        self.experiments = []
        self.running_processes = {}  # exp_name -> (process, gpu_id, start_time)
        self.completed_experiments = set()
        self.failed_experiments = set()
        self.load_config()
        
    def load_config(self):
        """Load experiment configuration"""
        with open(self.config_file, 'r') as f:
            config = json.load(f)
        self.experiments = config.get('experiments', [])
        self.max_gpus = config.get('max_gpus', 6)
        
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            available = []
            for line in result.stdout.strip().split('\n'):
                if line:
                    parts = line.split(', ')
                    gpu_id = int(parts[0])
                    mem_used = int(parts[1])
                    mem_total = int(parts[2])
                    mem_percent = (mem_used / mem_total) * 100
                    # Consider GPU available if memory usage < 80%
                    if mem_percent < 80:
                        available.append(gpu_id)
            return available[:self.max_gpus]
        except Exception as e:
            print(f"Error checking GPUs: {e}")
            return list(range(self.max_gpus))
    
    def is_experiment_running(self, exp_name: str) -> bool:
        """Check if experiment is currently running"""
        return exp_name in self.running_processes
    
    def is_experiment_completed(self, exp_name: str) -> bool:
        """Check if experiment has completed successfully"""
        exp_dir = pathlib.Path("experiments/retrieval") / exp_name
        results_file = exp_dir / "results.json"
        return results_file.exists()
    
    def get_experiment_status(self, exp_name: str) -> str:
        """Get status of experiment"""
        if exp_name in self.completed_experiments:
            return "completed"
        if exp_name in self.failed_experiments:
            return "failed"
        if exp_name in self.running_processes:
            return "running"
        if self.is_experiment_completed(exp_name):
            return "completed"
        return "pending"
    
    def start_experiment(self, exp: Dict, gpu_id: int) -> Optional[subprocess.Popen]:
        """Start a single experiment on specified GPU"""
        exp_name = exp['name']
        script = exp['script']
        config_path = exp['config']
        
        # Check if already completed
        if self.is_experiment_completed(exp_name):
            print(f"⏭️  Skipping {exp_name} (already completed)")
            self.completed_experiments.add(exp_name)
            return None
        
        # Check if config exists
        if not pathlib.Path(config_path).exists():
            print(f"❌ Config not found: {config_path}")
            return None
        
        # Create log directory
        exp_dir = pathlib.Path("experiments/retrieval") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        log_file = exp_dir / "training.log"
        
        # Build command
        cmd = [
            'python', script,
            '--config', config_path,
            '--gpu_id', str(gpu_id)
        ]
        
        if exp.get('resume', True):
            cmd.append('--resume')
        
        print(f"🚀 Starting {exp_name} on GPU {gpu_id}...")
        print(f"   Command: {' '.join(cmd)}")
        
        try:
            # Start process in background
            with open(log_file, 'w') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=pathlib.Path.cwd(),
                    preexec_fn=os.setsid  # Create new process group
                )
            
            self.running_processes[exp_name] = {
                'process': process,
                'gpu_id': gpu_id,
                'start_time': datetime.now(),
                'cmd': ' '.join(cmd)
            }
            
            print(f"   ✅ Started with PID: {process.pid}")
            return process
        
        except Exception as e:
            print(f"   ❌ Failed to start: {e}")
            return None
    
    def check_running_experiments(self):
        """Check status of running experiments"""
        completed = []
        failed = []
        
        for exp_name, info in list(self.running_processes.items()):
            process = info['process']
            gpu_id = info['gpu_id']
            
            # Check if process is still running
            if process.poll() is not None:
                # Process finished
                return_code = process.returncode
                elapsed = datetime.now() - info['start_time']
                
                if return_code == 0:
                    print(f"✅ {exp_name} completed successfully (GPU {gpu_id}, {elapsed})")
                    completed.append(exp_name)
                    self.completed_experiments.add(exp_name)
                else:
                    print(f"❌ {exp_name} failed with return code {return_code} (GPU {gpu_id}, {elapsed})")
                    failed.append(exp_name)
                    self.failed_experiments.add(exp_name)
                
                del self.running_processes[exp_name]
        
        return completed, failed
    
    def stop_experiment(self, exp_name: str, force: bool = False):
        """Stop a running experiment"""
        if exp_name not in self.running_processes:
            print(f"⚠️  Experiment {exp_name} is not running")
            return False
        
        info = self.running_processes[exp_name]
        process = info['process']
        gpu_id = info['gpu_id']
        
        try:
            if force:
                print(f"🛑 Force stopping {exp_name} (PID: {process.pid})...")
                process.kill()
            else:
                print(f"⏸️  Stopping {exp_name} gracefully (PID: {process.pid})...")
                # Send SIGTERM to process group
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            
            # Wait a bit for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if force:
                    process.kill()
                else:
                    print(f"⚠️  Process didn't terminate, killing...")
                    process.kill()
            
            del self.running_processes[exp_name]
            print(f"✅ Stopped {exp_name}")
            return True
        
        except Exception as e:
            print(f"❌ Error stopping {exp_name}: {e}")
            return False
    
    def stop_all_experiments(self, force: bool = False):
        """Stop all running experiments"""
        exp_names = list(self.running_processes.keys())
        for exp_name in exp_names:
            self.stop_experiment(exp_name, force)
    
    def run_experiments(self, max_parallel: Optional[int] = None):
        """Run experiments in parallel on available GPUs"""
        if max_parallel is None:
            max_parallel = self.max_gpus
        
        print(f"\n{'='*60}")
        print(f"Starting Phase 6 ACL Experiments")
        print(f"Max parallel: {max_parallel}")
        print(f"{'='*60}\n")
        
        # Filter experiments
        pending_experiments = [
            exp for exp in self.experiments
            if self.get_experiment_status(exp['name']) == 'pending'
        ]
        
        print(f"Pending experiments: {len(pending_experiments)}")
        print(f"Completed experiments: {len(self.completed_experiments)}")
        print(f"Running experiments: {len(self.running_processes)}\n")
        
        try:
            while pending_experiments or self.running_processes:
                # Check running experiments
                self.check_running_experiments()
                
                # Start new experiments if slots available
                available_gpus = self.get_available_gpus()
                running_count = len(self.running_processes)
                
                while pending_experiments and running_count < max_parallel and available_gpus:
                    # Find next pending experiment
                    exp = pending_experiments.pop(0)
                    exp_name = exp['name']
                    
                    # Skip if already running
                    if self.is_experiment_running(exp_name):
                        continue
                    
                    # Get available GPU
                    gpu_id = available_gpus.pop(0)
                    
                    # Start experiment
                    process = self.start_experiment(exp, gpu_id)
                    if process:
                        running_count += 1
                    else:
                        # Put back if failed to start
                        pending_experiments.insert(0, exp)
                
                # Wait a bit before checking again
                if pending_experiments or self.running_processes:
                    time.sleep(5)
        
        except KeyboardInterrupt:
            print("\n⚠️  Interrupted by user. Stopping all experiments...")
            self.stop_all_experiments(force=False)
            print("\n💾 Experiments stopped. Use --resume to continue.")
        
        print("\n" + "="*60)
        print("Experiment Manager Summary:")
        print("="*60)
        print(f"Completed: {len(self.completed_experiments)}")
        print(f"Failed: {len(self.failed_experiments)}")
        print(f"Still running: {len(self.running_processes)}")
        print("="*60)
    
    def print_status(self):
        """Print status of all experiments"""
        print("\n" + "="*60)
        print("Experiment Status:")
        print("="*60)
        
        for exp in self.experiments:
            exp_name = exp['name']
            status = self.get_experiment_status(exp_name)
            
            if status == "running":
                info = self.running_processes[exp_name]
                elapsed = datetime.now() - info['start_time']
                print(f"🟢 {exp_name:40s} | {status:10s} | GPU {info['gpu_id']} | {elapsed}")
            elif status == "completed":
                print(f"✅ {exp_name:40s} | {status:10s}")
            elif status == "failed":
                print(f"❌ {exp_name:40s} | {status:10s}")
            else:
                print(f"⏳ {exp_name:40s} | {status:10s}")
        
        print("="*60)

def main():
    parser = argparse.ArgumentParser(description='Phase 6 Experiment Manager')
    parser.add_argument('--config', type=str, default='phase6_experiments.json',
                        help='Experiment configuration file')
    parser.add_argument('--status', action='store_true', help='Show status and exit')
    parser.add_argument('--stop', type=str, help='Stop specific experiment')
    parser.add_argument('--stop-all', action='store_true', help='Stop all running experiments')
    parser.add_argument('--force', action='store_true', help='Force stop (kill)')
    parser.add_argument('--max-parallel', type=int, help='Maximum parallel experiments')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(args.config)
    
    if args.status:
        manager.print_status()
    elif args.stop:
        manager.stop_experiment(args.stop, force=args.force)
    elif args.stop_all:
        manager.stop_all_experiments(force=args.force)
    else:
        manager.run_experiments(max_parallel=args.max_parallel)

if __name__ == "__main__":
    main()

