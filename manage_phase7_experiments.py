"""
Experiment Manager for Phase 7 Novel Experiments
- Parallel execution on all GPUs
- Pause/Resume capability
- Checkpoint management
- Status monitoring
- Error detection and notification
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
    
    def check_experiment_errors(self, exp_name: str) -> Optional[str]:
        """Check for errors in experiment log file"""
        exp_dir = pathlib.Path("experiments/retrieval") / exp_name
        log_file = exp_dir / "training.log"
        
        if not log_file.exists():
            return None
        
        error_patterns = [
            r'Error:',
            r'Traceback',
            r'Exception:',
            r'IndexError',
            r'KeyError',
            r'ValueError',
            r'TypeError',
            r'AttributeError',
            r'RuntimeError',
            r'CUDA error',
            r'Out of memory',
            r'OOM',
            r'failed',
            r'Failed',
            r'FAILED'
        ]
        
        try:
            with open(log_file, 'r') as f:
                lines = f.readlines()
                # Check last 50 lines for errors
                for line in lines[-50:]:
                    for pattern in error_patterns:
                        if re.search(pattern, line, re.IGNORECASE):
                            return line.strip()
        except Exception:
            pass
        
        return None
    
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
            '--gpu_id', str(gpu_id),
            '--resume'
        ]
        
        print(f"🚀 Starting {exp_name} on GPU {gpu_id}...")
        print(f"   Command: {' '.join(cmd)}")
        print(f"   Log: {log_file}")
        
        # Start process
        try:
            with open(log_file, 'a') as f:
                process = subprocess.Popen(
                    cmd,
                    stdout=f,
                    stderr=subprocess.STDOUT,
                    cwd=pathlib.Path.cwd(),
                    preexec_fn=os.setsid  # Create new process group
                )
            
            self.running_processes[exp_name] = (process, gpu_id, datetime.now())
            print(f"   ✅ Started with PID: {process.pid}")
            return process
        except Exception as e:
            print(f"   ❌ Failed to start: {e}")
            return None
    
    def pause_experiment(self, exp_name: str) -> bool:
        """Pause an experiment gracefully"""
        if exp_name not in self.running_processes:
            print(f"⚠️  Experiment {exp_name} is not running")
            return False
        
        process, gpu_id, start_time = self.running_processes[exp_name]
        
        try:
            # Send SIGTERM for graceful shutdown
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            print(f"⏸️  Pausing {exp_name} (PID: {process.pid})...")
            
            # Wait for graceful shutdown
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                # Force kill if needed
                os.killpg(os.getpgid(process.pid), signal.SIGKILL)
                print(f"   Force killed {exp_name}")
            
            del self.running_processes[exp_name]
            print(f"   ✅ {exp_name} paused (checkpoint should be saved)")
            return True
        except Exception as e:
            print(f"   ❌ Error pausing {exp_name}: {e}")
            return False
    
    def resume_experiment(self, exp: Dict, gpu_id: int) -> Optional[subprocess.Popen]:
        """Resume a paused experiment"""
        return self.start_experiment(exp, gpu_id)  # Resume uses same start logic with --resume
    
    def check_running_experiments(self):
        """Check status of running experiments and detect failures"""
        failed = []
        
        for exp_name, (process, gpu_id, start_time) in list(self.running_processes.items()):
            # Check if process is still running
            if process.poll() is not None:
                # Process finished
                return_code = process.returncode
                
                if return_code == 0:
                    # Completed successfully
                    print(f"✅ {exp_name} completed successfully")
                    self.completed_experiments.add(exp_name)
                    del self.running_processes[exp_name]
                else:
                    # Failed
                    error_msg = self.check_experiment_errors(exp_name)
                    print(f"❌ {exp_name} failed (exit code: {return_code})")
                    if error_msg:
                        print(f"   Error: {error_msg}")
                    self.failed_experiments.add(exp_name)
                    failed.append((exp_name, return_code, error_msg))
                    del self.running_processes[exp_name]
            else:
                # Check for errors in log
                error_msg = self.check_experiment_errors(exp_name)
                if error_msg:
                    print(f"⚠️  Error detected in {exp_name}: {error_msg}")
                    failed.append((exp_name, None, error_msg))
        
        return failed
    
    def run(self, max_parallel: Optional[int] = None):
        """Run all experiments in parallel"""
        if max_parallel is None:
            max_parallel = self.max_gpus
        
        # Sort by priority
        sorted_experiments = sorted(self.experiments, key=lambda x: x.get('priority', 999))
        
        print(f"\n{'='*60}")
        print(f"Phase 7 Experiment Manager")
        print(f"{'='*60}")
        print(f"Total experiments: {len(sorted_experiments)}")
        print(f"Max parallel: {max_parallel}")
        print(f"{'='*60}\n")
        
        # Start experiments
        while sorted_experiments or self.running_processes:
            # Check running experiments
            failed = self.check_running_experiments()
            
            if failed:
                print(f"\n⚠️  {len(failed)} experiment(s) failed:")
                for exp_name, code, error in failed:
                    print(f"   - {exp_name}: {code}, {error}")
                print("\n💡 Check logs for details. You can resume failed experiments.")
            
            # Start new experiments if slots available
            available_gpus = self.get_available_gpus()
            running_count = len(self.running_processes)
            
            if sorted_experiments and running_count < max_parallel and available_gpus:
                # Find next experiment to start
                for exp in sorted_experiments[:]:
                    exp_name = exp['name']
                    status = self.get_experiment_status(exp_name)
                    
                    if status == "completed":
                        sorted_experiments.remove(exp)
                        continue
                    
                    if status == "running":
                        continue
                    
                    # Find available GPU
                    used_gpus = {gpu_id for _, gpu_id, _ in self.running_processes.values()}
                    free_gpus = [gpu for gpu in available_gpus if gpu not in used_gpus]
                    
                    if free_gpus:
                        gpu_id = free_gpus[0]
                        process = self.start_experiment(exp, gpu_id)
                        if process:
                            sorted_experiments.remove(exp)
                        break
            
            # Wait before next check
            time.sleep(10)
        
        print("\n✅ All experiments completed or failed")
        self.print_summary()
    
    def print_summary(self):
        """Print summary of experiment status"""
        print(f"\n{'='*60}")
        print("Experiment Summary")
        print(f"{'='*60}")
        print(f"Completed: {len(self.completed_experiments)}")
        print(f"Failed: {len(self.failed_experiments)}")
        print(f"Running: {len(self.running_processes)}")
        
        if self.completed_experiments:
            print(f"\n✅ Completed:")
            for exp_name in self.completed_experiments:
                print(f"   - {exp_name}")
        
        if self.failed_experiments:
            print(f"\n❌ Failed:")
            for exp_name in self.failed_experiments:
                print(f"   - {exp_name}")
        
        if self.running_processes:
            print(f"\n🔄 Running:")
            for exp_name, (_, gpu_id, start_time) in self.running_processes.items():
                print(f"   - {exp_name} (GPU {gpu_id})")

def main():
    parser = argparse.ArgumentParser(description='Phase 7 Experiment Manager')
    parser.add_argument('--config', type=str, default='phase7_experiments.json',
                       help='Path to experiment config JSON')
    parser.add_argument('--max-parallel', type=int, default=None,
                       help='Maximum parallel experiments')
    parser.add_argument('--pause', type=str, default=None,
                       help='Pause specific experiment by name')
    parser.add_argument('--resume', type=str, default=None,
                       help='Resume specific experiment by name')
    parser.add_argument('--status', action='store_true',
                       help='Show status of all experiments')
    parser.add_argument('--stop-all', action='store_true',
                       help='Stop all running experiments')
    
    args = parser.parse_args()
    
    manager = ExperimentManager(args.config)
    
    if args.status:
        manager.print_summary()
        return
    
    if args.pause:
        manager.pause_experiment(args.pause)
        return
    
    if args.resume:
        # Find experiment config
        exp_config = None
        for exp in manager.experiments:
            if exp['name'] == args.resume:
                exp_config = exp
                break
        
        if exp_config:
            available_gpus = manager.get_available_gpus()
            if available_gpus:
                manager.resume_experiment(exp_config, available_gpus[0])
            else:
                print("❌ No available GPUs")
        else:
            print(f"❌ Experiment {args.resume} not found")
        return
    
    if args.stop_all:
        for exp_name in list(manager.running_processes.keys()):
            manager.pause_experiment(exp_name)
        return
    
    # Run all experiments
    try:
        manager.run(args.max_parallel)
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        print("Pausing all running experiments...")
        for exp_name in list(manager.running_processes.keys()):
            manager.pause_experiment(exp_name)
        manager.print_summary()

if __name__ == "__main__":
    import re
    main()

