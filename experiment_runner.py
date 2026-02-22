"""
Robust Experiment Runner with Checkpoint/Resume, Parallel Execution, and Pause/Restart
Supports running multiple experiments across GPUs with full state management
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import pathlib
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import psutil
import torch

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

@dataclass
class ExperimentStatus:
    """Track experiment status and state"""
    experiment_name: str
    status: str  # pending, running, paused, completed, failed
    gpu_id: Optional[int] = None
    pid: Optional[int] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    checkpoint_path: Optional[str] = None
    config_path: Optional[str] = None
    log_path: Optional[str] = None
    error: Optional[str] = None
    progress: Dict = None
    
    def __post_init__(self):
        if self.progress is None:
            self.progress = {}

class ExperimentRunner:
    """Manages experiment execution with checkpoint/resume and parallel execution"""
    
    def __init__(self, experiments_dir: str = "experiments/retrieval", status_file: str = "experiment_status.json"):
        self.experiments_dir = pathlib.Path(experiments_dir)
        self.status_file = pathlib.Path(status_file)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Load existing status
        self.experiments: Dict[str, ExperimentStatus] = {}
        self.load_status()
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Received shutdown signal. Saving state and stopping experiments...")
        self.pause_all()
        self.save_status()
        sys.exit(0)
    
    def load_status(self):
        """Load experiment status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    self.experiments = {
                        name: ExperimentStatus(**status)
                        for name, status in data.items()
                    }
                logger.info(f"Loaded {len(self.experiments)} experiment statuses")
            except Exception as e:
                logger.warning(f"Error loading status: {e}")
                self.experiments = {}
        else:
            self.experiments = {}
    
    def save_status(self):
        """Save experiment status to file"""
        try:
            data = {
                name: asdict(status)
                for name, status in self.experiments.items()
            }
            with open(self.status_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs"""
        if not torch.cuda.is_available():
            return []
        
        num_gpus = torch.cuda.device_count()
        available = []
        
        for gpu_id in range(num_gpus):
            # Check if GPU is being used
            gpu_usage = self._get_gpu_usage(gpu_id)
            if gpu_usage < 0.5:  # Less than 50% utilization
                available.append(gpu_id)
        
        return available
    
    def _get_gpu_usage(self, gpu_id: int) -> float:
        """Get GPU utilization percentage"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True
            )
            for line in result.stdout.strip().split('\n'):
                parts = line.split(', ')
                if len(parts) == 2 and int(parts[0]) == gpu_id:
                    return float(parts[1]) / 100.0
        except:
            pass
        return 0.0
    
    def is_process_running(self, pid: int) -> bool:
        """Check if a process is still running"""
        try:
            process = psutil.Process(pid)
            return process.is_running()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return False
    
    def register_experiment(self, experiment_name: str, config: Dict):
        """Register a new experiment"""
        exp_dir = self.experiments_dir / experiment_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_path = exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Create status
        status = ExperimentStatus(
            experiment_name=experiment_name,
            status="pending",
            config_path=str(config_path),
            log_path=str(exp_dir / "training.log")
        )
        
        self.experiments[experiment_name] = status
        self.save_status()
        logger.info(f"Registered experiment: {experiment_name}")
    
    def start_experiment(self, experiment_name: str, script_path: str, gpu_id: Optional[int] = None) -> bool:
        """Start an experiment on a specific GPU"""
        if experiment_name not in self.experiments:
            logger.error(f"Experiment {experiment_name} not registered")
            return False
        
        status = self.experiments[experiment_name]
        
        # Check if already running
        if status.status == "running":
            if status.pid and self.is_process_running(status.pid):
                logger.info(f"Experiment {experiment_name} already running (PID: {status.pid})")
                return True
            else:
                # Process died, mark as failed
                status.status = "failed"
                status.error = "Process died unexpectedly"
        
        # Get GPU
        if gpu_id is None:
            available_gpus = self.get_available_gpus()
            if not available_gpus:
                logger.warning(f"No available GPUs for {experiment_name}")
                return False
            gpu_id = available_gpus[0]
        
        status.gpu_id = gpu_id
        status.status = "running"
        status.start_time = datetime.now().isoformat()
        
        # Build command
        exp_dir = self.experiments_dir / experiment_name
        log_file = exp_dir / "training.log"
        error_file = exp_dir / "error.log"
        
        cmd = [
            sys.executable,
            script_path,
            "--config", str(status.config_path),
            "--gpu_id", str(gpu_id)
        ]
        
        # Add resume checkpoint if exists
        if status.checkpoint_path and pathlib.Path(status.checkpoint_path).exists():
            cmd.extend(["--resume_from_checkpoint", status.checkpoint_path])
            logger.info(f"Resuming {experiment_name} from checkpoint: {status.checkpoint_path}")
        
        # Run in background
        try:
            with open(log_file, 'a') as log_f, open(error_file, 'a') as err_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=err_f,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)},
                    cwd=pathlib.Path.cwd()
                )
            
            status.pid = process.pid
            self.save_status()
            logger.info(f"Started {experiment_name} on GPU {gpu_id} (PID: {process.pid})")
            return True
        except Exception as e:
            status.status = "failed"
            status.error = str(e)
            self.save_status()
            logger.error(f"Failed to start {experiment_name}: {e}")
            return False
    
    def pause_experiment(self, experiment_name: str) -> bool:
        """Pause an experiment by sending SIGSTOP"""
        if experiment_name not in self.experiments:
            return False
        
        status = self.experiments[experiment_name]
        
        if status.status != "running" or not status.pid:
            logger.warning(f"Experiment {experiment_name} is not running")
            return False
        
        if not self.is_process_running(status.pid):
            status.status = "failed"
            status.error = "Process not found"
            self.save_status()
            return False
        
        try:
            process = psutil.Process(status.pid)
            process.suspend()  # Send SIGSTOP
            status.status = "paused"
            self.save_status()
            logger.info(f"Paused experiment {experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to pause {experiment_name}: {e}")
            return False
    
    def resume_experiment(self, experiment_name: str) -> bool:
        """Resume a paused experiment by sending SIGCONT"""
        if experiment_name not in self.experiments:
            return False
        
        status = self.experiments[experiment_name]
        
        if status.status != "paused" or not status.pid:
            logger.warning(f"Experiment {experiment_name} is not paused")
            return False
        
        if not self.is_process_running(status.pid):
            # Process died, need to restart from checkpoint
            logger.info(f"Process died, restarting {experiment_name} from checkpoint")
            return self.start_experiment(experiment_name, self._get_script_path(experiment_name), status.gpu_id)
        
        try:
            process = psutil.Process(status.pid)
            process.resume()  # Send SIGCONT
            status.status = "running"
            self.save_status()
            logger.info(f"Resumed experiment {experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to resume {experiment_name}: {e}")
            return False
    
    def stop_experiment(self, experiment_name: str) -> bool:
        """Stop an experiment gracefully"""
        if experiment_name not in self.experiments:
            return False
        
        status = self.experiments[experiment_name]
        
        if status.status not in ["running", "paused"] or not status.pid:
            return False
        
        if not self.is_process_running(status.pid):
            status.status = "failed"
            status.error = "Process not found"
            self.save_status()
            return False
        
        try:
            process = psutil.Process(status.pid)
            process.terminate()  # Send SIGTERM
            try:
                process.wait(timeout=10)
            except psutil.TimeoutExpired:
                process.kill()  # Force kill if needed
            
            status.status = "stopped"
            status.end_time = datetime.now().isoformat()
            self.save_status()
            logger.info(f"Stopped experiment {experiment_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop {experiment_name}: {e}")
            return False
    
    def pause_all(self):
        """Pause all running experiments"""
        for name, status in self.experiments.items():
            if status.status == "running":
                self.pause_experiment(name)
    
    def resume_all(self):
        """Resume all paused experiments"""
        for name, status in self.experiments.items():
            if status.status == "paused":
                self.resume_experiment(name)
    
    def update_status(self):
        """Update status of all experiments"""
        for name, status in list(self.experiments.items()):
            if status.status == "running" and status.pid:
                if not self.is_process_running(status.pid):
                    # Check if completed successfully
                    exp_dir = self.experiments_dir / name
                    results_file = exp_dir / "results.json"
                    if results_file.exists():
                        status.status = "completed"
                        status.end_time = datetime.now().isoformat()
                    else:
                        status.status = "failed"
                        status.error = "Process terminated unexpectedly"
                    self.save_status()
    
    def run_parallel(self, max_concurrent: Optional[int] = None):
        """Run pending experiments in parallel, distributing across all GPUs"""
        if max_concurrent is None:
            # Use all GPUs (0-5), not just available ones
            max_concurrent = torch.cuda.device_count() if torch.cuda.is_available() else 1
        
        running = sum(1 for s in self.experiments.values() if s.status == "running")
        pending = [name for name, s in self.experiments.items() if s.status == "pending"]
        
        # Get GPU assignments for running experiments
        used_gpus = set()
        for status in self.experiments.values():
            if status.status == "running" and status.gpu_id is not None:
                used_gpus.add(status.gpu_id)
        
        # Get all GPUs and find unused ones
        all_gpus = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
        available_gpus = [gpu for gpu in all_gpus if gpu not in used_gpus]
        
        gpu_index = 0
        while pending and running < max_concurrent and gpu_index < len(available_gpus):
            experiment_name = pending.pop(0)
            script_path = self._get_script_path(experiment_name)
            
            # Assign to next available GPU
            gpu_id = available_gpus[gpu_index] if available_gpus else None
            
            if script_path and self.start_experiment(experiment_name, script_path, gpu_id):
                running += 1
                gpu_index += 1
            time.sleep(1)  # Small delay between starts
    
    def _get_script_path(self, experiment_name: str) -> Optional[str]:
        """Determine script path based on experiment name"""
        # Map experiment names to scripts
        if "ensemble" in experiment_name:
            return "train_ensemble.py"
        elif "domain_specific" in experiment_name and "hard" in experiment_name:
            return "train_combined_techniques.py"
        elif "reranking" in experiment_name:
            return "train_reranking.py"
        elif "query_expansion" in experiment_name or "query_rewriting" in experiment_name:
            return "train_query_expansion.py"
        elif "hybrid" in experiment_name:
            return "train_hybrid_learned.py"
        elif "domain_specific" in experiment_name:
            return "train_domain_specific_bge.py"
        else:
            return "train_advanced_bge.py"
    
    def get_status_summary(self) -> Dict:
        """Get summary of all experiment statuses"""
        summary = {
            "total": len(self.experiments),
            "pending": 0,
            "running": 0,
            "paused": 0,
            "completed": 0,
            "failed": 0,
            "experiments": {}
        }
        
        for name, status in self.experiments.items():
            summary[status.status] = summary.get(status.status, 0) + 1
            summary["experiments"][name] = {
                "status": status.status,
                "gpu_id": status.gpu_id,
                "pid": status.pid,
                "start_time": status.start_time
            }
        
        return summary

def main():
    parser = argparse.ArgumentParser(description='Experiment Runner with Checkpoint/Resume')
    parser.add_argument('--action', type=str, required=True,
                        choices=['register', 'start', 'pause', 'resume', 'stop', 'status', 'run_parallel', 'update'],
                        help='Action to perform')
    parser.add_argument('--experiment', type=str, help='Experiment name')
    parser.add_argument('--config', type=str, help='Config file path (for register)')
    parser.add_argument('--script', type=str, help='Script path (for start)')
    parser.add_argument('--gpu_id', type=int, help='GPU ID (for start)')
    parser.add_argument('--max_concurrent', type=int, help='Max concurrent experiments (for run_parallel)')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.action == 'register':
        if not args.experiment or not args.config:
            parser.error("--experiment and --config required for register")
        with open(args.config, 'r') as f:
            config = json.load(f)
        runner.register_experiment(args.experiment, config)
    
    elif args.action == 'start':
        if not args.experiment or not args.script:
            parser.error("--experiment and --script required for start")
        runner.start_experiment(args.experiment, args.script, args.gpu_id)
    
    elif args.action == 'pause':
        if not args.experiment:
            parser.error("--experiment required for pause")
        runner.pause_experiment(args.experiment)
    
    elif args.action == 'resume':
        if not args.experiment:
            parser.error("--experiment required for resume")
        runner.resume_experiment(args.experiment)
    
    elif args.action == 'stop':
        if not args.experiment:
            parser.error("--experiment required for stop")
        runner.stop_experiment(args.experiment)
    
    elif args.action == 'status':
        summary = runner.get_status_summary()
        print(json.dumps(summary, indent=2))
    
    elif args.action == 'run_parallel':
        runner.update_status()
        runner.run_parallel(args.max_concurrent)
    
    elif args.action == 'update':
        runner.update_status()
        runner.save_status()
    
    runner.save_status()

if __name__ == "__main__":
    main()

