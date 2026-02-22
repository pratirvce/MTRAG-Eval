"""
Parallel Phase 4 Experiment Runner
Distributes experiments across multiple GPUs for faster execution
"""

import json
import pathlib
import subprocess
import sys
import logging
import time
import signal
import os
import threading
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import torch

# Setup logging
LOG_DIR = pathlib.Path("experiments/retrieval/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = pathlib.Path("experiments/retrieval/phase4_parallel_status.json")
PID_FILE = pathlib.Path("experiments/retrieval/phase4_parallel_runner.pid")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"phase4_parallel_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Import experiment configurations
from run_phase4_experiments import PHASE4_EXPERIMENTS

class GPUManager:
    """Manages GPU allocation and availability."""
    
    def __init__(self):
        self.available_gpus = self.get_available_gpus()
        self.gpu_locks = {gpu: threading.Lock() for gpu in self.available_gpus}
        self.gpu_usage = {gpu: None for gpu in self.available_gpus}  # Track which process uses which GPU
        
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPUs."""
        if not torch.cuda.is_available():
            logging.warning("CUDA not available")
            return []
        
        total_gpus = torch.cuda.device_count()
        available = []
        
        for gpu_id in range(total_gpus):
            # Check if GPU is free (low utilization and memory)
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used', 
                     '--format=csv,noheader,nounits', f'--id={gpu_id}'],
                    capture_output=True,
                    text=True,
                    timeout=5
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(', ')
                    if len(parts) >= 3:
                        idx = int(parts[0])
                        util = int(parts[1])
                        mem = int(parts[2])
                        # Consider GPU free if utilization < 10% and memory < 100 MB
                        if util < 10 and mem < 100:
                            available.append(idx)
            except:
                continue
        
        logging.info(f"Available GPUs: {available}")
        return available
    
    def acquire_gpu(self, timeout: float = 60.0) -> Optional[int]:
        """Acquire a free GPU. Returns GPU ID or None if none available."""
        start_time = time.time()
        while time.time() - start_time < timeout:
            for gpu_id in self.available_gpus:
                if self.gpu_locks[gpu_id].acquire(blocking=False):
                    if self.gpu_usage[gpu_id] is None:
                        self.gpu_usage[gpu_id] = os.getpid()
                        logging.info(f"Acquired GPU {gpu_id}")
                        return gpu_id
                    else:
                        self.gpu_locks[gpu_id].release()
            time.sleep(1)
        return None
    
    def release_gpu(self, gpu_id: int):
        """Release a GPU."""
        if gpu_id in self.gpu_locks:
            self.gpu_usage[gpu_id] = None
            self.gpu_locks[gpu_id].release()
            logging.info(f"Released GPU {gpu_id}")

class ParallelExperimentRunner:
    def __init__(self, max_parallel: int = 5):
        self.gpu_manager = GPUManager()
        self.max_parallel = min(max_parallel, len(self.gpu_manager.available_gpus))
        self.status = self.load_status()
        self.running_experiments = {}  # exp_name -> (process, gpu_id, thread)
        self.completed_experiments = set()
        
    def load_status(self) -> Dict:
        """Load experiment status from file."""
        if STATUS_FILE.exists():
            try:
                with open(STATUS_FILE, 'r') as f:
                    return json.load(f)
            except:
                pass
        return {
            "experiments": {},
            "started_at": None,
            "last_updated": None
        }
    
    def save_status(self):
        """Save experiment status to file."""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(STATUS_FILE, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def update_experiment_status(self, exp_name: str, status: str, gpu_id: Optional[int] = None, details: Optional[Dict] = None):
        """Update status of an experiment."""
        if exp_name not in self.status["experiments"]:
            self.status["experiments"][exp_name] = {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "gpu_id": None,
                "log_file": None,
                "error": None
            }
        
        exp_data = self.status["experiments"][exp_name]
        exp_data["status"] = status
        exp_data["last_updated"] = datetime.now().isoformat()
        
        if status == "running":
            if exp_data["started_at"] is None:
                exp_data["started_at"] = datetime.now().isoformat()
            if gpu_id is not None:
                exp_data["gpu_id"] = gpu_id
        elif status in ["completed", "failed"]:
            exp_data["completed_at"] = datetime.now().isoformat()
            if status == "failed" and details:
                exp_data["error"] = details.get("error", "Unknown error")
        
        if details:
            exp_data.update(details)
        
        self.save_status()
    
    def run_experiment_on_gpu(self, exp_name: str, exp_config: Dict, gpu_id: int):
        """Run a single experiment on a specific GPU."""
        logging.info(f"Starting experiment {exp_name} on GPU {gpu_id}")
        self.update_experiment_status(exp_name, "running", gpu_id=gpu_id)
        
        # Create experiment directory
        exp_dir = pathlib.Path("experiments/retrieval") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_file = exp_dir / "config.json"
        exp_config_copy = exp_config.copy()
        exp_config_copy['config']['gpu_id'] = gpu_id  # Add GPU ID to config
        with open(config_file, 'w') as f:
            json.dump(exp_config_copy['config'], f, indent=2)
        
        # Prepare log files
        log_file = exp_dir / "training.log"
        error_log = exp_dir / "error.log"
        
        self.update_experiment_status(exp_name, "running", gpu_id=gpu_id, details={
            "log_file": str(log_file),
            "config_file": str(config_file)
        })
        
        # Build command
        script = exp_config['script']
        cmd = [
            sys.executable, script,
            "--config", str(config_file)
        ]
        
        if 'domain' in exp_config['config']:
            cmd.extend(["--domain", str(exp_config['config']['domain'])])
        
        # Set CUDA_VISIBLE_DEVICES to use specific GPU
        # This makes the specified GPU appear as GPU 0 to the process
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        # Note: When CUDA_VISIBLE_DEVICES is set, the GPU appears as device 0 to the process
        
        try:
            # Run in background
            with open(log_file, 'w') as log_f, open(error_log, 'w') as err_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=err_f,
                    cwd=pathlib.Path(".").absolute(),
                    env=env
                )
            
            logging.info(f"Experiment {exp_name} started on GPU {gpu_id} (PID: {process.pid})")
            return process
            
        except Exception as e:
            logging.error(f"Error starting experiment {exp_name}: {e}")
            self.update_experiment_status(exp_name, "failed", details={"error": str(e)})
            return None
    
    def monitor_experiment(self, exp_name: str, process: subprocess.Popen, gpu_id: int):
        """Monitor an experiment process."""
        return_code = process.wait()
        
        # Remove from running experiments
        if exp_name in self.running_experiments:
            del self.running_experiments[exp_name]
        
        # Release GPU
        self.gpu_manager.release_gpu(gpu_id)
        
        if return_code == 0:
            logging.info(f"✅ Experiment {exp_name} completed successfully on GPU {gpu_id}")
            self.update_experiment_status(exp_name, "completed")
            self.completed_experiments.add(exp_name)
        else:
            logging.error(f"❌ Experiment {exp_name} failed on GPU {gpu_id} (return code: {return_code})")
            self.update_experiment_status(exp_name, "failed", details={
                "error": f"Process exited with code {return_code}",
                "return_code": return_code
            })
    
    def run_experiments_parallel(self, experiments: Dict):
        """Run experiments in parallel across multiple GPUs."""
        if not self.status.get("started_at"):
            self.status["started_at"] = datetime.now().isoformat()
            self.save_status()
        
        total = len(experiments)
        pending_experiments = {k: v for k, v in experiments.items() 
                              if k not in self.completed_experiments}
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting {total} Phase 4 experiments in parallel")
        logging.info(f"Available GPUs: {self.gpu_manager.available_gpus}")
        logging.info(f"Max parallel: {self.max_parallel}")
        logging.info(f"{'='*60}\n")
        
        while pending_experiments or self.running_experiments:
            # Start new experiments if we have free GPUs and pending experiments
            while len(self.running_experiments) < self.max_parallel and pending_experiments:
                exp_name, exp_config = pending_experiments.popitem()
                
                # Check if already completed
                if exp_name in self.status["experiments"]:
                    current_status = self.status["experiments"][exp_name].get("status", "pending")
                    if current_status == "completed":
                        logging.info(f"Skipping {exp_name} (already completed)")
                        continue
                
                # Acquire GPU
                gpu_id = self.gpu_manager.acquire_gpu(timeout=10.0)
                if gpu_id is None:
                    logging.warning(f"No free GPU available, waiting...")
                    pending_experiments[exp_name] = exp_config  # Put back
                    break
                
                # Start experiment
                process = self.run_experiment_on_gpu(exp_name, exp_config, gpu_id)
                if process:
                    # Start monitoring thread
                    monitor_thread = threading.Thread(
                        target=self.monitor_experiment,
                        args=(exp_name, process, gpu_id),
                        daemon=True
                    )
                    monitor_thread.start()
                    
                    self.running_experiments[exp_name] = (process, gpu_id, monitor_thread)
                    logging.info(f"Running experiments: {list(self.running_experiments.keys())}")
            
            # Wait a bit before checking again
            time.sleep(5)
        
        # Wait for all experiments to complete
        logging.info("Waiting for all experiments to complete...")
        while self.running_experiments:
            time.sleep(10)
            running_names = list(self.running_experiments.keys())
            logging.info(f"Still running: {running_names}")
        
        # Final summary
        completed = len(self.completed_experiments)
        failed = sum(1 for exp in self.status["experiments"].values() 
                    if exp.get("status") == "failed")
        
        logging.info(f"\n{'='*60}")
        logging.info(f"All experiments completed!")
        logging.info(f"Successful: {completed}/{total}")
        logging.info(f"Failed: {failed}/{total}")
        logging.info(f"{'='*60}")
        
        self.status["completed_at"] = datetime.now().isoformat()
        self.save_status()
    
    def get_status_summary(self) -> str:
        """Get a human-readable status summary."""
        summary = []
        summary.append(f"\n{'='*60}")
        summary.append("Phase 4 Parallel Experiments Status")
        summary.append(f"{'='*60}\n")
        
        if self.status.get("started_at"):
            summary.append(f"Started: {self.status['started_at']}")
        if self.status.get("last_updated"):
            summary.append(f"Last Updated: {self.status['last_updated']}")
        summary.append(f"Running: {len(self.running_experiments)} experiments")
        summary.append(f"Available GPUs: {self.gpu_manager.available_gpus}")
        summary.append("")
        
        experiments = self.status.get("experiments", {})
        if not experiments:
            summary.append("No experiments run yet.")
        else:
            summary.append(f"{'Experiment':<40} | {'GPU':<4} | {'Status':<12} | {'Started':<20}")
            summary.append("-" * 85)
            
            for exp_name, exp_data in experiments.items():
                status = exp_data.get("status", "unknown")
                gpu_id = exp_data.get("gpu_id", "N/A")
                started = exp_data.get("started_at", "N/A")[:19] if exp_data.get("started_at") else "N/A"
                
                status_icon = {
                    "completed": "✅",
                    "running": "🔄",
                    "failed": "❌",
                    "pending": "⏳"
                }.get(status, "❓")
                
                summary.append(f"{exp_name:<40} | {str(gpu_id):<4} | {status_icon} {status:<10} | {started:<20}")
        
        summary.append(f"{'='*60}\n")
        return "\n".join(summary)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Phase 4 experiments in parallel across GPUs')
    parser.add_argument('--experiment', type=str, help='Specific experiment to run (or "all")')
    parser.add_argument('--max_parallel', type=int, default=5, help='Maximum parallel experiments (default: 5)')
    parser.add_argument('--status', action='store_true', help='Show current status and exit')
    
    args = parser.parse_args()
    
    runner = ParallelExperimentRunner(max_parallel=args.max_parallel)
    
    if args.status:
        print(runner.get_status_summary())
        return
    
    # Save PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # Handle signals gracefully
    def signal_handler(signum, frame):
        logging.info("Received signal, saving status and exiting...")
        runner.save_status()
        sys.exit(0)
    
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        if args.experiment == "all" or args.experiment is None:
            experiments = PHASE4_EXPERIMENTS
        else:
            if args.experiment not in PHASE4_EXPERIMENTS:
                print(f"Error: Unknown experiment '{args.experiment}'")
                return
            experiments = {args.experiment: PHASE4_EXPERIMENTS[args.experiment]}
        
        runner.run_experiments_parallel(experiments)
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user, saving status...")
        runner.save_status()
    finally:
        if PID_FILE.exists():
            PID_FILE.unlink()

if __name__ == "__main__":
    main()

