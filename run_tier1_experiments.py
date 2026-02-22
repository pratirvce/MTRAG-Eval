#!/usr/bin/env python3
"""
Tier 1 Experiments Master Runner
Runs all 5 Tier 1 experiments in parallel with resume capability and email notifications
"""

import os
import sys
import json
import time
import signal
import subprocess
import argparse
import pathlib
import logging
import smtplib
from datetime import datetime
from typing import Dict, List, Optional
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
try:
    import torch
except ImportError:
    torch = None
    logging.warning("PyTorch not found. GPU detection will be limited.")

try:
    import psutil
except ImportError:
    psutil = None
    logging.warning("psutil not found. Process monitoring will be limited.")

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Email configuration
EMAIL_TO = "pratirvce@gmail.com"
EMAIL_FROM = "pratirvce@gmail.com"  # Update with your SMTP server
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")  # Set via environment variable

# Tier 1 Experiments Configuration
TIER1_EXPERIMENTS = [
    {
        "name": "tier1_cross_encoder_finetuned",
        "priority": 1,
        "script": "train_cross_encoder_finetuned_tier1.py",
        "expected_ndcg": "0.49-0.52",
        "time_estimate": "3-5 days",
        "gpu": None,  # Will be assigned
        "status": "pending"
    },
    {
        "name": "tier1_cross_attention_query_document",
        "priority": 2,
        "script": "train_cross_attention_tier1.py",
        "expected_ndcg": "0.49-0.52",
        "time_estimate": "4-6 days",
        "gpu": None,
        "status": "pending"
    },
    {
        "name": "tier1_hierarchical_multigranularity",
        "priority": 3,
        "script": "train_hierarchical_multigranularity_tier1.py",
        "expected_ndcg": "0.49-0.52",
        "time_estimate": "3-4 days",
        "gpu": None,
        "status": "pending"
    },
    {
        "name": "tier1_iterative_refinement_improved",
        "priority": 4,
        "script": "train_iterative_refinement_improved_tier1.py",
        "expected_ndcg": "0.50-0.54",
        "time_estimate": "3-4 days",
        "gpu": None,
        "status": "pending"
    },
    {
        "name": "tier1_contrastive_learning",
        "priority": 5,
        "script": "train_contrastive_learning_tier1.py",
        "expected_ndcg": "0.49-0.52",
        "time_estimate": "5-7 days",
        "gpu": None,
        "status": "pending"
    }
]

class Tier1ExperimentRunner:
    """Manages Tier 1 experiments with resume, parallel execution, and email notifications"""
    
    def __init__(self, experiments_dir: str = "experiments/retrieval", status_file: str = "tier1_experiments_status.json"):
        self.experiments_dir = pathlib.Path(experiments_dir)
        self.status_file = pathlib.Path(status_file)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        self.status_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing status
        self.experiments_status = {}
        self.load_status()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info("Received shutdown signal. Saving state...")
        self.save_status()
        sys.exit(0)
    
    def load_status(self):
        """Load experiment status from file"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    self.experiments_status = json.load(f)
                logger.info(f"Loaded status for {len(self.experiments_status)} experiments")
            except Exception as e:
                logger.warning(f"Error loading status: {e}")
                self.experiments_status = {}
        else:
            self.experiments_status = {}
    
    def save_status(self):
        """Save experiment status to file"""
        try:
            with open(self.status_file, 'w') as f:
                json.dump(self.experiments_status, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving status: {e}")
    
    def get_available_gpus(self) -> List[int]:
        """Get list of available GPUs"""
        if torch is None:
            # Fallback: use nvidia-smi to detect GPUs
            try:
                result = subprocess.run(['nvidia-smi', '--list-gpus'], 
                                      capture_output=True, text=True)
                if result.returncode == 0:
                    gpu_count = len(result.stdout.strip().split('\n'))
                    return list(range(gpu_count))
            except:
                pass
            return []
        
        if not torch.cuda.is_available():
            return []
        
        available = []
        for i in range(torch.cuda.device_count()):
            # Check if GPU is being used
            try:
                torch.cuda.set_device(i)
                # Try to allocate small tensor to check availability
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                torch.cuda.empty_cache()
                available.append(i)
            except:
                pass
        
        return available
    
    def send_email_notification(self, subject: str, body: str, is_error: bool = False):
        """Send email notification"""
        if not EMAIL_PASSWORD:
            logger.warning("EMAIL_PASSWORD not set. Skipping email notification.")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_FROM
            msg['To'] = EMAIL_TO
            msg['Subject'] = f"[Tier1 Experiments] {subject}"
            
            if is_error:
                body = f"❌ ERROR ALERT ❌\n\n{body}"
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def check_experiment_status(self, exp_name: str) -> Dict:
        """Check if experiment is running or completed"""
        exp_dir = self.experiments_dir / exp_name
        
        # Check if results exist
        results_file = exp_dir / "results.json"
        if results_file.exists():
            try:
                with open(results_file, 'r') as f:
                    results = json.load(f)
                    return {"status": "completed", "results": results}
            except:
                pass
        
        # Check if process is running
        if psutil:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    cmdline = ' '.join(proc.info['cmdline'] or [])
                    if exp_name in cmdline and 'python' in cmdline.lower():
                        return {"status": "running", "pid": proc.info['pid']}
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
        else:
            # Fallback: check via subprocess
            try:
                result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
                for line in result.stdout.split('\n'):
                    if exp_name in line and 'python' in line.lower():
                        parts = line.split()
                        if len(parts) > 1:
                            return {"status": "running", "pid": int(parts[1])}
            except:
                pass
        
        # Check checkpoint
        checkpoint_file = exp_dir / "checkpoint.json"
        if checkpoint_file.exists():
            try:
                with open(checkpoint_file, 'r') as f:
                    checkpoint = json.load(f)
                    return {"status": "paused", "checkpoint": checkpoint}
            except:
                pass
        
        return {"status": "pending"}
    
    def start_experiment(self, exp_config: Dict, gpu_id: int, resume: bool = False):
        """Start an experiment on specified GPU"""
        exp_name = exp_config["name"]
        script_name = exp_config["script"]
        exp_dir = self.experiments_dir / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Check if script exists
        script_path = pathlib.Path(script_name)
        if not script_path.exists():
            logger.error(f"Script not found: {script_name}")
            self.send_email_notification(
                f"Experiment {exp_name} Failed to Start",
                f"Script {script_name} not found. Please check the implementation.",
                is_error=True
            )
            return None
        
        # Prepare command
        log_file = exp_dir / "training.log"
        error_file = exp_dir / "error.log"
        
        cmd = [
            sys.executable, str(script_path),
            "--experiment_name", exp_name,
            "--gpu", str(gpu_id),
            "--output_dir", str(exp_dir),
            "--resume" if resume else "--no-resume"
        ]
        
        # Start process
        try:
            with open(log_file, 'w') as log_f, open(error_file, 'w') as err_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=err_f,
                    env={**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
                )
            
            # Update status
            self.experiments_status[exp_name] = {
                "status": "running",
                "gpu": gpu_id,
                "pid": process.pid,
                "start_time": datetime.now().isoformat(),
                "priority": exp_config["priority"],
                "script": script_name
            }
            self.save_status()
            
            logger.info(f"Started {exp_name} on GPU {gpu_id} (PID: {process.pid})")
            
            # Send notification
            self.send_email_notification(
                f"Experiment {exp_name} Started",
                f"Experiment {exp_name} has started on GPU {gpu_id}.\n"
                f"Expected nDCG@10: {exp_config['expected_ndcg']}\n"
                f"Time estimate: {exp_config['time_estimate']}\n"
                f"Log file: {log_file}"
            )
            
            return process
        except Exception as e:
            logger.error(f"Failed to start {exp_name}: {e}")
            self.send_email_notification(
                f"Experiment {exp_name} Failed to Start",
                f"Error: {str(e)}\n\nCommand: {' '.join(cmd)}",
                is_error=True
            )
            return None
    
    def monitor_experiments(self):
        """Monitor running experiments and detect failures"""
        while True:
            for exp_name, status in list(self.experiments_status.items()):
                if status.get("status") == "running":
                    pid = status.get("pid")
                    
                    # Check if process is still running
                    pid_exists = False
                    if psutil:
                        pid_exists = psutil.pid_exists(pid)
                    else:
                        # Fallback check
                        try:
                            subprocess.run(['kill', '-0', str(pid)], 
                                         capture_output=True, check=True)
                            pid_exists = True
                        except:
                            pid_exists = False
                    
                    if pid and not pid_exists:
                        # Process ended, check if it completed successfully
                        exp_dir = self.experiments_dir / exp_name
                        results_file = exp_dir / "results.json"
                        error_file = exp_dir / "error.log"
                        
                        if results_file.exists():
                            # Completed successfully
                            status["status"] = "completed"
                            status["end_time"] = datetime.now().isoformat()
                            logger.info(f"Experiment {exp_name} completed successfully")
                            
                            try:
                                with open(results_file, 'r') as f:
                                    results = json.load(f)
                                    avg_ndcg = results.get("average", {}).get("nDCG@10", 0)
                                    
                                self.send_email_notification(
                                    f"Experiment {exp_name} Completed Successfully",
                                    f"Experiment {exp_name} has completed.\n"
                                    f"Average nDCG@10: {avg_ndcg:.4f}\n"
                                    f"Results: {results_file}"
                                )
                            except Exception as e:
                                logger.error(f"Error reading results: {e}")
                        else:
                            # Failed
                            status["status"] = "failed"
                            status["end_time"] = datetime.now().isoformat()
                            
                            error_msg = "Unknown error"
                            if error_file.exists():
                                try:
                                    with open(error_file, 'r') as f:
                                        error_msg = f.read()[-2000:]  # Last 2000 chars
                                except:
                                    pass
                            
                            logger.error(f"Experiment {exp_name} failed")
                            self.send_email_notification(
                                f"Experiment {exp_name} Failed",
                                f"Experiment {exp_name} has failed.\n\n"
                                f"Error log (last 2000 chars):\n{error_msg}\n\n"
                                f"Full error log: {error_file}",
                                is_error=True
                            )
                        
                        self.save_status()
            
            time.sleep(60)  # Check every minute
    
    def run_all_experiments(self, max_parallel: int = 6, resume: bool = True):
        """Run all Tier 1 experiments in parallel"""
        logger.info("Starting Tier 1 experiments...")
        
        # Sort by priority
        experiments = sorted(TIER1_EXPERIMENTS, key=lambda x: x["priority"])
        
        # Get available GPUs
        available_gpus = self.get_available_gpus()
        if not available_gpus:
            logger.error("No available GPUs found!")
            self.send_email_notification(
                "Tier 1 Experiments Failed to Start",
                "No available GPUs found. Please check GPU availability.",
                is_error=True
            )
            return
        
        logger.info(f"Available GPUs: {available_gpus}")
        
        # Start experiments
        running_processes = {}
        gpu_assignments = {}
        
        for exp_config in experiments:
            exp_name = exp_config["name"]
            
            # Check if already completed
            status = self.check_experiment_status(exp_name)
            if status["status"] == "completed":
                logger.info(f"Experiment {exp_name} already completed, skipping")
                continue
            
            # Wait for available GPU
            while len(running_processes) >= min(max_parallel, len(available_gpus)):
                time.sleep(10)
                # Check for completed processes
                for name, proc in list(running_processes.items()):
                    if proc.poll() is not None:
                        del running_processes[name]
                        gpu = gpu_assignments.pop(name)
                        available_gpus.append(gpu)
                        logger.info(f"GPU {gpu} freed by {name}")
            
            # Assign GPU
            gpu_id = available_gpus.pop(0)
            gpu_assignments[exp_name] = gpu_id
            
            # Check if should resume
            should_resume = resume and status["status"] == "paused"
            
            # Start experiment
            logger.info(f"Starting {exp_name} on GPU {gpu_id} (Priority {exp_config['priority']})")
            process = self.start_experiment(exp_config, gpu_id, resume=should_resume)
            
            if process:
                running_processes[exp_name] = process
                exp_config["gpu"] = gpu_id
                exp_config["status"] = "running"
            else:
                available_gpus.append(gpu_id)  # Return GPU if failed to start
        
        # Start monitoring in background
        import threading
        monitor_thread = threading.Thread(target=self.monitor_experiments, daemon=True)
        monitor_thread.start()
        
        # Wait for all experiments
        logger.info(f"Started {len(running_processes)} experiments. Monitoring...")
        while running_processes:
            time.sleep(60)
            for name, proc in list(running_processes.items()):
                if proc.poll() is not None:
                    logger.info(f"Experiment {name} finished (exit code: {proc.returncode})")
                    del running_processes[name]
                    gpu = gpu_assignments.pop(name)
                    available_gpus.append(gpu)
        
        logger.info("All experiments completed!")

def main():
    parser = argparse.ArgumentParser(description="Run Tier 1 experiments")
    parser.add_argument("--max-parallel", type=int, default=6, help="Maximum parallel experiments")
    parser.add_argument("--no-resume", action="store_true", help="Don't resume from checkpoints")
    parser.add_argument("--experiment", type=str, help="Run specific experiment only")
    
    args = parser.parse_args()
    
    runner = Tier1ExperimentRunner()
    
    if args.experiment:
        # Run specific experiment
        exp_config = next((e for e in TIER1_EXPERIMENTS if e["name"] == args.experiment), None)
        if not exp_config:
            logger.error(f"Experiment {args.experiment} not found")
            return
        
        gpus = runner.get_available_gpus()
        if not gpus:
            logger.error("No available GPUs")
            return
        
        runner.start_experiment(exp_config, gpus[0], resume=not args.no_resume)
    else:
        # Run all experiments
        runner.run_all_experiments(
            max_parallel=args.max_parallel,
            resume=not args.no_resume
        )

if __name__ == "__main__":
    main()


