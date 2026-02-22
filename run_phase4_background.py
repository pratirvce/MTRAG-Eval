"""
Run Phase 4 experiments in the background with status tracking
"""

import json
import pathlib
import subprocess
import sys
import logging
import time
import signal
import os
from datetime import datetime
from typing import Dict, List, Optional

# Setup logging
LOG_DIR = pathlib.Path("experiments/retrieval/logs")
LOG_DIR.mkdir(parents=True, exist_ok=True)

STATUS_FILE = pathlib.Path("experiments/retrieval/phase4_status.json")
PID_FILE = pathlib.Path("experiments/retrieval/phase4_runner.pid")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_DIR / f"phase4_background_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
        logging.StreamHandler()
    ]
)

# Import experiment configurations
from run_phase4_experiments import PHASE4_EXPERIMENTS

class ExperimentRunner:
    def __init__(self):
        self.status = self.load_status()
        self.running_processes = {}
        
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
            "last_updated": None,
            "current_experiment": None
        }
    
    def save_status(self):
        """Save experiment status to file."""
        self.status["last_updated"] = datetime.now().isoformat()
        with open(STATUS_FILE, 'w') as f:
            json.dump(self.status, f, indent=2)
    
    def update_experiment_status(self, exp_name: str, status: str, details: Optional[Dict] = None):
        """Update status of an experiment."""
        if exp_name not in self.status["experiments"]:
            self.status["experiments"][exp_name] = {
                "status": "pending",
                "started_at": None,
                "completed_at": None,
                "log_file": None,
                "error": None
            }
        
        self.status["experiments"][exp_name]["status"] = status
        self.status["experiments"][exp_name]["last_updated"] = datetime.now().isoformat()
        
        if status == "running":
            if self.status["experiments"][exp_name]["started_at"] is None:
                self.status["experiments"][exp_name]["started_at"] = datetime.now().isoformat()
            self.status["current_experiment"] = exp_name
        elif status in ["completed", "failed"]:
            self.status["experiments"][exp_name]["completed_at"] = datetime.now().isoformat()
            if status == "failed" and details:
                self.status["experiments"][exp_name]["error"] = details.get("error", "Unknown error")
        
        if details:
            self.status["experiments"][exp_name].update(details)
        
        self.save_status()
    
    def run_experiment(self, exp_name: str, exp_config: Dict) -> bool:
        """Run a single experiment in the background."""
        logging.info(f"Starting experiment: {exp_name}")
        self.update_experiment_status(exp_name, "running")
        
        # Create experiment directory
        exp_dir = pathlib.Path("experiments/retrieval") / exp_name
        exp_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        config_file = exp_dir / "config.json"
        with open(config_file, 'w') as f:
            json.dump(exp_config['config'], f, indent=2)
        
        # Prepare log files
        log_file = exp_dir / "training.log"
        error_log = exp_dir / "error.log"
        
        self.update_experiment_status(exp_name, "running", {
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
        
        try:
            # Run in background
            with open(log_file, 'w') as log_f, open(error_log, 'w') as err_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=log_f,
                    stderr=err_f,
                    cwd=pathlib.Path(".").absolute()
                )
            
            self.running_processes[exp_name] = process
            logging.info(f"Experiment {exp_name} started (PID: {process.pid})")
            
            # Wait for completion (with timeout monitoring)
            return_code = process.wait()
            
            # Remove from running processes
            if exp_name in self.running_processes:
                del self.running_processes[exp_name]
            
            if return_code == 0:
                logging.info(f"✅ Experiment {exp_name} completed successfully")
                self.update_experiment_status(exp_name, "completed")
                return True
            else:
                logging.error(f"❌ Experiment {exp_name} failed (return code: {return_code})")
                self.update_experiment_status(exp_name, "failed", {
                    "error": f"Process exited with code {return_code}",
                    "return_code": return_code
                })
                return False
                
        except Exception as e:
            logging.error(f"❌ Error running experiment {exp_name}: {e}")
            self.update_experiment_status(exp_name, "failed", {
                "error": str(e)
            })
            return False
    
    def run_all_experiments(self, experiments: Dict, sequential: bool = True):
        """Run all experiments."""
        if not self.status.get("started_at"):
            self.status["started_at"] = datetime.now().isoformat()
            self.status["current_experiment"] = None
            self.save_status()
        
        total = len(experiments)
        completed = 0
        failed = 0
        
        logging.info(f"\n{'='*60}")
        logging.info(f"Starting {total} Phase 4 experiments")
        logging.info(f"{'='*60}\n")
        
        for i, (exp_name, exp_config) in enumerate(experiments.items(), 1):
            # Check if already completed
            if exp_name in self.status["experiments"]:
                current_status = self.status["experiments"][exp_name].get("status", "pending")
                if current_status == "completed":
                    logging.info(f"Skipping {exp_name} (already completed)")
                    completed += 1
                    continue
                elif current_status == "running":
                    logging.info(f"Skipping {exp_name} (currently running)")
                    continue
            
            logging.info(f"\n[{i}/{total}] Processing: {exp_name}")
            logging.info(f"Description: {exp_config['description']}")
            
            success = self.run_experiment(exp_name, exp_config)
            
            if success:
                completed += 1
            else:
                failed += 1
            
            # Progress update
            logging.info(f"Progress: {completed + failed}/{total} completed ({completed} successful, {failed} failed)")
            
            if not sequential and i < total:
                logging.info(f"Waiting 10 seconds before next experiment...")
                time.sleep(10)
        
        # Final summary
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
        summary.append("Phase 4 Experiments Status")
        summary.append(f"{'='*60}\n")
        
        if self.status.get("started_at"):
            summary.append(f"Started: {self.status['started_at']}")
        if self.status.get("last_updated"):
            summary.append(f"Last Updated: {self.status['last_updated']}")
        if self.status.get("current_experiment"):
            summary.append(f"Current: {self.status['current_experiment']}")
        summary.append("")
        
        experiments = self.status.get("experiments", {})
        if not experiments:
            summary.append("No experiments run yet.")
        else:
            summary.append(f"{'Experiment':<40} | {'Status':<12} | {'Started':<20}")
            summary.append("-" * 80)
            
            for exp_name, exp_data in experiments.items():
                status = exp_data.get("status", "unknown")
                started = exp_data.get("started_at", "N/A")[:19] if exp_data.get("started_at") else "N/A"
                
                # Color coding
                status_icon = {
                    "completed": "✅",
                    "running": "🔄",
                    "failed": "❌",
                    "pending": "⏳"
                }.get(status, "❓")
                
                summary.append(f"{exp_name:<40} | {status_icon} {status:<10} | {started:<20}")
        
        summary.append(f"{'='*60}\n")
        return "\n".join(summary)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Run Phase 4 experiments in background')
    parser.add_argument('--experiment', type=str, help='Specific experiment to run (or "all")')
    parser.add_argument('--sequential', action='store_true', default=True,
                        help='Run experiments sequentially (default: True)')
    parser.add_argument('--status', action='store_true', help='Show current status and exit')
    parser.add_argument('--resume', action='store_true', help='Resume from where we left off')
    
    args = parser.parse_args()
    
    runner = ExperimentRunner()
    
    if args.status:
        print(runner.get_status_summary())
        return
    
    # Save PID
    with open(PID_FILE, 'w') as f:
        f.write(str(os.getpid()))
    
    # Handle SIGTERM gracefully
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
        
        runner.run_all_experiments(experiments, sequential=args.sequential)
        
    except KeyboardInterrupt:
        logging.info("Interrupted by user, saving status...")
        runner.save_status()
    finally:
        # Clean up PID file
        if PID_FILE.exists():
            PID_FILE.unlink()

if __name__ == "__main__":
    main()

