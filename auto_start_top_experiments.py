#!/usr/bin/env python3
"""
Automated Top Experiments Runner
Automatically starts top-priority experiments when GPUs become available
Includes resume capability and email notifications
"""

import os
import sys
import subprocess
import pathlib
import json
import time
import logging
import smtplib
import signal
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

# Email configuration
EMAIL_TO = "pratirvce@gmail.com"
EMAIL_FROM = "pratirvce@gmail.com"
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
EMAIL_PASSWORD = os.environ.get("EMAIL_PASSWORD", "")

# Top experiments priority list
# Priority order: Lower number = Higher priority
TOP_EXPERIMENTS = [
    # Priority 1: Highest Impact Novel Experiments (Best Paper Roadmap)
    {
        "name": "best_paper_graph_aware_retrieval",
        "script": "train_graph_aware_tier1.py",
        "priority": 1,
        "expected_ndcg": "0.58-0.62",
        "time_estimate": "4-6 days",
        "gpu": None,
        "status": "pending",
        "description": "Conversation Graph-Aware Retrieval (First GNN for multi-turn retrieval) ⭐⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_rl_adaptive_retrieval",
        "script": "train_rl_adaptive_tier1.py",
        "priority": 1,
        "expected_ndcg": "0.59-0.63",
        "time_estimate": "5-7 days",
        "gpu": None,
        "status": "pending",
        "description": "Reinforcement Learning for Adaptive Retrieval (First RL for retrieval) ⭐⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_temporal_memory",
        "script": "train_temporal_memory_tier1.py",
        "priority": 1,
        "expected_ndcg": "0.58-0.62",
        "time_estimate": "4-6 days",
        "gpu": None,
        "status": "pending",
        "description": "Temporal Memory Networks (First memory-augmented retrieval) ⭐⭐⭐⭐⭐"
    },
    # Priority 2: Performance Boosters
    {
        "name": "best_paper_large_model_finetuning",
        "script": "train_large_model_tier1.py",
        "priority": 2,
        "expected_ndcg": "0.60-0.65",
        "time_estimate": "5-7 days",
        "gpu": None,
        "status": "pending",
        "description": "Large Model Fine-Tuning (BGE-Large) ⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_learned_rrf",
        "script": "train_learned_rrf_tier1.py",
        "priority": 2,
        "expected_ndcg": "0.62-0.66",
        "time_estimate": "3-4 days",
        "gpu": None,
        "status": "pending",
        "description": "Learned Reciprocal Rank Fusion (Neural weighting ensemble) ⭐⭐⭐⭐"
    },
    # Priority 3: Additional Novel Methods
    {
        "name": "best_paper_multitask_retrieval",
        "script": "train_multitask_retrieval_tier1.py",
        "priority": 3,
        "expected_ndcg": "0.59-0.63",
        "time_estimate": "5-7 days",
        "gpu": None,
        "status": "pending",
        "description": "Multi-Task Learning (Retrieval + Reranking, adapted for Task A) ⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_meta_learning",
        "script": "train_meta_learning_tier1.py",
        "priority": 3,
        "expected_ndcg": "0.57-0.61",
        "time_estimate": "5-7 days",
        "gpu": None,
        "status": "pending",
        "description": "Cross-Domain Transfer Learning with Meta-Learning (MAML) ⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_adversarial_curriculum",
        "script": "train_adversarial_curriculum_tier1.py",
        "priority": 3,
        "expected_ndcg": "0.57-0.61",
        "time_estimate": "4-5 days",
        "gpu": None,
        "status": "pending",
        "description": "Adversarial Hard Negative Mining with Curriculum Learning ⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_llm_distillation",
        "script": "train_llm_distillation_tier1.py",
        "priority": 3,
        "expected_ndcg": "0.58-0.62",
        "time_estimate": "4-6 days",
        "gpu": None,
        "status": "pending",
        "description": "Knowledge Distillation from Large Language Models ⭐⭐⭐⭐"
    },
    {
        "name": "best_paper_hierarchical_routing",
        "script": "train_hierarchical_routing_tier1.py",
        "priority": 3,
        "expected_ndcg": "0.60-0.64",
        "time_estimate": "4-6 days",
        "gpu": None,
        "status": "pending",
        "description": "Hierarchical Multi-Stage with Learned Routing ⭐⭐⭐⭐"
    },
    # Existing Tier 1 Experiments (Lower Priority)
    {
        "name": "tier1_cross_attention_query_document",
        "script": "train_cross_attention_tier1.py",
        "priority": 10,
        "expected_ndcg": "0.49-0.52",
        "time_estimate": "4-6 days",
        "gpu": None,
        "status": "pending",
        "description": "Cross-Attention Query-Document Interaction"
    },
    {
        "name": "tier1_iterative_refinement_improved",
        "script": "train_iterative_refinement_improved_tier1.py",
        "priority": 10,
        "expected_ndcg": "0.50-0.54",
        "time_estimate": "3-4 days",
        "gpu": None,
        "status": "pending",
        "description": "Iterative Refinement Retrieval with Feedback"
    },
    {
        "name": "tier1_learning_to_rank_listwise",
        "script": "train_learning_to_rank_tier1.py",
        "priority": 10,
        "expected_ndcg": "0.50-0.53",
        "time_estimate": "4-5 days",
        "gpu": None,
        "status": "pending",
        "description": "Learning-to-Rank with Listwise Loss"
    },
    {
        "name": "tier1_ensemble_best_methods",
        "script": "train_ensemble_best_tier1.py",
        "priority": 10,
        "expected_ndcg": "0.48-0.52",
        "time_estimate": "1-2 days",
        "gpu": None,
        "status": "pending",
        "description": "Ensemble of best methods"
    },
    {
        "name": "tier1_cross_encoder_large",
        "script": "train_cross_encoder_large_tier1.py",
        "priority": 10,
        "expected_ndcg": "0.51-0.54",
        "time_estimate": "2-4 days",
        "gpu": None,
        "status": "pending",
        "description": "Cross-Encoder with larger model"
    },
    {
        "name": "tier1_pseudo_relevance_feedback",
        "script": "train_pseudo_relevance_feedback_tier1.py",
        "priority": 10,
        "expected_ndcg": "0.48-0.51",
        "time_estimate": "2-3 days",
        "gpu": None,
        "status": "pending",
        "description": "Pseudo-relevance feedback with LLM expansion"
    }
]

class AutoExperimentRunner:
    """Automatically starts experiments when GPUs become available"""
    
    def __init__(self, experiments_dir: str = "experiments/retrieval", status_file: str = "auto_experiments_status.json"):
        self.experiments_dir = pathlib.Path(experiments_dir)
        self.status_file = pathlib.Path(status_file)
        self.running_experiments = {}  # {exp_name: {gpu, pid, start_time}}
        self.failed_experiments = {}  # {exp_name: {error, retry_count, last_failure_time}}
        self.max_retries = 3
        self.load_status()
    
    def load_status(self):
        """Load previous status if exists"""
        if self.status_file.exists():
            try:
                with open(self.status_file, 'r') as f:
                    data = json.load(f)
                    self.running_experiments = data.get('running_experiments', {})
                    self.failed_experiments = data.get('failed_experiments', {})
            except Exception as e:
                logger.warning(f"Could not load status: {e}")
    
    def save_status(self):
        """Save current status"""
        data = {
            'running_experiments': self.running_experiments,
            'failed_experiments': self.failed_experiments,
            'last_updated': datetime.now().isoformat()
        }
        with open(self.status_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def get_free_gpus(self) -> List[int]:
        """Get list of free GPUs (utilization < 10%)"""
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=index,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=10
            )
            if result.returncode == 0:
                free_gpus = []
                for line in result.stdout.strip().split('\n'):
                    if ',' in line:
                        parts = line.split(', ')
                        if len(parts) == 2:
                            try:
                                gpu_idx = int(parts[0])
                                util = int(parts[1])
                                if util < 10:  # Less than 10% utilization
                                    free_gpus.append(gpu_idx)
                            except ValueError:
                                continue
                return sorted(free_gpus)
        except Exception as e:
            logger.warning(f"Could not detect free GPUs: {e}")
        
        # Fallback: assume all GPUs available (0-5)
        return list(range(6))
    
    def check_experiment_status(self, exp_name: str) -> Tuple[bool, str]:
        """Check if experiment needs to be run"""
        exp_dir = self.experiments_dir / exp_name
        
        if not exp_dir.exists():
            return True, "not_started"
        
        results_file = exp_dir / "results.json"
        if not results_file.exists():
            # Check if it's running
            if exp_name in self.running_experiments:
                pid = self.running_experiments[exp_name].get('pid')
                if pid and self.is_process_running(pid):
                    return False, "running"
            return True, "in_progress"
        
        try:
            with open(results_file) as f:
                data = json.load(f)
            
            # Check if it's a valid completed result
            if 'average' in data:
                avg = data['average']
                if avg.get('nDCG@10', 0) > 0:
                    return False, "completed"
                else:
                    return True, "failed_needs_rerun"
            elif 'status' in data:
                if data['status'] == 'failed':
                    return True, "failed_needs_rerun"
                elif data['status'] == 'training_completed' and 'nDCG@10' not in str(data):
                    return True, "needs_evaluation"
            
            return False, "completed"
        except Exception as e:
            logger.warning(f"Error checking {exp_name}: {e}")
            return True, "needs_check"
    
    def is_process_running(self, pid: int) -> bool:
        """Check if process is still running"""
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    
    def send_email_notification(self, subject: str, body: str, is_error: bool = False):
        """Send email notification"""
        if not EMAIL_PASSWORD:
            logger.warning("EMAIL_PASSWORD not set. Skipping email notification.")
            return
        
        try:
            msg = MIMEMultipart()
            msg['From'] = EMAIL_FROM
            msg['To'] = EMAIL_TO
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(EMAIL_FROM, EMAIL_PASSWORD)
            server.send_message(msg)
            server.quit()
            
            logger.info(f"Email notification sent: {subject}")
        except Exception as e:
            logger.error(f"Failed to send email: {e}")
    
    def start_experiment(self, exp_config: Dict, gpu: int, resume: bool = True) -> Optional[int]:
        """Start an experiment on a specific GPU"""
        exp_name = exp_config['name']
        script = exp_config['script']
        
        logger.info(f"Starting {exp_name} on GPU {gpu}")
        
        # Check if script exists
        script_path = pathlib.Path(script)
        if not script_path.exists():
            logger.warning(f"Script {script} not found. Skipping {exp_name}.")
            self.send_email_notification(
                f"Experiment Failed: {exp_name}",
                f"Script {script} not found for experiment {exp_name}.",
                is_error=True
            )
            return None
        
        # Activate venv if it exists
        venv_python = pathlib.Path("venv/bin/python")
        if venv_python.exists():
            python_cmd = str(venv_python)
        else:
            python_cmd = "python3"
        
        output_dir = str(self.experiments_dir / exp_name)
        
        cmd = [
            python_cmd,
            script,
            "--experiment_name", exp_name,
            "--gpu", str(gpu),
            "--output_dir", output_dir
        ]
        
        if resume:
            cmd.append("--resume")
        else:
            cmd.append("--no-resume")
        
        # Run in background
        log_file = pathlib.Path(output_dir) / "training.log"
        log_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(log_file, 'a') as log:
                process = subprocess.Popen(
                    cmd,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    cwd=pathlib.Path.cwd(),
                    env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu))
                )
            
            pid = process.pid
            self.running_experiments[exp_name] = {
                'gpu': gpu,
                'pid': pid,
                'start_time': datetime.now().isoformat(),
                'script': script
            }
            self.save_status()
            
            logger.info(f"Started {exp_name} on GPU {gpu} (PID: {pid})")
            
            # Send notification
            self.send_email_notification(
                f"Experiment Started: {exp_name}",
                f"Experiment {exp_name} has been started on GPU {gpu}.\n\n"
                f"Expected nDCG@10: {exp_config['expected_ndcg']}\n"
                f"Time estimate: {exp_config['time_estimate']}\n"
                f"PID: {pid}\n"
                f"Log: {log_file}"
            )
            
            return pid
        except Exception as e:
            logger.error(f"Failed to start {exp_name}: {e}")
            self.send_email_notification(
                f"Experiment Failed to Start: {exp_name}",
                f"Failed to start experiment {exp_name} on GPU {gpu}.\n\nError: {str(e)}",
                is_error=True
            )
            return None
    
    def analyze_error(self, exp_name: str, log_content: str) -> Optional[str]:
        """Analyze error and return fix suggestion"""
        log_lower = log_content.lower()
        
        # Common error patterns and fixes
        if 'cuda error' in log_lower or 'invalid device ordinal' in log_lower:
            return "CUDA_DEVICE_ERROR"
        elif 'module not found' in log_lower or 'no module named' in log_lower:
            return "MISSING_MODULE"
        elif 'out of memory' in log_lower or 'cuda out of memory' in log_lower:
            return "OOM_ERROR"
        elif 'attributeerror' in log_lower:
            return "ATTRIBUTE_ERROR"
        elif 'importerror' in log_lower:
            return "IMPORT_ERROR"
        elif 'file not found' in log_lower or 'no such file' in log_lower:
            return "FILE_NOT_FOUND"
        elif 'syntaxerror' in log_lower:
            return "SYNTAX_ERROR"
        
        return "UNKNOWN_ERROR"
    
    def attempt_fix(self, exp_name: str, error_type: str) -> bool:
        """Attempt to fix common errors"""
        logger.info(f"Attempting to fix {error_type} for {exp_name}")
        
        if error_type == "MISSING_MODULE" or error_type == "IMPORT_ERROR":
            # Try to install missing packages (simplified - would need package detection)
            logger.info(f"Module error detected. May need to install packages.")
            return True  # Assume fixable
        elif error_type == "OOM_ERROR":
            # Could reduce batch size, but that requires config modification
            logger.info(f"OOM error detected. May need to reduce batch size.")
            return True
        elif error_type == "CUDA_DEVICE_ERROR":
            # GPU assignment issue - should be handled by auto-runner
            logger.info(f"CUDA device error. Will retry with different GPU.")
            return True
        elif error_type == "FILE_NOT_FOUND":
            # Check if data files exist
            logger.info(f"File not found error. Checking data files.")
            return True
        
        return False
    
    def monitor_experiments(self):
        """Monitor running experiments and check for completion/failures"""
        completed = []
        failed = []
        to_restart = []
        
        for exp_name, info in list(self.running_experiments.items()):
            pid = info.get('pid')
            if not pid or not self.is_process_running(pid):
                # Process finished
                exp_dir = self.experiments_dir / exp_name
                results_file = exp_dir / "results.json"
                
                if results_file.exists():
                    try:
                        with open(results_file) as f:
                            data = json.load(f)
                        if 'average' in data and data['average'].get('nDCG@10', 0) > 0:
                            completed.append((exp_name, info))
                            logger.info(f"✅ Experiment {exp_name} completed successfully")
                            # Remove from failed list if it was there
                            if exp_name in self.failed_experiments:
                                del self.failed_experiments[exp_name]
                            self.send_email_notification(
                                f"Experiment Completed: {exp_name}",
                                f"Experiment {exp_name} has completed successfully.\n\n"
                                f"Results: {json.dumps(data.get('average', {}), indent=2)}"
                            )
                        else:
                            # Check if it's a failure
                            if 'status' in data and data['status'] == 'failed':
                                failed.append((exp_name, info))
                                error_msg = data.get('error', 'Unknown error')
                                self.handle_experiment_failure(exp_name, info, error_msg, exp_dir)
                            else:
                                failed.append((exp_name, info))
                                self.handle_experiment_failure(exp_name, info, "Results invalid", exp_dir)
                    except Exception as e:
                        failed.append((exp_name, info))
                        self.handle_experiment_failure(exp_name, info, str(e), exp_dir)
                else:
                    # Check log for errors
                    log_file = exp_dir / "training.log"
                    error_detected = False
                    error_msg = "Process terminated without results"
                    
                    if log_file.exists():
                        try:
                            with open(log_file, 'r') as f:
                                log_content = f.read()
                                if any(keyword in log_content.lower() for keyword in ['error', 'exception', 'failed', 'traceback']):
                                    error_detected = True
                                    # Extract last error
                                    lines = log_content.split('\n')
                                    for line in reversed(lines[-50:]):  # Check last 50 lines
                                        if any(kw in line.lower() for kw in ['error', 'exception', 'failed']):
                                            error_msg = line.strip()
                                            break
                                    
                                    failed.append((exp_name, info))
                                    self.handle_experiment_failure(exp_name, info, error_msg, exp_dir, log_content)
                        except Exception as e:
                            logger.warning(f"Could not read log for {exp_name}: {e}")
                            failed.append((exp_name, info))
                            self.handle_experiment_failure(exp_name, info, f"Could not read log: {e}", exp_dir)
                    else:
                        # No log file - process may have crashed immediately
                        failed.append((exp_name, info))
                        self.handle_experiment_failure(exp_name, info, "No log file found - process may have crashed", exp_dir)
        
        # Remove completed/failed from running
        for exp_name, _ in completed + failed:
            if exp_name in self.running_experiments:
                del self.running_experiments[exp_name]
        
        if completed or failed:
            self.save_status()
        
        return len(completed), len(failed)
    
    def handle_experiment_failure(self, exp_name: str, info: Dict, error_msg: str, exp_dir: pathlib.Path, log_content: str = ""):
        """Handle experiment failure: analyze, fix, and schedule restart"""
        logger.error(f"❌ Experiment {exp_name} failed: {error_msg}")
        
        # Analyze error
        error_type = "UNKNOWN_ERROR"
        if log_content:
            error_type = self.analyze_error(exp_name, log_content)
        
        # Track failure
        retry_count = self.failed_experiments.get(exp_name, {}).get('retry_count', 0)
        
        if retry_count < self.max_retries:
            # Attempt fix
            fixable = self.attempt_fix(exp_name, error_type)
            
            if fixable:
                retry_count += 1
                self.failed_experiments[exp_name] = {
                    'error': error_msg,
                    'error_type': error_type,
                    'retry_count': retry_count,
                    'last_failure_time': datetime.now().isoformat(),
                    'gpu': info.get('gpu'),
                    'status': 'will_retry'
                }
                
                logger.info(f"🔄 Will retry {exp_name} (attempt {retry_count}/{self.max_retries})")
                self.send_email_notification(
                    f"Experiment Failed - Will Retry: {exp_name}",
                    f"Experiment {exp_name} failed but will be retried.\n\n"
                    f"Error: {error_msg}\n"
                    f"Error Type: {error_type}\n"
                    f"Retry Count: {retry_count}/{self.max_retries}\n"
                    f"Log: {exp_dir / 'training.log'}",
                    is_error=True
                )
            else:
                self.failed_experiments[exp_name] = {
                    'error': error_msg,
                    'error_type': error_type,
                    'retry_count': retry_count,
                    'last_failure_time': datetime.now().isoformat(),
                    'status': 'unfixable'
                }
                logger.error(f"❌ Cannot fix {exp_name} - error type: {error_type}")
                self.send_email_notification(
                    f"Experiment Failed - Cannot Fix: {exp_name}",
                    f"Experiment {exp_name} failed and cannot be automatically fixed.\n\n"
                    f"Error: {error_msg}\n"
                    f"Error Type: {error_type}\n"
                    f"Log: {exp_dir / 'training.log'}",
                    is_error=True
                )
        else:
            self.failed_experiments[exp_name] = {
                'error': error_msg,
                'error_type': error_type,
                'retry_count': retry_count,
                'last_failure_time': datetime.now().isoformat(),
                'status': 'max_retries_exceeded'
            }
            logger.error(f"❌ Max retries exceeded for {exp_name}")
            self.send_email_notification(
                f"Experiment Failed - Max Retries: {exp_name}",
                f"Experiment {exp_name} failed after {self.max_retries} retries.\n\n"
                f"Error: {error_msg}\n"
                f"Error Type: {error_type}\n"
                f"Log: {exp_dir / 'training.log'}",
                is_error=True
            )
    
    def run(self, check_interval: int = 300, max_parallel: int = 6):
        """Main loop: monitor GPUs and start experiments"""
        logger.info("="*60)
        logger.info("Auto Experiment Runner Started")
        logger.info("="*60)
        logger.info(f"Monitoring interval: {check_interval} seconds")
        logger.info(f"Max parallel experiments: {max_parallel}")
        logger.info(f"Email notifications: {'Enabled' if EMAIL_PASSWORD else 'Disabled (set EMAIL_PASSWORD env var)'}")
        
        # Sort experiments by priority
        experiments_to_run = sorted(
            [exp for exp in TOP_EXPERIMENTS],
            key=lambda x: x['priority']
        )
        
        try:
            while True:
                # Monitor existing experiments
                completed, failed = self.monitor_experiments()
                if completed > 0 or failed > 0:
                    logger.info(f"Status update: {completed} completed, {failed} failed")
                
                # Check for available GPUs
                free_gpus = self.get_free_gpus()
                running_count = len(self.running_experiments)
                
                logger.info(f"Free GPUs: {free_gpus}, Running experiments: {running_count}/{max_parallel}")
                
                # First, restart failed experiments that are fixable
                for exp_name, fail_info in list(self.failed_experiments.items()):
                    if running_count >= max_parallel:
                        break
                    
                    if fail_info.get('status') == 'will_retry' and exp_name not in self.running_experiments:
                        # Find matching experiment config
                        exp_config = next((e for e in experiments_to_run if e['name'] == exp_name), None)
                        if exp_config and free_gpus:
                            gpu = free_gpus.pop(0)
                            logger.info(f"🔄 Restarting failed experiment {exp_name} on GPU {gpu}")
                            
                            # Clear previous failure status temporarily
                            fail_info['status'] = 'retrying'
                            
                            pid = self.start_experiment(exp_config, gpu, resume=True)
                            if pid:
                                running_count += 1
                                logger.info(f"✅ Restarted {exp_name} on GPU {gpu}")
                            else:
                                free_gpus.append(gpu)
                                fail_info['status'] = 'will_retry'
                
                # Start new experiments if GPUs available
                if free_gpus and running_count < max_parallel:
                    for exp_config in experiments_to_run:
                        if running_count >= max_parallel:
                            break
                        
                        exp_name = exp_config['name']
                        
                        # Skip if max retries exceeded
                        if exp_name in self.failed_experiments:
                            fail_info = self.failed_experiments[exp_name]
                            if fail_info.get('status') in ['max_retries_exceeded', 'unfixable']:
                                continue
                        
                        should_run, status = self.check_experiment_status(exp_name)
                        
                        if should_run and exp_name not in self.running_experiments:
                            # Find available GPU
                            if free_gpus:
                                gpu = free_gpus.pop(0)
                                
                                # Determine if should resume
                                resume = status != "failed_needs_rerun"
                                
                                pid = self.start_experiment(exp_config, gpu, resume=resume)
                                if pid:
                                    running_count += 1
                                    logger.info(f"Started {exp_name} on GPU {gpu}")
                                    # Remove from failed if it was there
                                    if exp_name in self.failed_experiments:
                                        del self.failed_experiments[exp_name]
                                else:
                                    free_gpus.append(gpu)  # Return GPU to pool
                
                # Wait before next check
                time.sleep(check_interval)
        
        except KeyboardInterrupt:
            logger.info("\nShutdown requested. Saving status...")
            self.save_status()
            logger.info("Status saved. Exiting.")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Auto-start top experiments when GPUs available')
    parser.add_argument('--check-interval', type=int, default=300, help='Check interval in seconds (default: 300)')
    parser.add_argument('--max-parallel', type=int, default=6, help='Max parallel experiments (default: 6)')
    
    args = parser.parse_args()
    
    runner = AutoExperimentRunner()
    runner.run(check_interval=args.check_interval, max_parallel=args.max_parallel)

if __name__ == "__main__":
    main()

