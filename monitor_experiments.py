#!/usr/bin/env python3
"""
Experiment Monitoring Script
Monitors all running experiments, detects failures, and generates notifications
"""

import os
import json
import subprocess
import pathlib
import time
import argparse
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import re
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

class ExperimentMonitor:
    def __init__(self, check_interval: int = 60):
        self.check_interval = check_interval
        self.experiment_patterns = {
            'cross_encoder': r'train_cross_encoder_finetuned\.py',
            'multistage': r'train_multistage_retrieval\.py',
            'llm_expansion': r'train_llm_query_expansion\.py',
            'hybrid': r'train_hybrid_learned\.py',
            'reranking': r'train_reranking\.py',
            'ensemble': r'train_ensemble\.py',
            'query_expansion': r'train_query_expansion\.py',
            'conversation_aware': r'train_conversation_aware_retrieval\.py',
            'iterative_refinement': r'train_iterative_refinement_retrieval\.py'
        }
        self.error_patterns = [
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
        self.notification_file = pathlib.Path("experiment_failures.json")
        self.status_file = pathlib.Path("experiment_status.json")
        
    def get_running_experiments(self) -> List[Dict]:
        """Get list of currently running experiments"""
        experiments = []
        
        try:
            result = subprocess.run(
                ['ps', 'aux'],
                capture_output=True,
                text=True
            )
            
            for line in result.stdout.split('\n'):
                for exp_type, pattern in self.experiment_patterns.items():
                    if re.search(pattern, line):
                        parts = line.split()
                        if len(parts) >= 11:
                            pid = parts[1]
                            cmd = ' '.join(parts[10:])
                            
                            # Extract experiment name and GPU
                            exp_name = self._extract_experiment_name(cmd)
                            gpu_id = self._extract_gpu_id(cmd)
                            config_path = self._extract_config_path(cmd)
                            
                            experiments.append({
                                'pid': int(pid),
                                'type': exp_type,
                                'name': exp_name,
                                'gpu_id': gpu_id,
                                'config': config_path,
                                'cmd': cmd,
                                'status': 'running'
                            })
                            break
        except Exception as e:
            print(f"Error getting running experiments: {e}")
        
        return experiments
    
    def _extract_experiment_name(self, cmd: str) -> str:
        """Extract experiment name from command"""
        # Try to extract from config path
        match = re.search(r'--config\s+([^\s]+)', cmd)
        if match:
            config_path = match.group(1)
            # Extract experiment name from path
            parts = config_path.split('/')
            for part in reversed(parts):
                if part.startswith('phase'):
                    return part.replace('config.json', '').rstrip('/')
        return 'unknown'
    
    def _extract_gpu_id(self, cmd: str) -> Optional[int]:
        """Extract GPU ID from command"""
        match = re.search(r'--gpu_id\s+(\d+)', cmd)
        if match:
            return int(match.group(1))
        return None
    
    def _extract_config_path(self, cmd: str) -> Optional[str]:
        """Extract config path from command"""
        match = re.search(r'--config\s+([^\s]+)', cmd)
        if match:
            return match.group(1)
        return None
    
    def check_experiment_logs(self, experiment: Dict) -> Tuple[bool, List[str]]:
        """Check experiment log file for errors"""
        errors = []
        exp_name = experiment['name']
        
        # Find log file
        log_paths = [
            pathlib.Path(f"experiments/retrieval/{exp_name}/training.log"),
            pathlib.Path(f"experiments/retrieval/{exp_name}/eval.log"),
            pathlib.Path(f"experiments/retrieval/{exp_name}/log.txt"),
        ]
        
        log_file = None
        for path in log_paths:
            if path.exists():
                log_file = path
                break
        
        if not log_file:
            return False, []
        
        try:
            # Read last 100 lines of log
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                recent_lines = lines[-100:] if len(lines) > 100 else lines
                
                # Check for errors
                for i, line in enumerate(recent_lines):
                    for error_pattern in self.error_patterns:
                        if re.search(error_pattern, line, re.IGNORECASE):
                            # Get context (5 lines before and after)
                            start = max(0, i - 5)
                            end = min(len(recent_lines), i + 6)
                            context = ''.join(recent_lines[start:end])
                            errors.append({
                                'pattern': error_pattern,
                                'line': line.strip(),
                                'context': context,
                                'line_number': len(lines) - len(recent_lines) + i + 1
                            })
                            break
        except Exception as e:
            errors.append({
                'pattern': 'LogReadError',
                'line': f"Could not read log file: {e}",
                'context': '',
                'line_number': 0
            })
        
        return len(errors) > 0, errors
    
    def check_process_status(self, pid: int) -> Tuple[bool, Optional[int]]:
        """Check if process is still running and get exit code if finished"""
        try:
            result = subprocess.run(
                ['ps', '-p', str(pid), '-o', 'pid,stat'],
                capture_output=True,
                text=True
            )
            
            if result.returncode != 0:
                # Process not found - check if it exited
                # Try to get exit code from /proc
                proc_status = pathlib.Path(f"/proc/{pid}/status")
                if not proc_status.exists():
                    # Process definitely finished
                    return False, None
            
            # Process is running
            return True, None
        except Exception:
            return False, None
    
    def check_experiment_completion(self, experiment: Dict) -> bool:
        """Check if experiment completed successfully"""
        exp_name = experiment['name']
        results_file = pathlib.Path(f"experiments/retrieval/{exp_name}/results.json")
        return results_file.exists()
    
    def monitor_experiments(self, notify: bool = True) -> Dict:
        """Monitor all experiments and detect failures"""
        running_experiments = self.get_running_experiments()
        failed_experiments = []
        completed_experiments = []
        status_report = {
            'timestamp': datetime.now().isoformat(),
            'running': [],
            'failed': [],
            'completed': [],
            'warnings': []
        }
        
        # Check each running experiment
        for exp in running_experiments:
            exp_name = exp['name']
            pid = exp['pid']
            
            # Check if process is still running
            is_running, exit_code = self.check_process_status(pid)
            
            if not is_running:
                # Process stopped - check if it completed successfully
                if self.check_experiment_completion(exp):
                    completed_experiments.append(exp)
                    status_report['completed'].append({
                        'name': exp_name,
                        'pid': pid,
                        'completed_at': datetime.now().isoformat()
                    })
                else:
                    # Process stopped but no results - likely failed
                    has_errors, errors = self.check_experiment_logs(exp)
                    failed_exp = {
                        'name': exp_name,
                        'type': exp['type'],
                        'pid': pid,
                        'gpu_id': exp['gpu_id'],
                        'config': exp['config'],
                        'cmd': exp['cmd'],
                        'status': 'stopped',
                        'exit_code': exit_code,
                        'errors': errors,
                        'detected_at': datetime.now().isoformat(),
                        'log_file': f"experiments/retrieval/{exp_name}/training.log"
                    }
                    failed_experiments.append(failed_exp)
                    status_report['failed'].append(failed_exp)
            else:
                # Process still running - check for errors in logs
                has_errors, errors = self.check_experiment_logs(exp)
                if has_errors:
                    # Running but has errors - might be failing
                    status_report['warnings'].append({
                        'name': exp_name,
                        'pid': pid,
                        'errors': errors,
                        'status': 'running_with_errors'
                    })
                
                status_report['running'].append({
                    'name': exp_name,
                    'pid': pid,
                    'gpu_id': exp['gpu_id'],
                    'status': 'running'
                })
        
        # Save status report
        with open(self.status_file, 'w') as f:
            json.dump(status_report, f, indent=2)
        
        # Save failures if any
        if failed_experiments:
            failure_report = {
                'timestamp': datetime.now().isoformat(),
                'failed_experiments': failed_experiments,
                'total_failed': len(failed_experiments)
            }
            with open(self.notification_file, 'w') as f:
                json.dump(failure_report, f, indent=2)
            
            if notify:
                self.print_failure_notification(failed_experiments)
                # Send email notification
                try:
                    self.send_email_notification(failed_experiments)
                except Exception as e:
                    print(f"⚠️  Could not send email notification: {e}")
        
        return status_report
    
    def send_email_notification(self, failed_experiments: List[Dict]):
        """Send email notification about failed experiments"""
        email_to = "pratirvce@gmail.com"
        
        # Check if email was already sent for these failures (avoid spam)
        sent_failures_file = pathlib.Path("experiment_failures_sent.json")
        sent_failures = set()
        if sent_failures_file.exists():
            try:
                with open(sent_failures_file, 'r') as f:
                    data = json.load(f)
                    sent_failures = set(data.get('sent_failures', []))
            except:
                pass
        
        # Filter out already notified failures
        new_failures = []
        for exp in failed_experiments:
            failure_id = f"{exp['name']}_{exp['pid']}_{exp.get('detected_at', '')}"
            if failure_id not in sent_failures:
                new_failures.append(exp)
                sent_failures.add(failure_id)
        
        if not new_failures:
            return  # No new failures to report
        
        # Build email content
        subject = f"🚨 {len(new_failures)} Experiment(s) Failed - MTRAG Benchmark"
        
        body_text = f"""⚠️  EXPERIMENT FAILURES DETECTED

{len(new_failures)} experiment(s) failed and need attention:

"""
        
        for i, exp in enumerate(new_failures, 1):
            body_text += f"""
{i}. {exp['name']} ({exp['type']})
   PID: {exp['pid']}
   GPU: {exp.get('gpu_id', 'N/A')}
   Status: {exp['status']}
   Config: {exp.get('config', 'N/A')}
   Log: {exp.get('log_file', 'N/A')}
"""
            if exp.get('exit_code') is not None:
                body_text += f"   Exit Code: {exp['exit_code']}\n"
            
            if exp.get('errors'):
                body_text += f"   Errors Found: {len(exp['errors'])}\n"
                if exp['errors']:
                    first_error = exp['errors'][0]
                    body_text += f"   First Error: {first_error.get('pattern', 'Unknown')}\n"
                    error_line = first_error.get('line', '')[:200]
                    if error_line:
                        body_text += f"   Error Line: {error_line}\n"
            body_text += "\n"
        
        body_text += f"""
📝 To fix and restart:
   1. Check the log files for detailed error messages
   2. Fix the issues in the code/config
   3. Restart with: python <script> --config <config> --gpu_id <gpu> --resume

📄 Full failure report saved to: experiment_failures.json

Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Try to send email using system mail command (works on most Linux systems)
        try:
            # Try using system's mail command first
            import subprocess
            mail_cmd = f'echo "{body_text}" | mail -s "{subject}" {email_to}'
            result = subprocess.run(mail_cmd, shell=True, capture_output=True, text=True)
            
            if result.returncode == 0:
                print(f"✅ Email notification sent to {email_to} (via mail command)")
            else:
                # Fallback: try sendmail
                sendmail_cmd = f'echo -e "Subject: {subject}\n\n{body_text}" | sendmail {email_to}'
                result2 = subprocess.run(sendmail_cmd, shell=True, capture_output=True, text=True)
                
                if result2.returncode == 0:
                    print(f"✅ Email notification sent to {email_to} (via sendmail)")
                else:
                    # Try SMTP if configured
                    smtp_server = os.environ.get('SMTP_SERVER', '')
                    if smtp_server:
                        try:
                            smtp_port = int(os.environ.get('SMTP_PORT', '587'))
                            smtp_user = os.environ.get('SMTP_USER', '')
                            smtp_password = os.environ.get('SMTP_PASSWORD', '')
                            email_from = os.environ.get('EMAIL_FROM', f'experiments@{os.uname().nodename}')
                            
                            msg = MIMEMultipart()
                            msg['From'] = email_from
                            msg['To'] = email_to
                            msg['Subject'] = subject
                            msg.attach(MIMEText(body_text, 'plain'))
                            
                            server = smtplib.SMTP(smtp_server, smtp_port)
                            if smtp_user and smtp_password:
                                server.starttls()
                                server.login(smtp_user, smtp_password)
                            server.send_message(msg)
                            server.quit()
                            print(f"✅ Email notification sent to {email_to} (via SMTP)")
                        except Exception as smtp_e:
                            print(f"⚠️  Could not send email via SMTP: {smtp_e}")
                            raise
                    else:
                        print(f"⚠️  Could not send email (mail/sendmail not available, SMTP not configured)")
                        print(f"   Email would be sent to: {email_to}")
                        print(f"   Subject: {subject}")
                        raise Exception("No email method available")
            
            # Save sent failures
            with open(sent_failures_file, 'w') as f:
                json.dump({'sent_failures': list(sent_failures)}, f, indent=2)
            
        except Exception as e:
            print(f"⚠️  Could not send email notification: {e}")
            print(f"   Email would be sent to: {email_to}")
            print(f"   Subject: {subject}")
            # Don't raise - just log the error
    
    def print_failure_notification(self, failed_experiments: List[Dict]):
        """Print failure notification"""
        print("\n" + "="*80)
        print("⚠️  EXPERIMENT FAILURES DETECTED")
        print("="*80)
        print(f"\n🚨 {len(failed_experiments)} experiment(s) failed and need attention:\n")
        
        for i, exp in enumerate(failed_experiments, 1):
            print(f"{i}. {exp['name']} ({exp['type']})")
            print(f"   PID: {exp['pid']}")
            print(f"   GPU: {exp.get('gpu_id', 'N/A')}")
            print(f"   Status: {exp['status']}")
            if exp.get('exit_code') is not None:
                print(f"   Exit Code: {exp['exit_code']}")
            print(f"   Config: {exp.get('config', 'N/A')}")
            print(f"   Log: {exp.get('log_file', 'N/A')}")
            
            if exp.get('errors'):
                print(f"   Errors Found: {len(exp['errors'])}")
                # Show first error
                if exp['errors']:
                    first_error = exp['errors'][0]
                    print(f"   First Error: {first_error.get('pattern', 'Unknown')}")
                    error_line = first_error.get('line', '')[:100]
                    if error_line:
                        print(f"   Error Line: {error_line}...")
            
            print()
        
        print("="*80)
        print("\n📝 To fix and restart:")
        print("   1. Check the log files for detailed error messages")
        print("   2. Fix the issues in the code/config")
        print("   3. Restart with: python <script> --config <config> --gpu_id <gpu> --resume")
        print("\n📄 Full failure report saved to: experiment_failures.json")
        print("="*80 + "\n")
    
    def run_continuous_monitoring(self):
        """Run continuous monitoring loop"""
        print("🔍 Starting continuous experiment monitoring...")
        print(f"   Check interval: {self.check_interval} seconds")
        print(f"   Monitoring: {', '.join(self.experiment_patterns.keys())}")
        print("\nPress Ctrl+C to stop\n")
        
        try:
            while True:
                status = self.monitor_experiments(notify=True)
                
                # Print summary
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
                      f"Running: {len(status['running'])}, "
                      f"Failed: {len(status['failed'])}, "
                      f"Completed: {len(status['completed'])}, "
                      f"Warnings: {len(status['warnings'])}")
                
                time.sleep(self.check_interval)
        except KeyboardInterrupt:
            print("\n\n⏹️  Monitoring stopped by user")
        except Exception as e:
            print(f"\n❌ Monitoring error: {e}")

def main():
    parser = argparse.ArgumentParser(description='Monitor experiments for failures')
    parser.add_argument('--check-interval', type=int, default=60,
                        help='Check interval in seconds (default: 60)')
    parser.add_argument('--once', action='store_true',
                        help='Run check once and exit (default: continuous)')
    parser.add_argument('--no-notify', action='store_true',
                        help='Do not print notifications')
    
    args = parser.parse_args()
    
    monitor = ExperimentMonitor(check_interval=args.check_interval)
    
    if args.once:
        status = monitor.monitor_experiments(notify=not args.no_notify)
        print(f"\n✅ Check complete. Status saved to: {monitor.status_file}")
        if status['failed']:
            print(f"⚠️  {len(status['failed'])} failed experiment(s) detected!")
            print(f"   See: {monitor.notification_file}")
            return 1
        return 0
    else:
        monitor.run_continuous_monitoring()
        return 0

if __name__ == "__main__":
    exit(main())
