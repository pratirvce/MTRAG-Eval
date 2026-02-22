#!/usr/bin/env python3
"""
Automatically evaluate all completed Phase 4 experiments
Checks for experiments with completed training and runs evaluation on test set
"""

import json
import pathlib
import subprocess
import sys
import logging
import re
from datetime import datetime
from typing import Dict, List, Optional

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)

EXPERIMENTS_DIR = pathlib.Path("experiments/retrieval")
MODELS_DIR = pathlib.Path("models")

def check_training_completed(exp_name: str) -> tuple[bool, Optional[float]]:
    """Check if training has completed for an experiment."""
    log_file = EXPERIMENTS_DIR / exp_name / "training.log"
    config_file = EXPERIMENTS_DIR / exp_name / "config.json"
    
    if not log_file.exists():
        return False, None
    
    # Get expected epochs from config
    expected_epochs = None
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                expected_epochs = config.get('epochs', None)
        except:
            pass
    
    # Check log for completion
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            lines = content.split('\n')
            
            # Try to find final epoch from the log
            final_epoch = None
            for line in reversed(lines[-200:]):
                if "'epoch':" in line or '"epoch":' in line or '"epoch"' in line:
                    # Try to parse JSON-like dict
                    match = re.search(r"['\"]epoch['\"]\s*[:=]\s*([\d.]+)", line.lower())
                    if match:
                        try:
                            final_epoch = float(match.group(1))
                            break
                        except:
                            pass
            
            # Check if training completed message exists OR if we reached expected epochs
            has_completion_message = 'Training complete' in content or 'Training completed' in content
            
            # If we have expected epochs, check if we reached it
            if expected_epochs:
                if final_epoch is not None and final_epoch >= expected_epochs - 0.1:
                    # Reached expected epochs - training is done
                    return True, final_epoch
                elif has_completion_message:
                    # Has completion message - training is done
                    return True, final_epoch if final_epoch else expected_epochs
            else:
                # No expected epochs info, check for completion message
                if has_completion_message:
                    return True, final_epoch if final_epoch else 1.0
                # If final epoch exists and is >= 1, assume it might be complete
                # (but this is less reliable)
                if final_epoch is not None and final_epoch >= 1.0:
                    # Check if this is the last log entry (might indicate completion)
                    if len(lines) > 0 and ('epoch' in lines[-1].lower() or 'epoch' in lines[-2].lower()):
                        return True, final_epoch
    except Exception as e:
        logging.warning(f"Error reading log for {exp_name}: {e}")
    
    return False, None

def get_model_path(exp_name: str) -> Optional[pathlib.Path]:
    """Get the model path for an experiment."""
    # Try different possible paths
    possible_paths = [
        MODELS_DIR / exp_name,
        MODELS_DIR / exp_name.replace('phase4_', ''),
        pathlib.Path(".") / exp_name,
        pathlib.Path(".") / exp_name.replace('phase4_', ''),
        EXPERIMENTS_DIR.parent.parent / "models" / exp_name,
    ]
    
    # Also check config for output_path
    config_file = EXPERIMENTS_DIR / exp_name / "config.json"
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
                output_path = config.get('output_path', '')
                if output_path:
                    possible_paths.insert(0, pathlib.Path(output_path))
        except:
            pass
    
    for path in possible_paths:
        if path.exists() and (path / "config.json").exists():
            return path
        # Also check if it's the model directory itself
        if path.exists() and any((path / f).exists() for f in ["config.json", "modules.json", "1_Pooling"]):
            return path
    
    return None

def convert_results_format(exp_name: str, eval_results: Dict, config_file: pathlib.Path, model_path: pathlib.Path) -> Dict:
    """Convert evaluation results to the expected format matching phase1-3 results."""
    # Load config if available
    config = {}
    if config_file.exists():
        try:
            with open(config_file) as f:
                config = json.load(f)
        except:
            pass
    
    # Extract domain results and averages
    domain_results = {}
    for domain in ["clapnq", "fiqa", "govt", "cloud"]:
        if domain in eval_results:
            domain_results[domain] = {
                "Recall@5": eval_results[domain].get("Recall@5", 0),
                "Recall@10": eval_results[domain].get("Recall@10", 0),
                "nDCG@5": eval_results[domain].get("nDCG@5", 0),
                "nDCG@10": eval_results[domain].get("nDCG@10", 0),
            }
    
    # Calculate averages (if average key exists, use it; otherwise calculate)
    if "average" in eval_results:
        avg = eval_results["average"]
        averages = {
            "Recall@5": avg.get("Recall@5", 0),
            "Recall@10": avg.get("Recall@10", 0),
            "nDCG@5": avg.get("nDCG@5", 0),
            "nDCG@10": avg.get("nDCG@10", 0),
        }
    else:
        # Calculate from domain results
        if domain_results:
            averages = {
                "Recall@5": sum(r["Recall@5"] for r in domain_results.values()) / len(domain_results),
                "Recall@10": sum(r["Recall@10"] for r in domain_results.values()) / len(domain_results),
                "nDCG@5": sum(r["nDCG@5"] for r in domain_results.values()) / len(domain_results),
                "nDCG@10": sum(r["nDCG@10"] for r in domain_results.values()) / len(domain_results),
            }
        else:
            averages = {"Recall@5": 0, "Recall@10": 0, "nDCG@5": 0, "nDCG@10": 0}
    
    # Build final results structure
    final_results = {
        "experiment_name": exp_name,
        "model_path": str(model_path),
        "config": config,
        "domain_results": domain_results,
        "averages": averages,
    }
    
    # Add baseline comparison if we have baseline scores
    baseline_r10 = 0.3591  # From evaluation.log baseline
    baseline_n10 = 0.2810
    if averages["Recall@10"] and averages["nDCG@10"]:
        final_results["baseline_comparison"] = {
            "Recall@5": {
                "ours": averages["Recall@5"],
                "baseline": 0.2944,  # From evaluation.log
                "improvement": averages["Recall@5"] - 0.2944
            },
            "Recall@10": {
                "ours": averages["Recall@10"],
                "baseline": baseline_r10,
                "improvement": averages["Recall@10"] - baseline_r10
            },
            "nDCG@5": {
                "ours": averages["nDCG@5"],
                "baseline": 0.2509,  # From evaluation.log
                "improvement": averages["nDCG@5"] - 0.2509
            },
            "nDCG@10": {
                "ours": averages["nDCG@10"],
                "baseline": baseline_n10,
                "improvement": averages["nDCG@10"] - baseline_n10
            }
        }
    
    return final_results

def find_python_executable() -> str:
    """Find the correct Python executable (prefer venv if available)."""
    # Check for venv Python first
    venv_python = pathlib.Path("venv/bin/python3")
    if venv_python.exists():
        return str(venv_python.absolute())
    
    # Check for venv/bin/python
    venv_python2 = pathlib.Path("venv/bin/python")
    if venv_python2.exists():
        return str(venv_python2.absolute())
    
    # Fall back to system Python
    return sys.executable

def evaluate_experiment(exp_name: str, model_path: pathlib.Path, is_domain_specific: bool = False) -> bool:
    """Evaluate a single experiment."""
    logging.info(f"{'='*80}")
    logging.info(f"Evaluating: {exp_name}")
    logging.info(f"Model path: {model_path}")
    logging.info(f"{'='*80}")
    
    results_file = EXPERIMENTS_DIR / exp_name / "results.json"
    config_file = EXPERIMENTS_DIR / exp_name / "config.json"
    
    # Build evaluation command (save to temp file first)
    temp_results = EXPERIMENTS_DIR / exp_name / "temp_results.json"
    
    # Use venv Python if available
    python_exe = find_python_executable()
    
    if is_domain_specific:
        # Domain-specific models are evaluated differently
        # Extract domain from experiment name
        domain_match = re.search(r'domain_specific_(\w+)', exp_name)
        if domain_match:
            domain = domain_match.group(1)
            cmd = [
                python_exe,
                "evaluate_advanced_models.py",
                "--model_path", str(model_path),
                "--domains", domain,
                "--output", str(temp_results)
            ]
        else:
            logging.error(f"Cannot determine domain for {exp_name}")
            return False
    else:
        # Multi-domain model
        cmd = [
            python_exe,
            "evaluate_advanced_models.py",
            "--model_path", str(model_path),
            "--output", str(temp_results)
        ]
    
    logging.info(f"Running: {' '.join(cmd)}")
    
    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False  # Don't raise on non-zero exit
        )
        
        if result.returncode == 0:
            logging.info(f"✅ Evaluation completed for {exp_name}")
            
            # Check if temp results file was created
            if temp_results.exists():
                # Load and convert format
                try:
                    with open(temp_results) as f:
                        eval_results = json.load(f)
                    
                    # Convert to expected format
                    final_results = convert_results_format(exp_name, eval_results, config_file, model_path)
                    
                    # Save final results
                    with open(results_file, 'w') as f:
                        json.dump(final_results, f, indent=2)
                    
                    logging.info(f"   Results saved to: {results_file}")
                    
                    # Clean up temp file
                    temp_results.unlink()
                    
                    return True
                except Exception as e:
                    logging.error(f"   Error processing results: {e}")
                    return False
            else:
                logging.warning(f"   ⚠️  Results file not found at {temp_results}")
                # Show output for debugging
                if result.stdout:
                    logging.info(f"   stdout: {result.stdout[:500]}")
                return False
        else:
            logging.error(f"❌ Evaluation failed for {exp_name}")
            logging.error(f"   Exit code: {result.returncode}")
            if result.stderr:
                logging.error(f"   Error: {result.stderr[:500]}")
            if result.stdout:
                logging.error(f"   Output: {result.stdout[:500]}")
            return False
            
    except Exception as e:
        logging.error(f"❌ Exception during evaluation of {exp_name}: {e}")
        import traceback
        logging.error(traceback.format_exc())
        return False

def find_completed_experiments() -> List[Dict]:
    """Find all Phase 4 experiments that have completed training but not been evaluated."""
    completed = []
    
    # All Phase 4 experiments
    phase4_experiments = [
        'phase4_hard_negatives_cosine',
        'phase4_hard_negatives_triplet',
        'phase4_hard_negatives_5neg',
        'phase4_bge_large',
        'phase4_domain_specific_clapnq',
        'phase4_domain_specific_fiqa',
        'phase4_domain_specific_govt',
        'phase4_domain_specific_cloud',
    ]
    
    for exp_name in phase4_experiments:
        results_file = EXPERIMENTS_DIR / exp_name / "results.json"
        
        # Skip if already evaluated
        if results_file.exists():
            logging.debug(f"Skipping {exp_name}: already has results.json")
            continue
        
        # Check if training completed
        training_done, final_epoch = check_training_completed(exp_name)
        
        if training_done:
            # Find model path
            model_path = get_model_path(exp_name)
            
            if model_path:
                is_domain_specific = 'domain_specific' in exp_name
                completed.append({
                    'name': exp_name,
                    'model_path': model_path,
                    'final_epoch': final_epoch,
                    'is_domain_specific': is_domain_specific
                })
                logging.info(f"✅ Found completed training: {exp_name} (epoch {final_epoch})")
            else:
                logging.warning(f"⚠️  Training completed for {exp_name} but model path not found")
        else:
            # Check if it's currently running
            import subprocess
            ps_result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
            if exp_name in ps_result.stdout:
                logging.debug(f"Skipping {exp_name}: still running")
            else:
                logging.debug(f"Skipping {exp_name}: training not completed")
    
    return completed

def main():
    """Main function to evaluate all completed experiments."""
    logging.info("="*80)
    logging.info("Phase 4 Automatic Evaluation Script")
    logging.info("="*80)
    logging.info("")
    
    # Find completed experiments
    logging.info("Scanning for completed Phase 4 experiments...")
    completed = find_completed_experiments()
    
    if not completed:
        logging.info("✅ No experiments need evaluation. All are either:")
        logging.info("   - Already evaluated (have results.json)")
        logging.info("   - Still training")
        logging.info("   - Not started")
        return 0
    
    logging.info(f"\n📋 Found {len(completed)} experiment(s) ready for evaluation:")
    for exp in completed:
        logging.info(f"   • {exp['name']} (epoch {exp['final_epoch']})")
    
    logging.info("")
    
    # Evaluate each experiment
    results = {}
    for i, exp in enumerate(completed, 1):
        logging.info(f"\n[{i}/{len(completed)}] Processing {exp['name']}...")
        
        success = evaluate_experiment(
            exp['name'],
            exp['model_path'],
            exp['is_domain_specific']
        )
        
        results[exp['name']] = {
            'success': success,
            'timestamp': datetime.now().isoformat()
        }
        
        if success:
            logging.info(f"✅ Successfully evaluated {exp['name']}")
        else:
            logging.error(f"❌ Failed to evaluate {exp['name']}")
    
    # Summary
    logging.info("")
    logging.info("="*80)
    logging.info("EVALUATION SUMMARY")
    logging.info("="*80)
    
    successful = [name for name, res in results.items() if res['success']]
    failed = [name for name, res in results.items() if not res['success']]
    
    if successful:
        logging.info(f"\n✅ Successfully evaluated ({len(successful)}):")
        for name in successful:
            logging.info(f"   • {name}")
    
    if failed:
        logging.info(f"\n❌ Failed to evaluate ({len(failed)}):")
        for name in failed:
            logging.info(f"   • {name}")
        logging.info("\n   Check logs above for error details")
    
    logging.info("")
    logging.info("="*80)
    
    return 0 if not failed else 1

if __name__ == "__main__":
    sys.exit(main())

