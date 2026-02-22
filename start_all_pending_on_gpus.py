#!/usr/bin/env python3
"""
Start all pending experiments on available GPUs.
Finds free GPUs and distributes pending experiments across them.
"""

import subprocess
import pathlib
import json
import time
import logging
import sys
import re
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger(__name__)

def get_available_gpus(threshold_util=10, threshold_mem_pct=10) -> List[int]:
    """Get list of available GPU IDs."""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        available_gpus = []
        for line in result.stdout.strip().split('\n'):
            if ',' in line:
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    try:
                        gpu_idx = int(parts[0])
                        util = int(parts[1])
                        mem_used = int(parts[2])
                        mem_total = int(parts[3])
                        mem_pct = (mem_used / mem_total) * 100 if mem_total > 0 else 0
                        
                        if util < threshold_util and mem_pct < threshold_mem_pct:
                            available_gpus.append(gpu_idx)
                            logger.info(f"GPU {gpu_idx}: {util}% util, {mem_pct:.1f}% mem - AVAILABLE")
                        else:
                            logger.debug(f"GPU {gpu_idx}: {util}% util, {mem_pct:.1f}% mem - BUSY")
                    except ValueError:
                        continue
        return available_gpus
    except Exception as e:
        logger.error(f"Error checking GPUs: {e}")
        return []

def get_pending_experiments() -> List[str]:
    """Get list of pending experiments by checking directories."""
    try:
        experiments_dir = pathlib.Path(__file__).parent / "experiments" / "retrieval"
        if not experiments_dir.exists():
            logger.error(f"Experiments directory not found: {experiments_dir}")
            return []
        
        experiment_dirs = [d for d in experiments_dir.iterdir() if d.is_dir()]
        pending = []
        
        for exp_dir in experiment_dirs:
            exp_name = exp_dir.name
            if exp_name in ['logs', 'test']:  # Skip non-experiment directories
                continue
            
            # Check if experiment is completed
            results_file = exp_dir / "results.json"
            checkpoint_file = exp_dir / "checkpoints" / "checkpoint.json"
            
            is_complete = False
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        results = json.load(f)
                        # Check if results have valid scores
                        if 'average' in results:
                            avg = results['average']
                            if avg.get('Recall@10', 0) > 0 or avg.get('nDCG@10', 0) > 0:
                                is_complete = True
                except:
                    pass
            
            # Check if running
            is_running = is_experiment_running(exp_name)
            
            # Consider pending if not complete and not running
            if not is_complete and not is_running:
                # Check checkpoint to see if it's in progress
                has_progress = False
                if checkpoint_file.exists():
                    try:
                        with open(checkpoint_file, 'r') as f:
                            checkpoint = json.load(f)
                            # Check if any domain has progress
                            for domain_data in checkpoint.values():
                                if isinstance(domain_data, dict) and 'results' in domain_data:
                                    results = domain_data['results']
                                    if results.get('Recall@10', 0) > 0 or results.get('nDCG@10', 0) > 0:
                                        has_progress = True
                                        break
                    except:
                        pass
                
                # Include if no checkpoint or has progress (can resume)
                if not checkpoint_file.exists() or has_progress:
                    pending.append(exp_name)
        
        return sorted(pending)
    except Exception as e:
        logger.error(f"Error getting pending experiments: {e}", exc_info=True)
        return []

def is_experiment_running(exp_name: str) -> bool:
    """Check if experiment is already running."""
    try:
        result = subprocess.run(
            ['pgrep', '-f', exp_name],
            capture_output=True,
            text=True
        )
        return result.returncode == 0
    except:
        return False

def map_experiment_to_script(exp_name: str) -> Optional[str]:
    """Map experiment name to its training script."""
    # Pattern matching for experiment names to scripts
    mappings = {
        # Tier 1 experiments
        'tier1_cross_attention_query_document': 'train_cross_attention_retrieval.py',
        'tier1_cross_attention_rerun': 'train_cross_attention_retrieval.py',
        'tier1_learning_to_rank_listwise': 'train_learning_to_rank_tier1.py',
        'tier1_qdit_transformer': 'train_qdit_transformer_tier1.py',
        'tier1_pseudo_relevance_feedback': 'train_pseudo_relevance_feedback_tier1.py',
        'tier1_iterative_refinement_improved': 'train_iterative_refinement_improved_tier1.py',
        'tier1_contrastive_learning': 'train_contrastive_learning_tier1.py',
        'tier1_enhanced_contrastive_hardnegatives': 'train_enhanced_contrastive_tier1.py',
        'tier1_cross_encoder_domain_specific': 'train_cross_encoder_domain_specific_tier1.py',
        'tier1_cross_encoder_evaluation': 'train_cross_encoder_evaluation.py',
        'tier1_cross_encoder_finetuned': 'train_cross_encoder_finetuned_tier1.py',
        'tier1_cross_encoder_large': 'train_cross_encoder_large_tier1.py',
        'tier1_ensemble_best_methods': 'train_ensemble_best_tier1.py',
        'tier1_hierarchical_multigranularity': 'train_hierarchical_multigranularity_tier1.py',
        'tier1_llm_query_expansion': 'train_llm_query_expansion_tier1.py',
        'tier1_multistage_2stage': 'train_multistage_tier1.py',
        'tier1_multistage_3stage_finetuned': 'train_multistage_tier1.py',
        
        # Best paper experiments
        'best_paper_adversarial_curriculum': 'train_adversarial_curriculum_tier1.py',
        'best_paper_graph_aware_retrieval': 'train_graph_aware_tier1.py',
        'best_paper_hierarchical_routing': 'train_hierarchical_routing_tier1.py',
        'best_paper_large_model_finetuning': 'train_large_model_tier1.py',
        'best_paper_learned_rrf': 'train_learned_rrf_tier1.py',
        'best_paper_llm_distillation': 'train_llm_distillation_tier1.py',
        'best_paper_meta_learning': 'train_meta_learning_tier1.py',
        'best_paper_multitask_retrieval': 'train_multitask_retrieval_tier1.py',
        'best_paper_rl_adaptive_retrieval': 'train_rl_adaptive_tier1.py',
        'best_paper_temporal_memory': 'train_temporal_memory_tier1.py',
        
        # Tier 2
        'tier2_ensemble_advanced': 'train_ensemble.py',
        'tier2_multistage_3stage': 'train_multistage_retrieval.py',
        
        # Phase experiments
        'multistage_retrieval': 'train_multistage_retrieval.py',
    }
    
    # Direct mapping
    if exp_name in mappings:
        return mappings[exp_name]
    
    # Pattern-based mapping
    if 'cross_attention' in exp_name.lower():
        return 'train_cross_attention_retrieval.py'
    elif 'learning_to_rank' in exp_name.lower() or 'listwise' in exp_name.lower():
        return 'train_learning_to_rank_tier1.py'
    elif 'qdit' in exp_name.lower():
        return 'train_qdit_transformer_tier1.py'
    elif 'pseudo_relevance' in exp_name.lower() or 'prf' in exp_name.lower():
        return 'train_pseudo_relevance_feedback_tier1.py'
    elif 'iterative_refinement' in exp_name.lower():
        return 'train_iterative_refinement_improved_tier1.py'
    elif 'contrastive' in exp_name.lower():
        if 'enhanced' in exp_name.lower() or 'hardnegatives' in exp_name.lower():
            return 'train_enhanced_contrastive_tier1.py'
        return 'train_contrastive_learning_tier1.py'
    elif 'cross_encoder' in exp_name.lower():
        if 'domain_specific' in exp_name.lower():
            return 'train_cross_encoder_domain_specific_tier1.py'
        elif 'large' in exp_name.lower():
            return 'train_cross_encoder_large_tier1.py'
        elif 'finetuned' in exp_name.lower():
            return 'train_cross_encoder_finetuned_tier1.py'
        elif 'evaluation' in exp_name.lower():
            return 'train_cross_encoder_evaluation.py'
        return 'train_cross_encoder_finetuned_tier1.py'
    elif 'ensemble' in exp_name.lower():
        if 'best' in exp_name.lower():
            return 'train_ensemble_best_tier1.py'
        return 'train_ensemble.py'
    elif 'hierarchical' in exp_name.lower():
        if 'routing' in exp_name.lower():
            return 'train_hierarchical_routing_tier1.py'
        return 'train_hierarchical_multigranularity_tier1.py'
    elif 'llm_query_expansion' in exp_name.lower() or 'query_expansion' in exp_name.lower():
        return 'train_llm_query_expansion_tier1.py'
    elif 'multistage' in exp_name.lower():
        return 'train_multistage_tier1.py'
    elif 'adversarial_curriculum' in exp_name.lower():
        return 'train_adversarial_curriculum_tier1.py'
    elif 'graph_aware' in exp_name.lower():
        return 'train_graph_aware_tier1.py'
    elif 'learned_rrf' in exp_name.lower():
        return 'train_learned_rrf_tier1.py'
    elif 'llm_distillation' in exp_name.lower():
        return 'train_llm_distillation_tier1.py'
    elif 'meta_learning' in exp_name.lower():
        return 'train_meta_learning_tier1.py'
    elif 'multitask' in exp_name.lower():
        return 'train_multitask_retrieval_tier1.py'
    elif 'rl_adaptive' in exp_name.lower():
        return 'train_rl_adaptive_tier1.py'
    elif 'temporal_memory' in exp_name.lower():
        return 'train_temporal_memory_tier1.py'
    elif 'large_model' in exp_name.lower():
        return 'train_large_model_tier1.py'
    
    logger.warning(f"Could not map experiment '{exp_name}' to a script")
    return None

def get_config_for_experiment(exp_name: str, output_dir: pathlib.Path) -> Optional[dict]:
    """Create or load config for experiment."""
    config_file = output_dir / "config.json"
    
    # If config exists, use it
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except:
            pass
    
    # Create default config
    default_config = {
        "model_path": "BAAI/bge-base-en-v1.5",
        "output_dir": str(output_dir),
        "domains": ["clapnq", "fiqa", "govt", "cloud"],
        "use_data_splits": True,
        "top_k": 100,
        "batch_size": 32,
        "doc_batch_size": 1000,
        "resume": True,
        "use_attention": True
    }
    
    # Save default config
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(config_file, 'w') as f:
        json.dump(default_config, f, indent=2)
    
    return default_config

def start_experiment(exp_name: str, script: str, gpu_id: int, resume: bool = True) -> bool:
    """Start an experiment on a specific GPU."""
    output_dir = pathlib.Path("experiments/retrieval") / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / "training.log"
    
    # Check if script exists
    script_path = pathlib.Path(script)
    if not script_path.exists():
        logger.error(f"Script not found: {script}")
        return False
    
    # Determine Python command
    venv_python = pathlib.Path("venv/bin/python3")
    if venv_python.exists():
        python_cmd = str(venv_python)
    else:
        python_cmd = "python3"
    
    # Build command
    if script == 'train_cross_attention_retrieval.py':
        # This script uses --config and --gpu_id
        config_file = output_dir / "config.json"
        if not config_file.exists():
            get_config_for_experiment(exp_name, output_dir)
        
        cmd = [
            python_cmd,
            script,
            '--config', str(config_file),
            '--gpu_id', str(gpu_id)
        ]
        if not resume:
            cmd.append('--no-resume')
    else:
        # Standard tier1 scripts use --experiment_name, --gpu, --output_dir
        cmd = [
            python_cmd,
            script,
            '--experiment_name', exp_name,
            '--gpu', str(gpu_id),
            '--output_dir', str(output_dir)
        ]
        if resume:
            cmd.append('--resume')
        else:
            cmd.append('--no-resume')
    
    logger.info(f"Starting {exp_name} on GPU {gpu_id}...")
    logger.info(f"Command: {' '.join(cmd)}")
    
    # Write to log file
    with open(log_file, 'a') as f:
        f.write(f"\n{'='*80}\n")
        f.write(f"Started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"GPU: {gpu_id}\n")
        f.write(f"Command: {' '.join(cmd)}\n")
        f.write(f"{'='*80}\n")
    
    # Start process in background
    try:
        with open(log_file, 'a') as log_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=subprocess.STDOUT,
                cwd=pathlib.Path(__file__).parent
            )
        
        logger.info(f"✅ Started {exp_name} on GPU {gpu_id} (PID: {process.pid})")
        return True
    except Exception as e:
        logger.error(f"❌ Failed to start {exp_name}: {e}")
        return False

def main():
    """Main function to start all pending experiments."""
    logger.info("="*80)
    logger.info("Starting All Pending Experiments on Free GPUs")
    logger.info("="*80)
    
    # Get available GPUs
    available_gpus = get_available_gpus()
    if not available_gpus:
        logger.error("No free GPUs available!")
        return
    
    logger.info(f"Found {len(available_gpus)} available GPUs: {available_gpus}")
    
    # Get pending experiments
    pending = get_pending_experiments()
    logger.info(f"Found {len(pending)} pending experiments")
    
    if not pending:
        logger.info("No pending experiments to start")
        return
    
    # Filter out experiments that are already running
    truly_pending = []
    for exp in pending:
        if not is_experiment_running(exp):
            truly_pending.append(exp)
        else:
            logger.info(f"Skipping {exp} - already running")
    
    logger.info(f"Starting {len(truly_pending)} experiments...")
    
    # Start experiments, distributing across GPUs
    started = 0
    gpu_index = 0
    
    for exp_name in truly_pending:
        if gpu_index >= len(available_gpus):
            logger.warning(f"All GPUs are now in use. {len(truly_pending) - started} experiments still pending.")
            break
        
        # Map to script
        script = map_experiment_to_script(exp_name)
        if not script:
            logger.warning(f"Skipping {exp_name} - could not determine script")
            continue
        
        # Check if script exists
        if not pathlib.Path(script).exists():
            logger.warning(f"Skipping {exp_name} - script not found: {script}")
            continue
        
        # Start experiment
        gpu_id = available_gpus[gpu_index]
        if start_experiment(exp_name, script, gpu_id, resume=True):
            started += 1
            gpu_index += 1
            time.sleep(2)  # Small delay between starts
    
    logger.info("="*80)
    logger.info(f"✅ Started {started} experiments across {gpu_index} GPUs")
    logger.info(f"   GPUs used: {available_gpus[:gpu_index]}")
    logger.info(f"   Remaining pending: {len(truly_pending) - started}")
    logger.info("="*80)

if __name__ == "__main__":
    main()

