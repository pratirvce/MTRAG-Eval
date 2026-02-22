"""
Run domain-specific experiments in parallel on separate GPUs
Each domain gets its own GPU for maximum efficiency
"""

import subprocess
import sys
import logging
import time
import pathlib
import json
from datetime import datetime
import threading
import os
import torch

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

DOMAINS = ["clapnq", "fiqa", "govt", "cloud"]
CONFIG_TEMPLATE = {
    "base_model": "BAAI/bge-base-en-v1.5",
    "epochs": 7,
    "batch_size": 32,
    "learning_rate": 1e-5,
    "use_pretrained_multi_domain": True,
    "pretrained_multi_domain_path": "./models/phase1_epochs5",
    "use_validation": True,
    "use_data_splits": True,
    "warmup_steps": 50,
    "evaluation_steps": 200,
    "save_best_model": True,
    "seed": 42
}

def get_available_gpus():
    """Get list of available GPUs."""
    if not torch.cuda.is_available():
        return []
    
    available = []
    for gpu_id in range(torch.cuda.device_count()):
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
                    util = int(parts[1])
                    mem = int(parts[2])
                    if util < 10 and mem < 100:
                        available.append(gpu_id)
        except:
            continue
    return available

def run_domain_on_gpu(domain: str, gpu_id: int):
    """Run domain-specific training on a specific GPU."""
    logging.info(f"Starting {domain} training on GPU {gpu_id}")
    
    # Create config
    exp_dir = pathlib.Path("experiments/retrieval") / f"domain_specific_{domain}_gpu{gpu_id}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    config = CONFIG_TEMPLATE.copy()
    config["domain"] = domain
    config["gpu_id"] = gpu_id
    config["output_path"] = f"./models/domain_specific_{domain}"
    
    config_file = exp_dir / "config.json"
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Prepare logs
    log_file = exp_dir / "training.log"
    error_log = exp_dir / "error.log"
    
    # Build command
    cmd = [
        sys.executable, "train_domain_specific_bge.py",
        "--domain", domain,
        "--config", str(config_file)
    ]
    
    # Set CUDA device
    env = os.environ.copy()
    env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    
    try:
        with open(log_file, 'w') as log_f, open(error_log, 'w') as err_f:
            process = subprocess.Popen(
                cmd,
                stdout=log_f,
                stderr=err_f,
                cwd=pathlib.Path(".").absolute(),
                env=env
            )
        
        logging.info(f"{domain} started on GPU {gpu_id} (PID: {process.pid})")
        return_code = process.wait()
        
        if return_code == 0:
            logging.info(f"✅ {domain} completed successfully on GPU {gpu_id}")
            return True
        else:
            logging.error(f"❌ {domain} failed on GPU {gpu_id} (return code: {return_code})")
            return False
            
    except Exception as e:
        logging.error(f"Error running {domain} on GPU {gpu_id}: {e}")
        return False

def main():
    available_gpus = get_available_gpus()
    logging.info(f"Available GPUs: {available_gpus}")
    
    if len(available_gpus) < len(DOMAINS):
        logging.warning(f"Only {len(available_gpus)} GPUs available, but {len(DOMAINS)} domains to train")
        logging.info("Will run domains sequentially if needed")
    
    threads = []
    gpu_idx = 0
    
    for domain in DOMAINS:
        if gpu_idx < len(available_gpus):
            gpu_id = available_gpus[gpu_idx]
            thread = threading.Thread(
                target=run_domain_on_gpu,
                args=(domain, gpu_id),
                daemon=False
            )
            thread.start()
            threads.append((thread, domain, gpu_id))
            gpu_idx += 1
        else:
            # Wait for a GPU to free up
            logging.info(f"Waiting for GPU for {domain}...")
            while True:
                available = get_available_gpus()
                if available:
                    gpu_id = available[0]
                    thread = threading.Thread(
                        target=run_domain_on_gpu,
                        args=(domain, gpu_id),
                        daemon=False
                    )
                    thread.start()
                    threads.append((thread, domain, gpu_id))
                    break
                time.sleep(10)
    
    # Wait for all to complete
    logging.info("Waiting for all domain experiments to complete...")
    for thread, domain, gpu_id in threads:
        thread.join()
        logging.info(f"{domain} on GPU {gpu_id} finished")
    
    logging.info("✅ All domain-specific experiments completed!")

if __name__ == "__main__":
    main()


