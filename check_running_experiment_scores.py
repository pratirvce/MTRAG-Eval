#!/usr/bin/env python3
"""
Check nDCG@10 scores for running experiments.
Shows scores from results.json files if available, and reference scores from similar experiments.
"""

import json
import os
from pathlib import Path
from datetime import datetime

# Running experiments
RUNNING_EXPERIMENTS = [
    'tier1_mixture_experts_fixed',
    'tier1_curriculum_contrastive_fixed',
    'tier1_multi_turn_state_tracking_fixed',
    'tier1_causal_inference_fixed',
    'tier1_differentiable_retrieval_fixed',
    'tier1_learned_indices_fixed',
    'tier1_uncertainty_aware_fixed',
    'tier1_neural_ndcg',
]

# Reference experiments (non-fixed versions)
REFERENCE_EXPERIMENTS = {
    'tier1_causal_inference_fixed': 'tier1_causal_inference',
    'tier1_neural_ndcg': None,  # check if exists
}

base_path = Path('experiments/retrieval')

def get_score_from_results(results_file):
    """Extract nDCG@10 score from results.json file."""
    try:
        with open(results_file, 'r') as f:
            data = json.load(f)
            if 'average' in data and 'nDCG@10' in data['average']:
                return data['average']['nDCG@10']
            elif 'nDCG@10' in data:
                return data['nDCG@10']
    except Exception as e:
        return None
    return None

print("=" * 80)
print("nDCG@10 SCORES FOR RUNNING EXPERIMENTS")
print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

has_scores = False
no_scores = []

for exp_name in RUNNING_EXPERIMENTS:
    exp_path = base_path / exp_name
    results_file = exp_path / 'results.json'
    
    print(f"📊 {exp_name}")
    print("-" * 80)
    
    if results_file.exists():
        score = get_score_from_results(results_file)
        if score is not None:
            has_scores = True
            # Check status
            try:
                with open(results_file, 'r') as f:
                    data = json.load(f)
                    status = data.get('status', 'completed')
                    if status == 'failed':
                        error = data.get('error', 'Unknown error')
                        error_short = (error[:70] + '...') if len(error) > 70 else error
                        print(f"   ❌ Status: FAILED")
                        print(f"   Error: {error_short}")
                        print(f"   nDCG@10: {score:.4f} (from partial results)")
                    else:
                        print(f"   ✅ Status: COMPLETED")
                        print(f"   nDCG@10: {score:.4f}")
                        
                        # Show domain breakdown if available
                        if 'domains' in data:
                            print(f"   Domain breakdown:")
                            for domain, domain_data in data['domains'].items():
                                if isinstance(domain_data, dict) and 'nDCG@10' in domain_data:
                                    print(f"     {domain}: {domain_data['nDCG@10']:.4f}")
            except:
                print(f"   ✅ nDCG@10: {score:.4f}")
        else:
            print(f"   ⏳ Status: Still running (results.json exists but no score yet)")
            no_scores.append(exp_name)
    else:
        print(f"   ⏳ Status: Still running (no results.json found)")
        no_scores.append(exp_name)
        
        # Check for reference scores from non-fixed versions
        ref_name = REFERENCE_EXPERIMENTS.get(exp_name)
        if ref_name:
            ref_results = base_path / ref_name / 'results.json'
            if ref_results.exists():
                ref_score = get_score_from_results(ref_results)
                if ref_score is not None:
                    print(f"   📌 Reference (non-fixed version '{ref_name}'): nDCG@10 = {ref_score:.4f}")
    
    print()

if no_scores:
    print("=" * 80)
    print("REFERENCE SCORES")
    print("=" * 80)
    print()
    print("Since the above experiments are still running, here are reference scores")
    print("from similar completed experiments:")
    print()
    
    # Find similar completed experiments
    similar_exps = [
        ('tier1_causal_inference', 'tier1_causal_inference_fixed'),
        ('tier1_cross_domain_transfer', None),
        ('tier1_explainable_retrieval', None),
        ('tier1_adversarial_robustness', None),
        ('tier1_temporal_attention', None),
        ('tier1_graph_aware_retrieval', None),
        ('tier1_hybrid_lexical_semantic', None),
        ('tier1_momentum_contrastive', None),
    ]
    
    print(f"{'Experiment':<50} {'nDCG@10':<10} {'Recall@10'}")
    print("-" * 80)
    
    for exp_name, _ in similar_exps:
        results_file = base_path / exp_name / 'results.json'
        if results_file.exists():
            score = get_score_from_results(results_file)
            if score is not None:
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        recall = data.get('average', {}).get('Recall@10', 'N/A')
                        if recall != 'N/A':
                            print(f"{exp_name:<50} {score:<10.4f} {recall:.4f}")
                        else:
                            print(f"{exp_name:<50} {score:<10.4f}")
                except:
                    print(f"{exp_name:<50} {score:<10.4f}")

print()
print("=" * 80)
if has_scores:
    print("✅ Some experiments have completed and have scores available")
else:
    print("⏳ All listed experiments are still running - scores will be available when they complete")
print("=" * 80)

