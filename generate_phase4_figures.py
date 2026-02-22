#!/usr/bin/env python3
"""
Generate Phase 4 visualization figures for the LaTeX report
"""

import matplotlib.pyplot as plt
import numpy as np
import json
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams['figure.figsize'] = (6, 4)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 10
plt.rcParams['axes.titlesize'] = 11
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9

# Data
domains = ['CLAPNQ', 'FIQA', 'GOVT', 'CLOUD']

# Phase 2 Augmentation results
phase2_aug = {
    'CLAPNQ': {'R@10': 0.5816, 'nDCG@10': 0.4786},
    'FIQA': {'R@10': 0.4911, 'nDCG@10': 0.4024},
    'GOVT': {'R@10': 0.5072, 'nDCG@10': 0.4041},
    'CLOUD': {'R@10': 0.4598, 'nDCG@10': 0.3543}
}

# Phase 4 Domain-Specific results
phase4_domain = {
    'CLAPNQ': {'R@10': 0.6016, 'nDCG@10': 0.4981},
    'FIQA': {'R@10': 0.5119, 'nDCG@10': 0.4026},
    'GOVT': {'R@10': 0.5511, 'nDCG@10': 0.4628},
    'CLOUD': {'R@10': 0.5293, 'nDCG@10': 0.4104}
}

# Phase 4 full metrics
phase4_full = {
    'CLAPNQ': {'R@5': 0.4529, 'R@10': 0.6016, 'nDCG@5': 0.4399, 'nDCG@10': 0.4981},
    'FIQA': {'R@5': 0.3869, 'R@10': 0.5119, 'nDCG@5': 0.3500, 'nDCG@10': 0.4026},
    'GOVT': {'R@5': 0.4435, 'R@10': 0.5511, 'nDCG@5': 0.4210, 'nDCG@10': 0.4628},
    'CLOUD': {'R@5': 0.4293, 'R@10': 0.5293, 'nDCG@5': 0.3643, 'nDCG@10': 0.4104}
}

# All phases average results
all_phases = {
    'Baseline': {'R@10': 0.3800, 'nDCG@10': 0.3000},
    'Phase 1\n(epochs5)': {'R@10': 0.4693, 'nDCG@10': 0.3671},
    'Phase 2\n(augmentation)': {'R@10': 0.5099, 'nDCG@10': 0.4098},
    'Phase 3\n(hybrid)': {'R@10': 0.3388, 'nDCG@10': 0.2709},
    'Phase 4\n(domain-specific)': {'R@10': 0.5485, 'nDCG@10': 0.4435}
}

output_dir = Path('report_figures')
output_dir.mkdir(exist_ok=True)

# Figure 1: Domain Comparison
print("Generating phase4_domain_comparison.png...")
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(domains))
width = 0.35

phase2_r10 = [phase2_aug[d]['R@10'] for d in domains]
phase4_r10 = [phase4_domain[d]['R@10'] for d in domains]

bars1 = ax.bar(x - width/2, phase2_r10, width, label='Phase 2: Augmentation', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, phase4_r10, width, label='Phase 4: Domain-Specific', color='#2ecc71', alpha=0.8)

# Add improvement labels
improvements = [3.4, 4.2, 8.7, 15.1]
for i, (b1, b2, imp) in enumerate(zip(bars1, bars2, improvements)):
    height = max(b1.get_height(), b2.get_height())
    ax.text(b2.get_x() + b2.get_width()/2., height + 0.01,
            f'+{imp}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Domain')
ax.set_ylabel('Recall@10')
ax.set_title('Domain-Specific vs Multi-Domain Model Performance')
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.legend()
ax.set_ylim([0, 0.65])
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / 'phase4_domain_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 2: Phase 4 Results (all metrics)
print("Generating phase4_results.png...")
fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(domains))
width = 0.2

metrics = ['R@5', 'R@10', 'nDCG@5', 'nDCG@10']
colors = ['#e74c3c', '#3498db', '#f39c12', '#2ecc71']

for i, (metric, color) in enumerate(zip(metrics, colors)):
    values = [phase4_full[d][metric] for d in domains]
    ax.bar(x + i*width, values, width, label=metric, color=color, alpha=0.8)

ax.set_xlabel('Domain')
ax.set_ylabel('Score')
ax.set_title('Phase 4 Domain-Specific Model Results (All Metrics)')
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(domains)
ax.legend()
ax.set_ylim([0, 0.65])
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / 'phase4_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 3: Transfer Learning Approach (diagram)
print("Generating transfer_learning_approach.png...")
fig, ax = plt.subplots(figsize=(10, 6))
ax.axis('off')

# Stage 1 box
stage1_box = plt.Rectangle((1, 4), 2, 1, linewidth=2, edgecolor='#3498db', facecolor='#ecf0f1', alpha=0.7)
ax.add_patch(stage1_box)
ax.text(2, 4.5, 'Stage 1: Multi-Domain\nPre-training\n(phase1_epochs5)', 
        ha='center', va='center', fontsize=10, fontweight='bold')

# Arrows to Stage 2
for i, domain in enumerate(domains):
    y_pos = 2.5 - i * 0.5
    ax.arrow(3, 4.5, 1, y_pos - 4.5, head_width=0.15, head_length=0.1, 
             fc='#2ecc71', ec='#2ecc71', linewidth=1.5)
    
    # Stage 2 boxes
    stage2_box = plt.Rectangle((4.5, y_pos - 0.3), 1.5, 0.6, 
                               linewidth=1.5, edgecolor='#2ecc71', facecolor='#d5f4e6', alpha=0.7)
    ax.add_patch(stage2_box)
    ax.text(5.25, y_pos, f'Stage 2:\n{domain}\nDomain-Specific', 
            ha='center', va='center', fontsize=8)

ax.text(5.25, 3.2, 'Domain-Specific\nFine-tuning', 
        ha='center', va='center', fontsize=11, fontweight='bold', color='#27ae60')

ax.set_xlim([0, 7])
ax.set_ylim([0, 5.5])
ax.set_title('Two-Stage Transfer Learning Approach', fontsize=12, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(output_dir / 'transfer_learning_approach.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 4: All Phases Comparison
print("Generating all_phases_comparison.png...")
fig, ax = plt.subplots(figsize=(8, 5))
phases = list(all_phases.keys())
x = np.arange(len(phases))
width = 0.35

r10_values = [all_phases[p]['R@10'] for p in phases]
ndcg10_values = [all_phases[p]['nDCG@10'] for p in phases]

bars1 = ax.bar(x - width/2, r10_values, width, label='Recall@10', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, ndcg10_values, width, label='nDCG@10', color='#2ecc71', alpha=0.8)

# Highlight Phase 4
ax.bar(x[-1] - width/2, r10_values[-1], width, color='#e74c3c', alpha=0.9)
ax.bar(x[-1] + width/2, ndcg10_values[-1], width, color='#e74c3c', alpha=0.9)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

ax.set_xlabel('Phase')
ax.set_ylabel('Score')
ax.set_title('Performance Progression Across All Phases')
ax.set_xticks(x)
ax.set_xticklabels(phases, rotation=0, ha='center')
ax.legend()
ax.set_ylim([0, 0.6])
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / 'all_phases_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# Figure 5: Domain Improvements
print("Generating domain_improvements.png...")
fig, ax = plt.subplots(figsize=(7, 4.5))
x = np.arange(len(domains))
width = 0.35

r10_improvements = [3.4, 4.2, 8.7, 15.1]
ndcg10_improvements = [4.1, 0.0, 14.5, 15.8]

bars1 = ax.bar(x - width/2, r10_improvements, width, label='Recall@10 Improvement', color='#3498db', alpha=0.8)
bars2 = ax.bar(x + width/2, ndcg10_improvements, width, label='nDCG@10 Improvement', color='#2ecc71', alpha=0.8)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        if height > 0:
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                    f'+{height:.1f}%', ha='center', va='bottom', fontsize=8, fontweight='bold')

ax.set_xlabel('Domain')
ax.set_ylabel('Improvement (%)')
ax.set_title('Domain-Specific Model Improvements Over Multi-Domain Baseline')
ax.set_xticks(x)
ax.set_xticklabels(domains)
ax.legend()
ax.set_ylim([0, 18])
ax.grid(axis='y', alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(output_dir / 'domain_improvements.png', dpi=300, bbox_inches='tight')
plt.close()

print("\n✅ All Phase 4 figures generated successfully!")
print(f"Figures saved to: {output_dir}/")

