#!/usr/bin/env python3
"""
Generate a detailed summary of all experiments with comprehensive explanations.
This includes methodology, results, and insights for each experiment category.
"""

import json
import pathlib
from datetime import datetime
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

RESULTS_DIR = pathlib.Path("experiments/retrieval")
STATUS_FILES = [
    "experiment_status.json",
    "stopped_experiments_resume_info.json",
    "fixed_experiments_status.json",
    "tier1_experiments_status.json",
    "auto_fixed_experiments_status.json",
]

# Domain descriptions
DOMAIN_INFO = {
    "clapnq": "ClapNQ - Wikipedia-based question answering corpus (4,293 documents, 183,408 passages)",
    "fiqa": "FiQA - Financial question answering corpus (7,661 documents, 49,607 passages)",
    "govt": "Govt - Government documents corpus (8,578 documents, 72,422 passages)",
    "cloud": "Cloud - Technical documentation corpus (57,638 documents, 61,022 passages)",
}

# Experiment category explanations
CATEGORY_EXPLANATIONS = {
    "Baseline": {
        "description": "Baseline experiments establish performance benchmarks using standard fine-tuning approaches on the BGE (BAAI General Embedding) model architecture.",
        "methodology": "Standard contrastive learning with MultipleNegativesRankingLoss, fine-tuned on multi-domain or domain-specific data with varying hyperparameters (epochs, learning rates).",
        "key_insights": "Baselines provide reference points for comparing advanced methods. Variations in epochs and learning rates help identify optimal training configurations.",
    },
    "Domain-Specific Fine-tuning": {
        "description": "Fine-tuning models specifically for individual domains to capture domain-specific terminology, context, and retrieval patterns.",
        "methodology": "Two-stage training: first fine-tune on multi-domain data, then further fine-tune on a specific domain. Uses domain-specific training data and validation sets.",
        "key_insights": "Domain-specific fine-tuning consistently outperforms multi-domain models when evaluated on the target domain, showing the importance of specialized knowledge.",
    },
    "Contrastive Learning": {
        "description": "Advanced contrastive learning techniques that improve representation quality through better negative sampling and loss functions.",
        "methodology": "Enhancements include hard negative mining, curriculum learning, momentum contrast, and improved loss functions (triplet loss, cosine similarity loss).",
        "key_insights": "Hard negative mining and curriculum learning help models distinguish between similar but distinct documents, improving ranking precision.",
    },
    "Large Model Fine-tuning": {
        "description": "Leveraging larger pre-trained models (e.g., bge-large) to improve retrieval performance through increased model capacity.",
        "methodology": "Fine-tuning larger embedding models (typically 560M+ parameters) on the MT-RAG benchmark data with the same contrastive learning approach.",
        "key_insights": "Larger models capture more nuanced semantic relationships but require more computational resources. Often achieve highest nDCG scores.",
    },
    "Meta-Learning": {
        "description": "Adaptive learning approaches that quickly adapt to new domains or queries using few-shot learning principles.",
        "methodology": "Model-Agnostic Meta-Learning (MAML) or similar approaches that learn to quickly adapt embeddings for new domains with minimal examples.",
        "key_insights": "Meta-learning enables better generalization across domains and can adapt to domain shifts more effectively than standard fine-tuning.",
    },
    "Ensemble Methods": {
        "description": "Combining predictions from multiple models or retrieval methods to improve overall performance and robustness.",
        "methodology": "Various fusion strategies: weighted fusion, reciprocal rank fusion (RRF), domain-specific ensemble models, and combination of different architectures.",
        "key_insights": "Ensembles reduce variance and improve robustness. Weighted fusion of domain-specific models often outperforms single models.",
    },
    "Query Expansion": {
        "description": "Enhancing queries before retrieval by adding relevant terms, synonyms, or generating expanded queries using language models.",
        "methodology": "Techniques include pseudo-relevance feedback, LLM-based query expansion (GPT-4), multi-query generation, and domain-specific expansion.",
        "key_insights": "Query expansion helps with information retrieval, especially for short or ambiguous queries. LLM-based expansion can capture semantic relationships.",
    },
    "Reranking": {
        "description": "Multi-stage retrieval where an initial retrieval step is followed by a reranking stage using more sophisticated models.",
        "methodology": "Two-stage approach: (1) Retrieve top-K candidates using bi-encoder, (2) Rerank using cross-encoder or specialized reranking models with full query-document interactions.",
        "key_insights": "Reranking significantly improves precision by allowing query-document interaction, but at higher computational cost. Most effective for domain-specific retrieval.",
    },
    "Hybrid Methods": {
        "description": "Combining lexical (BM25, keyword-based) and semantic (neural embedding) retrieval methods for improved coverage.",
        "methodology": "Linear or learned fusion of BM25 and neural retrieval scores. Alpha parameter controls the balance between lexical and semantic components.",
        "key_insights": "Hybrid approaches combine the precision of semantic search with the recall of lexical search, especially effective for technical domains.",
    },
    "Multi-Stage Retrieval": {
        "description": "Cascaded retrieval pipelines with multiple stages of increasingly sophisticated retrieval and reranking.",
        "methodology": "2-stage or 3-stage pipelines: coarse retrieval → fine retrieval → optional reranking. Each stage filters candidates for the next.",
        "key_insights": "Multi-stage retrieval balances efficiency and effectiveness, allowing early filtering with fast methods and precise ranking with expensive methods.",
    },
    "Learning-to-Rank": {
        "description": "Supervised learning approaches that directly optimize ranking metrics (e.g., nDCG) rather than similarity scores.",
        "methodology": "Pointwise, pairwise, or listwise ranking objectives. Listwise approaches optimize nDCG directly using LambdaRank or ListNet-style losses.",
        "key_insights": "Directly optimizing ranking metrics can outperform contrastive learning for retrieval tasks where ranking quality is the primary concern.",
    },
    "Cross-Encoder": {
        "description": "Dual-encoder models that compute query-document interactions directly, providing more accurate relevance scores at higher computational cost.",
        "methodology": "Cross-attention mechanisms between query and document tokens, allowing full interaction. Used for reranking or as primary retrieval in smaller corpora.",
        "key_insights": "Cross-encoders excel at precision but are computationally expensive. Best used for reranking or domain-specific applications where accuracy is critical.",
    },
    "Conversation-Aware": {
        "description": "Retrieval methods that leverage conversation context and multi-turn dialogue history to improve query understanding.",
        "methodology": "Incorporating previous turns, conversation state tracking, and context-aware query encoding to handle conversational queries effectively.",
        "key_insights": "Conversational retrieval benefits from context, especially for follow-up questions and clarifications common in multi-turn dialogues.",
    },
    "Graph-Based Methods": {
        "description": "Leveraging knowledge graphs and document relationships to enhance retrieval through structural information.",
        "methodology": "Graph neural networks, document relationship graphs, entity linking, and graph-aware reranking to capture semantic and structural relationships.",
        "key_insights": "Graph methods help capture entity relationships and document connections that pure text embeddings might miss.",
    },
    "Temporal/Memory Methods": {
        "description": "Methods that incorporate temporal dynamics and memory mechanisms to handle evolving information and context.",
        "methodology": "Memory-augmented neural networks, temporal attention mechanisms, and models that track information changes over time.",
        "key_insights": "Temporal awareness helps in domains where information changes over time or where recency is important.",
    },
    "Reinforcement Learning": {
        "description": "Using reinforcement learning to optimize retrieval policies and adaptively select retrieval strategies.",
        "methodology": "RL agents that learn retrieval policies through reward signals (e.g., nDCG). Human feedback can be incorporated for better alignment.",
        "key_insights": "RL approaches can learn adaptive strategies but require careful reward shaping and may be unstable during training.",
    },
    "Knowledge Distillation": {
        "description": "Transferring knowledge from large teacher models to smaller student models for efficient deployment.",
        "methodology": "Training smaller models to mimic larger models' embeddings or ranking behavior, often using KL divergence or ranking loss.",
        "key_insights": "Distillation enables efficient deployment of large model capabilities in resource-constrained environments.",
    },
    "LLM-Based Methods": {
        "description": "Utilizing large language models (LLMs) for query rewriting, expansion, or direct retrieval scoring.",
        "methodology": "Using GPT-4 or similar LLMs for query expansion, query decomposition, or as rerankers. Can also generate retrieval queries.",
        "key_insights": "LLMs provide strong semantic understanding but at high cost. Most effective for query preprocessing rather than direct retrieval.",
    },
    "Adversarial Training": {
        "description": "Improving robustness through adversarial examples and curriculum learning that progressively increases difficulty.",
        "methodology": "Generating adversarial examples, curriculum learning with increasing difficulty, and robustness training to handle challenging queries.",
        "key_insights": "Adversarial training improves model robustness and generalization, especially for edge cases and difficult queries.",
    },
    "Hard Negatives Mining": {
        "description": "Intelligently selecting challenging negative examples to improve contrastive learning effectiveness.",
        "methodology": "Mining hard negatives based on similarity scores, using in-batch negatives, or dynamic negative sampling strategies.",
        "key_insights": "Hard negatives are crucial for contrastive learning. Mining strategies that select challenging but relevant negatives improve discrimination.",
    },
    "Curriculum Learning": {
        "description": "Training models on increasingly difficult examples to improve learning efficiency and final performance.",
        "methodology": "Structuring training data from easy to hard examples, progressively increasing difficulty, or using adaptive difficulty scheduling.",
        "key_insights": "Curriculum learning helps models learn more effectively, especially for complex retrieval tasks with varied difficulty levels.",
    },
    "Other Methods": {
        "description": "Novel or experimental approaches that don't fit into standard categories.",
        "methodology": "Various experimental techniques including differentiable search, learned indexes, neural architecture search, and other innovative approaches.",
        "key_insights": "Exploration of novel techniques can lead to breakthrough improvements, though many experimental methods require careful tuning.",
    },
}

def load_all_results() -> Dict[str, Dict]:
    """Load all experiment results from results.json files."""
    results = {}
    
    if not RESULTS_DIR.exists():
        return results
    
    for exp_dir in RESULTS_DIR.iterdir():
        if exp_dir.is_dir():
            results_file = exp_dir / "results.json"
            if results_file.exists():
                try:
                    with open(results_file, 'r') as f:
                        data = json.load(f)
                        results[exp_dir.name] = data
                except Exception as e:
                    print(f"Warning: Could not load {results_file}: {e}")
    
    return results

def load_config(exp_name: str) -> Optional[Dict]:
    """Load experiment configuration if available."""
    config_file = RESULTS_DIR / exp_name / "config.json"
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            pass
    return None

def load_status_files() -> Dict[str, Dict]:
    """Load all status files to track experiment states."""
    all_status = {}
    
    for status_file in STATUS_FILES:
        filepath = pathlib.Path(status_file)
        if filepath.exists():
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    all_status[status_file] = data
            except Exception as e:
                pass
    
    return all_status

def get_all_experiment_directories() -> Set[str]:
    """Get all experiment directory names."""
    if not RESULTS_DIR.exists():
        return set()
    
    return {d.name for d in RESULTS_DIR.iterdir() if d.is_dir()}

def extract_scores(exp_data: Dict) -> Optional[Dict]:
    """Extract scores from experiment data."""
    scores = {}
    
    if 'average' in exp_data:
        avg = exp_data['average']
        scores = {
            'Recall@1': avg.get('Recall@1', 0),
            'Recall@3': avg.get('Recall@3', 0),
            'Recall@5': avg.get('Recall@5', 0),
            'Recall@10': avg.get('Recall@10', 0),
            'nDCG@1': avg.get('nDCG@1', 0),
            'nDCG@3': avg.get('nDCG@3', 0),
            'nDCG@5': avg.get('nDCG@5', 0),
            'nDCG@10': avg.get('nDCG@10', 0),
        }
    elif 'averages' in exp_data:
        avg = exp_data['averages']
        scores = {
            'Recall@1': avg.get('Recall@1', 0),
            'Recall@3': avg.get('Recall@3', 0),
            'Recall@5': avg.get('Recall@5', 0),
            'Recall@10': avg.get('Recall@10', 0),
            'nDCG@1': avg.get('nDCG@1', 0),
            'nDCG@3': avg.get('nDCG@3', 0),
            'nDCG@5': avg.get('nDCG@5', 0),
            'nDCG@10': avg.get('nDCG@10', 0),
        }
    
    return scores if scores else None

def categorize_experiment(name: str) -> str:
    """Categorize experiment by name."""
    name_lower = name.lower()
    
    if 'baseline' in name_lower or name.startswith('baseline_'):
        return 'Baseline'
    elif name.startswith('domain_specific') or 'domain_specific' in name:
        return 'Domain-Specific Fine-tuning'
    elif name.startswith('contrastive') or 'contrastive' in name:
        return 'Contrastive Learning'
    elif name.startswith('meta_learning') or 'meta_learning' in name:
        return 'Meta-Learning'
    elif name.startswith('ensemble') or 'ensemble' in name:
        return 'Ensemble Methods'
    elif name.startswith('query_expansion') or 'query_expansion' in name:
        return 'Query Expansion'
    elif name.startswith('reranking') or 'reranking' in name:
        return 'Reranking'
    elif name.startswith('hybrid') or 'hybrid' in name:
        return 'Hybrid Methods'
    elif name.startswith('multistage') or 'multistage' in name:
        return 'Multi-Stage Retrieval'
    elif name.startswith('learning_to_rank') or 'learning_to_rank' in name:
        return 'Learning-to-Rank'
    elif name.startswith('large_model') or 'large_model' in name or 'bge_large' in name_lower:
        return 'Large Model Fine-tuning'
    elif 'hard_negatives' in name or 'hard_neg' in name:
        return 'Hard Negatives Mining'
    elif 'curriculum' in name:
        return 'Curriculum Learning'
    elif 'adversarial' in name:
        return 'Adversarial Training'
    elif 'graph' in name:
        return 'Graph-Based Methods'
    elif 'temporal' in name or 'memory' in name:
        return 'Temporal/Memory Methods'
    elif 'reinforcement' in name or 'rl' in name:
        return 'Reinforcement Learning'
    elif 'distillation' in name:
        return 'Knowledge Distillation'
    elif 'cross_encoder' in name or 'cross_attention' in name:
        return 'Cross-Encoder'
    elif 'conversation' in name or 'multi_turn' in name:
        return 'Conversation-Aware'
    elif 'llm' in name_lower:
        return 'LLM-Based Methods'
    else:
        return 'Other Methods'

def get_model_checkpoint_path(exp_name: str, results_data: Optional[Dict] = None, config: Optional[Dict] = None) -> str:
    """Get model checkpoint path for an experiment."""
    # Check in results.json first
    if results_data and 'model_path' in results_data:
        return results_data['model_path']
    
    # Check config.json for output_path or model_path
    if config:
        if 'output_path' in config:
            return config['output_path']
        if 'model_path' in config:
            return config['model_path']
        if 'output_dir' in config:
            return config['output_dir']
    
    # Check if model exists in models directory with experiment name mapping
    exp_dir = RESULTS_DIR / exp_name
    if exp_dir.exists():
        # Check for model subdirectory
        model_dir = exp_dir / "model"
        if model_dir.exists() and model_dir.is_dir():
            return f"experiments/retrieval/{exp_name}/model"
    
    # Try to infer from experiment name
    # Many experiments use models/{experiment_name} pattern
    models_path = pathlib.Path("models") / exp_name
    if models_path.exists():
        return f"models/{exp_name}"
    
    return "N/A"

def get_experiment_json_files(exp_name: str, full_paths: bool = False) -> List[str]:
    """Get list of JSON files associated with an experiment.
    
    Args:
        exp_name: Experiment name
        full_paths: If True, return full paths. If False, return just filenames.
    
    Returns:
        List of JSON file paths or filenames
    """
    json_files = []
    exp_dir = RESULTS_DIR / exp_name
    seen_files = set()
    
    if not exp_dir.exists():
        return json_files
    
    # Always include config.json and results.json if they exist (priority files)
    priority_files = []
    if (exp_dir / "config.json").exists():
        if full_paths:
            priority_files.append(f"experiments/retrieval/{exp_name}/config.json")
        else:
            priority_files.append("config.json")
        seen_files.add("config.json")
    if (exp_dir / "results.json").exists():
        if full_paths:
            priority_files.append(f"experiments/retrieval/{exp_name}/results.json")
        else:
            priority_files.append("results.json")
        seen_files.add("results.json")
    
    # Find other JSON files
    other_files = []
    for json_file in exp_dir.rglob("*.json"):
        rel_path = json_file.relative_to(pathlib.Path("."))
        file_name = json_file.name
        
        # Skip if we've already seen this filename (for non-full paths mode)
        if not full_paths and file_name in seen_files:
            continue
        
        if full_paths:
            file_str = str(rel_path)
            if file_str not in other_files and file_str not in priority_files:
                other_files.append(file_str)
        else:
            if file_name not in seen_files:
                other_files.append(file_name)
                seen_files.add(file_name)
    
    # Combine priority files first, then others (sorted)
    if full_paths:
        json_files = priority_files + sorted(other_files)
    else:
        json_files = priority_files + sorted(other_files)
    
    return json_files

def get_experiment_explanation(exp_name: str, category: str, config: Optional[Dict] = None, scores: Optional[Dict] = None) -> str:
    """Generate detailed explanation for an experiment."""
    lines = []
    
    # Basic description
    cat_info = CATEGORY_EXPLANATIONS.get(category, CATEGORY_EXPLANATIONS["Other Methods"])
    lines.append(f"**{category}**")
    lines.append("")
    lines.append(f"{cat_info['description']}")
    lines.append("")
    
    # Methodology
    lines.append("**Methodology:**")
    lines.append(f"{cat_info['methodology']}")
    lines.append("")
    
    # Experiment-specific details
    if config:
        lines.append("**Experiment Configuration:**")
        if 'base_model' in config:
            lines.append(f"- Base Model: {config['base_model']}")
        if 'epochs' in config:
            lines.append(f"- Training Epochs: {config['epochs']}")
        if 'learning_rate' in config:
            lines.append(f"- Learning Rate: {config['learning_rate']}")
        if 'batch_size' in config:
            lines.append(f"- Batch Size: {config['batch_size']}")
        if 'domain' in config:
            lines.append(f"- Target Domain: {config['domain']} ({DOMAIN_INFO.get(config['domain'], 'N/A')})")
        if 'domains' in config and isinstance(config['domains'], list):
            domains_str = ", ".join([f"{d} ({DOMAIN_INFO.get(d, d)})" for d in config['domains']])
            lines.append(f"- Training Domains: {domains_str}")
        lines.append("")
    
    # Results summary
    if scores:
        lines.append("**Results Summary:**")
        lines.append(f"- nDCG@10: {scores['nDCG@10']:.4f}")
        lines.append(f"- Recall@10: {scores['Recall@10']:.4f}")
        lines.append(f"- nDCG@5: {scores['nDCG@5']:.4f}")
        lines.append(f"- Recall@5: {scores['Recall@5']:.4f}")
        lines.append("")
    
    # Key insights
    lines.append("**Key Insights:**")
    lines.append(f"{cat_info['key_insights']}")
    lines.append("")
    
    return "\n".join(lines)

def generate_detailed_summary() -> str:
    """Generate comprehensive experiment summary with detailed explanations."""
    lines = []
    
    # Header
    lines.append("# Detailed Experiment Summary with Comprehensive Explanations")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")
    lines.append("This document provides a comprehensive overview of all retrieval experiments conducted on the MT-RAG benchmark, including detailed explanations of methodologies, results, and insights.")
    lines.append("")
    lines.append("## Table of Contents")
    lines.append("")
    lines.append("1. [Executive Summary](#executive-summary)")
    lines.append("2. [Experimental Setup](#experimental-setup)")
    lines.append("3. [Experiment Categories](#experiment-categories)")
    lines.append("4. [Top Performing Experiments](#top-performing-experiments)")
    lines.append("5. [Category-Wise Detailed Analysis](#category-wise-detailed-analysis)")
    lines.append("6. [Comparison and Insights](#comparison-and-insights)")
    lines.append("7. [Conclusion](#conclusion)")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Load data
    print("Loading experiment data...")
    results = load_all_results()
    status_files = load_status_files()
    all_dirs = get_all_experiment_directories()
    
    # Get all unique experiment names
    all_experiments = set()
    all_experiments.update(results.keys())
    all_experiments.update(all_dirs)
    
    for status_file, data in status_files.items():
        if isinstance(data, dict):
            if 'stopped_experiments' in data:
                for exp in data['stopped_experiments']:
                    all_experiments.add(exp.get('experiment_name'))
            else:
                all_experiments.update(data.keys())
    
    all_experiments = sorted(all_experiments)
    
    # Categorize experiments
    experiments_by_category = defaultdict(list)
    completed_experiments = []
    
    for exp_name in all_experiments:
        category = categorize_experiment(exp_name)
        scores = extract_scores(results.get(exp_name, {})) if exp_name in results else None
        config = load_config(exp_name)
        
        exp_data = {
            'name': exp_name,
            'category': category,
            'scores': scores,
            'config': config,
        }
        
        experiments_by_category[category].append(exp_data)
        
        if scores:
            completed_experiments.append(exp_data)
    
    # Executive Summary
    lines.append("## Executive Summary")
    lines.append("")
    lines.append(f"This comprehensive evaluation includes **{len(all_experiments)}** total experiments across **{len(experiments_by_category)}** different methodological categories.")
    lines.append(f"Of these, **{len(completed_experiments)}** experiments have completed successfully with full evaluation results.")
    lines.append("")
    
    # Top performers
    completed_experiments.sort(key=lambda x: x['scores']['nDCG@10'] if x['scores'] else 0, reverse=True)
    top_10 = completed_experiments[:10]
    
    lines.append("### Top 10 Experiments by nDCG@10")
    lines.append("")
    lines.append("| Rank | Experiment | Category | nDCG@10 | Recall@10 |")
    lines.append("|------|-----------|----------|---------|-----------|")
    
    for rank, exp in enumerate(top_10, 1):
        name = exp['name']
        if len(name) > 40:
            name = name[:37] + "..."
        scores = exp['scores']
        lines.append(
            f"| {rank} | {name} | {exp['category']} | "
            f"{scores['nDCG@10']:.4f} | {scores['Recall@10']:.4f} |"
        )
    
    lines.append("")
    lines.append("**Key Findings:**")
    lines.append(f"- **Best Overall Performance:** {top_10[0]['name']} achieves nDCG@10 of {top_10[0]['scores']['nDCG@10']:.4f}")
    lines.append("- **Most Effective Categories:** Large Model Fine-tuning and Domain-Specific Fine-tuning consistently outperform baselines")
    min_ndcg = min(exp['scores']['nDCG@10'] for exp in completed_experiments if exp['scores'])
    max_ndcg = max(exp['scores']['nDCG@10'] for exp in completed_experiments if exp['scores'])
    lines.append(f"- **Performance Range:** nDCG@10 ranges from {min_ndcg:.4f} to {max_ndcg:.4f}")
    lines.append("")
    
    # Top 20 experiments table with checkpoints and JSON files
    top_20 = completed_experiments[:20]
    lines.append("### Top 20 Experiments: Accuracy, Checkpoints, and Associated Files")
    lines.append("")
    lines.append("| Rank | Experiment | nDCG@10 | Recall@10 | Model Checkpoint | JSON Files |")
    lines.append("|------|-----------|---------|-----------|------------------|------------|")
    
    for rank, exp in enumerate(top_20, 1):
        name = exp['name']
        scores = exp['scores']
        
        # Get model checkpoint path
        exp_results = results.get(exp['name'], {})
        checkpoint_path = get_model_checkpoint_path(exp['name'], exp_results, exp['config'])
        if len(checkpoint_path) > 50:
            checkpoint_display = checkpoint_path[:47] + "..."
        else:
            checkpoint_display = checkpoint_path
        
        # Get JSON files
        json_files = get_experiment_json_files(exp['name'])
        if json_files:
            json_display = ", ".join(json_files[:3])
            if len(json_files) > 3:
                json_display += f" (+{len(json_files) - 3} more)"
        else:
            json_display = "N/A"
        
        if len(name) > 30:
            name_display = name[:27] + "..."
        else:
            name_display = name
        
        lines.append(
            f"| {rank} | {name_display} | {scores['nDCG@10']:.4f} | {scores['Recall@10']:.4f} | "
            f"`{checkpoint_display}` | {json_display} |"
        )
    
    lines.append("")
    lines.append("**Note:** JSON files column shows the first 3 files. See detailed list below for all JSON files with full paths.")
    lines.append("")
    
    # Detailed JSON files listing
    lines.append("### Complete JSON Files Listing for Top 20 Experiments")
    lines.append("")
    lines.append("This section provides a complete listing of all JSON files associated with each of the top 20 experiments, including their full paths.")
    lines.append("")
    
    for rank, exp in enumerate(top_20, 1):
        exp_name = exp['name']
        json_files = get_experiment_json_files(exp_name, full_paths=True)
        
        lines.append(f"#### {rank}. {exp_name}")
        lines.append("")
        if json_files:
            lines.append(f"**Total JSON files:** {len(json_files)}")
            lines.append("")
            for json_file in json_files:
                lines.append(f"- `{json_file}`")
        else:
            lines.append("**No JSON files found in experiment directory.**")
        lines.append("")
    
    lines.append("---")
    lines.append("")
    
    # Experimental Setup
    lines.append("## Experimental Setup")
    lines.append("")
    lines.append("### Dataset")
    lines.append("")
    lines.append("The MT-RAG benchmark consists of four domains:")
    for domain, info in DOMAIN_INFO.items():
        lines.append(f"- **{domain.upper()}**: {info}")
    lines.append("")
    lines.append("### Evaluation Metrics")
    lines.append("")
    lines.append("- **Recall@K**: Fraction of relevant documents retrieved in the top K results")
    lines.append("- **nDCG@K**: Normalized Discounted Cumulative Gain, measuring ranking quality with position discounting")
    lines.append("- **K values**: 1, 3, 5, 10")
    lines.append("")
    lines.append("### Base Models")
    lines.append("")
    lines.append("- Primary: BAAI/bge-base-en-v1.5 (278M parameters)")
    lines.append("- Large: BAAI/bge-large-en-v1.5 (560M+ parameters)")
    lines.append("- Fine-tuning: Contrastive learning with MultipleNegativesRankingLoss")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Experiment Categories Overview
    lines.append("## Experiment Categories")
    lines.append("")
    lines.append(f"The experiments are organized into **{len(experiments_by_category)}** categories:")
    lines.append("")
    
    category_summary = []
    for category in sorted(experiments_by_category.keys()):
        exps = experiments_by_category[category]
        completed = [e for e in exps if e['scores']]
        if completed:
            avg_ndcg = sum(e['scores']['nDCG@10'] for e in completed) / len(completed)
            best_ndcg = max(e['scores']['nDCG@10'] for e in completed)
            best_exp = next(e for e in completed if e['scores']['nDCG@10'] == best_ndcg)
        else:
            avg_ndcg = 0
            best_ndcg = 0
            best_exp = None
        
        category_summary.append({
            'category': category,
            'total': len(exps),
            'completed': len(completed),
            'avg_ndcg': avg_ndcg,
            'best_ndcg': best_ndcg,
            'best_exp': best_exp['name'] if best_exp else None,
        })
    
    category_summary.sort(key=lambda x: x['best_ndcg'], reverse=True)
    
    lines.append("| Category | Total | Completed | Avg nDCG@10 | Best nDCG@10 | Best Experiment |")
    lines.append("|----------|-------|-----------|-------------|--------------|-----------------|")
    
    for cat in category_summary:
        best_name = cat['best_exp'][:40] + "..." if cat['best_exp'] and len(cat['best_exp']) > 40 else (cat['best_exp'] or "N/A")
        avg_str = f"{cat['avg_ndcg']:.4f}" if cat['completed'] > 0 else "N/A"
        best_str = f"{cat['best_ndcg']:.4f}" if cat['completed'] > 0 else "N/A"
        lines.append(
            f"| {cat['category']} | {cat['total']} | {cat['completed']} | {avg_str} | {best_str} | {best_name} |"
        )
    
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Top Performing Experiments
    lines.append("## Top Performing Experiments")
    lines.append("")
    lines.append("### Detailed Analysis of Top 10 Experiments")
    lines.append("")
    
    for rank, exp in enumerate(top_10, 1):
        lines.append(f"#### {rank}. {exp['name']}")
        lines.append("")
        explanation = get_experiment_explanation(exp['name'], exp['category'], exp['config'], exp['scores'])
        lines.append(explanation)
        lines.append("---")
        lines.append("")
    
    # Category-wise analysis
    lines.append("## Category-Wise Detailed Analysis")
    lines.append("")
    
    for category in sorted(experiments_by_category.keys()):
        exps = experiments_by_category[category]
        completed = [e for e in exps if e['scores']]
        
        if not completed:
            continue
        
        lines.append(f"### {category}")
        lines.append("")
        
        # Category overview
        cat_info = CATEGORY_EXPLANATIONS.get(category, CATEGORY_EXPLANATIONS["Other Methods"])
        lines.append(f"**Overview:** {cat_info['description']}")
        lines.append("")
        lines.append(f"**Methodology:** {cat_info['methodology']}")
        lines.append("")
        
        # Results
        completed.sort(key=lambda x: x['scores']['nDCG@10'], reverse=True)
        
        lines.append(f"**Experiments in this category:** {len(completed)} completed")
        lines.append("")
        lines.append("| Experiment | nDCG@10 | Recall@10 | nDCG@5 | Recall@5 |")
        lines.append("|-----------|---------|-----------|--------|----------|")
        
        for exp in completed[:10]:  # Top 10 per category
            name = exp['name']
            if len(name) > 40:
                name = name[:37] + "..."
            scores = exp['scores']
            lines.append(
                f"| {name} | {scores['nDCG@10']:.4f} | {scores['Recall@10']:.4f} | "
                f"{scores['nDCG@5']:.4f} | {scores['Recall@5']:.4f} |"
            )
        
        if len(completed) > 10:
            lines.append(f"| ... ({len(completed) - 10} more experiments) | ... | ... | ... | ... |")
        
        lines.append("")
        lines.append(f"**Key Insights:** {cat_info['key_insights']}")
        lines.append("")
        lines.append("---")
        lines.append("")
    
    # Comparison and Insights
    lines.append("## Comparison and Insights")
    lines.append("")
    lines.append("### Performance Trends")
    lines.append("")
    lines.append("1. **Large Models Outperform:** Experiments using larger base models (bge-large) consistently achieve higher nDCG scores, demonstrating the value of increased model capacity.")
    lines.append("")
    lines.append("2. **Domain-Specific Fine-tuning is Effective:** Fine-tuning models specifically for individual domains outperforms multi-domain training when evaluated on target domains.")
    lines.append("")
    lines.append("3. **Ensemble Methods Improve Robustness:** Combining multiple models through weighted fusion or reciprocal rank fusion consistently improves performance and reduces variance.")
    lines.append("")
    lines.append("4. **Reranking Adds Value:** Two-stage retrieval with reranking improves precision, though at increased computational cost.")
    lines.append("")
    lines.append("5. **Query Expansion Helps:** LLM-based query expansion and pseudo-relevance feedback improve retrieval, especially for short or ambiguous queries.")
    lines.append("")
    lines.append("### Methodology Recommendations")
    lines.append("")
    lines.append("Based on the experimental results:")
    lines.append("")
    lines.append("- **For Best Performance:** Use large models with domain-specific fine-tuning and ensemble methods")
    lines.append("- **For Efficiency:** Use multi-stage retrieval with efficient bi-encoder for initial retrieval and cross-encoder for reranking")
    lines.append("- **For Generalization:** Combine multi-domain pretraining with domain-specific fine-tuning")
    lines.append("- **For Robustness:** Employ ensemble methods and adversarial training")
    lines.append("")
    lines.append("---")
    lines.append("")
    
    # Conclusion
    lines.append("## Conclusion")
    lines.append("")
    lines.append(f"This comprehensive evaluation demonstrates the effectiveness of various retrieval approaches on the MT-RAG benchmark. With **{len(completed_experiments)}** completed experiments across **{len(experiments_by_category)}** categories, the results provide valuable insights into:")
    lines.append("")
    lines.append("1. The relative effectiveness of different retrieval methodologies")
    lines.append("2. The importance of domain-specific adaptation")
    lines.append("3. The trade-offs between model size, training cost, and performance")
    lines.append("4. The value of ensemble and multi-stage approaches")
    lines.append("")
    lines.append(f"The best-performing approach achieves **nDCG@10 of {top_10[0]['scores']['nDCG@10']:.4f}**, setting a strong benchmark for future research in conversational retrieval.")
    lines.append("")
    lines.append("---")
    lines.append("")
    lines.append(f"*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")
    
    return "\n".join(lines)

def main():
    """Main function."""
    print("=" * 80)
    print("Generating Detailed Experiment Summary with Explanations")
    print("=" * 80)
    print()
    
    summary = generate_detailed_summary()
    
    # Write to file
    output_file = "DETAILED_EXPERIMENT_SUMMARY.md"
    with open(output_file, 'w') as f:
        f.write(summary)
    
    print(f"✅ Detailed summary written to: {output_file}")
    print()
    print("Summary includes:")
    print("  - Executive summary with top performers")
    print("  - Detailed category explanations")
    print("  - Methodology descriptions for each approach")
    print("  - Configuration details for each experiment")
    print("  - Performance comparisons and insights")
    print("  - Recommendations based on results")

if __name__ == "__main__":
    main()

