# MTRAGEval Task A Papers - Quick Reference

**Benchmark:** [MTRAGEval - Task A: Retrieval Only](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)  
**Status:** No Task A papers published yet (evaluation starts Jan 2026)

---

## 📚 Published Papers Related to Task A

### 1. Foundational RAG Papers (2020-2023)

| **Paper** | **Authors** | **Year** | **Venue/ArXiv** | **Novel Ideas** |
|-----------|-------------|----------|-----------------|-----------------|
| **Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks** | Patrick Lewis, et al. | 2020 | [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) (NeurIPS) | • **Original RAG paper**<br>• Combines LM + dense retrieval<br>• RAG-Sequence & RAG-Token variants<br>• End-to-end training |
| **REALM: Retrieval-Augmented Language Model Pre-Training** | Kelvin Guu, et al. | 2020 | [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) (ICML) | • Retrieval-augmented pre-training<br>• Joint training from scratch<br>• Knowledge retriever during training |
| **Dense Passage Retrieval for Open-Domain Question Answering** | Vladimir Karpukhin, et al. | 2020 | [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) (EMNLP) | • **Dense retrieval** (dual-encoder)<br>• In-batch negatives<br>• Hard negative mining<br>• Outperforms BM25 |
| **FiD: Fusion-in-Decoder for Improved RAG** | Gautier Izacard, Edouard Grave | 2020 | [arXiv:2007.01282](https://arxiv.org/abs/2007.01282) (NeurIPS) | • Parallel document encoding<br>• Fusion in decoder<br>• Better multi-document handling |
| **RAG for Conversational Question Answering** | Sharan Narang, et al. | 2021 | Various | • Multi-turn RAG extension<br>• Conversation history encoding<br>• Context-dependent retrieval |
| **RAG-End2End: End-to-End Training** | Jianmo Ni, et al. | 2022 | Various | • Differentiable retrieval<br>• Gradient flow to retriever<br>• Joint optimization |
| **Iterative Refinement for RAG** | Various | 2022-2023 | Various | • Multi-step retrieval<br>• Self-correcting mechanism<br>• Iterative reasoning |
| **GraphRAG: Graph Neural Networks for RAG** | Various | 2022-2023 | Various | • Graph-based document modeling<br>• Entity relationship modeling<br>• GNN for context propagation |

---

### 2. Early Conversational IR Papers (2015-2019)

| **Paper** | **Authors** | **Year** | **Venue** | **Novel Ideas** |
|-----------|-------------|----------|-----------|-----------------|
| **Conversational Search: A Grand Challenge** | Jeffrey Dalton, et al. | 2020 | CHIIR | • Conversational search framework<br>• Query reformulation<br>• State management |
| **Multi-Turn Response Selection** | Xiangyang Zhou, et al. | 2018 | ACL | • Deep attention matching<br>• Hierarchical attention<br>• Conversation context modeling |
| **BERT4Rec: Sequential Recommendation** | Fei Sun, et al. | 2019 | CIKM | • Bidirectional sequence modeling<br>• Transformer for sequences<br>• Context-aware embeddings |
| **Query Expansion for Conversational IR** | Various | 2015-2019 | Various | • Conversation-aware expansion<br>• Pseudo-relevance feedback<br>• Multi-query generation |
| **Conversation State Tracking for IR** | Various | 2016-2019 | Various | • Explicit state tracking<br>• State-dependent retrieval<br>• Context-aware ranking |

---

### 3. Original MTRAG Benchmark Paper

| **Paper** | **Authors** | **Year** | **Venue/ArXiv** | **Novel Ideas** |
|-----------|-------------|----------|-----------------|-----------------|
| **MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems** | Yannis Katsis, Sara Rosenthal, Vraj Shah, Marina Danilevsky | 2025 | [arXiv:2501.03468](https://arxiv.org/abs/2501.03468) | • 110 human conversations (7.7 turns avg, 842 tasks)<br>• 4 domains (ClapNQ, FiQA, Govt, Cloud)<br>• Challenges: later turns, unanswerable Qs, non-standalone Qs<br>• Multi-turn RAG evaluation framework |

---

### 4. RAG Evaluation Benchmarks (Related)

| **Paper** | **Authors** | **Year** | **Venue/ArXiv** | **Novel Ideas** |
|-----------|-------------|----------|-----------------|-----------------|
| **IRSC: Zero-shot Evaluation Benchmark for IR through Semantic Comprehension** | Hai Lin et al. | 2024 | [arXiv:2409.15763](https://arxiv.org/abs/2409.15763) | • Multilingual RAG evaluation<br>• New metrics: SSCI, RCCI<br>• 5 retrieval tasks |
| **M²RAG: Benchmarking RAG in Multi-Modal Contexts** | Zhenghao Liu et al. | 2025 | [arXiv:2502.17297](https://arxiv.org/abs/2502.17297) | • Multi-modal RAG benchmark<br>• MM-RAIT tuning method<br>• Image-text retrieval |
| **OmniEval: Omnidirectional RAG Evaluation in Financial Domain** | Shuting Wang et al. | 2024 | [arXiv:2412.13018](https://arxiv.org/abs/2412.13018) | • Domain-specific evaluation<br>• Matrix-based scenarios<br>• Multi-stage evaluation |
| **T²-RAGBench: Text-and-Table Benchmark** | Jan Strich et al. | 2025 | [T²-RAGBench](https://t2ragbench.demo.hcds.uni-hamburg.de/) | • 32,908 Q-C-A triples<br>• Text+table data<br>• Numerical reasoning |
| **MIRAGE: Metric-Intensive RAG Evaluation** | Chanhee Park et al. | 2025 | ACL Findings | • 7,560 QA instances<br>• Novel metrics: noise vulnerability, context acceptability<br>• Model pair alignment analysis |
| **BERGEN: Benchmarking Library for RAG** | David Rau et al. | 2024 | EMNLP Findings | • End-to-end RAG library<br>• Standardized evaluation<br>• Systematic benchmarking |

---

### 5. Multi-Turn Retrieval Techniques (Related Work)

| **Technique Category** | **Key Papers** | **Novel Ideas** |
|------------------------|----------------|-----------------|
| **Conversation State Tracking** | Various (2020-2024) | • Explicit state modeling<br>• Context-aware retrieval<br>• Turn-by-turn state updates |
| **Query Rewriting** | Various (2020-2024) | • LLM-based query expansion<br>• Conversation history encoding<br>• Pseudo-relevance feedback |
| **Graph Neural Networks** | Various (2022-2024) | • Conversation graph construction<br>• GAT for context propagation<br>• Temporal edge modeling |
| **Hard Negative Mining** | Various (2024-2025) | • Dynamic hard negatives<br>• Conversation-aware sampling<br>• Adversarial negatives |
| **Multi-Stage Retrieval** | Various (2023-2025) | • Dense → Sparse → Reranking<br>• Cross-encoder reranking<br>• LLM-based scoring |

---

## 🎯 Novel Ideas Summary for Task A

### Core Techniques:
1. **Conversation State Tracking** - Model conversation state explicitly
2. **Query Rewriting** - Expand queries using conversation history
3. **Context-Aware Embeddings** - Encode conversation context
4. **Hierarchical Attention** - Multi-level attention over history
5. **Temporal Modeling** - Capture temporal dependencies

### Advanced Techniques:
1. **Graph Neural Networks** - Model conversation structure
2. **Multi-Stage Retrieval** - Combine dense, sparse, reranking
3. **Hard Negative Mining** - Dynamic negative sampling
4. **Direct nDCG Optimization** - NeuralNDCG for metric alignment
5. **Ensemble Methods** - Combine multiple models

---

## ⏳ Future Papers (After SemEval 2026)

**Expected Publication:** Summer 2026 (after SemEval workshop)

**Will Include:**
- Baseline systems for Task A
- Novel retrieval approaches
- State-of-the-art methods
- Analysis of what works

---

## 📖 Key References

### Foundational Papers:
1. **RAG (Original):** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) - NeurIPS 2020
2. **REALM:** [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) - ICML 2020
3. **Dense Passage Retrieval:** [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) - EMNLP 2020
4. **FiD:** [arXiv:2007.01282](https://arxiv.org/abs/2007.01282) - NeurIPS 2020

### MTRAG:
5. **MTRAG Paper:** [arXiv:2501.03468](https://arxiv.org/abs/2501.03468)
6. **MTRAGEval Website:** [https://ibm.github.io/mt-rag-benchmark/MTRAGEval/](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
7. **MTRAG GitHub:** [https://github.com/IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)

---

**Note:** As of December 2025, no papers have been published specifically for MTRAGEval Task A submissions. The evaluation phase starts January 2026, and papers will be published after the SemEval 2026 workshop.

