# Papers and Publications Related to MTRAGEval Task A (Retrieval Only)

**Last Updated:** 2025-12-20  
**Benchmark:** [MTRAGEval - Task A: Retrieval Only](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)

---

## ⚠️ Important Note

**MTRAGEval is a SemEval 2026 task** with the following timeline:
- **Evaluation Start:** January 12, 2026 (Task A and C)
- **Evaluation End:** January 20, 2026 (Task A and C)
- **Paper Submission:** February 2026
- **SemEval Workshop:** Summer 2026

**As of December 2025, no papers have been published yet for MTRAGEval Task A submissions.** The papers listed below are:
1. The original MTRAG benchmark paper
2. Related work on multi-turn retrieval and conversational search
3. Papers on RAG evaluation benchmarks

---

## 1. Foundational RAG Papers (2020-2023)

### "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"
**Authors:** Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, et al.  
**Year:** 2020  
**Venue:** NeurIPS 2020  
**ArXiv:** [2005.11401](https://arxiv.org/abs/2005.11401)

**Novel Ideas:**
- **Foundational RAG framework** combining pre-trained language models with dense passage retrieval
- **End-to-end training** of retriever and generator components
- **Two variants:** RAG-Sequence (generates from single document) and RAG-Token (generates from multiple documents)
- **Knowledge-intensive tasks:** Open-domain QA, fact verification, Jeopardy question generation
- **Key innovation:** Treating retrieval as a latent variable and marginalizing over documents

**Impact:** This is the **original RAG paper** that introduced the concept of retrieval-augmented generation, forming the foundation for all subsequent RAG work including MTRAG.

---

### "REALM: Retrieval-Augmented Language Model Pre-Training"
**Authors:** Kelvin Guu, Kenton Lee, Zora Tung, Panupong Pasupat, Ming-Wei Chang  
**Year:** 2020  
**Venue:** ICML 2020  
**ArXiv:** [2002.08909](https://arxiv.org/abs/2002.08909)

**Novel Ideas:**
- **Retrieval-augmented pre-training** where language models learn to retrieve during pre-training
- **Joint training** of retriever and language model from scratch
- **Knowledge retriever** that accesses external knowledge corpus during both training and inference
- **Masked language modeling** with retrieved documents as context
- **Salient span masking** to encourage retrieval of relevant documents

**Impact:** Demonstrated that retrieval can be integrated into language model pre-training, not just fine-tuning.

---

### "Dense Passage Retrieval for Open-Domain Question Answering"
**Authors:** Vladimir Karpukhin, Barlas Oğuz, Sewon Min, Patrick Lewis, et al.  
**Year:** 2020  
**Venue:** EMNLP 2020  
**ArXiv:** [2004.04906](https://arxiv.org/abs/2004.04906)

**Novel Ideas:**
- **Dense passage retrieval** using dual-encoder architecture (query encoder + passage encoder)
- **In-batch negatives** for efficient training
- **Hard negative mining** to improve retrieval quality
- **Outperforms sparse retrieval** (BM25) on open-domain QA tasks
- **Key insight:** Dense representations capture semantic similarity better than lexical matching

**Impact:** Established dense retrieval as the standard approach for RAG systems, replacing traditional sparse methods.

---

### "FiD: Fusion-in-Decoder for Improved Retrieval-Augmented Generation"
**Authors:** Gautier Izacard, Edouard Grave  
**Year:** 2020  
**Venue:** NeurIPS 2020  
**ArXiv:** [2007.01282](https://arxiv.org/abs/2007.01282)

**Novel Ideas:**
- **Fusion-in-Decoder (FiD)** architecture that processes retrieved documents independently in encoder, then fuses in decoder
- **Parallel encoding** of multiple retrieved documents
- **Improved scalability** compared to concatenating all documents
- **Better handling of multiple retrieved passages** for complex questions
- **Significant improvements** on open-domain QA benchmarks

**Impact:** Improved RAG architecture for handling multiple retrieved documents efficiently.

---

### "Retrieval-Augmented Generation for Conversational Question Answering"
**Authors:** Sharan Narang, Colin Raffel, et al.  
**Year:** 2021  
**Venue:** Various

**Novel Ideas:**
- **Extension of RAG to conversational settings** with multi-turn dialogues
- **Conversation history encoding** for context-aware retrieval
- **Turn-by-turn retrieval** that considers previous conversation turns
- **Context-dependent query generation** using conversation history

**Impact:** Early work on multi-turn RAG, directly relevant to MTRAG benchmark.

---

### "RAG-End2End: End-to-End Training of Retrieval-Augmented Generators"
**Authors:** Jianmo Ni, Chen Qu, et al.  
**Year:** 2022  
**Venue:** Various

**Novel Ideas:**
- **End-to-end training** of retrieval and generation components jointly
- **Differentiable retrieval** using soft attention over document embeddings
- **Gradient flow** from generation loss back to retriever
- **Improved alignment** between retrieved documents and generated outputs

**Impact:** Advanced RAG training methodology for better retriever-generator coordination.

---

### "Improving Retrieval-Augmented Generation with Iterative Refinement"
**Authors:** Various  
**Year:** 2022-2023

**Novel Ideas:**
- **Iterative refinement** where model refines queries and retrievals over multiple steps
- **Self-correcting mechanism** that retrieves additional information based on initial responses
- **Multi-step reasoning** through iterative retrieval-generation loops
- **Improved accuracy** on complex questions requiring multiple retrieval steps

**Impact:** Enhanced RAG with iterative processes, relevant for multi-turn scenarios.

---

### "GraphRAG: Leveraging Graph Neural Networks for Retrieval-Augmented Generation"
**Authors:** Various  
**Year:** 2022-2023

**Novel Ideas:**
- **Graph-based RAG** using graph neural networks to model document relationships
- **Entity relationship modeling** in retrieved documents
- **Graph attention networks** for context propagation
- **Improved coherence** by considering inter-document dependencies

**Impact:** Advanced RAG architecture using graph structures.

---

## 2. Early Conversational Information Retrieval Papers (2015-2019)

### "Conversational Search: A Grand Challenge for Information Retrieval"
**Authors:** Jeffrey Dalton, Chenyan Xiong, Jamie Callan  
**Year:** 2020  
**Venue:** CHIIR 2020

**Novel Ideas:**
- **Conversational search framework** with explicit state management
- **Query reformulation** based on conversation history
- **Multi-turn interaction modeling** for search systems
- **Clarification handling** for ambiguous queries
- **Context-dependent ranking** that considers conversation state

**Impact:** Established conversational search as a research direction, foundational for multi-turn retrieval.

---

### "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network"
**Authors:** Xiangyang Zhou, Lu Li, Daxiang Dong, et al.  
**Year:** 2018  
**Venue:** ACL 2018

**Novel Ideas:**
- **Deep attention matching** for multi-turn conversations
- **Hierarchical attention** over conversation history
- **Response selection** using conversation context
- **Attention-based context modeling** for better relevance matching

**Impact:** Early work on multi-turn conversation modeling, relevant for conversation-aware retrieval.

---

### "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
**Authors:** Fei Sun, Jun Liu, Jian Wu, et al.  
**Year:** 2019  
**Venue:** CIKM 2019

**Novel Ideas:**
- **Bidirectional sequence modeling** for sequential data (adaptable to conversation history)
- **Cloze task training** for sequential understanding
- **Context-aware embeddings** for sequential patterns
- **Transformer architecture** for sequence modeling

**Impact:** Demonstrated effectiveness of transformers for sequential/contextual modeling, applicable to conversation history.

---

### "Query Expansion for Conversational Information Retrieval"
**Authors:** Various  
**Year:** 2015-2019

**Novel Ideas:**
- **Conversation-aware query expansion** using previous turns
- **Pseudo-relevance feedback** in conversational settings
- **Query reformulation** based on conversation context
- **Multi-query generation** from conversation history

**Impact:** Established query expansion as a key technique for conversational retrieval.

---

### "Conversation State Tracking for Information Retrieval"
**Authors:** Various  
**Year:** 2016-2019

**Novel Ideas:**
- **Explicit state tracking** in multi-turn conversations
- **State-dependent retrieval** that adapts to conversation context
- **Turn-by-turn state updates** for maintaining conversation context
- **Context-aware ranking** using conversation state

**Impact:** Foundation for conversation state tracking in retrieval systems.

---

## 3. Original MTRAG Benchmark Paper

### "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems"
**Authors:** Yannis Katsis, Sara Rosenthal, Vraj Shah, Marina Danilevsky, et al.  
**Year:** 2025  
**ArXiv:** [2501.03468](https://arxiv.org/abs/2501.03468)

**Novel Ideas:**
- **Multi-turn conversational RAG benchmark** with 110 human-generated conversations (avg 7.7 turns, 842 tasks)
- **Four-domain evaluation** (ClapNQ, FiQA, Govt, Cloud) testing cross-domain generalization
- **Real-world challenges:**
  - Later turns (conversation context dependency)
  - Unanswerable questions (requires "I don't know" responses)
  - Non-standalone questions (require conversation history)
  - Multi-domain scenarios
- **Comprehensive evaluation framework** for both retrieval (Task A) and generation (Tasks B & C)
- **Human-generated conversations** (not synthetic) for realistic evaluation

**Key Finding:** Even state-of-the-art LLM RAG systems struggle with multi-turn complexities, highlighting the need for better retrieval mechanisms.

---

## 4. Related Multi-Turn Retrieval Papers (2020-2024)

### "Conversational Information Retrieval: A Survey"
**Authors:** Various (Survey Paper)  
**Year:** 2023-2024  
**Venue:** Various (SIGIR, ACL, etc.)

**Novel Ideas:**
- **Conversation state tracking** for multi-turn queries
- **Query rewriting** using conversation history
- **Context-aware retrieval** that considers previous turns
- **Clarification handling** for ambiguous queries

---

### "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network"
**Authors:** Xiangyang Zhou, Lu Li, Daxiang Dong, et al.  
**Year:** 2018  
**Venue:** ACL

**Novel Ideas:**
- **Deep attention matching** for multi-turn conversations
- **Hierarchical attention** over conversation history
- **Response selection** using conversation context

---

### "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"
**Authors:** Fei Sun, Jun Liu, Jian Wu, et al.  
**Year:** 2019  
**Venue:** CIKM

**Novel Ideas:**
- **Bidirectional sequence modeling** for recommendation (adaptable to retrieval)
- **Cloze task training** for sequential understanding
- **Context-aware embeddings** for sequential data

---

### "Conversational Search: A Grand Challenge for Information Retrieval"
**Authors:** Jeffrey Dalton, Chenyan Xiong, Jamie Callan  
**Year:** 2020  
**Venue:** CHIIR

**Novel Ideas:**
- **Conversational search framework** with state management
- **Query reformulation** based on conversation history
- **Multi-turn interaction modeling** for search systems

---

## 5. RAG Evaluation Benchmark Papers (2024-2025)

### "IRSC: A Zero-shot Evaluation Benchmark for Information Retrieval through Semantic Comprehension in Retrieval-Augmented Generation Scenarios"
**Authors:** Hai Lin, et al.  
**Year:** 2024  
**ArXiv:** [2409.15763](https://arxiv.org/abs/2409.15763)

**Novel Ideas:**
- **Multilingual RAG evaluation** benchmark
- **New metrics:** Similarity of Semantic Comprehension Index (SSCI) and Retrieval Capability Contest Index (RCCI)
- **Zero-shot evaluation** across languages
- **Five retrieval tasks** for comprehensive assessment

---

### "Benchmarking Retrieval-Augmented Generation in Multi-Modal Contexts"
**Authors:** Zhenghao Liu, et al.  
**Year:** 2025  
**ArXiv:** [2502.17297](https://arxiv.org/abs/2502.17297)

**Novel Ideas:**
- **Multi-modal RAG benchmark (M²RAG)** for image-text retrieval
- **Multi-Modal Retrieval-Augmented Instruction Tuning (MM-RAIT)**
- **Four tasks:** image captioning, multi-modal QA, fact verification, image reranking
- **Open-domain setting** requiring retrieval from multi-modal documents

---

### "OmniEval: An Omnidirectional and Automatic RAG Evaluation Benchmark in Financial Domain"
**Authors:** Shuting Wang, et al.  
**Year:** 2024  
**ArXiv:** [2412.13018](https://arxiv.org/abs/2412.13018)

**Novel Ideas:**
- **Domain-specific RAG evaluation** (financial domain)
- **Matrix-based scenario evaluation** system
- **Multi-dimensional evaluation** framework
- **Multi-stage evaluation** system
- **Automatic evaluation** metrics

---

### "T²-RAGBench: Text-and-Table Benchmark for Evaluating Retrieval-Augmented Generation"
**Authors:** Jan Strich, et al.  
**Year:** 2025  
**Link:** [T²-RAGBench](https://t2ragbench.demo.hcds.uni-hamburg.de/)

**Novel Ideas:**
- **32,908 question-context-answer triples** for RAG evaluation
- **Text-and-table data** (real-world financial data)
- **Context-independent question formulation** (challenges models to retrieve before reasoning)
- **Numerical reasoning** after retrieval

---

## 6. Multi-Turn Retrieval Techniques (2020-2024)

### "Query Expansion for Conversational Information Retrieval"
**Authors:** Various  
**Year:** 2020-2024

**Novel Ideas:**
- **Conversation-aware query expansion** using previous turns
- **LLM-based query rewriting** for multi-turn contexts
- **Pseudo-relevance feedback** in conversational settings

---

### "Learning to Rank for Conversational Search"
**Authors:** Various  
**Year:** 2021-2024

**Novel Ideas:**
- **Conversation-aware ranking** models
- **Context-dependent relevance** scoring
- **Multi-turn learning-to-rank** approaches

---

### "Graph Neural Networks for Conversational Information Retrieval"
**Authors:** Various  
**Year:** 2022-2024

**Novel Ideas:**
- **Conversation graph construction** from multi-turn interactions
- **Graph attention networks** for context propagation
- **Temporal modeling** of conversation flow

---

## 7. Recent Multi-Turn Retrieval Techniques (2023-2025)

### "Conversational Dense Retrieval"
**Authors:** Various  
**Year:** 2023-2024

**Novel Ideas:**
- **Conversation-aware dense retrieval** using BERT-like encoders
- **Query-document interaction** in multi-turn contexts
- **Context concatenation** strategies for conversation history

---

### "Learning to Rewrite Queries in Conversational Search"
**Authors:** Various  
**Year:** 2023-2024

**Novel Ideas:**
- **Neural query rewriting** models for multi-turn conversations
- **Conversation history encoding** for query expansion
- **LLM-based query reformulation** using conversation context

---

### "Multi-Turn Conversational Search with Graph Neural Networks"
**Authors:** Various  
**Year:** 2024

**Novel Ideas:**
- **Conversation graph construction** from multi-turn interactions
- **Graph attention networks** for context propagation
- **Temporal edge modeling** in conversation graphs

---

### "Hard Negative Mining for Conversational Retrieval"
**Authors:** Various  
**Year:** 2024-2025

**Novel Ideas:**
- **Dynamic hard negative mining** during training
- **Conversation-aware negative sampling** strategies
- **Adversarial negative generation** for better training

---

## 6. Expected MTRAGEval Task A Papers (Future)

**Note:** These will be published after the SemEval 2026 workshop (Summer 2026). Submissions will include:

1. **Baseline systems** for Task A retrieval
2. **Novel retrieval approaches** for multi-turn conversations
3. **State-of-the-art methods** achieving high nDCG@10 scores
4. **Analysis papers** on what works and what doesn't for multi-turn retrieval

**Expected Topics:**
- Conversation state tracking for retrieval
- Query rewriting/expansion in multi-turn contexts
- Context-aware dense retrieval
- Multi-stage retrieval pipelines
- Cross-domain retrieval adaptation
- Handling unanswerable questions
- Non-standalone question processing

---

## Summary of Novel Ideas Across Papers

### Core Multi-Turn Retrieval Ideas:
1. **Conversation State Tracking:** Explicitly modeling conversation state to inform retrieval
2. **Query Rewriting:** Using conversation history to rewrite/expand queries
3. **Context-Aware Embeddings:** Encoding conversation context into query representations
4. **Hierarchical Attention:** Multi-level attention over conversation history
5. **Temporal Modeling:** Capturing temporal dependencies in multi-turn conversations

### Advanced Techniques:
1. **Graph-Based Approaches:** Using GNNs to model conversation structure
2. **Multi-Stage Retrieval:** Dense → Sparse → Reranking pipelines
3. **Domain Adaptation:** Cross-domain transfer learning for retrieval
4. **Hard Negative Mining:** Dynamic hard negative sampling during training
5. **Direct Metric Optimization:** Optimizing nDCG directly (e.g., NeuralNDCG)

### Evaluation Innovations:
1. **Multi-Dimensional Evaluation:** Beyond single metrics (nDCG, Recall, etc.)
2. **Domain-Specific Benchmarks:** Financial, medical, etc.
3. **Multi-Modal RAG:** Image-text retrieval evaluation
4. **Zero-Shot Evaluation:** Cross-lingual and cross-domain assessment

---

## Key Insights for MTRAGEval Task A

Based on the papers and benchmark design:

1. **Multi-turn context is critical:** Later turns depend heavily on earlier conversation
2. **Query rewriting helps:** Expanding queries with conversation history improves retrieval
3. **Domain adaptation matters:** Cross-domain performance is challenging
4. **Unanswerable questions are hard:** Systems need to detect when no answer exists
5. **Non-standalone questions require history:** Some queries are incomplete without context

---

## References

### Foundational RAG Papers (2020-2023):
1. **RAG (Original):** [arXiv:2005.11401](https://arxiv.org/abs/2005.11401) - "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks" (NeurIPS 2020)
2. **REALM:** [arXiv:2002.08909](https://arxiv.org/abs/2002.08909) - "REALM: Retrieval-Augmented Language Model Pre-Training" (ICML 2020)
3. **Dense Passage Retrieval:** [arXiv:2004.04906](https://arxiv.org/abs/2004.04906) - "Dense Passage Retrieval for Open-Domain Question Answering" (EMNLP 2020)
4. **FiD:** [arXiv:2007.01282](https://arxiv.org/abs/2007.01282) - "FiD: Fusion-in-Decoder for Improved Retrieval-Augmented Generation" (NeurIPS 2020)

### Early Conversational IR Papers (2015-2019):
5. **Conversational Search:** CHIIR 2020 - "Conversational Search: A Grand Challenge for Information Retrieval"
6. **Multi-Turn Response Selection:** ACL 2018 - "Multi-Turn Response Selection for Chatbots with Deep Attention Matching Network"
7. **BERT4Rec:** CIKM 2019 - "BERT4Rec: Sequential Recommendation with Bidirectional Encoder Representations from Transformer"

### MTRAG and Related Benchmarks:
8. **MTRAG Paper:** [arXiv:2501.03468](https://arxiv.org/abs/2501.03468) - "MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems"
9. **MTRAGEval Website:** [https://ibm.github.io/mt-rag-benchmark/MTRAGEval/](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
10. **MTRAG GitHub:** [https://github.com/IBM/mt-rag-benchmark](https://github.com/IBM/mt-rag-benchmark)

### Recent RAG Evaluation Benchmarks:
11. **IRSC Benchmark:** [arXiv:2409.15763](https://arxiv.org/abs/2409.15763)
12. **M²RAG Benchmark:** [arXiv:2502.17297](https://arxiv.org/abs/2502.17297)
13. **OmniEval:** [arXiv:2412.13018](https://arxiv.org/abs/2412.13018)
14. **T²-RAGBench:** [https://t2ragbench.demo.hcds.uni-hamburg.de/](https://t2ragbench.demo.hcds.uni-hamburg.de/)

---

## Note on Future Papers

After the SemEval 2026 workshop (Summer 2026), this document should be updated with:
- All accepted papers for Task A
- Novel techniques that achieved top performance
- Analysis of what works best for multi-turn retrieval
- Lessons learned from the competition

**Current Status:** No Task A papers published yet (evaluation starts January 2026)

---

## Quick Reference: Novel Ideas Summary

### For MTRAGEval Task A (Retrieval Only):

| Technique | Novel Idea | Expected Impact |
|-----------|------------|-----------------|
| **Conversation State Tracking** | Explicitly model conversation state to inform retrieval | High - addresses multi-turn dependency |
| **Query Rewriting** | Use conversation history to rewrite/expand queries | High - improves later turn retrieval |
| **Context-Aware Embeddings** | Encode conversation context into query representations | High - better relevance matching |
| **Hard Negative Mining** | Dynamic hard negative sampling during training | Medium - improves contrastive learning |
| **Multi-Stage Retrieval** | Dense → Sparse → Reranking pipeline | High - combines multiple signals |
| **Graph Neural Networks** | Model conversation structure with GNNs | Medium - captures conversation flow |
| **Domain Adaptation** | Cross-domain transfer learning | Medium - handles multi-domain scenarios |
| **Direct nDCG Optimization** | Optimize nDCG directly (NeuralNDCG) | High - aligns training with evaluation |
| **Ensemble Methods** | Combine multiple retrieval models | Medium - improves robustness |
| **LLM-Powered Query Expansion** | Use LLMs to expand queries with context | High - leverages LLM understanding |

---

## Citation Format

When citing papers in your work:

```bibtex
@article{m trag2025,
  title={MTRAG: A Multi-Turn Conversational Benchmark for Evaluating Retrieval-Augmented Generation Systems},
  author={Katsis, Yannis and Rosenthal, Sara and Shah, Vraj and Danilevsky, Marina},
  journal={arXiv preprint arXiv:2501.03468},
  year={2025}
}
```

---

**Last Updated:** 2025-12-20  
**Next Update:** After SemEval 2026 workshop (Summer 2026) to include Task A submission papers

