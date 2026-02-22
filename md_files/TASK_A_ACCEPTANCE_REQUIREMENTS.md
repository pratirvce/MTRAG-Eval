# Task A - Retrieval Only: Acceptance Requirements

**Competition**: SemEval 2026 Multi-Turn RAG (MTRAGEval)  
**Task**: Task A - Retrieval Only  
**Official Page**: [MTRAGEval Competition](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)

---

## 📊 Primary Evaluation Metric

### **nDCG@10** (Normalized Discounted Cumulative Gain at 10)
- **Primary Ranking Metric**: nDCG@10
- **Secondary Metrics**: Recall@10, Recall@5, Recall@3, Recall@1
- **Leaderboard Ranking**: Based on **nDCG@10** score
- **Evaluation Period**: January 10-20, 2026

---

## 🎯 Performance Targets for Acceptance

### **Baseline Scores (From MTRAG Paper)**

| Retriever | Setup | nDCG@10 | Status |
|-----------|-------|---------|--------|
| BM25 | Last Turn | 0.21 | Baseline |
| BM25 | Query Rewrite | 0.25 | Baseline |
| BGE-base 1.5 | Last Turn | 0.30 | Baseline |
| BGE-base 1.5 | Query Rewrite | 0.38 | Baseline |
| **Elser** | Last Turn | **0.49** | Strong baseline |
| **Elser** | Query Rewrite | **0.54** | ⭐ **State-of-the-Art** |

**SOTA Baseline**: **0.54 nDCG@10** (Elser with Query Rewrite)

---

### **Competition Acceptance Targets**

#### **Minimum to be Competitive**:
- **nDCG@10**: **0.50+**
- **Status**: Competitive but may not rank high

#### **Top 10 Leaderboard**:
- **nDCG@10**: **0.52-0.54**
- **Status**: Competitive, likely to rank in top 10
- **Requirement**: Beat or match Elser's 0.54

#### **Top 5 Leaderboard**:
- **nDCG@10**: **0.54-0.56**
- **Status**: Strong performance, top 5 ranking
- **Requirement**: Beat Elser's 0.54 with clear margin

#### **Top 3 Leaderboard**:
- **nDCG@10**: **0.56-0.58**
- **Status**: Exceptional performance, podium finish
- **Requirement**: Significant improvement over SOTA

#### **Winner**:
- **nDCG@10**: **0.58+**
- **Status**: Best performance, likely winner
- **Requirement**: Exceptional improvement over SOTA

---

### **Tier 1 Conference Publication Targets**

For **ACL, NeurIPS, ICML** acceptance with **novel contribution**:

#### **Minimum for Acceptance**:
- **nDCG@10**: **0.55-0.57**
- **Requirement**: Beat SOTA (0.54) with clear margin + strong novelty
- **Acceptance Probability**: 50-65% (EMNLP, NAACL, Findings)

#### **Strong Acceptance**:
- **nDCG@10**: **0.57-0.60**
- **Requirement**: Significant improvement over SOTA + novel contribution
- **Acceptance Probability**: 60-75% (ACL, EMNLP, NAACL)

#### **Top Leaderboard + Novel**:
- **nDCG@10**: **0.60-0.65**
- **Requirement**: Exceptional performance + highly novel + strong analysis
- **Acceptance Probability**: 80-90% (ACL, NeurIPS, ICML)

#### **Best Paper Candidate**:
- **nDCG@10**: **0.65+**
- **Requirement**: Exceptional performance + highly novel + comprehensive analysis
- **Acceptance Probability**: 90%+ (ACL, NeurIPS, ICML)

---

## 📋 Task A Requirements

### **Evaluation Criteria**

1. **Primary Metric**: **nDCG@10** (Normalized Discounted Cumulative Gain at 10)
2. **Task Type**: **Retrieval Only** (no generation required)
3. **Evaluation Period**: January 10-20, 2026
4. **Ranking Method**: Leaderboard ranking based on nDCG@10

### **Output Format Requirements**

Results must be in JSONL format with this structure:

```json
{
  "task_id": "unique_id",
  "contexts": [
    {
      "document_id": "822086267_6698-7277-0-579",  // REQUIRED
      "score": 18.759138,                           // REQUIRED
      "text": "...",                                // Optional
      "title": "...",                               // Optional
      "source": ""                                  // Optional
    }
  ]
}
```

**Required Fields**:
- `document_id`: Must match corpus document IDs exactly
- `score`: Retrieval score (higher = more relevant)

**Optional Fields**:
- `text`, `title`, `source`: Can be included but not required

---

## ✅ What's ALLOWED (During Training/Evaluation)

### **During Training (Trial/Training Data)**:

1. **Metadata Access**: ✅ Can use all metadata:
   - Question type (factoid, composite, etc.)
   - Answerability (answerable, unanswerable, etc.)
   - Multi-turn type (follow-up, clarification)
   - Domain information
   - Full conversation history

2. **Data Usage**: ✅ Can use:
   - Training data (MTRAG Benchmark)
   - Trial data
   - Any publicly available models/embeddings
   - Any training techniques (fine-tuning, domain-specific, etc.)

3. **Model Training**: ✅ Can train:
   - Domain-specific models
   - Multi-domain models
   - Ensemble models
   - Any retrieval architecture

4. **Techniques**: ✅ Allowed:
   - Query rewriting/expansion (LLM-based or rule-based)
   - Domain-specific models
   - Ensemble methods (RRF, weighted average, etc.)
   - Reranking (cross-encoder, etc.)
   - Hybrid retrieval (BM25 + dense)
   - Hard negative mining
   - Contrastive learning
   - Any publicly available models/APIs

### **During Evaluation**:

1. **Domain Information**: ✅ Will be provided
   - Can use domain to select appropriate model/strategy

2. **Query Text**: ✅ Will be provided
   - Can use query for retrieval

3. **Trained Models**: ✅ Can use:
   - Any models trained on training data
   - Pre-trained models
   - External models/APIs

---

## ❌ What's NOT ALLOWED (During Evaluation)

### **Metadata NOT Provided**:

1. **Question Type**: ❌ Not provided
   - Cannot use question type during evaluation

2. **Answerability**: ❌ Not provided
   - Cannot use answerability information

3. **Multi-Turn Type**: ❌ Not provided
   - Cannot use follow-up/clarification labels

4. **Conversation History Details**: ❌ Not provided
   - Only query text will be provided

### **Data Restrictions**:

1. **Evaluation Data**: ❌ Cannot use for training
   - Must not train on evaluation data

2. **Metadata Dependency**: ❌ Cannot depend on metadata not provided
   - Models must work with only domain + query text

---

## 🎯 Your Current Status vs. Requirements

### **Current Performance**:
- **Your Best**: 0.45755 nDCG@10 (Contrastive Learning)
- **SOTA Baseline**: 0.54 nDCG@10 (Elser with Query Rewrite)
- **Gap**: -0.08245 (15.3% improvement needed)

### **Targets**:

| Goal | nDCG@10 | Gap from Current | Status |
|------|---------|------------------|--------|
| **Beat SOTA** | 0.55 | +0.09245 | 🎯 Target |
| **Top 10** | 0.52-0.54 | +0.06245 to +0.08245 | 🎯 Target |
| **Top 5** | 0.54-0.56 | +0.08245 to +0.10245 | 🎯 Target |
| **Tier 1 Min** | 0.55-0.57 | +0.09245 to +0.11245 | 🎯 Target |
| **Tier 1 Strong** | 0.57-0.60 | +0.11245 to +0.14245 | 🎯 Stretch Goal |

---

## 🚀 Running Experiments (Expected Impact)

### **Novel Experiments**:
1. **Enhanced Contrastive Learning**: Expected 0.50-0.54 nDCG@10
2. **QDIT Transformer**: Expected 0.52-0.56 nDCG@10

### **Fixed Experiments**:
3. **Learning-to-Rank**: Expected 0.50-0.53 nDCG@10
4. **Pseudo-Relevance Feedback**: Expected 0.48-0.51 nDCG@10
5. **Cross-Attention**: Expected 0.49-0.52 nDCG@10

### **Combined Potential**:
- **Best Single Method**: 0.52-0.56 nDCG@10
- **Ensemble of Best**: **0.54-0.58 nDCG@10** ✅
  - **Beats SOTA (0.54)!**
  - **Top 5 Leaderboard!**
  - **Tier 1 Publication Quality!**

---

## 📝 Key Requirements Summary

### **For Competition Acceptance**:

1. **Performance**: 
   - Minimum: 0.50+ nDCG@10 (competitive)
   - Top 10: 0.52-0.54 nDCG@10
   - Top 5: 0.54-0.56 nDCG@10
   - Winner: 0.58+ nDCG@10

2. **Output Format**:
   - Must match required JSONL format
   - `document_id` and `score` required
   - Results must be evaluable with official scripts

3. **Compliance**:
   - Cannot use evaluation data for training
   - Cannot depend on metadata not provided
   - Can use domain information (will be provided)

### **For Tier 1 Conference Publication**:

1. **Performance**:
   - Minimum: 0.55-0.57 nDCG@10 + strong novelty
   - Strong: 0.57-0.60 nDCG@10 + novel contribution
   - Exceptional: 0.60-0.65 nDCG@10 + highly novel

2. **Novelty**:
   - Novel architecture or training method
   - Strong theoretical foundation
   - Comprehensive analysis (ablations, error analysis)

3. **Generalizability**:
   - Works across multiple domains
   - Not just dataset-specific improvements

---

## ✅ Compliance Checklist

Before submission, ensure:

- [ ] **Performance**: Achieve target nDCG@10 score
- [ ] **Format**: Output matches required JSONL format
- [ ] **Compliance**: No evaluation data leakage
- [ ] **Metadata**: No dependency on metadata not provided
- [ ] **Domain**: Only use domain information (will be provided)
- [ ] **Evaluation**: Results evaluable with official scripts
- [ ] **Documentation**: All techniques documented

---

## 🎯 Your Path to Acceptance

### **Phase 1: Beat SOTA (0.54)**
- **Target**: 0.55-0.57 nDCG@10
- **Status**: ✅ Running experiments expected to reach this
- **Timeline**: Current experiments should achieve this

### **Phase 2: Top Leaderboard (0.57-0.60)**
- **Target**: 0.57-0.60 nDCG@10
- **Status**: ⏳ Ensemble of best methods
- **Timeline**: After current experiments complete

### **Phase 3: Tier 1 Publication (0.60+)**
- **Target**: 0.60-0.65 nDCG@10
- **Status**: ⏳ Optimal ensemble + novel methods
- **Timeline**: Final optimization phase

---

## 📚 References

- [MTRAGEval Competition Website](https://ibm.github.io/mt-rag-benchmark/MTRAGEval/)
- [MTRAG Benchmark GitHub](https://github.com/IBM/mt-rag-benchmark/)
- [MTRAG Retrieval Results](https://github.com/IBM/mt-rag-benchmark/blob/main/human/retrieval_tasks/README.md)
- [Evaluation README](scripts/evaluation/README.md)

---

**Summary**: To get accepted in Task A, you need to achieve **0.54+ nDCG@10** (beat Elser) for competitive ranking, and **0.55-0.60+ nDCG@10** for Tier 1 conference publication with novel contributions.

---

*Last Updated: 2025-12-17*

