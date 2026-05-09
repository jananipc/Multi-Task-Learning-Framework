# SentiFraudRAG: Multi-Task Learning Framework for Joint Sentiment Analysis and Fraud Detection in E-Commerce Platforms

A three-layer architecture that jointly performs sentiment analysis and fraud detection on e-commerce product reviews, augmented with retrieval-based explanation generation and hallucination mitigation.

## Architecture

The framework consists of three layers:

### Layer 1 — Multi-Task Classification Model
- **Backbone:** DistilBERT (`distilbert-base-uncased`) with the first 4 transformer layers frozen
- **Input fusion:** CLS token embedding (768-d) concatenated with 12 engineered numerical features
- **Shared MLP:** 768+12 → 256 → 128 with ReLU and dropout
- **Task heads:**
  - Sentiment head: 3-class classification (positive / neutral / negative)
  - Fraud head: binary classification (genuine / fake)
- **Loss:** weighted combination — `0.4 * CrossEntropy(sentiment) + 0.6 * BCEWithLogits(fraud)` with class balancing

### Layer 2 — Sentiment-Aware Hierarchical Retrieval
- FAISS vector index over a curated knowledge base of 8 fraud patterns and 8 behavioral rules
- Sentence-BERT (`all-MiniLM-L6-v2`) encodes review text for cosine similarity search
- Retrieval scope adapts based on fraud probability and sentiment cluster

### Layer 3 — LLM Explanation Generator
- Flan-T5-small generates natural language explanations grounded in retrieved fraud patterns
- Hallucination mitigation module validates outputs against retrieved evidence and flags conflicts

## Dataset

| Property | Value |
|----------|-------|
| Size | 20,000 reviews |
| Sentiment classes | positive (9,541), neutral (5,438), negative (5,021) |
| Fraud labels | genuine (18,051), fake (1,949) |
| Split | 80/20 train/test, stratified by fraud label |

## Engineered Features

| Category | Features |
|----------|----------|
| Textual | exclamation_count, superlative_count, review_length, avg_word_length, capital_ratio |
| Behavioral | burst_activity_flag, rating_deviation, review_count |
| Metadata | verified_purchase, suspicious_rating_flag, helpfulness_ratio, rating |

## Results

| Metric | Score |
|--------|-------|
| Sentiment accuracy | 98% |
| Sentiment weighted F1 | 0.98 |
| Fraud ROC-AUC | ~1.00 |
| Fraud F1 (weighted) | 1.00 |
| MCC (sentiment) | 0.9655 |
| MCC (fraud) | 1.00 |

### Ablation Study

| Configuration | Weighted F1 |
|---------------|-------------|
| Layer 1 only (base model) | 0.940 |
| Layer 1 + 2 (model + retrieval) | 0.960 |
| Full SentiFraudRAG (all 3 layers) | 0.980 |

## Requirements

```
torch
transformers
sentence-transformers
faiss-cpu
scikit-learn
pandas
numpy
matplotlib
seaborn
joblib
```

## Usage

```python
result = predict_review_mode2(
    review_text="This is the best product ever! Buy now! Amazing deal!!!",
    star_rating=5,
    verified_purchase=False
)
print_result(result)
```

**Output includes:**
- Sentiment label and class probabilities
- Fraud probability and binary decision (with optimized threshold)
- Matched fraud pattern IDs from the knowledge base
- Triggered behavioral rules with descriptions
- Natural language explanation
- Confidence score (adjusted for hallucination penalty)

## Saved Artifacts

| File | Description |
|------|-------------|
| `sentifraud_mode2_model.pth` | Trained model weights |
| `distilbert_tokenizer/` | Saved tokenizer |
| `sentiment_encoder.pkl` | Label encoder for sentiment classes |
| `best_threshold.json` | Optimized fraud detection threshold (0.9994) |

## Training Configuration

| Parameter | Value |
|-----------|-------|
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 |
| Weight decay | 0.01 |
| Max sequence length | 128 |
| Optimizer | AdamW |
| Gradient clipping | 1.0 |
| Fraud pos_weight | 9.26 |
