# Day 18 - Threshold Optimization

## Objective

Learn how to optimize the decision threshold of a binary classification model using validation data.

The model outputs probabilities, and different thresholds produce different classification results.

---

## Key Concepts

- Classification probability
- Decision threshold
- Precision
- Recall
- F1 Score
- Precision-Recall trade-off
- Threshold search
- Best threshold selection
- Validation-based model selection
- Data leakage

---

## Pipeline

```text
Best Model
    ↓
Validation Dataset
    ↓
Logits
    ↓
Sigmoid
    ↓
Probabilities
    ↓
Threshold Search
    ↓
Classification Metrics
    ↓
Best Threshold