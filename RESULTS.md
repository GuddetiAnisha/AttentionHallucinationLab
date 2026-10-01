# Real HaluEval Validation Results

## Final semantic-boost experiment

The completed experiment used **300 HaluEval dialogue groups (600 labeled responses)** with leakage-aware grouped evaluation.

### Training-group cross-validation

| Classifier | CV ROC-AUC | CV Accuracy |
|---|---:|---:|
| Hybrid TF-IDF Logistic Regression | 0.838 | 0.744 |
| Numeric Logistic Regression | 0.833 | 0.733 |
| **Random Forest** | **0.854** | **0.771** |
| HistGradientBoosting | 0.838 | 0.756 |

The classifier selected by training-group cross-validation was **Random Forest**.

### Held-out results

| Metric | Result |
|---|---:|
| Accuracy | **0.700** |
| Precision | **0.650** |
| Recall | **0.867** |
| F1 | **0.743** |
| ROC-AUC | **0.786** |
| Brier score | **0.190** |
| Confusion matrix | `[[40, 35], [10, 65]]` |

### Interpretation

The final semantic-boosted pipeline reached **70% held-out accuracy**, **0.79 ROC-AUC**, and **86.7% recall**. The evaluation kept paired correct/hallucinated responses from the same dialogue together, so the held-out result is not based on direct pair leakage.

These results are benchmark-specific and should not be presented as universal hallucination-detection performance.
