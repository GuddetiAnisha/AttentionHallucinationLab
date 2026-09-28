"""AttentionHallucinationLab package."""

from .features import attention_feature_vector, head_entropy, sink_statistics
from .risk import train_risk_estimator, evaluate_binary_predictions

__all__ = [
    "attention_feature_vector",
    "head_entropy",
    "sink_statistics",
    "train_risk_estimator",
    "evaluate_binary_predictions",
]
