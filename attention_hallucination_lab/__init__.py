"""AttentionHallucinationLab package."""

from .features import attention_feature_vector, head_entropy, sink_statistics
from .risk import train_risk_estimator, evaluate_binary_predictions
from .internal_explainer import (
    ActivationExplanation,
    AgentExplanation,
    explain_hidden_state,
    multi_agent_summary,
    token_overlap,
    top_token_projection,
)

__all__ = [
    "attention_feature_vector",
    "head_entropy",
    "sink_statistics",
    "train_risk_estimator",
    "evaluate_binary_predictions",
    "ActivationExplanation",
    "AgentExplanation",
    "explain_hidden_state",
    "multi_agent_summary",
    "token_overlap",
    "top_token_projection",
]
