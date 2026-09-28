import pandas as pd

from attention_hallucination_lab.risk import (
    ablation_feature_sets,
    evaluate_binary_predictions,
    train_risk_estimator,
)


def demo_frame():
    return pd.DataFrame(
        {
            "sink_max_mean": [0.10, 0.12, 0.60, 0.68, 0.15, 0.72],
            "sink_topk_mean_mean": [0.09, 0.11, 0.55, 0.61, 0.14, 0.65],
            "entropy_mean_mean": [0.90, 0.88, 0.52, 0.48, 0.85, 0.44],
            "entropy_min_mean": [0.80, 0.78, 0.40, 0.35, 0.76, 0.31],
            "is_hallucination": [0, 0, 1, 1, 0, 1],
        }
    )


def test_train_risk_estimator():
    result = train_risk_estimator(demo_frame())
    assert result.feature_names
    assert 0.0 <= result.metrics["accuracy"] <= 1.0
    assert 0.0 <= result.metrics["brier_score"] <= 1.0


def test_evaluate_binary_predictions():
    metrics = evaluate_binary_predictions([0, 1, 1], [0.1, 0.8, 0.6])
    assert metrics["accuracy"] == 1.0
    assert metrics["roc_auc"] == 1.0


def test_ablation_feature_sets():
    groups = ablation_feature_sets(demo_frame().columns)
    assert groups["sink_only"]
    assert groups["entropy_only"]
    assert len(groups["sink_plus_entropy"]) >= 4
