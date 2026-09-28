"""Hallucination-risk estimation and evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


@dataclass
class RiskModelResult:
    model: Pipeline
    feature_names: list[str]
    metrics: dict[str, object]


def evaluate_binary_predictions(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float = 0.5,
) -> dict[str, object]:
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)

    if len(y_true) != len(probabilities):
        raise ValueError("y_true and probabilities must have the same length")
    if len(y_true) == 0:
        raise ValueError("evaluation requires at least one example")

    predictions = (probabilities >= threshold).astype(int)

    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision_score(y_true, predictions, zero_division=0)),
        "recall": float(recall_score(y_true, predictions, zero_division=0)),
        "f1": float(f1_score(y_true, predictions, zero_division=0)),
        "brier_score": float(brier_score_loss(y_true, probabilities)),
        "confusion_matrix": confusion_matrix(
            y_true,
            predictions,
            labels=[0, 1],
        ).tolist(),
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(roc_auc_score(y_true, probabilities))
    else:
        metrics["roc_auc"] = None

    return metrics


def train_risk_estimator(
    frame: pd.DataFrame,
    label_column: str = "is_hallucination",
    feature_columns: Sequence[str] | None = None,
    random_state: int = 42,
) -> RiskModelResult:
    if label_column not in frame:
        raise ValueError(f"missing label column: {label_column}")

    if feature_columns is None:
        feature_columns = [
            column
            for column in frame.columns
            if column != label_column and pd.api.types.is_numeric_dtype(frame[column])
        ]

    feature_columns = list(feature_columns)
    if not feature_columns:
        raise ValueError("no numeric features available")

    X = frame[feature_columns].astype(float)
    y = frame[label_column].astype(int)

    if y.nunique() < 2:
        raise ValueError("training requires both correct and hallucinated examples")

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    random_state=random_state,
                    max_iter=2000,
                ),
            ),
        ]
    )
    model.fit(X, y)

    probabilities = model.predict_proba(X)[:, 1]
    metrics = evaluate_binary_predictions(y, probabilities)

    return RiskModelResult(
        model=model,
        feature_names=feature_columns,
        metrics=metrics,
    )


def ablation_feature_sets(columns: Sequence[str]) -> dict[str, list[str]]:
    columns = list(columns)
    sink = [c for c in columns if "sink" in c]
    entropy = [c for c in columns if "entropy" in c]

    return {
        "sink_only": sink,
        "entropy_only": entropy,
        "sink_plus_entropy": sorted(set(sink + entropy)),
    }
