"""Experiment helpers for feature extraction and ablations."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

import pandas as pd

from .risk import ablation_feature_sets, train_risk_estimator


def run_ablation_experiment(
    frame: pd.DataFrame,
    label_column: str = "is_hallucination",
) -> dict[str, dict]:
    feature_sets = ablation_feature_sets(
        [column for column in frame.columns if column != label_column]
    )

    results: dict[str, dict] = {}

    for name, columns in feature_sets.items():
        if not columns:
            results[name] = {"skipped": True, "reason": "no matching features"}
            continue

        trained = train_risk_estimator(
            frame,
            label_column=label_column,
            feature_columns=columns,
        )
        results[name] = {
            "features": columns,
            "metrics": trained.metrics,
        }

    return results


def save_json(data: dict, path: str | Path) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2), encoding="utf-8")
