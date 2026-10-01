"""Approximate machine-unlearning baselines and audit utilities.

The module is deliberately model-agnostic and small enough for reproducible
experiments.  It supports:

* gradient-ascent forgetting on a PyTorch classifier/LM-style model;
* optional retain-set regularisation during unlearning;
* multi-seed counterfactual reference summaries;
* behavioral leakage scoring from target-answer probabilities;
* linear representation probes for residual target knowledge;
* knowledge-recovery / relearning curves;
* sequential deletion audit summaries.

This is a research baseline, not a claim of reproducing TOFU, LeakPro, or any
published unlearning method exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

import copy
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from torch import nn


@dataclass(frozen=True)
class CounterfactualSummary:
    mean: float
    std: float
    lower: float
    upper: float
    n_seeds: int


@dataclass(frozen=True)
class ProbeReport:
    accuracy: float
    roc_auc: float | None
    n_samples: int


@dataclass(frozen=True)
class RecoveryPoint:
    step: int
    score: float


def _extract_logits(output: object) -> torch.Tensor:
    """Return logits from common PyTorch / Hugging Face output styles."""
    if isinstance(output, torch.Tensor):
        return output
    if hasattr(output, "logits"):
        return output.logits
    if isinstance(output, Mapping) and "logits" in output:
        return output["logits"]
    raise TypeError("Model output must be a Tensor or expose a 'logits' field")


def _batch_xy(batch: object) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract inputs and labels from tuple/list or mapping batches."""
    if isinstance(batch, Mapping):
        if "labels" not in batch:
            raise KeyError("Mapping batch must contain 'labels'")
        labels = batch["labels"]
        if "input_ids" in batch:
            inputs = batch["input_ids"]
        elif "inputs" in batch:
            inputs = batch["inputs"]
        else:
            raise KeyError("Mapping batch must contain 'input_ids' or 'inputs'")
        return inputs, labels
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    raise TypeError("Batch must be a mapping or (inputs, labels) pair")


def _classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Cross entropy for 2-D classifier logits or token-level 3-D LM logits."""
    if logits.ndim == 2:
        return nn.functional.cross_entropy(logits, labels.long())
    if logits.ndim == 3:
        vocab = logits.shape[-1]
        return nn.functional.cross_entropy(
            logits.reshape(-1, vocab), labels.reshape(-1).long(), ignore_index=-100
        )
    raise ValueError("Expected logits with shape [B,C] or [B,T,V]")


def gradient_ascent_unlearn(
    model: nn.Module,
    forget_batches: Iterable[object],
    *,
    retain_batches: Iterable[object] | None = None,
    steps: int = 20,
    lr: float = 1e-3,
    retain_weight: float = 1.0,
    device: str | torch.device = "cpu",
) -> nn.Module:
    """Run a simple approximate-unlearning baseline.

    The forget loss is *maximised* (gradient ascent) while an optional retain
    loss is minimised.  A deep copy of ``model`` is returned, leaving the
    input model unchanged.
    """
    if steps <= 0:
        raise ValueError("steps must be positive")
    if lr <= 0:
        raise ValueError("lr must be positive")

    unlearned = copy.deepcopy(model).to(device)
    unlearned.train()
    optimiser = torch.optim.Adam(unlearned.parameters(), lr=lr)

    forget_cache = list(forget_batches)
    if not forget_cache:
        raise ValueError("forget_batches must not be empty")
    retain_cache = list(retain_batches) if retain_batches is not None else []

    for step in range(steps):
        f_inputs, f_labels = _batch_xy(forget_cache[step % len(forget_cache)])
        f_inputs = f_inputs.to(device)
        f_labels = f_labels.to(device)
        forget_loss = _classification_loss(_extract_logits(unlearned(f_inputs)), f_labels)

        objective = -forget_loss
        if retain_cache:
            r_inputs, r_labels = _batch_xy(retain_cache[step % len(retain_cache)])
            r_inputs = r_inputs.to(device)
            r_labels = r_labels.to(device)
            retain_loss = _classification_loss(_extract_logits(unlearned(r_inputs)), r_labels)
            objective = objective + retain_weight * retain_loss

        optimiser.zero_grad(set_to_none=True)
        objective.backward()
        optimiser.step()

    return unlearned.eval()


def summarize_counterfactual(
    seed_scores: Sequence[float], confidence_z: float = 1.96
) -> CounterfactualSummary:
    """Summarise a multi-seed target-omitted counterfactual distribution."""
    values = np.asarray(seed_scores, dtype=float)
    if values.size < 2:
        raise ValueError("At least two counterfactual seed scores are required")
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    margin = confidence_z * std
    return CounterfactualSummary(
        mean=mean,
        std=std,
        lower=mean - margin,
        upper=mean + margin,
        n_seeds=int(values.size),
    )


def behavioral_leakage(target_probabilities: Sequence[float]) -> float:
    """Mean probability assigned to the target answer after unlearning."""
    values = np.asarray(target_probabilities, dtype=float)
    if values.size == 0:
        raise ValueError("target_probabilities must not be empty")
    if np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must lie in [0, 1]")
    return float(values.mean())


def residual_knowledge_probe(
    representations: np.ndarray,
    labels: Sequence[int],
    *,
    train_fraction: float = 0.7,
    random_state: int = 0,
) -> ProbeReport:
    """Train a linear probe to detect residual target knowledge.

    ``labels`` should encode whether a representation corresponds to target
    knowledge (1) or a control/retain example (0).
    """
    x = np.asarray(representations, dtype=float)
    y = np.asarray(labels, dtype=int)
    if x.ndim != 2 or len(x) != len(y):
        raise ValueError("representations must be [N,D] and match labels")
    if len(np.unique(y)) < 2:
        raise ValueError("labels must contain at least two classes")
    if not 0.0 < train_fraction < 1.0:
        raise ValueError("train_fraction must lie in (0, 1)")

    rng = np.random.default_rng(random_state)
    order = rng.permutation(len(y))
    split = max(2, min(len(y) - 2, int(round(len(y) * train_fraction))))
    train_idx, test_idx = order[:split], order[split:]

    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(x[train_idx], y[train_idx])
    pred = model.predict(x[test_idx])
    prob = model.predict_proba(x[test_idx])[:, 1]

    auc = None
    if len(np.unique(y[test_idx])) == 2:
        auc = float(roc_auc_score(y[test_idx], prob))
    return ProbeReport(
        accuracy=float(accuracy_score(y[test_idx], pred)),
        roc_auc=auc,
        n_samples=int(len(test_idx)),
    )


def knowledge_recovery_audit(
    model: nn.Module,
    train_batches: Iterable[object],
    score_fn: Callable[[nn.Module], float],
    *,
    steps: int = 10,
    lr: float = 1e-3,
    device: str | torch.device = "cpu",
) -> list[RecoveryPoint]:
    """Measure how quickly target knowledge is recovered during fine-tuning."""
    if steps < 0:
        raise ValueError("steps must be non-negative")
    candidate = copy.deepcopy(model).to(device)
    candidate.train()
    optimiser = torch.optim.Adam(candidate.parameters(), lr=lr)
    batches = list(train_batches)
    if not batches:
        raise ValueError("train_batches must not be empty")

    curve = [RecoveryPoint(step=0, score=float(score_fn(candidate.eval())))]
    candidate.train()
    for step in range(1, steps + 1):
        inputs, labels = _batch_xy(batches[(step - 1) % len(batches)])
        inputs = inputs.to(device)
        labels = labels.to(device)
        loss = _classification_loss(_extract_logits(candidate(inputs)), labels)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        curve.append(RecoveryPoint(step=step, score=float(score_fn(candidate.eval()))))
        candidate.train()
    return curve


def sequential_unlearning_profile(
    deletion_scores: Sequence[float],
    counterfactual: CounterfactualSummary,
) -> list[dict[str, float | int | bool]]:
    """Compare repeated-deletion audit scores with a counterfactual band."""
    profile: list[dict[str, float | int | bool]] = []
    for index, score in enumerate(deletion_scores, start=1):
        value = float(score)
        profile.append(
            {
                "deletion_index": index,
                "score": value,
                "inside_counterfactual_band": counterfactual.lower
                <= value
                <= counterfactual.upper,
                "distance_from_counterfactual_mean": abs(value - counterfactual.mean),
            }
        )
    return profile
