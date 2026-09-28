"""Attention-sink and attention-entropy feature extraction."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


_EPS = 1e-12


def _validate_attention(attention: np.ndarray) -> np.ndarray:
    array = np.asarray(attention, dtype=float)
    if array.ndim != 3:
        raise ValueError("attention must have shape [heads, query_tokens, key_tokens]")
    if array.shape[1] != array.shape[2]:
        raise ValueError("query_tokens and key_tokens must have equal length")
    if np.any(array < 0):
        raise ValueError("attention weights must be non-negative")
    return array


def head_entropy(attention: np.ndarray, normalized: bool = True) -> np.ndarray:
    """Return one mean entropy value per attention head.

    Entropy is averaged over query-token distributions.
    """
    array = _validate_attention(attention)
    probs = array / np.clip(array.sum(axis=-1, keepdims=True), _EPS, None)
    entropy = -(probs * np.log(np.clip(probs, _EPS, None))).sum(axis=-1)

    if normalized and array.shape[-1] > 1:
        entropy = entropy / math.log(array.shape[-1])

    return entropy.mean(axis=-1)


def sink_strength(attention: np.ndarray) -> np.ndarray:
    """Accumulated attention received by each key token.

    Values are averaged across heads and query positions so sequence lengths
    remain comparable.
    """
    array = _validate_attention(attention)
    return array.mean(axis=0).mean(axis=0)


def sink_statistics(
    attention: np.ndarray,
    strong_sink_threshold: float = 0.20,
    top_k: int = 3,
) -> dict[str, float]:
    strengths = sink_strength(attention)
    if strengths.size == 0:
        raise ValueError("attention sequence must contain at least one token")

    top_k = max(1, min(int(top_k), strengths.size))
    sorted_strengths = np.sort(strengths)[::-1]

    return {
        "sink_max": float(sorted_strengths[0]),
        "sink_topk_mean": float(sorted_strengths[:top_k].mean()),
        "sink_mean": float(strengths.mean()),
        "strong_sink_count": float((strengths >= strong_sink_threshold).sum()),
    }


def layer_feature_vector(attention: np.ndarray) -> dict[str, float]:
    entropies = head_entropy(attention)
    sinks = sink_statistics(attention)
    return {
        **sinks,
        "entropy_mean": float(entropies.mean()),
        "entropy_min": float(entropies.min()),
        "entropy_max": float(entropies.max()),
        "entropy_std": float(entropies.std()),
    }


def attention_feature_vector(attentions: Iterable[np.ndarray]) -> dict[str, float]:
    """Aggregate attention features across layers."""
    layers = [layer_feature_vector(a) for a in attentions]
    if not layers:
        raise ValueError("at least one attention layer is required")

    result: dict[str, float] = {"num_layers": float(len(layers))}
    keys = list(layers[0].keys())

    for key in keys:
        values = np.asarray([layer[key] for layer in layers], dtype=float)
        result[f"{key}_mean"] = float(values.mean())
        result[f"{key}_max"] = float(values.max())
        result[f"{key}_min"] = float(values.min())

    return result
