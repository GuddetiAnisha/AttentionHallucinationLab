"""Lightweight internal-activation explainers for single- and multi-agent analysis.

This module provides transparent baselines for inspecting hidden states from
Hugging Face causal language models. It is intentionally *not* an
implementation of Jacobian Lens or Natural Language Autoencoders.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import torch


@dataclass
class ActivationExplanation:
    layer: int
    position: int
    token: str
    top_tokens: list[tuple[str, float]]
    activation_norm: float


@dataclass
class AgentExplanation:
    agent_id: str
    role: str
    explanation: ActivationExplanation


def top_token_projection(
    hidden_state: np.ndarray,
    output_weight: np.ndarray,
    vocabulary: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Project one hidden vector into vocabulary space and return top tokens."""
    hidden = np.asarray(hidden_state, dtype=float)
    weight = np.asarray(output_weight, dtype=float)

    if hidden.ndim != 1:
        raise ValueError("hidden_state must be a 1D vector")
    if weight.ndim != 2 or weight.shape[1] != hidden.shape[0]:
        raise ValueError("output_weight must have shape [vocab, hidden_dim]")
    if len(vocabulary) != weight.shape[0]:
        raise ValueError("vocabulary length must match output_weight rows")

    logits = weight @ hidden
    top_k = max(1, min(int(top_k), len(vocabulary)))
    indexes = np.argsort(logits)[::-1][:top_k]
    return [(vocabulary[i], float(logits[i])) for i in indexes]


def explain_hidden_state(
    hidden_state: np.ndarray,
    output_weight: np.ndarray,
    vocabulary: list[str],
    *,
    layer: int,
    position: int,
    token: str,
    top_k: int = 5,
) -> ActivationExplanation:
    hidden = np.asarray(hidden_state, dtype=float)
    return ActivationExplanation(
        layer=layer,
        position=position,
        token=token,
        top_tokens=top_token_projection(hidden, output_weight, vocabulary, top_k),
        activation_norm=float(np.linalg.norm(hidden)),
    )


def token_overlap(
    first: ActivationExplanation,
    second: ActivationExplanation,
) -> float:
    """Jaccard overlap between the readable top-token projections."""
    a = {token for token, _ in first.top_tokens}
    b = {token for token, _ in second.top_tokens}
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def multi_agent_summary(
    explanations: Iterable[AgentExplanation],
) -> dict[str, object]:
    """Summarise agreement/divergence between agent activation explanations."""
    rows = list(explanations)
    if not rows:
        raise ValueError("at least one agent explanation is required")

    pairwise = []
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            score = token_overlap(rows[i].explanation, rows[j].explanation)
            pairwise.append(
                {
                    "agent_a": rows[i].agent_id,
                    "agent_b": rows[j].agent_id,
                    "top_token_overlap": score,
                }
            )

    mean_overlap = (
        float(np.mean([row["top_token_overlap"] for row in pairwise]))
        if pairwise
        else 1.0
    )

    return {
        "num_agents": len(rows),
        "roles": [row.role for row in rows],
        "mean_pairwise_top_token_overlap": mean_overlap,
        "pairwise": pairwise,
    }


def extract_hidden_state_explanations(
    model_name: str,
    prompt: str,
    layers: list[int] | None = None,
    top_k: int = 5,
    device: str | None = None,
) -> list[ActivationExplanation]:
    """Extract hidden states and project selected layers to readable tokens.

    This uses the model's output embedding / language-model head as a simple,
    transparent projection baseline. It should not be described as J-Lens,
    NLA, or a causal explanation method.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_hidden_states=True,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {key: value.to(device) for key, value in encoded.items()}

    with torch.no_grad():
        outputs = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    hidden_states = outputs.hidden_states
    if hidden_states is None:
        raise RuntimeError("model did not return hidden states")

    if layers is None:
        candidate = sorted(set([1, max(1, len(hidden_states)//2), len(hidden_states)-1]))
        layers = [idx for idx in candidate if 0 <= idx < len(hidden_states)]

    output_embeddings = model.get_output_embeddings()
    if output_embeddings is None:
        raise RuntimeError("model does not expose output embeddings")

    weight = output_embeddings.weight.detach().float().cpu().numpy()
    vocabulary = [
        tokenizer.convert_ids_to_tokens(i)
        for i in range(weight.shape[0])
    ]

    input_ids = encoded["input_ids"][0].detach().cpu().tolist()
    position = len(input_ids) - 1
    token = tokenizer.convert_ids_to_tokens(input_ids[position])

    explanations = []
    for layer in layers:
        if layer < 0 or layer >= len(hidden_states):
            raise ValueError(f"layer index out of range: {layer}")
        hidden = hidden_states[layer][0, position].detach().float().cpu().numpy()
        explanations.append(
            explain_hidden_state(
                hidden,
                weight,
                vocabulary,
                layer=layer,
                position=position,
                token=token,
                top_k=top_k,
            )
        )

    return explanations
