import numpy as np

from attention_hallucination_lab.internal_explainer import (
    AgentExplanation,
    explain_hidden_state,
    multi_agent_summary,
    token_overlap,
    top_token_projection,
)


def test_top_token_projection():
    hidden = np.array([1.0, 0.0])
    weight = np.array([
        [1.0, 0.0],
        [0.5, 0.0],
        [-1.0, 0.0],
    ])
    vocab = ["alpha", "beta", "gamma"]
    result = top_token_projection(hidden, weight, vocab, top_k=2)
    assert result[0][0] == "alpha"
    assert result[1][0] == "beta"


def test_token_overlap_and_multi_agent_summary():
    hidden_a = np.array([1.0, 0.0])
    hidden_b = np.array([0.9, 0.1])
    weight = np.array([
        [1.0, 0.0],
        [0.8, 0.1],
        [0.0, 1.0],
    ])
    vocab = ["route", "queue", "alarm"]

    exp_a = explain_hidden_state(
        hidden_a, weight, vocab, layer=1, position=0, token="x", top_k=2
    )
    exp_b = explain_hidden_state(
        hidden_b, weight, vocab, layer=1, position=0, token="y", top_k=2
    )

    assert 0.0 <= token_overlap(exp_a, exp_b) <= 1.0

    summary = multi_agent_summary([
        AgentExplanation("agent-1", "planner", exp_a),
        AgentExplanation("agent-2", "diagnoser", exp_b),
    ])
    assert summary["num_agents"] == 2
    assert len(summary["pairwise"]) == 1
    assert 0.0 <= summary["mean_pairwise_top_token_overlap"] <= 1.0
