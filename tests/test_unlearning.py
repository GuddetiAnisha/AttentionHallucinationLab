import numpy as np
import torch
from torch import nn

from attention_hallucination_lab.unlearning import (
    behavioral_leakage,
    gradient_ascent_unlearn,
    knowledge_recovery_audit,
    residual_knowledge_probe,
    sequential_unlearning_profile,
    summarize_counterfactual,
)


def _toy_model():
    torch.manual_seed(0)
    return nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 2))


def _train_toy_model(model):
    x = torch.tensor(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]], dtype=torch.float32
    )
    y = torch.tensor([0, 0, 1, 1])
    opt = torch.optim.Adam(model.parameters(), lr=0.05)
    for _ in range(200):
        loss = nn.functional.cross_entropy(model(x), y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model


def test_counterfactual_summary_and_profile():
    summary = summarize_counterfactual([0.10, 0.12, 0.08, 0.11])
    assert summary.n_seeds == 4
    assert summary.lower < summary.mean < summary.upper

    profile = sequential_unlearning_profile([0.10, 0.50], summary)
    assert profile[0]["inside_counterfactual_band"] is True
    assert profile[1]["inside_counterfactual_band"] is False


def test_behavioral_leakage():
    assert np.isclose(behavioral_leakage([0.1, 0.2, 0.3]), 0.2)


def test_residual_probe_separable_data():
    rng = np.random.default_rng(1)
    negatives = rng.normal(-2.0, 0.2, size=(30, 4))
    positives = rng.normal(2.0, 0.2, size=(30, 4))
    x = np.vstack([negatives, positives])
    y = np.array([0] * 30 + [1] * 30)

    report = residual_knowledge_probe(x, y, random_state=2)
    assert report.accuracy >= 0.9
    assert report.roc_auc is None or report.roc_auc >= 0.9


def test_gradient_ascent_unlearning_changes_forget_confidence():
    model = _train_toy_model(_toy_model())
    x_forget = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y_forget = torch.tensor([1, 1])
    x_retain = torch.tensor([[0.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    y_retain = torch.tensor([0, 0])

    with torch.no_grad():
        before = torch.softmax(model(x_forget), dim=-1)[:, 1].mean().item()

    unlearned = gradient_ascent_unlearn(
        model,
        [(x_forget, y_forget)],
        retain_batches=[(x_retain, y_retain)],
        steps=40,
        lr=0.01,
        retain_weight=1.0,
    )

    with torch.no_grad():
        after = torch.softmax(unlearned(x_forget), dim=-1)[:, 1].mean().item()

    assert after < before


def test_knowledge_recovery_curve_has_expected_length():
    model = _train_toy_model(_toy_model())
    x = torch.tensor([[1.0, 0.0], [1.0, 1.0]], dtype=torch.float32)
    y = torch.tensor([1, 1])

    def score_fn(candidate):
        with torch.no_grad():
            return float(torch.softmax(candidate(x), dim=-1)[:, 1].mean())

    curve = knowledge_recovery_audit(
        model,
        [(x, y)],
        score_fn,
        steps=3,
        lr=0.001,
    )
    assert [p.step for p in curve] == [0, 1, 2, 3]
