import numpy as np

from attention_hallucination_lab.features import (
    attention_feature_vector,
    head_entropy,
    sink_statistics,
)


def normalized_attention(heads=2, tokens=4):
    base = np.ones((heads, tokens, tokens), dtype=float)
    return base / base.sum(axis=-1, keepdims=True)


def test_head_entropy_uniform_is_high():
    attn = normalized_attention()
    entropy = head_entropy(attn)
    assert entropy.shape == (2,)
    assert np.allclose(entropy, 1.0, atol=1e-6)


def test_sink_statistics_detects_concentrated_token():
    attn = normalized_attention()
    attn[:, :, 0] = 0.7
    attn[:, :, 1:] = 0.1
    attn = attn / attn.sum(axis=-1, keepdims=True)

    stats = sink_statistics(attn, strong_sink_threshold=0.5)
    assert stats["sink_max"] > 0.5
    assert stats["strong_sink_count"] >= 1


def test_attention_feature_vector_has_expected_groups():
    layers = [normalized_attention(), normalized_attention()]
    features = attention_feature_vector(layers)

    assert features["num_layers"] == 2.0
    assert "sink_max_mean" in features
    assert "entropy_mean_mean" in features
