"""Plotting helpers for attention/hallucination experiments."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def plot_layer_metric(values, ylabel: str, title: str, output_path: str | Path):
    values = np.asarray(values, dtype=float)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(len(values)), values, marker="o")
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)


def plot_attention_heatmap(
    attention_matrix,
    tokens,
    title: str,
    output_path: str | Path,
):
    matrix = np.asarray(attention_matrix, dtype=float)
    if matrix.ndim != 2:
        raise ValueError("attention_matrix must be 2D")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    image = ax.imshow(matrix, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")

    if len(tokens) <= 25:
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90, fontsize=7)
        ax.set_yticklabels(tokens, fontsize=7)

    fig.colorbar(image, ax=ax, label="Attention")
    fig.tight_layout()
    fig.savefig(output, dpi=160)
    plt.close(fig)
