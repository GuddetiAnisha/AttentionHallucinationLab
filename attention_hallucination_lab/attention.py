"""Transformer attention extraction utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass
class AttentionExtraction:
    model_name: str
    tokens: list[str]
    attentions: list[np.ndarray]


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().float().cpu().numpy()


def extract_attentions(
    model_name: str,
    prompt: str,
    max_new_tokens: int = 0,
    device: str | None = None,
) -> AttentionExtraction:
    """Extract per-layer attention tensors from a Hugging Face causal LM.

    Returns arrays shaped [heads, query_tokens, key_tokens] for each layer.
    For analysis reproducibility, this function uses eval mode and disables
    sampling. It does not label hallucinations; labels belong to datasets.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
    )

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    model.eval()

    encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(device) for k, v in encoded.items()}

    if max_new_tokens > 0:
        with torch.no_grad():
            generated = model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_attentions=False,
            )
        input_ids = generated.sequences
    else:
        input_ids = encoded["input_ids"]

    attention_mask = torch.ones_like(input_ids, device=device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            use_cache=False,
            return_dict=True,
        )

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    attentions = [_to_numpy(layer[0]) for layer in outputs.attentions]

    return AttentionExtraction(
        model_name=model_name,
        tokens=tokens,
        attentions=attentions,
    )
