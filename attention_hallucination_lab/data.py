"""Dataset helpers for hallucination experiments."""

from __future__ import annotations

import json
from pathlib import Path


VALID_LABELS = {"correct", "intrinsic", "extrinsic"}


def load_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            label = row.get("label")
            if label not in VALID_LABELS:
                raise ValueError(
                    f"line {line_number}: label must be one of {sorted(VALID_LABELS)}"
                )
            rows.append(row)
    return rows


def is_hallucination(label: str) -> int:
    if label not in VALID_LABELS:
        raise ValueError(f"unknown label: {label}")
    return int(label != "correct")
