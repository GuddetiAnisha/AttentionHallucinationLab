"""Command-line interface for AttentionHallucinationLab."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from attention_hallucination_lab.attention import extract_attentions
from attention_hallucination_lab.features import attention_feature_vector
from attention_hallucination_lab.experiment import run_ablation_experiment, save_json
from attention_hallucination_lab.risk import train_risk_estimator


def cmd_extract(args):
    extraction = extract_attentions(
        model_name=args.model,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
    )
    features = attention_feature_vector(extraction.attentions)

    payload = {
        "model": extraction.model_name,
        "tokens": extraction.tokens,
        "features": features,
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {output}")


def cmd_train(args):
    frame = pd.read_csv(args.features)
    result = train_risk_estimator(frame)
    save_json(
        {
            "feature_names": result.feature_names,
            "metrics": result.metrics,
        },
        args.output,
    )
    print(f"Wrote {args.output}")


def cmd_ablate(args):
    frame = pd.read_csv(args.features)
    report = run_ablation_experiment(frame)
    save_json(report, args.output)
    print(f"Wrote {args.output}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Attention-based hallucination analysis research toolkit."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    extract = sub.add_parser("extract", help="extract attention-derived features")
    extract.add_argument("--model", required=True)
    extract.add_argument("--prompt", required=True)
    extract.add_argument("--max-new-tokens", type=int, default=0)
    extract.add_argument("--output", default="generated/attention_features.json")
    extract.set_defaults(func=cmd_extract)

    train = sub.add_parser("train", help="train hallucination-risk estimator")
    train.add_argument("--features", required=True)
    train.add_argument("--output", default="generated/risk_report.json")
    train.set_defaults(func=cmd_train)

    ablate = sub.add_parser("ablate", help="compare sink/entropy feature groups")
    ablate.add_argument("--features", required=True)
    ablate.add_argument("--output", default="generated/ablation_report.json")
    ablate.set_defaults(func=cmd_ablate)

    return parser


def main():
    args = build_parser().parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
