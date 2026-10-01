"""Semantic-boosted, leakage-aware HaluEval validation.

Adds optional NLI semantic features on top of the previous accuracy-boosted
pipeline. The aim is to improve genuine semantic discrimination rather than
force an artificial 100% score.

Key safeguards:
- correct/hallucinated responses from one dialogue always stay in the same group
- model selection uses grouped CV on training groups only
- threshold tuning uses out-of-fold training predictions only
- held-out test labels are never used for selection/tuning
"""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from transformers import AutoModelForCausalLM, AutoModelForSequenceClassification, AutoTokenizer

from attention_hallucination_lab.features import attention_feature_vector
from attention_hallucination_lab.risk import evaluate_binary_predictions

DEFAULT_MODEL = "gpt2"
DEFAULT_NLI_MODEL = "facebook/bart-large-mnli"
DEFAULT_DIALOGUES = 300
RANDOM_STATE = 42


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def normalize_text(text: str) -> str:
    return " ".join(str(text).lower().split())


def tokenize_simple(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9']+", normalize_text(text)))


def overlap_features(knowledge: str, dialogue: str, response: str) -> dict[str, float]:
    k = tokenize_simple(knowledge)
    d = tokenize_simple(dialogue)
    r = tokenize_simple(response)

    def jaccard(a, b):
        if not a and not b:
            return 1.0
        return len(a & b) / max(1, len(a | b))

    def containment(a, b):
        return len(a & b) / max(1, len(a))

    negations = {"no", "not", "never", "none", "without", "neither", "nor"}
    return {
        "knowledge_response_jaccard": jaccard(k, r),
        "dialogue_response_jaccard": jaccard(d, r),
        "response_in_knowledge": containment(r, k),
        "response_in_dialogue": containment(r, d),
        "response_unique_tokens": float(len(r)),
        "knowledge_unique_tokens": float(len(k)),
        "dialogue_unique_tokens": float(len(d)),
        "response_char_length": float(len(response)),
        "knowledge_char_length": float(len(knowledge)),
        "dialogue_char_length": float(len(dialogue)),
        "response_negation_count": float(len(r & negations)),
        "knowledge_negation_count": float(len(k & negations)),
        "negation_mismatch": float(abs(len(r & negations) - len(k & negations))),
    }


def build_prompt(knowledge: str, dialogue: str, response: str) -> str:
    return f"Knowledge: {knowledge}\nDialogue: {dialogue}\nResponse: {response}"


def safe_perplexity(mean_nll: float) -> float:
    if not np.isfinite(mean_nll):
        return float("nan")
    return float(min(math.exp(min(mean_nll, 20.0)), 1e8))


def load_causal_lm(model_name: str, device: str):
    print(f"Loading causal LM once: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        output_attentions=True,
        output_hidden_states=True,
        attn_implementation="eager",
    ).to(device)
    model.eval()
    return tok, model


def extract_transformer_features(tokenizer, model, prompt: str, device: str):
    encoded = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.to(device)

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_attentions=True,
            output_hidden_states=True,
            use_cache=False,
            return_dict=True,
        )

    attentions = [layer[0].detach().float().cpu().numpy() for layer in outputs.attentions]
    features = attention_feature_vector(attentions)
    features["sequence_length"] = float(input_ids.shape[1])

    logits = outputs.logits[:, :-1, :]
    targets = input_ids[:, 1:]
    if targets.numel() > 0:
        token_nll = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="none",
        )
        mean_nll = float(token_nll.mean().cpu())
        features.update({
            "mean_token_nll": mean_nll,
            "std_token_nll": float(token_nll.std(unbiased=False).cpu()),
            "nll_q25": float(torch.quantile(token_nll, 0.25).cpu()),
            "nll_q75": float(torch.quantile(token_nll, 0.75).cpu()),
            "nll_max": float(token_nll.max().cpu()),
            "perplexity": safe_perplexity(mean_nll),
        })

    hidden_states = outputs.hidden_states
    selected = sorted(set([1, len(hidden_states)//3, 2*len(hidden_states)//3, len(hidden_states)-1]))
    for idx in selected:
        idx = max(0, min(idx, len(hidden_states)-1))
        hidden = hidden_states[idx][0].detach().float()
        norms = hidden.norm(dim=-1)
        features[f"hidden_l{idx}_norm_mean"] = float(norms.mean().cpu())
        features[f"hidden_l{idx}_norm_std"] = float(norms.std(unbiased=False).cpu())
        features[f"hidden_l{idx}_abs_mean"] = float(hidden.abs().mean().cpu())
        features[f"hidden_l{idx}_last_norm"] = float(hidden[-1].norm().cpu())

    return features


class NLIScorer:
    def __init__(self, model_name: str, device: str):
        print(f"Loading NLI model once: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name).to(device)
        self.model.eval()
        self.device = device

        id2label = {int(k): str(v).lower() for k, v in self.model.config.id2label.items()}
        self.entailment_idx = next((k for k, v in id2label.items() if "entail" in v), None)
        self.contradiction_idx = next((k for k, v in id2label.items() if "contrad" in v), None)
        self.neutral_idx = next((k for k, v in id2label.items() if "neutral" in v), None)

        if self.entailment_idx is None:
            self.entailment_idx = 2
        if self.contradiction_idx is None:
            self.contradiction_idx = 0
        if self.neutral_idx is None:
            self.neutral_idx = 1

    def score(self, premise: str, hypothesis: str) -> dict[str, float]:
        encoded = self.tokenizer(
            premise,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits[0]
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

        return {
            "nli_entailment": float(probs[self.entailment_idx]),
            "nli_contradiction": float(probs[self.contradiction_idx]),
            "nli_neutral": float(probs[self.neutral_idx]),
            "nli_entail_minus_contradiction": float(
                probs[self.entailment_idx] - probs[self.contradiction_idx]
            ),
        }


def build_candidates(numeric_columns: list[str], seed: int):
    hybrid = ColumnTransformer([
        ("num", Pipeline([("scale", StandardScaler())]), numeric_columns),
        ("word", TfidfVectorizer(
            lowercase=True, strip_accents="unicode", ngram_range=(1, 2),
            min_df=2, max_df=0.98, max_features=45000, sublinear_tf=True
        ), "text_for_model"),
        ("char", TfidfVectorizer(
            analyzer="char_wb", lowercase=True, ngram_range=(3, 5),
            min_df=2, max_features=55000, sublinear_tf=True
        ), "text_for_model"),
    ], sparse_threshold=0.3)

    hybrid_logistic = Pipeline([
        ("features", hybrid),
        ("classifier", LogisticRegression(
            C=3.0, max_iter=5000, class_weight="balanced",
            random_state=seed, solver="liblinear"
        )),
    ])

    numeric_logistic = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(
            C=1.5, max_iter=4000, class_weight="balanced", random_state=seed
        )),
    ])

    rf = RandomForestClassifier(
        n_estimators=800, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1
    )

    hgb = HistGradientBoostingClassifier(
        learning_rate=0.03, max_iter=450, max_leaf_nodes=15,
        min_samples_leaf=8, l2_regularization=2.0, random_state=seed
    )

    return {
        "hybrid_tfidf_logistic": ("full", hybrid_logistic),
        "numeric_logistic": ("numeric", numeric_logistic),
        "random_forest": ("numeric", rf),
        "hist_gradient_boosting": ("numeric", hgb),
    }


def select_X(frame, mode, numeric_columns):
    if mode == "full":
        return frame[["text_for_model", *numeric_columns]]
    return frame[numeric_columns]


def cross_validate_models(frame_train, y_train, groups_train, numeric_columns, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    report = {}
    candidates = build_candidates(numeric_columns, seed)

    for name, (mode, model) in candidates.items():
        aucs, accs = [], []
        X = select_X(frame_train, mode, numeric_columns)

        for tr, va in cv.split(X, y_train, groups_train):
            est = clone(model)
            est.fit(X.iloc[tr], y_train.iloc[tr])
            probs = est.predict_proba(X.iloc[va])[:, 1]
            metrics = evaluate_binary_predictions(y_train.iloc[va], probs)
            accs.append(metrics["accuracy"])
            if metrics["roc_auc"] is not None:
                aucs.append(metrics["roc_auc"])

        report[name] = {
            "mode": mode,
            "mean_cv_roc_auc": float(np.mean(aucs)),
            "std_cv_roc_auc": float(np.std(aucs)),
            "mean_cv_accuracy": float(np.mean(accs)),
        }
        print(
            f"{name}: CV ROC-AUC={report[name]['mean_cv_roc_auc']:.3f}, "
            f"CV accuracy={report[name]['mean_cv_accuracy']:.3f}"
        )

    best = max(report, key=lambda n: report[n]["mean_cv_roc_auc"])
    return best, report


def tune_threshold_oof(frame_train, y_train, groups_train, numeric_columns, mode, model, seed):
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    X = select_X(frame_train, mode, numeric_columns)
    oof = np.zeros(len(y_train), dtype=float)

    for tr, va in cv.split(X, y_train, groups_train):
        est = clone(model)
        est.fit(X.iloc[tr], y_train.iloc[tr])
        oof[va] = est.predict_proba(X.iloc[va])[:, 1]

    thresholds = np.linspace(0.25, 0.75, 101)
    scores = [accuracy_score(y_train, (oof >= t).astype(int)) for t in thresholds]
    i = int(np.argmax(scores))
    return float(thresholds[i]), float(scores[i])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--nli-model", default=DEFAULT_NLI_MODEL)
    parser.add_argument("--n-dialogues", type=int, default=DEFAULT_DIALOGUES)
    parser.add_argument("--seed", type=int, default=RANDOM_STATE)
    parser.add_argument("--skip-nli", action="store_true")
    parser.add_argument("--skip-transformer-features", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    ds = load_dataset("shunk031/HaluEval", "dialogue", split="train")
    ds = ds.select(range(min(args.n_dialogues, len(ds))))

    lm_tok = lm = None
    if not args.skip_transformer_features:
        lm_tok, lm = load_causal_lm(args.model, device)

    nli = None if args.skip_nli else NLIScorer(args.nli_model, device)

    rows = []
    for i, item in enumerate(ds):
        knowledge = str(item["knowledge"])
        dialogue = str(item["dialogue_history"])

        for response, label, kind in [
            (str(item["right_response"]), 0, "correct"),
            (str(item["hallucinated_response"]), 1, "hallucinated"),
        ]:
            row = {
                "dialogue_id": i,
                "response_type": kind,
                "is_hallucination": label,
                "knowledge": knowledge,
                "dialogue": dialogue,
                "response": response,
                "text_for_model": f"knowledge: {knowledge} dialogue: {dialogue} response: {response}",
            }
            row.update(overlap_features(knowledge, dialogue, response))

            if lm_tok is not None and lm is not None:
                row.update(
                    extract_transformer_features(
                        lm_tok, lm, build_prompt(knowledge, dialogue, response), device
                    )
                )

            if nli is not None:
                row.update(nli.score(knowledge, response))

            rows.append(row)

        if (i + 1) % 10 == 0 or i + 1 == len(ds):
            print(f"Processed {i + 1}/{len(ds)}")

    out = Path("generated")
    out.mkdir(exist_ok=True)
    frame = pd.DataFrame(rows)
    feature_path = out / "halueval_semantic_boost_features.csv"
    frame.to_csv(feature_path, index=False)

    excluded = {
        "dialogue_id", "response_type", "is_hallucination",
        "knowledge", "dialogue", "response", "text_for_model"
    }
    numeric_columns = [
        c for c in frame.columns
        if c not in excluded and pd.api.types.is_numeric_dtype(frame[c])
    ]
    frame[numeric_columns] = (
        frame[numeric_columns]
        .replace([np.inf, -np.inf], np.nan)
        .fillna(frame[numeric_columns].median(numeric_only=True))
        .fillna(0.0)
    )

    y = frame["is_hallucination"].astype(int)
    groups = frame["dialogue_id"].astype(int)

    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=args.seed)
    train_idx, test_idx = next(splitter.split(frame, y, groups))

    train = frame.iloc[train_idx].reset_index(drop=True)
    test = frame.iloc[test_idx].reset_index(drop=True)
    y_train = y.iloc[train_idx].reset_index(drop=True)
    y_test = y.iloc[test_idx].reset_index(drop=True)
    groups_train = groups.iloc[train_idx].reset_index(drop=True)

    print("\n=== TRAINING-GROUP MODEL COMPARISON ===")
    best_name, cv_report = cross_validate_models(
        train, y_train, groups_train, numeric_columns, args.seed
    )

    mode, best_model = build_candidates(numeric_columns, args.seed)[best_name]
    threshold, oof_acc = tune_threshold_oof(
        train, y_train, groups_train, numeric_columns, mode, best_model, args.seed
    )

    X_train = select_X(train, mode, numeric_columns)
    X_test = select_X(test, mode, numeric_columns)
    best_model.fit(X_train, y_train)
    probs = best_model.predict_proba(X_test)[:, 1]

    heldout = evaluate_binary_predictions(y_test, probs, threshold=threshold)
    default = evaluate_binary_predictions(y_test, probs, threshold=0.5)
    baseline = evaluate_binary_predictions(
        y_test,
        np.full(len(y_test), float(y_train.mean())),
        threshold=0.5,
    )

    payload = {
        "dataset": "HaluEval/dialogue",
        "causal_lm": None if args.skip_transformer_features else args.model,
        "nli_model": None if args.skip_nli else args.nli_model,
        "n_dialogues": int(len(ds)),
        "n_rows": int(len(frame)),
        "numeric_feature_count": int(len(numeric_columns)),
        "selected_classifier": best_name,
        "tuned_threshold": threshold,
        "oof_training_accuracy": oof_acc,
        "cross_validation": cv_report,
        "heldout_metrics_tuned_threshold": heldout,
        "heldout_metrics_default_threshold": default,
        "majority_baseline": baseline,
        "notes": [
            "Grouped split prevents paired-response leakage.",
            "Model selection and threshold tuning use training groups only.",
            "NLI semantic features measure knowledge/response entailment and contradiction.",
            "No 100% accuracy target is enforced; results must be reported as observed.",
        ],
    }

    metrics_path = out / "halueval_semantic_boost_metrics.json"
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n=== SEMANTIC-BOOSTED HELD-OUT RESULTS ===")
    for k, v in heldout.items():
        print(f"{k}: {v}")

    print(f"\nFeatures: {feature_path}")
    print(f"Metrics:  {metrics_path}")


if __name__ == "__main__":
    main()
