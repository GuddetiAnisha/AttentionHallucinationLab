# AttentionHallucinationLab

A reproducible Python research prototype for studying whether transformer attention statistics can help estimate hallucination risk in language-model outputs.

The project is designed as a software-only experimental framework for:
- extracting layer/head attention tensors from Hugging Face causal language models
- computing attention-sink statistics
- computing attention-head entropy
- comparing correct, intrinsic-hallucination, and extrinsic-hallucination examples
- training a simple hallucination-risk estimator from attention-derived features
- running feature ablations
- exporting metrics and plots for reproducible analysis

## Important scope

This is an independent research prototype. It does **not** claim to reproduce SinkProbe, Head Entropy, or any specific published method exactly unless an experiment explicitly implements and validates that method against its paper. The included sink and entropy features are transparent baseline implementations intended for controlled experiments and portfolio research.

## Project structure

```text
attention_hallucination_lab/
  __init__.py
  attention.py
  features.py
  risk.py
  experiment.py
  data.py
cli.py
samples/
  demo_dataset.jsonl
tests/
  test_features.py
  test_risk.py
requirements.txt
```

## Hallucination labels

The framework uses three labels:

- `correct`: output is supported by the reference/context.
- `intrinsic`: output contradicts information explicitly present in the supplied context/reference.
- `extrinsic`: output introduces unsupported information that is absent from the supplied context/reference.

These labels are supplied by the dataset or annotation process; the framework does not pretend to infer ground truth automatically.

## Attention features

For each attention tensor, the project derives features such as:

- maximum accumulated sink strength
- mean top-k sink strength
- number of strong sink tokens
- mean attention entropy
- minimum attention entropy
- layer-wise sink and entropy summaries

## Risk estimation

A lightweight logistic-regression model estimates hallucination probability from attention-derived features.

Evaluation includes:
- accuracy
- precision
- recall
- F1-score
- ROC-AUC when both classes are present
- Brier score
- confusion matrix

Ablation experiments compare:
- sink features only
- entropy features only
- sink + entropy features

## Install

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux/macOS
source .venv/bin/activate

pip install -r requirements.txt
```

## Run a demo without downloading a large model

The feature and risk modules can be tested using synthetic attention tensors:

```bash
pytest -q
```

## Run attention extraction with a Hugging Face model

```bash
python cli.py extract \
  --model sshleifer/tiny-gpt2 \
  --prompt "Context: Stockholm is the capital of Sweden. Question: What is the capital of Sweden?" \
  --output generated/attention_features.json
```

The first model run may download model weights from Hugging Face.

## Train a risk estimator from a feature table

```bash
python cli.py train \
  --features samples/demo_features.csv \
  --output generated/risk_report.json
```

## Research directions

Useful next steps include:
- reproducing published SinkProbe and Head Entropy methods from their original papers
- evaluating multiple model families and tasks
- QA, mathematical reasoning, and code-generation benchmarks
- carefully annotated intrinsic/extrinsic hallucination datasets
- calibration analysis across model sizes
- layer/head ablations
- memorisation-sensitive evaluation
- bootstrap confidence intervals and statistical significance tests

## CV-safe description

**AttentionHallucinationLab — Attention-Based Hallucination Risk Estimation**

- Built a PyTorch/Hugging Face research prototype for extracting transformer attention tensors and deriving attention-sink and attention-entropy features.
- Structured experiments around correct, intrinsic-hallucination, and extrinsic-hallucination examples.
- Implemented a logistic-regression risk estimator with accuracy, precision, recall, F1, ROC-AUC, Brier score, and confusion-matrix evaluation.
- Added sink-only, entropy-only, and combined feature ablations, reproducible sample data, automated tests, and JSON reporting.

## License

MIT.


## Internal activation explainer extension for agentic AI

A small internal-activation explanation module has been added to support experiments related to explainable LLM agents and multi-agent systems.

### What it adds

- hidden-state extraction from Hugging Face causal language models
- readable token projections from selected transformer layers using the model output head
- per-layer activation norms
- comparison of top projected tokens between agents
- pairwise overlap scores for multi-agent explanation comparison
- a sample two-agent diagnostic workflow
- automated tests for projection and multi-agent comparison logic

### Important limitation

This extension is a transparent baseline for inspecting hidden states. It is **not** an implementation of Jacobian Lens, Natural Language Autoencoders, or a causal mechanistic-interpretability method. It is intended as a research starting point for studying how single-agent explanation signals could be compared across multiple agents.

### CV-safe description

- Extended AttentionHallucinationLab with a hidden-state explanation module for LLM-based agents using readable token projections from selected transformer layers.
- Added a multi-agent comparison workflow that measures agreement and divergence between agent-level internal explanations without exposing hidden chain-of-thought.
