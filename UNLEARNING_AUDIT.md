# Machine Unlearning Audit Extension

This repository now includes a software-only baseline for studying the central question behind the AI Sweden thesis **"Erased or Suppressed? Auditing LLM Unlearning"**: does an approximate unlearning procedure truly remove target information, or does it only suppress its observable output?

## Implemented components

The module `attention_hallucination_lab/unlearning.py` contains:

1. **Approximate unlearning baseline**
   - gradient-ascent forgetting on target/forget examples
   - optional retain-set regularization to preserve non-target behavior
   - works with standard PyTorch classifiers and simple Hugging Face-style logits outputs

2. **Multi-seed counterfactual summary**
   - aggregates evaluation scores from independently trained target-omitted reference models
   - reports mean, sample standard deviation, and a configurable comparison band
   - intended to avoid treating one reference checkpoint as the entire counterfactual

3. **Behavioral leakage audit**
   - measures residual probability assigned to target answers after unlearning
   - provides a simple output-level leakage baseline

4. **Representation-level residual knowledge probe**
   - trains a logistic-regression probe on hidden representations
   - reports accuracy and ROC-AUC when available
   - can be combined with hidden-state extraction already present in AttentionHallucinationLab

5. **Knowledge-recovery audit**
   - fine-tunes an unlearned model on target data for a controlled number of steps
   - records a recovery curve to study how rapidly target behavior returns
   - useful for distinguishing stable forgetting from easily recoverable suppression

6. **Sequential unlearning profile**
   - compares successive deletion-request scores against a multi-seed counterfactual band
   - supports longitudinal experiments over repeated deletion requests

## Tests

`tests/test_unlearning.py` adds deterministic tests for:

- counterfactual summaries
- sequential deletion profiles
- behavioral leakage scoring
- residual-knowledge probes
- gradient-ascent unlearning behavior
- relearning/recovery curves

Run the complete test suite with:

```bash
pytest -q
```

## Important scope

This is an independent research baseline. It does **not** claim to reproduce TOFU, LeakPro, or any published machine-unlearning algorithm exactly. It is intended to provide a transparent experimental foundation that can later be connected to:

- target-omitted multi-seed language-model training
- TOFU-style forget/retain datasets
- Hugging Face hidden-state extraction
- adversarial prompting and steering tests
- quantization and fine-tuning stress tests
- sequential or overlapping deletion requests
- LeakPro-compatible result export

## CV-safe wording

**AttentionHallucinationLab — LLM Unlearning Audit Extension**

- Implemented a research baseline for approximate machine unlearning using gradient-ascent forgetting with optional retain-set regularization.
- Added multi-seed counterfactual summaries, behavioral leakage scoring, representation-level linear probes, knowledge-recovery curves, and sequential deletion auditing.
- Connected the audit design to existing hidden-state analysis and reproducible evaluation utilities, with deterministic Pytest coverage.
- Designed the extension to distinguish surface-level suppression from residual internal knowledge without claiming exact reproduction of TOFU, LeakPro, or a published unlearning method.
