# Project Description

## AttentionHallucinationLab

AttentionHallucinationLab is a software-only research prototype for studying whether transformer attention statistics provide useful signals for hallucination-risk estimation.

### Research questions

1. Do attention-sink statistics differ between correct and hallucinated outputs?
2. Do attention-head entropy statistics provide complementary information?
3. Do these signals behave differently for intrinsic and extrinsic hallucinations?
4. Does combining sink and entropy features improve a simple risk-estimation baseline?
5. Which layers or feature groups contribute most consistently across tasks?

### Pipeline

```text
Prompt / Context
      |
Hugging Face Causal LM
      |
Layer/Head Attention Tensors
      |
+-----------------------------+
| Attention Sink Features     |
| Attention Entropy Features  |
+-----------------------------+
      |
Feature Table + Ground-Truth Label
      |
Risk Estimator
      |
Hallucination Probability
      |
Evaluation + Ablation + Error Analysis
```

### Software components

- `attention.py`: Hugging Face model loading and attention extraction
- `features.py`: sink-strength and head-entropy statistics
- `data.py`: controlled hallucination-label loading
- `risk.py`: logistic-regression risk model and metrics
- `experiment.py`: feature ablations and JSON reporting
- `visualize.py`: layer plots and attention heatmaps
- `cli.py`: command-line entry points
- `tests/`: deterministic unit tests
- `samples/`: small synthetic examples

### Evaluation

The project reports accuracy, precision, recall, F1, ROC-AUC where defined, Brier score, and a confusion matrix. Experiments can be separated by hallucination type and task.

### Scientific limitations

The current implementation provides transparent attention-derived baselines. It does not claim exact reproduction of published SinkProbe or Head Entropy methods, causal interpretation of attention, or generalisation to production telecom systems. Those claims require paper-faithful reproduction, larger datasets, multiple model families, careful annotations, and statistical validation.


### Internal explainer extension

The repository also includes `internal_explainer.py`, a lightweight baseline for analysing transformer hidden states. It projects selected hidden-state vectors through the model's output embedding to obtain human-readable top-token summaries. These summaries can be attached to different agent roles and compared using pairwise token-overlap metrics.

This supports small experiments on how an explanation representation for one LLM agent might be compared across multiple agents in a workflow. The method is intentionally simple and should not be described as Jacobian Lens, Natural Language Autoencoders, or formal mechanistic interpretability.
