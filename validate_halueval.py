import json
from pathlib import Path

import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from attention_hallucination_lab.attention import extract_attentions
from attention_hallucination_lab.features import attention_feature_vector
from attention_hallucination_lab.risk import evaluate_binary_predictions

MODEL = "sshleifer/tiny-gpt2"
N_DIALOGUES = 40  # 40 correct + 40 hallucinated = 80 examples


def build_prompt(item, response):
    return (
        f"Knowledge: {item['knowledge']}\n"
        f"Dialogue: {item['dialogue_history']}\n"
        f"Response: {response}"
    )


def main():
    print("Downloading HaluEval...")
    dataset = load_dataset(
        "shunk031/HaluEval",
        "dialogue",
        split="train",
    ).select(range(N_DIALOGUES))

    rows = []
    print("Extracting real-model attention features...")

    for i, item in enumerate(dataset):
        examples = [
            (item["right_response"], 0),
            (item["hallucinated_response"], 1),
        ]

        for response, label in examples:
            prompt = build_prompt(item, response)
            extraction = extract_attentions(
                MODEL,
                prompt,
                max_new_tokens=0,
            )
            features = attention_feature_vector(extraction.attentions)
            rows.append({
                **features,
                "is_hallucination": label,
            })

        print(f"Processed {i + 1}/{N_DIALOGUES}")

    out = Path("generated")
    out.mkdir(exist_ok=True)

    frame = pd.DataFrame(rows)
    feature_path = out / "halueval_features.csv"
    frame.to_csv(feature_path, index=False)

    print(f"\nSaved features to: {feature_path}")
    print("Examples:", len(frame))

    feature_columns = [c for c in frame.columns if c != "is_hallucination"]
    X = frame[feature_columns].astype(float)
    y = frame["is_hallucination"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("scale", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])
    model.fit(X_train, y_train)

    probs = model.predict_proba(X_test)[:, 1]
    metrics = evaluate_binary_predictions(y_test, probs)

    print("\n=== REAL HALUEVAL VALIDATION ===")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    metrics_path = out / "halueval_validation_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nValidation results saved to: {metrics_path}")


if __name__ == "__main__":
    main()
