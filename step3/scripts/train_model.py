"""Train the binary triage classifier and persist it as a versioned artifact.

Baseline: TF-IDF + Logistic Regression over the cleaned dataset produced by
`prepare_dataset.py`. The model is binary (`urgent` vs `non-urgent`, the
only ground truth the dataset actually supports — see `docs/dataset.md`).

Two probability thresholds are derived from the held-out test set so the API
contract's three classes (`normal` / `attention` / `urgent`) can be served
from this binary model without inventing a supervised label that doesn't
exist in the data:

  p(urgent) < low_threshold             -> normal
  low_threshold <= p(urgent) < high     -> attention
  p(urgent) >= high_threshold           -> urgent

These thresholds are picked from the 33rd/67th percentile of predicted
probabilities on the test set -- a distributional heuristic, not a
clinically validated cutoff. They need human review before any use beyond
this educational project (see the disclaimer in the root README).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

LABEL_MAP = {"non-urgent": 0, "urgent": 1}
LOW_PERCENTILE = 0.33
HIGH_PERCENTILE = 0.67
MODEL_VERSION = "step2-tfidf-logreg-v1"

DEFAULT_INPUT = (
    Path(__file__).resolve().parent.parent / "data" / "processed" / "triage_dataset.parquet"
)
DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def build_pipeline() -> Pipeline:
    """Build the untrained TF-IDF + Logistic Regression pipeline."""
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(max_features=20_000, ngram_range=(1, 2), min_df=2),
            ),
            (
                "classifier",
                LogisticRegression(max_iter=1000, class_weight="balanced"),
            ),
        ]
    )


def train(
    df: pd.DataFrame, test_size: float = 0.2, random_state: int = 42
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Train the classifier and return (model bundle, evaluation metrics)."""
    x = df["question"]
    y = df["triage"].map(LABEL_MAP)
    if y.isnull().any():
        raise ValueError(f"Unmapped labels found: {sorted(set(df['triage']) - set(LABEL_MAP))}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=test_size, random_state=random_state, stratify=y
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    y_proba = pipeline.predict_proba(x_test)[:, 1]
    y_pred = (y_proba >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    low_threshold = float(pd.Series(y_proba).quantile(LOW_PERCENTILE))
    high_threshold = float(pd.Series(y_proba).quantile(HIGH_PERCENTILE))

    metrics = {
        "model_version": MODEL_VERSION,
        "train_rows": len(x_train),
        "test_rows": len(x_test),
        "accuracy": float((y_pred == y_test).mean()),
        "precision_urgent": float(precision_score(y_test, y_pred)),
        "recall_urgent": float(recall_score(y_test, y_pred)),
        "f1_urgent": float(f1_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "confusion_matrix": {
            "true_non_urgent_pred_non_urgent": int(tn),
            "true_non_urgent_pred_urgent": int(fp),
            "true_urgent_pred_non_urgent": int(fn),
            "true_urgent_pred_urgent": int(tp),
        },
        "three_class_thresholds": {
            "normal_below": low_threshold,
            "urgent_at_or_above": high_threshold,
            "method": f"{LOW_PERCENTILE:.0%}/{HIGH_PERCENTILE:.0%} percentile of p(urgent) "
            "on the held-out test set — a heuristic, not a clinically validated cutoff",
        },
    }

    bundle = {
        "pipeline": pipeline,
        "label_map": LABEL_MAP,
        "version": MODEL_VERSION,
        "thresholds": metrics["three_class_thresholds"],
    }
    return bundle, metrics


def save_bundle(bundle: dict[str, Any], output_path: Path) -> None:
    """Persist a trained model bundle to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Processed dataset not found at {args.input}. Run prepare_dataset.py first."
        )

    df = pd.read_parquet(args.input)
    bundle, metrics = train(df)

    # Flat layout, matching the root .gitignore's `**/models/*.joblib` (one level
    # deep, not `models/<run_id>/model.joblib`): the binary artifact is
    # regenerable and ignored, while the small per-run metrics file is a
    # deliberate audit trail and gets committed, same as
    # step1/benchmarks/step1-baseline.json.
    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    model_path = args.models_dir / f"{run_id}.joblib"
    save_bundle(bundle, model_path)

    metrics_dir = args.models_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = metrics_dir / f"{run_id}.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    pointer = {"run_id": run_id, "model_path": str(model_path)}
    (args.models_dir / "current_model.json").write_text(
        json.dumps(pointer, indent=2), encoding="utf-8"
    )

    print(json.dumps(metrics, indent=2))
    print(f"\nSaved model to {model_path}")
    print(f"Saved metrics to {metrics_path}")
    print(f"Updated pointer at {args.models_dir / 'current_model.json'}")


if __name__ == "__main__":
    main()
