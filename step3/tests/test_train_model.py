"""Unit tests for the training pipeline, using a small synthetic dataset with
an obvious separating signal so the pipeline exercises real learning without
needing the full 40k-row processed dataset."""

import pandas as pd
import pytest
from train_model import LABEL_MAP, save_bundle, train

URGENT_TEMPLATES = [
    "Severe chest pain and difficulty breathing, started {n} minutes ago, need help now.",
    "Sudden loss of consciousness and cardiac arrest symptoms, unresponsive, {n} years old.",
    "Massive bleeding from a deep wound after an accident, patient is {n} years old and pale.",
    "Signs of stroke: face drooping, slurred speech, one arm weak, onset {n} minutes ago.",
]
NON_URGENT_TEMPLATES = [
    "Mild headache for the past {n} days, no fever, otherwise feeling fine.",
    "Small itchy rash on my arm for {n} days, not spreading, no other symptoms.",
    "Occasional dry cough for {n} days, no fever, wondering if I should see a doctor.",
    "Question about a vitamin supplement I've been taking for {n} weeks, no side effects.",
]


@pytest.fixture
def synthetic_df() -> pd.DataFrame:
    rows = []
    for i in range(15):
        urgent_text = URGENT_TEMPLATES[i % len(URGENT_TEMPLATES)].format(n=i + 1)
        rows.append({"question": urgent_text, "triage": "urgent"})
        non_urgent_text = NON_URGENT_TEMPLATES[i % len(NON_URGENT_TEMPLATES)].format(n=i + 1)
        rows.append({"question": non_urgent_text, "triage": "non-urgent"})
    return pd.DataFrame(rows)


def test_train_returns_bundle_and_metrics(synthetic_df: pd.DataFrame) -> None:
    bundle, metrics = train(synthetic_df, test_size=0.3, random_state=0)

    assert set(bundle) == {"pipeline", "label_map", "version", "thresholds"}
    assert bundle["label_map"] == LABEL_MAP
    assert 0.0 <= metrics["accuracy"] <= 1.0
    assert 0.0 <= metrics["roc_auc"] <= 1.0
    assert metrics["train_rows"] + metrics["test_rows"] == len(synthetic_df)


def test_thresholds_are_ordered(synthetic_df: pd.DataFrame) -> None:
    _, metrics = train(synthetic_df, test_size=0.3, random_state=0)
    thresholds = metrics["three_class_thresholds"]
    assert 0.0 <= thresholds["normal_below"] <= thresholds["urgent_at_or_above"] <= 1.0


def test_bundle_predicts_probabilities_for_new_text(synthetic_df: pd.DataFrame) -> None:
    bundle, _ = train(synthetic_df, test_size=0.3, random_state=0)
    proba = bundle["pipeline"].predict_proba(["Chest pain and can't breathe, please help"])
    assert proba.shape == (1, 2)
    assert abs(proba.sum() - 1.0) < 1e-6


def test_rejects_unmapped_labels() -> None:
    bad_df = pd.DataFrame(
        {
            "question": ["Some report text that is long enough to pass any length check."] * 4,
            "triage": ["unmapped"] * 4,
        }
    )
    with pytest.raises(ValueError, match="Unmapped labels"):
        train(bad_df)


def test_save_bundle_writes_joblib_file(synthetic_df: pd.DataFrame, tmp_path) -> None:
    bundle, _ = train(synthetic_df, test_size=0.3, random_state=0)
    output_path = tmp_path / "nested" / "model.joblib"

    save_bundle(bundle, output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
