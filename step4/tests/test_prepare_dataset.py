"""Unit tests for the dataset cleaning logic, using a small synthetic fixture
instead of the real 42k-row dataset so the suite stays fast and CI-friendly."""

import pandas as pd
import pytest
from prepare_dataset import MIN_CHARS, clean


@pytest.fixture
def raw_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "question": [
                "Patient reports severe   chest pain and shortness of breath.",
                "Patient reports severe   chest pain and shortness of breath.",  # exact dup
                "  Mild headache for two days, no other symptoms.  ",
                "Mild headache for two days, no other symptoms.",  # same text, conflicting label
                "Hi",  # too short
                "Age",  # too short
            ],
            "triage": [
                "urgent",
                "urgent",
                "non-urgent",
                "urgent",  # conflicts with the row above once whitespace is normalized
                "non-urgent",
                "non-urgent",
            ],
        }
    )


def test_drops_exact_duplicates(raw_df: pd.DataFrame) -> None:
    cleaned, report = clean(raw_df)
    assert report["raw_rows"] == 6
    assert report["after_exact_dedup"] == 5  # one exact (question, triage) duplicate removed


def test_drops_conflicting_labels(raw_df: pd.DataFrame) -> None:
    cleaned, report = clean(raw_df)
    assert report["conflicting_label_texts"] == 1
    assert "mild headache" not in " ".join(cleaned["question"].str.lower())


def test_drops_short_texts(raw_df: pd.DataFrame) -> None:
    cleaned, report = clean(raw_df)
    assert report["too_short_dropped"] == 2
    assert cleaned["question"].str.len().min() >= MIN_CHARS


def test_normalizes_whitespace_and_labels(raw_df: pd.DataFrame) -> None:
    cleaned, _ = clean(raw_df)
    remaining = cleaned.iloc[0]
    assert "  " not in remaining["question"]
    assert remaining["triage"] in {"urgent", "non-urgent"}


def test_rejects_unknown_labels() -> None:
    bad_df = pd.DataFrame(
        {
            "question": ["Some report text that is long enough to pass the length check."],
            "triage": ["unknown-label"],
        }
    )
    with pytest.raises(ValueError, match="Unexpected label values"):
        clean(bad_df)


def test_final_rows_matches_dataframe_length(raw_df: pd.DataFrame) -> None:
    cleaned, report = clean(raw_df)
    assert report["final_rows"] == len(cleaned)
