"""Validate and clean the raw triage dataset into a training-ready artifact.

Applies the cleaning rules documented in `docs/dataset.md`:
  1. Collapse duplicate whitespace / trim each question text.
  2. Drop exact duplicate (question, triage) rows.
  3. Drop question texts that carry conflicting labels across duplicates
     (ambiguous ground truth, cannot be trusted).
  4. Drop texts shorter than MIN_CHARS after stripping (greetings, stray
     metadata like "Age" or "[URL=...", not real clinical content).
  5. Normalize the label column (lowercase, stripped).

Writes the cleaned dataset to `data/processed/triage_dataset.parquet` plus a
JSON validation report (row counts per stage, class balance, length stats)
so every run is auditable.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

MIN_CHARS = 15
VALID_LABELS = {"urgent", "non-urgent"}

DEFAULT_INPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "triage_dataset.parquet"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Clean the raw dataframe, returning the result and a stage-by-stage report."""
    report: dict[str, Any] = {"raw_rows": len(df)}

    working = df.copy()
    working["question"] = working["question"].str.strip().str.split().str.join(" ")
    working["triage"] = working["triage"].str.strip().str.lower()

    unknown_labels = sorted(set(working["triage"].unique()) - VALID_LABELS)
    if unknown_labels:
        raise ValueError(f"Unexpected label values found: {unknown_labels}")

    working = working.drop_duplicates(subset=["question", "triage"])
    report["after_exact_dedup"] = len(working)

    conflicting_texts = working.groupby("question")["triage"].nunique()
    conflicting_texts = conflicting_texts[conflicting_texts > 1].index
    report["conflicting_label_texts"] = len(conflicting_texts)
    working = working[~working["question"].isin(conflicting_texts)]
    report["after_conflict_removal"] = len(working)

    too_short = working["question"].str.len() < MIN_CHARS
    report["too_short_dropped"] = int(too_short.sum())
    working = working[~too_short]
    report["after_short_text_removal"] = len(working)

    working = working.reset_index(drop=True)
    report["final_rows"] = len(working)
    report["label_distribution"] = working["triage"].value_counts().to_dict()
    report["question_length_chars"] = {
        "min": int(working["question"].str.len().min()),
        "p50": float(working["question"].str.len().median()),
        "p95": float(working["question"].str.len().quantile(0.95)),
        "max": int(working["question"].str.len().max()),
    }
    return working, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    if not args.input.exists():
        raise SystemExit(
            f"Raw dataset not found at {args.input}. Run download_dataset.py first."
        )

    raw = pd.read_parquet(args.input)
    cleaned, report = clean(raw)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "triage_dataset.parquet"
    cleaned.to_parquet(output_path, index=False)

    report_path = args.output_dir / "validation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(json.dumps(report, indent=2))
    print(f"\nSaved cleaned dataset to {output_path}")
    print(f"Saved validation report to {report_path}")


if __name__ == "__main__":
    main()
