"""Airflow DAG: ingest -> clean -> train -> persist the triage classifier.

This is an orchestration layer only. Every task calls straight into the same
plain functions used by `scripts/download_dataset.py`, `scripts/prepare_dataset.py`
and `scripts/train_model.py` — the ones covered by `tests/` — so the DAG
never re-implements logic that's already unit-tested.

Requires the whole `step3/` directory mounted into the Airflow container
(not just `dags/`), so `scripts/`, `data/` and `models/` are reachable as
siblings of this file. Unlike Step 2, this repo doesn't ship a
docker-compose.airflow.yml for this step — see step3/README.md for how
training here is actually run (the `trainer` service in docker-compose.yml).
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from airflow.sdk import dag, task

STEP2_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = STEP2_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RAW_PATH = STEP2_ROOT / "data" / "raw" / "triage_dataset.parquet"
PROCESSED_DIR = STEP2_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "triage_dataset.parquet"
MODELS_DIR = STEP2_ROOT / "models"


@dag(
    dag_id="triage_retraining",
    description="Ingest, clean, train and persist the triage text classifier.",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1, tzinfo=UTC),
    catchup=False,
    tags=["step2", "triage"],
)
def triage_retraining():
    @task
    def ingest() -> str:
        """Download the raw dataset (no-op if already present)."""
        from download_dataset import download

        return str(download(RAW_PATH, force=False))

    @task
    def prepare(raw_path: str) -> str:
        """Clean the raw dataset and write the processed parquet + report."""
        import pandas as pd
        from prepare_dataset import clean

        raw = pd.read_parquet(raw_path)
        cleaned, report = clean(raw)

        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        cleaned.to_parquet(PROCESSED_PATH, index=False)
        (PROCESSED_DIR / "validation_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return str(PROCESSED_PATH)

    @task
    def train_and_persist(processed_path: str) -> dict:
        """Train the classifier and persist model + metrics + pointer."""
        import pandas as pd
        from train_model import save_bundle, train

        df = pd.read_parquet(processed_path)
        bundle, metrics = train(df)

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        model_path = MODELS_DIR / f"{run_id}.joblib"
        save_bundle(bundle, model_path)

        metrics_dir = MODELS_DIR / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        (metrics_dir / f"{run_id}.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )

        pointer = {"run_id": run_id, "model_path": str(model_path)}
        (MODELS_DIR / "current_model.json").write_text(
            json.dumps(pointer, indent=2), encoding="utf-8"
        )
        return {"run_id": run_id, "metrics": metrics}

    train_and_persist(prepare(ingest()))


triage_retraining()
