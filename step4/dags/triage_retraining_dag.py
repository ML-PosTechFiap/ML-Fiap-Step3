"""Airflow DAG: ingest -> clean -> train -> persist -> convert to ONNX.

This is an orchestration layer only. Every task calls straight into the same
plain functions used by `scripts/download_dataset.py`, `scripts/prepare_dataset.py`,
`scripts/train_model.py` and `scripts/convert_to_onnx.py` — the ones covered
by `tests/` — so the DAG never re-implements logic that's already unit-tested.

Requires the whole `step4/` directory mounted into the Airflow container
(not just `dags/`), so `scripts/`, `data/` and `models/` are reachable as
siblings of this file. This repo doesn't ship a docker-compose.airflow.yml
for this step (same as Step 3) — see step4/README.md for how training is
actually run here (the `trainer` service in docker-compose.yml), and for
the extra dependencies (skl2onnx, onnxruntime) plus the locale setup a
production Airflow image would need to run convert_and_persist_onnx.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

from airflow.sdk import dag, task

STEP4_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = STEP4_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

RAW_PATH = STEP4_ROOT / "data" / "raw" / "triage_dataset.parquet"
PROCESSED_DIR = STEP4_ROOT / "data" / "processed"
PROCESSED_PATH = PROCESSED_DIR / "triage_dataset.parquet"
MODELS_DIR = STEP4_ROOT / "models"


@dag(
    dag_id="triage_retraining",
    description="Ingest, clean, train, persist and ONNX-convert the triage text classifier.",
    schedule="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["step4", "triage"],
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
    def train_and_persist(processed_path: str) -> str:
        """Train the classifier and persist model + metrics + pointer.

        Returns the sklearn model's run_id (not the full metrics dict): the
        next task re-derives its own run_id when it converts to ONNX, so
        passing a big dict through XCom for no reason is avoided.
        """
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
        return run_id

    @task
    def convert_and_persist_onnx(sklearn_run_id: str) -> dict:
        """Convert the just-trained model to ONNX, verify it against the
        sklearn pipeline's predict_proba, and persist model + sidecar +
        pointer.

        Loads the model by `sklearn_run_id` (the exact model
        `train_and_persist` just built, passed through XCom) rather than
        re-reading `current_model.json` — a concurrent DAG run could have
        overwritten that pointer in between the two tasks, which would
        silently convert the wrong model and mislabel the sidecar's
        `source_run_id`.
        """
        import joblib
        from convert_to_onnx import convert, verify

        model_path = MODELS_DIR / f"{sklearn_run_id}.joblib"
        bundle = joblib.load(model_path)

        onnx_model = convert(bundle)
        verify(bundle, onnx_model)

        run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        onnx_path = MODELS_DIR / f"{run_id}.onnx"
        onnx_path.write_bytes(onnx_model.SerializeToString())

        sidecar = {
            "thresholds": bundle["thresholds"],
            "version": f"{bundle['version']}-onnx",
            "source_run_id": sklearn_run_id,
        }
        Path(f"{onnx_path}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

        onnx_pointer = {"run_id": run_id, "model_path": str(onnx_path)}
        (MODELS_DIR / "current_onnx_model.json").write_text(
            json.dumps(onnx_pointer, indent=2), encoding="utf-8"
        )
        return {"run_id": run_id, "source_sklearn_run_id": sklearn_run_id}

    convert_and_persist_onnx(train_and_persist(prepare(ingest())))


triage_retraining()
