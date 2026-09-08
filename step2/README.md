# Step 2 — CI/CD and Automated Training Pipeline

Evolution of Step 1. This snapshot adds a reproducible data pipeline, CI, and an Airflow
retraining DAG, while preserving the API and Docker behavior delivered in Step 1.

## CI (done)

Two workflows under `../.github/workflows/`:

- **`ci.yml`** — runs on every push/PR to `main`: lint (`ruff check`) + tests for both Step 1 and
  Step 2, each in its own job with its own `uv.lock`. Step 2's job only runs the cleaning/training
  unit tests (synthetic fixtures), so it has no external dependency and stays fast.
- **`dataset-smoke-test.yml`** — weekly (Monday 06:00 UTC) and manually triggerable. Runs the real
  `download_dataset.py` + `prepare_dataset.py` against the live Hugging Face mirror, asserts row
  counts and label set are sane, and publishes the validation report to the job summary. Kept out
  of the main CI gate so an external-service hiccup never blocks a PR.

## Data pipeline (done)

The training dataset is fetched and validated by two scripts, so the whole thing is reproducible
from a clean checkout with no manual downloads or accounts:

```bash
cd step2
uv sync
uv run python scripts/download_dataset.py   # fetch raw data/raw/triage_dataset.parquet
uv run python scripts/prepare_dataset.py     # clean -> data/processed/triage_dataset.parquet
uv run pytest                                # unit tests (synthetic fixtures, fast)
```

See [`docs/dataset.md`](docs/dataset.md) for the dataset source, license, validation findings, and
the label-mapping decision (binary dataset → three-class API contract). Neither `data/raw/` nor
`data/processed/` is versioned — both regenerate from the scripts above.

## Training (done)

`scripts/train_model.py` trains a TF-IDF + Logistic Regression baseline (binary: `urgent` vs
`non-urgent`) and persists a versioned artifact:

```bash
uv run python scripts/train_model.py
# -> models/<run_id>.joblib          (binary artifact, gitignored)
# -> models/metrics/<run_id>.json    (small metrics record, committed)
# -> models/current_model.json       (pointer to the latest run, gitignored)
```

Latest measured baseline (`models/metrics/20260908T173852Z.json`, full 40,715-row processed
dataset, 80/20 stratified split): accuracy 0.717, ROC-AUC 0.765, recall on `urgent` 0.645. Modest
numbers, expected from a linear bag-of-words baseline on noisy patient-authored text — a fine
starting point for Step 4 to improve on (e.g. a stronger model, or better calibration of the
three-class thresholds). See `docs/dataset.md` for how the three-class thresholds are derived from
this model's output probabilities.

## Airflow DAG (done)

`dags/triage_retraining_dag.py` orchestrates `ingest -> prepare -> train_and_persist`, calling the
exact same functions used above (and covered by `tests/`) — it's an orchestration layer, not a
second implementation. Run it locally with Docker:

```bash
cd step2
docker compose -f docker-compose.airflow.yml up
# watch the logs for the auto-generated admin password, then open http://localhost:8080
```

This starts a single-container Airflow (webserver + scheduler + triggerer, SQLite backend) with
the whole `step2/` directory mounted in, so the DAG can import `scripts/` and read/write `data/`
and `models/` exactly like it does outside Airflow. It's a local dev setup, not a production
topology (Airflow prints this warning itself: "Airflow Standalone is for development purposes
only").

Verified end-to-end in this environment via `airflow dags test triage_retraining <date>` inside
the same image (Airflow 3.3.1): all three tasks (`ingest`, `prepare`, `train_and_persist`)
completed with `state=success`, and separately via `docker compose up`, where the API health
endpoint reported `metadatabase`, `scheduler`, `triggerer` and `dag_processor` all `healthy` with
the DAG parsed with 0 errors.

## Still planned

Nothing — Step 3 (`../step3/`) copies this pipeline forward, adds the API back (serving
`models/current_model.json` instead of the Step 1 rule-based classifier), and layers Prometheus +
Grafana on top.
