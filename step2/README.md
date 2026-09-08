# Step 2 — CI/CD and Automated Training Pipeline

Evolution of Step 1. This snapshot adds a reproducible data pipeline, and will add GitHub Actions
and an Airflow training DAG, while preserving the API and Docker behavior delivered in Step 1.

## Data pipeline (done)

The training dataset is fetched and validated by two scripts, so the whole thing is reproducible
from a clean checkout with no manual downloads or accounts:

```bash
cd step2
uv sync
uv run python scripts/download_dataset.py   # fetch raw data/raw/triage_dataset.parquet
uv run python scripts/prepare_dataset.py     # clean -> data/processed/triage_dataset.parquet
uv run pytest                                # cleaning-logic tests (synthetic fixture, fast)
```

See [`docs/dataset.md`](docs/dataset.md) for the dataset source, license, validation findings, and
the label-mapping decision (binary dataset → three-class API contract). Neither `data/raw/` nor
`data/processed/` is versioned — both regenerate from the scripts above.

## Still planned

- GitHub Actions workflow: lint + tests on every push (Step 1's API tests plus this step's
  data-pipeline tests).
- Airflow DAG: ingestion → training → model artifact persistence, reusing
  `prepare_dataset.py`'s cleaning logic.
- Copy Step 1's API into this snapshot once the DAG produces a model artifact for it to serve.
