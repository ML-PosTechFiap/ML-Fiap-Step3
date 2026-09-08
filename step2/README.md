# Step 2 — CI/CD and Automated Training Pipeline

Evolution of Step 1. This snapshot adds a reproducible data pipeline and CI, and will add an
Airflow training DAG, while preserving the API and Docker behavior delivered in Step 1.

## CI (done)

Two workflows under `../.github/workflows/`:

- **`ci.yml`** — runs on every push/PR to `main`: lint (`ruff check`) + tests for both Step 1 and
  Step 2, each in its own job with its own `uv.lock`. Step 2's job only runs the cleaning-logic
  unit tests (synthetic fixture), so it has no external dependency and stays fast.
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
uv run pytest                                # cleaning-logic tests (synthetic fixture, fast)
```

See [`docs/dataset.md`](docs/dataset.md) for the dataset source, license, validation findings, and
the label-mapping decision (binary dataset → three-class API contract). Neither `data/raw/` nor
`data/processed/` is versioned — both regenerate from the scripts above.

## Still planned

- Airflow DAG: ingestion → training → model artifact persistence, reusing
  `prepare_dataset.py`'s cleaning logic.
- Copy Step 1's API into this snapshot once the DAG produces a model artifact for it to serve.
