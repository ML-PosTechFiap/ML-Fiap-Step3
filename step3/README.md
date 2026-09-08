# Step 3 — Monitoring and Observability

Evolution of Step 1 + Step 2: the API from Step 1, evolved to serve the model trained in Step 2
instead of the rule-based baseline, instrumented with Prometheus metrics and observed through a
provisioned Grafana dashboard.

## What's here

- `src/triage_api/` — Step 1's API, with a new `MLTriageClassifier` (in `classifier.py`) that
  loads the Step 2 model bundle and buckets its `P(urgent)` output into the API's three classes
  via the two calibrated thresholds persisted alongside it. `main.py` loads it automatically if
  `models/current_model.json` exists, and **falls back to the Step 1 rule-based classifier
  otherwise** — the API always boots, even before any model has been trained.
- `scripts/`, `dags/` — copied from Step 2 unchanged: the data pipeline, training script and
  Airflow DAG. See `docs/dataset.md` for the dataset write-up.
- Prometheus instrumentation: a small middleware in `main.py` (no external instrumentator
  library — see "Why not prometheus-fastapi-instrumentator" below) exposing `/metrics` with
  request counts, latency histograms and status codes, all labelled by route.
- `docker-compose.yml` — the full local stack: `trainer` (runs the Step 2 pipeline once into a
  shared volume) → `api` (serves it) → `prometheus` (scrapes it) → `grafana` (dashboards it).

## Running it

```bash
cd step3
docker compose up --build
```

This builds one image (used by both `trainer` and `api` — they share every dependency), runs the
trainer to completion (download → clean → train, ~1-2 minutes with a cold cache), then starts the
API already pointed at that model, plus Prometheus and Grafana:

- API: http://localhost:8000/docs
- Prometheus: http://localhost:9090
- Grafana: http://localhost:3000 (anonymous viewer access enabled — no login needed for the
  provisioned "Triage API" dashboard; sign in as `admin`/`admin` only if you need to edit)

Send it some traffic and watch the dashboard update:

```bash
for i in $(seq 1 20); do
  curl -s -X POST http://localhost:8000/predict \
    -H "Content-Type: application/json" \
    -d '{"report":"Patient reports severe chest pain and shortness of breath."}' > /dev/null
done
```

Without Docker: `uv sync && uv run uvicorn triage_api.main:app --app-dir src --reload` (falls back
to the rule-based classifier unless you've run `scripts/train_model.py` locally first, which
writes into `./models/`, picked up automatically).

## Verified in this environment

- `uv run pytest` — 31 tests (API, both classifiers, data cleaning, training), all passing.
- `uv run ruff check .` — clean.
- Real Docker Compose run: `trainer` completed the full pipeline against the live 40,715-row
  dataset and produced a model; `api` loaded it (`/health` reports the trained model's version,
  not the rule-based fallback); `/metrics` showed real counts after driving traffic through
  `/predict`; Prometheus's target for `api:8000` was `up`; Grafana's provisioned dashboard queried
  real data back from Prometheus.

## Why not `prometheus-fastapi-instrumentator`

Tried first — it transitively pins an older Starlette, which downgraded the whole project below
the Starlette version whose `TestClient` supports `httpx2` (Step 1's actual HTTP test dependency;
see its `pyproject.toml`), breaking every test that touches the API. A dozen lines of
`prometheus-client` + one ASGI middleware exposes the exact same three things the rubric asks for
(requests, latency, errors) without importing a stale package or fighting the rest of the stack's
already-current versions.

## Still planned (Step 4)

- Replace the linear baseline with a stronger model and/or better-calibrated thresholds.
- Convert to ONNX and benchmark original vs. optimized latency, reusing `step1/scripts/benchmark.py`.
