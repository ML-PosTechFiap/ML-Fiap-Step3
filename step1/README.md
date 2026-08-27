# Step 1 — Architecture Decision and Initial API

This snapshot contains the initial FastAPI service, the temporary rule-based classifier,
Docker packaging, tests, and the local HTTP latency baseline.

The complete architecture decision, dataset rationale, benchmark results, and evolution history
are documented in the [project README](../README.md).

## Run

```bash
uv sync
uv run uvicorn triage_api.main:app --app-dir src --reload
```

## Verify

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
docker build -t medical-triage-api:step1 .
```
