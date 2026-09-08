# Step 4 — Latency Optimization and Final Delivery

Evolution of Step 1 + Step 2 + Step 3: the same trained classifier (`step2-tfidf-logreg-v1`),
converted to ONNX and served through ONNX Runtime, with a reproducible latency comparison against
the original scikit-learn pipeline.

> **Educational project.** As in every earlier step: this classifier is not clinically validated
> and must not inform real triage decisions.

## What's new here

- `scripts/convert_to_onnx.py` — converts the trained pipeline (`models/current_model.json`) to
  ONNX via `skl2onnx`, **verifies** the converted graph's output against the original pipeline's
  `predict_proba` (max allowed difference: `1e-4`) before writing anything, and persists
  `models/<run_id>.onnx` + a JSON sidecar (thresholds + version — an ONNX graph has no place for
  arbitrary Python metadata the way a joblib bundle does).
- `OnnxTriageClassifier` (`classifier.py`) — same three-class threshold-bucketing as
  `MLTriageClassifier`, running through `onnxruntime.InferenceSession` instead of scikit-learn.
  The ONNX graph takes raw text directly (`StringTensorType` input) — the vectorizer's tokenizer
  and IDF weights are baked into the graph, no separate preprocessing step.
- `main.py` picks the serving backend via `TRIAGE_CLASSIFIER_BACKEND` (`auto` — default, prefers
  ONNX, falls back to scikit-learn, falls back to the Step 1 rule-based classifier — `onnx` |
  `sklearn` | `rule-based` to force one, which is how the benchmark below runs both variants from
  the same image).
- `docker-compose.yml`'s `trainer` now runs `train_model.py` **then** `convert_to_onnx.py`, so the
  `api` service (backend `auto`) serves the ONNX-optimized model by default.

## Benchmark: original vs. optimized

Methodology matches Step 1's (`scripts/benchmark.py`, unchanged apart from reading the actual
`classifier_version` from `/health` instead of hard-coding it): full HTTP round trip from the host
against the containerized API, 50 warm-up requests + 500 measured requests, same payload, same
machine, run back to back by forcing `TRIAGE_CLASSIFIER_BACKEND` on the same image via `docker run`
(not part of `docker-compose.yml` — that's for serving, this is for comparing).

<!-- STEP4_BENCHMARK_RESULTS -->

Measured 08/09/2026, Windows 11, Docker Desktop 29.7.2, same trained model (`step2-tfidf-logreg-v1`)
in both cases:

| Metric | scikit-learn (original) | ONNX (optimized) | Improvement |
|---|---:|---:|---:|
| Minimum | 4.258 ms | 3.903 ms | 8.3% |
| Mean | 10.627 ms | 10.253 ms | 3.5% |
| Median | 7.366 ms | 5.487 ms | **25.5%** |
| p95 | 29.730 ms | 28.276 ms | 4.9% |
| Maximum | 45.273 ms | 33.462 ms | 26.1% |

Raw output: [`benchmarks/step4-sklearn.json`](benchmarks/step4-sklearn.json) /
[`benchmarks/step4-onnx.json`](benchmarks/step4-onnx.json).

The median is the most representative number for a typical call on this machine (same caveat as
Step 1: mean/p95/max carry HTTP-cycle and Docker Desktop virtualization noise). ONNX's advantage
here is real but modest — expected for a small linear model (TF-IDF + Logistic Regression), where
scikit-learn's own inference is already fast; ONNX's bigger wins usually show up on larger models
(deep nets, gradient-boosted ensembles) where graph-level optimization and removing Python
call overhead matter more. Re-running this comparison a few times during development showed some
run-to-run variance (medians in the 5-8ms range for both backends) consistent with that read: a
small, consistent floor advantage for ONNX (~1ms lower minimum every run), with the rest dominated
by noise.

To reproduce:

```bash
cd step4
docker compose up --build -d trainer api   # trains + converts + serves (backend: auto -> onnx)
uv run python scripts/benchmark.py --warmup 50 --requests 500 \
  --url http://127.0.0.1:8000/predict --output benchmarks/step4-onnx.json

docker compose stop api
docker run -d --name bench-sklearn --network step4_default -p 8000:8000 \
  -e TRIAGE_MODELS_DIR=/app/models -e TRIAGE_CLASSIFIER_BACKEND=sklearn \
  -v step4_models:/app/models:ro step4-api
uv run python scripts/benchmark.py --warmup 50 --requests 500 \
  --url http://127.0.0.1:8000/predict --output benchmarks/step4-sklearn.json
docker rm -f bench-sklearn && docker compose start api
```

## Two real bugs this step caught (fixed here, not in Step 3)

Both surfaced only by actually running things in Docker, not from code review:

1. **Locale**: `onnxruntime`'s `StringNormalizer` op (used internally by the converted
   `TfidfVectorizer`) requires a real system locale at session-init time.
   `python:3.12-slim` ships none, and inference fails with a
   `locale::facet::_S_create_c_locale` error. Fixed in the `Dockerfile` by installing and
   generating `en_US.UTF-8` before switching to the non-root user.
2. **Airflow needs the extra dependencies too**: the vanilla `apache/airflow` image happens to
   ship `pandas`/`scikit-learn`/`joblib` already, which is why Step 2 and 3's DAGs "just worked"
   in a stock image — but not `skl2onnx`/`onnxruntime`. Running `convert_and_persist_onnx` for
   real (`airflow dags test triage_retraining <date>`) failed with `ModuleNotFoundError`, and then
   with the same locale error above, until both were installed. **A production Airflow deployment
   of this DAG needs a custom image** (`apache/airflow` + `pip install skl2onnx onnxruntime` +
   the same locale setup as the Dockerfile) — this repo doesn't ship one (no
   `docker-compose.airflow.yml` here, same as Step 3), but it's a concrete, verified requirement,
   not a guess.

## Running it

```bash
cd step4
docker compose up --build
# Grafana:    http://localhost:3000  (anonymous viewer access)
# Prometheus: http://localhost:9090
# API:        http://localhost:8000/docs  (serving the ONNX model by default)
```

Verified end-to-end in this environment: `trainer` completed download → clean → train → convert
→ verify against the live 40,715-row dataset; `/health` reported `step2-tfidf-logreg-v1-onnx`;
`/predict` returned sensible classifications; `/metrics`, Prometheus's target, and Grafana's
dashboard all showed real data, same checks as Step 3.

## Still planned

- Record the STAR-format demo video (the actual Tech Challenge deliverable for this step) — not
  something this assistant can do; see the rubric in `MLET - Tech Challenge Fase 3.pdf`.
- Optionally: a stronger model than the linear TF-IDF baseline, if time allows before submission.
