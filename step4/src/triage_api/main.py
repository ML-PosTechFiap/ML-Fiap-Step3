"""FastAPI application entry point."""

import json
import logging
import os
import time
from pathlib import Path
from time import perf_counter_ns

from fastapi import FastAPI, Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

from triage_api import __version__
from triage_api.classifier import (
    MLTriageClassifier,
    OnnxTriageClassifier,
    RuleBasedTriageClassifier,
    TriageClassifier,
)
from triage_api.schemas import HealthResponse, TriageRequest, TriageResponse

logger = logging.getLogger(__name__)

# Module-level (not created inside create_app): prometheus_client's default
# registry raises on a duplicate metric name, and create_app() is called
# more than once (tests inject different classifiers via create_app(...)).
HTTP_REQUESTS_TOTAL = Counter(
    "http_requests_total",
    "Total HTTP requests, labelled by method, path and status code.",
    ["method", "path", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "http_request_duration_seconds",
    "HTTP request latency in seconds, labelled by method and path.",
    ["method", "path"],
)


def _default_models_dir() -> Path:
    """Resolve the models directory: `TRIAGE_MODELS_DIR` env var if set
    (how Docker points this at the trainer's shared volume), otherwise the
    step3/models/ sibling directory for local, non-containerized runs."""
    override = os.environ.get("TRIAGE_MODELS_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parent.parent.parent / "models"


def _try_load_onnx_classifier(models_dir: Path) -> TriageClassifier | None:
    pointer_path = models_dir / "current_onnx_model.json"
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        return OnnxTriageClassifier.load(Path(pointer["model_path"]))
    except Exception:
        logger.exception("Failed to load ONNX model from %s.", pointer_path)
        return None


def _try_load_sklearn_classifier(models_dir: Path) -> TriageClassifier | None:
    pointer_path = models_dir / "current_model.json"
    if not pointer_path.exists():
        return None
    try:
        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        return MLTriageClassifier.load(Path(pointer["model_path"]))
    except Exception:
        logger.exception("Failed to load scikit-learn model from %s.", pointer_path)
        return None


def _load_default_classifier() -> TriageClassifier:
    """Pick a classifier by `TRIAGE_CLASSIFIER_BACKEND` env var
    (`onnx` | `sklearn` | `rule-based` | `auto`, default `auto`: ONNX if
    available, else scikit-learn, else the Step 1 rule-based fallback).

    Training and ONNX conversion are separate steps (see scripts/train_model.py,
    scripts/convert_to_onnx.py, or the trainer service in docker-compose.yml),
    not a hard dependency of booting the API — whatever hasn't been produced
    yet is simply skipped, and the API always starts.

    Forcing a specific backend (rather than always taking `auto`'s best
    pick) is what lets the same image benchmark scikit-learn vs ONNX
    side by side — see step4/README.md.
    """
    models_dir = _default_models_dir()
    backend = os.environ.get("TRIAGE_CLASSIFIER_BACKEND", "auto").lower()

    if backend == "onnx":
        return _try_load_onnx_classifier(models_dir) or RuleBasedTriageClassifier()
    if backend == "sklearn":
        return _try_load_sklearn_classifier(models_dir) or RuleBasedTriageClassifier()
    if backend == "rule-based":
        return RuleBasedTriageClassifier()

    # auto: prefer the faster, converted model if it exists.
    return (
        _try_load_onnx_classifier(models_dir)
        or _try_load_sklearn_classifier(models_dir)
        or RuleBasedTriageClassifier()
    )


def create_app(classifier: TriageClassifier | None = None) -> FastAPI:
    """Create the API with an injectable classifier implementation."""
    active_classifier = classifier or _load_default_classifier()
    application = FastAPI(
        title="Medical Triage API",
        version=__version__,
        description=(
            "Educational service for urgency classification. "
            "It must not be used for clinical decisions."
        ),
    )

    @application.get("/health", tags=["operations"])
    async def health() -> HealthResponse:
        return HealthResponse(
            status="healthy",
            service_version=__version__,
            classifier_version=active_classifier.version,
        )

    @application.post("/predict", tags=["triage"])
    async def predict(request: TriageRequest) -> TriageResponse:
        started_at = perf_counter_ns()
        urgency = active_classifier.classify(request.report)
        elapsed_ms = (perf_counter_ns() - started_at) / 1_000_000
        return TriageResponse(
            urgency=urgency,
            classifier_version=active_classifier.version,
            processing_time_ms=round(elapsed_ms, 4),
        )

    # Request count, latency and status-code (error) metrics, scraped by
    # Prometheus at /metrics — see prometheus/prometheus.yml and
    # docker-compose.yml. `path` uses the matched route template (e.g.
    # "/predict"), not the raw URL, so it stays a low-cardinality label —
    # an unmatched request (404, no route) falls back to a fixed
    # "unmatched" label instead of the raw URL, which a scanner or a
    # client hitting many distinct nonexistent paths could otherwise use
    # to grow this metric's cardinality without bound.
    @application.middleware("http")
    async def record_metrics(request: Request, call_next):
        started_at = time.perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            return response
        finally:
            route = request.scope.get("route")
            path = route.path if route is not None else "unmatched"
            HTTP_REQUEST_DURATION_SECONDS.labels(method=request.method, path=path).observe(
                time.perf_counter() - started_at
            )
            HTTP_REQUESTS_TOTAL.labels(
                method=request.method, path=path, status_code=status_code
            ).inc()

    @application.get("/metrics", include_in_schema=False)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    return application


app = create_app()
