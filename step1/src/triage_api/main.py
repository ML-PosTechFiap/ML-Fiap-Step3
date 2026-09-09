"""FastAPI application entry point."""

from time import perf_counter_ns

from fastapi import FastAPI

from triage_api import __version__
from triage_api.classifier import RuleBasedTriageClassifier, TriageClassifier
from triage_api.schemas import HealthResponse, TriageRequest, TriageResponse


def create_app(classifier: TriageClassifier | None = None) -> FastAPI:
    """Create the API with an injectable classifier implementation."""
    active_classifier = classifier or RuleBasedTriageClassifier()
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

    return application


app = create_app()
