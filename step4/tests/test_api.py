"""Integration tests for the HTTP API.

Most tests inject a `RuleBasedTriageClassifier` explicitly via `create_app`,
so they're deterministic regardless of whether a trained model happens to be
present on disk locally. The module-level `app` (used in production) picks
its classifier based on that environment state — see
`triage_api.main._load_default_classifier` — so it only gets a lightweight
smoke test here, not classifier-specific assertions.
"""

import pytest
from fastapi.testclient import TestClient

from triage_api.classifier import RuleBasedTriageClassifier
from triage_api.main import app, create_app


def test_default_app_health_check_is_reachable() -> None:
    client = TestClient(app)
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_default_app_exposes_prometheus_metrics() -> None:
    client = TestClient(app)
    response = client.get("/metrics")

    assert response.status_code == 200
    # Prometheus text exposition format always carries HELP/TYPE comments,
    # even before any request has been instrumented.
    assert "# HELP" in response.text
    assert "# TYPE" in response.text


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app(classifier=RuleBasedTriageClassifier()))


def test_health_returns_service_and_classifier_versions(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service_version": "0.1.0",
        "classifier_version": "step1-rule-based-v1",
    }


def test_predict_returns_classification_metadata(client: TestClient) -> None:
    response = client.post(
        "/predict",
        json={"report": "Patient presents with severe chest pain."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["urgency"] == "urgent"
    assert payload["classifier_version"] == "step1-rule-based-v1"
    assert payload["processing_time_ms"] >= 0


def test_predict_rejects_blank_report(client: TestClient) -> None:
    response = client.post("/predict", json={"report": "   "})

    assert response.status_code == 422


def test_predict_rejects_missing_report(client: TestClient) -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422
