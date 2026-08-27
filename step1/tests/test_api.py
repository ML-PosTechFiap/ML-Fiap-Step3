"""Integration tests for the HTTP API."""

from fastapi.testclient import TestClient

from triage_api.main import app

client = TestClient(app)


def test_health_returns_service_and_classifier_versions() -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "healthy",
        "service_version": "0.1.0",
        "classifier_version": "step1-rule-based-v1",
    }


def test_predict_returns_classification_metadata() -> None:
    response = client.post(
        "/predict",
        json={"report": "Patient presents with severe chest pain."},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["urgency"] == "urgent"
    assert payload["classifier_version"] == "step1-rule-based-v1"
    assert payload["processing_time_ms"] >= 0


def test_predict_rejects_blank_report() -> None:
    response = client.post("/predict", json={"report": "   "})

    assert response.status_code == 422


def test_predict_rejects_missing_report() -> None:
    response = client.post("/predict", json={})

    assert response.status_code == 422
