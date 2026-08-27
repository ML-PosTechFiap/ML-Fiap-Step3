"""HTTP request and response schemas."""

from typing import Annotated

from pydantic import BaseModel, Field, field_validator

from triage_api.classifier import UrgencyLevel


class TriageRequest(BaseModel):
    """Payload accepted by the triage endpoint."""

    report: Annotated[str, Field(min_length=3, max_length=10_000)]

    @field_validator("report")
    @classmethod
    def normalize_report(cls, value: str) -> str:
        """Reject blank reports and remove surrounding whitespace."""
        normalized_value = value.strip()
        if len(normalized_value) < 3:
            raise ValueError("report must contain at least three non-whitespace characters")
        return normalized_value


class TriageResponse(BaseModel):
    """Classification returned by the triage endpoint."""

    urgency: UrgencyLevel
    classifier_version: str
    processing_time_ms: float


class HealthResponse(BaseModel):
    """Service health information."""

    status: str
    service_version: str
    classifier_version: str
