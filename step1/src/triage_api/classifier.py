"""Classification contract and the temporary Step 1 implementation."""

from enum import StrEnum
from typing import Protocol


class UrgencyLevel(StrEnum):
    """Urgency levels exposed by the public API."""

    NORMAL = "normal"
    ATTENTION = "attention"
    URGENT = "urgent"


class TriageClassifier(Protocol):
    """Contract implemented by every classifier version."""

    @property
    def version(self) -> str:
        """Return the classifier version identifier."""

    def classify(self, report: str) -> UrgencyLevel:
        """Classify a medical report by urgency."""


class RuleBasedTriageClassifier:
    """Deterministic Step 1 baseline that will be replaced by a trained model."""

    _URGENT_SIGNALS = (
        "anaphylaxis",
        "cardiac arrest",
        "chest pain",
        "difficulty breathing",
        "loss of consciousness",
        "severe bleeding",
        "stroke",
        "unconscious",
    )
    _ATTENTION_SIGNALS = (
        "dizziness",
        "fever",
        "fracture",
        "infection",
        "persistent pain",
        "shortness of breath",
        "vomiting",
    )

    @property
    def version(self) -> str:
        """Return the semantic identifier for the temporary classifier."""
        return "step1-rule-based-v1"

    def classify(self, report: str) -> UrgencyLevel:
        """Classify text using ordered keyword groups."""
        normalized_report = " ".join(report.casefold().split())

        if self._contains_any(normalized_report, self._URGENT_SIGNALS):
            return UrgencyLevel.URGENT
        if self._contains_any(normalized_report, self._ATTENTION_SIGNALS):
            return UrgencyLevel.ATTENTION
        return UrgencyLevel.NORMAL

    @staticmethod
    def _contains_any(report: str, signals: tuple[str, ...]) -> bool:
        return any(signal in report for signal in signals)
