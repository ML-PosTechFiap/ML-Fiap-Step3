"""Classification contract, the Step 1 rule-based baseline, and the Step 3
ML classifier trained in Step 2."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


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


class MLTriageClassifier:
    """Step 3: the binary TF-IDF + Logistic Regression model trained in
    Step 2 (see step2/scripts/train_model.py), bucketed into the API's
    three classes via the probability thresholds persisted alongside it.

    Those thresholds are a distributional heuristic (percentiles of
    predicted P(urgent) on a held-out set), not a clinically validated
    cutoff — see step2/docs/dataset.md and the disclaimer in the root
    README. This classifier is not fit for real triage decisions.
    """

    def __init__(self, bundle: dict[str, Any]) -> None:
        self._pipeline = bundle["pipeline"]
        self._thresholds = bundle["thresholds"]
        self._version = bundle["version"]

    @property
    def version(self) -> str:
        """Return the trained model's version identifier."""
        return self._version

    def classify(self, report: str) -> UrgencyLevel:
        """Classify text using the model's P(urgent) against two thresholds."""
        probability_urgent = self._pipeline.predict_proba([report])[0][1]

        if probability_urgent < self._thresholds["normal_below"]:
            return UrgencyLevel.NORMAL
        if probability_urgent >= self._thresholds["urgent_at_or_above"]:
            return UrgencyLevel.URGENT
        return UrgencyLevel.ATTENTION

    @classmethod
    def load(cls, model_path: Path) -> "MLTriageClassifier":
        """Load a model bundle persisted by `scripts/train_model.py`."""
        import joblib

        bundle = joblib.load(model_path)
        return cls(bundle)
