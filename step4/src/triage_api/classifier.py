"""Classification contract, the Step 1 rule-based baseline, the Step 3 ML
classifier (scikit-learn) and the Step 4 ONNX-optimized classifier."""

from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol


def _bucket_by_threshold(
    probability_urgent: float, thresholds: dict[str, float]
) -> "UrgencyLevel":
    """Shared three-class bucketing: identical logic for the scikit-learn
    and ONNX classifiers, since ONNX is meant to be a drop-in, faster
    replacement for the same trained model, not a different one."""
    if probability_urgent < thresholds["normal_below"]:
        return UrgencyLevel.NORMAL
    if probability_urgent >= thresholds["urgent_at_or_above"]:
        return UrgencyLevel.URGENT
    return UrgencyLevel.ATTENTION


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
        return _bucket_by_threshold(probability_urgent, self._thresholds)

    @classmethod
    def load(cls, model_path: Path) -> "MLTriageClassifier":
        """Load a model bundle persisted by `scripts/train_model.py`."""
        import joblib

        bundle = joblib.load(model_path)
        return cls(bundle)


class OnnxTriageClassifier:
    """Step 4: the same model as `MLTriageClassifier`, converted to ONNX by
    `scripts/convert_to_onnx.py` and served through ONNX Runtime instead of
    scikit-learn — same predictions (same trained weights, just a faster
    inference path), same threshold-bucketing. See step4/README.md for the
    original-vs-optimized latency comparison.
    """

    def __init__(self, session: Any, thresholds: dict[str, float], version: str) -> None:
        self._session = session
        self._input_name = session.get_inputs()[0].name
        self._thresholds = thresholds
        self._version = version

    @property
    def version(self) -> str:
        """Return the converted model's version identifier."""
        return self._version

    def classify(self, report: str) -> UrgencyLevel:
        """Classify text by running the ONNX graph directly on raw text."""
        import numpy as np

        input_tensor = np.array([[report]], dtype=object)
        _, probabilities = self._session.run(None, {self._input_name: input_tensor})
        probability_urgent = float(probabilities[0][1])
        return _bucket_by_threshold(probability_urgent, self._thresholds)

    @classmethod
    def load(cls, model_path: Path) -> "OnnxTriageClassifier":
        """Load an ONNX model persisted by `scripts/convert_to_onnx.py`,
        plus its JSON sidecar (thresholds + version — ONNX graphs carry no
        place for arbitrary Python metadata)."""
        import json

        import onnxruntime as ort

        metadata = json.loads(Path(f"{model_path}.json").read_text(encoding="utf-8"))
        session = ort.InferenceSession(str(model_path))
        return cls(session, metadata["thresholds"], metadata["version"])
