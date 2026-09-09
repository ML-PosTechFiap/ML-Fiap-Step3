"""Unit tests for the Step 1 rule-based classifier and the Step 3 ML classifier."""

import pandas as pd
import pytest
from train_model import save_bundle, train

from triage_api.classifier import MLTriageClassifier, RuleBasedTriageClassifier, UrgencyLevel


@pytest.fixture
def classifier() -> RuleBasedTriageClassifier:
    return RuleBasedTriageClassifier()


@pytest.mark.parametrize(
    ("report", "expected_urgency"),
    [
        ("Patient reports crushing chest pain and nausea.", UrgencyLevel.URGENT),
        ("Persistent fever and vomiting since yesterday.", UrgencyLevel.ATTENTION),
        ("Routine follow-up with no new symptoms.", UrgencyLevel.NORMAL),
    ],
)
def test_classify_returns_expected_urgency(
    classifier: RuleBasedTriageClassifier,
    report: str,
    expected_urgency: UrgencyLevel,
) -> None:
    assert classifier.classify(report) is expected_urgency


def test_urgent_signal_takes_precedence(
    classifier: RuleBasedTriageClassifier,
) -> None:
    report = "Patient has fever followed by loss of consciousness."

    assert classifier.classify(report) is UrgencyLevel.URGENT


def test_classifier_exposes_step_version(classifier: RuleBasedTriageClassifier) -> None:
    assert classifier.version == "step1-rule-based-v1"


class _StubPipeline:
    """Fake sklearn pipeline returning a fixed P(urgent), to test the
    threshold-bucketing logic in isolation from real model training."""

    def __init__(self, probability_urgent: float) -> None:
        self._probability_urgent = probability_urgent

    def predict_proba(self, _texts: list[str]) -> list[list[float]]:
        return [[1 - self._probability_urgent, self._probability_urgent]]


def _bundle_with_probability(probability_urgent: float) -> dict:
    return {
        "pipeline": _StubPipeline(probability_urgent),
        "thresholds": {"normal_below": 0.3, "urgent_at_or_above": 0.7},
        "version": "step2-tfidf-logreg-v1",
    }


@pytest.mark.parametrize(
    ("probability_urgent", "expected_urgency"),
    [
        (0.1, UrgencyLevel.NORMAL),
        (0.29, UrgencyLevel.NORMAL),
        (0.3, UrgencyLevel.ATTENTION),
        (0.5, UrgencyLevel.ATTENTION),
        (0.69, UrgencyLevel.ATTENTION),
        (0.7, UrgencyLevel.URGENT),
        (0.95, UrgencyLevel.URGENT),
    ],
)
def test_ml_classifier_buckets_by_threshold(
    probability_urgent: float, expected_urgency: UrgencyLevel
) -> None:
    classifier = MLTriageClassifier(_bundle_with_probability(probability_urgent))
    assert classifier.classify("irrelevant, the stub ignores this") is expected_urgency


def test_ml_classifier_exposes_bundle_version() -> None:
    classifier = MLTriageClassifier(_bundle_with_probability(0.5))
    assert classifier.version == "step2-tfidf-logreg-v1"


def test_ml_classifier_loads_a_persisted_bundle(tmp_path) -> None:
    """Round-trip: train a tiny real model, save it, load it back through
    MLTriageClassifier.load, and confirm it classifies without error."""
    df = pd.DataFrame(
        {
            "question": [
                "Severe chest pain and can't breathe, please help now.",
                "Sudden collapse and unresponsive, need help immediately.",
                "Mild headache for two days, no other symptoms.",
                "Small itchy rash on my arm, not spreading.",
            ]
            * 5,
            "triage": ["urgent", "urgent", "non-urgent", "non-urgent"] * 5,
        }
    )
    bundle, _ = train(df, test_size=0.3, random_state=0)
    model_path = tmp_path / "model.joblib"
    save_bundle(bundle, model_path)

    classifier = MLTriageClassifier.load(model_path)

    assert classifier.version == bundle["version"]
    assert classifier.classify("Chest pain, please help") in set(UrgencyLevel)
