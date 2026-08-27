"""Unit tests for the Step 1 classifier."""

import pytest

from triage_api.classifier import RuleBasedTriageClassifier, UrgencyLevel


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
