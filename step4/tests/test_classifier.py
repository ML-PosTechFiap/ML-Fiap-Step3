"""Unit tests for the rule-based, scikit-learn, and ONNX classifiers."""

import pandas as pd
import pytest
from convert_to_onnx import convert, verify
from train_model import save_bundle, train

from triage_api.classifier import (
    MLTriageClassifier,
    OnnxTriageClassifier,
    RuleBasedTriageClassifier,
    UrgencyLevel,
)


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

    def predict_proba(self, texts: list[str]) -> list[list[float]]:
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


@pytest.fixture
def trained_bundle() -> dict:
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
    return bundle


def test_onnx_conversion_matches_sklearn_predictions(trained_bundle: dict) -> None:
    """`convert_to_onnx.verify` raises on a mismatch — this test just
    confirms it doesn't, i.e. the ONNX graph reproduces the sklearn
    pipeline's predict_proba within tolerance."""
    onnx_model = convert(trained_bundle)
    verify(trained_bundle, onnx_model)  # raises ValueError on mismatch


def test_onnx_classifier_loads_and_classifies(trained_bundle: dict, tmp_path) -> None:
    """Round-trip: convert a real trained model to ONNX, save it plus its
    sidecar exactly as convert_to_onnx.py would, load it back through
    OnnxTriageClassifier.load, and confirm it classifies without error."""
    import json

    onnx_model = convert(trained_bundle)
    model_path = tmp_path / "model.onnx"
    model_path.write_bytes(onnx_model.SerializeToString())
    sidecar = {
        "thresholds": trained_bundle["thresholds"],
        "version": f"{trained_bundle['version']}-onnx",
        "source_run_id": "test-run",
    }
    (tmp_path / "model.onnx.json").write_text(json.dumps(sidecar))

    classifier = OnnxTriageClassifier.load(model_path)

    assert classifier.version == f"{trained_bundle['version']}-onnx"
    assert classifier.classify("Chest pain, please help") in set(UrgencyLevel)


def test_onnx_and_sklearn_classifiers_agree_on_the_same_inputs(trained_bundle: dict) -> None:
    """The whole point of Step 4 is "same model, faster inference" — so for
    a batch of varied texts, both classifiers must reach the same verdict."""
    import onnxruntime as ort

    sklearn_classifier = MLTriageClassifier(trained_bundle)

    onnx_model = convert(trained_bundle)
    session = ort.InferenceSession(onnx_model.SerializeToString())
    onnx_classifier = OnnxTriageClassifier(
        session, trained_bundle["thresholds"], trained_bundle["version"]
    )

    texts = [
        "Severe chest pain and can't breathe right now.",
        "Mild headache, feeling mostly fine.",
        "Small itchy rash, not spreading.",
        "Sudden collapse, need help immediately.",
    ]
    for text in texts:
        assert sklearn_classifier.classify(text) == onnx_classifier.classify(text)
