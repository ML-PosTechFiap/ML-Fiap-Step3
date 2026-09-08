"""Convert the Step 2 scikit-learn model to ONNX for faster inference.

Reads the trained pipeline pointed at by `models/current_model.json`
(produced by `train_model.py`), converts it with skl2onnx, and writes:

  models/<run_id>.onnx        -- the ONNX graph (raw text in, [P(non-urgent),
                                  P(urgent)] out — the vectorizer runs
                                  inside the graph, no separate preprocessing)
  models/<run_id>.onnx.json   -- thresholds + version sidecar, since an ONNX
                                  graph has no place to carry arbitrary
                                  Python metadata the way a joblib bundle does
  models/current_onnx_model.json -- pointer to the latest ONNX artifact,
                                     mirroring current_model.json

Correctness is checked before writing anything: the ONNX graph's output on a
few sample texts must match the source pipeline's `predict_proba` closely
(see `verify`), since a silent conversion mismatch would defeat the point of
"same model, faster inference".
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

MAX_PROBABILITY_DIFFERENCE = 1e-4

DEFAULT_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
VERIFICATION_SAMPLES = (
    "Severe chest pain and difficulty breathing, started minutes ago.",
    "Mild headache for two days, no fever, feeling fine otherwise.",
    "Patient is asking a general question about vitamin supplements.",
)


def convert(bundle: dict[str, Any]) -> Any:
    """Convert the bundle's sklearn pipeline to an ONNX ModelProto."""
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import StringTensorType

    pipeline = bundle["pipeline"]
    return convert_sklearn(
        pipeline,
        initial_types=[("input", StringTensorType([None, 1]))],
        options={id(pipeline): {"zipmap": False}},
    )


def verify(bundle: dict[str, Any], onnx_model: Any) -> None:
    """Fail loudly if the ONNX graph doesn't reproduce the sklearn
    pipeline's predict_proba within a small numerical tolerance."""
    import numpy as np
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_model.SerializeToString())
    input_name = session.get_inputs()[0].name

    sklearn_probabilities = bundle["pipeline"].predict_proba(list(VERIFICATION_SAMPLES))
    onnx_input = np.array([[text] for text in VERIFICATION_SAMPLES], dtype=object)
    _, onnx_probabilities = session.run(None, {input_name: onnx_input})

    max_difference = float(np.abs(sklearn_probabilities - onnx_probabilities).max())
    if max_difference > MAX_PROBABILITY_DIFFERENCE:
        raise ValueError(
            f"ONNX conversion mismatch: max |sklearn - onnx| probability "
            f"difference is {max_difference}, exceeding the "
            f"{MAX_PROBABILITY_DIFFERENCE} tolerance."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models-dir", type=Path, default=DEFAULT_MODELS_DIR)
    args = parser.parse_args()

    pointer_path = args.models_dir / "current_model.json"
    if not pointer_path.exists():
        raise SystemExit(
            f"No trained scikit-learn model found at {pointer_path}. Run train_model.py first."
        )
    pointer = json.loads(pointer_path.read_text(encoding="utf-8"))

    import joblib

    bundle = joblib.load(pointer["model_path"])

    onnx_model = convert(bundle)
    verify(bundle, onnx_model)

    run_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    onnx_path = args.models_dir / f"{run_id}.onnx"
    onnx_path.write_bytes(onnx_model.SerializeToString())

    sidecar = {
        "thresholds": bundle["thresholds"],
        "version": f"{bundle['version']}-onnx",
        "source_run_id": pointer["run_id"],
    }
    Path(f"{onnx_path}.json").write_text(json.dumps(sidecar, indent=2), encoding="utf-8")

    onnx_pointer = {"run_id": run_id, "model_path": str(onnx_path)}
    (args.models_dir / "current_onnx_model.json").write_text(
        json.dumps(onnx_pointer, indent=2), encoding="utf-8"
    )

    print(f"Converted and verified (max diff <= {MAX_PROBABILITY_DIFFERENCE}).")
    print(f"Saved ONNX model to {onnx_path}")
    print(f"Updated pointer at {args.models_dir / 'current_onnx_model.json'}")


if __name__ == "__main__":
    main()
