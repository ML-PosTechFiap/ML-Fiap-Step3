"""Measure HTTP latency for the containerized prediction endpoint."""

import argparse
import json
import statistics
import time
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

DEFAULT_PAYLOAD = {"report": "Patient reports persistent chest pain and difficulty breathing."}


def send_request(url: str) -> float:
    """Send one prediction request and return elapsed milliseconds."""
    # url is a developer-supplied --url CLI argument (a local/CI benchmark
    # target), never untrusted input, so the scheme it could carry isn't a
    # real risk here.
    request = urllib.request.Request(  # noqa: S310
        url,
        data=json.dumps(DEFAULT_PAYLOAD).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started_at = time.perf_counter_ns()
    with urllib.request.urlopen(request, timeout=5) as response:  # noqa: S310
        if response.status != 200:
            raise RuntimeError(f"unexpected HTTP status: {response.status}")
        response.read()
    return (time.perf_counter_ns() - started_at) / 1_000_000


def percentile(values: list[float], percentile_value: float) -> float:
    """Calculate a percentile using linear interpolation."""
    ordered_values = sorted(values)
    position = (len(ordered_values) - 1) * percentile_value
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered_values) - 1)
    fraction = position - lower_index
    return (
        ordered_values[lower_index]
        + (ordered_values[upper_index] - ordered_values[lower_index]) * fraction
    )


def fetch_classifier_version(predict_url: str) -> str:
    """Ask the running service's /health for the classifier actually
    serving predict_url, instead of hard-coding which one that is — the
    same image serves rule-based, scikit-learn, or ONNX depending on
    TRIAGE_CLASSIFIER_BACKEND, so the caller shouldn't have to know which."""
    health_url = predict_url.rsplit("/", 1)[0] + "/health"
    with urllib.request.urlopen(health_url, timeout=5) as response:  # noqa: S310
        return json.loads(response.read())["classifier_version"]


def benchmark(url: str, warmup: int, requests: int) -> dict[str, Any]:
    """Run warm-up calls followed by measured requests."""
    for _ in range(warmup):
        send_request(url)

    measurements = [send_request(url) for _ in range(requests)]
    return {
        "classifier": fetch_classifier_version(url),
        "measured_at_utc": datetime.now(UTC).isoformat(),
        "endpoint": url,
        "warmup_requests": warmup,
        "measured_requests": requests,
        "latency_ms": {
            "minimum": round(min(measurements), 3),
            "mean": round(statistics.fmean(measurements), 3),
            "median": round(statistics.median(measurements), 3),
            "p95": round(percentile(measurements, 0.95), 3),
            "maximum": round(max(measurements), 3),
        },
    }


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    result = benchmark(arguments.url, arguments.warmup, arguments.requests)
    serialized_result = json.dumps(result, indent=2)
    print(serialized_result)
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(f"{serialized_result}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
