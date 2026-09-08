"""Download the raw triage dataset from its public Hugging Face mirror.

Source: https://huggingface.co/datasets/myothiha/triage_dataset (MIT license).
No authentication required. The file is fetched through Hugging Face's public
parquet-conversion API so the download is reproducible without extra tooling
(no `datasets`/`huggingface_hub` dependency, no Kaggle credentials).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import requests

DATASET_API_URL = (
    "https://huggingface.co/api/datasets/myothiha/triage_dataset/parquet/default/train"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent.parent / "data" / "raw" / "triage_dataset.parquet"


def resolve_parquet_url() -> str:
    """Ask the Hugging Face API for the current parquet shard URL(s)."""
    response = requests.get(DATASET_API_URL, timeout=30)
    response.raise_for_status()
    urls = response.json()
    if not urls:
        raise RuntimeError("Hugging Face API returned no parquet shards for this dataset.")
    if len(urls) > 1:
        raise RuntimeError(
            f"Expected a single parquet shard, got {len(urls)}. "
            "Update this script to concatenate shards before proceeding."
        )
    return urls[0]


def download(output_path: Path, force: bool) -> Path:
    """Download the dataset to `output_path` unless it already exists."""
    if output_path.exists() and not force:
        print(f"Already downloaded: {output_path} (use --force to re-download)")
        return output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    url = resolve_parquet_url()
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    output_path.write_bytes(response.content)
    print(f"Downloaded {len(response.content):,} bytes to {output_path}")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Destination file (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if the file already exists.",
    )
    args = parser.parse_args()
    download(args.output, args.force)


if __name__ == "__main__":
    main()
