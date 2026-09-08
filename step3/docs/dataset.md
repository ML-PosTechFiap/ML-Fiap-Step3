# Dataset — triage text classification

## Source

[`myothiha/triage_dataset`](https://huggingface.co/datasets/myothiha/triage_dataset) — also
mirrored on [Kaggle](https://www.kaggle.com/datasets/myothiha/triage-dataset). MIT license, no
credentialing required. Downloaded through Hugging Face's public parquet-conversion API (see
`scripts/download_dataset.py`), which needs no account, API key, or extra SDK.

This replaces MIMIC-IV-ED as the dataset actually used for training. MIMIC-IV-ED remains
documented in the root `README.md` as the clinically-grounded option, but it requires PhysioNet
credentialing (account, CITI training, signed DUA) that blocks immediate, reproducible use in an
academic pipeline. `myothiha/triage_dataset` is used instead so Step 2's pipeline and Step 4's
model can run end-to-end without an external approval step.

## Raw shape

- 42,513 rows, 2 columns: `question` (free text, patient-authored health question) and `triage`
  (binary label: `urgent` / `non-urgent`).
- Text is closer to an online health-forum question than a clinical chief-complaint note — expect
  informal language, first person, occasional greetings/metadata noise.

## Validation findings (see `scripts/prepare_dataset.py`)

| Issue | Count | Handling |
|---|---:|---|
| Exact duplicate `(question, triage)` rows | 1,447 | Dropped, keep first occurrence |
| Question texts with conflicting labels across duplicates | 163 texts | Dropped entirely — ground truth is ambiguous, can't be trusted either way |
| Texts under 15 characters after stripping (`"Hi"`, `"Age"`, `"."`, stray `[URL=...]`) | 25 | Dropped — not real clinical content |
| Long-tail length (up to 71,913 chars) | n/a | Kept as-is; length capping is a training-time concern (tokenizer/vectorizer max length), not a data-quality issue, so it's left to the Step 4 modeling code |

Result: **40,715 clean rows** — `non-urgent`: 29,533 (72.5%), `urgent`: 11,182 (27.5%). Moderate
class imbalance; Step 4 training should account for it (class weights or stratified
resampling) rather than relying on plain accuracy.

Run `uv run python scripts/prepare_dataset.py` to regenerate
`data/processed/triage_dataset.parquet` and `data/processed/validation_report.json` from the raw
file. Neither `data/raw/` nor `data/processed/` is versioned (see root `.gitignore`); both are
reproducible from `download_dataset.py` + `prepare_dataset.py`.

## Label mapping: binary dataset → three-class API contract

The API contract (`normal` / `attention` / `urgent`, see `step1/src/triage_api/classifier.py`)
has three classes; this dataset only supports two (`urgent` / `non-urgent`). Rather than inventing
a fake `attention` label with no ground truth behind it, the plan for Step 4 is:

1. Train a binary classifier (`urgent` vs `non-urgent`) on this dataset — that's what the labels
   actually support.
2. At inference, use the model's predicted probability for `urgent` and bucket it into the
   three-class contract with two thresholds calibrated on a held-out split, e.g.:
   - `p < t_low` → `normal`
   - `t_low <= p < t_high` → `attention`
   - `p >= t_high` → `urgent`

This keeps every label the model is trained on honest, and documents `attention` as a
confidence-band heuristic rather than a supervised class — worth calling out explicitly in the
Step 4 write-up and in the "educational only" disclaimer already in the root README.
