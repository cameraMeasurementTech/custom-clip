# Nexisgen Protocol Spec (v1.0)

This document freezes the v1 protocol contract used by miner and validator implementations.

## Interval Model

- Interval mode: `blocks`
- Interval length: `50` blocks
- Interval ID: interval start block (example: `7756900` for range `7756900-7756950`)
- Upload reserve/deadline buffer: `2` blocks before interval close
- Validator cadence: evaluate every closed 50-block interval
- Weight cadence: submit `set_weights` every `250` blocks

## Submission Package

Each miner submission must include:

1. `dataset.parquet` containing clip rows
2. `manifest.json` containing interval metadata and dataset digest
3. Referenced clip/frame assets under interval prefix (`clips/...`, `frames/...`)

Canonical object prefix:

`{interval_id}/`

## Dataset Spec Compatibility

- Manifest should include `spec_id` (legacy manifests without it are treated as `video_v1`).
- `dataset_type` is retained as a compatibility alias and is normalized to `spec_id`.
- Validators reject:
  - unknown spec IDs
  - incompatible `protocol_version` / `schema_version` for the declared spec
  - specs outside validator allow-list (`NEXIS_VALIDATOR_ENABLED_SPECS`)

## Required Clip Schema

Row schema is defined in `nexis/models.py` as `ClipRecord`.

Key requirements:

- Clip duration is `5s` (tolerance allowed for codec boundaries)
- `clip_sha256` and `first_frame_sha256` are mandatory
- `source_video_url` must point to YouTube (`youtube.com` or `youtu.be`)

## Validation Policies

### Hard Failures (Reject Interval Immediately)

- Non-YouTube source URL
- Invalid schema or missing required fields
- Digest mismatch for clip/frame assets
- Overlapping clip windows from same source identity (`source_video_id`) where clip starts differ by `< 4.5s`
- Caption mismatch above threshold for sampled rows
- Sampled clip/frame asset hash mismatch against row SHA256 fields
- Source authenticity mismatch (up to 3 sampled rows per miner)

## Sampling Policy

- Miner sampling per interval:
  - `sample_rate=0.25`
  - `sample_min=2`
  - `sample_max=35`
- Per-miner row sampling:
  - If row count `<= 10`, sample all rows
  - Else sample `20%`, capped at `10` rows
  - **Hard checks run on full records** (row sampling is for expensive checks such as asset verification and semantic/source checks)

## Cross-Validator Record Intelligence

- Owner validator hotkey: `5EUdjwHz9pW4ftQQQga9PKq7knGiGW9wcHUjkDSih7zpovPy`
- Owner publishes accepted miner interval bundles to shared bucket `nexis-dataset`
- Owner publishes/updates global overlap index JSON in shared bucket `nexis-record-info`
  - Current payload shape:
    - `{"<source_video_url>": ["0.000", "5.000", ...]}`
  - Validators still accept legacy wrapped payloads and normalize legacy keys to canonical source URLs
- All validators load this index each interval and prune overlapping rows only (row-level rejection, not full-interval rejection)
- Additional same-interval cross-miner arbitration:
  - if rows overlap across miners, earliest `manifest.created_at` wins
  - tie-breaker: lexicographic hotkey

## Scoring

For accepted intervals only:

`score = pow(sample_numbers, 3)`

Where `sample_numbers` is the number of sampled rows that passed hard checks.

## Failure Memory

- Validators keep a rolling failure lookback of `10` intervals.
- Recent hard-failure miners are eligible for weight dampening/zeroing.

