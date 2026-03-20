# Multi-Spec Migration Runbook

This runbook describes a safe rollout from video-only payloads to multi-spec payloads.

## Phase 0: Baseline Lock

- Ensure golden tests for current `video_v1` behavior are green.
- Record current validator acceptance/rejection distributions.

## Phase 1: Dual-Read Introduction

- Deploy validator code that can read:
  - legacy manifests (no `spec_id`)
  - new manifests (`spec_id` + `dataset_type`)
- Keep miner output unchanged (`video_v1` only).
- Configure validators with `NEXIS_VALIDATOR_ENABLED_SPECS=video_v1`.

## Phase 2: Spec Metadata Write

- Enable miner write path to include `spec_id=video_v1` and `dataset_type=video_v1`.
- Continue validating only `video_v1`.
- Monitor rejection reasons:
  - `unknown_spec:*`
  - `incompatible_spec_version:*`
  - `spec_not_enabled:*`

## Phase 3: New Spec Pilot

- Add new spec adapter and tests.
- Enable a small validator subset with:
  - `NEXIS_VALIDATOR_ENABLED_SPECS=video_v1,<new_spec>`
- Keep most validators on `video_v1` while collecting metrics.

## Phase 4: Broad Enablement

- Roll out validator allow-list update to all validators.
- Enable miner CLI selection for the new spec where intended.
- Keep backward-compatible parsing enabled.

## Rollback

- Immediately restrict validators to `video_v1`:
  - `NEXIS_VALIDATOR_ENABLED_SPECS=video_v1`
- Revert miners to `nexis mine --spec video_v1`.
- Keep dual-read parser enabled to avoid orphaning legacy submissions.

## Validator Upgrade Order

1. Owner validator
2. Non-owner validators (staged groups)
3. Miners
