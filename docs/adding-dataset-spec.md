# Adding a New Dataset Spec

Use this checklist when integrating a new company-specific schema/validation policy.

## 1) Register the Spec

- Implement a spec adapter under `nexis/specs/`.
- Provide:
  - `spec_id`
  - supported protocol/schema versions
  - row model
  - hard-check implementation
  - asset verifier (or explicit `None`)
  - overlap key strategy
- Register it in `DatasetSpecRegistry.with_defaults()`.

## 2) Compatibility Contract

- Document accepted `protocol_version` and `schema_version`.
- Add validator compatibility tests for:
  - accepted version
  - incompatible version rejection
  - unknown spec rejection

## 3) Ingestion Path

- Add/extend source provider adapters if needed.
- Ensure miner can emit rows for the new row model.
- Ensure manifest writes `spec_id` and `dataset_type`.

## 4) Validation Path

- Wire hard checks through the spec adapter.
- Add/extend sampled asset verification for spec assets.
- Verify cross-spec isolation through overlap/index key namespace.

## 5) Tests (required)

- miner pipeline package/output tests
- validator load/compatibility tests
- security tests (no cross-spec bypass)
- owner sync + overlap index round-trip tests

## 6) Docs + Rollout

- Add `docs/specs/<spec_id>.md`.
- Update `docs/spec.md` compatibility matrix.
- Follow `docs/multi-spec-migration-runbook.md` for staged rollout.
