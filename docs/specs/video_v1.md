# Dataset Spec: `video_v1`

`video_v1` is the preserved v1 dataset spec for Nexisgen.

## Manifest Compatibility

- `spec_id`: `video_v1`
- `protocol_version`: `1.0.0`
- `schema_version`: `1.0.0`
- `dataset_type`: compatibility alias; must normalize to `video_v1`

## Row Model

Rows are validated as `ClipRecord` with required fields:

- clip/frame URIs and SHA256 digests
- source identity (`source_video_id`, `source_video_url`)
- fixed-duration clip metadata (`duration_sec ~= 5.0`)
- caption and source proof fields

## Hard Rules

- source URL must be YouTube (`youtube.com` / `youtu.be`)
- no same-source overlap under `4.5s`
- captions must satisfy lexical minimum checks
- sampled clip/frame assets must match declared SHA256
- optional source-auth and semantic checks can append failures

## Overlap Index Keys

Owner/validator indexing uses both:

- legacy keys (`source_video_id`, canonical URL, raw URL)
- namespaced keys (`video_v1:<key>`)

This keeps backward compatibility while enabling cross-spec separation.
