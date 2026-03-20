# Dataset Schema

Nexisgen miner datasets use Parquet rows with the following columns:

```text
clip_id STRING
clip_uri STRING
clip_sha256 STRING
first_frame_uri STRING
first_frame_sha256 STRING
source_video_id STRING
split_group_id STRING
split STRING
clip_start_sec FLOAT
duration_sec FLOAT
width INT
height INT
fps FLOAT
num_frames INT
has_audio BOOLEAN
caption STRING
source_video_url STRING
source_proof JSON
```

## Extra Protocol Constraints

- `clip_id` must be deterministic from:
  - `source_video_id`
  - `clip_start_sec`
  - `duration_sec`
- `duration_sec` must be close to `5.0` for v1.
- `source_video_url` must be YouTube.
- Validators apply hard checks over full record sets per interval.
- Overlap rows may be pruned (row-level rejection) against:
  - global shared record index (`nexis-record-info`)
  - same-interval cross-miner conflicts (earliest manifest `created_at` wins)

## Manifest Fields

`manifest.json` includes:

- `protocol_version`
- `schema_version`
- `spec_id` (default: `video_v1`; legacy manifests without this field are treated as `video_v1`)
- `dataset_type` (compatibility alias; normalized to `spec_id`)
- `netuid`
- `miner_hotkey`
- `interval_id`
- `created_at`
- `record_count`
- `dataset_sha256`
- optional `source` metadata

