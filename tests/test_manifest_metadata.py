from __future__ import annotations

import json

from nexis.models import IntervalManifest


def test_manifest_defaults_to_video_v1_spec_metadata() -> None:
    manifest = IntervalManifest(
        netuid=1,
        miner_hotkey="miner1",
        interval_id=5,
        record_count=0,
        dataset_sha256="a" * 64,
    )
    assert manifest.spec_id == "video_v1"
    assert manifest.dataset_type == "video_v1"


def test_manifest_backfills_spec_id_from_legacy_dataset_type() -> None:
    payload = {
        "protocol_version": "1.0.0",
        "schema_version": "1.0.0",
        "dataset_type": "video_v1",
        "netuid": 1,
        "miner_hotkey": "miner1",
        "interval_id": 6,
        "record_count": 1,
        "dataset_sha256": "b" * 64,
    }
    manifest = IntervalManifest.model_validate(payload)
    assert manifest.spec_id == "video_v1"
    assert manifest.dataset_type == "video_v1"
    encoded = json.loads(manifest.model_dump_json())
    assert encoded["spec_id"] == "video_v1"
    assert encoded["dataset_type"] == "video_v1"
