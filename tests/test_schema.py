from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pyarrow")

from nexis.models import ClipRecord, IntervalManifest
from nexis.serialization import read_dataset_parquet, read_manifest, write_dataset_parquet, write_manifest


def test_schema_roundtrip(tmp_path: Path) -> None:
    row = ClipRecord(
        clip_id="abc123",
        clip_uri="clips/abc123.mp4",
        clip_sha256="a" * 64,
        first_frame_uri="frames/abc123.jpg",
        first_frame_sha256="b" * 64,
        source_video_id="ytid",
        split_group_id="ytid:1",
        split="train",
        clip_start_sec=0.0,
        duration_sec=5.0,
        width=1280,
        height=720,
        fps=30.0,
        num_frames=150,
        has_audio=True,
        caption="A person walking across a city street.",
        source_video_url="https://youtube.com/watch?v=123",
        source_proof={"extractor": "yt-dlp"},
    )
    dataset_path = tmp_path / "dataset.parquet"
    write_dataset_parquet([row], dataset_path)
    loaded = read_dataset_parquet(dataset_path)
    assert len(loaded) == 1
    assert loaded[0].clip_id == "abc123"

    manifest = IntervalManifest(
        netuid=1,
        miner_hotkey="hotkey",
        interval_id=10,
        record_count=1,
        dataset_sha256="c" * 64,
    )
    manifest_path = tmp_path / "manifest.json"
    write_manifest(manifest, manifest_path)
    loaded_manifest = read_manifest(manifest_path)
    assert loaded_manifest.interval_id == 10

