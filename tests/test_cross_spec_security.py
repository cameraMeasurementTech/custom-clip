from __future__ import annotations

import json
from pathlib import Path

from nexis.hash_utils import sha256_file
from nexis.models import ClipRecord, IntervalManifest
from nexis.serialization import write_dataset_parquet, write_manifest
from nexis.validator.pipeline import ValidatorPipeline
from .helpers import LocalObjectStore, run_async


def _row(url: str) -> ClipRecord:
    return ClipRecord(
        clip_id="c1",
        clip_uri="clips/c1.mp4",
        clip_sha256="a" * 64,
        first_frame_uri="frames/c1.jpg",
        first_frame_sha256="b" * 64,
        source_video_id="ytid",
        split_group_id="ytid:1",
        split="train",
        clip_start_sec=0.0,
        duration_sec=5.0,
        width=640,
        height=360,
        fps=30.0,
        num_frames=150,
        has_audio=True,
        caption="A moving object in a short scene.",
        source_video_url=url,
        source_proof={"extractor": "yt-dlp"},
    )


def test_unknown_spec_cannot_bypass_video_rules(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        hotkey = "miner1"
        interval_id = 21
        row = _row("https://example.com/not-youtube")
        clip = tmp_path / "clip.mp4"
        frame = tmp_path / "frame.jpg"
        clip.write_bytes(b"clip")
        frame.write_bytes(b"frame")
        row.clip_sha256 = sha256_file(clip)
        row.first_frame_sha256 = sha256_file(frame)
        dataset = tmp_path / "dataset.parquet"
        write_dataset_parquet([row], dataset)
        manifest = IntervalManifest(
            netuid=1,
            miner_hotkey=hotkey,
            interval_id=interval_id,
            record_count=1,
            dataset_sha256=sha256_file(dataset),
        )
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest, manifest_path)
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["spec_id"] = "unknown_spec"
        payload["dataset_type"] = "video_v1"
        manifest_path.write_text(json.dumps(payload), encoding="utf-8")

        key_base = f"{interval_id}"
        await store.upload_file(f"{key_base}/dataset.parquet", dataset)
        await store.upload_file(f"{key_base}/manifest.json", manifest_path)
        await store.upload_file(f"{key_base}/{row.clip_uri}", clip)
        await store.upload_file(f"{key_base}/{row.first_frame_uri}", frame)

        pipeline = ValidatorPipeline(store_for_hotkey=lambda _: store)
        decisions, _ = await pipeline.validate_interval(candidate_hotkeys=[hotkey], interval_id=interval_id)
        assert decisions[0].accepted is False
        assert "unknown_spec:unknown_spec" in decisions[0].failures
        assert not any(item.startswith("non_youtube_source:") for item in decisions[0].failures)

    run_async(run())


def test_video_spec_id_wins_over_dataset_type_override(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        hotkey = "miner1"
        interval_id = 22
        row = _row("https://example.com/not-youtube")
        clip = tmp_path / "clip.mp4"
        frame = tmp_path / "frame.jpg"
        clip.write_bytes(b"clip")
        frame.write_bytes(b"frame")
        row.clip_sha256 = sha256_file(clip)
        row.first_frame_sha256 = sha256_file(frame)
        dataset = tmp_path / "dataset.parquet"
        write_dataset_parquet([row], dataset)
        manifest = IntervalManifest(
            netuid=1,
            miner_hotkey=hotkey,
            interval_id=interval_id,
            record_count=1,
            dataset_sha256=sha256_file(dataset),
            spec_id="video_v1",
            dataset_type="unknown_spec",
        )
        manifest_path = tmp_path / "manifest.json"
        write_manifest(manifest, manifest_path)

        key_base = f"{interval_id}"
        await store.upload_file(f"{key_base}/dataset.parquet", dataset)
        await store.upload_file(f"{key_base}/manifest.json", manifest_path)
        await store.upload_file(f"{key_base}/{row.clip_uri}", clip)
        await store.upload_file(f"{key_base}/{row.first_frame_uri}", frame)

        pipeline = ValidatorPipeline(store_for_hotkey=lambda _: store)
        decisions, _ = await pipeline.validate_interval(candidate_hotkeys=[hotkey], interval_id=interval_id)
        assert decisions[0].accepted is False
        assert any(item.startswith("non_youtube_source:") for item in decisions[0].failures)

    run_async(run())
