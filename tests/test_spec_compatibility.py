from __future__ import annotations

import json
from pathlib import Path

from nexis.hash_utils import sha256_file
from nexis.models import ClipRecord, IntervalManifest
from nexis.serialization import write_dataset_parquet, write_manifest
from nexis.validator.pipeline import ValidatorPipeline
from .helpers import LocalObjectStore, run_async


def _record(clip_id: str, start: float, url: str = "https://youtube.com/watch?v=abc") -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        clip_uri=f"clips/{clip_id}.mp4",
        clip_sha256="a" * 64,
        first_frame_uri=f"frames/{clip_id}.jpg",
        first_frame_sha256="b" * 64,
        source_video_id="ytid",
        split_group_id="ytid:1",
        split="train",
        clip_start_sec=start,
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


async def _upload_valid_submission(
    *,
    store: LocalObjectStore,
    tmp_path: Path,
    hotkey: str,
    interval_id: int,
    manifest_payload: dict | None = None,
) -> None:
    row = _record("c1", 0.0)
    clip = tmp_path / "clip.mp4"
    frame = tmp_path / "frame.jpg"
    clip.write_bytes(b"clip")
    frame.write_bytes(b"frame")
    row.clip_sha256 = sha256_file(clip)
    row.first_frame_sha256 = sha256_file(frame)
    dataset = tmp_path / "dataset.parquet"
    manifest = tmp_path / "manifest.json"
    write_dataset_parquet([row], dataset)
    if manifest_payload is None:
        write_manifest(
            IntervalManifest(
                netuid=1,
                miner_hotkey=hotkey,
                interval_id=interval_id,
                record_count=1,
                dataset_sha256=sha256_file(dataset),
            ),
            manifest,
        )
    else:
        manifest.write_text(json.dumps(manifest_payload), encoding="utf-8")

    key_base = f"{interval_id}"
    await store.upload_file(f"{key_base}/dataset.parquet", dataset)
    await store.upload_file(f"{key_base}/manifest.json", manifest)
    await store.upload_file(f"{key_base}/{row.clip_uri}", clip)
    await store.upload_file(f"{key_base}/{row.first_frame_uri}", frame)


def test_validator_accepts_legacy_manifest_without_spec_id(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        hotkey = "miner1"
        interval_id = 11

        dataset = tmp_path / "dataset.parquet"
        manifest = tmp_path / "manifest.json"
        row = _record("c1", 0.0)
        clip = tmp_path / "clip.mp4"
        frame = tmp_path / "frame.jpg"
        clip.write_bytes(b"clip")
        frame.write_bytes(b"frame")
        row.clip_sha256 = sha256_file(clip)
        row.first_frame_sha256 = sha256_file(frame)
        write_dataset_parquet([row], dataset)
        write_manifest(
            IntervalManifest(
                netuid=1,
                miner_hotkey=hotkey,
                interval_id=interval_id,
                record_count=1,
                dataset_sha256=sha256_file(dataset),
            ),
            manifest,
        )
        payload = json.loads(manifest.read_text(encoding="utf-8"))
        payload.pop("spec_id", None)
        payload.pop("dataset_type", None)
        await _upload_valid_submission(
            store=store,
            tmp_path=tmp_path,
            hotkey=hotkey,
            interval_id=interval_id,
            manifest_payload=payload,
        )

        pipeline = ValidatorPipeline(store_for_hotkey=lambda _: store)
        decisions, _ = await pipeline.validate_interval(candidate_hotkeys=[hotkey], interval_id=interval_id)
        assert decisions[0].accepted is True

    run_async(run())


def test_validator_rejects_unknown_spec_id(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        hotkey = "miner1"
        interval_id = 12
        await _upload_valid_submission(
            store=store,
            tmp_path=tmp_path,
            hotkey=hotkey,
            interval_id=interval_id,
            manifest_payload={
                "protocol_version": "1.0.0",
                "schema_version": "1.0.0",
                "spec_id": "unknown_spec",
                "dataset_type": "unknown_spec",
                "netuid": 1,
                "miner_hotkey": hotkey,
                "interval_id": interval_id,
                "record_count": 1,
                    "dataset_sha256": "a" * 64,
            },
        )
        pipeline = ValidatorPipeline(store_for_hotkey=lambda _: store)
        decisions, _ = await pipeline.validate_interval(candidate_hotkeys=[hotkey], interval_id=interval_id)
        assert decisions[0].accepted is False
        assert "unknown_spec:unknown_spec" in decisions[0].failures

    run_async(run())


def test_validator_rejects_incompatible_schema_version(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        hotkey = "miner1"
        interval_id = 13
        await _upload_valid_submission(
            store=store,
            tmp_path=tmp_path,
            hotkey=hotkey,
            interval_id=interval_id,
            manifest_payload={
                "protocol_version": "1.0.0",
                "schema_version": "9.9.9",
                "spec_id": "video_v1",
                "dataset_type": "video_v1",
                "netuid": 1,
                "miner_hotkey": hotkey,
                "interval_id": interval_id,
                "record_count": 1,
                    "dataset_sha256": "a" * 64,
            },
        )
        pipeline = ValidatorPipeline(store_for_hotkey=lambda _: store)
        decisions, _ = await pipeline.validate_interval(candidate_hotkeys=[hotkey], interval_id=interval_id)
        assert decisions[0].accepted is False
        assert any(item.startswith("incompatible_spec_version:video_v1") for item in decisions[0].failures)

    run_async(run())


def test_validator_rejects_spec_when_not_enabled(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        hotkey = "miner1"
        interval_id = 14
        await _upload_valid_submission(
            store=store,
            tmp_path=tmp_path,
            hotkey=hotkey,
            interval_id=interval_id,
        )
        pipeline = ValidatorPipeline(store_for_hotkey=lambda _: store, enabled_specs=["some_other_spec"])
        decisions, _ = await pipeline.validate_interval(candidate_hotkeys=[hotkey], interval_id=interval_id)
        assert decisions[0].accepted is False
        assert "spec_not_enabled:video_v1" in decisions[0].failures

    run_async(run())
