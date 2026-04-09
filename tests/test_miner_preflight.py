from __future__ import annotations

from nexis.models import ClipRecord
from nexis.miner.pending_pack import (
    PENDING_RECORDS_JSONL,
    PendingSaveValidatorConfig,
    merge_validated_pending_record,
)
from nexis.miner.preflight import prune_until_hard_checks_pass
from nexis.specs import DatasetSpecRegistry

from .helpers import VALID_TEST_CAPTION


def _row(
    clip_id: str,
    start: float,
    *,
    caption: str = VALID_TEST_CAPTION,
    url: str = "https://youtube.com/watch?v=abc",
    source_id: str = "abc",
) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        clip_uri=f"clips/{clip_id}.mp4",
        clip_sha256="a" * 64,
        first_frame_uri=f"frames/{clip_id}.jpg",
        first_frame_sha256="b" * 64,
        source_video_id=source_id,
        split_group_id=f"{source_id}:1",
        split="train",
        clip_start_sec=start,
        duration_sec=5.0,
        width=1280,
        height=720,
        fps=30.0,
        num_frames=150,
        has_audio=True,
        caption=caption,
        source_video_url=url,
        source_proof={"extractor": "yt-dlp"},
    )


def test_prune_drops_short_caption() -> None:
    spec = DatasetSpecRegistry.with_defaults().get("video_v1")
    bad = _row("c1", 0.0, caption="hi")
    good = _row("c2", 10.0)
    out = prune_until_hard_checks_pass([bad, good], spec)
    assert len(out) == 1
    assert out[0].clip_id == "c2"


def test_prune_drops_later_overlapping_clip() -> None:
    spec = DatasetSpecRegistry.with_defaults().get("video_v1")
    a = _row("c1", 0.0)
    b = _row("c2", 2.0)
    out = prune_until_hard_checks_pass([a, b], spec)
    assert len(out) == 1
    assert out[0].clip_id == "c1"


def test_prune_drops_non_youtube() -> None:
    spec = DatasetSpecRegistry.with_defaults().get("video_v1")
    bad = _row("c1", 0.0, url="https://example.com/v")
    good = _row("c2", 10.0, source_id="def", url="https://youtube.com/watch?v=def")
    out = prune_until_hard_checks_pass([bad, good], spec)
    assert len(out) == 1
    assert out[0].clip_id == "c2"


def _make_cfg(*, run_local_asset_verify: bool = False) -> PendingSaveValidatorConfig:
    return PendingSaveValidatorConfig(
        enabled=True,
        spec_id="video_v1",
        registry=DatasetSpecRegistry.with_defaults(),
        miner_hotkey="test",
        prepare_sample_interval_id=0,
        run_local_asset_verify=run_local_asset_verify,
        caption_semantic_checker=None,
        category_checker=None,
    )


def test_merge_rejects_short_caption_without_asset_verify(tmp_path) -> None:
    wd = tmp_path
    bad = _row("c1", 0.0, caption="no")
    ok, removed = merge_validated_pending_record(wd, bad, _make_cfg())
    assert ok is False
    assert len(removed) == 1
    assert removed[0].clip_id == "c1"
    assert not (wd / PENDING_RECORDS_JSONL).is_file()


def test_merge_writes_valid_row_without_asset_verify(tmp_path) -> None:
    wd = tmp_path
    good = _row("c1", 0.0)
    ok, removed = merge_validated_pending_record(wd, good, _make_cfg())
    assert ok is True
    assert removed == []
    text = (wd / PENDING_RECORDS_JSONL).read_text(encoding="utf-8").strip()
    assert "c1" in text


def test_merge_rejects_new_overlap_against_existing(tmp_path) -> None:
    wd = tmp_path
    existing = _row("c1", 0.0, source_id="src", url="https://youtube.com/watch?v=src")
    (wd / PENDING_RECORDS_JSONL).write_text(existing.model_dump_json() + "\n", encoding="utf-8")
    new = _row("c2", 2.0, source_id="src", url="https://youtube.com/watch?v=src")
    ok, removed = merge_validated_pending_record(wd, new, _make_cfg())
    assert ok is False
    assert removed[0].clip_id == "c2"
