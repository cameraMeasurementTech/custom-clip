from __future__ import annotations

from pathlib import Path

from nexis.models import ClipRecord
from nexis.validator.category_check import (
    NatureCategoryChecker,
    caption_gate_check_nature,
    caption_gate_decision_from_caption,
    get_middle_three_frame_paths,
    parse_strict_pass,
)


def _row(*, caption: str) -> ClipRecord:
    return ClipRecord(
        clip_id="c1",
        clip_uri="clips/c1.mp4",
        clip_sha256="a" * 64,
        first_frame_uri="frames/c1.jpg",
        first_frame_sha256="b" * 64,
        source_video_id="vid",
        split_group_id="vid:1",
        split="train",
        clip_start_sec=0.0,
        duration_sec=5.0,
        width=1280,
        height=720,
        fps=30.0,
        num_frames=150,
        has_audio=True,
        caption=caption,
        source_video_url="https://youtube.com/watch?v=abc",
        source_proof={"extractor": "yt-dlp"},
    )


def test_caption_gate_decision_rejects_non_nature() -> None:
    result = caption_gate_check_nature("A man driving a car through city traffic.")
    assert caption_gate_decision_from_caption(result) == "reject"


def test_caption_gate_decision_borderline_nature() -> None:
    result = caption_gate_check_nature("A scenic mountain landscape with a road in view.")
    assert caption_gate_decision_from_caption(result) == "borderline"


def test_middle_three_frame_selection_handles_current_frame_shape(tmp_path: Path) -> None:
    frames = [tmp_path / f"f{i}.jpg" for i in range(6)]
    for frame in frames:
        frame.write_bytes(b"x")
    frame_paths = [frames[0], frames[1], frames[2], frames[3], frames[4], frames[5], frames[0]]
    middle = get_middle_three_frame_paths(frame_paths)
    assert middle == [frames[2], frames[3], frames[4]]


def test_parse_strict_pass_requires_three_frames() -> None:
    parsed = parse_strict_pass({"frames": [{"winner": "nature"}]})
    assert parsed is None


def test_category_checker_strict_unavailable_without_key(tmp_path: Path) -> None:
    frame_paths = [tmp_path / f"f{i}.jpg" for i in range(6)]
    for frame in frame_paths:
        frame.write_bytes(b"f")
    checker = NatureCategoryChecker(
        enabled=True,
        api_key="",
        timeout_sec=20,
        max_samples=8,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    failures = checker.check(
        sampled=[_row(caption="A scenic mountain landscape with a road in view.")],
        frame_paths_by_clip_id={"c1": frame_paths},
    )
    assert failures == ["category_strict_api_key_missing:c1"]
