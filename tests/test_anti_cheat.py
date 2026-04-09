from __future__ import annotations

from nexis.models import ClipRecord
from nexis.validator.checks import run_hard_checks

from .helpers import VALID_TEST_CAPTION


def _row(clip_id: str, start: float, url: str, caption: str = VALID_TEST_CAPTION) -> ClipRecord:
    return ClipRecord(
        clip_id=clip_id,
        clip_uri=f"clips/{clip_id}.mp4",
        clip_sha256="a" * 64,
        first_frame_uri=f"frames/{clip_id}.jpg",
        first_frame_sha256="b" * 64,
        source_video_id="vid",
        split_group_id="vid:1",
        split="train",
        clip_start_sec=start,
        duration_sec=5.0,
        width=640,
        height=360,
        fps=30.0,
        num_frames=150,
        has_audio=True,
        caption=caption,
        source_video_url=url,
        source_proof={"extractor": "yt-dlp"},
    )


def test_overlap_detection() -> None:
    rows = [
        _row("c1", 0.0, "https://youtube.com/watch?v=abc"),
        _row("c2", 2.0, "https://youtube.com/watch?v=abc"),
    ]
    result = run_hard_checks(rows)
    assert any("overlap_lt_5s" in item for item in result.failures)


def test_overlap_detection_uses_source_video_id() -> None:
    rows = [
        _row("c1", 0.0, "https://youtube.com/watch?v=abc"),
        _row("c2", 2.0, "https://youtu.be/abc?t=2"),
    ]
    rows[1].source_video_id = rows[0].source_video_id
    result = run_hard_checks(rows)
    assert any("overlap_lt_5s" in item for item in result.failures)


def test_non_youtube_detection() -> None:
    rows = [_row("c1", 0.0, "https://example.com/video")]
    result = run_hard_checks(rows)
    assert any("non_youtube_source" in item for item in result.failures)


def test_malicious_youtube_like_domain_detection() -> None:
    rows = [_row("c1", 0.0, "https://youtube.com.evil.tld/watch?v=abc")]
    result = run_hard_checks(rows)
    assert any("non_youtube_source" in item for item in result.failures)


def test_short_caption_rejects_under_min_words() -> None:
    rows = [_row("c1", 0.0, "https://youtube.com/watch?v=abc", caption="one two three four five")]
    result = run_hard_checks(rows)
    assert any(item.startswith("short_caption:") for item in result.failures)


def test_short_caption_rejects_exactly_twenty_words() -> None:
    twenty = " ".join(f"w{i}" for i in range(20))
    rows = [_row("c1", 0.0, "https://youtube.com/watch?v=abc", caption=twenty)]
    result = run_hard_checks(rows)
    assert any(item.startswith("short_caption:") for item in result.failures)


def test_caption_at_min_words_passes_lexical_gate() -> None:
    from nexis.protocol import MIN_CAPTION_WORDS

    ok = " ".join(f"w{i}" for i in range(MIN_CAPTION_WORDS))
    rows = [_row("c1", 0.0, "https://youtube.com/watch?v=abc", caption=ok)]
    result = run_hard_checks(rows)
    assert not any(item.startswith("short_caption:") for item in result.failures)

