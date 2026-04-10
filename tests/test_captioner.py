from __future__ import annotations

from pathlib import Path

from nexis.miner.captioner import CaptionClipResult, Captioner


def test_missing_api_key_skips_without_fallback_caption() -> None:
    captioner = Captioner(api_key="", model="gpt-4o-mini")
    result = captioner.caption_clip(
        clip_path=Path("clip.mp4"),
        source_url="https://youtube.com/watch?v=abc",
        first_frame_path=None,
    )
    assert isinstance(result, CaptionClipResult)
    assert result.caption is None
    assert result.category_proof is not None
    assert result.category_proof.get("error") == "missing_api_key"

