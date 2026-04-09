from __future__ import annotations

from pathlib import Path

from nexis.miner.captioner import Captioner


def test_fallback_caption_avoids_url_like_text() -> None:
    captioner = Captioner(api_key="", model="gpt-4o-mini")
    caption = captioner.caption_clip(
        clip_path=Path("clip.mp4"),
        source_url="https://youtube.com/watch?v=abc",
        first_frame_path=None,
    )
    assert caption is not None
    assert "http://" not in caption.lower()
    assert "https://" not in caption.lower()
    assert len(caption.split()) >= 3

