"""Tests for atomic URL queue pop."""

from __future__ import annotations

from nexis.miner.sources_queue import pop_next_source_url


def test_pop_next_source_url_empties_file(tmp_path) -> None:
    src = tmp_path / "urls.txt"
    src.write_text("https://a.example/a\n\n# skip\nhttps://b.example/b\n", encoding="utf-8")
    assert pop_next_source_url(src) == "https://a.example/a"
    assert pop_next_source_url(src) == "https://b.example/b"
    assert pop_next_source_url(src) is None
    assert src.read_text(encoding="utf-8") == "# skip\n"


def test_pop_next_source_url_skips_comments_and_blank(tmp_path) -> None:
    src = tmp_path / "u.txt"
    src.write_text("\n# c\n  \nhttps://x.test\n", encoding="utf-8")
    assert pop_next_source_url(src) == "https://x.test"
    assert pop_next_source_url(src) is None
