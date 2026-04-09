"""Tests for thread key file parsing."""

from __future__ import annotations

import pytest

from nexis.miner.thread_keys import parse_thread_keys_file


def test_parse_thread_keys_file(tmp_path) -> None:
    p = tmp_path / "k.txt"
    p.write_text("a,b\nc, d\n\n", encoding="utf-8")
    rows = parse_thread_keys_file(p)
    assert rows == [["a", "b"], ["c", "d"]]


def test_parse_thread_keys_file_missing(tmp_path) -> None:
    with pytest.raises(ValueError, match="not found"):
        parse_thread_keys_file(tmp_path / "nope.txt")
