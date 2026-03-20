from __future__ import annotations

from pathlib import Path

from nexis.cli import _load_hotkeys_from_file, _parse_exclude_hotkeys


def test_load_hotkeys_from_file(tmp_path: Path) -> None:
    path = tmp_path / "blacklist.txt"
    path.write_text(
        "\n".join(
            [
                "",
                "# comment",
                "hotkey1",
                " hotkey2 ",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert _load_hotkeys_from_file(path) == {"hotkey1", "hotkey2"}


def test_parse_exclude_hotkeys() -> None:
    assert _parse_exclude_hotkeys("") == set()
    assert _parse_exclude_hotkeys("  ") == set()
    assert _parse_exclude_hotkeys("a,b, c ,,") == {"a", "b", "c"}

