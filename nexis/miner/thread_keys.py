"""Parse per-thread OpenAI key lists for mine-prepare."""

from __future__ import annotations

from pathlib import Path


def parse_thread_keys_file(path: Path) -> list[list[str]]:
    """One line per worker: comma-separated API keys (stripped, empty segments skipped).

    Raises ValueError if the file is missing or has no non-empty key lines.
    """
    p = path.expanduser().resolve()
    if not p.is_file():
        raise ValueError(f"thread keys file not found: {p}")
    lines = p.read_text(encoding="utf-8").splitlines()
    result: list[list[str]] = []
    for line in lines:
        parts = [x.strip() for x in line.replace("\n", ",").split(",")]
        keys = [x for x in parts if x]
        if keys:
            result.append(keys)
    if not result:
        raise ValueError(f"no key lines in thread keys file: {p}")
    return result
