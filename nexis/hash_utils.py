"""Hashing and deterministic id helpers."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        while True:
            chunk = fp.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def deterministic_clip_id(source_video_id: str, clip_start_sec: float, duration_sec: float) -> str:
    payload = f"{source_video_id}:{clip_start_sec:.3f}:{duration_sec:.3f}"
    return sha256_text(payload)[:24]

