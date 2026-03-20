"""Source provider abstraction for miner ingestion."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlparse

from .youtube import (
    create_clip,
    download_youtube_video,
    extract_caption_frames,
    extract_first_frame,
    probe_video,
    read_sources,
)


class SourceProvider(Protocol):
    def read_sources(self, path: Path) -> list[str]: ...

    def source_video_id(self, url: str) -> str: ...

    def download(self, url: str, output_dir: Path) -> Path: ...

    def probe(self, path: Path) -> dict[str, Any]: ...

    def create_clip(self, src: Path, dst: Path, start_sec: float, duration_sec: float) -> None: ...

    def extract_first_frame(self, src: Path, dst: Path) -> None: ...

    def extract_caption_frames(self, src: Path, output_dir: Path, frame_count: int) -> list[Path]: ...


class YouTubeSourceProvider:
    """Default source provider for video_v1."""

    def read_sources(self, path: Path) -> list[str]:
        return read_sources(path)

    def source_video_id(self, url: str) -> str:
        parsed = urlparse(url)
        if parsed.netloc.endswith("youtu.be"):
            return parsed.path.strip("/")
        query = parsed.query
        for part in query.split("&"):
            if part.startswith("v="):
                return part.split("=", 1)[1]
        return parsed.path.strip("/").replace("/", "_")

    def download(self, url: str, output_dir: Path) -> Path:
        return download_youtube_video(url, output_dir)

    def probe(self, path: Path) -> dict[str, Any]:
        return probe_video(path)

    def create_clip(self, src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
        create_clip(src, dst, start_sec, duration_sec)

    def extract_first_frame(self, src: Path, dst: Path) -> None:
        extract_first_frame(src, dst)

    def extract_caption_frames(self, src: Path, output_dir: Path, frame_count: int) -> list[Path]:
        return extract_caption_frames(src, output_dir, frame_count=frame_count)
