"""YouTube acquisition and clip extraction utilities for miners."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

YT_DLP_DOWNLOAD_TIMEOUT_SECONDS = 300
FFPROBE_TIMEOUT_SEC = 30
FFMPEG_TIMEOUT_SEC = 120
YTDLP_RETRIES = 2
logger = logging.getLogger(__name__)


def _env_str(name: str, default: str) -> str:
    value = os.getenv(name)
    return value if value is not None and value != "" else default


TARGET_RESOLUTION = _env_str("TARGET_RESOLUTION", "720")


def _build_yt_dlp_cmd(args: list[str]) -> list[str]:
    return ["yt-dlp", *args]


def _run_command(
    cmd: list[str],
    *,
    timeout_sec: int,
    capture_output: bool = False,
    text: bool = False,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        check=True,
        timeout=timeout_sec,
        capture_output=capture_output,
        text=text,
    )


def _run_subprocess(
    cmd: list[str],
    *,
    timeout: int,
) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        timeout=timeout,
        capture_output=True,
        text=True,
        check=False,
    )


def read_sources(path: Path) -> list[str]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [line for line in lines if line and not line.startswith("#")]


def download_youtube_video(url: str, output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    output_template = str(output_dir / "%(id)s.%(ext)s")
    cmd = _build_yt_dlp_cmd(
        [
            "--no-simulate",
            "-f",
            (
                "bestvideo[height<="
                f"{TARGET_RESOLUTION}]+bestaudio/"
                f"best[height<={TARGET_RESOLUTION}]/best"
            ),
            "--merge-output-format",
            "mp4",
            "--recode-video",
            "mp4",
            "-o",
            output_template,
            "--no-playlist",
            "--no-overwrites",
            "--print",
            "%(id)s",
            url,
        ]
    )
    last_error: Exception | None = None
    for attempt in range(1, YTDLP_RETRIES + 1):
        try:
            logger.info("yt-dlp download start url=%s attempt=%d/%d", url, attempt, YTDLP_RETRIES)
            result = _run_subprocess(
                cmd,
                timeout=YT_DLP_DOWNLOAD_TIMEOUT_SECONDS,
            )
            if result.returncode != 0:
                logger.warning(
                    "yt-dlp download failed attempt=%d/%d rc=%d err=%s",
                    attempt,
                    YTDLP_RETRIES,
                    result.returncode,
                    (result.stderr or "")[:200],
                )
                last_error = RuntimeError(
                    f"yt-dlp download failed: {(result.stderr or '')[:200]}"
                )
                continue

            video_id = ""
            for line in reversed(result.stdout.splitlines()):
                candidate = line.strip()
                if candidate:
                    video_id = candidate
                    break
            if video_id:
                preferred = output_dir / f"{video_id}.mp4"
                if preferred.exists():
                    logger.info("yt-dlp download complete url=%s path=%s", url, preferred)
                    return preferred

            mp4_candidates = sorted(
                output_dir.glob("*.mp4"),
                key=lambda p: p.stat().st_mtime,
                reverse=True,
            )
            if mp4_candidates:
                logger.info("yt-dlp download complete (fallback) url=%s path=%s", url, mp4_candidates[0])
                return mp4_candidates[0]
            last_error = RuntimeError("yt-dlp completed but output file was not found")
        except (subprocess.TimeoutExpired, OSError) as exc:
            logger.warning("yt-dlp download exception attempt=%d/%d: %s", attempt, YTDLP_RETRIES, exc)
            last_error = exc
    raise RuntimeError(f"failed to download source video for url={url}") from last_error


def probe_video(path: Path) -> dict:
    logger.debug("ffprobe start path=%s", path)
    cmd = [
        "ffprobe",
        "-v",
        "quiet",
        "-print_format",
        "json",
        "-show_streams",
        "-show_format",
        str(path),
    ]
    proc = _run_command(
        cmd,
        timeout_sec=FFPROBE_TIMEOUT_SEC,
        capture_output=True,
        text=True,
    )
    payload = json.loads(proc.stdout)
    logger.debug("ffprobe complete path=%s", path)
    return payload


def create_clip(src: Path, dst: Path, start_sec: float, duration_sec: float = 5.0) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-ss",
        f"{start_sec:.3f}",
        "-i",
        str(src),
        "-t",
        f"{duration_sec:.3f}",
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        str(dst),
    ]
    logger.debug("ffmpeg create clip src=%s dst=%s start=%.3f duration=%.3f", src, dst, start_sec, duration_sec)
    _run_command(cmd, timeout_sec=FFMPEG_TIMEOUT_SEC)


def extract_first_frame(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        "select=eq(n\\,0)",
        "-frames:v",
        "1",
        str(dst),
    ]
    logger.debug("ffmpeg extract first frame src=%s dst=%s", src, dst)
    _run_command(cmd, timeout_sec=FFMPEG_TIMEOUT_SEC)


def extract_caption_frames(src: Path, output_dir: Path, frame_count: int = 12) -> list[Path]:
    """Extract timeline-sampled frames for multimodal captioning."""
    if frame_count <= 0:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    output_pattern = output_dir / "caption_%03d.jpg"
    fps = max(frame_count / 5.0, 0.2)
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(src),
        "-vf",
        f"fps={fps:.6f}",
        "-frames:v",
        str(frame_count),
        str(output_pattern),
    ]
    try:
        logger.debug(
            "ffmpeg extract caption frames src=%s out_dir=%s frame_count=%d fps=%.6f",
            src,
            output_dir,
            frame_count,
            fps,
        )
        _run_command(cmd, timeout_sec=FFMPEG_TIMEOUT_SEC)
    except Exception:
        logger.warning("caption frame extraction failed src=%s", src)
        return []
    frames = sorted(output_dir.glob("caption_*.jpg"))
    logger.debug("caption frame extraction complete src=%s extracted=%d", src, len(frames))
    return frames[:frame_count]

