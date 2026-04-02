"""Miner interval pipeline: collect clips, build dataset, upload package."""

from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Any

from ..hash_utils import deterministic_clip_id, sha256_file
from ..models import ClipRecord, IntervalManifest
from ..protocol import CLIP_DURATION_SEC, SCHEMA_VERSION
from ..serialization import write_dataset_parquet, write_manifest
from ..specs import DEFAULT_SPEC_ID
from .captioner import Captioner
from .providers import SourceProvider, YouTubeSourceProvider

logger = logging.getLogger(__name__)
CAPTION_FRAME_COUNT = 12


def _video_stream(info: dict[str, Any]) -> dict[str, Any]:
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            return stream
    raise ValueError("video stream not found")


def _audio_present(info: dict[str, Any]) -> bool:
    return any(stream.get("codec_type") == "audio" for stream in info.get("streams", []))


class MinerPipeline:
    def __init__(
        self,
        store: Any,
        captioner: Captioner,
        source_provider: SourceProvider | None = None,
        spec_id: str = DEFAULT_SPEC_ID,
        dataset_category: str = "nature_landscape_scenery",
    ):
        self.store = store
        self.captioner = captioner
        self.source_provider = source_provider or YouTubeSourceProvider()
        self.spec_id = spec_id
        self.dataset_category = dataset_category.strip()

    async def run_interval(
        self,
        *,
        sources_file: Path,
        netuid: int,
        miner_hotkey: str,
        interval_id: int,
        workdir: Path,
    ) -> tuple[Path, Path]:
        logger.info("miner pipeline start interval_id=%d hotkey=%s", interval_id, miner_hotkey)
        workdir.mkdir(parents=True, exist_ok=True)
        raw_dir = workdir / "raw"
        clips_dir = workdir / "clips"
        frames_dir = workdir / "frames"
        out_dir = workdir / "out" / str(interval_id)
        out_dir.mkdir(parents=True, exist_ok=True)

        records: list[ClipRecord] = []
        assets_to_upload: dict[str, Path] = {}
        for url in self.source_provider.read_sources(sources_file):
            source_id = self.source_provider.source_video_id(url)
            logger.info("processing source source_id=%s url=%s", source_id, url)
            raw_path = self.source_provider.download(url, raw_dir)
            probe = self.source_provider.probe(raw_path)
            format_info = probe.get("format", {})
            duration = float(format_info.get("duration", 0.0))
            total_segments = int(math.floor(duration / CLIP_DURATION_SEC))
            logger.debug(
                "source stats source_id=%s duration=%.3f total_segments=%d",
                source_id,
                duration,
                total_segments,
            )
            if total_segments <= 0:
                logger.warning("source has no 5s segments source_id=%s url=%s", source_id, url)
                continue
            stream = _video_stream(probe)
            width = int(stream.get("width", 1))
            height = int(stream.get("height", 1))
            r_frame_rate = stream.get("r_frame_rate", "0/1")
            numerator, denominator = r_frame_rate.split("/")
            fps = float(numerator) / max(float(denominator), 1.0)
            has_audio = _audio_present(probe)

            for idx in range(total_segments):
                start = idx * CLIP_DURATION_SEC
                clip_id = deterministic_clip_id(source_id, start, CLIP_DURATION_SEC)
                clip_path = clips_dir / f"{clip_id}.mp4"
                frame_path = frames_dir / f"{clip_id}.jpg"
                self.source_provider.create_clip(raw_path, clip_path, start, CLIP_DURATION_SEC)
                self.source_provider.extract_first_frame(clip_path, frame_path)
                caption_frames = self.source_provider.extract_caption_frames(
                    clip_path,
                    frames_dir / clip_id / "caption",
                    CAPTION_FRAME_COUNT,
                )
                if not caption_frames and frame_path.exists():
                    caption_frames = [frame_path]
                caption = self.captioner.caption_clip(
                    clip_path,
                    url,
                    frame_paths=caption_frames,
                )
                logger.debug(
                    "built clip record clip_id=%s source_id=%s start_sec=%.3f",
                    clip_id,
                    source_id,
                    start,
                )
                record = ClipRecord(
                    clip_id=clip_id,
                    clip_uri=f"clips/{clip_path.name}",
                    clip_sha256=sha256_file(clip_path),
                    first_frame_uri=f"frames/{frame_path.name}",
                    first_frame_sha256=sha256_file(frame_path),
                    source_video_id=source_id,
                    split_group_id=f"{source_id}:{interval_id}",
                    split="train",
                    clip_start_sec=start,
                    duration_sec=CLIP_DURATION_SEC,
                    width=width,
                    height=height,
                    fps=max(fps, 1.0),
                    num_frames=max(int(round(CLIP_DURATION_SEC * max(fps, 1.0))), 1),
                    has_audio=has_audio,
                    caption=caption,
                    source_video_url=url,
                    source_proof={
                        "extractor": "yt-dlp",
                        "source_video_id": source_id,
                    },
                )
                records.append(record)
                assets_to_upload[record.clip_uri] = clip_path
                assets_to_upload[record.first_frame_uri] = frame_path

        dataset_path = out_dir / "dataset.parquet"
        write_dataset_parquet(records, dataset_path)
        logger.info("dataset written interval=%d records=%d", interval_id, len(records))
        manifest = IntervalManifest(
            protocol_version="1.0.0",
            schema_version=SCHEMA_VERSION,
            spec_id=self.spec_id,
            dataset_type=self.spec_id,
            category=self.dataset_category or None,
            netuid=netuid,
            miner_hotkey=miner_hotkey,
            interval_id=interval_id,
            record_count=len(records),
            dataset_sha256=sha256_file(dataset_path),
        )
        manifest_path = out_dir / "manifest.json"
        write_manifest(manifest, manifest_path)

        # Bucket name is already miner hotkey; interval directory is enough.
        base_key = f"{interval_id}"
        await self.store.upload_file(f"{base_key}/dataset.parquet", dataset_path, use_write=True)
        await self.store.upload_file(f"{base_key}/manifest.json", manifest_path, use_write=True)
        for relative_uri, local_path in assets_to_upload.items():
            await self.store.upload_file(
                f"{base_key}/{relative_uri.lstrip('/')}",
                local_path,
                use_write=True,
            )
        logger.debug("uploaded assets interval=%d count=%d", interval_id, len(assets_to_upload))
        logger.info("uploaded interval package hotkey=%s interval=%d", miner_hotkey, interval_id)
        return dataset_path, manifest_path

