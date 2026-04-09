"""Miner interval pipeline: collect clips, build dataset, upload package."""

from __future__ import annotations

import logging
import math
import shutil
from pathlib import Path
from typing import Any

from ..hash_utils import deterministic_clip_id, sha256_file
from ..models import ClipRecord, IntervalManifest
from ..protocol import CLIP_DURATION_SEC, PREP_MAX_CONSECUTIVE_REJECTS_PER_URL, SCHEMA_VERSION
from ..serialization import write_dataset_parquet, write_manifest
from ..specs import DEFAULT_SPEC_ID
from .captioner import Captioner
from .pending_pack import (
    PendingSaveValidatorConfig,
    append_pending_record,
    append_pending_record_locked,
    load_segment_cursor,
    load_worker_segment_cursor,
    merge_validated_pending_record,
    save_segment_cursor,
    save_worker_segment_cursor,
)
from .providers import SourceProvider, YouTubeSourceProvider
from .sources_queue import pop_next_source_url
from .youtube import probe_video

logger = logging.getLogger(__name__)
CAPTION_FRAME_COUNT = 12
_VALIDATOR_CLIP_WIDTH = 1280
_VALIDATOR_CLIP_HEIGHT = 720


def _try_unlink_raw(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
            logger.info("deleted raw video path=%s", path)
    except OSError as exc:
        logger.warning("failed to delete raw path=%s err=%s", path, exc)


def _try_unlink_file(path: Path) -> None:
    try:
        if path.is_file():
            path.unlink()
            logger.info("deleted file %s", path)
    except OSError as exc:
        logger.warning("failed to delete file %s err=%s", path, exc)


def _rmtree_if_dir(path: Path) -> None:
    if not path.is_dir():
        return
    try:
        shutil.rmtree(path, ignore_errors=False)
        logger.info("deleted directory %s", path)
    except OSError as exc:
        logger.warning("failed rmtree %s err=%s", path, exc)


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
        store: Any | None,
        captioner: Captioner,
        source_provider: SourceProvider | None = None,
        spec_id: str = DEFAULT_SPEC_ID,
        dataset_category: str = "nature_landscape_scenery",
        pending_save: PendingSaveValidatorConfig | None = None,
    ):
        self.store = store  # None allowed for prepare-only (mine_one_segment dequeue mode)
        self.captioner = captioner
        self.source_provider = source_provider or YouTubeSourceProvider()
        self.spec_id = spec_id
        self.dataset_category = dataset_category.strip()
        self._pending_save = pending_save

    def _persist_pending_record(self, workdir: Path, record: ClipRecord, *, locked: bool) -> bool:
        """Write pending row. Returns False if validator prepare gate rejected the new clip."""
        if self._pending_save is not None and self._pending_save.enabled:
            accepted, _removed = merge_validated_pending_record(workdir, record, self._pending_save)
            return accepted
        if locked:
            append_pending_record_locked(workdir, record)
        else:
            append_pending_record(workdir, record)
        return True

    def _clip_output_dimensions(self, clip_path: Path) -> tuple[int, int] | None:
        try:
            info = probe_video(clip_path)
            for stream in info.get("streams", []):
                if str(stream.get("codec_type", "")).lower() == "video":
                    return int(stream.get("width", 0)), int(stream.get("height", 0))
        except Exception as exc:
            logger.warning("probe output clip failed path=%s err=%s", clip_path, exc)
        return None

    def _scrub_rejected_segment_files(
        self,
        clip_path: Path,
        frame_path: Path,
        frames_dir: Path,
        clip_id: str,
    ) -> None:
        _try_unlink_file(clip_path)
        _try_unlink_file(frame_path)
        _rmtree_if_dir(frames_dir / clip_id)

    def _early_reject_bad_output_resolution(
        self,
        clip_path: Path,
        frame_path: Path,
        frames_dir: Path,
        clip_id: str,
        *,
        source_url: str,
    ) -> bool:
        """If prepare validation requires 1280×720 assets, drop segment files before caption when output mismatches.

        Returns True when the encoded clip is rejected; caller should **skip the entire source URL** (not only
        advance to the next 5s segment), since resolution is typically consistent for the whole download.
        """
        if not (
            self._pending_save is not None
            and self._pending_save.enabled
            and self._pending_save.run_local_asset_verify
        ):
            return False
        dims = self._clip_output_dimensions(clip_path)
        if dims is None:
            logger.warning(
                "prepare validation REJECT (probe failed) clip_id=%s url=%s — skipping entire source",
                clip_id,
                source_url,
            )
            self._scrub_rejected_segment_files(clip_path, frame_path, frames_dir, clip_id)
            return True
        w, h = dims
        if w == _VALIDATOR_CLIP_WIDTH and h == _VALIDATOR_CLIP_HEIGHT:
            return False
        logger.warning(
            "prepare validation REJECT (resolution) clip_id=%s url=%s got %dx%d require %dx%d — skipping entire source",
            clip_id,
            source_url,
            w,
            h,
            _VALIDATOR_CLIP_WIDTH,
            _VALIDATOR_CLIP_HEIGHT,
        )
        self._scrub_rejected_segment_files(clip_path, frame_path, frames_dir, clip_id)
        return True

    def mine_one_segment(
        self,
        workdir: Path,
        sources_file: Path,
        *,
        worker_id: str | None = None,
        dequeue_sources: bool = False,
    ) -> bool:
        """Build one 5s clip + caption and append to miner_pending_records.jsonl.

        Legacy: full sources file + global cursor (``nexis mine``).

        Queue mode: ``dequeue_sources=True`` and ``worker_id`` set — atomically pop next URL from
        ``sources_file``, per-worker cursor under ``cursors/``, raw under ``raw/{worker_id}/``.
        """
        if dequeue_sources:
            if not worker_id:
                raise ValueError("worker_id is required when dequeue_sources=True")
            return self._mine_one_segment_dequeue(workdir, sources_file, worker_id)

        workdir = workdir.expanduser().resolve()
        workdir.mkdir(parents=True, exist_ok=True)
        all_urls = list(self.source_provider.read_sources(sources_file))
        if not all_urls:
            return False

        n = len(all_urls)
        raw_dir = workdir / "raw"
        clips_dir = workdir / "clips"
        frames_dir = workdir / "frames"

        cur = load_segment_cursor(workdir)
        url_index = int(cur["url_index"]) % n
        seg_index = int(cur["segment_index"])
        active_raw: str | None = cur.get("active_raw")

        url = all_urls[url_index]
        source_id = self.source_provider.source_video_id(url)

        if seg_index == 0:
            raw_path = self.source_provider.download(url, raw_dir)
            active_raw = str(raw_path.relative_to(workdir))
        else:
            if not active_raw:
                save_segment_cursor(workdir, url_index, 0, active_raw=None, prep_reject_streak=0)
                return self.mine_one_segment(workdir, sources_file)
            raw_path = workdir / active_raw
            if not raw_path.is_file():
                save_segment_cursor(workdir, url_index, 0, active_raw=None, prep_reject_streak=0)
                return self.mine_one_segment(workdir, sources_file)

        probe = self.source_provider.probe(raw_path)
        format_info = probe.get("format", {})
        duration = float(format_info.get("duration", 0.0))
        total_segments = int(math.floor(duration / CLIP_DURATION_SEC))

        if total_segments <= 0:
            logger.warning("source has no 5s segments source_id=%s url=%s", source_id, url)
            _try_unlink_raw(raw_path)
            save_segment_cursor(
                workdir, (url_index + 1) % n, 0, active_raw=None, prep_reject_streak=0
            )
            return True

        if seg_index >= total_segments:
            _try_unlink_raw(raw_path)
            save_segment_cursor(
                workdir, (url_index + 1) % n, 0, active_raw=None, prep_reject_streak=0
            )
            return True

        stream = _video_stream(probe)
        width = int(stream.get("width", 1))
        height = int(stream.get("height", 1))
        r_frame_rate = stream.get("r_frame_rate", "0/1")
        numerator, denominator = r_frame_rate.split("/")
        fps = float(numerator) / max(float(denominator), 1.0)
        has_audio = _audio_present(probe)

        start = float(seg_index) * CLIP_DURATION_SEC
        clip_id = deterministic_clip_id(source_id, start, CLIP_DURATION_SEC)
        clip_path = clips_dir / f"{clip_id}.mp4"
        frame_path = frames_dir / f"{clip_id}.jpg"
        self.source_provider.create_clip(raw_path, clip_path, start, CLIP_DURATION_SEC)
        self.source_provider.extract_first_frame(clip_path, frame_path)
        if self._early_reject_bad_output_resolution(
            clip_path, frame_path, frames_dir, clip_id, source_url=url
        ):
            _try_unlink_raw(raw_path)
            save_segment_cursor(
                workdir, (url_index + 1) % n, 0, active_raw=None, prep_reject_streak=0
            )
            return True

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
        if caption is None:
            logger.info(
                "prepare skip segment reason=caption_too_short clip_id=%s url=%s",
                clip_id,
                url,
            )
            self._scrub_rejected_segment_files(clip_path, frame_path, frames_dir, clip_id)
            reject_streak = max(0, int(cur.get("prep_reject_streak", 0)))
            next_seg = seg_index + 1
            if next_seg >= total_segments:
                _try_unlink_raw(raw_path)
                save_segment_cursor(
                    workdir, (url_index + 1) % n, 0, active_raw=None, prep_reject_streak=0
                )
            else:
                save_segment_cursor(
                    workdir,
                    url_index,
                    next_seg,
                    active_raw=active_raw,
                    prep_reject_streak=reject_streak,
                )
            return True

        record = ClipRecord(
            clip_id=clip_id,
            clip_uri=f"clips/{clip_path.name}",
            clip_sha256=sha256_file(clip_path),
            first_frame_uri=f"frames/{frame_path.name}",
            first_frame_sha256=sha256_file(frame_path),
            source_video_id=source_id,
            split_group_id=f"{source_id}:pending",
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
        if self._persist_pending_record(workdir, record, locked=True):
            reject_streak = 0
        else:
            logger.warning(
                "prepare segment done but row rejected (see prepare merge logs above) clip_id=%s url=%s; advancing segment",
                record.clip_id,
                url,
            )
            reject_streak = max(0, int(cur.get("prep_reject_streak", 0))) + 1

        if reject_streak >= PREP_MAX_CONSECUTIVE_REJECTS_PER_URL:
            logger.warning(
                "prepare skip entire source after %d consecutive prepare rejects url=%s url_index=%d",
                reject_streak,
                url,
                url_index,
            )
            _try_unlink_raw(raw_path)
            save_segment_cursor(
                workdir, (url_index + 1) % n, 0, active_raw=None, prep_reject_streak=0
            )
            return True

        next_seg = seg_index + 1
        if next_seg >= total_segments:
            _try_unlink_raw(raw_path)
            save_segment_cursor(
                workdir, (url_index + 1) % n, 0, active_raw=None, prep_reject_streak=0
            )
        else:
            save_segment_cursor(
                workdir,
                url_index,
                next_seg,
                active_raw=active_raw,
                prep_reject_streak=reject_streak,
            )

        return True

    def _mine_one_segment_dequeue(self, workdir: Path, sources_file: Path, worker_id: str) -> bool:
        wd = workdir.expanduser().resolve()
        wd.mkdir(parents=True, exist_ok=True)
        raw_base = wd / "raw" / worker_id
        clips_dir = wd / "clips"
        frames_dir = wd / "frames"
        cur = load_worker_segment_cursor(wd, worker_id)
        reject_streak = max(0, int(cur.get("prep_reject_streak", 0)))
        current_url: str | None = cur.get("current_url")
        seg_index = int(cur.get("segment_index", 0))
        active_raw: str | None = cur.get("active_raw")

        if not active_raw:
            reject_streak = 0
            url = pop_next_source_url(sources_file)
            if not url:
                save_worker_segment_cursor(
                    wd,
                    worker_id,
                    current_url=None,
                    segment_index=0,
                    active_raw=None,
                    prep_reject_streak=0,
                )
                return False
            raw_path = self.source_provider.download(url, raw_base)
            active_raw = str(raw_path.relative_to(wd))
            current_url = url
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=url,
                segment_index=0,
                active_raw=active_raw,
                prep_reject_streak=0,
            )
        else:
            raw_path = wd / active_raw
            url = current_url
            if not url or not raw_path.is_file():
                save_worker_segment_cursor(
                    wd,
                    worker_id,
                    current_url=None,
                    segment_index=0,
                    active_raw=None,
                    prep_reject_streak=0,
                )
                return self.mine_one_segment(
                    wd, sources_file, worker_id=worker_id, dequeue_sources=True
                )

        source_id = self.source_provider.source_video_id(url)
        probe = self.source_provider.probe(raw_path)
        format_info = probe.get("format", {})
        duration = float(format_info.get("duration", 0.0))
        total_segments = int(math.floor(duration / CLIP_DURATION_SEC))

        if total_segments <= 0:
            logger.warning("source has no 5s segments source_id=%s url=%s", source_id, url)
            _try_unlink_raw(raw_path)
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=None,
                segment_index=0,
                active_raw=None,
                prep_reject_streak=0,
            )
            return True

        if seg_index >= total_segments:
            _try_unlink_raw(raw_path)
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=None,
                segment_index=0,
                active_raw=None,
                prep_reject_streak=0,
            )
            return True

        stream = _video_stream(probe)
        width = int(stream.get("width", 1))
        height = int(stream.get("height", 1))
        r_frame_rate = stream.get("r_frame_rate", "0/1")
        numerator, denominator = r_frame_rate.split("/")
        fps = float(numerator) / max(float(denominator), 1.0)
        has_audio = _audio_present(probe)

        start = float(seg_index) * CLIP_DURATION_SEC
        clip_id = deterministic_clip_id(source_id, start, CLIP_DURATION_SEC)
        clip_path = clips_dir / f"{clip_id}.mp4"
        frame_path = frames_dir / f"{clip_id}.jpg"
        self.source_provider.create_clip(raw_path, clip_path, start, CLIP_DURATION_SEC)
        self.source_provider.extract_first_frame(clip_path, frame_path)
        if self._early_reject_bad_output_resolution(
            clip_path, frame_path, frames_dir, clip_id, source_url=url
        ):
            _try_unlink_raw(raw_path)
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=None,
                segment_index=0,
                active_raw=None,
                prep_reject_streak=0,
            )
            logger.info(
                "prepare dequeue worker=%s: finished bad-resolution source; next run pops next line from sources",
                worker_id,
            )
            return True

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
        if caption is None:
            logger.info(
                "prepare skip segment reason=caption_too_short clip_id=%s url=%s worker=%s",
                clip_id,
                url,
                worker_id,
            )
            self._scrub_rejected_segment_files(clip_path, frame_path, frames_dir, clip_id)
            next_seg = seg_index + 1
            if next_seg >= total_segments:
                _try_unlink_raw(raw_path)
                save_worker_segment_cursor(
                    wd,
                    worker_id,
                    current_url=None,
                    segment_index=0,
                    active_raw=None,
                    prep_reject_streak=0,
                )
            else:
                save_worker_segment_cursor(
                    wd,
                    worker_id,
                    current_url=url,
                    segment_index=next_seg,
                    active_raw=active_raw,
                    prep_reject_streak=reject_streak,
                )
            return True

        record = ClipRecord(
            clip_id=clip_id,
            clip_uri=f"clips/{clip_path.name}",
            clip_sha256=sha256_file(clip_path),
            first_frame_uri=f"frames/{frame_path.name}",
            first_frame_sha256=sha256_file(frame_path),
            source_video_id=source_id,
            split_group_id=f"{source_id}:pending",
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
        if self._persist_pending_record(wd, record, locked=True):
            reject_streak = 0
        else:
            logger.warning(
                "prepare segment done but row rejected (see prepare merge logs above) clip_id=%s url=%s worker=%s",
                record.clip_id,
                url,
                worker_id,
            )
            reject_streak += 1

        if reject_streak >= PREP_MAX_CONSECUTIVE_REJECTS_PER_URL:
            logger.warning(
                "prepare dequeue worker=%s skip entire source after %d consecutive prepare rejects url=%s",
                worker_id,
                reject_streak,
                url,
            )
            _try_unlink_raw(raw_path)
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=None,
                segment_index=0,
                active_raw=None,
                prep_reject_streak=0,
            )
            return True

        next_seg = seg_index + 1
        if next_seg >= total_segments:
            _try_unlink_raw(raw_path)
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=None,
                segment_index=0,
                active_raw=None,
                prep_reject_streak=0,
            )
        else:
            save_worker_segment_cursor(
                wd,
                worker_id,
                current_url=url,
                segment_index=next_seg,
                active_raw=active_raw,
                prep_reject_streak=reject_streak,
            )
        return True

    async def run_interval(
        self,
        *,
        sources_file: Path,
        netuid: int,
        miner_hotkey: str,
        interval_id: int,
        workdir: Path,
    ) -> tuple[Path, Path]:
        if self.store is None:
            raise RuntimeError("MinerPipeline requires store for run_interval")
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
                if caption is None:
                    logger.info(
                        "skip clip in interval build reason=caption_too_short clip_id=%s source_id=%s",
                        clip_id,
                        source_id,
                    )
                    self._scrub_rejected_segment_files(clip_path, frame_path, frames_dir, clip_id)
                    continue
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

