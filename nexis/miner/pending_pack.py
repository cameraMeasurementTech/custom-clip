"""Pending clip queue, upload cadence state, and deduplicated R2 packs."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from collections.abc import Awaitable, Callable
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..models import ClipRecord
from ..specs import DatasetSpecRegistry
from ..validator.assets import VideoAssetVerifier
from .preflight import (
    clip_ids_from_all_preflight_failures,
    clip_ids_from_caption_semantic_failures,
    prune_until_hard_checks_pass,
    run_preflight_on_candidates,
)
from .submission_draft import materialize_hashes_from_workdir, upload_records_as_interval_package

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[misc, assignment]

PENDING_RECORDS_JSONL = "miner_pending_records.jsonl"
SEGMENT_CURSOR_JSON = "miner_segment_cursor.json"
UPLOADED_CLIP_IDS_TXT = "miner_uploaded_clip_ids.txt"
UPLOAD_STATE_JSON = "miner_upload_state.json"
R2_UPLOAD_INTERVAL_HISTORY_JSON = "miner_r2_upload_interval_history.json"
CURSORS_SUBDIR = "cursors"


def _pending_lock_path(workdir: Path) -> Path:
    return workdir.expanduser().resolve() / f"{PENDING_RECORDS_JSONL}.lock"


@contextmanager
def _posix_flock_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "a+b")
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()


def append_pending_record_locked(workdir: Path, record: ClipRecord) -> None:
    """Append one JSON line with exclusive lock (POSIX fcntl). Falls back to unlocked append if no fcntl."""
    wd = workdir.expanduser().resolve()
    path = wd / PENDING_RECORDS_JSONL
    line = record.model_dump_json() + "\n"
    wd.mkdir(parents=True, exist_ok=True)
    if fcntl is None:
        logger.warning("fcntl unavailable; pending append is not cross-process safe")
        with path.open("a", encoding="utf-8") as f:
            f.write(line)
    else:
        with _posix_flock_lock(_pending_lock_path(wd)):
            with path.open("a", encoding="utf-8") as f:
                f.write(line)
    logger.info("appended pending clip_id=%s start_sec=%.3f", record.clip_id, record.clip_start_sec)


def append_pending_record(workdir: Path, record: ClipRecord) -> None:
    """Append one line (locked when fcntl is available)."""
    append_pending_record_locked(workdir, record)


def load_pending_records(workdir: Path) -> list[ClipRecord]:
    path = workdir.expanduser() / PENDING_RECORDS_JSONL
    if not path.is_file():
        return []
    out: list[ClipRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ClipRecord.model_validate_json(line))
    return out


def _parse_pending_lines(text: str) -> list[ClipRecord]:
    out: list[ClipRecord] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        out.append(ClipRecord.model_validate_json(line))
    return out


@dataclass(frozen=True)
class PendingSaveValidatorConfig:
    """If enabled, mine_one_segment commits via merge_validated_pending_record instead of blind append."""

    enabled: bool
    spec_id: str
    registry: DatasetSpecRegistry
    miner_hotkey: str
    prepare_sample_interval_id: int = 0
    run_local_asset_verify: bool = True
    caption_semantic_checker: Any | None = None
    category_checker: Any | None = None


def merge_validated_pending_record(
    workdir: Path,
    new_record: ClipRecord,
    cfg: PendingSaveValidatorConfig,
) -> tuple[bool, list[ClipRecord]]:
    """Merge ``new_record`` with existing JSONL using validator hard checks + local asset/LLM on the new row only.

    Pending file updates run under the pending lock. **Local clip/frame files for rejected or superseded rows
    are deleted immediately after the lock is released** (same call), so bad segments do not linger on disk.

    Returns ``(accepted, rows_that_were_deleted_from_disk)`` — the second value lists rows whose assets were
    removed (for logging/tests); files are already gone when this returns.
    """
    wd = workdir.expanduser().resolve()
    path = wd / PENDING_RECORDS_JSONL
    wd.mkdir(parents=True, exist_ok=True)
    spec = cfg.registry.get(cfg.spec_id)
    want_llm = (
        cfg.caption_semantic_checker is not None
        and getattr(cfg.caption_semantic_checker, "active", False)
    ) or (cfg.category_checker is not None and getattr(cfg.category_checker, "active", False))

    def _attempt(existing: list[ClipRecord]) -> tuple[bool, list[ClipRecord], list[ClipRecord] | None]:
        combined = existing + [new_record]
        hard_before_prune = spec.run_hard_checks(combined)
        kept_hard = prune_until_hard_checks_pass(combined, spec)
        kept_ids = {r.clip_id for r in kept_hard}
        if new_record.clip_id not in kept_ids:
            logger.warning(
                "prepare merge REJECT (hard checks / overlap) clip_id=%s source_url=%s caption=%r failures=%s",
                new_record.clip_id,
                new_record.source_video_url,
                (new_record.caption or "")[:200],
                hard_before_prune.failures,
            )
            return False, [new_record], None

        ordered = [r for r in combined if r.clip_id in kept_ids]
        verifier = spec.build_asset_verifier()
        asset_out = None
        if isinstance(verifier, VideoAssetVerifier) and (cfg.run_local_asset_verify or want_llm):
            miner_dir = wd / ".prepare_preflight_assets"
            miner_dir.mkdir(parents=True, exist_ok=True)
            asset_out = verifier.verify_local_workdir(
                workdir=wd, sampled=[new_record], miner_dir=miner_dir
            )
            bad_asset = clip_ids_from_all_preflight_failures(asset_out.failures, [new_record])
            if new_record.clip_id in bad_asset:
                logger.warning(
                    "prepare merge REJECT (local assets) clip_id=%s source_url=%s failures=%s",
                    new_record.clip_id,
                    new_record.source_video_url,
                    asset_out.failures,
                )
                return False, [new_record], None

        failures_llm: list[str] = []
        if want_llm and asset_out is not None:
            semantic_failures: list[str] = []
            if cfg.caption_semantic_checker is not None and getattr(
                cfg.caption_semantic_checker, "active", False
            ):
                chk = getattr(cfg.caption_semantic_checker, "check", None)
                if callable(chk):
                    semantic_failures = chk(
                        sampled=[new_record],
                        frame_paths_by_clip_id=asset_out.semantic_frames_by_clip_id,
                    )
                    failures_llm.extend(semantic_failures)
            bad_semantic = clip_ids_from_caption_semantic_failures(semantic_failures)
            if new_record.clip_id not in bad_semantic and cfg.category_checker is not None and getattr(
                cfg.category_checker, "active", False
            ):
                cchk = getattr(cfg.category_checker, "check", None)
                if callable(cchk):
                    failures_llm.extend(
                        cchk(
                            sampled=[new_record],
                            frame_paths_by_clip_id=asset_out.semantic_frames_by_clip_id,
                        )
                    )
            bad_llm = clip_ids_from_all_preflight_failures(failures_llm, [new_record])
            if new_record.clip_id in bad_llm:
                logger.warning(
                    "prepare merge REJECT (LLM gate) clip_id=%s source_url=%s failures=%s",
                    new_record.clip_id,
                    new_record.source_video_url,
                    failures_llm,
                )
                return False, [new_record], None

        if want_llm and asset_out is None:
            logger.warning(
                "prepare validate: LLM checks requested but asset verifier unavailable for spec=%s; skipping LLM",
                cfg.spec_id,
            )

        final_ids = {r.clip_id for r in ordered}
        removed_existing = [r for r in existing if r.clip_id not in final_ids]
        return True, removed_existing, ordered

    accepted = False
    to_delete: list[ClipRecord] = []

    def _run_merge_locked_body() -> None:
        nonlocal accepted, to_delete
        existing = _parse_pending_lines(path.read_text(encoding="utf-8")) if path.is_file() else []
        acc, td, ordered = _attempt(existing)
        accepted = acc
        to_delete = td
        if accepted and ordered is not None:
            content = "\n".join(r.model_dump_json() for r in ordered) + ("\n" if ordered else "")
            path.write_text(content, encoding="utf-8")
            cap_preview = (new_record.caption or "").strip().replace("\n", " ")[:120]
            logger.info(
                "prepare merge ACCEPT clip_id=%s url=%s pending_rows_after_merge=%d "
                "older_pending_rows_removed=%d caption_preview=%r asset_verify=%s llm_gate=%s",
                new_record.clip_id,
                new_record.source_video_url,
                len(ordered),
                len(to_delete),
                cap_preview,
                cfg.run_local_asset_verify,
                want_llm,
            )
        elif not accepted:
            logger.info(
                "prepare merge finished REJECT clip_id=%s (details at WARNING above if logged)",
                new_record.clip_id,
            )

    if fcntl is None:
        logger.warning("fcntl unavailable; merge_validated_pending_record is not cross-process safe")
        _run_merge_locked_body()
    else:
        with _posix_flock_lock(_pending_lock_path(wd)):
            _run_merge_locked_body()

    if to_delete:
        delete_local_assets_for_records(wd, to_delete)

    return accepted, to_delete


def load_segment_cursor(workdir: Path) -> dict[str, Any]:
    path = workdir.expanduser() / SEGMENT_CURSOR_JSON
    if not path.is_file():
        return {"url_index": 0, "segment_index": 0, "active_raw": None, "prep_reject_streak": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ar = data.get("active_raw")
        return {
            "url_index": max(0, int(data.get("url_index", 0))),
            "segment_index": max(0, int(data.get("segment_index", 0))),
            "active_raw": str(ar) if ar else None,
            "prep_reject_streak": max(0, int(data.get("prep_reject_streak", 0))),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"url_index": 0, "segment_index": 0, "active_raw": None, "prep_reject_streak": 0}


def save_segment_cursor(
    workdir: Path,
    url_index: int,
    segment_index: int,
    *,
    active_raw: str | None,
    prep_reject_streak: int = 0,
) -> None:
    path = workdir.expanduser() / SEGMENT_CURSOR_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "url_index": url_index,
        "segment_index": segment_index,
        "active_raw": active_raw,
        "prep_reject_streak": max(0, int(prep_reject_streak)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _worker_cursor_path(workdir: Path, worker_id: str) -> Path:
    return workdir.expanduser().resolve() / CURSORS_SUBDIR / f"miner_segment_cursor.{worker_id}.json"


def load_worker_segment_cursor(workdir: Path, worker_id: str) -> dict[str, Any]:
    path = _worker_cursor_path(workdir, worker_id)
    if not path.is_file():
        return {"current_url": None, "segment_index": 0, "active_raw": None, "prep_reject_streak": 0}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ar = data.get("active_raw")
        cu = data.get("current_url")
        return {
            "current_url": str(cu).strip() if cu else None,
            "segment_index": max(0, int(data.get("segment_index", 0))),
            "active_raw": str(ar) if ar else None,
            "prep_reject_streak": max(0, int(data.get("prep_reject_streak", 0))),
        }
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return {"current_url": None, "segment_index": 0, "active_raw": None, "prep_reject_streak": 0}


def save_worker_segment_cursor(
    workdir: Path,
    worker_id: str,
    *,
    current_url: str | None,
    segment_index: int,
    active_raw: str | None,
    prep_reject_streak: int = 0,
) -> None:
    path = _worker_cursor_path(workdir, worker_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "current_url": current_url,
        "segment_index": segment_index,
        "active_raw": active_raw,
        "prep_reject_streak": max(0, int(prep_reject_streak)),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _resolved_path_under_workdir(workdir: Path, relative_uri: str) -> Path | None:
    root = workdir.expanduser().resolve()
    candidate = (root / relative_uri.lstrip("/")).resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        logger.warning("skip delete path outside workdir: %s", candidate)
        return None
    return candidate


def delete_local_assets_for_records(workdir: Path, records: list[ClipRecord]) -> None:
    """Remove clip, frame, and caption scratch dirs for each record; paths must stay under workdir.

    Deletes by record URIs and by canonical ``clips/{clip_id}.mp4`` / ``frames/{clip_id}.jpg`` so
    local files are removed even if stored paths ever diverge from URIs.
    """
    wd = workdir.expanduser().resolve()
    for row in records:
        file_paths: list[Path] = []
        for rel in (
            row.clip_uri,
            row.first_frame_uri,
            f"clips/{row.clip_id}.mp4",
            f"frames/{row.clip_id}.jpg",
        ):
            p = _resolved_path_under_workdir(wd, rel)
            if p is not None and p.is_file():
                file_paths.append(p)

        seen: set[Path] = set()
        for p in file_paths:
            resolved = p.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            try:
                resolved.unlink()
                logger.info("deleted local file %s", resolved)
            except OSError as exc:
                logger.warning("failed delete file %s: %s", resolved, exc)

        cap_dir = wd / "frames" / row.clip_id
        if not cap_dir.is_dir():
            continue
        try:
            cap_resolved = cap_dir.resolve()
            cap_resolved.relative_to(wd)
        except ValueError:
            logger.warning("skip delete caption dir outside workdir: %s", cap_dir)
            continue
        except OSError as exc:
            logger.warning("skip delete caption dir %s: %s", cap_dir, exc)
            continue
        try:
            shutil.rmtree(cap_resolved, ignore_errors=False)
            logger.info("deleted local caption dir %s", cap_resolved)
        except OSError as exc:
            logger.warning("failed rmtree %s: %s", cap_resolved, exc)

        for scratch_root in (".prepare_preflight_assets", ".preflight_assets"):
            sem_dir = (wd / scratch_root / "semantic" / row.clip_id).resolve()
            try:
                sem_dir.relative_to(wd)
            except ValueError:
                continue
            if sem_dir.is_dir():
                try:
                    shutil.rmtree(sem_dir, ignore_errors=False)
                    logger.info("deleted validator scratch dir %s", sem_dir)
                except OSError as exc:
                    logger.warning("failed rmtree %s: %s", sem_dir, exc)


def delete_local_out_interval(workdir: Path, interval_id: int) -> None:
    out_dir = workdir.expanduser().resolve() / "out" / str(interval_id)
    if out_dir.is_dir():
        try:
            shutil.rmtree(out_dir, ignore_errors=False)
            logger.info("deleted local out dir %s", out_dir)
        except OSError as exc:
            logger.warning("failed rmtree out dir %s: %s", out_dir, exc)


def load_uploaded_clip_ids(workdir: Path) -> set[str]:
    path = workdir.expanduser() / UPLOADED_CLIP_IDS_TXT
    if not path.is_file():
        return set()
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def append_uploaded_clip_ids(workdir: Path, clip_ids: list[str]) -> None:
    if not clip_ids:
        return
    path = workdir.expanduser() / UPLOADED_CLIP_IDS_TXT
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for cid in clip_ids:
            f.write(f"{cid}\n")
    logger.info("recorded %d clip_id(s) as uploaded (dedupe log)", len(clip_ids))


def remove_clip_ids_from_pending(workdir: Path, remove_ids: set[str]) -> None:
    """Rewrite ``miner_pending_records.jsonl`` without rows whose ``clip_id`` is in ``remove_ids``.

    Used after a successful R2 pack upload (and after preflight drops) so uploaded clips do not stay in the
    pending queue. Uses the same POSIX lock as append/merge when ``fcntl`` is available.
    """
    if not remove_ids:
        return
    wd = workdir.expanduser().resolve()
    path = wd / PENDING_RECORDS_JSONL
    if not path.is_file():
        return

    def _rewrite() -> tuple[int, int]:
        kept: list[str] = []
        removed_lines = 0
        for line in path.read_text(encoding="utf-8").splitlines():
            s = line.strip()
            if not s:
                continue
            rec = ClipRecord.model_validate_json(s)
            if rec.clip_id in remove_ids:
                removed_lines += 1
            else:
                kept.append(s)
        path.write_text("\n".join(kept) + ("\n" if kept else ""), encoding="utf-8")
        return removed_lines, len(kept)

    if fcntl is None:
        logger.warning("fcntl unavailable; pending jsonl rewrite is not cross-process safe")
        removed_lines, remaining = _rewrite()
    else:
        with _posix_flock_lock(_pending_lock_path(wd)):
            removed_lines, remaining = _rewrite()

    logger.info(
        "miner_pending_records.jsonl: removed %d line(s) for uploaded/processed clip_ids; %d line(s) remain path=%s",
        removed_lines,
        remaining,
        path,
    )


def load_last_pack_block(workdir: Path) -> int | None:
    path = workdir.expanduser() / UPLOAD_STATE_JSON
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        v = data.get("last_pack_block")
        return int(v) if v is not None else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def save_last_pack_block(workdir: Path, block: int) -> None:
    path = workdir.expanduser() / UPLOAD_STATE_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"last_pack_block": block}, indent=2), encoding="utf-8")


def load_r2_upload_interval_history(workdir: Path) -> list[int]:
    path = workdir.expanduser().resolve() / R2_UPLOAD_INTERVAL_HISTORY_JSON
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        raw = data.get("interval_ids", [])
        return [int(x) for x in raw]
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        logger.warning("failed to read R2 upload history; starting fresh path=%s", path)
        return []


def save_r2_upload_interval_history(workdir: Path, interval_ids: list[int]) -> None:
    path = workdir.expanduser().resolve() / R2_UPLOAD_INTERVAL_HISTORY_JSON
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps({"interval_ids": interval_ids}, indent=2),
        encoding="utf-8",
    )


async def maybe_prune_old_r2_interval_prefix(
    *,
    store: Any,
    workdir: Path,
    interval_id: int,
    enabled: bool,
    uploads_ago: int,
) -> None:
    """After a successful upload to ``interval_id``, drop the oldest tracked id when at capacity and delete that prefix on R2."""
    wd = workdir.expanduser().resolve()
    ago = max(1, int(uploads_ago))
    hist = load_r2_upload_interval_history(wd)
    if enabled and len(hist) >= ago:
        victim = hist.pop(0)
        if victim != interval_id:
            try:
                n = await store.delete_prefix(str(victim), use_write=True)
                logger.info(
                    "pruned old R2 interval prefix %s (%d objects); current upload interval_id=%s",
                    victim,
                    n,
                    interval_id,
                )
            except Exception:
                logger.exception(
                    "failed to prune R2 prefix victim=%s (current interval_id=%s)",
                    victim,
                    interval_id,
                )
        else:
            logger.warning(
                "r2 prune skipped victim equals current interval_id=%s (history may be stale)",
                interval_id,
            )
    hist.append(interval_id)
    while len(hist) > ago:
        hist.pop(0)
    save_r2_upload_interval_history(wd, hist)


def records_for_interval_upload(records: list[ClipRecord], interval_id: int) -> list[ClipRecord]:
    return [
        r.model_copy(update={"split_group_id": f"{r.source_video_id}:{interval_id}"})
        for r in records
    ]


async def wait_until_r2_manifest_slot(
    *,
    store: Any,
    interval_id: int,
    wait_sec: float,
    interval_refresher: Callable[[], Awaitable[int]] | None,
) -> int:
    """Block until ``{interval_id}/manifest.json`` is absent on R2, recomputing interval when requested.

    If ``wait_sec`` <= 0, returns ``interval_id`` immediately without checking R2.
    """
    if wait_sec <= 0:
        return interval_id
    id_cur = interval_id
    while True:
        key = f"{id_cur}/manifest.json"
        if not await store.object_exists(key):
            if id_cur != interval_id:
                logger.info("R2 manifest slot free interval_id=%s (was waiting for %s)", id_cur, interval_id)
            return id_cur
        logger.info(
            "R2 manifest already exists key=%s; sleeping %.0fs then retrying%s",
            key,
            wait_sec,
            " (refresh interval from chain)" if interval_refresher is not None else "",
        )
        await asyncio.sleep(wait_sec)
        if interval_refresher is not None:
            id_cur = await interval_refresher()


async def pack_deduped_pending_and_upload(
    *,
    store: Any,
    workdir: Path,
    interval_id: int,
    netuid: int,
    miner_hotkey: str,
    spec_id: str,
    category: str | None,
    r2_prune_old_prefix: bool = True,
    r2_prune_uploads_ago: int = 2,
    manifest_busy_wait_sec: float = 1200.0,
    interval_refresher: Callable[[], Awaitable[int]] | None = None,
    preflight_before_upload: bool = False,
    preflight_semantic_checker: Any | None = None,
    preflight_category_checker: Any | None = None,
    spec_registry: DatasetSpecRegistry | None = None,
) -> tuple[Path, Path, list[ClipRecord], int]:
    """Upload pending clips not yet in miner_uploaded_clip_ids.txt; trim pending.

    When there are no new rows (or preflight removes all), still uploads an **empty** ``dataset.parquet``
    plus manifest (``record_count=0``) so the validator receives both objects.

    Returns ``(dataset_path, manifest_path, uploaded_rows, interval_id)``; ``uploaded_rows`` may be empty.
    """
    wd = workdir.expanduser()
    uploaded = load_uploaded_clip_ids(wd)
    pending = load_pending_records(wd)
    candidates = [r for r in pending if r.clip_id not in uploaded]

    interval_id = await wait_until_r2_manifest_slot(
        store=store,
        interval_id=interval_id,
        wait_sec=manifest_busy_wait_sec,
        interval_refresher=interval_refresher,
    )

    if candidates and preflight_before_upload:
        kept, removed = run_preflight_on_candidates(
            workdir=wd,
            records=candidates,
            spec_id=spec_id,
            miner_hotkey=miner_hotkey,
            interval_id=interval_id,
            registry=spec_registry,
            caption_semantic_checker=preflight_semantic_checker,
            category_checker=preflight_category_checker,
        )
        if removed:
            remove_clip_ids_from_pending(wd, {r.clip_id for r in removed})
            delete_local_assets_for_records(wd, removed)
            logger.info(
                "preflight before upload interval=%d removed=%d kept=%d",
                interval_id,
                len(removed),
                len(kept),
            )
        candidates = kept

    if not candidates:
        logger.info(
            "pack interval=%d uploading empty dataset+manifest (0 new rows; satisfies validator file presence)",
            interval_id,
        )
        ds, mf = await upload_records_as_interval_package(
            store=store,
            records=[],
            interval_id=interval_id,
            netuid=netuid,
            miner_hotkey=miner_hotkey,
            workdir=wd,
            spec_id=spec_id,
            category=category,
        )
        await maybe_prune_old_r2_interval_prefix(
            store=store,
            workdir=wd,
            interval_id=interval_id,
            enabled=r2_prune_old_prefix,
            uploads_ago=r2_prune_uploads_ago,
        )
        return ds, mf, [], interval_id

    rows = records_for_interval_upload(candidates, interval_id)
    logger.info(
        "pack interval=%d new_records=%d (honest clip_id / clip_start_sec; deduped)",
        interval_id,
        len(rows),
    )
    hashed = materialize_hashes_from_workdir(rows, wd)
    ds, mf = await upload_records_as_interval_package(
        store=store,
        records=hashed,
        interval_id=interval_id,
        netuid=netuid,
        miner_hotkey=miner_hotkey,
        workdir=wd,
        spec_id=spec_id,
        category=category,
    )
    clip_ids = [r.clip_id for r in candidates]
    append_uploaded_clip_ids(wd, clip_ids)
    remove_clip_ids_from_pending(wd, set(clip_ids))
    await maybe_prune_old_r2_interval_prefix(
        store=store,
        workdir=wd,
        interval_id=interval_id,
        enabled=r2_prune_old_prefix,
        uploads_ago=r2_prune_uploads_ago,
    )
    return ds, mf, candidates, interval_id
