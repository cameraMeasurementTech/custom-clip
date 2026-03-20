"""Owner validator sync helpers for shared Hippius buckets."""

from __future__ import annotations

import asyncio
import json
import logging
import shutil
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Callable

from ..hash_utils import sha256_file
from ..models import ClipRecord, ValidationDecision
from ..serialization import write_dataset_parquet, write_manifest
from ..specs import DEFAULT_SPEC_ID
from .pipeline import ValidatorPipeline

logger = logging.getLogger(__name__)


def _run_async(coro: Any) -> Any:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    close_coro = getattr(coro, "close", None)
    if callable(close_coro):
        close_coro()
    raise RuntimeError("Synchronous owner sync helper cannot run inside an active event loop")


def normalize_relative_uri(value: str) -> str | None:
    text = value.strip().lstrip("/")
    if not text:
        return None
    parts = PurePosixPath(text).parts
    if any(part in {"", ".", ".."} for part in parts):
        return None
    return "/".join(parts)


def parse_record_info(raw: str) -> dict[str, list[float]]:
    try:
        payload = json.loads(raw)
    except Exception:
        return {}
    if isinstance(payload, dict) and isinstance(payload.get(DEFAULT_SPEC_ID), dict):
        payload = payload[DEFAULT_SPEC_ID]
    if not isinstance(payload, dict):
        return {}
    parsed: dict[str, list[float]] = {}
    for source_url, values in payload.items():
        if not isinstance(source_url, str) or not isinstance(values, list):
            continue
        starts: list[float] = []
        for item in values:
            try:
                starts.append(float(item))
            except Exception:
                continue
        if starts:
            parsed[source_url] = sorted(starts)
    return parsed


def canonical_source_key_from_url(source_video_url: str) -> str:
    return source_video_url.strip()


def serialize_record_info(record_index: dict[str, list[float]]) -> str:
    payload = {
        f"{DEFAULT_SPEC_ID}": {
            source_url: [f"{value:.3f}" for value in sorted(values)]
            for source_url, values in sorted(record_index.items())
        },
    }
    return json.dumps(payload, indent=2)


def load_record_info_snapshot(
    *,
    record_info_store: Any | None,
    object_key: str,
    workdir: Path,
) -> dict[str, list[float]]:
    return _run_async(
        load_record_info_snapshot_async(
            record_info_store=record_info_store,
            object_key=object_key,
            workdir=workdir,
        )
    )


async def load_record_info_snapshot_async(
    *,
    record_info_store: Any | None,
    object_key: str,
    workdir: Path,
) -> dict[str, list[float]]:
    if record_info_store is None:
        return {}
    local = workdir / "record-info" / "snapshot.json"
    ok = await record_info_store.download_file(object_key, local)
    if not ok or not local.exists():
        return {}
    return parse_record_info(local.read_text(encoding="utf-8"))


def merge_records_into_index(
    *,
    record_index: dict[str, list[float]],
    records: list[ClipRecord],
) -> None:
    for row in records:
        source_url = row.source_video_url
        values = record_index.setdefault(source_url, [])
        values.append(float(row.clip_start_sec))
    for source_url, values in record_index.items():
        deduped = sorted(set(round(value, 3) for value in values))
        record_index[source_url] = deduped


def upload_record_info_snapshot(
    *,
    record_info_store: Any | None,
    object_key: str,
    workdir: Path,
    record_index: dict[str, list[float]],
) -> None:
    _run_async(
        upload_record_info_snapshot_async(
            record_info_store=record_info_store,
            object_key=object_key,
            workdir=workdir,
            record_index=record_index,
        )
    )


async def upload_record_info_snapshot_async(
    *,
    record_info_store: Any | None,
    object_key: str,
    workdir: Path,
    record_index: dict[str, list[float]],
) -> None:
    if record_info_store is None:
        return
    local = workdir / "record-info" / "snapshot.json"
    local.parent.mkdir(parents=True, exist_ok=True)
    local.write_text(serialize_record_info(record_index), encoding="utf-8")
    await record_info_store.upload_file(object_key, local, use_write=True)


def upload_validated_datasets_to_owner_bucket(
    *,
    owner_store: Any | None,
    source_store_for_hotkey: Callable[[str], Any],
    validator: ValidatorPipeline,
    decisions: list[ValidationDecision],
    interval_id: int,
    workdir: Path,
) -> dict[str, list[ClipRecord]]:
    return _run_async(
        upload_validated_datasets_to_owner_bucket_async(
            owner_store=owner_store,
            source_store_for_hotkey=source_store_for_hotkey,
            validator=validator,
            decisions=decisions,
            interval_id=interval_id,
            workdir=workdir,
        )
    )


async def upload_validated_datasets_to_owner_bucket_async(
    *,
    owner_store: Any | None,
    source_store_for_hotkey: Callable[[str], Any],
    validator: ValidatorPipeline,
    decisions: list[ValidationDecision],
    interval_id: int,
    workdir: Path,
) -> dict[str, list[ClipRecord]]:
    published_rows_by_hotkey: dict[str, list[ClipRecord]] = {}
    if owner_store is None:
        return published_rows_by_hotkey
    artifacts = validator.last_interval_artifacts
    if artifacts.interval_id != interval_id:
        return published_rows_by_hotkey
    for decision in decisions:
        if not decision.accepted:
            continue
        hotkey = decision.miner_hotkey
        records = artifacts.records_by_hotkey.get(hotkey, [])
        manifest = artifacts.manifests_by_hotkey.get(hotkey)
        if manifest is None:
            continue

        out_dir = workdir / "owner-upload" / str(interval_id) / hotkey
        out_dir.mkdir(parents=True, exist_ok=True)
        key_prefix = f"{interval_id}/{hotkey}"
        source_store = source_store_for_hotkey(hotkey)
        try:
            publishable_rows: list[ClipRecord] = []
            for row in records:
                pending_assets: list[tuple[str, Path]] = []
                row_asset_missing = False
                for relative_uri in (row.clip_uri, row.first_frame_uri):
                    safe_uri = normalize_relative_uri(relative_uri)
                    if safe_uri is None:
                        row_asset_missing = True
                        break
                    src_key = f"{interval_id}/{safe_uri}"
                    local_asset = out_dir / "assets" / safe_uri
                    ok = await source_store.download_file(src_key, local_asset)
                    if not ok:
                        row_asset_missing = True
                        break
                    expected_sha = (
                        row.clip_sha256
                        if relative_uri == row.clip_uri
                        else row.first_frame_sha256
                    )
                    if sha256_file(local_asset) != expected_sha:
                        row_asset_missing = True
                        break
                    pending_assets.append((safe_uri, local_asset))
                if row_asset_missing:
                    continue
                for safe_uri, local_asset in pending_assets:
                    await owner_store.upload_file(
                        f"{key_prefix}/{safe_uri}",
                        local_asset,
                        use_write=True,
                    )
                publishable_rows.append(row)

            if not publishable_rows:
                continue

            dataset_path = out_dir / "dataset.parquet"
            manifest_path = out_dir / "manifest.json"
            write_dataset_parquet(publishable_rows, dataset_path)
            published_manifest = manifest.model_copy(deep=True)
            published_manifest.record_count = len(publishable_rows)
            published_manifest.dataset_sha256 = sha256_file(dataset_path)
            write_manifest(published_manifest, manifest_path)

            await owner_store.upload_file(
                f"{key_prefix}/dataset.parquet",
                dataset_path,
                use_write=True,
            )
            await owner_store.upload_file(
                f"{key_prefix}/manifest.json",
                manifest_path,
                use_write=True,
            )
            published_rows_by_hotkey[hotkey] = publishable_rows
        finally:
            _cleanup_owner_upload_workdir(out_dir)
    return published_rows_by_hotkey


def _cleanup_owner_upload_workdir(out_dir: Path) -> None:
    try:
        if out_dir.exists():
            shutil.rmtree(out_dir)
        interval_dir = out_dir.parent
        try:
            interval_dir.rmdir()
        except OSError:
            pass
        try:
            interval_dir.parent.rmdir()
        except OSError:
            pass
    except Exception as exc:
        logger.warning("failed to remove owner upload cache dir=%s error=%s", out_dir, exc)
