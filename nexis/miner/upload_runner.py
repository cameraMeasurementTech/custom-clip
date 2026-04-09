"""Shared mine-upload logic (CLI and python -m entry points)."""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path

from ..config import Settings
from ..specs import DatasetSpecRegistry
from .pending_pack import (
    delete_local_assets_for_records,
    delete_local_out_interval,
    pack_deduped_pending_and_upload,
)
from .preflight import build_preflight_llm_checkers
from ..storage.r2 import R2S3Store

logger = logging.getLogger(__name__)


async def _mine_upload_async(
    *,
    store: R2S3Store,
    workdir: Path,
    interval_id: int,
    netuid: int,
    miner_hotkey: str,
    active_spec: str,
    active_category: str,
    delete_local: bool,
    r2_prune_old_prefix: bool,
    r2_prune_uploads_ago: int,
    manifest_busy_wait_sec: float,
    interval_refresher: Callable[[], Awaitable[int]] | None,
    emit: Callable[[str], None],
    settings: Settings | None = None,
) -> None:
    semantic_checker = None
    category_checker = None
    if settings is not None and settings.miner_preflight_before_upload:
        if settings.miner_preflight_semantic or settings.miner_preflight_category:
            semantic_checker, category_checker = build_preflight_llm_checkers(settings)
    packed = await pack_deduped_pending_and_upload(
        store=store,
        workdir=workdir,
        interval_id=interval_id,
        netuid=netuid,
        miner_hotkey=miner_hotkey,
        spec_id=active_spec,
        category=active_category or None,
        r2_prune_old_prefix=r2_prune_old_prefix,
        r2_prune_uploads_ago=r2_prune_uploads_ago,
        manifest_busy_wait_sec=manifest_busy_wait_sec,
        interval_refresher=interval_refresher,
        preflight_before_upload=bool(settings.miner_preflight_before_upload) if settings else True,
        preflight_semantic_checker=semantic_checker,
        preflight_category_checker=category_checker,
        spec_registry=DatasetSpecRegistry.with_defaults(),
    )
    if packed is None:
        emit("mine-upload: skipped (unexpected)")
        return
    _ds, _mf, uploaded_rows, used_interval = packed
    if uploaded_rows:
        emit(
            f"mine-upload: uploaded interval={used_interval} records={len(uploaded_rows)} "
            f"to R2 prefix {used_interval}/"
        )
    else:
        emit(
            f"mine-upload: uploaded empty package interval={used_interval} "
            f"(dataset.parquet + manifest, 0 clips) to R2 prefix {used_interval}/"
        )
    if delete_local:
        delete_local_assets_for_records(workdir, uploaded_rows)
        delete_local_out_interval(workdir, used_interval)
        emit("mine-upload: deleted local assets for uploaded clips and out dir")


def run_mine_upload(
    *,
    settings: Settings,
    store: R2S3Store,
    workdir: Path,
    interval_id: int,
    active_spec: str,
    active_category: str,
    spec_registry_enabled_ids: set[str],
    spec_registry_all_ids: set[str],
    delete_local: bool,
    miner_hotkey: str,
    emit: Callable[[str], None] | None = None,
    r2_prune_old_prefix: bool | None = None,
    r2_prune_uploads_ago: int | None = None,
    manifest_busy_wait_sec: float | None = None,
    interval_refresher: Callable[[], Awaitable[int]] | None = None,
) -> None:
    """Upload deduped pending pack. Raises ValueError on bad input."""
    if not active_category:
        raise ValueError("NEXIS_DATASET_CATEGORY must not be empty")
    if active_spec not in spec_registry_enabled_ids:
        raise ValueError(
            f"Spec '{active_spec}' is not enabled for miner (enabled: {', '.join(sorted(spec_registry_enabled_ids))})"
        )
    if active_spec not in spec_registry_all_ids:
        raise ValueError(f"Unknown dataset spec: {active_spec}")

    wd = workdir.expanduser().resolve()
    if not wd.is_dir():
        raise ValueError(f"workdir is not a directory: {wd}")

    prune = settings.miner_r2_prune_old_prefix if r2_prune_old_prefix is None else r2_prune_old_prefix
    prune_ago = (
        settings.miner_r2_prune_uploads_ago if r2_prune_uploads_ago is None else r2_prune_uploads_ago
    )
    busy_wait = (
        settings.miner_upload_manifest_busy_wait_sec
        if manifest_busy_wait_sec is None
        else manifest_busy_wait_sec
    )

    def _emit(msg: str) -> None:
        if emit is not None:
            emit(msg)
        else:
            logger.info("%s", msg)

    asyncio.run(
        _mine_upload_async(
            store=store,
            workdir=wd,
            interval_id=interval_id,
            netuid=settings.netuid,
            miner_hotkey=miner_hotkey,
            active_spec=active_spec,
            active_category=active_category,
            delete_local=delete_local,
            r2_prune_old_prefix=prune,
            r2_prune_uploads_ago=prune_ago,
            manifest_busy_wait_sec=busy_wait,
            interval_refresher=interval_refresher,
            emit=_emit,
            settings=settings,
        )
    )
