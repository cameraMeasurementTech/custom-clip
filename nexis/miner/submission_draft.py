"""Parquet + manifest build and upload for miner submissions."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from ..hash_utils import sha256_file
from ..models import ClipRecord, IntervalManifest
from ..protocol import SCHEMA_VERSION
from ..serialization import write_dataset_parquet, write_manifest

logger = logging.getLogger(__name__)


def materialize_hashes_from_workdir(records: list[ClipRecord], workdir: Path) -> list[ClipRecord]:
    """Set clip_sha256 / first_frame_sha256 from files under workdir."""
    resolved: list[ClipRecord] = []
    for row in records:
        clip_path = workdir / row.clip_uri.lstrip("/")
        frame_path = workdir / row.first_frame_uri.lstrip("/")
        if not clip_path.is_file():
            raise FileNotFoundError(f"missing clip asset for clip_id={row.clip_id}: {clip_path}")
        if not frame_path.is_file():
            raise FileNotFoundError(f"missing frame asset for clip_id={row.clip_id}: {frame_path}")
        resolved.append(
            row.model_copy(
                update={
                    "clip_sha256": sha256_file(clip_path),
                    "first_frame_sha256": sha256_file(frame_path),
                }
            )
        )
    return resolved


async def upload_records_as_interval_package(
    *,
    store: Any,
    records: list[ClipRecord],
    interval_id: int,
    netuid: int,
    miner_hotkey: str,
    workdir: Path,
    spec_id: str,
    category: str | None,
) -> tuple[Path, Path]:
    """Write Parquet + manifest under workdir/out/{interval_id}/ and upload to store."""
    out_dir = workdir / "out" / str(interval_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    dataset_path = out_dir / "dataset.parquet"
    write_dataset_parquet(records, dataset_path)
    manifest = IntervalManifest(
        protocol_version="1.0.0",
        schema_version=SCHEMA_VERSION,
        spec_id=spec_id,
        dataset_type=spec_id,
        category=category,
        netuid=netuid,
        miner_hotkey=miner_hotkey,
        interval_id=interval_id,
        record_count=len(records),
        dataset_sha256=sha256_file(dataset_path),
    )
    manifest_path = out_dir / "manifest.json"
    write_manifest(manifest, manifest_path)

    assets: dict[str, Path] = {}
    for row in records:
        assets[row.clip_uri] = workdir / row.clip_uri.lstrip("/")
        assets[row.first_frame_uri] = workdir / row.first_frame_uri.lstrip("/")

    base_key = f"{interval_id}"
    await store.upload_file(f"{base_key}/dataset.parquet", dataset_path, use_write=True)
    await store.upload_file(f"{base_key}/manifest.json", manifest_path, use_write=True)
    for relative_uri, local_path in assets.items():
        await store.upload_file(
            f"{base_key}/{relative_uri.lstrip('/')}",
            local_path,
            use_write=True,
        )
    logger.info("uploaded interval package interval=%d records=%d", interval_id, len(records))
    return dataset_path, manifest_path
