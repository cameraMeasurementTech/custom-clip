"""Parquet + manifest serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .models import ClipRecord, IntervalManifest
from .protocol import MIN_CAPTION_WORDS

_cliprecord_empty_parquet_schema: pa.Schema | None = None


def _get_cliprecord_parquet_schema() -> pa.Schema:
    """Schema for 0-row ClipRecord Parquet (validator must read empty datasets)."""
    global _cliprecord_empty_parquet_schema
    if _cliprecord_empty_parquet_schema is None:
        cap = " ".join(["word"] * MIN_CAPTION_WORDS)
        ref = ClipRecord(
            clip_id="0" * 24,
            clip_uri="clips/000000000000000000000000.mp4",
            clip_sha256="0" * 64,
            first_frame_uri="frames/000000000000000000000000.jpg",
            first_frame_sha256="0" * 64,
            source_video_id="x",
            split_group_id="x:0",
            split="train",
            clip_start_sec=0.0,
            duration_sec=5.0,
            width=1280,
            height=720,
            fps=30.0,
            num_frames=150,
            has_audio=True,
            caption=cap,
            source_video_url="https://youtube.com/watch?v=x",
            source_proof={"extractor": "yt-dlp"},
        ).model_dump(mode="python")
        _cliprecord_empty_parquet_schema = pa.Table.from_pylist([ref]).schema
    return _cliprecord_empty_parquet_schema


def write_dataset_parquet(records: Iterable[ClipRecord], output_path: Path) -> None:
    rows = [r.model_dump(mode="python") for r in records]
    if not rows:
        schema = _get_cliprecord_parquet_schema()
        table = pa.Table.from_pylist([], schema=schema)
        pq.write_table(table, output_path)
        return
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, output_path)


def read_dataset_parquet(path: Path) -> list[ClipRecord]:
    table = pq.read_table(path)
    rows = table.to_pylist()
    return [ClipRecord.model_validate(row) for row in rows]


def read_dataset_parquet_as_model(path: Path, row_model: type) -> list:
    table = pq.read_table(path)
    rows = table.to_pylist()
    return [row_model.model_validate(row) for row in rows]


def write_manifest(manifest: IntervalManifest, output_path: Path) -> None:
    output_path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")


def read_manifest(path: Path) -> IntervalManifest:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return IntervalManifest.model_validate(payload)
