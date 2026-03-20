"""Parquet + manifest serialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq

from .models import ClipRecord, IntervalManifest


def write_dataset_parquet(records: Iterable[ClipRecord], output_path: Path) -> None:
    rows = [r.model_dump(mode="python") for r in records]
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

