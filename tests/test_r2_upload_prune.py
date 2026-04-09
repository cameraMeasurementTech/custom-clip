"""Tests for R2 interval-prefix pruning after successful uploads."""

from __future__ import annotations

import asyncio
from pathlib import Path

from nexis.miner.pending_pack import (
    R2_UPLOAD_INTERVAL_HISTORY_JSON,
    load_r2_upload_interval_history,
    maybe_prune_old_r2_interval_prefix,
)


def test_prune_r2_deletes_prefix_from_two_uploads_ago(tmp_path: Path) -> None:
    class FakeStore:
        def __init__(self) -> None:
            self.calls: list[str] = []

        async def delete_prefix(self, prefix: str, *, use_write: bool = True) -> int:
            self.calls.append(prefix)
            return 1

    store = FakeStore()
    wd = tmp_path

    async def run() -> None:
        await maybe_prune_old_r2_interval_prefix(
            store=store,
            workdir=wd,
            interval_id=10,
            enabled=True,
            uploads_ago=2,
        )
        await maybe_prune_old_r2_interval_prefix(
            store=store,
            workdir=wd,
            interval_id=20,
            enabled=True,
            uploads_ago=2,
        )
        await maybe_prune_old_r2_interval_prefix(
            store=store,
            workdir=wd,
            interval_id=30,
            enabled=True,
            uploads_ago=2,
        )

    asyncio.run(run())

    assert store.calls == ["10"]
    assert load_r2_upload_interval_history(wd) == [20, 30]
    assert (wd / R2_UPLOAD_INTERVAL_HISTORY_JSON).is_file()
