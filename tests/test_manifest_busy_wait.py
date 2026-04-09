"""Tests for R2 manifest busy wait before pack upload."""

from __future__ import annotations

import asyncio

from nexis.miner.pending_pack import wait_until_r2_manifest_slot


def test_wait_until_slot_sleeps_until_manifest_missing() -> None:
    class Store:
        def __init__(self) -> None:
            self.n = 0

        async def object_exists(self, key: str) -> bool:
            self.n += 1
            return self.n < 3

    store = Store()

    async def run() -> int:
        return await wait_until_r2_manifest_slot(
            store=store,
            interval_id=100,
            wait_sec=0.01,
            interval_refresher=None,
        )

    result = asyncio.run(run())
    assert result == 100
    assert store.n == 3


def test_wait_skipped_when_wait_sec_zero() -> None:
    class Store:
        async def object_exists(self, key: str) -> bool:
            raise AssertionError("should not check when wait_sec is 0")

    async def run() -> int:
        return await wait_until_r2_manifest_slot(
            store=Store(),
            interval_id=42,
            wait_sec=0.0,
            interval_refresher=None,
        )

    assert asyncio.run(run()) == 42


def test_wait_refreshes_interval() -> None:
    class Store:
        def __init__(self) -> None:
            self.seen: list[int] = []

        async def object_exists(self, key: str) -> bool:
            if key == "10/manifest.json":
                self.seen.append(10)
                return True
            if key == "20/manifest.json":
                self.seen.append(20)
                return False
            raise AssertionError(key)

    store = Store()
    calls = {"n": 0}

    async def refresh() -> int:
        calls["n"] += 1
        return 20

    async def run() -> int:
        return await wait_until_r2_manifest_slot(
            store=store,
            interval_id=10,
            wait_sec=0.01,
            interval_refresher=refresh,
        )

    assert asyncio.run(run()) == 20
    assert calls["n"] == 1
    assert store.seen == [10, 20]
