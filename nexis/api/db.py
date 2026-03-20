"""Async Postgres database helpers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import asyncpg


class Database:
    """Thin asyncpg pool wrapper used by API repository."""

    def __init__(self, dsn: str):
        self._dsn = dsn
        self._pool: asyncpg.Pool | None = None

    async def connect(self) -> None:
        if self._pool is not None:
            return
        self._pool = await asyncpg.create_pool(dsn=self._dsn, min_size=1, max_size=10)

    async def close(self) -> None:
        if self._pool is None:
            return
        await self._pool.close()
        self._pool = None

    async def execute(self, query: str, *args: Any) -> str:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def execute_many(self, query: str, args_list: Sequence[Sequence[Any]]) -> None:
        if not args_list:
            return
        pool = self._require_pool()
        async with pool.acquire() as conn:
            await conn.executemany(query, args_list)

    async def fetch(self, query: str, *args: Any) -> list[asyncpg.Record]:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        pool = self._require_pool()
        async with pool.acquire() as conn:
            return await conn.fetchval(query, *args)

    def _require_pool(self) -> asyncpg.Pool:
        if self._pool is None:
            raise RuntimeError("database pool is not initialized")
        return self._pool

