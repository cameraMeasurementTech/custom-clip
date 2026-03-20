"""Metagraph-backed validator allowlist sync."""

from __future__ import annotations

import asyncio
import inspect
import logging
from collections.abc import Mapping
from datetime import datetime, timezone
from typing import Any

from ..chain.metagraph import _open_subtensor

logger = logging.getLogger(__name__)


async def _resolve_maybe_awaitable(value: Any) -> Any:
    if inspect.isawaitable(value):
        return await value
    return value


def extract_hotkeys_with_min_stake(
    *,
    metagraph: Any,
    min_stake: float,
) -> dict[str, float]:
    hotkeys = list(getattr(metagraph, "hotkeys", []))
    stakes = list(getattr(metagraph, "S", []))
    if not hotkeys or not stakes:
        return {}
    count = min(len(hotkeys), len(stakes))
    allowlist: dict[str, float] = {}
    for idx in range(count):
        hotkey = hotkeys[idx]
        if not isinstance(hotkey, str) or not hotkey:
            continue
        try:
            stake_value = float(stakes[idx])
        except Exception:
            continue
        if stake_value >= min_stake:
            allowlist[hotkey] = stake_value
    return allowlist


class ValidatorAllowlistCache:
    """Thread-safe validator hotkey/stake snapshot."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._hotkey_to_stake: dict[str, float] = {}
        self._updated_at: datetime | None = None

    async def replace(self, hotkey_to_stake: Mapping[str, float]) -> None:
        async with self._lock:
            self._hotkey_to_stake = {
                str(hotkey): float(stake) for hotkey, stake in hotkey_to_stake.items()
            }
            self._updated_at = datetime.now(timezone.utc)

    async def contains(self, hotkey: str) -> bool:
        async with self._lock:
            return hotkey in self._hotkey_to_stake

    async def snapshot(self) -> tuple[dict[str, float], datetime | None]:
        async with self._lock:
            return dict(self._hotkey_to_stake), self._updated_at


class MetagraphAllowlistSync:
    """Background refresher for validator allowlist cache."""

    def __init__(
        self,
        *,
        netuid: int,
        network: str,
        min_stake: float,
        refresh_sec: int,
        cache: ValidatorAllowlistCache,
    ):
        self._netuid = int(netuid)
        self._network = network
        self._min_stake = float(min_stake)
        self._refresh_sec = max(int(refresh_sec), 5)
        self._cache = cache
        self._task: asyncio.Task[None] | None = None
        self._stop = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None:
            return
        self._stop.clear()
        self._task = asyncio.create_task(self._run(), name="validator-allowlist-sync")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._stop.set()
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    async def refresh_once(self) -> dict[str, float]:
        async with _open_subtensor(self._network) as subtensor:
            metagraph = await _resolve_maybe_awaitable(subtensor.metagraph(self._netuid))
        allowlist = extract_hotkeys_with_min_stake(
            metagraph=metagraph,
            min_stake=self._min_stake,
        )
        await self._cache.replace(allowlist)
        return allowlist

    async def _run(self) -> None:
        while not self._stop.is_set():
            try:
                allowlist = await self.refresh_once()
                logger.info(
                    "validator allowlist refreshed count=%d min_stake=%.2f",
                    len(allowlist),
                    self._min_stake,
                )
            except Exception as exc:
                logger.warning("validator allowlist refresh failed: %s", exc)
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=float(self._refresh_sec))
            except asyncio.TimeoutError:
                continue

