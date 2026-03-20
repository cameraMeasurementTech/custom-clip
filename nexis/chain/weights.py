"""Chain weight submission helpers for validators."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .metagraph import _open_subtensor, _resolve_maybe_awaitable, _run_async
import logging

logger = logging.getLogger(__name__)

@dataclass(frozen=True)
class ChainWeightPayload:
    """Dense UID-aligned weight payload for on-chain submission."""

    uids: list[int]
    weights: list[float]
    unknown_hotkeys: list[str]


@dataclass(frozen=True)
class WeightSubmissionResult:
    """Result metadata for validator weight submission."""

    submitted: bool
    reason: str = ""
    unknown_hotkeys: list[str] | None = None


def build_chain_weight_payload(
    *,
    metagraph_hotkeys: Iterable[str],
    metagraph_uids: Iterable[int],
    weights_by_hotkey: dict[str, float],
) -> ChainWeightPayload:
    """Build dense UID-aligned weights from hotkey-indexed weights."""
    hotkeys = list(metagraph_hotkeys)
    uids = [int(uid) for uid in metagraph_uids]
    uid_by_hotkey = dict(zip(hotkeys, uids, strict=True))

    unknown_hotkeys: list[str] = []
    dense: dict[int, float] = {uid: 0.0 for uid in uids}
    for hotkey, weight in weights_by_hotkey.items():
        uid = uid_by_hotkey.get(hotkey)
        if uid is None:
            unknown_hotkeys.append(hotkey)
            continue
        dense[uid] = max(0.0, float(weight))

    total = sum(dense.values())
    if total > 0:
        dense = {uid: value / total for uid, value in dense.items()}
    elif uids:
        # Required fallback: if no valid miner weight exists, route full weight to UID 0.
        target_uid = 0 if 0 in dense else uids[0]
        dense = {uid: 0.0 for uid in uids}
        dense[target_uid] = 1.0

    ordered_weights = [dense[uid] for uid in uids]
    return ChainWeightPayload(uids=uids, weights=ordered_weights, unknown_hotkeys=unknown_hotkeys)


async def submit_weights_to_chain_async(
    *,
    netuid: int,
    network: str,
    wallet_name: str,
    wallet_hotkey: str,
    wallet_path: Path,
    weights_by_hotkey: dict[str, float],
    subtensor: object | None = None,
) -> WeightSubmissionResult:
    """Submit validator-computed weights using bittensor set_weights."""
    import bittensor as bt

    if subtensor is None:
        async with _open_subtensor(network) as owned_subtensor:
            return await submit_weights_to_chain_async(
                netuid=netuid,
                network=network,
                wallet_name=wallet_name,
                wallet_hotkey=wallet_hotkey,
                wallet_path=wallet_path,
                weights_by_hotkey=weights_by_hotkey,
                subtensor=owned_subtensor,
            )
    active_subtensor = subtensor
    if active_subtensor is None:
        return WeightSubmissionResult(submitted=False, reason="subtensor_unavailable")
    metagraph = await _resolve_maybe_awaitable(active_subtensor.metagraph(netuid))
    payload = build_chain_weight_payload(
        metagraph_hotkeys=list(metagraph.hotkeys),
        metagraph_uids=list(metagraph.uids),
        weights_by_hotkey=weights_by_hotkey,
    )
    if not payload.uids:
        return WeightSubmissionResult(
            submitted=False,
            reason="empty_metagraph",
            unknown_hotkeys=payload.unknown_hotkeys,
        )

    wallet = bt.wallet(
        name=wallet_name,
        hotkey=wallet_hotkey,
        path=str(wallet_path.expanduser()),
    )
    logger.info(f"submitting weights to chain: {payload.uids} {payload.weights}")
    attempt = 0
    submitted = False
    while attempt < 3:
        result = await _resolve_maybe_awaitable(
            active_subtensor.set_weights(
                wallet=wallet,
                netuid=netuid,
                uids=payload.uids,
                weights=payload.weights,
                wait_for_inclusion=True,
            )
        )

        submitted = True
        if isinstance(result, tuple):
            submitted = bool(result[0])
        elif isinstance(result, bool):
            submitted = result
        if submitted:
            break
        logger.error(f"set_weights failed: {result} on attempt {attempt}")
        attempt += 1
        if attempt < 3:
            await asyncio.sleep(10)

    return WeightSubmissionResult(
        submitted=submitted,
        reason="" if submitted else "set_weights_returned_false",
        unknown_hotkeys=payload.unknown_hotkeys,
    )


def submit_weights_to_chain(
    *,
    netuid: int,
    network: str,
    wallet_name: str,
    wallet_hotkey: str,
    wallet_path: Path,
    weights_by_hotkey: dict[str, float],
) -> WeightSubmissionResult:
    return _run_async(
        submit_weights_to_chain_async(
            netuid=netuid,
            network=network,
            wallet_name=wallet_name,
            wallet_hotkey=wallet_hotkey,
            wallet_path=wallet_path,
            weights_by_hotkey=weights_by_hotkey,
        )
    )

