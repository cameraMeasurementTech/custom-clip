from __future__ import annotations

from nexis.chain.metagraph import (
    _build_subtensor,
    fetch_current_block_async,
    fetch_hotkeys_from_metagraph_async,
)
from .helpers import FakeMetagraphSubtensor, patch_bittensor_async_subtensor, run_async


def test_build_subtensor_uses_network(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    calls: list[str] = []

    def fake_subtensor(*, network: str) -> dict[str, str]:
        calls.append(network)
        return {"network": network}

    patch_bittensor_async_subtensor(monkeypatch, fake_subtensor)

    result = _build_subtensor("test")

    assert calls == ["test"]
    assert result == {"network": "test"}


def test_fetch_hotkeys_from_metagraph_async_uses_provided_subtensor() -> None:
    hotkeys = run_async(
        fetch_hotkeys_from_metagraph_async(
            netuid=42,
            network="test",
            subtensor=FakeMetagraphSubtensor(
                hotkeys=["hk1", "", 123, "hk2"],  # type: ignore[list-item]
                expected_netuid=42,
            ),
        )
    )

    assert hotkeys == ["hk1", "hk2"]


def test_fetch_current_block_async_uses_provided_subtensor() -> None:
    block = run_async(
        fetch_current_block_async(
            network="test",
            subtensor=FakeMetagraphSubtensor(block=123456),
        )
    )

    assert block == 123456
