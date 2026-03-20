from __future__ import annotations

from pathlib import Path

from nexis.chain.weights import build_chain_weight_payload, submit_weights_to_chain_async
from .helpers import FakeWeightSubtensor, patch_bittensor_wallet, run_async


def test_build_chain_weight_payload_maps_hotkeys_to_uids_and_normalizes() -> None:
    payload = build_chain_weight_payload(
        metagraph_hotkeys=["hk0", "hk1", "hk2"],
        metagraph_uids=[10, 11, 12],
        weights_by_hotkey={"hk1": 3.0, "hk2": 1.0},
    )

    assert payload.uids == [10, 11, 12]
    assert payload.unknown_hotkeys == []
    assert payload.weights == [0.0, 0.75, 0.25]


def test_build_chain_weight_payload_tracks_unknown_hotkeys() -> None:
    payload = build_chain_weight_payload(
        metagraph_hotkeys=["hk0", "hk1"],
        metagraph_uids=[1, 2],
        weights_by_hotkey={"hk1": 1.0, "missing_hotkey": 2.0},
    )

    assert payload.uids == [1, 2]
    assert payload.weights == [0.0, 1.0]
    assert payload.unknown_hotkeys == ["missing_hotkey"]


def test_build_chain_weight_payload_falls_back_to_uid_zero_when_all_zero() -> None:
    payload = build_chain_weight_payload(
        metagraph_hotkeys=["hk0", "hk1", "hk2"],
        metagraph_uids=[0, 1, 2],
        weights_by_hotkey={},
    )

    assert payload.uids == [0, 1, 2]
    assert payload.weights == [1.0, 0.0, 0.0]


def test_build_chain_weight_payload_falls_back_to_first_uid_if_zero_missing() -> None:
    payload = build_chain_weight_payload(
        metagraph_hotkeys=["hk10", "hk11"],
        metagraph_uids=[10, 11],
        weights_by_hotkey={},
    )

    assert payload.uids == [10, 11]
    assert payload.weights == [1.0, 0.0]


def test_submit_weights_to_chain_async_uses_provided_subtensor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    wallet_calls = patch_bittensor_wallet(monkeypatch)
    fake_subtensor = FakeWeightSubtensor(
        hotkeys=["hk0", "hk1"],
        uids=[0, 1],
        expected_netuid=7,
    )

    result = run_async(
        submit_weights_to_chain_async(
            netuid=7,
            network="test",
            wallet_name="w",
            wallet_hotkey="hk",
            wallet_path=Path("/tmp/wallet"),
            weights_by_hotkey={"hk1": 2.0},
            subtensor=fake_subtensor,
        )
    )

    assert result.submitted is True
    assert result.unknown_hotkeys == []
    assert len(wallet_calls) == 1
    assert len(fake_subtensor.set_weight_calls) == 1
    assert fake_subtensor.set_weight_calls[0]["uids"] == [0, 1]
    assert fake_subtensor.set_weight_calls[0]["weights"] == [0.0, 1.0]


def test_submit_weights_to_chain_async_retries_and_succeeds(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    sleep_calls: list[float] = []
    fake_subtensor = FakeWeightSubtensor(
        hotkeys=["hk0", "hk1"],
        uids=[0, 1],
        expected_netuid=7,
        set_weight_results=[False, (False, "temporary_error"), True],
    )

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    _ = patch_bittensor_wallet(monkeypatch)
    monkeypatch.setattr("nexis.chain.weights.asyncio.sleep", fake_sleep)

    result = run_async(
        submit_weights_to_chain_async(
            netuid=7,
            network="test",
            wallet_name="w",
            wallet_hotkey="hk",
            wallet_path=Path("/tmp/wallet"),
            weights_by_hotkey={"hk1": 2.0},
            subtensor=fake_subtensor,
        )
    )

    assert result.submitted is True
    assert result.reason == ""
    assert len(fake_subtensor.set_weight_calls) == 3
    assert sleep_calls == [10, 10]


def test_submit_weights_to_chain_async_retries_and_fails(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    sleep_calls: list[float] = []
    fake_subtensor = FakeWeightSubtensor(
        hotkeys=["hk0", "hk1"],
        uids=[0, 1],
        expected_netuid=7,
        set_weight_results=[False, False, False],
    )

    async def fake_sleep(seconds: float) -> None:
        sleep_calls.append(seconds)

    _ = patch_bittensor_wallet(monkeypatch)
    monkeypatch.setattr("nexis.chain.weights.asyncio.sleep", fake_sleep)

    result = run_async(
        submit_weights_to_chain_async(
            netuid=7,
            network="test",
            wallet_name="w",
            wallet_hotkey="hk",
            wallet_path=Path("/tmp/wallet"),
            weights_by_hotkey={"hk1": 2.0},
            subtensor=fake_subtensor,
        )
    )

    assert result.submitted is False
    assert result.reason == "set_weights_returned_false"
    assert len(fake_subtensor.set_weight_calls) == 3
    assert sleep_calls == [10, 10]


def test_submit_weights_to_chain_async_returns_empty_metagraph(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    fake_subtensor = FakeWeightSubtensor(
        hotkeys=[],
        uids=[],
        expected_netuid=7,
    )
    _ = patch_bittensor_wallet(monkeypatch)

    result = run_async(
        submit_weights_to_chain_async(
            netuid=7,
            network="test",
            wallet_name="w",
            wallet_hotkey="hk",
            wallet_path=Path("/tmp/wallet"),
            weights_by_hotkey={"hk1": 1.0},
            subtensor=fake_subtensor,
        )
    )

    assert result.submitted is False
    assert result.reason == "empty_metagraph"
    assert result.unknown_hotkeys == ["hk1"]
    assert fake_subtensor.set_weight_calls == []

