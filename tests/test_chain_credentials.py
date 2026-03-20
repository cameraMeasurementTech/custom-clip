from __future__ import annotations

from pathlib import Path

from nexis.chain.credentials import ReadCredentialCommitmentManager
from nexis.storage.hippius import HippiusCredentials
from .helpers import patch_bittensor_wallet, run_async


def _manager() -> ReadCredentialCommitmentManager:
    return ReadCredentialCommitmentManager(
        netuid=1,
        network="finney",
        wallet_name="w",
        wallet_hotkey="h",
        wallet_path=Path("~/.bittensor/wallets"),
        hippius_endpoint_url="https://s3.hippius.com",
        hippius_region="decentralized",
    )


def test_payload_v2_roundtrip_includes_bucket_name() -> None:
    manager = _manager()
    payload = manager._encode_payload(  # type: ignore[attr-defined]
        bucket_name="my-custom-bucket",
        read_access_key="read-ak",
        read_secret_key="read-sk",
    )
    decoded = manager._decode_payload(payload)  # type: ignore[attr-defined]
    assert decoded is not None
    assert decoded["bucket_name"] == "my-custom-bucket"
    assert decoded["read_access_key"] == "read-ak"
    assert decoded["read_secret_key"] == "read-sk"


def test_invalid_payload_decode_returns_none() -> None:
    manager = _manager()
    v1_payload = "nexis1|cmVhZC1haw|cmVhZC1zaw"
    decoded = manager._decode_payload(v1_payload)  # type: ignore[attr-defined]
    assert decoded is None


def test_build_credentials_uses_committed_bucket_name() -> None:
    manager = _manager()
    committed = {
        "bucket_name": "team-bucket-42",
        "read_access_key": "ak",
        "read_secret_key": "sk",
    }
    creds = manager.build_hippius_credentials(committed)
    assert creds is not None
    assert creds.bucket_name == "team-bucket-42"
    assert creds.write_access_key == "ak"
    assert creds.write_secret_key == "sk"


def test_build_credentials_requires_bucket_name() -> None:
    manager = _manager()
    committed = {
        "bucket_name": "",
        "read_access_key": "ak",
        "read_secret_key": "sk",
    }
    assert manager.build_hippius_credentials(committed) is None


def test_get_all_credentials_async_uses_provided_subtensor() -> None:
    manager = _manager()
    payload = manager._encode_payload(  # type: ignore[attr-defined]
        bucket_name="bucket-1",
        read_access_key="ak",
        read_secret_key="sk",
    )

    manager._decode_hotkey = lambda key: str(key[0])  # type: ignore[method-assign]
    manager._extract_commitment_string = lambda value: str(value)  # type: ignore[method-assign]

    class FakeSubstrate:
        async def query_map(self, **kwargs) -> list[tuple[tuple[str], str]]:  # type: ignore[no-untyped-def]
            assert kwargs["module"] == "Commitments"
            assert kwargs["storage_function"] == "CommitmentOf"
            return [(("hk1",), payload)]

    class FakeSubtensor:
        substrate = FakeSubstrate()

    committed = run_async(manager.get_all_credentials_async(subtensor=FakeSubtensor()))
    assert "hk1" in committed
    assert committed["hk1"]["bucket_name"] == "bucket-1"
    assert committed["hk1"]["read_access_key"] == "ak"
    assert committed["hk1"]["read_secret_key"] == "sk"


def test_commit_read_credentials_async_uses_provided_subtensor(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    manager = _manager()
    wallet_calls = patch_bittensor_wallet(monkeypatch)
    commit_calls: list[tuple[object, int, str]] = []

    class FakeSubtensor:
        async def commit(self, wallet: object, netuid: int, payload: str) -> None:
            commit_calls.append((wallet, netuid, payload))

    creds = HippiusCredentials(
        bucket_name="bucket-2",
        endpoint_url="https://s3.example.com",
        region="region-1",
        read_access_key="read-ak",
        read_secret_key="read-sk",
        write_access_key="write-ak",
        write_secret_key="write-sk",
    )

    result = run_async(
        manager.commit_read_credentials_async(
            "hk-test",
            creds,
            subtensor=FakeSubtensor(),
        )
    )

    assert result == creds.read_commitment
    assert len(wallet_calls) == 1
    assert len(commit_calls) == 1
    assert commit_calls[0][1] == manager.netuid
