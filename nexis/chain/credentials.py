"""Read credential commitment manager backed by on-chain commitments."""

from __future__ import annotations

import base64
from collections.abc import AsyncIterator
import logging
from pathlib import Path
from typing import Any

from .metagraph import _open_subtensor, _resolve_maybe_awaitable, _run_async
from ..storage.hippius import HippiusCredentials

logger = logging.getLogger(__name__)
_COMMITMENT_PREFIX = "nexis"


class ReadCredentialCommitmentManager:
    """Commits and fetches read credentials through chain commitments."""

    def __init__(
        self,
        *,
        netuid: int,
        network: str,
        wallet_name: str,
        wallet_hotkey: str,
        wallet_path: Path,
        hippius_endpoint_url: str,
        hippius_region: str,
    ):
        self.netuid = netuid
        self.network = network
        self.wallet_name = wallet_name
        self.wallet_hotkey = wallet_hotkey
        self.wallet_path = wallet_path
        self.hippius_endpoint_url = hippius_endpoint_url
        self.hippius_region = hippius_region

    def commit_read_credentials(self, hotkey: str, credentials: HippiusCredentials) -> str:
        return _run_async(self.commit_read_credentials_async(hotkey, credentials))

    async def commit_read_credentials_async(
        self,
        hotkey: str,
        credentials: HippiusCredentials,
        subtensor: Any | None = None,
    ) -> str:
        import bittensor as bt

        credentials.validate_bucket_name()
        commitment_payload = self._encode_payload(
            bucket_name=credentials.bucket_name,
            read_access_key=credentials.read_access_key,
            read_secret_key=credentials.read_secret_key,
        )
        if len(commitment_payload) > 128:
            raise ValueError(
                f"commitment payload too long ({len(commitment_payload)} chars); "
                "shorter read credentials are required"
            )

        if subtensor is None:
            async with _open_subtensor(self.network) as owned_subtensor:
                return await self.commit_read_credentials_async(
                    hotkey=hotkey,
                    credentials=credentials,
                    subtensor=owned_subtensor,
                )
        wallet = bt.wallet(
            name=self.wallet_name,
            hotkey=self.wallet_hotkey,
            path=str(self.wallet_path.expanduser()),
        )
        await _resolve_maybe_awaitable(subtensor.commit(wallet, self.netuid, commitment_payload))
        logger.info("committed read credentials on-chain for hotkey=%s", hotkey)
        return credentials.read_commitment

    def get_credentials_for_hotkey(self, hotkey: str) -> dict | None:
        payload = self.get_all_credentials()
        return payload.get(hotkey)

    def get_all_credentials(self) -> dict[str, dict]:
        return _run_async(self.get_all_credentials_async())

    async def get_all_credentials_async(self, subtensor: Any | None = None) -> dict[str, dict]:
        commitments: dict[str, dict] = {}
        try:
            if subtensor is None:
                async with _open_subtensor(self.network) as owned_subtensor:
                    return await self.get_all_credentials_async(subtensor=owned_subtensor)
            substrate = getattr(subtensor, "substrate", None)
            if substrate is None:
                return commitments
            query_result = await _resolve_maybe_awaitable(
                substrate.query_map(
                    module="Commitments",
                    storage_function="CommitmentOf",
                    params=[self.netuid],
                    block_hash=None,
                )
            )
            if query_result is None:
                return commitments

            async for key, value in self._iter_query_entries(query_result):
                hotkey = self._decode_hotkey(key)
                if not hotkey:
                    continue
                commitment_str = self._extract_commitment_string(value)
                if not commitment_str:
                    continue
                decoded = self._decode_payload(commitment_str)
                if decoded is None:
                    continue
                commitments[hotkey] = {
                    "bucket_name": decoded["bucket_name"],
                    "endpoint_url": self.hippius_endpoint_url,
                    "region": self.hippius_region,
                    "read_access_key": decoded["read_access_key"],
                    "read_secret_key": decoded["read_secret_key"],
                    "commitment": commitment_str,
                }
        except Exception as exc:
            logger.warning("failed to fetch chain commitments: %s", exc)
        return commitments

    def build_hippius_credentials(
        self,
        committed: dict | None,
    ) -> HippiusCredentials | None:
        if committed is None:
            return None
        bucket_name = str(committed.get("bucket_name", "")).strip()
        read_access_key = str(committed.get("read_access_key", "")).strip()
        read_secret_key = str(committed.get("read_secret_key", "")).strip()
        if not bucket_name or not read_access_key or not read_secret_key:
            return None
        # Validator path is read-only; write fields mirror read keys for compatibility.
        return HippiusCredentials(
            bucket_name=bucket_name,
            endpoint_url=self.hippius_endpoint_url,
            region=self.hippius_region,
            read_access_key=read_access_key,
            read_secret_key=read_secret_key,
            write_access_key=read_access_key,
            write_secret_key=read_secret_key,
        )

    def _encode_payload(self, *, bucket_name: str, read_access_key: str, read_secret_key: str) -> str:
        bucket = self._b64encode(bucket_name)
        key = self._b64encode(read_access_key)
        secret = self._b64encode(read_secret_key)
        return f"{_COMMITMENT_PREFIX}|{bucket}|{key}|{secret}"

    def _decode_payload(self, payload: str) -> dict[str, str] | None:
        if payload.startswith(f"{_COMMITMENT_PREFIX}|"):
            parts = payload.split("|", 3)
            if len(parts) != 4:
                return None
            bucket_name = self._b64decode(parts[1])
            read_access_key = self._b64decode(parts[2])
            read_secret_key = self._b64decode(parts[3])
            if bucket_name is None or read_access_key is None or read_secret_key is None:
                return None
            return {
                "bucket_name": bucket_name,
                "read_access_key": read_access_key,
                "read_secret_key": read_secret_key,
            }
        else:
            return None

    def _b64encode(self, value: str) -> str:
        return base64.urlsafe_b64encode(value.encode("utf-8")).decode("ascii").rstrip("=")

    def _b64decode(self, value: str) -> str | None:
        try:
            padded = value + ("=" * (-len(value) % 4))
            return base64.urlsafe_b64decode(padded.encode("ascii")).decode("utf-8")
        except Exception:
            return None

    def _decode_hotkey(self, key: Any) -> str | None:
        try:
            if isinstance(key, (list, tuple)) and key:
                from bittensor.core.chain_data import decode_account_id

                return decode_account_id(key[0])
        except Exception:
            return None
        return None

    def _extract_commitment_string(self, value: Any) -> str | None:
        try:
            payload = getattr(value, "value", value)
            fields = payload.get("info", {}).get("fields", [])
            if not fields:
                return None
            encoded_map = fields[0][0]
            if not isinstance(encoded_map, dict) or not encoded_map:
                return None
            encoded_value = encoded_map[next(iter(encoded_map.keys()))]
            if isinstance(encoded_value, (list, tuple)) and encoded_value:
                first = encoded_value[0]
                if isinstance(first, (list, tuple)):
                    return bytes(first).decode("utf-8")
                if isinstance(first, (bytes, bytearray)):
                    return bytes(first).decode("utf-8")
            if isinstance(encoded_value, str):
                return encoded_value
        except Exception:
            return None
        return None

    async def _iter_query_entries(self, query_result: Any) -> AsyncIterator[tuple[Any, Any]]:
        if hasattr(query_result, "__aiter__"):
            async for key, value in query_result:
                yield key, value
            return
        for key, value in query_result:
            yield key, value

