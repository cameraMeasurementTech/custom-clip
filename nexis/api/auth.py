"""Request authentication for validation evidence API."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass

from fastapi import HTTPException, Request, status

from .metagraph_sync import ValidatorAllowlistCache
from .repository import ValidationEvidenceRepository

logger = logging.getLogger(__name__)

HEADER_VALIDATOR_HOTKEY = "x-validator-hotkey"
HEADER_SIGNATURE = "x-signature"
HEADER_TIMESTAMP = "x-timestamp"
HEADER_NONCE = "x-nonce"


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_auth_message(
    *,
    method: str,
    path: str,
    body_sha256: str,
    timestamp: int,
    nonce: str,
) -> bytes:
    payload = f"{method.upper()}|{path}|{body_sha256}|{timestamp}|{nonce}"
    return payload.encode("utf-8")


def verify_hotkey_signature(
    *,
    hotkey: str,
    signature_hex: str,
    message: bytes,
) -> bool:
    try:
        import bittensor as bt

        clean_signature = signature_hex.removeprefix("0x").removeprefix("0X")
        signature = bytes.fromhex(clean_signature)
        keypair = bt.Keypair(ss58_address=hotkey)
        return bool(keypair.verify(data=message, signature=signature))
    except Exception:
        return False


@dataclass
class AuthContext:
    validator_hotkey: str
    signature: str
    timestamp: int
    nonce: str
    body_sha256: str


class RequestAuthenticator:
    """Validate signed headers and replay protection."""

    def __init__(
        self,
        *,
        allowlist_cache: ValidatorAllowlistCache,
        repository: ValidationEvidenceRepository,
        max_time_skew_sec: int,
        nonce_max_age_sec: int,
    ):
        self._allowlist_cache = allowlist_cache
        self._repository = repository
        self._max_time_skew_sec = max(int(max_time_skew_sec), 1)
        self._nonce_max_age_sec = max(int(nonce_max_age_sec), 1)

    async def authenticate(self, request: Request, body: bytes) -> AuthContext:
        hotkey = request.headers.get(HEADER_VALIDATOR_HOTKEY, "").strip()
        signature = request.headers.get(HEADER_SIGNATURE, "").strip()
        timestamp_raw = request.headers.get(HEADER_TIMESTAMP, "").strip()
        nonce = request.headers.get(HEADER_NONCE, "").strip()
        if not hotkey or not signature or not timestamp_raw or not nonce:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="missing required authentication headers",
            )

        try:
            timestamp = int(timestamp_raw)
        except ValueError as exc:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="invalid timestamp header",
            ) from exc

        now_sec = int(time.time())
        skew = abs(now_sec - timestamp)
        if skew > self._max_time_skew_sec:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="request timestamp outside allowed window",
            )

        if not await self._allowlist_cache.contains(hotkey):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="validator hotkey is not in allowlist",
            )

        body_hash = sha256_hex(body)
        message = build_auth_message(
            method=request.method,
            path=request.url.path,
            body_sha256=body_hash,
            timestamp=timestamp,
            nonce=nonce,
        )
        if not verify_hotkey_signature(
            hotkey=hotkey,
            signature_hex=signature,
            message=message,
        ):
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="signature verification failed",
            )

        nonce_inserted = await self._repository.register_nonce_once(
            validator_hotkey=hotkey,
            nonce=nonce,
            signature_timestamp=timestamp,
            max_age_sec=self._nonce_max_age_sec,
        )
        if not nonce_inserted:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="replayed request nonce",
            )
        return AuthContext(
            validator_hotkey=hotkey,
            signature=signature,
            timestamp=timestamp,
            nonce=nonce,
            body_sha256=body_hash,
        )

