"""Validator-side HTTP reporter for signed interval decisions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import secrets
import time
from dataclasses import dataclass
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

from ..models import ValidationDecision

logger = logging.getLogger(__name__)


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_auth_message(
    *,
    method: str,
    path: str,
    body_sha256: str,
    timestamp: int,
    nonce: str,
) -> bytes:
    raw = f"{method.upper()}|{path}|{body_sha256}|{timestamp}|{nonce}"
    return raw.encode("utf-8")


def decision_to_payload(decision: ValidationDecision) -> dict[str, Any]:
    notes = decision.notes or {}
    return {
        "miner_hotkey": decision.miner_hotkey,
        "accepted": bool(decision.accepted),
        "failures": list(decision.failures),
        "record_count": int(decision.record_count),
        "global_overlap_pruned_count": int(notes.get("global_overlap_pruned_count", 0) or 0),
        "cross_miner_overlap_pruned_count": int(
            notes.get("cross_miner_overlap_pruned_count", 0) or 0
        ),
    }


def build_interval_payload(interval_id: int, decisions: list[ValidationDecision]) -> bytes:
    payload = {
        "interval_id": int(interval_id),
        "decisions": [decision_to_payload(item) for item in decisions],
    }
    # Compact JSON keeps body hash deterministic.
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


@dataclass
class ValidationResultReporter:
    endpoint_url: str
    hotkey_ss58: str
    hotkey_signer: Any
    timeout_sec: float = 10.0

    async def report_interval(
        self,
        *,
        interval_id: int,
        decisions: list[ValidationDecision],
    ) -> bool:
        if not self.endpoint_url or not decisions:
            return False

        body = build_interval_payload(interval_id, decisions)
        timestamp = int(time.time())
        nonce = secrets.token_hex(16)
        body_sha256 = _sha256_hex(body)
        endpoint_path = self._endpoint_path()
        message = build_auth_message(
            method="POST",
            path=endpoint_path,
            body_sha256=body_sha256,
            timestamp=timestamp,
            nonce=nonce,
        )
        signature = self.hotkey_signer.sign(data=message).hex()
        headers = {
            "Content-Type": "application/json",
            "X-Validator-Hotkey": self.hotkey_ss58,
            "X-Signature": signature,
            "X-Timestamp": str(timestamp),
            "X-Nonce": nonce,
        }
        try:
            status_code = await asyncio.to_thread(self._post_sync, body, headers)
            if status_code < 200 or status_code >= 300:
                logger.warning(
                    "validation evidence POST failed interval=%d status=%d",
                    interval_id,
                    status_code,
                )
                return False
            logger.info(
                "validation evidence submitted interval=%d decisions=%d",
                interval_id,
                len(decisions),
            )
            return True
        except Exception as exc:
            logger.warning("validation evidence report failed interval=%d error=%s", interval_id, exc)
            return False

    def _endpoint_path(self) -> str:
        try:
            from urllib.parse import urlparse

            parsed = urlparse(self.endpoint_url)
            return parsed.path or "/v1/validation-results"
        except Exception:
            return "/v1/validation-results"

    def _post_sync(self, body: bytes, headers: dict[str, str]) -> int:
        req = urllib_request.Request(
            self.endpoint_url,
            data=body,
            headers=headers,
            method="POST",
        )
        try:
            with urllib_request.urlopen(req, timeout=float(self.timeout_sec)) as response:
                return int(getattr(response, "status", 200))
        except urllib_error.HTTPError as exc:
            return int(exc.code)

