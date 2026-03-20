from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from nexis.api.auth import RequestAuthenticator, build_auth_message
from .helpers import run_async


class _FakeAllowlistCache:
    def __init__(self, allowed: set[str]):
        self._allowed = allowed

    async def contains(self, hotkey: str) -> bool:
        return hotkey in self._allowed


class _FakeRepository:
    def __init__(self):
        self._seen: set[tuple[str, str]] = set()

    async def register_nonce_once(
        self,
        *,
        validator_hotkey: str,
        nonce: str,
        signature_timestamp: int,
        max_age_sec: int,
    ) -> bool:
        _ = signature_timestamp
        _ = max_age_sec
        key = (validator_hotkey, nonce)
        if key in self._seen:
            return False
        self._seen.add(key)
        return True


def test_build_auth_message_layout() -> None:
    msg = build_auth_message(
        method="post",
        path="/v1/validation-results",
        body_sha256="abc123",
        timestamp=123,
        nonce="n1",
    )
    assert msg == b"POST|/v1/validation-results|abc123|123|n1"


def test_authenticator_rejects_replayed_nonce(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = int(time.time())
    cache = _FakeAllowlistCache({"hk-ok"})
    repo = _FakeRepository()
    auth = RequestAuthenticator(
        allowlist_cache=cache,  # type: ignore[arg-type]
        repository=repo,  # type: ignore[arg-type]
        max_time_skew_sec=120,
        nonce_max_age_sec=600,
    )

    monkeypatch.setattr("nexis.api.auth.verify_hotkey_signature", lambda **_kwargs: True)

    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/v1/validation-results"),
        headers={
            "x-validator-hotkey": "hk-ok",
            "x-signature": "abcd",
            "x-timestamp": str(now),
            "x-nonce": "nonce-1",
        },
    )

    first = run_async(auth.authenticate(request, b'{"interval_id":0}'))
    assert first.validator_hotkey == "hk-ok"

    with pytest.raises(HTTPException) as replay_err:
        run_async(auth.authenticate(request, b'{"interval_id":0}'))
    assert replay_err.value.status_code == 409


def test_authenticator_rejects_hotkey_not_allowlisted(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    now = int(time.time())
    auth = RequestAuthenticator(
        allowlist_cache=_FakeAllowlistCache(set()),  # type: ignore[arg-type]
        repository=_FakeRepository(),  # type: ignore[arg-type]
        max_time_skew_sec=120,
        nonce_max_age_sec=600,
    )
    monkeypatch.setattr("nexis.api.auth.verify_hotkey_signature", lambda **_kwargs: True)
    request = SimpleNamespace(
        method="POST",
        url=SimpleNamespace(path="/v1/validation-results"),
        headers={
            "x-validator-hotkey": "hk-missing",
            "x-signature": "abcd",
            "x-timestamp": str(now),
            "x-nonce": "nonce-x",
        },
    )
    with pytest.raises(HTTPException) as err:
        run_async(auth.authenticate(request, b"{}"))
    assert err.value.status_code == 403

