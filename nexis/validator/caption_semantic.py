"""Semantic caption-to-frame validation for sampled rows."""

from __future__ import annotations

import base64
import json
import logging
import re
import time
from pathlib import Path

from ..models import ClipRecord

logger = logging.getLogger(__name__)

# Miner caption prompts import this so generation targets the same bar as validation (keep in sync with
# ``_judge_match``).
SEMANTIC_JUDGE_CRITERIA_FOR_MINER = (
    "The judge returns JSON {\"match\": true} or {\"match\": false} using ONLY these rules:\n"
    "Return false if:\n"
    "- any part of the caption is contradicted by the frames\n"
    "- any important detail is not visually supported\n"
    "- the caption is overly generic and fails to capture the main visible subject or action\n"
    "- the caption includes speculation or inference beyond the frames\n"
    "Return true only if the caption is both visually supported and specific enough to describe the clip's "
    "main content.\n"
    "The images are timeline-sampled from the same short clip (time order left-to-right in the request); "
    "every substantive caption claim must be supported across what they show together."
)

_PROMPT_INJECTION_CAPTION_RE = re.compile(r"\b(?:match|true)\b", re.IGNORECASE)
_TRANSIENT_LLM_STATUS_CODES = {408, 409, 425, 429, 500, 502, 503, 504}
_TRANSIENT_LLM_ERROR_HINTS = (
    "timeout",
    "timed out",
    "rate limit",
    "too many requests",
    "temporarily unavailable",
    "internal server error",
    "service unavailable",
    "bad gateway",
    "gateway timeout",
    "connection error",
    "connection reset",
)


class _TransientLLMError(Exception):
    """Transient LLM server-side issue; caller may rotate keys / retry sweeps."""


class RateLimitSemanticError(Exception):
    """Every API key in a sweep returned 429 / TPM — clip fails after rotation with no successful call."""

    pass


def _retry_delay_from_message(message: str, attempt: int, base_sleep_sec: float, cap_sec: float) -> float:
    """Prefer provider-suggested wait (e.g. OpenAI 429 body), else exponential backoff."""
    m = re.search(r"try again in ([0-9.]+)\s*s", message, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 0.25, cap_sec)
    return min(base_sleep_sec * (2**attempt), cap_sec)


def _is_rate_limit_message(msg: str) -> bool:
    """Detect 429 / TPM from exception text (including wrapped ``_TransientLLMError``)."""
    lowered = msg.lower()
    if "error code: 429" in lowered or "status code: 429" in lowered:
        return True
    if "rate_limit_exceeded" in lowered.replace(" ", ""):
        return True
    if "tokens per min" in lowered or "(tpm)" in lowered:
        return True
    if "rate limit reached" in lowered:
        return True
    if "too many requests" in lowered and "429" in lowered:
        return True
    return False


def _is_transient_llm_error(error: Exception) -> bool:
    """Includes 429 — rotate to next key; ``_judge_with_key_failover`` may fail if all keys 429."""
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in _TRANSIENT_LLM_STATUS_CODES:
        return True
    lowered = str(error).lower()
    return any(hint in lowered for hint in _TRANSIENT_LLM_ERROR_HINTS)


_OPENAI_CLIENT_MAX_RETRIES = 0


class CaptionSemanticChecker:
    """Optional semantic checker using OpenAI-compatible vision APIs.

    Transient errors (429, 5xx, timeouts): try the **next API key immediately** (no sleep between keys).
    If **every key** in a sweep returns **429 / TPM**, the clip fails immediately with
    ``caption_semantic_rate_limited`` (no extra sweeps).
    Otherwise, after a full pass through all keys, **sleep**, then retry from the **first** key.
    After ``max_key_rotation_rounds`` full passes without success, fail with ``caption_semantic_transient_exhausted``.

    Other behavior:
    - disabled checker or missing keys returns no failures
    - client import failure skips that key only
    - unparseable model outputs are treated as mismatch (same as before)
    """

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str = "",
        api_keys: list[str] | None = None,
        model: str,
        timeout_sec: int,
        max_samples: int,
        max_key_rotation_rounds: int = 2,
        retry_base_sleep_sec: float = 2.0,
        retry_sleep_cap_sec: float = 120.0,
        provider: str = "openai",
        base_url: str | None = None,
    ):
        self._enabled = enabled
        if api_keys is not None:
            self._api_keys = [k.strip() for k in api_keys if k.strip()]
        else:
            self._api_keys = [api_key.strip()] if api_key.strip() else []
        self._model = model
        self._timeout_sec = timeout_sec
        self._max_samples = max(0, max_samples)
        self._max_key_rotation_rounds = max(1, int(max_key_rotation_rounds))
        self._retry_base_sleep_sec = max(0.1, float(retry_base_sleep_sec))
        self._retry_sleep_cap_sec = max(1.0, float(retry_sleep_cap_sec))
        self._provider = provider
        self._base_url = base_url

    @property
    def active(self) -> bool:
        return self._enabled and bool(self._api_keys) and self._max_samples > 0

    def _try_build_client(self, api_key: str) -> object | None:
        try:
            from openai import OpenAI

            client_kwargs: dict[str, object] = {
                "api_key": api_key,
                "timeout": self._timeout_sec,
                "max_retries": _OPENAI_CLIENT_MAX_RETRIES,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            return OpenAI(**client_kwargs)
        except Exception:
            return None

    def _judge_with_key_failover(
        self,
        *,
        caption: str,
        frame_paths: list[Path],
        clip_id: str,
    ) -> tuple[bool, bool | None]:
        """Return ``(transient_exhausted, verdict)``. ``verdict`` is meaningful only if not exhausted."""
        max_rounds = self._max_key_rotation_rounds
        last_exc: _TransientLLMError | None = None
        for round_i in range(max_rounds):
            attempted_call = False
            keys_tried = 0
            keys_rate_limited = 0
            saw_non_rate_transient = False
            for idx, key in enumerate(self._api_keys):
                client = self._try_build_client(key)
                if client is None:
                    continue
                attempted_call = True
                keys_tried += 1
                try:
                    verdict = self._judge_match(
                        client=client,
                        caption=caption,
                        frame_paths=frame_paths,
                    )
                    logger.info(
                        "Caption semantic LLM OK clip_id=%s key_index=%d (of %d keys) "
                        "sweep_round=%d/%d (vision judge returned a verdict)",
                        clip_id,
                        idx,
                        len(self._api_keys),
                        round_i + 1,
                        max_rounds,
                    )
                    return (False, verdict)
                except _TransientLLMError as exc:
                    last_exc = exc
                    if _is_rate_limit_message(str(exc)):
                        keys_rate_limited += 1
                    else:
                        saw_non_rate_transient = True
                    logger.warning(
                        "Caption semantic transient LLM error clip_id=%s key_index=%d/%d sweep=%d/%d: %s",
                        clip_id,
                        idx,
                        len(self._api_keys) - 1,
                        round_i + 1,
                        max_rounds,
                        exc,
                    )
            if not attempted_call:
                logger.error("Caption semantic OpenAI client unavailable for all keys clip_id=%s", clip_id)
                return (True, None)
            if (
                keys_tried > 0
                and keys_rate_limited == keys_tried
                and not saw_non_rate_transient
            ):
                raise RateLimitSemanticError(
                    f"every API key returned 429/TPM ({keys_tried} key(s)) in sweep {round_i + 1}"
                )
            if round_i >= max_rounds - 1:
                logger.warning(
                    "Caption semantic transient exhausted clip_id=%s after %d full key sweep(s) (%d key(s)): %s",
                    clip_id,
                    max_rounds,
                    len(self._api_keys),
                    last_exc,
                )
                return (True, None)
            delay = _retry_delay_from_message(
                str(last_exc or ""),
                round_i,
                self._retry_base_sleep_sec,
                self._retry_sleep_cap_sec,
            )
            logger.warning(
                "Caption semantic all keys transient clip_id=%s; sleeping %.1fs then sweep %d/%d from first key",
                clip_id,
                delay,
                round_i + 2,
                max_rounds,
            )
            time.sleep(delay)
        return (True, None)

    def check(
        self,
        *,
        sampled: list[ClipRecord],
        frame_paths_by_clip_id: dict[str, list[Path]],
    ) -> list[str]:
        if not self.active:
            return []

        failures: list[str] = []
        checked = 0
        for row in sampled:
            if checked >= self._max_samples:
                break
            if self._contains_prompt_injection_terms(row.caption):
                failures.append(f"caption_semantic_injection_keyword:{row.clip_id}")
                checked += 1
                continue
            frame_paths = [
                path
                for path in frame_paths_by_clip_id.get(row.clip_id, [])
                if path.exists()
            ]
            if not frame_paths:
                failures.append(f"caption_semantic_frames_missing:{row.clip_id}")
                checked += 1
                continue
            try:
                exhausted, verdict = self._judge_with_key_failover(
                    caption=row.caption,
                    frame_paths=frame_paths[:12],
                    clip_id=row.clip_id,
                )
            except RateLimitSemanticError as exc:
                logger.warning(
                    "Caption semantic all keys rate limited (clip rejected) clip_id=%s: %s",
                    row.clip_id,
                    exc,
                )
                failures.append(f"caption_semantic_rate_limited:{row.clip_id}")
                checked += 1
                continue
            if exhausted:
                failures.append(f"caption_semantic_transient_exhausted:{row.clip_id}")
                checked += 1
                continue
            if verdict is False or verdict is None:
                failures.append(f"caption_semantic_mismatch:{row.clip_id}")
            checked += 1
        return failures

    @staticmethod
    def _contains_prompt_injection_terms(caption: str) -> bool:
        return bool(_PROMPT_INJECTION_CAPTION_RE.search(caption))

    def _judge_match(self, *, client: object, caption: str, frame_paths: list[Path]) -> bool | None:
        prompt = (
            "You are validating whether a caption is accurately grounded in timeline-sampled frames from a "
            "short video clip.\n\n"
            f"{SEMANTIC_JUDGE_CRITERIA_FOR_MINER}\n\n"
            "Output: a single JSON object with one boolean field \"match\" (no markdown, no other keys).\n\n"
            f"Caption: {caption}"
        )
        content = [{"type": "text", "text": prompt}]
        for frame_path in frame_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_data_uri(frame_path)},
                }
            )
        create_kwargs: dict[str, object] = {
            "model": self._model,
            "messages": [{"role": "user", "content": content}],
            "max_tokens": 60,
            "temperature": 0,
        }
        # Gemini OpenAI-compat vision calls may reject json_object mode; OpenAI supports it for gpt-4o-class.
        if self._provider == "openai":
            create_kwargs["response_format"] = {"type": "json_object"}
        try:
            response = client.chat.completions.create(**create_kwargs)  # type: ignore[arg-type]
            choice = response.choices[0]
            message = getattr(choice, "message", None)
            output_text = getattr(message, "content", "") if message is not None else ""
            return self._parse_match(str(output_text))
        except Exception as exc:
            if _is_transient_llm_error(exc):
                raise _TransientLLMError(str(exc)) from exc
            return None

    def _frame_data_uri(self, frame_path: Path) -> str:
        payload = base64.b64encode(frame_path.read_bytes()).decode("ascii")
        return f"data:image/jpeg;base64,{payload}"

    def _parse_match(self, output_text: str) -> bool | None:
        text = output_text.strip()
        if not text:
            return None

        # Attempt strict JSON first.
        try:
            data = json.loads(text)
            value = data.get("match")
            if isinstance(value, bool):
                return value
        except Exception:
            pass

        # Attempt to parse JSON from fenced/code-rich output.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
                value = data.get("match")
                if isinstance(value, bool):
                    return value
            except Exception:
                pass

        lowered = text.lower()
        if '"match": false' in lowered or lowered == "false":
            return False
        if '"match": true' in lowered or lowered == "true":
            return True
        return None

