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
    """Transient LLM server-side issue; caller may retry."""


def _retry_delay_from_message(message: str, attempt: int, base_sleep_sec: float, cap_sec: float) -> float:
    """Prefer provider-suggested wait (e.g. OpenAI 429 body), else exponential backoff."""
    m = re.search(r"try again in ([0-9.]+)\s*s", message, re.IGNORECASE)
    if m:
        return min(float(m.group(1)) + 0.25, cap_sec)
    return min(base_sleep_sec * (2**attempt), cap_sec)


def _is_transient_llm_error(error: Exception) -> bool:
    status_code = getattr(error, "status_code", None)
    if isinstance(status_code, int) and status_code in _TRANSIENT_LLM_STATUS_CODES:
        return True
    lowered = str(error).lower()
    return any(hint in lowered for hint in _TRANSIENT_LLM_ERROR_HINTS)


class CaptionSemanticChecker:
    """Optional semantic checker using OpenAI-compatible vision APIs.

    Transient API errors (429, 5xx, timeouts) are retried; after exhausting retries the clip
    fails with ``caption_semantic_transient_exhausted``.

    Other behavior:
    - disabled checker or missing key returns no failures
    - client import failure returns no failures
    - unparseable model outputs are treated as mismatch (same as before)
    """

    def __init__(
        self,
        *,
        enabled: bool,
        api_key: str,
        model: str,
        timeout_sec: int,
        max_samples: int,
        max_transient_retries: int = 5,
        retry_base_sleep_sec: float = 2.0,
        retry_sleep_cap_sec: float = 120.0,
        provider: str = "openai",
        base_url: str | None = None,
    ):
        self._enabled = enabled
        self._api_key = api_key.strip()
        self._model = model
        self._timeout_sec = timeout_sec
        self._max_samples = max(0, max_samples)
        self._max_transient_retries = max(0, int(max_transient_retries))
        self._retry_base_sleep_sec = max(0.1, float(retry_base_sleep_sec))
        self._retry_sleep_cap_sec = max(1.0, float(retry_sleep_cap_sec))
        self._provider = provider
        self._base_url = base_url

    @property
    def active(self) -> bool:
        return self._enabled and bool(self._api_key) and self._max_samples > 0

    def check(
        self,
        *,
        sampled: list[ClipRecord],
        frame_paths_by_clip_id: dict[str, list[Path]],
    ) -> list[str]:
        if not self.active:
            return []

        try:
            from openai import OpenAI

            client_kwargs: dict[str, object] = {
                "api_key": self._api_key,
                "timeout": self._timeout_sec,
            }
            if self._base_url:
                client_kwargs["base_url"] = self._base_url
            client = OpenAI(**client_kwargs)
        except Exception:
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
                continue
            max_attempts = 1 + self._max_transient_retries
            verdict: bool | None = None
            exhausted = False
            for attempt in range(max_attempts):
                try:
                    verdict = self._judge_match(
                        client=client,
                        caption=row.caption,
                        frame_paths=frame_paths[:12],
                    )
                    break
                except _TransientLLMError as exc:
                    if attempt >= max_attempts - 1:
                        logger.warning(
                            "Caption semantic transient errors exhausted for clip_id=%s after %d attempt(s): %s",
                            row.clip_id,
                            max_attempts,
                            exc,
                        )
                        failures.append(f"caption_semantic_transient_exhausted:{row.clip_id}")
                        exhausted = True
                        break
                    delay = _retry_delay_from_message(
                        str(exc),
                        attempt,
                        self._retry_base_sleep_sec,
                        self._retry_sleep_cap_sec,
                    )
                    logger.warning(
                        "Caption semantic transient LLM error clip_id=%s attempt %d/%d; retry in %.1fs: %s",
                        row.clip_id,
                        attempt + 1,
                        max_attempts,
                        delay,
                        exc,
                    )
                    time.sleep(delay)
            if exhausted:
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
        prompt = f"""
You are validating whether a caption is accurately grounded in timeline-sampled frames from a short video clip.

Return JSON only:
{{"match": true}} or {{"match": false}}

Return false if:
- any part of the caption is contradicted by the frames
- any important detail is not visually supported
- the caption is overly generic and fails to capture the main visible subject or action
- the caption includes speculation or inference beyond the frames

Return true only if the caption is both visually supported and specific enough to describe the clip's main content.

Caption: {caption}
"""
        content = [{"type": "text", "text": prompt}]
        for frame_path in frame_paths:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_data_uri(frame_path)},
                }
            )
        try:
            response = client.chat.completions.create(  # type: ignore[attr-defined]
                model=self._model,
                messages=[{"role": "user", "content": content}],
                max_tokens=60,
            )
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

