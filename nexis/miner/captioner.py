"""Caption generation wrapper for miner pipeline."""

from __future__ import annotations

import base64
import json
import logging
import random
import re
import threading
import time
from pathlib import Path
from typing import Any, NamedTuple

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    OpenAI,
    RateLimitError,
)

from ..protocol import CAPTION_SKIP_IF_FEWER_WORDS, MIN_CAPTION_WORDS
from ..validator.category_check import (
    get_middle_three_frame_paths,
    parse_strict_pass,
    strict_pass_decision,
)

logger = logging.getLogger(__name__)

# Generation goals aligned with validator hard checks, caption_semantic judge, and injection filter
# (see nexis.validator.checks._check_caption_alignment, nexis.validator.caption_semantic.CaptionSemanticChecker).
# Validator vision judge: match only if every important claim is visible AND the caption is specific to THIS clip's
# main subject/action (not a reusable generic paragraph).
_CAPTION_CONTENT_RULES = (
    "You write captions for 5-second clips used in training. You only see timeline-sampled frames—those pixels "
    "are the sole evidence. A separate automated vision model will read your caption and the same images; it "
    "returns pass only if your text is a faithful, specific description of what is on screen. Write so that pass "
    "is likely.\n\n"
    "Lexical / platform gates (must all hold):\n"
    f"- Output at least {MIN_CAPTION_WORDS} words (whitespace-separated). "
    f"Below that length the caption is discarded.\n"
    "- One or two sentences; never paste URLs.\n"
    "- No http://, https://, or URL-like strings.\n"
    "- Never use the whole words 'match' or 'true' (case-insensitive; blocked downstream).\n\n"
    "Semantic check (same bar as validation—optimize for this):\n"
    "The judge fails the caption if: anything in it is contradicted by the frames; any important stated detail "
    "is not clearly visible; the text is too generic to identify this clip's main visible subject or action; or "
    "the text guesses beyond what the frames show (causes you cannot see, audio, off-screen events, unstated "
    "identities, mood without visible cues).\n"
    "The judge passes only if every substantive claim is visible in the frames AND the caption is specific "
    "enough that it describes this clip's main content, not a generic scene.\n\n"
    "How to describe the frames exactly (do this in order):\n"
    "1) State in plain words the single clearest visible focus: the main object, region, person, animal, or "
    "action that occupies attention (what a viewer would say is 'what this clip shows').\n"
    "2) Add only details you can literally point to in the images: colors, materials, weather/light (haze, sun, "
    "shadows), water or ground texture, built structures, vegetation type, horizon or skyline shape, and where "
    "things sit (foreground, midground, background, left/right). Phrase each detail as something visible, not "
    "as a story or interpretation.\n"
    "3) If frames show motion or change across time, say what moves or changes in simple verbs; if the view is "
    "static, say so.\n"
    "4) Reach the word minimum by adding more precise visible detail—not generic filler. Avoid slogan-like or "
    "travel-brochure wording unless each phrase is tied to something you see (e.g. 'pale mist along the ridge' "
    "only if that mist appears).\n\n"
    "Hard avoids (instant semantic fail):\n"
    "- Inventing people, animals, objects, text, or buildings not clearly in the frames.\n"
    "- Exact counts (three birds, five windows) unless plainly countable.\n"
    "- Stock descriptions that could fit many videos; every clause should fit only this view.\n"
    "- Sounds, speech, music, narrator intent, or 'the camera' / production language.\n\n"
    "Before you output, check: each noun phrase refers to something visible; the main subject or action is "
    "unambiguous; nothing is implied that the frames do not show."
)

_CAPTION_PLAIN_OUTPUT_LINE = (
    "Output only the caption text, with no title, quotes, or preamble."
)

# Plain-text caption path: rules + how to format the assistant reply.
_CAPTION_GENERATION_INSTRUCTIONS = f"{_CAPTION_CONTENT_RULES}\n\n{_CAPTION_PLAIN_OUTPUT_LINE}"


def _nature_miner_json_response_spec(
    *,
    middle_indices: tuple[int, int, int],
    clip_name: str,
    source_url: str,
) -> str:
    """Single structured-output contract for the nature JSON vision call (parseable by the miner).

    Includes clip file name and source URL so the model can disambiguate logs/metadata; both are also
    referenced in caption rules (do not paste the URL into the caption text).
    """
    i0, i1, i2 = middle_indices
    return (
        "Context (for this clip only; do not paste the URL or filename into the caption text):\n"
        f"- Clip file name: {clip_name}\n"
        f"- Source URL: {source_url}\n\n"
        "=== STRUCTURED OUTPUT (mandatory) ===\n"
        "Your entire reply must be one JSON object only (no markdown fences, no commentary). "
        'It must have exactly two top-level keys: "caption" and "frames".\n\n'
        'Key "caption": string. It must obey every caption rule in the section above (same constraints as a '
        "standalone caption).\n\n"
        'Key "frames": JSON array of length exactly 3. Each element is one object with these keys only:\n'
        '  - "frame_index": integer, one of 0, 1, 2 (chronological order within the three middle samples)\n'
        '  - "winner": string, one of: nature, people, animal, vehicle, urban, indoor, other\n'
        '  - "nature_score", "people_score", "animal_score", "vehicle_score", "urban_score", "indoor_score": '
        "each a number from 0.0 through 1.0\n\n"
        "The images below are in strict time order; image index 0 is the earliest. "
        "Score only the three MIDDLE timeline frames: they are at positions "
        f"{i0}, {i1}, and {i2} (0-based indices in the image list). "
        "Produce one frames[] object per middle frame in chronological order; set frame_index to 0, 1, and 2.\n\n"
        "Category scoring rules (each frames[] object must be consistent with these):\n"
        "- nature wins only when natural scenery is the main subject.\n"
        "- If a person is central and dominant, winner must not be nature.\n"
        "- If an animal is the main subject, winner must not be nature.\n"
        "- If a vehicle, city/urban, or indoor scene dominates, winner must not be nature.\n\n"
        "Example shape (values are illustrative; replace with your scores and caption):\n"
        '{"caption": "(long grounded caption here …)", "frames": ['
        '{"frame_index":0,"winner":"nature","nature_score":0.8,"people_score":0.1,'
        '"animal_score":0.0,"vehicle_score":0.0,"urban_score":0.0,"indoor_score":0.0}, '
        '{"frame_index":1,"winner":"urban","nature_score":0.2,"people_score":0.15,'
        '"animal_score":0.0,"vehicle_score":0.0,"urban_score":0.8,"indoor_score":0.0}, '
        '{"frame_index":2,"winner":"nature","nature_score":0.78,"people_score":0.12,'
        '"animal_score":0.0,"vehicle_score":0.0,"urban_score":0.0,"indoor_score":0.0}'
        "]}\n"
    )


class CaptionClipResult(NamedTuple):
    """``caption`` is ``None`` when the segment should be skipped (too short, not nature, or bad JSON)."""

    caption: str | None
    category_proof: dict[str, Any] | None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    raw = text.strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    m = re.search(r"\{[\s\S]*\}", raw)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None

# OpenAI SDK default retries fight TPM windows; we handle 429 with API-suggested delays.
_OPENAI_CLIENT_MAX_RETRIES = 0
# Transient HTTP statuses: retry with other keys / outer backoff (same as provider hiccups).
_RETRYABLE_OPENAI_STATUS_CODES = frozenset({408, 500, 502, 503, 504})
_TRY_AGAIN_MESSAGE_RE = re.compile(
    r"try\s+again\s+in\s+([0-9]+(?:\.[0-9]+)?)\s*s",
    re.IGNORECASE,
)


def merge_openai_api_keys(primary: str, extra_csv: str) -> list[str]:
    """Deduplicated list: OPENAI_API_KEY first, then comma/newline-separated NEXIS_OPENAI_API_KEYS."""
    keys: list[str] = []
    segments: list[str] = [primary]
    if extra_csv.strip():
        for line in extra_csv.replace("\n", ",").split(","):
            segments.append(line)
    for segment in segments:
        k = segment.strip()
        if k and k not in keys:
            keys.append(k)
    return keys


def _safe_source_url_for_prompt(url: str) -> str:
    """Avoid control characters / huge URLs that can break JSON request bodies to OpenAI."""
    cleaned = "".join(ch for ch in url if ord(ch) >= 32)
    return cleaned[:2048] if len(cleaned) > 2048 else cleaned


def _retry_delay_seconds(exc: Exception) -> float | None:
    """Best-effort seconds to wait from Retry-After header or 'try again in Xs' in the error body."""
    response = getattr(exc, "response", None)
    if response is not None:
        headers = getattr(response, "headers", None)
        if headers is not None:
            for name in ("retry-after", "Retry-After"):
                value = headers.get(name)
                if value:
                    try:
                        return float(str(value).strip())
                    except ValueError:
                        pass
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, dict):
            msg = str(err.get("message", ""))
        else:
            msg = str(err or "")
    else:
        msg = str(exc)
    match = _TRY_AGAIN_MESSAGE_RE.search(msg)
    if match:
        return float(match.group(1))
    return None


def _parse_rate_limit_sleep_seconds(exc: RateLimitError) -> float | None:
    return _retry_delay_seconds(exc)


class Captioner:
    """OpenAI chat captions with optional per-key RPM spacing and multi-key 429 failover."""

    def __init__(
        self,
        api_key: str = "",
        model: str = "gpt-4o",
        timeout_sec: int = 30,
        *,
        api_keys: list[str] | None = None,
        provider: str = "openai",
        base_url: str | None = None,
        rate_limit_max_attempts: int = 15,
        rate_limit_max_sleep_sec: float = 120.0,
        openai_rpm_per_key: int = 0,
    ):
        if api_keys is not None:
            self._api_keys = [k.strip() for k in api_keys if k.strip()]
        else:
            self._api_keys = [api_key.strip()] if api_key.strip() else []
        self._model = model
        self._timeout_sec = timeout_sec
        self._provider = provider
        self._base_url = base_url
        self._rate_limit_max_attempts = max(1, int(rate_limit_max_attempts))
        self._rate_limit_max_sleep_sec = max(1.0, float(rate_limit_max_sleep_sec))
        rpm = max(0, int(openai_rpm_per_key))
        self._rpm_per_key = rpm
        self._min_interval = (60.0 / rpm) if rpm > 0 else 0.0
        self._schedule_lock = threading.Lock()
        self._key_next_available: list[float] = [0.0] * len(self._api_keys)

    def _frame_data_uri(self, frame_path: Path) -> str:
        raw = frame_path.read_bytes()
        b64 = base64.b64encode(raw).decode("ascii")
        return f"data:image/jpeg;base64,{b64}"

    def _sleep_between_outer_rounds(
        self,
        *,
        exc: Exception,
        attempt_index: int,
        clip_name: str,
    ) -> bool:
        """Sleep after all keys failed in a sweep (429, 5xx, etc.); return True if caller should retry."""
        suggested = _retry_delay_seconds(exc)
        if suggested is not None:
            delay = min(max(suggested, 0.5), self._rate_limit_max_sleep_sec)
        else:
            delay = min(6.0 * (1.35**attempt_index), self._rate_limit_max_sleep_sec)
        delay += random.uniform(0.0, 0.75)
        if attempt_index + 1 >= self._rate_limit_max_attempts:
            logger.error(
                "caption API retries exhausted clip=%s attempts=%d last_error=%s",
                clip_name,
                self._rate_limit_max_attempts,
                exc,
            )
            return False
        logger.warning(
            "caption backoff clip=%s round=%d/%d sleeping=%.1fs after transient error (keys exhausted this sweep)",
            clip_name,
            attempt_index + 1,
            self._rate_limit_max_attempts,
            delay,
        )
        time.sleep(delay)
        return True

    def _pick_key_index_least_wait(self, tried: set[int]) -> int | None:
        """Choose the key with the earliest next-available time (max aggregate RPM)."""
        with self._schedule_lock:
            best_i: int | None = None
            best_t = float("inf")
            for i in range(len(self._api_keys)):
                if i in tried:
                    continue
                t = self._key_next_available[i]
                if t < best_t:
                    best_t = t
                    best_i = i
            return best_i

    def _wait_and_reserve_rpm_slot(self, idx: int) -> None:
        """Sleep until key idx may send a request, then reserve one slot (60/rpm spacing)."""
        if self._min_interval <= 0:
            return
        while True:
            wait: float
            with self._schedule_lock:
                now = time.monotonic()
                wait = self._key_next_available[idx] - now
                if wait <= 0:
                    self._key_next_available[idx] = now + self._min_interval
                    return
            logger.debug(
                "caption rpm spacing key_index=%d wait=%.2fs (target ~%d req/min per key)",
                idx,
                wait,
                self._rpm_per_key,
            )
            time.sleep(wait)

    def _penalize_key_after_429(self, idx: int, exc: RateLimitError) -> None:
        """Push key cooldown from API hint or at least one spacing interval."""
        suggested = _parse_rate_limit_sleep_seconds(exc)
        if suggested is not None:
            bump = min(max(suggested, 0.5), self._rate_limit_max_sleep_sec)
        else:
            base = self._min_interval if self._min_interval > 0 else 6.0
            bump = min(max(base, 1.0), self._rate_limit_max_sleep_sec)
        with self._schedule_lock:
            now = time.monotonic()
            self._key_next_available[idx] = max(self._key_next_available[idx], now + bump)
        logger.warning(
            "caption 429 penalty key_index=%d cooldown=%.1fs",
            idx,
            bump,
        )

    def _build_openai_client(self, api_key: str) -> OpenAI:
        client_kwargs: dict[str, object] = {
            "api_key": api_key,
            "timeout": self._timeout_sec,
            "max_retries": _OPENAI_CLIENT_MAX_RETRIES,
        }
        if self._base_url:
            client_kwargs["base_url"] = self._base_url
        return OpenAI(**client_kwargs)

    def _chat_completion(
        self,
        client: OpenAI,
        *,
        clip_path: Path,
        source_url: str,
        valid_frames: list[Path],
        extra_user_text: str = "",
    ) -> str:
        safe_url = _safe_source_url_for_prompt(source_url)
        prompt = (
            f"{_CAPTION_GENERATION_INSTRUCTIONS}\n\n"
            f"Clip file name: {clip_path.name}. Source URL (do not repeat or quote in the caption): {safe_url}."
        )
        if extra_user_text.strip():
            prompt = f"{prompt} {extra_user_text.strip()}"
        frames: list[dict[str, object]] = []
        for frame_path in valid_frames:
            frames.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_data_uri(frame_path)},
                }
            )
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}] + frames
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=300,
            temperature=0,
        )
        return (response.choices[0].message.content or "").strip()

    def _chat_completion_nature_json(
        self,
        client: OpenAI,
        *,
        clip_path: Path,
        source_url: str,
        valid_frames: list[Path],
        middle_indices: tuple[int, int, int],
    ) -> str:
        safe_url = _safe_source_url_for_prompt(source_url)
        # Rules + JSON schema; clip name + URL are inside _nature_miner_json_response_spec (top "Context" block).
        prompt = (
            f"{_CAPTION_CONTENT_RULES}\n\n"
            f"{_nature_miner_json_response_spec(middle_indices=middle_indices, clip_name=clip_path.name, source_url=safe_url)}"
        )
        frames: list[dict[str, object]] = []
        for frame_path in valid_frames:
            frames.append(
                {
                    "type": "image_url",
                    "image_url": {"url": self._frame_data_uri(frame_path)},
                }
            )
        content: list[dict[str, object]] = [{"type": "text", "text": prompt}] + frames
        response = client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            max_tokens=900,
            temperature=0,
            response_format={"type": "json_object"},
        )
        return (response.choices[0].message.content or "").strip()

    def _evaluate_nature_json_response(self, raw: str, *, clip_name: str) -> CaptionClipResult:
        data = _extract_json_object(raw)
        if not isinstance(data, dict):
            logger.warning("caption+nature JSON parse failed clip=%s", clip_name)
            return CaptionClipResult(None, {"error": "json_parse_failed", "raw_preview": raw[:400]})

        caption = str(data.get("caption", "")).strip()
        wc = len(caption.lower().split()) if caption else 0
        if wc < CAPTION_SKIP_IF_FEWER_WORDS:
            logger.warning(
                "caption too short clip=%s words=%d need>=%d",
                clip_name,
                wc,
                CAPTION_SKIP_IF_FEWER_WORDS,
            )
            return CaptionClipResult(None, {"error": "caption_too_short", "words": wc, "data": data})

        frames_raw = data.get("frames")
        parsed = parse_strict_pass({"frames": frames_raw}) if isinstance(frames_raw, list) else None
        if parsed is None:
            logger.warning("caption+nature strict frames parse failed clip=%s", clip_name)
            return CaptionClipResult(
                None,
                {"error": "category_frames_invalid", "data": data},
            )

        decision = strict_pass_decision(parsed)
        proof: dict[str, Any] = {
            "stage": "miner_caption_nature_json",
            "strict_decision": decision.decision,
            "strict_reason": decision.reason,
            "nature_score": decision.nature_score,
            "rival_score": decision.rival_score,
            "margin": decision.margin,
            "frames": frames_raw,
        }
        if decision.decision != "accept":
            logger.info(
                "prepare skip segment reason=category_not_nature clip=%s decision=%s %s",
                clip_name,
                decision.decision,
                decision.reason,
            )
            return CaptionClipResult(None, proof)

        return CaptionClipResult(caption, proof)

    def caption_clip(
        self,
        clip_path: Path,
        source_url: str,
        first_frame_path: Path | None = None,
        frame_paths: list[Path] | None = None,
        *,
        dataset_category: str | None = None,
    ) -> CaptionClipResult:
        """Return caption and optional category proof, or skip (caption ``None``) if too short / not nature."""
        if not self._api_keys:
            logger.warning(
                "caption skip reason=missing_%s_api_key",
                self._provider,
            )
            return CaptionClipResult(None, {"error": "missing_api_key", "provider": self._provider})

        valid_frames = [p for p in (frame_paths or []) if p.exists()]
        if not valid_frames and first_frame_path and first_frame_path.exists():
            valid_frames = [first_frame_path]
        valid_frames = valid_frames[:12]

        use_nature_json = (
            self._provider == "openai"
            and bool(dataset_category and "nature" in dataset_category.lower())
        )
        middle_three = get_middle_three_frame_paths(valid_frames) if use_nature_json else None
        if use_nature_json and middle_three is None:
            logger.warning(
                "prepare skip segment reason=need_three_frames_for_category clip=%s",
                clip_path.name,
            )
            return CaptionClipResult(None, {"error": "insufficient_frames_for_category"})

        middle_indices: tuple[int, int, int] | None = None
        if use_nature_json and middle_three is not None:
            by_path = {str(p.resolve()): i for i, p in enumerate(valid_frames)}
            try:
                middle_indices = (
                    by_path[str(middle_three[0].resolve())],
                    by_path[str(middle_three[1].resolve())],
                    by_path[str(middle_three[2].resolve())],
                )
            except KeyError:
                logger.warning("prepare skip segment reason=middle_frame_index clip=%s", clip_path.name)
                return CaptionClipResult(None, {"error": "middle_frame_resolve_failed"})

        n = len(self._api_keys)
        logger.debug(
            "caption request clip=%s frames=%d keys=%d nature_json=%s rpm_per_key=%s",
            clip_path.name,
            len(valid_frames),
            n,
            use_nature_json,
            self._rpm_per_key if self._rpm_per_key > 0 else "off",
        )

        outer = 0
        last_transient: Exception | None = None
        rate_limit_events = 0

        while outer < self._rate_limit_max_attempts:
            tried: set[int] = set()
            while len(tried) < n:
                idx = self._pick_key_index_least_wait(tried)
                if idx is None:
                    break
                self._wait_and_reserve_rpm_slot(idx)
                key = self._api_keys[idx]
                client = self._build_openai_client(key)
                try:
                    if use_nature_json and middle_indices is not None:
                        raw = self._chat_completion_nature_json(
                            client,
                            clip_path=clip_path,
                            source_url=source_url,
                            valid_frames=valid_frames,
                            middle_indices=middle_indices,
                        )
                        logger.debug("caption+nature JSON clip=%s raw_len=%d", clip_path.name, len(raw))
                        logger.info(
                            "caption OpenAI OK clip=%s key_index=%d (of %d keys) "
                            "outer_backoff_round=%d/%d prior_429_count=%d phase=caption+nature_json",
                            clip_path.name,
                            idx,
                            n,
                            outer + 1,
                            self._rate_limit_max_attempts,
                            rate_limit_events,
                        )
                        return self._evaluate_nature_json_response(raw, clip_name=clip_path.name)

                    text = self._chat_completion(
                        client,
                        clip_path=clip_path,
                        source_url=source_url,
                        valid_frames=valid_frames,
                    )
                    logger.debug("caption response clip=%s text_len=%d", clip_path.name, len(text))
                    wc = len(text.strip().lower().split()) if text else 0
                    if wc < CAPTION_SKIP_IF_FEWER_WORDS:
                        logger.warning(
                            "caption too short clip=%s words=%d need>=%d; skipping segment (no retry)",
                            clip_path.name,
                            wc,
                            CAPTION_SKIP_IF_FEWER_WORDS,
                        )
                        return CaptionClipResult(None, None)
                    logger.info(
                        "caption OpenAI OK clip=%s key_index=%d (of %d keys) "
                        "outer_backoff_round=%d/%d prior_429_count=%d phase=plain_caption",
                        clip_path.name,
                        idx,
                        n,
                        outer + 1,
                        self._rate_limit_max_attempts,
                        rate_limit_events,
                    )
                    return CaptionClipResult(text, None)
                except RateLimitError as exc:
                    rate_limit_events += 1
                    last_transient = exc
                    self._penalize_key_after_429(idx, exc)
                    tried.add(idx)
                    if n > 1:
                        if idx < n - 1:
                            logger.warning(
                                "caption 429 clip=%s key_index=%d; next=try different API key (%d keys total)",
                                clip_path.name,
                                idx,
                                n,
                            )
                        else:
                            logger.warning(
                                "caption 429 clip=%s key_index=%d (last key this sweep); "
                                "next=outer backoff then retry from key 0",
                                clip_path.name,
                                idx,
                            )
                    else:
                        logger.warning(
                            "caption 429 clip=%s single API key; next=sleep outer round %d/%d",
                            clip_path.name,
                            outer + 1,
                            self._rate_limit_max_attempts,
                        )
                    continue
                except APIStatusError as exc:
                    # RateLimitError is handled above; here: 5xx/408 retry, other statuses fail.
                    if exc.status_code in _RETRYABLE_OPENAI_STATUS_CODES:
                        rate_limit_events += 1
                        last_transient = exc
                        tried.add(idx)
                        logger.warning(
                            "caption OpenAI HTTP %s clip=%s key_index=%d; rotating/retrying (%s)",
                            exc.status_code,
                            clip_path.name,
                            idx,
                            exc.__class__.__name__,
                        )
                        continue
                    if exc.status_code == 400:
                        logger.error(
                            "caption OpenAI HTTP 400 (bad request — often invalid payload or URL) clip=%s: %s",
                            clip_path.name,
                            exc,
                        )
                        return CaptionClipResult(
                            None,
                            {"error": "caption_bad_request", "http_status": 400},
                        )
                    logger.error(
                        "caption OpenAI HTTP %s (not retrying) clip=%s: %s",
                        exc.status_code,
                        clip_path.name,
                        exc,
                    )
                    return CaptionClipResult(
                        None,
                        {
                            "error": "caption_api_error",
                            "http_status": exc.status_code,
                        },
                    )
                except (APIConnectionError, APITimeoutError) as exc:
                    rate_limit_events += 1
                    last_transient = exc
                    tried.add(idx)
                    logger.warning(
                        "caption OpenAI connection/timeout clip=%s key_index=%d: %s",
                        clip_path.name,
                        idx,
                        exc,
                    )
                    continue
                except Exception:
                    logger.exception("caption generation failed clip=%s", clip_path.name)
                    return CaptionClipResult(None, {"error": "caption_generation_failed"})

            if last_transient is None:
                return CaptionClipResult(None, {"error": "caption_unavailable"})
            if not self._sleep_between_outer_rounds(
                exc=last_transient,
                attempt_index=outer,
                clip_name=clip_path.name,
            ):
                return CaptionClipResult(None, {"error": "caption_rate_limit_exhausted"})
            outer += 1

        return CaptionClipResult(None, {"error": "caption_rate_limit_exhausted"})
