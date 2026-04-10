"""Drop pending clips that would fail validator hard checks or sampled asset verification."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from ..models import ClipRecord
from ..specs import DatasetSpecRegistry
from ..validator.assets import VideoAssetVerifier
from ..validator.sampling import select_row_indices

if TYPE_CHECKING:
    from ..config import Settings

logger = logging.getLogger(__name__)

_MAX_PREFLIGHT_ROUNDS = 64
_INTERVAL_SEED_PREFIX = "interval:"

# Failure codes that identify a single clip_id as the first segment after the prefix.
_SINGLE_CLIP_FAILURE_PREFIXES = frozenset(
    {
        "non_youtube_source",
        "empty_caption",
        "short_caption",
        "url_like_caption",
        "missing_clip_asset",
        "clip_sha256_mismatch",
        "missing_first_frame_asset",
        "first_frame_sha256_mismatch",
        "invalid_resolution",
        "caption_semantic_injection_keyword",
        "caption_semantic_mismatch",
        "caption_semantic_transient_exhausted",
        "caption_semantic_rate_limited",
        "category_strict_frames_missing",
        "category_strict_api_key_missing",
        "category_strict_client_unavailable",
        "category_strict_response_invalid",
        "category_strict_reject",
        "category_strict_transient_exhausted",
        "caption_semantic_frames_missing",
    }
)


def interval_seed_for_validator(interval_id: int) -> str:
    return hashlib.sha256(f"{_INTERVAL_SEED_PREFIX}{interval_id}".encode("utf-8")).hexdigest()


def _resolve_overlap_drop(failure: str, by_id: dict[str, ClipRecord]) -> set[str]:
    if not failure.startswith("overlap_lt_5s:"):
        return set()
    tail = failure[len("overlap_lt_5s:") :]
    parts = tail.rsplit(":", 2)
    if len(parts) != 3:
        return set()
    _source_key, prev_id, curr_id = parts
    prev_r = by_id.get(prev_id)
    curr_r = by_id.get(curr_id)
    if prev_r is not None and curr_r is not None:
        if curr_r.clip_start_sec >= prev_r.clip_start_sec:
            return {curr_id}
        return {prev_id}
    return {prev_id, curr_id}


def _resolve_hard_check_drops(failures: list[str], records: list[ClipRecord]) -> set[str]:
    by_id = {r.clip_id: r for r in records}
    drop: set[str] = set()
    for f in failures:
        overlap = _resolve_overlap_drop(f, by_id)
        if overlap:
            drop |= overlap
            continue
        if ":" not in f:
            continue
        prefix, rest = f.split(":", 1)
        if prefix in _SINGLE_CLIP_FAILURE_PREFIXES:
            if prefix == "invalid_resolution":
                sub = rest.split(":", 1)
                drop.add(sub[0])
            else:
                drop.add(rest.split(":", 1)[0])
    return drop


def prune_until_hard_checks_pass(records: list[ClipRecord], spec: DatasetSpec) -> list[ClipRecord]:
    remaining = list(records)
    for _ in range(_MAX_PREFLIGHT_ROUNDS):
        if not remaining:
            return remaining
        result = spec.run_hard_checks(remaining)
        if not result.failures:
            return remaining
        to_drop = _resolve_hard_check_drops(result.failures, remaining)
        if not to_drop:
            logger.warning(
                "preflight hard checks still failing but no removable clip ids; dropping all %d rows: %s",
                len(remaining),
                result.failures[:5],
            )
            return []
        remaining = [r for r in remaining if r.clip_id not in to_drop]
    logger.warning("preflight hard checks exceeded %d rounds; stopping with %d rows", _MAX_PREFLIGHT_ROUNDS, len(remaining))
    return remaining


def clip_ids_from_asset_and_llm_failures(failures: list[str]) -> set[str]:
    """Map verifier / optional LLM failure lines to clip_ids."""
    ids: set[str] = set()
    for f in failures:
        if f.startswith("invalid_asset_uri:"):
            continue
        if f.startswith("overlap_lt_5s:"):
            tail = f[len("overlap_lt_5s:") :].rsplit(":", 2)
            if len(tail) == 3:
                ids.update([tail[1], tail[2]])
            continue
        if f.startswith("invalid_resolution:"):
            parts = f.split(":", 2)
            if len(parts) >= 2:
                ids.add(parts[1])
            continue
        if ":" not in f:
            continue
        prefix, rest = f.split(":", 1)
        if prefix in _SINGLE_CLIP_FAILURE_PREFIXES:
            if prefix == "invalid_resolution":
                ids.add(rest.split(":", 1)[0])
            else:
                ids.add(rest.split(":", 1)[0])
    return ids


def _match_invalid_uri_failure(failure: str, records: list[ClipRecord]) -> set[str]:
    if not failure.startswith("invalid_asset_uri:"):
        return set()
    uri = failure[len("invalid_asset_uri:") :]
    return {r.clip_id for r in records if r.clip_uri == uri or r.first_frame_uri == uri}


def clip_ids_from_all_preflight_failures(failures: list[str], records: list[ClipRecord]) -> set[str]:
    ids = clip_ids_from_asset_and_llm_failures(failures)
    for f in failures:
        ids |= _match_invalid_uri_failure(f, records)
    return ids


_CAPTION_SEMANTIC_FAILURE_PREFIXES = (
    "caption_semantic_injection_keyword:",
    "caption_semantic_mismatch:",
    "caption_semantic_rate_limited:",
    "caption_semantic_transient_exhausted:",
    "caption_semantic_frames_missing:",
)


def clip_ids_from_caption_semantic_failures(failures: list[str]) -> set[str]:
    """Clip ids that failed (or could not complete) caption semantic validation."""
    ids: set[str] = set()
    for f in failures:
        for p in _CAPTION_SEMANTIC_FAILURE_PREFIXES:
            if f.startswith(p):
                ids.add(f[len(p) :])
                break
    return ids


def run_preflight_on_candidates(
    *,
    workdir: Path,
    records: list[ClipRecord],
    spec_id: str,
    miner_hotkey: str,
    interval_id: int,
    registry: DatasetSpecRegistry | None = None,
    caption_semantic_checker: object | None = None,
    category_checker: object | None = None,
) -> tuple[list[ClipRecord], list[ClipRecord]]:
    """Return (kept, removed). Uses same row sampling seed as the validator pipeline."""
    reg = registry or DatasetSpecRegistry.with_defaults()
    spec = reg.get(spec_id)
    remaining = prune_until_hard_checks_pass(list(records), spec)

    verifier = spec.build_asset_verifier()
    if not isinstance(verifier, VideoAssetVerifier):
        if remaining:
            logger.info("preflight skip local asset verify (no VideoAssetVerifier for spec=%s)", spec_id)
        kept_ids = {r.clip_id for r in remaining}
        removed = [r for r in records if r.clip_id not in kept_ids]
        return remaining, removed

    seed = interval_seed_for_validator(interval_id)
    miner_dir = workdir.expanduser().resolve() / ".preflight_assets"
    miner_dir.mkdir(parents=True, exist_ok=True)

    for round_i in range(_MAX_PREFLIGHT_ROUNDS):
        if not remaining:
            break
        remaining = prune_until_hard_checks_pass(remaining, spec)
        if not remaining:
            break

        idx = select_row_indices(len(remaining), miner_hotkey, seed)
        sampled = [remaining[i] for i in idx]
        asset_out = verifier.verify_local_workdir(workdir=workdir, sampled=sampled, miner_dir=miner_dir)
        failures = list(asset_out.failures)

        semantic_failures: list[str] = []
        if caption_semantic_checker is not None and getattr(caption_semantic_checker, "active", False):
            check = getattr(caption_semantic_checker, "check", None)
            if callable(check):
                semantic_failures = check(
                    sampled=sampled,
                    frame_paths_by_clip_id=asset_out.semantic_frames_by_clip_id,
                )
                failures.extend(semantic_failures)
        bad_semantic = clip_ids_from_caption_semantic_failures(semantic_failures)

        if category_checker is not None and getattr(category_checker, "active", False):
            ccheck = getattr(category_checker, "check", None)
            if callable(ccheck):
                sampled_category = [r for r in sampled if r.clip_id not in bad_semantic]
                if sampled_category:
                    failures.extend(
                        ccheck(
                            sampled=sampled_category,
                            frame_paths_by_clip_id=asset_out.semantic_frames_by_clip_id,
                        )
                    )

        if not failures:
            break
        bad = clip_ids_from_all_preflight_failures(failures, sampled)
        if not bad:
            logger.warning("preflight asset/LLM failures but no clip ids parsed: %s", failures[:8])
            break
        logger.info(
            "preflight round=%d dropping %d clip(s) from sample failures: %s",
            round_i,
            len(bad),
            sorted(bad)[:12],
        )
        remaining = [r for r in remaining if r.clip_id not in bad]

    kept_ids = {r.clip_id for r in remaining}
    removed = [r for r in records if r.clip_id not in kept_ids]
    return remaining, removed


def build_preflight_llm_checkers(settings: "Settings") -> tuple[object | None, object | None]:
    """Optional semantic/category checkers when miner preflight mirrors validator LLM gates."""
    from ..miner.llm_runtime import GEMINI_OPENAI_BASE_URL, openai_api_keys_merged, resolve_llm_runtime
    from ..validator.caption_semantic import CaptionSemanticChecker
    from ..validator.category_check import NatureCategoryChecker

    merged_openai = openai_api_keys_merged(settings)

    semantic: object | None = None
    if settings.miner_preflight_semantic:
        gemini_key = settings.gemini_api_key.strip()
        if settings.miner_preflight_semantic_use_gemini and gemini_key:
            prov = "gemini"
            model = settings.miner_preflight_semantic_gemini_model.strip() or "gemini-3.1-flash-lite-preview"
            base = GEMINI_OPENAI_BASE_URL
            sem_keys = [gemini_key]
            logger.info(
                "preflight semantic checker: Gemini model=%s (NEXIS_MINER_PREFLIGHT_SEMANTIC_USE_GEMINI=true)",
                model,
            )
        elif settings.miner_preflight_semantic_use_gemini and not gemini_key:
            logger.warning(
                "NEXIS_MINER_PREFLIGHT_SEMANTIC_USE_GEMINI=true but GEMINI_API_KEY is empty; "
                "using OpenAI for preflight semantic (NEXIS_VALIDATOR_SEMANTIC_MODEL)"
            )
            prov, key, model, base, _route = resolve_llm_runtime(
                settings,
                openai_model=settings.validator_semantic_model,
            )
            if prov == "openai":
                sem_keys = merged_openai if merged_openai else ([key] if key.strip() else [])
            else:
                sem_keys = [key] if key.strip() else []
        else:
            prov, key, model, base, _route = resolve_llm_runtime(
                settings,
                openai_model=settings.validator_semantic_model,
            )
            if prov == "openai":
                sem_keys = merged_openai if merged_openai else ([key] if key.strip() else [])
            else:
                sem_keys = [key] if key.strip() else []
        semantic = CaptionSemanticChecker(
            enabled=True,
            api_keys=sem_keys,
            model=model,
            timeout_sec=settings.validator_semantic_timeout_sec,
            max_samples=settings.validator_semantic_max_samples,
            max_key_rotation_rounds=settings.validator_semantic_max_key_rotation_rounds,
            retry_base_sleep_sec=settings.validator_semantic_retry_base_sleep_sec,
            retry_sleep_cap_sec=settings.validator_semantic_retry_sleep_cap_sec,
            provider=prov,
            base_url=base,
        )
    category: object | None = None
    if settings.miner_preflight_category:
        cprov, ckey, cmodel, cbase, _croute = resolve_llm_runtime(
            settings,
            openai_model=settings.validator_category_model,
        )
        if cprov == "openai":
            cat_keys = merged_openai if merged_openai else ([ckey] if ckey.strip() else [])
        else:
            cat_keys = [ckey] if ckey.strip() else []
        category = NatureCategoryChecker(
            enabled=True,
            api_keys=cat_keys,
            timeout_sec=settings.validator_category_timeout_sec,
            max_samples=settings.validator_category_max_samples,
            base_url=cbase,
            model=cmodel,
            max_key_rotation_rounds=settings.validator_semantic_max_key_rotation_rounds,
            retry_base_sleep_sec=settings.validator_semantic_retry_base_sleep_sec,
            retry_sleep_cap_sec=settings.validator_semantic_retry_sleep_cap_sec,
        )
    return semantic, category
