"""Shared mine-prepare logic (CLI and python -m entry points)."""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from ..config import Settings
from ..specs import DatasetSpecRegistry
from .captioner import Captioner, merge_openai_api_keys
from .llm_runtime import resolve_llm_runtime
from .pending_pack import PendingSaveValidatorConfig
from .pipeline import MinerPipeline
from .preflight import build_preflight_llm_checkers
from .thread_keys import parse_thread_keys_file

logger = logging.getLogger(__name__)


def mine_prepare_worker(
    worker_index: int,
    keys: list[str],
    *,
    workdir: Path,
    sources_path: Path,
    caption_provider: str,
    caption_model: str,
    caption_base_url: str | None,
    caption_timeout: int,
    rate_limit_max_attempts: int,
    rate_limit_max_sleep_sec: float,
    rpm: int,
    max_segments_per_worker: int,
    active_spec: str,
    active_category: str,
    pending_save: PendingSaveValidatorConfig | None,
) -> None:
    worker_id = f"w{worker_index}"
    if caption_provider not in ("openai", "gemini"):
        raise RuntimeError("mine-prepare supports OpenAI or Gemini caption providers only.")
    if not keys:
        raise RuntimeError(f"no API keys configured for worker {worker_id}")
    use_rpm = rpm if caption_provider == "openai" else 0
    captioner = Captioner(
        api_keys=keys,
        model=caption_model,
        timeout_sec=caption_timeout,
        provider=caption_provider,
        base_url=caption_base_url,
        rate_limit_max_attempts=rate_limit_max_attempts,
        rate_limit_max_sleep_sec=rate_limit_max_sleep_sec,
        openai_rpm_per_key=use_rpm,
    )
    pipeline = MinerPipeline(
        store=None,
        captioner=captioner,
        spec_id=active_spec,
        dataset_category=active_category,
        pending_save=pending_save,
    )
    done = 0
    while True:
        if max_segments_per_worker > 0 and done >= max_segments_per_worker:
            logger.info("mine-prepare worker %s stop reason=max_segments", worker_id)
            break
        try:
            ok = pipeline.mine_one_segment(
                workdir,
                sources_path,
                worker_id=worker_id,
                dequeue_sources=True,
            )
        except Exception:
            logger.exception("mine-prepare worker %s failed", worker_id)
            break
        if not ok:
            logger.info("mine-prepare worker %s idle (no URL or queue empty)", worker_id)
            break
        done += 1


def run_mine_prepare(
    *,
    settings: Settings,
    workdir: Path,
    sources_path: Path,
    workers: int,
    thread_keys_file: Path | None,
    max_segments_per_worker: int,
    rpm_per_key: int | None,
    active_spec: str,
    active_category: str,
    spec_registry: DatasetSpecRegistry,
    spec_registry_enabled_ids: set[str],
    spec_registry_all_ids: set[str],
) -> None:
    """Validate settings and run parallel prepare workers. Raises ValueError on bad input."""
    if not active_category:
        raise ValueError("NEXIS_DATASET_CATEGORY must not be empty")
    if active_spec not in spec_registry_enabled_ids:
        raise ValueError(
            f"Spec '{active_spec}' is not enabled for miner (enabled: {', '.join(sorted(spec_registry_enabled_ids))})"
        )
    if active_spec not in spec_registry_all_ids:
        raise ValueError(f"Unknown dataset spec: {active_spec}")

    (
        caption_provider,
        _single_key,
        caption_model,
        caption_base_url,
        caption_route,
    ) = resolve_llm_runtime(settings, openai_model=settings.caption_model)
    if caption_route == "no_api_key":
        raise ValueError("OPENAI_API_KEY (and/or keys file) required for mine-prepare")

    rpm = settings.caption_openai_rpm_per_key if rpm_per_key is None else rpm_per_key
    key_matrix: list[list[str]]
    if caption_provider == "gemini":
        if workers != 1:
            raise ValueError("mine-prepare with Gemini requires --workers 1")
        if thread_keys_file is not None:
            raise ValueError("--thread-keys-file is not used with Gemini; use GEMINI_API_KEY")
        gk = settings.gemini_api_key.strip()
        if not gk:
            raise ValueError("GEMINI_API_KEY required for mine-prepare with Gemini")
        key_matrix = [[gk]]
    elif workers > 1:
        if thread_keys_file is None:
            raise ValueError("--thread-keys-file is required when --workers > 1")
        key_rows = parse_thread_keys_file(thread_keys_file)
        if len(key_rows) != workers:
            raise ValueError(
                f"--thread-keys-file has {len(key_rows)} non-empty key line(s); "
                f"must equal --workers ({workers})"
            )
        key_matrix = key_rows
    else:
        if thread_keys_file is not None:
            rows = parse_thread_keys_file(thread_keys_file)
            key_matrix = [rows[0]] if rows else []
            if not key_matrix or not key_matrix[0]:
                raise ValueError("thread keys file has no keys on first line")
        else:
            merged = merge_openai_api_keys(settings.openai_api_key, settings.openai_api_keys_extra)
            if not merged:
                raise ValueError(
                    "No OpenAI keys: set OPENAI_API_KEY / NEXIS_OPENAI_API_KEYS or --thread-keys-file"
                )
            key_matrix = [merged]

    workdir.expanduser().resolve().mkdir(parents=True, exist_ok=True)
    sources_resolved = sources_path.expanduser().resolve()

    pending_save: PendingSaveValidatorConfig | None = None
    if settings.miner_prepare_validate_before_save:
        sem: object | None = None
        cat: object | None = None
        if settings.miner_preflight_semantic or settings.miner_preflight_category:
            sem, cat = build_preflight_llm_checkers(settings)
        if settings.miner_prepare_semantic_only:
            cat = None
        hotkey_label = f"{settings.bt_wallet_name}:{settings.bt_wallet_hotkey}"
        pending_save = PendingSaveValidatorConfig(
            enabled=True,
            spec_id=active_spec,
            registry=spec_registry,
            miner_hotkey=hotkey_label,
            prepare_sample_interval_id=0,
            run_local_asset_verify=settings.miner_prepare_asset_verify,
            caption_semantic_checker=sem,
            category_checker=cat,
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                mine_prepare_worker,
                i,
                key_matrix[i],
                workdir=workdir,
                sources_path=sources_resolved,
                caption_provider=caption_provider,
                caption_model=caption_model,
                caption_base_url=caption_base_url,
                caption_timeout=settings.caption_timeout_sec,
                rate_limit_max_attempts=settings.caption_rate_limit_max_attempts,
                rate_limit_max_sleep_sec=settings.caption_rate_limit_max_sleep_sec,
                rpm=rpm,
                max_segments_per_worker=max_segments_per_worker,
                active_spec=active_spec,
                active_category=active_category,
                pending_save=pending_save,
            )
            for i in range(workers)
        ]
        for fut in as_completed(futures):
            fut.result()
