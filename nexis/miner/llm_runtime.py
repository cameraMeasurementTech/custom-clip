"""Resolve caption / LLM provider from Settings (OpenAI vs Gemini)."""

from __future__ import annotations

import logging

from ..config import Settings
from .captioner import merge_openai_api_keys

logger = logging.getLogger(__name__)

OPENAI_PRIMARY_MODEL = "gpt-4o"
GEMINI_PRIMARY_MODEL = "gemini-3.1-flash-lite-preview"
GEMINI_OPENAI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai/"


def resolve_llm_runtime(
    settings: Settings,
    *,
    openai_model: str,
) -> tuple[str, str, str, str | None, str]:
    resolved_openai_model = openai_model.strip()
    if not resolved_openai_model or resolved_openai_model == "gpt-4o-mini":
        resolved_openai_model = OPENAI_PRIMARY_MODEL
    openai_api_key = settings.openai_api_key.strip()
    gemini_api_key = settings.gemini_api_key.strip()
    if openai_api_key and gemini_api_key:
        logger.info("both OPENAI_API_KEY and GEMINI_API_KEY are set; preferring OpenAI")
    if openai_api_key:
        return (
            "openai",
            openai_api_key,
            resolved_openai_model,
            None,
            "openai_key",
        )
    if gemini_api_key:
        return (
            "gemini",
            gemini_api_key,
            GEMINI_PRIMARY_MODEL,
            GEMINI_OPENAI_BASE_URL,
            "gemini_key",
        )
    return (
        "openai",
        "",
        resolved_openai_model,
        None,
        "no_api_key",
    )


def openai_api_keys_merged(settings: Settings) -> list[str]:
    """All OpenAI keys (primary + ``NEXIS_OPENAI_API_KEYS``), deduped — same order as captioning."""
    return merge_openai_api_keys(settings.openai_api_key, settings.openai_api_keys_extra)
