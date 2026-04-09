"""Runtime configuration for Nexisgen."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from .protocol import INTERVAL_LENGTH_BLOCKS


load_dotenv(override=False)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    netuid: int = Field(default=0, alias="NEXIS_NETUID")
    log_level: str = Field(default="INFO", alias="NEXIS_LOG_LEVEL")
    bt_network: str = Field(default="finney", alias="BT_NETWORK")
    bt_wallet_name: str = Field(default="default", alias="BT_WALLET_NAME")
    bt_wallet_hotkey: str = Field(default="default", alias="BT_WALLET_HOTKEY")
    bt_wallet_path: Path = Field(default=Path("~/.bittensor/wallets"), alias="BT_WALLET_PATH")

    r2_account_id: str = Field(default="", alias="R2_ACCOUNT_ID")
    r2_region: str = Field(default="auto", alias="R2_REGION")
    r2_read_access_key: str = Field(default="", alias="R2_READ_ACCESS_KEY")
    r2_read_secret_key: str = Field(default="", alias="R2_READ_SECRET_KEY")
    r2_write_access_key: str = Field(default="", alias="R2_WRITE_ACCESS_KEY")
    r2_write_secret_key: str = Field(default="", alias="R2_WRITE_SECRET_KEY")

    sources_file: Path = Field(default=Path("sources.txt"), alias="NEXIS_SOURCES_FILE")
    workdir: Path = Field(default=Path(".nexis"), alias="NEXIS_WORKDIR")
    block_poll_sec: float = Field(default=6.0, alias="NEXIS_BLOCK_POLL_SEC")
    dataset_spec_default: str = Field(default="video_v1", alias="NEXIS_DATASET_SPEC_DEFAULT")
    dataset_category: str = Field(default="nature_landscape_scenery", alias="NEXIS_DATASET_CATEGORY")
    miner_enabled_specs: str = Field(default="video_v1", alias="NEXIS_MINER_ENABLED_SPECS")
    miner_continuous_pack_mode: bool = Field(
        default=True,
        alias="NEXIS_MINER_CONTINUOUS_PACK_MODE",
        description=(
            "Mine one 5s clip per loop into miner_pending_records.jsonl; upload deduped batches on cadence. "
            "If false, use legacy one full run_interval per INTERVAL_LENGTH_BLOCKS chain interval."
        ),
    )
    miner_upload_cadence_blocks: int = Field(
        default=INTERVAL_LENGTH_BLOCKS,
        alias="NEXIS_MINER_UPLOAD_CADENCE_BLOCKS",
        description=(
            "Minimum blocks between R2 pack upload attempts in continuous mode. "
            f"Default equals INTERVAL_LENGTH_BLOCKS ({INTERVAL_LENGTH_BLOCKS}): one attempt per chain interval."
        ),
    )
    miner_r2_prune_old_prefix: bool = Field(
        default=True,
        alias="NEXIS_MINER_R2_PRUNE_OLD_PREFIX",
        description=(
            "After a successful interval pack upload, delete the R2 object prefix from N uploads ago "
            "(see NEXIS_MINER_R2_PRUNE_UPLOADS_AGO) to save bucket space."
        ),
    )
    miner_r2_prune_uploads_ago: int = Field(
        default=2,
        alias="NEXIS_MINER_R2_PRUNE_UPLOADS_AGO",
        description=(
            "When pruning, remove the interval folder from this many successful uploads before the current one "
            "(e.g. 2 = delete two uploads ago while uploading the latest)."
        ),
    )
    miner_upload_manifest_busy_wait_sec: float = Field(
        default=1200.0,
        alias="NEXIS_MINER_UPLOAD_MANIFEST_BUSY_WAIT_SEC",
        description=(
            "If {interval_id}/manifest.json already exists on R2 before upload, sleep this many seconds "
            "and retry (0 = do not wait / skip busy check). When an interval refresher is set, the interval id "
            "is recomputed from chain after each wait."
        ),
    )
    miner_prepare_validate_before_save: bool = Field(
        default=True,
        alias="NEXIS_MINER_PREPARE_VALIDATE_BEFORE_SAVE",
        description=(
            "When appending to miner_pending_records.jsonl (mine-prepare / mine_one_segment), merge under lock: "
            "keep only rows that pass validator hard checks, fix overlaps with existing pending, verify the new "
            "row's local assets (and optional LLM gates). Rejected clips are not written; removed older overlaps "
            "are deleted from the file and from disk."
        ),
    )
    miner_prepare_asset_verify: bool = Field(
        default=True,
        alias="NEXIS_MINER_PREPARE_ASSET_VERIFY",
        description=(
            "With prepare validation enabled, run local 720p/sha checks on each new row before writing pending. "
            "Set false to only enforce hard checks (YouTube, caption, overlap) if clips are not yet 1280×720."
        ),
    )
    miner_preflight_before_upload: bool = Field(
        default=True,
        alias="NEXIS_MINER_PREFLIGHT_BEFORE_UPLOAD",
        description=(
            "Before R2 pack upload, re-run validator-style filtering on pending (default off when "
            "NEXIS_MINER_PREPARE_VALIDATE_BEFORE_SAVE=true). Enable for legacy workdirs or extra safety."
        ),
    )
    miner_preflight_semantic: bool = Field(
        default=True,
        alias="NEXIS_MINER_PREFLIGHT_SEMANTIC",
        description=(
            "During miner preflight, also run caption semantic checks (validator-style). "
            "Uses LLM quota; enable only if you want miner-side parity with semantic validation."
        ),
    )
    miner_preflight_category: bool = Field(
        default=True,
        alias="NEXIS_MINER_PREFLIGHT_CATEGORY",
        description=(
            "During miner preflight, also run strict category checks (validator-style). "
            "Uses LLM quota."
        ),
    )
    validator_enabled_specs: str = Field(
        default="video_v1",
        alias="NEXIS_VALIDATOR_ENABLED_SPECS",
    )
    validator_blacklist_file: Path = Field(
        default=Path("validator_blacklist_hotkeys.txt"),
        alias="NEXIS_VALIDATOR_BLACKLIST_FILE",
    )

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_api_keys_extra: str = Field(
        default="",
        alias="NEXIS_OPENAI_API_KEYS",
        description=(
            "Comma or newline separated extra OpenAI API keys for miner captions. Merged with OPENAI_API_KEY "
            "(deduplicated). Use NEXIS_CAPTION_OPENAI_RPM_PER_KEY=3 per key (~6/min with two keys)."
        ),
    )
    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    caption_model: str = Field(default="gpt-4o", alias="NEXIS_CAPTION_MODEL")
    caption_timeout_sec: int = Field(default=30, alias="NEXIS_CAPTION_TIMEOUT_SEC")
    caption_rate_limit_max_attempts: int = Field(
        default=15,
        alias="NEXIS_CAPTION_RATE_LIMIT_MAX_ATTEMPTS",
        description="OpenAI-compatible caption calls: max tries per clip on 429 rate limits.",
    )
    caption_rate_limit_max_sleep_sec: float = Field(
        default=120.0,
        alias="NEXIS_CAPTION_RATE_LIMIT_MAX_SLEEP_SEC",
        description="Cap single sleep between 429 retries (seconds).",
    )
    caption_openai_rpm_per_key: int = Field(
        default=3,
        alias="NEXIS_CAPTION_OPENAI_RPM_PER_KEY",
        description=(
            "Max caption requests per minute per OpenAI API key (proactive spacing). "
            "With 2 keys at 3 RPM each you get up to ~6 requests/minute. Set 0 to disable spacing."
        ),
    )
    validator_semantic_check_enabled: bool = Field(
        default=True,
        alias="NEXIS_VALIDATOR_SEMANTIC_CHECK_ENABLED",
    )
    validator_semantic_model: str = Field(
        default="gpt-4o",
        alias="NEXIS_VALIDATOR_SEMANTIC_MODEL",
    )
    validator_semantic_timeout_sec: int = Field(
        default=20,
        alias="NEXIS_VALIDATOR_SEMANTIC_TIMEOUT_SEC",
    )
    validator_semantic_max_samples: int = Field(
        default=8,
        alias="NEXIS_VALIDATOR_SEMANTIC_MAX_SAMPLES",
    )
    validator_semantic_max_transient_retries: int = Field(
        default=5,
        alias="NEXIS_VALIDATOR_SEMANTIC_MAX_TRANSIENT_RETRIES",
        description=(
            "After a transient LLM error (429, 5xx, timeout), retry this many times before "
            "failing the clip with caption_semantic_transient_exhausted."
        ),
    )
    validator_semantic_retry_base_sleep_sec: float = Field(
        default=2.0,
        alias="NEXIS_VALIDATOR_SEMANTIC_RETRY_BASE_SLEEP_SEC",
        description="Base seconds for exponential backoff when the API does not suggest a wait time.",
    )
    validator_semantic_retry_sleep_cap_sec: float = Field(
        default=120.0,
        alias="NEXIS_VALIDATOR_SEMANTIC_RETRY_SLEEP_CAP_SEC",
        description="Maximum sleep between semantic-check retries (seconds).",
    )
    validator_category_check_enabled: bool = Field(
        default=True,
        alias="NEXIS_VALIDATOR_CATEGORY_CHECK_ENABLED",
    )
    validator_category_model: str = Field(
        default="gpt-4o",
        alias="NEXIS_VALIDATOR_CATEGORY_MODEL",
    )
    validator_category_timeout_sec: int = Field(
        default=20,
        alias="NEXIS_VALIDATOR_CATEGORY_TIMEOUT_SEC",
    )
    validator_category_max_samples: int = Field(
        default=8,
        alias="NEXIS_VALIDATOR_CATEGORY_MAX_SAMPLES",
    )

    owner_validator_hotkey: str = Field(
        default="5EUdjwHz9pW4ftQQQga9PKq7knGiGW9wcHUjkDSih7zpovPy",
        alias="NEXIS_OWNER_VALIDATOR_HOTKEY",
    )

    owner_db_bucket: str = Field(default="nexis-dataset", alias="NEXIS_OWNER_DB_BUCKET")
    owner_db_account_id: str = Field(default="", alias="NEXIS_OWNER_DB_ACCOUNT_ID")
    owner_db_read_access_key: str = Field(default="", alias="NEXIS_OWNER_DB_READ_ACCESS_KEY")
    owner_db_read_secret_key: str = Field(default="", alias="NEXIS_OWNER_DB_READ_SECRET_KEY")
    owner_db_write_access_key: str = Field(default="", alias="NEXIS_OWNER_DB_WRITE_ACCESS_KEY")
    owner_db_write_secret_key: str = Field(default="", alias="NEXIS_OWNER_DB_WRITE_SECRET_KEY")

    record_info_bucket: str = Field(default="nexis-record-info", alias="NEXIS_RECORD_INFO_BUCKET")
    # record_info_account_id: str = Field(default="", alias="NEXIS_RECORD_INFO_ACCOUNT_ID")
    # record_info_read_access_key: str = Field(default="", alias="NEXIS_RECORD_INFO_READ_ACCESS_KEY")
    # record_info_read_secret_key: str = Field(default="", alias="NEXIS_RECORD_INFO_READ_SECRET_KEY")
    record_info_account_id: str = "cce499ad4f3a4703b069771d8ff4215a"
    record_info_read_access_key: str = "0fa291e03819c60474fed86a4932e652"
    record_info_read_secret_key: str = "7bfbc213f3295c0a7f88db3f069490ce474e82520b4455b6a7bc7aa5e66224ee"
    record_info_write_access_key: str = Field(default="", alias="NEXIS_RECORD_INFO_WRITE_ACCESS_KEY")
    record_info_write_secret_key: str = Field(default="", alias="NEXIS_RECORD_INFO_WRITE_SECRET_KEY")
    record_info_object_key: str = Field(default="record_info.json", alias="NEXIS_RECORD_INFO_OBJECT_KEY")

    # Optional validator -> evidence API reporting
    validation_api_url: str = Field(default="https://api.nexisgen.ai/v1/validation-results", alias="NEXIS_VALIDATION_API_URL")
    validation_api_timeout_sec: float = Field(default=120.0, alias="NEXIS_VALIDATION_API_TIMEOUT_SEC")
    latest_result_timeout_sec: float = Field(
        default=120.0,
        alias="NEXIS_LATEST_RESULT_TIMEOUT_SEC",
        description="GET /v1/get_latest_result can return a large JSON window; allow a generous read timeout.",
    )

    # Evidence API server settings
    validation_api_postgres_dsn: str = Field(
        default="postgresql://nexis:nexis@localhost:5432/nexis_validation",
        alias="NEXIS_VALIDATION_API_POSTGRES_DSN",
    )
    validation_api_allowlist_refresh_sec: int = Field(
        default=300,
        alias="NEXIS_VALIDATION_API_ALLOWLIST_REFRESH_SEC",
    )
    validation_api_min_validator_stake: float = Field(
        default=5000.0,
        alias="NEXIS_VALIDATION_API_MIN_VALIDATOR_STAKE",
    )
    validation_api_auth_max_skew_sec: int = Field(
        default=300,
        alias="NEXIS_VALIDATION_API_AUTH_MAX_SKEW_SEC",
    )
    validation_api_nonce_max_age_sec: int = Field(
        default=86400,
        alias="NEXIS_VALIDATION_API_NONCE_MAX_AGE_SEC",
    )

    @field_validator("miner_upload_cadence_blocks")
    @classmethod
    def _cadence_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("NEXIS_MINER_UPLOAD_CADENCE_BLOCKS must be >= 1")
        return value

    @field_validator("miner_r2_prune_uploads_ago")
    @classmethod
    def _prune_uploads_ago_positive(cls, value: int) -> int:
        if value < 1:
            raise ValueError("NEXIS_MINER_R2_PRUNE_UPLOADS_AGO must be >= 1")
        return value

    @field_validator("miner_upload_manifest_busy_wait_sec")
    @classmethod
    def _manifest_busy_wait_non_negative(cls, value: float) -> float:
        if value < 0:
            raise ValueError("NEXIS_MINER_UPLOAD_MANIFEST_BUSY_WAIT_SEC must be >= 0")
        return value

    @field_validator("caption_openai_rpm_per_key")
    @classmethod
    def _caption_rpm_non_negative(cls, value: int) -> int:
        if value < 0:
            raise ValueError("NEXIS_CAPTION_OPENAI_RPM_PER_KEY must be >= 0")
        return value


def load_settings() -> Settings:
    return Settings()

