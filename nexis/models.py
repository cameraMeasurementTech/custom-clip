"""Core schema models for Nexisgen miner submissions."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .protocol import CLIP_DURATION_SEC, SCHEMA_VERSION

_DEFAULT_SPEC_ID = "video_v1"


class ClipRecord(BaseModel):
    """Single training clip row submitted by a miner."""

    clip_id: str = Field(min_length=1)
    clip_uri: str = Field(min_length=1)
    clip_sha256: str = Field(min_length=64, max_length=64)
    first_frame_uri: str = Field(min_length=1)
    first_frame_sha256: str = Field(min_length=64, max_length=64)
    source_video_id: str = Field(min_length=1)
    split_group_id: str = Field(min_length=1)
    split: str = Field(min_length=1)
    clip_start_sec: float = Field(ge=0.0)
    duration_sec: float = Field(gt=0.0)
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    fps: float = Field(gt=0.0)
    num_frames: int = Field(gt=0)
    has_audio: bool
    caption: str = Field(min_length=1)
    source_video_url: str = Field(min_length=1)
    source_proof: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(str_strip_whitespace=True)

    @field_validator("duration_sec")
    @classmethod
    def validate_duration(cls, value: float) -> float:
        if value < CLIP_DURATION_SEC - 0.15 or value > CLIP_DURATION_SEC + 0.15:
            raise ValueError(
                f"duration_sec must be close to {CLIP_DURATION_SEC} seconds for v1 protocol"
            )
        return value

class IntervalManifest(BaseModel):
    """Interval-level metadata for miner submission package."""

    protocol_version: str = Field(default="1.0.0")
    schema_version: str = Field(default=SCHEMA_VERSION)
    spec_id: str = Field(default=_DEFAULT_SPEC_ID, min_length=1)
    dataset_type: str = Field(default=_DEFAULT_SPEC_ID, min_length=1)
    netuid: int = Field(ge=0)
    miner_hotkey: str = Field(min_length=1)
    interval_id: int = Field(ge=0)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    record_count: int = Field(ge=0)
    dataset_sha256: str = Field(min_length=64, max_length=64)

    model_config = ConfigDict(str_strip_whitespace=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_spec_metadata(cls, payload: Any) -> Any:
        if not isinstance(payload, dict):
            return payload
        data = dict(payload)
        spec_id = str(data.get("spec_id", "")).strip()
        dataset_type = str(data.get("dataset_type", "")).strip()
        resolved = spec_id or dataset_type or _DEFAULT_SPEC_ID
        data["spec_id"] = resolved
        data["dataset_type"] = resolved
        return data


class ValidationDecision(BaseModel):
    """Per-miner validator decision for one interval."""

    miner_hotkey: str
    interval_id: int
    accepted: bool
    failures: list[str] = Field(default_factory=list)
    record_count: int = 0
    sampled_rows: int = 0
    checked_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    notes: dict[str, Any] = Field(default_factory=dict)

