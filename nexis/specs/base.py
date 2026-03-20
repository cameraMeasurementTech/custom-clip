"""Dataset spec interfaces for multi-spec subnet support."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from pydantic import BaseModel

from ..models import ClipRecord
from ..validator.assets import SampledAssetVerifier
from ..validator.checks import CheckResult


class DatasetSpec(Protocol):
    spec_id: str
    supported_protocol_versions: set[str]
    supported_schema_versions: set[str]
    row_model: type[BaseModel]

    def run_hard_checks(self, records: list[ClipRecord]) -> CheckResult: ...

    def build_asset_verifier(self) -> SampledAssetVerifier | None: ...

    def source_identity_key(self, row: ClipRecord) -> str: ...

    def source_identity_keys(self, row: ClipRecord) -> list[str]: ...

    def overlap_index_keys(self, row: ClipRecord) -> list[str]: ...

    def is_compatible(self, *, protocol_version: str, schema_version: str) -> bool: ...


@dataclass(frozen=True)
class SpecCompatibilityResult:
    compatible: bool
    reason: str
