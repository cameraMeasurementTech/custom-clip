"""Video v1 dataset spec adapter."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import parse_qs, urlparse

from ..models import ClipRecord
from ..validator.assets import SampledAssetVerifier, VideoAssetVerifier
from ..validator.checks import CheckResult, run_hard_checks


@dataclass(frozen=True)
class VideoV1Spec:
    spec_id: str = "video_v1"
    supported_protocol_versions: set[str] = frozenset({"1.0.0"})  # type: ignore[assignment]
    supported_schema_versions: set[str] = frozenset({"1.0.0"})  # type: ignore[assignment]
    row_model: type[ClipRecord] = ClipRecord

    def run_hard_checks(self, records: list[ClipRecord]) -> CheckResult:
        return run_hard_checks(records)

    def build_asset_verifier(self) -> SampledAssetVerifier:
        return VideoAssetVerifier()

    def source_identity_key(self, row: ClipRecord) -> str:
        canonical = self._canonical_source_key_from_url(row.source_video_url)
        if canonical:
            return canonical
        return row.source_video_id.strip() or row.source_video_url.strip()

    def source_identity_keys(self, row: ClipRecord) -> list[str]:
        keys: list[str] = []
        canonical = self._canonical_source_key_from_url(row.source_video_url)
        if canonical:
            keys.append(canonical)
        source_id = row.source_video_id.strip()
        if source_id:
            keys.append(source_id)
        source_url = row.source_video_url.strip()
        if source_url and source_url not in keys:
            keys.append(source_url)
        return keys

    def overlap_index_keys(self, row: ClipRecord) -> list[str]:
        keys: list[str] = []
        for key in self.source_identity_keys(row):
            keys.append(key)
            keys.append(f"{self.spec_id}:{key}")
        return keys

    def is_compatible(self, *, protocol_version: str, schema_version: str) -> bool:
        return (
            protocol_version in self.supported_protocol_versions
            and schema_version in self.supported_schema_versions
        )

    def _canonical_source_key_from_url(self, source_video_url: str) -> str:
        parsed = urlparse(source_video_url.strip())
        host = (parsed.hostname or "").lower()
        if host == "youtu.be":
            key = parsed.path.strip("/")
            return key or source_video_url.strip()
        if host == "youtube.com" or host.endswith(".youtube.com"):
            query = parse_qs(parsed.query)
            values = query.get("v", [])
            if values and values[0]:
                return values[0].strip()
            path_parts = [part for part in parsed.path.split("/") if part]
            if len(path_parts) >= 2 and path_parts[0] in {"shorts", "embed", "v"}:
                return path_parts[1].strip()
        return source_video_url.strip()
