"""Dataset spec registry."""

from __future__ import annotations

from dataclasses import dataclass

from .base import DatasetSpec, SpecCompatibilityResult
from .video_v1 import VideoV1Spec


DEFAULT_SPEC_ID = "video_v1"


@dataclass
class DatasetSpecRegistry:
    _specs: dict[str, DatasetSpec]

    @classmethod
    def with_defaults(cls) -> "DatasetSpecRegistry":
        return cls(_specs={DEFAULT_SPEC_ID: VideoV1Spec()})

    def get(self, spec_id: str) -> DatasetSpec:
        key = (spec_id or "").strip()
        if key not in self._specs:
            raise KeyError(f"unknown dataset spec: {spec_id}")
        return self._specs[key]

    def list_spec_ids(self) -> list[str]:
        return sorted(self._specs.keys())

    def compatibility(
        self,
        *,
        spec_id: str,
        protocol_version: str,
        schema_version: str,
    ) -> SpecCompatibilityResult:
        try:
            spec = self.get(spec_id)
        except KeyError:
            return SpecCompatibilityResult(compatible=False, reason=f"unknown_spec:{spec_id}")
        if not spec.is_compatible(
            protocol_version=protocol_version,
            schema_version=schema_version,
        ):
            return SpecCompatibilityResult(
                compatible=False,
                reason=(
                    f"incompatible_spec_version:{spec_id}:"
                    f"protocol={protocol_version}:schema={schema_version}"
                ),
            )
        return SpecCompatibilityResult(compatible=True, reason="")
