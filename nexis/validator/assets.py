"""Asset verifier interfaces and default video implementation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from pathlib import PurePosixPath
from typing import Any, Protocol

from ..hash_utils import sha256_file
from ..miner.youtube import extract_caption_frames
from ..models import ClipRecord

_SEMANTIC_FRAME_COUNT = 6


@dataclass
class AssetVerificationResult:
    failures: list[str]
    semantic_frames_by_clip_id: dict[str, list[Path]]
    first_frames_by_clip_id: dict[str, Path]


class SampledAssetVerifier(Protocol):
    async def verify(
        self,
        *,
        store: Any,
        key_base: str,
        sampled: list[ClipRecord],
        miner_dir: Path,
    ) -> AssetVerificationResult: ...


class VideoAssetVerifier:
    """Verifies sampled clip/frame assets for video_v1 submissions."""

    async def verify(
        self,
        *,
        store: Any,
        key_base: str,
        sampled: list[ClipRecord],
        miner_dir: Path,
    ) -> AssetVerificationResult:
        failures: list[str] = []
        semantic_frames_by_clip_id: dict[str, list[Path]] = {}
        first_frames_by_clip_id: dict[str, Path] = {}
        for row in sampled:
            clip_path = miner_dir / row.clip_uri.lstrip("/")
            frame_path = miner_dir / row.first_frame_uri.lstrip("/")
            clip_asset_failure = await self._verify_asset(
                store=store,
                key_base=key_base,
                relative_uri=row.clip_uri,
                expected_sha256=row.clip_sha256,
                target_path=clip_path,
                missing_code=f"missing_clip_asset:{row.clip_id}",
                mismatch_code=f"clip_sha256_mismatch:{row.clip_id}",
            )
            if clip_asset_failure:
                failures.append(clip_asset_failure)

            frame_asset_failure = await self._verify_asset(
                store=store,
                key_base=key_base,
                relative_uri=row.first_frame_uri,
                expected_sha256=row.first_frame_sha256,
                target_path=frame_path,
                missing_code=f"missing_first_frame_asset:{row.clip_id}",
                mismatch_code=f"first_frame_sha256_mismatch:{row.clip_id}",
            )
            if frame_asset_failure:
                failures.append(frame_asset_failure)
            else:
                first_frames_by_clip_id[row.clip_id] = frame_path

            if clip_asset_failure is None:
                clip_frames = extract_caption_frames(
                    clip_path,
                    miner_dir / "semantic" / row.clip_id,
                    frame_count=_SEMANTIC_FRAME_COUNT,
                )
                if frame_asset_failure is None:
                    if not clip_frames:
                        clip_frames = [frame_path]
                    while len(clip_frames) < _SEMANTIC_FRAME_COUNT:
                        clip_frames.append(frame_path)
                    clip_frames.append(frame_path)
                semantic_frames_by_clip_id[row.clip_id] = [
                    frame for frame in clip_frames if frame.exists()
                ]
        return AssetVerificationResult(
            failures=failures,
            semantic_frames_by_clip_id=semantic_frames_by_clip_id,
            first_frames_by_clip_id=first_frames_by_clip_id,
        )

    async def _verify_asset(
        self,
        *,
        store: Any,
        key_base: str,
        relative_uri: str,
        expected_sha256: str,
        target_path: Path,
        missing_code: str,
        mismatch_code: str,
    ) -> str | None:
        safe_uri = self._normalize_relative_uri(relative_uri)
        if safe_uri is None:
            return f"invalid_asset_uri:{relative_uri}"
        ok = await store.download_file(f"{key_base}/{safe_uri}", target_path)
        if not ok:
            return missing_code
        if sha256_file(target_path) != expected_sha256:
            return mismatch_code
        return None

    def _normalize_relative_uri(self, value: str) -> str | None:
        text = value.strip().lstrip("/")
        if not text:
            return None
        parts = PurePosixPath(text).parts
        if any(part in {"", ".", ".."} for part in parts):
            return None
        return "/".join(parts)
