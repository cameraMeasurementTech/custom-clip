from __future__ import annotations

from pathlib import Path
from typing import Any

from nexis.hash_utils import sha256_file
from nexis.miner.pipeline import MinerPipeline
from nexis.serialization import read_dataset_parquet, read_manifest
from .helpers import LocalObjectStore, run_async


class _StubCaptioner:
    def caption_clip(
        self,
        clip_path: Path,
        source_url: str,
        first_frame_path: Path | None = None,
        frame_paths: list[Path] | None = None,
    ) -> str:
        _ = source_url, first_frame_path, frame_paths
        return f"caption for {clip_path.stem}"


class _StubSourceProvider:
    def read_sources(self, path: Path) -> list[str]:
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

    def source_video_id(self, url: str) -> str:
        _ = url
        return "video-abc"

    def download(self, url: str, output_dir: Path) -> Path:
        _ = url
        output_dir.mkdir(parents=True, exist_ok=True)
        target = output_dir / "video-abc.mp4"
        target.write_bytes(b"raw-video")
        return target

    def probe(self, path: Path) -> dict[str, Any]:
        _ = path
        return {
            "format": {"duration": "10.2"},
            "streams": [
                {"codec_type": "video", "width": 1280, "height": 720, "r_frame_rate": "30/1"},
                {"codec_type": "audio"},
            ],
        }

    def create_clip(self, src: Path, dst: Path, start_sec: float, duration_sec: float) -> None:
        _ = src, duration_sec
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f"clip-{start_sec:.3f}".encode("utf-8"))

    def extract_first_frame(self, src: Path, dst: Path) -> None:
        _ = src
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(f"frame-{dst.stem}".encode("utf-8"))

    def extract_caption_frames(self, src: Path, output_dir: Path, frame_count: int) -> list[Path]:
        _ = src
        output_dir.mkdir(parents=True, exist_ok=True)
        frames: list[Path] = []
        for index in range(min(2, frame_count)):
            frame = output_dir / f"caption_{index:03d}.jpg"
            frame.write_bytes(f"caption-{index}".encode("utf-8"))
            frames.append(frame)
        return frames


def test_miner_pipeline_writes_and_uploads_interval_package(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        pipeline = MinerPipeline(
            store=store,
            captioner=_StubCaptioner(),  # type: ignore[arg-type]
            source_provider=_StubSourceProvider(),  # type: ignore[arg-type]
            spec_id="video_v1",
        )
        sources_file = tmp_path / "sources.txt"
        sources_file.write_text("https://youtube.com/watch?v=abc\n", encoding="utf-8")
        interval_id = 101
        dataset_path, manifest_path = await pipeline.run_interval(
            sources_file=sources_file,
            netuid=1,
            miner_hotkey="miner1",
            interval_id=interval_id,
            workdir=tmp_path / "work",
        )

        records = read_dataset_parquet(dataset_path)
        assert len(records) == 2
        assert all(row.source_video_id == "video-abc" for row in records)
        assert all(row.duration_sec == 5.0 for row in records)

        manifest = read_manifest(manifest_path)
        assert manifest.spec_id == "video_v1"
        assert manifest.dataset_type == "video_v1"
        assert manifest.record_count == 2
        assert manifest.dataset_sha256 == sha256_file(dataset_path)

        assert await store.object_exists(f"{interval_id}/dataset.parquet")
        assert await store.object_exists(f"{interval_id}/manifest.json")
        for row in records:
            assert await store.object_exists(f"{interval_id}/{row.clip_uri}")
            assert await store.object_exists(f"{interval_id}/{row.first_frame_uri}")

    run_async(run())
