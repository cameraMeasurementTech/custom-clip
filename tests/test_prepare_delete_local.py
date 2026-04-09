"""Tests for post-upload local asset deletion."""

from __future__ import annotations

from pathlib import Path

from nexis.hash_utils import deterministic_clip_id
from nexis.miner.pending_pack import delete_local_assets_for_records, delete_local_out_interval
from nexis.models import ClipRecord
from nexis.protocol import CLIP_DURATION_SEC


def test_delete_local_assets_for_records_removes_clip_frame_and_caption_dir(tmp_path: Path) -> None:
    wd = tmp_path / "pool"
    cid = deterministic_clip_id("vid1", 0.0, CLIP_DURATION_SEC)
    (wd / "clips").mkdir(parents=True)
    (wd / "frames").mkdir(parents=True)
    clip_p = wd / "clips" / f"{cid}.mp4"
    frame_p = wd / "frames" / f"{cid}.jpg"
    clip_p.write_text("x", encoding="utf-8")
    frame_p.write_text("y", encoding="utf-8")
    cap_dir = wd / "frames" / cid / "caption"
    cap_dir.mkdir(parents=True)
    (cap_dir / "f.jpg").write_bytes(b"\xff")

    row = ClipRecord(
        clip_id=cid,
        clip_uri=f"clips/{cid}.mp4",
        clip_sha256="0" * 64,
        first_frame_uri=f"frames/{cid}.jpg",
        first_frame_sha256="0" * 64,
        source_video_id="vid1",
        split_group_id="vid1:pending",
        split="train",
        clip_start_sec=0.0,
        duration_sec=CLIP_DURATION_SEC,
        width=1280,
        height=720,
        fps=30.0,
        num_frames=150,
        has_audio=False,
        caption="test",
        source_video_url="https://youtube.com/watch?v=x",
        source_proof={"extractor": "yt-dlp", "source_video_id": "vid1"},
    )
    delete_local_assets_for_records(wd, [row])
    assert not clip_p.exists()
    assert not frame_p.exists()
    assert not (wd / "frames" / cid).exists()


def test_delete_local_assets_by_clip_id_when_uri_differs(tmp_path: Path) -> None:
    """Canonical clips/frames/{clip_id}.* are removed even if record URIs are wrong."""
    wd = tmp_path / "pool"
    cid = deterministic_clip_id("vid1", 0.0, CLIP_DURATION_SEC)
    (wd / "clips").mkdir(parents=True)
    (wd / "frames").mkdir(parents=True)
    clip_p = wd / "clips" / f"{cid}.mp4"
    frame_p = wd / "frames" / f"{cid}.jpg"
    clip_p.write_text("x", encoding="utf-8")
    frame_p.write_text("y", encoding="utf-8")

    row = ClipRecord(
        clip_id=cid,
        clip_uri="clips/wrong-name.mp4",
        clip_sha256="0" * 64,
        first_frame_uri="frames/wrong.jpg",
        first_frame_sha256="0" * 64,
        source_video_id="vid1",
        split_group_id="vid1:pending",
        split="train",
        clip_start_sec=0.0,
        duration_sec=CLIP_DURATION_SEC,
        width=1280,
        height=720,
        fps=30.0,
        num_frames=150,
        has_audio=False,
        caption="test",
        source_video_url="https://youtube.com/watch?v=x",
        source_proof={"extractor": "yt-dlp", "source_video_id": "vid1"},
    )
    delete_local_assets_for_records(wd, [row])
    assert not clip_p.exists()
    assert not frame_p.exists()


def test_delete_local_out_interval(tmp_path: Path) -> None:
    wd = tmp_path / "w"
    out = wd / "out" / "12345"
    out.mkdir(parents=True)
    (out / "manifest.json").write_text("{}", encoding="utf-8")
    delete_local_out_interval(wd, 12345)
    assert not out.exists()
