from __future__ import annotations

from pathlib import Path

import pytest

from nexis.storage.hippius import HippiusCredentials
from .helpers import LocalObjectStore, run_async


def test_local_store_upload_download(tmp_path: Path) -> None:
    async def run() -> None:
        store = LocalObjectStore(tmp_path / "store")
        src = tmp_path / "src.txt"
        dst = tmp_path / "dst.txt"
        src.write_text("hello", encoding="utf-8")
        await store.upload_file("prefix/file.txt", src)
        assert await store.object_exists("prefix/file.txt")
        ok = await store.download_file("prefix/file.txt", dst)
        assert ok
        assert dst.read_text(encoding="utf-8") == "hello"
        keys = await store.list_prefix("prefix/")
        assert keys == ["prefix/file.txt"]

    run_async(run())


def test_hippius_credentials_bucket_name_validation_is_not_hotkey_bound() -> None:
    creds = HippiusCredentials(
        bucket_name="custom-bucket",
        endpoint_url="https://s3.hippius.com",
        region="decentralized",
        read_access_key="ak",
        read_secret_key="sk",
        write_access_key="wk",
        write_secret_key="ws",
    )
    creds.validate_bucket_name()
    # Backward-compatible method should no longer enforce hotkey equality.
    creds.validate_bucket_for_hotkey("different-hotkey")

    with pytest.raises(ValueError):
        HippiusCredentials(
            bucket_name="",
            endpoint_url="https://s3.hippius.com",
            region="decentralized",
            read_access_key="ak",
            read_secret_key="sk",
            write_access_key="wk",
            write_secret_key="ws",
        ).validate_bucket_name()

