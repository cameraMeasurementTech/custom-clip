"""Hippius S3-compatible storage client."""

from __future__ import annotations

import hashlib
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from aiobotocore.config import AioConfig
from aiobotocore.session import get_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HippiusCredentials:
    bucket_name: str
    endpoint_url: str
    region: str
    read_access_key: str
    read_secret_key: str
    write_access_key: str
    write_secret_key: str

    def validate_bucket_name(self) -> None:
        if not self.bucket_name.strip():
            raise ValueError("bucket name is required")

    def validate_bucket_for_hotkey(self, hotkey: str) -> None:
        # Backward-compatible alias retained for callers that haven't migrated.
        _ = hotkey
        self.validate_bucket_name()

    @property
    def read_commitment(self) -> str:
        payload = f"{self.bucket_name}:{self.read_access_key}:{self.read_secret_key}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class HippiusS3Store:
    """S3 client wrapper for Hippius object storage."""

    def __init__(self, credentials: HippiusCredentials):
        self.credentials = credentials
        self._session = get_session()

    async def upload_file(self, key: str, src: Path, use_write: bool = True) -> None:
        access, secret = self._select_keys(use_write=use_write)
        logger.debug(
            "upload start bucket=%s key=%s use_write=%s size_bytes=%d",
            self.credentials.bucket_name,
            key,
            str(use_write),
            src.stat().st_size if src.exists() else -1,
        )
        async with self._session.create_client(
            "s3",
            endpoint_url=self.credentials.endpoint_url,
            region_name=self.credentials.region,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            config=AioConfig(signature_version="s3v4", connect_timeout=5, read_timeout=30),
        ) as client:
            with src.open("rb") as fp:
                await client.put_object(Bucket=self.credentials.bucket_name, Key=key, Body=fp)
        logger.info("upload complete bucket=%s key=%s", self.credentials.bucket_name, key)

    async def download_file(self, key: str, dst: Path) -> bool:
        access, secret = self._select_keys(use_write=False)
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self.credentials.endpoint_url,
                region_name=self.credentials.region,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                config=AioConfig(signature_version="s3v4", connect_timeout=5, read_timeout=30),
            ) as client:
                resp = await client.get_object(Bucket=self.credentials.bucket_name, Key=key)
                body = await resp["Body"].read()
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_bytes(body)
                logger.debug(
                    "download complete bucket=%s key=%s bytes=%d",
                    self.credentials.bucket_name,
                    key,
                    len(body),
                )
                return True
        except Exception as exc:  # pragma: no cover - network service dependent
            logger.warning("download failed for key=%s: %s", key, exc)
            return False

    async def object_exists(self, key: str) -> bool:
        access, secret = self._select_keys(use_write=False)
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self.credentials.endpoint_url,
                region_name=self.credentials.region,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                config=AioConfig(signature_version="s3v4"),
            ) as client:
                await client.head_object(Bucket=self.credentials.bucket_name, Key=key)
                logger.debug("object exists bucket=%s key=%s", self.credentials.bucket_name, key)
                return True
        except Exception:
            logger.debug("object missing bucket=%s key=%s", self.credentials.bucket_name, key)
            return False

    async def list_prefix(self, prefix: str) -> list[str]:
        access, secret = self._select_keys(use_write=False)
        async with self._session.create_client(
            "s3",
            endpoint_url=self.credentials.endpoint_url,
            region_name=self.credentials.region,
            aws_access_key_id=access,
            aws_secret_access_key=secret,
            config=AioConfig(signature_version="s3v4"),
        ) as client:
            paginator = client.get_paginator("list_objects_v2")
            keys: list[str] = []
            async for page in paginator.paginate(Bucket=self.credentials.bucket_name, Prefix=prefix):
                for item in page.get("Contents", []):
                    keys.append(item["Key"])
            return keys

    async def get_object_last_modified(self, key: str) -> datetime | None:
        access, secret = self._select_keys(use_write=False)
        try:
            async with self._session.create_client(
                "s3",
                endpoint_url=self.credentials.endpoint_url,
                region_name=self.credentials.region,
                aws_access_key_id=access,
                aws_secret_access_key=secret,
                config=AioConfig(signature_version="s3v4"),
            ) as client:
                resp = await client.head_object(Bucket=self.credentials.bucket_name, Key=key)
                value = resp.get("LastModified")
                return value if isinstance(value, datetime) else None
        except Exception:
            return None

    def _select_keys(self, use_write: bool) -> tuple[str, str]:
        if use_write:
            return self.credentials.write_access_key, self.credentials.write_secret_key
        return self.credentials.read_access_key, self.credentials.read_secret_key

