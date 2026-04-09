"""Run mine-upload without the ``nexis`` console script.

Computes chain ``interval_id`` (aligned to ``INTERVAL_LENGTH_BLOCKS``) from the
current block unless you pass ``--interval-id``. Repeats on a wall-clock
schedule (default: every 30 minutes).

Example::

    python3 -m nexis.tools.mine_upload_main --workdir /data/pool

One shot (no loop), still using chain-derived interval id::

    python3 -m nexis.tools.mine_upload_main --workdir /data/pool --every once

Override interval id (advanced)::

    python3 -m nexis.tools.mine_upload_main --workdir /data/pool --interval-id 123456

Wallet and R2 settings come from ``.env`` / environment (same as ``nexis mine-upload``).
Before uploading, if ``{interval_id}/manifest.json`` already exists on R2, the process sleeps
(default **20 minutes** from ``NEXIS_MINER_UPLOAD_MANIFEST_BUSY_WAIT_SEC``) and retries; with a
chain-derived interval id, the id is **refreshed** after each wait. Override with ``--manifest-busy-wait-sec``.

After each successful upload, the R2 prefix from **two uploads ago** can be deleted
(``NEXIS_MINER_R2_PRUNE_OLD_PREFIX``, ``NEXIS_MINER_R2_PRUNE_UPLOADS_AGO``); use ``--no-r2-prune`` to disable.
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from collections.abc import Awaitable, Callable
from pathlib import Path

from nexis.chain.metagraph import fetch_current_block
from nexis.cli import (
    _build_remote_credentials,
    _configure_logging,
    _resolve_enabled_specs,
    _resolve_hotkey_ss58_from_wallet,
)
from nexis.config import Settings, load_settings
from nexis.miner.upload_runner import run_mine_upload
from nexis.protocol import INTERVAL_LENGTH_BLOCKS
from nexis.specs import DEFAULT_SPEC_ID, DatasetSpecRegistry
from nexis.storage.r2 import R2S3Store


def _parse_every_seconds(value: str) -> float:
    """Return seconds between upload attempts; 0 means run once and exit."""
    v = value.strip().lower().replace(" ", "")
    if v in ("0", "once", "one-shot", "single"):
        return 0.0
    m = re.fullmatch(r"(\d+(?:\.\d+)?)([smh])", v)
    if not m:
        raise argparse.ArgumentTypeError(
            "expected duration like 30m, 1h, 90s, or once / 0 for a single run"
        )
    num, unit = float(m.group(1)), m.group(2)
    mult = {"s": 1.0, "m": 60.0, "h": 3600.0}[unit]
    return num * mult


def _interval_id_for_chain(settings, override: int | None) -> int:
    if override is not None:
        return override
    block = fetch_current_block(network=settings.bt_network)
    return block - (block % INTERVAL_LENGTH_BLOCKS)


def _make_async_interval_refresher(settings: Settings):
    """Recompute chain interval id inside the async upload path (after each manifest busy wait)."""

    async def refresh() -> int:
        from nexis.chain.metagraph import fetch_current_block_async

        b = await fetch_current_block_async(network=settings.bt_network)
        return b - (b % INTERVAL_LENGTH_BLOCKS)

    return refresh


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Upload pending clips to R2; chain interval id is derived from the current block "
            f"(multiple of {INTERVAL_LENGTH_BLOCKS}) unless --interval-id is set. "
            "Repeats on a wall-clock schedule: --interval / --every (default 30m); use once for one upload."
        )
    )
    p.add_argument("--workdir", type=Path, required=True, help="Directory with pending clips.")
    p.add_argument(
        "--every",
        "--interval",
        type=_parse_every_seconds,
        default=_parse_every_seconds("30m"),
        dest="every",
        metavar="DURATION",
        help=(
            "Wall-clock time between upload attempts (same as --interval): "
            "e.g. 30m, 1h, 90s; once or 0 = single run (default: 30m)."
        ),
    )
    p.add_argument(
        "--interval-id",
        type=int,
        default=None,
        help=(
            "R2 prefix / chain interval start block (multiple of "
            f"{INTERVAL_LENGTH_BLOCKS}). Default: derived from current chain block."
        ),
    )
    p.add_argument(
        "--delete-local",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="After upload, delete local clips/frames (default: true).",
    )
    p.add_argument(
        "--no-r2-prune",
        action="store_true",
        help="Do not delete older R2 interval prefixes (see NEXIS_MINER_R2_PRUNE_OLD_PREFIX).",
    )
    p.add_argument(
        "--r2-prune-uploads-ago",
        type=int,
        default=None,
        help="Override NEXIS_MINER_R2_PRUNE_UPLOADS_AGO (how many uploads back to delete on R2).",
    )
    p.add_argument(
        "--manifest-busy-wait-sec",
        type=float,
        default=None,
        help=(
            "Override NEXIS_MINER_UPLOAD_MANIFEST_BUSY_WAIT_SEC: seconds to sleep when "
            f"{INTERVAL_LENGTH_BLOCKS}-block interval manifest already exists on R2 (0 disables wait/check)."
        ),
    )
    p.add_argument(
        "--spec",
        type=str,
        default="",
        help="Dataset spec ID (default NEXIS_DATASET_SPEC_DEFAULT).",
    )
    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    return p.parse_args(argv)


def _run_one_upload(
    *,
    settings: Settings,
    store: R2S3Store,
    wd: Path,
    interval_id: int,
    active_spec: str,
    active_category: str,
    enabled_specs: set[str],
    spec_all_ids: set[str],
    delete_local: bool,
    hotkey_ss58: str,
    r2_prune_old_prefix: bool | None,
    r2_prune_uploads_ago: int | None,
    manifest_busy_wait_sec: float | None,
    interval_refresher: Callable[[], Awaitable[int]] | None,
) -> int:
    try:
        run_mine_upload(
            settings=settings,
            store=store,
            workdir=wd,
            interval_id=interval_id,
            active_spec=active_spec,
            active_category=active_category,
            spec_registry_enabled_ids=enabled_specs,
            spec_registry_all_ids=spec_all_ids,
            delete_local=delete_local,
            miner_hotkey=hotkey_ss58,
            emit=print,
            r2_prune_old_prefix=r2_prune_old_prefix,
            r2_prune_uploads_ago=r2_prune_uploads_ago,
            manifest_busy_wait_sec=manifest_busy_wait_sec,
            interval_refresher=interval_refresher,
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 0


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    settings = load_settings()
    spec_registry = DatasetSpecRegistry.with_defaults()
    try:
        enabled_specs_list = _resolve_enabled_specs(settings.miner_enabled_specs, spec_registry)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    active_spec = args.spec.strip() or settings.dataset_spec_default.strip() or DEFAULT_SPEC_ID
    active_category = settings.dataset_category.strip()
    _configure_logging("INFO", debug=args.debug)

    try:
        hotkey_ss58 = _resolve_hotkey_ss58_from_wallet(settings)
    except Exception as exc:
        print(f"error: wallet/hotkey: {exc}", file=sys.stderr)
        return 2

    creds = _build_remote_credentials(settings, hotkey=hotkey_ss58)
    try:
        creds.validate_account_id()
        creds.validate_read_key_lengths()
        creds.validate_bucket_name()
        creds.validate_bucket_for_hotkey(hotkey_ss58)
    except Exception as exc:
        print(f"error: R2 credentials: {exc}", file=sys.stderr)
        return 2

    store = R2S3Store(creds)
    wd = args.workdir.expanduser().resolve()
    enabled_specs = set(enabled_specs_list)
    spec_all_ids = set(spec_registry.list_spec_ids())
    r2_prune_old_prefix: bool | None = False if args.no_r2_prune else None
    r2_prune_uploads_ago: int | None = args.r2_prune_uploads_ago
    if r2_prune_uploads_ago is not None and r2_prune_uploads_ago < 1:
        print("error: --r2-prune-uploads-ago must be >= 1", file=sys.stderr)
        return 2

    interval_refresher = _make_async_interval_refresher(settings) if args.interval_id is None else None
    manifest_busy_wait_sec = args.manifest_busy_wait_sec

    every = args.every
    try:
        while True:
            interval_id = _interval_id_for_chain(settings, args.interval_id)
            src = "override" if args.interval_id is not None else "chain"
            print(
                f"mine_upload_main: interval_id={interval_id} ({src}, "
                f"INTERVAL_LENGTH_BLOCKS={INTERVAL_LENGTH_BLOCKS})"
            )
            rc = _run_one_upload(
                settings=settings,
                store=store,
                wd=wd,
                interval_id=interval_id,
                active_spec=active_spec,
                active_category=active_category,
                enabled_specs=enabled_specs,
                spec_all_ids=spec_all_ids,
                delete_local=args.delete_local,
                hotkey_ss58=hotkey_ss58,
                r2_prune_old_prefix=r2_prune_old_prefix,
                r2_prune_uploads_ago=r2_prune_uploads_ago,
                manifest_busy_wait_sec=manifest_busy_wait_sec,
                interval_refresher=interval_refresher,
            )
            if rc != 0:
                return rc
            if every <= 0:
                break
            print(f"mine_upload_main: sleeping {every:.0f}s until next upload attempt")
            time.sleep(every)
    except KeyboardInterrupt:
        print("mine_upload_main: stopped", file=sys.stderr)
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
