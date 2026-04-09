"""Run mine-prepare without the ``nexis`` console script.

Example::

    python -m nexis.tools.mine_prepare_main \\
      --workdir /data/pool --sources urls.txt \\
      --openai-api-key sk-... --openai-api-keys-extra sk-b,sk-c

Or use ``.env`` and only pass paths::

    python -m nexis.tools.mine_prepare_main --workdir /data/pool --sources urls.txt
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from nexis.cli import _configure_logging, _resolve_enabled_specs
from nexis.config import load_settings
from nexis.miner.prepare_runner import run_mine_prepare
from nexis.specs import DEFAULT_SPEC_ID, DatasetSpecRegistry


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Mine-prepare (no nexis CLI entrypoint).")
    p.add_argument("--workdir", type=Path, required=True, help="Shared output directory.")
    p.add_argument(
        "--sources",
        type=Path,
        required=True,
        help="URL queue file (first line popped per claim).",
    )
    p.add_argument("--workers", type=int, default=1, help="Parallel workers (default 1).")
    p.add_argument(
        "--thread-keys-file",
        type=Path,
        default=None,
        help="One line per worker: comma-separated keys (required if workers > 1).",
    )
    p.add_argument(
        "--max-segments-per-worker",
        type=int,
        default=0,
        help="Stop each worker after N clips (0 = until queue empty).",
    )
    p.add_argument(
        "--rpm-per-key",
        type=int,
        default=None,
        help="Override NEXIS_CAPTION_OPENAI_RPM_PER_KEY (0 disables spacing).",
    )
    p.add_argument(
        "--spec",
        type=str,
        default="",
        help="Dataset spec ID (default NEXIS_DATASET_SPEC_DEFAULT).",
    )
    p.add_argument("--debug", action="store_true", help="Verbose logging.")
    p.add_argument(
        "--openai-api-key",
        type=str,
        default=None,
        help="Set OPENAI_API_KEY for this process (before loading .env settings).",
    )
    p.add_argument(
        "--openai-api-keys-extra",
        type=str,
        default=None,
        help="Set NEXIS_OPENAI_API_KEYS (comma-separated extra keys).",
    )
    p.add_argument(
        "--gemini-api-key",
        type=str,
        default=None,
        help="Set GEMINI_API_KEY for this process.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.openai_api_key is not None:
        os.environ["OPENAI_API_KEY"] = args.openai_api_key
    if args.openai_api_keys_extra is not None:
        os.environ["NEXIS_OPENAI_API_KEYS"] = args.openai_api_keys_extra
    if args.gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = args.gemini_api_key

    settings = load_settings()
    spec_registry = DatasetSpecRegistry.with_defaults()
    try:
        enabled_specs = _resolve_enabled_specs(settings.miner_enabled_specs, spec_registry)
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    active_spec = args.spec.strip() or settings.dataset_spec_default.strip() or DEFAULT_SPEC_ID
    active_category = settings.dataset_category.strip()
    _configure_logging("INFO", debug=args.debug)

    sources_path = args.sources.expanduser().resolve()
    if not sources_path.is_file():
        print(f"error: sources file not found: {sources_path}", file=sys.stderr)
        return 2

    thread_keys: Path | None = None
    if args.thread_keys_file is not None:
        thread_keys = args.thread_keys_file.expanduser().resolve()
        if not thread_keys.is_file():
            print(f"error: thread keys file not found: {thread_keys}", file=sys.stderr)
            return 2

    print(
        f"mine-prepare: workers={args.workers} workdir={args.workdir} sources={sources_path} "
        "(URLs are removed from the file as claimed)"
    )
    try:
        run_mine_prepare(
            settings=settings,
            workdir=args.workdir,
            sources_path=sources_path,
            workers=args.workers,
            thread_keys_file=thread_keys,
            max_segments_per_worker=args.max_segments_per_worker,
            rpm_per_key=args.rpm_per_key,
            active_spec=active_spec,
            active_category=active_category,
            spec_registry=spec_registry,
            spec_registry_enabled_ids=set(enabled_specs),
            spec_registry_all_ids=set(spec_registry.list_spec_ids()),
        )
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print("mine-prepare: all workers finished")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
