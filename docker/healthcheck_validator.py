"""Container healthcheck for validator runtime."""

from __future__ import annotations

from pathlib import Path
import sys


def main() -> int:
    try:
        raw = Path("/proc/1/cmdline").read_bytes()
    except OSError:
        return 1

    command = raw.replace(b"\x00", b" ").decode("utf-8", errors="ignore").strip().lower()
    if "nexis" in command and "validate" in command:
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
