"""Atomic pop of next URL from a shared sources file (multi-worker prepare)."""

from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

try:
    import fcntl
except ImportError:  # pragma: no cover
    fcntl = None  # type: ignore[misc, assignment]


def _sources_lock_path(sources_file: Path) -> Path:
    return sources_file.expanduser().resolve().with_name(
        sources_file.name + ".queue.lock"
    )


@contextmanager
def _posix_flock_lock(lock_path: Path):
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_fd = open(lock_path, "a+b")
    try:
        if fcntl is not None:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        if fcntl is not None:
            fcntl.flock(lock_fd.fileno(), fcntl.LOCK_UN)
        lock_fd.close()


def pop_next_source_url(sources_file: Path) -> str | None:
    """Remove and return the first non-empty, non-comment line from sources_file; return None if none.

    Uses a dedicated lock file next to the sources file. POSIX only (fcntl).
    """
    path = sources_file.expanduser().resolve()
    if fcntl is None:
        logger.warning("fcntl unavailable; pop_next_source_url is not atomic on this platform")
        if not path.is_file():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        popped: str | None = None
        rest: list[str] = []
        for line in lines:
            s = line.strip()
            if popped is None and s and not s.startswith("#"):
                popped = s
            else:
                rest.append(line)
        if popped is not None:
            path.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")
        return popped

    lock_path = _sources_lock_path(path)
    with _posix_flock_lock(lock_path):
        if not path.is_file():
            return None
        lines = path.read_text(encoding="utf-8").splitlines()
        popped: str | None = None
        rest: list[str] = []
        for line in lines:
            s = line.strip()
            if popped is None and s and not s.startswith("#"):
                popped = s
            else:
                rest.append(line)
        if popped is not None:
            path.write_text("\n".join(rest) + ("\n" if rest else ""), encoding="utf-8")
        return popped
