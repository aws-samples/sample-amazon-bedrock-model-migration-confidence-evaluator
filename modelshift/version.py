"""Build identity — lets the UI detect a stale server.

BUILD_ID is derived from the latest mtime + a content hash of the package's
source files, computed once at import (i.e. at server start). If any .py / web
asset changes and the server is restarted on the new code, the id changes;
a stale server keeps serving its old id, which the UI surfaces.
"""
from __future__ import annotations

import hashlib
import time
from datetime import datetime, timezone
from pathlib import Path

from . import __version__

_PKG_DIR = Path(__file__).resolve().parent
_WEB_DIR = _PKG_DIR.parent / "web"

# Process start time — distinguishes two servers started from identical code.
STARTED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
_STARTED_MONO = time.time()


def _source_fingerprint() -> tuple[str, str]:
    """(short content hash, latest source mtime ISO) across .py + web assets."""
    hasher = hashlib.sha256()
    latest = 0.0
    files = sorted(_PKG_DIR.rglob("*.py"))
    if _WEB_DIR.is_dir():
        files += sorted(_WEB_DIR.rglob("*"))
    for f in files:
        if not f.is_file() or "__pycache__" in f.parts:
            continue
        try:
            data = f.read_bytes()
        except OSError:
            continue
        hasher.update(f.name.encode())
        hasher.update(data)
        latest = max(latest, f.stat().st_mtime)
    mtime_iso = datetime.fromtimestamp(latest, timezone.utc).replace(microsecond=0).isoformat() if latest else ""
    return hasher.hexdigest()[:8], mtime_iso


_SRC_HASH, _SRC_MTIME = _source_fingerprint()

# Build id: version + source hash. Stable across restarts of identical code,
# changes whenever any source file changes.
BUILD_ID = f"{__version__}+{_SRC_HASH}"


def version_info() -> dict:
    return {
        "build_id": BUILD_ID,
        "version": __version__,
        "source_hash": _SRC_HASH,
        "source_mtime": _SRC_MTIME,
        "started_at": STARTED_AT,
        "uptime_seconds": int(time.time() - _STARTED_MONO),
    }
