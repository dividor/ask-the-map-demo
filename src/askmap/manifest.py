"""``manifest.json``: one entry per downloaded or built file with its source, SHA-256 and time written.

Adapted from the author's earlier NASA POWER downloader (``powerwm.data.power`` in the jepa-world-model
project). A file counts as done only if its SHA-256 still matches the manifest, so interrupted builds
resume safely.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while chunk := fh.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


class Manifest:
    """``manifest.json`` in ``root``: one entry per file name with source, SHA-256, size and time written."""

    def __init__(self, root: Path):
        self.path = root / "manifest.json"
        self.entries: dict[str, dict] = json.loads(self.path.read_text()) if self.path.exists() else {}

    def is_current(self, file: Path) -> bool:
        entry = self.entries.get(file.name)
        return entry is not None and file.exists() and sha256_of(file) == entry["sha256"]

    def record(self, file: Path, **info) -> None:
        self.entries[file.name] = {
            **info,
            "sha256": sha256_of(file),
            "bytes": file.stat().st_size,
            "written_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.entries, indent=2, sort_keys=True))
