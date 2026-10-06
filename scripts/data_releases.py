"""Release bookkeeping for the GHGRP raw snapshots in data/raw/.

Rules: raw files are immutable. A new release is a new file plus a new
manifest entry. Nothing here ever edits or deletes a raw file.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MANIFEST = RAW_DIR / "manifest.json"

# Locked decision: quarantine above this rate stops the refresh.
QUARANTINE_STOP_RATE = 0.02

# Locked decision: each annual refresh re-pulls the full history,
# because ECCC revises prior reporting years.
FULL_REPULL = True


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_manifest() -> dict:
    return json.loads(MANIFEST.read_text())


def save_manifest(manifest: dict) -> None:
    MANIFEST.write_text(json.dumps(manifest, indent=2) + "\n")


def latest_release() -> dict:
    """The newest release entry in the manifest."""
    manifest = load_manifest()
    releases = manifest["releases"]
    return max(releases, key=lambda r: r["years_covered"][1])


def release_path(release: dict) -> Path:
    return RAW_DIR / release["file"]


def add_release(*, release_id: str, filename: str, source: str,
                retrieval_date: str, years_covered: list,
                rows: int, notes: str = "") -> dict:
    """Register a new raw snapshot. The file must already exist."""
    path = RAW_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"raw snapshot missing: {path}")
    manifest = load_manifest()
    if any(r["release_id"] == release_id for r in manifest["releases"]):
        raise ValueError(f"release_id already registered: {release_id}")
    entry = {
        "release_id": release_id,
        "file": filename,
        "source": source,
        "retrieval_date": retrieval_date,
        "years_covered": years_covered,
        "rows": rows,
        "checksum_sha256": _sha256(path),
        "notes": notes,
    }
    manifest["releases"].append(entry)
    save_manifest(manifest)
    return entry


def verify_release(release: dict) -> bool:
    """True when the file on disk still matches its recorded checksum."""
    return _sha256(release_path(release)) == release["checksum_sha256"]
