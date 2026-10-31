"""``manifest.jsonl`` helpers (one JSON object per downloaded patch, ``"file"`` relative to the manifest's folder)."""

from __future__ import annotations

import json
import os
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_rows(path: Path, rows: list[dict]) -> None:
    """Atomic rewrite, so an interrupted run never leaves a half-written manifest."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    os.replace(tmp, path)


def append_row(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def find_manifests(root: Path) -> list[Path]:
    return sorted(root.rglob("manifest.jsonl")) if root.is_dir() else [root]


def song_key(row: dict) -> str | None:
    artist, song = (row.get("artist") or "").strip().lower(), (row.get("song") or "").strip().lower()
    return f"{artist}|{song}" if artist and song else None
