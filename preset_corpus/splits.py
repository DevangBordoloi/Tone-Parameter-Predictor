"""Deterministic train / val / test split of community presets, by author (so one author's style never straddles the split)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

FRACTIONS = (("train", 0.7), ("val", 0.1), ("test", 0.2))


def split_of(meta: dict, preset_id: str) -> str:
    """Key = ``split_group`` (artist|song, set by patch_corpus so one song never straddles the split), else the author, else the id."""
    key = (meta.get("split_group") or meta.get("author") or "").strip().lower() or preset_id
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in FRACTIONS:
        acc += frac
        if h < acc:
            return name
    return "train"


def load_split(preset_dir: str | Path, plugin_name: str, split: str) -> list[dict]:
    """Decoded presets (as written by decode.export_corpus) that belong to `split`."""
    d = Path(preset_dir) / plugin_name
    out = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        doc = json.loads(f.read_text(encoding="utf-8"))
        if split_of(doc.get("meta", {}), f.stem) == split:
            doc["_id"] = f.stem
            out.append(doc)
    return out
