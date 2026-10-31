"""Song / artist / rating metadata for a patch file.

Neither .hlx nor .tsl carries the song, the artist or the community rating: those live on the website page. The scraper
therefore has to save them next to each download. Three layouts are understood (first hit wins):

  1. sidecar        ``<file>.meta.json`` or ``<file stem>.json`` next to the patch
  2. manifest       ``manifest.jsonl`` / ``manifest.json`` in the patch's folder (or a parent), one entry per file with a
                    ``"file"`` key (file name or relative path)
  3. text           "Artist - Song" / "Song (Artist)" / "Song by Artist" in the patch name or file name (low confidence)

Accepted keys (all optional): title, artist, song, author, rating, rating_count, downloads, source_url, tags.
``rating`` may be on any scale; pass ``rating_max`` to normalise to 0-5 (default: values > 5 are read as percentages /
out of 10 / out of 100).
"""

from __future__ import annotations

import json
import re
from functools import lru_cache
from pathlib import Path
from typing import Any

from patch_corpus.schema import PatchMeta

_ALIASES = {
    "title": ("title", "name", "patch_name", "tone_name"),
    "artist": ("artist", "band", "performer"),
    "song": ("song", "song_title", "track", "track_title"),
    "author": ("author", "creator", "uploader", "user", "username"),
    "rating": ("rating", "stars", "score", "avg_rating", "average_rating"),
    "rating_count": ("rating_count", "votes", "num_ratings", "ratings"),
    "downloads": ("downloads", "download_count"),
    "source_url": ("source_url", "url", "link", "page"),
    "tags": ("tags", "keywords", "genres"),
}
_SEP = re.compile(r"\s+[-–—:|]\s+")


def normalise_rating(value: Any, rating_max: float | None = None) -> float | None:
    """Any common scale -> 0-5 stars."""
    try:
        r = float(value)
    except (TypeError, ValueError):
        return None
    if rating_max:
        return max(0.0, min(5.0, r / rating_max * 5.0))
    if r <= 5:
        return max(0.0, r)
    if r <= 10:
        return r / 2.0
    return min(5.0, r / 20.0)


def _pick(d: dict, key: str) -> Any:
    low = {k.lower(): v for k, v in d.items()}
    for alias in _ALIASES[key]:
        if alias in low and low[alias] not in (None, ""):
            return low[alias]
    return None


def from_dict(d: dict, origin: str, rating_max: float | None = None) -> PatchMeta:
    tags = _pick(d, "tags") or []
    if isinstance(tags, str):
        tags = [t.strip() for t in re.split(r"[,;|]", tags) if t.strip()]
    count = _pick(d, "rating_count")
    dl = _pick(d, "downloads")
    return PatchMeta(
        title=_str(_pick(d, "title")), artist=_str(_pick(d, "artist")), song=_str(_pick(d, "song")),
        author=_str(_pick(d, "author")), rating=normalise_rating(_pick(d, "rating"), rating_max),
        rating_count=_int(count), downloads=_int(dl), source_url=_str(_pick(d, "source_url")),
        tags=[str(t) for t in tags], meta_origin=origin)


def _str(v) -> str | None:
    return None if v is None else str(v).strip() or None


def _int(v) -> int | None:
    try:
        return int(str(v).replace(",", ""))
    except (TypeError, ValueError):
        return None


def artist_song_from_text(text: str) -> tuple[str | None, str | None]:
    """'Artist - Song', 'Song (Artist)', 'Song by Artist' -> (artist, song). Returns (None, None) when unsure."""
    text = re.sub(r"[_]+", " ", text).strip()
    m = re.match(r"^(.+?)\s+by\s+(.+)$", text, re.I)
    if m:
        return m.group(2).strip(), m.group(1).strip()
    m = re.match(r"^(.+?)\s*\(([^)]+)\)\s*$", text)
    if m and len(m.group(2).split()) <= 5:
        return m.group(2).strip(), m.group(1).strip()
    parts = _SEP.split(text, maxsplit=1)
    if len(parts) == 2 and all(p.strip() for p in parts):
        return parts[0].strip(), parts[1].strip()
    return None, None


@lru_cache(maxsize=256)
def _load_manifest(path: str) -> dict[str, dict]:
    p = Path(path)
    out: dict[str, dict] = {}
    try:
        if p.suffix == ".jsonl":
            rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
        else:
            doc = json.loads(p.read_text(encoding="utf-8"))
            rows = doc if isinstance(doc, list) else doc.get("items") or doc.get("files") or [
                {"file": k, **v} for k, v in doc.items() if isinstance(v, dict)]
    except Exception:
        return out
    for r in rows:
        if isinstance(r, dict) and r.get("file"):
            out[str(r["file"]).replace("\\", "/")] = r
    return out


def find_meta(patch_path: Path, patch_name: str | None = None, root: Path | None = None, rating_max: float | None = None) -> PatchMeta:
    for cand in (patch_path.with_name(patch_path.name + ".meta.json"), patch_path.with_suffix(".json")):
        if cand.exists() and cand != patch_path:
            try:
                return from_dict(json.loads(cand.read_text(encoding="utf-8")), "sidecar", rating_max)
            except Exception:
                pass
    folder = patch_path.parent
    for d in [folder, *folder.parents]:
        for name in ("manifest.jsonl", "manifest.json"):
            mp = d / name
            if mp.exists():
                m = _load_manifest(str(mp))
                rel = patch_path.relative_to(d).as_posix() if patch_path.is_relative_to(d) else patch_path.name
                hit = m.get(rel) or m.get(patch_path.name)
                if hit:
                    return from_dict(hit, "manifest", rating_max)
        if root is not None and d == root:
            break
    for text, origin in ((patch_name, "patch_name"), (patch_path.stem, "filename")):
        if text:
            artist, song = artist_song_from_text(text)
            if song:
                return PatchMeta(title=patch_name or patch_path.stem, artist=artist, song=song, meta_origin=origin)
    return PatchMeta(title=patch_name or patch_path.stem)
