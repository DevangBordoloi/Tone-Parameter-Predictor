"""Genre tagging from the artist's MusicBrainz tags (Last.fm top tags as the fallback), mapped onto a fixed genre list.

    python -m patch_scraper.genre tag [presets_dir]      # LASTFM_API_KEY is optional

Tags describe the artist, not the song, so a genre is an approximation. Lookups are cached in ``<root>/genre_cache.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.parse
from collections import defaultdict
from pathlib import Path
from typing import Callable

from patch_scraper.http import PoliteClient
from patch_scraper.manifest import find_manifests, read_rows, write_rows
from patch_scraper.paths import PRESETS

log = logging.getLogger(__name__)

# first-listed order only breaks ties; a tag counts for every genre whose keyword it contains
GENRE_KEYWORDS: dict[str, tuple[str, ...]] = {
    "metal": ("metal", "djent", "thrash", "deathcore", "grindcore"),
    "punk": ("punk", "hardcore", "ska"),
    "progressive": ("progressive", "prog ", "prog-", "math rock", "post-rock"),
    "fusion": ("fusion",),
    "blues": ("blues",),
    "jazz": ("jazz", "bebop", "swing"),
    "funk": ("funk", "disco"),
    "r&b/soul": ("soul", "r&b", "rnb", "motown", "gospel"),
    "country": ("country", "bluegrass", "americana", "folk"),
    "pop": ("pop", "synth", "dance"),
    "indie/alternative": ("indie", "alternative", "shoegaze", "grunge", "britpop"),
    "rock": ("rock",),
}
GENRES = [*GENRE_KEYWORDS, "other"]


def map_tags(tags: list[tuple[str, int]]) -> str:
    """Weighted vote of (tag, count) pairs onto GENRES. No usable tag -> 'other'."""
    score: dict[str, float] = defaultdict(float)
    for name, count in tags:
        low = f" {name.lower()} "
        for genre, words in GENRE_KEYWORDS.items():
            if any(w in low for w in words):
                score[genre] += max(1, count)
    return max(score, key=score.get) if score else "other"


def musicbrainz_tags(client: PoliteClient, artist: str) -> list[tuple[str, int]]:
    url = "https://musicbrainz.org/ws/2/artist/?" + urllib.parse.urlencode({"query": f'artist:"{artist}"', "fmt": "json", "limit": 3})
    hits = json.loads(client.get(url)).get("artists", [])
    exact = [h for h in hits if h.get("name", "").lower() == artist.lower()]
    best = (exact or hits or [None])[0]
    return [(t["name"], int(t.get("count", 1))) for t in (best or {}).get("tags", [])]


def lastfm_tags(client: PoliteClient, artist: str, api_key: str) -> list[tuple[str, int]]:
    url = "https://ws.audioscrobbler.com/2.0/?" + urllib.parse.urlencode(
        {"method": "artist.gettoptags", "artist": artist, "api_key": api_key, "format": "json"})
    tags = json.loads(client.get(url)).get("toptags", {}).get("tag", [])
    return [(t["name"], int(t.get("count", 1))) for t in tags]


def default_fetch(root: Path) -> Callable[[str], list[tuple[str, int]]]:
    cache = root / "_cache"
    mb = PoliteClient("https://musicbrainz.org", delay_sec=1.1, cache_dir=cache, check_robots=False)
    key = os.environ.get("LASTFM_API_KEY")
    lf = PoliteClient("https://ws.audioscrobbler.com", delay_sec=0.3, cache_dir=cache, check_robots=False)

    def fetch(artist: str) -> list[tuple[str, int]]:
        tags = []
        try:
            tags = musicbrainz_tags(mb, artist)
        except Exception as e:
            log.warning("MusicBrainz lookup failed for %s: %s", artist, e)
        if not tags and key:
            try:
                tags = lastfm_tags(lf, artist, key)
            except Exception as e:
                log.warning("Last.fm lookup failed for %s: %s", artist, e)
        return tags
    return fetch


def tag(root: Path, fetch: Callable[[str], list[tuple[str, int]]] | None = None, overwrite: bool = False) -> dict[str, int]:
    """Add ``genre`` (and the artist's top raw tags) to every manifest row under ``root``. Returns patches per genre."""
    root = Path(root)
    fetch = fetch or default_fetch(root if root.is_dir() else root.parent)
    cache_path = (root if root.is_dir() else root.parent) / "genre_cache.json"
    cache: dict[str, list] = json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
    totals: dict[str, int] = defaultdict(int)
    for mp in find_manifests(root):
        rows = read_rows(mp)
        for r in rows:
            artist = (r.get("artist") or "").strip()
            if not artist or (r.get("genre") and not overwrite):
                totals[r.get("genre") or "other"] += 1
                continue
            key = artist.lower()
            if key not in cache:
                cache[key] = [list(t) for t in fetch(artist)]
                cache_path.write_text(json.dumps(cache, indent=1), encoding="utf-8")
            tags = [(n, c) for n, c in cache[key]]
            r["genre"] = map_tags(tags)
            top = [n for n, _ in sorted(tags, key=lambda t: -t[1])[:5]]
            r["tags"] = list(dict.fromkeys([*(r.get("tags") or []), *top]))
            totals[r["genre"]] += 1
        write_rows(mp, rows)
    return dict(totals)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tag")
    t.add_argument("root", nargs="?", default=str(PRESETS))
    t.add_argument("--overwrite", action="store_true", help="re-tag rows that already have a genre")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    for g, n in sorted(tag(Path(a.root), overwrite=a.overwrite).items(), key=lambda kv: -kv[1]):
        print(f"{g:20s} {n}")


if __name__ == "__main__":
    main()
