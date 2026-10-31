"""Shared crawl engine: adapter pages -> filter (famous artists / rating / song) -> detail -> download -> manifest.jsonl.

A site adapter (``guitarpatches.py``) knows one site's HTML and provides:

    pages(client, query) -> iterator of lists of Item     one list per listing page, until the site runs out
    enrich(client, item)                                  optional detail-page fetch (fills rating / download_url)

Rows are written in the layout ``patch_corpus/metadata.py`` reads. A row whose file could not be downloaded (Line 6 needs a login)
is still stored, with ``"file": null`` and ``"status": "login_required"``; running again with the cookie set fills the file in.
Both formats are JSON: a download that does not parse as JSON (a login page, an error page) is rejected instead of saved.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import logging
import re
import urllib.parse
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterator, Protocol

from common.paths import ensure
from patch_corpus.metadata import artist_song_from_text, normalise_rating
from patch_scraper.artists import FamousArtists, norm
from patch_scraper.http import PoliteClient
from patch_scraper.manifest import append_row, read_rows, write_rows

log = logging.getLogger(__name__)


@dataclass
class Item:
    url: str                                   # detail-page URL, also the manifest key
    row: dict                                  # manifest fields found so far
    download_url: str | None = None
    needs_detail: bool = False                 # True when rating / download link are only on the detail page


class Adapter(Protocol):
    name: str
    file_ext: str
    base_url: str
    cookie_env: str | None

    def pages(self, client: PoliteClient, query: str | None) -> Iterator[list[Item]]: ...
    def enrich(self, client: PoliteClient, item: Item) -> None: ...
    # optional: validate(data: bytes) -> bool, for adapters whose files are not JSON (the default check)


def text(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", " ", s or "")).replace("\xa0", " ").strip().replace("  ", " ")


GENERIC = r"(?i)(part|patch|preset|tone|clean|lead|rhythm|crunch|test|version|take|v)\s*\d*"


def clean_song(raw: str | None, artist: str | None) -> str | None:
    """Patch / tone name -> song title: drop a leading 'Artist -' and reject names that are only the artist and/or a generic word."""
    s = (raw or "").strip()
    if artist:
        s = re.sub(rf"^\s*{re.escape(artist)}\s*[-–:]\s*", "", s, flags=re.I)
        rest = re.sub(rf"^\s*{re.escape(artist)}(?!\w)", "", s, flags=re.I).strip(" -–:")
        if rest != s and (not rest or re.fullmatch(GENERIC, rest)):          # "U2 part 2" for artist "U2"
            return None
    s = s.strip(" -–:")
    if not s or norm(s) == norm(artist or "") or re.fullmatch(GENERIC, s):
        return None
    return s


@dataclass
class CrawlStats:
    new: int = 0
    filled: int = 0
    already: int = 0
    not_wanted: int = 0
    no_song: int = 0
    low_rating: int = 0
    login_required: int = 0
    bad_file: int = 0
    duplicate: int = 0

    def __str__(self) -> str:
        return ", ".join(f"{k}={v}" for k, v in asdict(self).items())


def crawl(adapter: Adapter, out_dir: Path, client: PoliteClient | None = None, *, queries: list[str | None] | None = None,
          famous: FamousArtists | None = None, min_rating: float = 0.0, limit: int | None = None, allow_artist_only: bool = False,
          progress: Callable[[str], None] = print) -> CrawlStats:
    """Resumable crawl. ``famous=None`` keeps every artist; otherwise only artists in the list (their genre is recorded).
    Unrated patches are kept (``min_rating`` only drops patches that carry a lower rating)."""
    out_dir = ensure(Path(out_dir))
    client = client or PoliteClient(adapter.base_url, adapter.cookie_env, 2.0, cache_dir=out_dir / "_cache")
    manifest = out_dir / "manifest.jsonl"
    rows = read_rows(manifest)
    by_url = {r["source_url"]: r for r in rows if r.get("source_url")}
    hashes = {r.get("sha1") for r in rows if r.get("sha1")}
    stats, rewrite = CrawlStats(), False

    def store(item: Item, row: dict) -> str:
        """Download, validate and write the file. Returns 'saved', 'none' (no download link), 'bad' or 'duplicate'."""
        if not item.download_url:
            return "none"
        data = client.get(item.download_url, cache=False)
        check = getattr(adapter, "validate", None)
        try:
            if check is not None:
                if not check(data):
                    raise ValueError("rejected by the adapter")
            else:
                json.loads(data.decode("utf-8-sig"))
        except Exception:
            log.warning("%s did not return a valid %s patch (login page or error?)", item.url, adapter.file_ext)
            stats.bad_file += 1
            return "bad"
        digest = hashlib.sha1(data).hexdigest()
        if digest in hashes:
            stats.duplicate += 1
            return "duplicate"
        hashes.add(digest)
        name = f"{row['source_id']}{adapter.file_ext}"
        (out_dir / name).write_bytes(data)
        row.update(file=name, sha1=digest, status="ok")
        return "saved"

    for query in (queries or [None]):
        seen_ids: set[str] = set()
        for items in adapter.pages(client, query):
            fresh = [i for i in items if i.url not in seen_ids]
            seen_ids.update(i.url for i in items)
            if not fresh:
                break
            for item in fresh:
                known = by_url.get(item.url)
                if known is not None and (known.get("file") or not item.download_url):
                    stats.already += 1
                    continue
                row = known if known is not None else item.row
                hit = famous.match(row.get("artist")) if famous else None
                if famous and not hit:
                    stats.not_wanted += 1
                    continue
                if item.needs_detail and known is None:
                    adapter.enrich(client, item)
                    row = item.row
                if hit:
                    row["famous_artist"] = hit[1]
                    if hit[0] != "other":
                        row.setdefault("genre", hit[0])
                if not (row.get("artist") and (row.get("song") or allow_artist_only)):
                    stats.no_song += 1
                    continue
                if row.get("rating") is not None and row["rating"] < min_rating:
                    stats.low_rating += 1
                    continue
                result = store(item, row)
                if result in ("bad", "duplicate"):         # not stored, so the next run tries again
                    continue
                saved = result == "saved"
                if not saved:
                    row.update(file=None, status="login_required")
                    if known is None:
                        stats.login_required += 1
                if known is None:
                    rows.append(row)
                    by_url[item.url] = row
                    append_row(manifest, row)
                    stats.new += 1
                elif saved:
                    rewrite = True
                    stats.filled += 1
                if stats.new and stats.new % 25 == 0 and known is None:
                    progress(f"[{adapter.name}] {stats}")
                if limit and stats.new + stats.filled >= limit:
                    if rewrite:
                        write_rows(manifest, rows)
                    return stats
    if rewrite:
        write_rows(manifest, rows)
    return stats


def build_parser(prog: str, default_out: Path | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog=prog, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(default_out) if default_out else None)
    ap.add_argument("--all", action="store_true", help="keep every artist, not only the famous-artist list")
    ap.add_argument("--genre", action="append", default=[], help="restrict the famous list to these genres (repeatable)")
    ap.add_argument("--artist", action="append", default=[], help="crawl these artists only (repeatable; implies they are 'famous')")
    ap.add_argument("--min-rating", type=float, default=0.0, help="stars (0-5); unrated patches are kept")
    ap.add_argument("--limit", type=int, help="stop after this many new patches")
    ap.add_argument("--max-pages", type=int, help="listing pages per artist query (Line 6) or per unit (GuitarPatches)")
    ap.add_argument("--allow-artist-only", action="store_true", help="keep patches that name an artist but no song")
    return ap


def famous_filter(args) -> FamousArtists | None:
    if args.all and not args.artist:
        return None
    table = FamousArtists().table
    if args.artist:
        table = {"other": args.artist}
    elif args.genre:
        table = {g: table[g] for g in args.genre}
    return FamousArtists(table)
