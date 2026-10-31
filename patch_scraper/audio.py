"""Song URLs for each scraped patch: Spotify (metadata + track URL) and YouTube (via yt-dlp search).

    $env:SPOTIFY_CLIENT_ID = "..."; $env:SPOTIFY_CLIENT_SECRET = "..."
    python -m patch_scraper.audio match [presets_dir ...] [--out data/data_scraping/urls] [--no-spotify] [--threshold 0.8] [--limit N]
    python -m patch_scraper.audio match data/data_scraping/presets/katana data/data_scraping/presets/neuraldsp --no-spotify --retry-unmatched

Spotify gives the canonical title / artist / ISRC / duration / track URL; it does not serve full tracks. The YouTube candidate must
match the title and duration and must not look like a live version, cover, lesson or backing track. Matches below ``--threshold``
get no ``youtube_url``; they are listed in ``<out>/unmatched.jsonl`` with the best candidate as ``youtube_candidate`` for review.
The URLs are written to ``<out>/song_urls.jsonl``; audio is downloaded only with ``--download-audio``.

Without Spotify (``--no-spotify``) the reference duration comes from the public iTunes Search API, then the Deezer API (no key; ``duration_source``
in the row says which one answered); only when both fail is the score renormalised over title / artist / channel, and those matches deserve more
suspicion. If Spotify rate-limits a run (HTTP 429 with a long Retry-After), the run does not wait: it switches to that fallback for the remaining
songs (``spotify_fallback`` in the stats). ``--retry-unmatched`` re-scores songs whose last attempt was unmatched, ``--reverify`` also re-scores
matches that never had a reference duration; a search that failed on a network error is stored as ``search_failed`` and is always retried.

Both fallback APIs are documented public APIs (iTunes ~20 calls / minute, Deezer 50 / 5 s), but their robots.txt files disallow ``/search*`` and
``/*`` for crawlers, so this lookup is the one place in the package that skips the robots check (like the Spotify client): one call per song at a
conservative rate, responses cached under ``<out>/_cache``, and a 403 / 429 switches that provider off for the rest of the run. Downloading audio from YouTube is against its terms unless you hold the rights: keep it local.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import re
import urllib.error
import urllib.parse
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable

from patch_scraper.http import PoliteClient
from patch_scraper.manifest import find_manifests, read_rows, song_key, write_rows
from patch_scraper.paths import PRESETS, URLS

log = logging.getLogger(__name__)

# a candidate is rejected if its title has one of these and the song's own title does not
BANNED = ("live", "cover", "lesson", "tutorial", "tabs", "backing track", "karaoke", "guitar only", "isolated", "reaction",
          "slowed", "sped up", "8d", "remix", "playthrough", "how to play", "instrumental version")
W_TITLE, W_ARTIST, W_DURATION, W_CHANNEL = 0.4, 0.2, 0.3, 0.1
URL_FIELDS = ("spotify_id", "spotify_url", "isrc", "duration_s", "duration_source", "youtube_url", "youtube_title", "youtube_candidate", "match_score",
              "song_audio")
ITUNES_SEARCH = "https://itunes.apple.com/search?media=music&entity=song&limit=5&term="
DEEZER_SEARCH = "https://api.deezer.com/search?limit=5&q="


def _norm(s: str) -> str:
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s.lower())
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", s).split())


@dataclass
class Track:
    spotify_id: str
    title: str
    artist: str
    isrc: str | None
    duration_s: float
    source: str = "spotify"                                  # which service gave the duration: spotify / itunes / deezer


_SUFFIX = re.compile(r"\s*[-–(\[]\s*(?:\d{4}\s+)?(?:digital\s+)?(?:remaster(?:ed)?|mono|stereo|single version|album version|radio edit|"
                     r"deluxe|bonus track|\d{4}\s+mix)\b.*$", re.I)


def clean_spotify_title(title: str) -> str:
    """'Whole Lotta Love - Remaster' / 'Enter Sandman - Remastered 2021' -> the plain song title."""
    return _SUFFIX.sub("", title).strip() or title


def _best_by_similarity(items: list[tuple[str, list[str]]], song: str, artist: str) -> int | None:
    """Index of the (title, artist names) pair closest to the wanted song (0.6 * title + 0.4 * artist similarity >= 0.75, or its title contained in the
    wanted song with a matching artist), or None."""
    best, best_score = None, 0.0
    for i, (title, names) in enumerate(items):
        t = SequenceMatcher(None, _norm(title), _norm(song)).ratio()
        a = max((SequenceMatcher(None, _norm(n), _norm(artist)).ratio() for n in names), default=0.0)
        if (s := 0.6 * t + 0.4 * a) > best_score:
            best, best_score = i, s
    if best is not None and best_score >= 0.75:
        return best
    # preset titles are noisy ("Nothing Else Matters Cln & Dst"): also accept a result whose whole title sits inside the wanted song, when the
    # artist matches well; the longest such title wins
    wanted, hits = f" {_norm(song)} ", []
    for i, (title, names) in enumerate(items):
        n = _norm(title)
        if len(n) >= 4 and f" {n} " in wanted and max((SequenceMatcher(None, _norm(x), _norm(artist)).ratio() for x in names), default=0.0) >= 0.8:
            hits.append((len(n), i))
    return max(hits)[1] if hits else None


def pick_spotify_track(payload: dict, song: str, artist: str) -> Track | None:
    """Best of the search results by title+artist similarity (>= 0.75), or None."""
    items = payload.get("tracks", {}).get("items", [])
    i = _best_by_similarity([(it["name"], [a["name"] for a in it.get("artists", [])]) for it in items], song, artist)
    if i is None:
        return None
    best = items[i]
    return Track(best["id"], clean_spotify_title(best["name"]), best["artists"][0]["name"] if best.get("artists") else artist,
                 (best.get("external_ids") or {}).get("isrc"), best["duration_ms"] / 1000.0)


def pick_itunes_track(payload: dict, song: str, artist: str) -> Track | None:
    """iTunes Search API result -> Track (only the duration is used downstream)."""
    items = [it for it in payload.get("results", []) if it.get("trackTimeMillis")]
    i = _best_by_similarity([(it.get("trackName", ""), [it.get("artistName", "")]) for it in items], song, artist)
    if i is None:
        return None
    it = items[i]
    return Track(str(it.get("trackId", "")), clean_spotify_title(it["trackName"]), it.get("artistName", artist), None,
                 it["trackTimeMillis"] / 1000.0, "itunes")


def pick_deezer_track(payload: dict, song: str, artist: str) -> Track | None:
    """Deezer search result -> Track."""
    items = [it for it in payload.get("data", []) if it.get("duration")]
    i = _best_by_similarity([(it.get("title", ""), [(it.get("artist") or {}).get("name", "")]) for it in items], song, artist)
    if i is None:
        return None
    it = items[i]
    return Track(str(it.get("id", "")), clean_spotify_title(it["title"]), (it.get("artist") or {}).get("name", artist), it.get("isrc"),
                 float(it["duration"]), "deezer")


class DurationLookup:
    """Reference duration without Spotify: iTunes Search API first, Deezer second. A provider that answers 403 / 429 (or fails to connect twice
    in a row) is switched off for the rest of the run; calling the object returns None when nobody knows the song."""

    def __init__(self, cache_dir: Path | None = None, itunes_delay: float = 3.2, deezer_delay: float = 0.3):
        def mk(base: str, delay: float) -> PoliteClient:
            return PoliteClient(base, delay_sec=delay, cache_dir=cache_dir, check_robots=False, max_retry_after=30, retries=2)

        self.providers = [("itunes", mk("https://itunes.apple.com", itunes_delay), ITUNES_SEARCH, pick_itunes_track),
                          ("deezer", mk("https://api.deezer.com", deezer_delay), DEEZER_SEARCH, pick_deezer_track)]
        self.dead: set[str] = set()
        self.errors: dict[str, int] = {}

    def __call__(self, song: str, artist: str) -> Track | None:
        for name, client, base, pick in self.providers:
            if name in self.dead:
                continue
            try:
                payload = json.loads(client.get(base + urllib.parse.quote(f"{artist} {song}")))
                self.errors[name] = 0
            except urllib.error.HTTPError as e:
                if e.code in (403, 429):
                    log.warning("%s refused the lookup (HTTP %s): not using it for the rest of the run", name, e.code)
                    self.dead.add(name)
                continue
            except Exception as e:
                self.errors[name] = self.errors.get(name, 0) + 1
                log.warning("%s lookup failed for %s - %s: %s", name, artist, song, e)
                if self.errors[name] >= 2:
                    self.dead.add(name)
                continue
            if tr := pick(payload, song, artist):
                return tr
        return None


def score_candidate(cand: dict, song: str, artist: str, duration_s: float | None) -> float:
    """0-1 match score for a yt-dlp flat-search entry ({title, duration, channel/uploader}). 0 = rejected."""
    title = cand.get("title") or ""
    low, song_low = title.lower(), song.lower()
    if any(b in low and b not in song_low for b in BANNED):
        return 0.0
    chan = (cand.get("channel") or cand.get("uploader") or "")
    song_tokens = set(_norm(song).split())
    title_tokens = set(_norm(title).split())
    title_part = len(song_tokens & title_tokens) / len(song_tokens) if song_tokens else 0.0
    artist_part = 1.0 if _norm(artist) and (_norm(artist) in _norm(title) or _norm(artist) in _norm(chan)) else 0.0
    chan_part = 1.0 if chan.endswith("- Topic") or "vevo" in chan.lower() or _norm(artist) == _norm(chan.replace("- Topic", "")) else 0.0
    got = W_TITLE * title_part + W_ARTIST * artist_part + W_CHANNEL * chan_part
    total = W_TITLE + W_ARTIST + W_CHANNEL
    if duration_s is not None and cand.get("duration"):
        diff = abs(float(cand["duration"]) - duration_s)
        got += W_DURATION * max(0.0, 1.0 - max(0.0, diff - 5.0) / 10.0)      # full credit within 5 s, none beyond 15 s
        total += W_DURATION
    return round(got / total, 3)


# ---------------------------------------------------------------------------------------------------------------------
def spotify_token(client: PoliteClient, cid: str, secret: str) -> str:
    auth = base64.b64encode(f"{cid}:{secret}".encode()).decode()
    body = client.get("https://accounts.spotify.com/api/token", headers={"Authorization": f"Basic {auth}"},
                      data=b"grant_type=client_credentials", cache=False)
    return json.loads(body)["access_token"]


def spotify_search(client: PoliteClient, token: str, song: str, artist: str) -> Track | None:
    q = urllib.parse.urlencode({"q": f"track:{song} artist:{artist}", "type": "track", "limit": 5})
    body = client.get("https://api.spotify.com/v1/search?" + q, headers={"Authorization": f"Bearer {token}"})
    return pick_spotify_track(json.loads(body), song, artist)


def youtube_search(query: str, n: int = 5) -> list[dict]:
    from guitar_stem_isolator.downloader import use_system_trust_store

    use_system_trust_store()
    import yt_dlp

    opts = {"quiet": True, "no_warnings": True, "extract_flat": True, "skip_download": True}
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(f"ytsearch{n}:{query}", download=False)
    out = []
    for e in info.get("entries", []):
        url = e.get("url") or (f"https://www.youtube.com/watch?v={e['id']}" if e.get("id") else None)
        if url:
            out.append({**e, "url": url})
    return out


def best_youtube(song: str, artist: str, duration_s: float | None,
                 search: Callable[[str], list[dict]] = youtube_search) -> tuple[dict | None, float]:
    best, best_score = None, 0.0
    for q in (f"{artist} - {song} official audio", f"{artist} {song}"):
        for cand in search(q):
            if (s := score_candidate(cand, song, artist, duration_s)) > best_score:
                best, best_score = cand, s
        if best_score >= 0.9:
            break
    return best, best_score


def match(root: Path | list[Path], out: Path, *, threshold: float = 0.8, use_spotify: bool = True, limit: int | None = None,
          download_audio: bool = False, writeback: bool = True, search: Callable[[str], list[dict]] = youtube_search,
          download: Callable[..., Path] | None = None, spotify: Callable[[str, str], Track | None] | None = None,
          retry_unmatched: bool = False, duration_lookup: bool | Callable[[str, str], Track | None] = False,
          reverify: bool = False) -> dict[str, int]:
    """Find a Spotify and a YouTube URL for every unique artist|song under ``root``.

    Writes ``<out>/song_urls.jsonl`` (one row per song), ``<out>/unmatched.jsonl`` (below ``threshold``), and the same fields into
    every manifest row of that song. Audio is only downloaded with ``download_audio=True`` (into ``<out>/audio``).

    ``duration_lookup`` supplies the reference duration for songs Spotify does not answer for (``--no-spotify``, a 429 fallback, no hit):
    True = iTunes then Deezer, or pass a ``(song, artist) -> Track | None``. ``reverify`` also re-scores matches that never had a duration."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if download_audio and download is None:
        from guitar_stem_isolator.downloader import download_audio as _dl

        def download(url, duration_s, name):
            return _dl(url, start_sec=0.0, end_sec=duration_s, out_dir=out / "audio", name=name)
    if use_spotify and spotify is None:
        cid, secret = os.environ.get("SPOTIFY_CLIENT_ID"), os.environ.get("SPOTIFY_CLIENT_SECRET")
        if not (cid and secret):
            raise RuntimeError("set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET, or pass --no-spotify")
        sp = PoliteClient("https://api.spotify.com", delay_sec=0.2, cache_dir=out / "_cache", check_robots=False, max_retry_after=60)
        token = spotify_token(sp, cid, secret)
        spotify = lambda song, artist: spotify_search(sp, token, song, artist)       # noqa: E731
    reference = DurationLookup(out / "_cache") if duration_lookup is True else (duration_lookup or None)
    roots = [Path(r) for r in (root if isinstance(root, (list, tuple)) else [root])]
    manifests = {mp: read_rows(mp) for r in roots for mp in find_manifests(r)}
    results: dict[str, dict] = {}
    songs_file, unmatched_file = out / "song_urls.jsonl", out / "unmatched.jsonl"
    done = {song_key(r): r for r in read_rows(songs_file) if song_key(r)}
    stats = {"songs": 0, "matched": 0, "unmatched": 0, "search_failed": 0, "no_spotify": 0, "already": 0, "spotify_fallback": 0,
             "duration_itunes": 0, "duration_deezer": 0, "duration_none": 0}
    for rows in manifests.values():
        for r in rows:
            k = song_key(r)
            if not k or k in results:
                continue
            old = done.get(k, {})
            unverified = old.get("status") == "matched" and not (old.get("spotify_id") or old.get("duration_source"))
            if k in done and old.get("status") != "search_failed" and not (retry_unmatched and old.get("status") == "unmatched") \
                    and not (reverify and unverified):
                results[k] = {f: done[k].get(f) for f in URL_FIELDS}
                stats["already"] += 1
                continue
            if limit is not None and stats["songs"] >= limit:
                continue
            stats["songs"] += 1
            song, artist, dur = r["song"], r["artist"], None
            res: dict = {}
            tr = None
            if use_spotify:
                try:
                    tr = spotify(song, artist)
                except urllib.error.HTTPError as e:
                    if e.code != 429:
                        raise
                    log.warning("Spotify rate limit (HTTP 429): continuing with YouTube-only scoring for the remaining songs")
                    use_spotify, stats["spotify_fallback"] = False, 1
                if use_spotify and tr is None:
                    stats["no_spotify"] += 1
                elif tr is not None:
                    song, artist, dur = tr.title, tr.artist, tr.duration_s
                    res.update(spotify_id=tr.spotify_id, spotify_url=f"https://open.spotify.com/track/{tr.spotify_id}", isrc=tr.isrc,
                               duration_s=tr.duration_s, duration_source="spotify")
            if tr is None and reference is not None:
                try:
                    ref = reference(song, artist)
                except Exception as e:
                    log.warning("duration lookup failed for %s - %s: %s", artist, song, e)
                    ref = None
                if ref is not None:                                  # keep our own title / artist for the search: only the duration is borrowed
                    dur = ref.duration_s
                    res.update(isrc=ref.isrc, duration_s=ref.duration_s, duration_source=ref.source)
                    stats[f"duration_{ref.source}"] += 1
                else:
                    stats["duration_none"] += 1
            failed = False
            try:
                cand, score = best_youtube(song, artist, dur, search)
            except Exception as e:
                log.warning("YouTube search failed for %s - %s: %s", artist, song, e)
                cand, score, failed = None, 0.0, True
            ok = bool(cand) and score >= threshold
            res.update(match_score=score, youtube_url=cand["url"] if ok else None, youtube_title=cand.get("title") if cand else None,
                       youtube_candidate=None if ok or not cand else cand["url"], song_audio=None)
            if ok and download is not None:
                try:
                    name = re.sub(r"[^a-z0-9]+", "-", f"{artist}-{song}".lower()).strip("-")[:80]
                    res["song_audio"] = str(download(cand["url"], dur or cand.get("duration"), name))
                except Exception as e:
                    log.warning("download failed for %s - %s: %s", artist, song, e)
            full = {"artist": r["artist"], "song": r["song"], "genre": r.get("genre"), **{f: res.get(f) for f in URL_FIELDS}}
            full["status"] = "matched" if ok else "search_failed" if failed else "unmatched"
            full["spotify_checked"] = tr is not None
            with songs_file.open("a", encoding="utf-8") as f:
                f.write(json.dumps(full, ensure_ascii=False) + "\n")
            stats[full["status"]] += 1
            done[k] = full
            results[k] = {f: res.get(f) for f in URL_FIELDS}
    # unmatched.jsonl = the latest attempt of every song that is still unmatched (a re-run rewrites it, no stale rows)
    write_rows(unmatched_file, [r for r in done.values() if r.get("status") == "unmatched"])
    if writeback:          # song_urls.jsonl is complete either way
        for mp in manifests:
            rows = read_rows(mp)       # re-read: a crawler may have appended rows since this run started (the copy loaded at the start is stale)
            for r in rows:
                if (k := song_key(r)) in results:
                    r.update(results[k])
            write_rows(mp, rows)
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("match")
    m.add_argument("root", nargs="*", default=[str(PRESETS)], help="one or more preset folders (default: all of presets/)")
    m.add_argument("--out", default=str(URLS))
    m.add_argument("--threshold", type=float, default=0.8)
    m.add_argument("--no-spotify", action="store_true")
    m.add_argument("--download-audio", action="store_true", help="also download the matched YouTube audio into <out>/audio")
    m.add_argument("--limit", type=int, help="process at most this many new songs")
    m.add_argument("--retry-unmatched", action="store_true", help="re-score songs whose last attempt was unmatched")
    m.add_argument("--no-duration-lookup", action="store_true", help="do not ask iTunes / Deezer for a reference duration when Spotify is not used")
    m.add_argument("--reverify", action="store_true", help="also re-score matches that never had a reference duration")
    m.add_argument("--no-writeback", action="store_true", help="do not copy the URLs into the manifests (use while a crawl is running)")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    print(match([Path(r) for r in a.root], Path(a.out), retry_unmatched=a.retry_unmatched, threshold=a.threshold, use_spotify=not a.no_spotify, limit=a.limit,
                download_audio=a.download_audio, writeback=not a.no_writeback, duration_lookup=not a.no_duration_lookup, reverify=a.reverify))


if __name__ == "__main__":
    main()
