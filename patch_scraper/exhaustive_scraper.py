"""Genre-gap collection: the presets installed with your Neural DSP plugins, Fender Mustang patches, targeted forum threads, gap-genre Boss libraries.

    python -m patch_scraper.exhaustive_scraper local                     # factory + artist presets of the 4 installed plugins -> presets/neuraldsp
    python -m patch_scraper.exhaustive_scraper mustang                   # GuitarPatches mustang1 + mustang3 (.fuse) -> presets/fender_mustang
    python -m patch_scraper.exhaustive_scraper forum                     # Unity forum topics whose title / tags name a plugin or a gap genre
    python -m patch_scraper.exhaustive_scraper gaps                      # Katana + GT libraries, famous artists of the gap genres only
    python -m patch_scraper.exhaustive_scraper tag                       # (re)tag genre evidence on the neuraldsp / fender manifests
    python -m patch_scraper.exhaustive_scraper all

Everything goes through the package's own modules: ``http.PoliteClient`` (robots.txt, crawl delay, cache), ``archive.Sink`` (validate, hash
de-duplication across sources, ``manifest.jsonl``), ``sites.crawl``, ``artists.FamousArtists`` and ``genre``.

Genre is only ever set from evidence, and the evidence is recorded in ``genre_origin``:

  famous_artist   the preset's creator / artist is on ``config/famous_artists.json``
  creator_table   the creator is in ``CREATOR_GENRE`` below (well-known players the list does not cover)
  name_keyword    a whole-word keyword in the preset name or sub-folder (``Glassy SRV Tone`` -> blues)
  fuse_tags       a Fender FUSE patch's tags / title

A preset with no evidence gets no ``genre``; the style the plugin is known for goes in ``genre_prior`` and is NOT counted by
``balance report``. (John Mayer X's ``Artists/`` folders hold guitarists from many styles, and its factory folder holds bass and metal
presets next to blues ones; tagging all of it "blues" would only make the table look balanced.)

robots.txt is honoured everywhere. The forum is not searched (``/search`` is disallowed for crawlers): its category listings are filtered by
keyword instead. The Katana / GT libraries have no search either, so the "gap" crawl re-pages the (cached) listings and downloads only
patches whose artist is a famous artist of the gap genres; it cannot create patches the site does not have.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
from collections import Counter
from pathlib import Path

from patch_corpus.parsers.fender_fuse import is_fuse, parse_fuse
from patch_scraper.archive import Sink, valid
from patch_scraper.artists import FamousArtists
from patch_scraper.download import Fetcher, Ledger
from patch_scraper.guitarpatches import GT_UNITS, UNITS, GuitarPatches
from patch_scraper.http import PoliteClient
from patch_scraper.manifest import find_manifests, read_rows, write_rows
from patch_scraper.neural_forum import BASE as FORUM_BASE, crawl as forum_crawl, keyword_re
from patch_scraper.paths import BOSS_GT, FENDER, KATANA, NEURALDSP, URLS
from patch_scraper.sites import crawl

log = logging.getLogger("patch_scraper.exhaustive")

GAP_GENRES = ["blues", "jazz", "country", "funk", "r&b/soul", "pop", "fusion"]

# installed plugin folder -> (schema name, style the plugin is known for: a prior, never counted as the preset's genre)
OWNED = {
    "Archetype John Mayer X": ("Archetype John Mayer X", ["blues", "pop", "r&b/soul", "rock"]),
    "Archetype Gojira": ("Archetype Gojira", ["metal", "progressive"]),
    "Archetype Misha Mansoor X": ("Archetype Misha Mansoor X", ["progressive", "metal"]),
    "Archetype Petrucci X": ("Archetype Petrucci X", ["progressive", "metal", "fusion"]),
}
# a folder directly under the plugin folder named after an artist holds that artist's signature presets
SIGNATURE = {"john mayer": "John Mayer", "misha mansoor": "Misha Mansoor", "john petrucci": "John Petrucci", "gojira": "Gojira"}
FACTORY = {"neural dsp", "factory"}
NOT_PRESETS = {"default.xml", "reset all settings.xml", "midi.xml", "settings.xml", "preferences.xml"}

# well-known players on the plugins' Artists/ pages whose main style is public knowledge; the famous-artist list is consulted first
CREATOR_GENRE = {
    "cory wong": "funk", "mark lettieri": "funk", "eric krasno": "funk", "joey landreth": "blues", "ariel posen": "blues", "justus west": "blues",
    "lindsay ell": "country", "james valentine": "pop", "kiko loureiro": "metal", "lee malia": "metal", "jake bowen": "progressive",
    "adam nolly getgood": "progressive", "plini": "progressive", "joe satriani": "rock",
}
# whole-word evidence in a preset name / sub-folder, rare genres first; deliberately narrow
NAME_GENRE = [
    ("jazz", r"jazz\w*|bebop|hollow ?body|archtop|wes montgomery"),
    ("country", r"country|nashville|twang\w*|chicken pick\w*|bluegrass|pedal steel"),
    ("r&b/soul", r"soul\w*|neo[- ]?soul|motown|gospel|r&b|rnb|d'angelo"),
    ("funk", r"funk\w*|clav\w*"),
    ("fusion", r"fusion"),
    ("blues", r"blues\w*|srv|stevie ray|bb king|b\.b\. king|clapton|bonamassa"),
    ("pop", r"pop"),
    ("punk", r"punk"),
    ("metal", r"metal\w*|djent|thrash\w*|chug+|death"),
    ("progressive", r"prog\w*"),
    ("indie/alternative", r"indie|shoegaze|midwest emo"),
]


THREAD_OF_PLUGIN = re.compile(r"(?i)\barchetype\b|\bpresets?\s+thread\b")


def name_genre(*texts: str | None) -> str | None:
    text = " ".join(t for t in texts if t).lower()
    for genre, rx in NAME_GENRE:
        if re.search(rf"(?<![a-z])(?:{rx})(?![a-z])", text):
            return genre
    return None


def evidence(creator: str | None, artist: str | None, *texts: str | None, famous: FamousArtists) -> tuple[str | None, str | None]:
    """(genre, origin) from the strongest evidence available, else (None, None)."""
    for who, origin in ((artist, "famous_artist"), (creator, "famous_artist")):
        hit = famous.match(who) if who else None
        if hit and hit[0] != "other":
            return hit[0], origin
    if creator and creator.lower() in CREATOR_GENRE:
        return CREATOR_GENRE[creator.lower()], "creator_table"
    g = name_genre(*texts)
    return (g, "name_keyword") if g else (None, None)


# ----------------------------------------------------------------------------------------------------------- 1. local plugin presets
def default_roots() -> list[Path]:
    return [Path(r"C:\ProgramData\Neural DSP"), Path.home() / "AppData" / "Roaming" / "Neural DSP", Path.home() / "Documents" / "Neural DSP"]


def harvest_local(roots: list[Path] | None = None, plugins: list[str] | None = None, sink: Sink | None = None) -> dict:
    """Copy every factory / artist ``.xml`` preset of the owned plugins into ``presets/neuraldsp`` and log it in its manifest.

    Layout under a plugin folder: ``Artists/<creator>/<name>.xml`` (presets made by that guitarist), ``Neural DSP/`` or ``Factory/`` (factory),
    ``<Artist>/`` for a signature artist (``John Mayer/``), ``User/`` (the owner's own), plus ``Default.xml`` / settings files, which are skipped."""
    sink, famous = sink or Sink(), FamousArtists()
    wanted = {p.lower() for p in (plugins or OWNED)}
    stats: Counter = Counter()
    for root in roots or default_roots():
        if not root.exists():
            stats["missing_root"] += 1
            continue
        for plugin_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name.lower() in wanted):
            schema_name, prior = OWNED.get(plugin_dir.name, (plugin_dir.name, []))
            for f in sorted(plugin_dir.rglob("*.xml")):
                if f.name.lower() in NOT_PRESETS:
                    stats["skipped_non_preset"] += 1
                    continue
                parts = f.relative_to(plugin_dir).parts[:-1]
                top = parts[0].lower() if parts else ""
                if top == "artists" and len(parts) >= 2:
                    category, creator, artist, sub = "artist", parts[1], None, "/".join(parts[2:])
                elif top in FACTORY:
                    category, creator, artist, sub = "factory", None, None, "/".join(parts[1:])
                elif top in SIGNATURE:
                    category, creator, artist, sub = "signature", None, SIGNATURE[top], "/".join(parts[1:])
                elif top == "user":
                    category, creator, artist, sub = "user", None, None, "/".join(parts[1:])
                else:
                    category, creator, artist, sub = "other", None, None, "/".join(parts)
                data = f.read_bytes()
                if not valid(data, ".xml"):
                    stats["not_a_preset"] += 1
                    continue
                genre, origin = evidence(creator, artist, f.stem, sub, famous=famous)
                meta = {"source": "neuraldsp_local", "source_url": None, "title": f.stem, "plugin_family": schema_name, "category": category,
                        "subfolder": sub or None, "local_path": str(f), "tags": [], "genre_prior": prior,
                        "instrument_hint": "bass" if re.search(r"(?<![a-z])bass(?![a-z])", f.stem.lower()) else None}
                if creator:
                    meta["author"] = creator
                if artist:
                    meta["artist"], meta["famous_artist"] = artist, artist
                if genre:
                    meta["genre"], meta["genre_origin"] = genre, origin
                rows = sink.add(data, f.name, meta)
                stats[f"stored:{schema_name}"] += len(rows)
                stats["stored"] += len(rows)
    return {**stats, **{f"sink_{k}": v for k, v in sink.stats.items()}}


# ------------------------------------------------------------------------------------------------------------ 2. Fender Mustang
class MustangSite(GuitarPatches):
    """GuitarPatches' Mustang libraries (``mustang1`` = Mustang I-IV, ``mustang3`` = Mustang v3 / GT): plain-XML ``.fuse`` downloads."""

    name = "guitarpatches-mustang"
    file_ext = ".fuse"

    def __init__(self, max_pages: int = 50):
        super().__init__(("mustang1", "mustang3"), max_pages)

    def pages(self, client, query):
        for items in super().pages(client, query):
            for it in items:
                it.row["source_id"] = it.row["source_id"].replace("boss-", "fender-", 1)
                it.row["source"] = "guitarpatches"
            yield items

    @staticmethod
    def validate(data: bytes) -> bool:
        return is_fuse(data)


def tag_fender(folder: Path = FENDER, famous: FamousArtists | None = None) -> dict:
    """Add what the .fuse file itself says (author, tags, rating, Fender genre codes) and genre evidence to the manifest rows."""
    famous, stats = famous or FamousArtists(), Counter()
    for mp in find_manifests(folder):
        rows = read_rows(mp)
        for r in rows:
            if not r.get("file"):
                continue
            try:
                fp = parse_fuse(mp.parent / r["file"])
            except Exception as e:
                log.warning("%s: %s", r["file"], e)
                stats["unreadable"] += 1
                continue
            r.update(fuse_name=fp.name, fuse_author=fp.author, fuse_tags=fp.tags, fuse_genre_codes=list(fp.genre_codes), amp_model=fp.amp_name)
            if fp.rating is not None and r.get("rating") is None:
                r["rating"] = fp.rating
            if not r.get("genre"):
                genre, origin = evidence(None, r.get("artist"), r.get("title"), " ".join(fp.tags), famous=famous)
                if genre:
                    r["genre"], r["genre_origin"] = genre, origin
            stats["tagged"] += 1
            stats["with_genre"] += bool(r.get("genre"))
        write_rows(mp, rows)
    return dict(stats)


def crawl_mustang(out: Path = FENDER, limit: int | None = None) -> dict:
    stats = crawl(MustangSite(), out, famous=None, allow_artist_only=True, limit=limit)
    return {"crawl": str(stats), "tag": tag_fender(out)}


# ------------------------------------------------------------------------------------------------------------- 3. forum / 4. gap genres
FORUM_KEYWORDS = ["John Mayer", "Mayer", "Misha", "Mansoor", "Petrucci", "Gojira", "blues", "jazz", "funk", "country", "neo soul", "soul", "fusion",
                  "pop", "r&b", "clean", "edge of breakup"]


def tag_neuraldsp(folder: Path = NEURALDSP, famous: FamousArtists | None = None) -> dict:
    """Genre evidence for rows that have none (forum rows: thread title / file name / tags; the artists' own names)."""
    famous, stats = famous or FamousArtists(), Counter()
    for mp in find_manifests(folder):
        rows = read_rows(mp)
        for r in rows:
            if r.get("famous_artist") and r.get("source") == "neuraldsp_forum" and THREAD_OF_PLUGIN.search(r.get("title") or ""):
                # "Archetype: Plini Presets Thread": the artist is the plugin's namesake, not who the preset sounds like, so its genre is a prior only
                if r.get("genre"):
                    r["genre_prior"] = list(dict.fromkeys([*(r.get("genre_prior") or []), r.pop("genre")]))
                    r.pop("genre_origin", None)
                r["plugin_artist"] = r.pop("famous_artist")
                r.pop("artist", None)
                stats["demoted_plugin_thread"] += 1
            if r.get("genre"):
                continue
            genre, origin = evidence(r.get("author") if r.get("category") == "artist" else None, r.get("artist"), r.get("title"),
                                     r.get("original_name"), " ".join(r.get("tags") or []), famous=famous)
            if genre:
                r["genre"], r["genre_origin"] = genre, origin
                stats["tagged"] += 1
        write_rows(mp, rows)
    return dict(stats)


def crawl_forum(keywords: list[str] | None = None, delay: float = 2.0, limit: int | None = None) -> dict:
    client = PoliteClient(FORUM_BASE, None, delay, cache_dir=URLS / "_cache" / "neural_forum")
    fetcher = Fetcher(delay_sec=delay, clients={FORUM_BASE: client})
    sink, ledger = Sink(), Ledger()
    stats = forum_crawl(sink, fetcher, ledger, client, limit=limit, pattern=keyword_re(keywords or FORUM_KEYWORDS))
    return {**stats, "sink": sink.stats, "genre_evidence": tag_neuraldsp()}


def crawl_gaps(genres: list[str] | None = None, limit: int | None = None) -> dict:
    """Katana and Boss GT libraries restricted to the famous artists of the gap genres (their cached listings make a re-run cheap)."""
    table = FamousArtists().table
    famous = FamousArtists({g: table[g] for g in (genres or GAP_GENRES) if g in table})
    return {"katana": str(crawl(GuitarPatches(UNITS), KATANA, famous=famous, limit=limit)),
            "boss_gt": str(crawl(GuitarPatches(GT_UNITS), BOSS_GT, famous=famous, limit=limit))}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m patch_scraper.exhaustive_scraper", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["local", "mustang", "forum", "gaps", "tag", "all"])
    ap.add_argument("--root", action="append", help="local mode: folder holding the plugin folders (repeatable; default the 3 Neural DSP locations)")
    ap.add_argument("--plugin", action="append", help="local mode: plugin folder name (repeatable; default the 4 owned plugins)")
    ap.add_argument("--keyword", action="append", help="forum mode: topic keyword (repeatable; default the plugin + gap-genre list)")
    ap.add_argument("--genre", action="append", help="gaps mode: genre (repeatable; default the 7 gap genres)")
    ap.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    out = {}
    if a.mode in ("local", "all"):
        out["local"] = harvest_local([Path(r) for r in a.root] if a.root else None, a.plugin)
    if a.mode in ("mustang", "all"):
        out["mustang"] = crawl_mustang(limit=a.limit)
    if a.mode in ("forum", "all"):
        out["forum"] = crawl_forum(a.keyword, limit=a.limit)
    if a.mode in ("gaps", "all"):
        out["gaps"] = crawl_gaps(a.genre, a.limit)
    if a.mode in ("tag", "all"):
        out["tag"] = {"neuraldsp": tag_neuraldsp(), "fender": tag_fender()}
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))


if __name__ == "__main__":
    main()
