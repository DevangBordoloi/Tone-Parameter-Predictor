"""GuitarPatches.com (Boss .tsl livesets) crawler: Katana amps, and the GT / ME / GX multi-effects as a second hardware family.

    python -m patch_scraper.guitarpatches --limit 20                 # Katana libraries, famous artists, all genres
    python -m patch_scraper.guitarpatches --genre jazz --genre country
    python -m patch_scraper.guitarpatches --all                      # every artist on the site
    python -m patch_scraper.guitarpatches --target boss_gt           # GT-1000, GT-100, GT-1, ME-80, ME-90, GX-100 -> presets/boss_gt
    python -m patch_scraper.guitarpatches --target boss_gt --unit GT100 --unit ME80

The GT / ME / GX units save the same Boss Tone Studio liveset JSON as the Katana, but with named parameters
(``patch_corpus/parsers/boss_tsl.py``); they are kept in their own folder so they can serve as an unseen-hardware test set.

The site has no search, so each Katana library (Katana, KATANAMKII, KATANAMKIII) is paged through (18 patches per page) and only
patches whose artist is on the famous-artist list get a detail request (rating, uploader, download link) and a download. The
download link is public; no login is needed. Both ``robots.txt`` and this client ask for 10 s between requests, so a full pass over a
library takes a while (listing pages are cached, so a re-run is quick).
"""

from __future__ import annotations

import logging
import re
import urllib.parse
from typing import Iterator

from patch_corpus.metadata import normalise_rating
from patch_scraper.http import PoliteClient
from patch_scraper.paths import BOSS_GT, KATANA
from patch_scraper.sites import Item, build_parser, clean_song, crawl, famous_filter, text

BASE = "https://guitarpatches.com"
UNITS = ("Katana", "KATANAMKII", "KATANAMKIII")
# Boss Tone Studio units with guitar presets on units.php (the bass GT-1B and the pre-BTS GT-10 / GT-8 / ME-50 file formats are left out)
GT_UNITS = ("GT1000", "GT100", "GT100v2", "GT1", "ME80", "ME90", "GX100")
TARGETS = {"katana": (UNITS, KATANA), "boss_gt": (GT_UNITS, BOSS_GT)}


def parse_listing(page: str, unit: str) -> list[Item]:
    items = []
    for cell in page.split('<div class="cell">')[1:]:
        m = re.search(r"patches\.php\?mode=show&(?:amp;)?unit=([^&']+)&(?:amp;)?ID=(\d+)", cell)
        if not m:
            continue
        name = re.search(r"<h1[^>]*>(.*?)</h1>", cell, re.S)
        artist = re.search(r"<h2[^>]*>(.*?)</h2>", cell, re.S)
        date = re.search(r'<p class="is-size-7">\s*([\d-]+)', cell)
        pid = m.group(2)
        a = text(artist.group(1)) if artist else None
        n = text(name.group(1)) if name else None
        song = clean_song(n, a)
        url = f"{BASE}/patches.php?mode=show&unit={m.group(1)}&ID={pid}"
        prefix = "katana" if m.group(1).lower().startswith("katana") else "boss"
        row = {"source_id": f"{prefix}-{m.group(1).lower()}-{pid}", "source": "guitarpatches", "source_url": url, "unit": m.group(1),
               "title": n, "artist": a, "song": song, "song_origin": "patch_name" if song else None, "date": date.group(1) if date else None,
               "tags": []}
        items.append(Item(url=url, row=row, needs_detail=True))
    return items


def parse_detail(page: str) -> dict:
    """rating / uploader / download link from a patch page. ``rating_text`` keeps the raw rating box for auditing."""
    box = re.search(r"Rating</p>\s*<h6[^>]*>(.*?)</h6>", page, re.S)
    rating_text = text(box.group(1)) if box else ""
    rating = count = None
    if rating_text and not re.search(r"(?i)no ratings", rating_text):
        n = re.search(r"\d+(?:\.\d+)?", rating_text)
        rating = float(n.group(0)) if n else None
        c = re.search(r"(\d+)\s*(?:vote|rating|review)", rating_text, re.I)
        count = int(c.group(1)) if c else None
    contributor = re.search(r"Contributor</label>\s*<h3[^>]*>(.*?)</h3>", page, re.S)
    link = re.search(r'href="([^"]*download\.php\?[^"]*mode=download[^"]*)"', page)
    return {"rating": None if rating is None else normalise_rating(rating), "rating_count": count, "rating_text": rating_text or None,
            "author": text(contributor.group(1)) if contributor else None,
            "download": urllib.parse.urljoin(BASE + "/", link.group(1).replace("&amp;", "&")) if link else None}


class GuitarPatches:
    name = "guitarpatches"
    file_ext = ".tsl"
    base_url = BASE
    cookie_env = None

    def __init__(self, units: tuple[str, ...] = UNITS, max_pages: int = 200):
        self.units, self.max_pages = units, max_pages

    def pages(self, client: PoliteClient, query: str | None) -> Iterator[list[Item]]:
        for unit in self.units:
            for page in range(1, self.max_pages + 1):
                url = f"{BASE}/patches.php?unit={unit}&sort=date&page={page}"
                items = parse_listing(client.get(url).decode("utf-8", "replace"), unit)
                if not items:
                    break
                yield items

    def enrich(self, client: PoliteClient, item: Item) -> None:
        d = parse_detail(client.get(item.url).decode("utf-8", "replace"))
        item.download_url = d.pop("download")
        item.row.update({k: v for k, v in d.items() if v is not None})
        item.needs_detail = False


def main(argv: list[str] | None = None) -> None:
    ap = build_parser("python -m patch_scraper.guitarpatches", None)
    ap.add_argument("--target", choices=sorted(TARGETS), default="katana", help="unit family and output folder (default katana)")
    ap.add_argument("--unit", action="append", help=f"library to scan (katana: {', '.join(UNITS)}; boss_gt: {', '.join(GT_UNITS)})")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    famous = famous_filter(args)
    units, out = TARGETS[args.target]
    site = GuitarPatches(tuple(args.unit or units), args.max_pages or 200)
    print(crawl(site, args.out or out, famous=famous, min_rating=args.min_rating, limit=args.limit,
                allow_artist_only=args.allow_artist_only))


if __name__ == "__main__":
    main()
