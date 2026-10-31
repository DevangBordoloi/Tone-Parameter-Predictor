"""ToneHunt / TONE3000 spider (Playwright, async): amp-model captures (.nam), impulse responses (.wav) and packs (.zip).

    pip install playwright ; python -m playwright install chromium
    python -m patch_scraper.tonehunt --limit 50                                # default amp-model queries
    python -m patch_scraper.tonehunt --query "JCM800" --query "Mesa Rectifier" --artist Metallica
    $env:TONE3000_STORAGE_STATE = "C:\\path\\tone3000_state.json"              # a logged-in Playwright session, see below

ToneHunt is now TONE3000 (``tonehunt.org`` redirects to ``tone3000.com``); its ``robots.txt`` allows crawling. The site is a client-rendered
Next.js app, so the pages are driven with a real browser. Downloads normally need a free account: log in once with
``python -m playwright codegen --save-storage tone3000_state.json https://www.tone3000.com`` and point ``TONE3000_STORAGE_STATE`` at the file
(cookies are read by the browser only, never copied into this project). TONE3000 also publishes an official API (``/api/v1``, OAuth, own terms),
which is the sturdier route for a large crawl; this spider is the browser alternative.

UNVERIFIED: the CSS selectors below (``SELECTORS``) were written without being able to run a browser against the live site in this environment.
If a run finds no tones, adjust them there; nothing else is site-specific. Output is validated and unpacked by ``archive.Sink`` and lands in
``presets/nam`` (.nam, .wav). Titles and tags are passed through ``artists.py`` / ``genre.py`` so rows carry artist and genre where known.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import urllib.parse
from pathlib import Path

from patch_scraper.archive import Sink
from patch_scraper.artists import FamousArtists
from patch_scraper.download import Ledger, link_ledger_summary
from patch_scraper.http import PoliteClient, USER_AGENT
from patch_scraper.paths import URLS

log = logging.getLogger(__name__)

BASE = "https://www.tone3000.com"
STATE_ENV = "TONE3000_STORAGE_STATE"
AMP_QUERIES = ["Marshall JCM800", "Marshall Plexi", "Mesa Boogie Rectifier", "Fender Twin Reverb", "Fender Deluxe", "Vox AC30", "Peavey 5150",
               "EVH 5150", "Orange Rockerverb", "Soldano SLO", "Friedman BE-100", "Bogner Ecstasy", "Dumble", "Mesa Mark IV", "Diezel", "ENGL Powerball"]
SELECTORS = {
    "tone_link": "a[href*='/tones/'], a[href*='/tone/']",            # anchor to a tone's page in the search results
    "download": "button:has-text('Download'), a:has-text('Download')",
    "title": "h1",
    "tags": "a[href*='tag='], [data-testid='tag']",
}
DOWNLOAD_EXTS = (".nam", ".wav", ".zip")


def search_url(query: str, page: int = 1) -> str:
    return f"{BASE}/search?" + urllib.parse.urlencode({"query": query, "page": page})


def queries_for(artists: list[str] | None, queries: list[str] | None) -> list[str]:
    return list(dict.fromkeys([*(queries or []), *(f"{a} tone" for a in artists or [])])) or AMP_QUERIES


async def tone_links(page, query: str, max_pages: int, delay: float) -> list[str]:
    links: list[str] = []
    for n in range(1, max_pages + 1):
        await page.goto(search_url(query, n), wait_until="networkidle")
        hrefs = await page.eval_on_selector_all(SELECTORS["tone_link"], "els => els.map(e => e.href)")
        fresh = [h for h in dict.fromkeys(hrefs) if h not in links]
        if not fresh:
            break
        links += fresh
        await asyncio.sleep(delay)
    return links


async def fetch_tone(page, url: str, delay: float) -> tuple[dict, list[tuple[str, bytes]]]:
    """Open one tone page; return (metadata, [(filename, bytes), ...]) for every download the Download button produces."""
    await page.goto(url, wait_until="networkidle")
    title = (await page.text_content(SELECTORS["title"]) or "").strip() if await page.query_selector(SELECTORS["title"]) else ""
    tags = [t.strip() for t in await page.eval_on_selector_all(SELECTORS["tags"], "els => els.map(e => e.textContent)")]
    files: list[tuple[str, bytes]] = []
    for button in await page.query_selector_all(SELECTORS["download"]):
        try:
            async with page.expect_download(timeout=30_000) as info:
                await button.click()
            download = await info.value
            path = await download.path()
            if path:
                files.append((download.suggested_filename or "download", Path(path).read_bytes()))
        except Exception as e:                                       # login wall, timeout, button that is not a file download
            log.info("no download from %s: %s", url, type(e).__name__)
        await asyncio.sleep(delay)
    return {"title": title, "tags": [t for t in tags if t]}, files


async def crawl(sink: Sink, ledger: Ledger, *, queries: list[str], max_pages: int = 1, per_query: int = 20, limit: int | None = None,
                delay: float = 3.0, headless: bool = True, check_robots: bool = True, storage_state: str | None = None) -> dict:
    try:
        from playwright.async_api import async_playwright
    except ImportError as e:
        raise SystemExit("Playwright is not installed: pip install playwright && python -m playwright install chromium") from e
    robots = PoliteClient(BASE, None, delay, check_robots=check_robots).robots if check_robots else None
    stats = {"queries": 0, "tones": 0, "downloaded": 0, "stored": 0, "skipped_robots": 0}
    famous = sink.famous or FamousArtists()
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=headless)
        ctx = await browser.new_context(user_agent=USER_AGENT, storage_state=storage_state or None, accept_downloads=True)
        page = await ctx.new_page()
        try:
            for query in queries:
                stats["queries"] += 1
                try:
                    links = (await tone_links(page, query, max_pages, delay))[:per_query]
                except Exception as e:
                    log.warning("search %r failed: %s", query, e)
                    continue
                for url in links:
                    if ledger.done(url):
                        continue
                    if robots is not None and not robots.can_fetch(USER_AGENT, url):
                        stats["skipped_robots"] += 1
                        ledger.log(url, "robots_blocked", source="tonehunt")
                        continue
                    stats["tones"] += 1
                    try:
                        info, files = await fetch_tone(page, url, delay)
                    except Exception as e:                           # dead page, navigation timeout
                        ledger.log(url, "network_error", reason=type(e).__name__, source="tonehunt")
                        continue
                    stored = 0
                    meta = {"source": "tonehunt", "source_url": url, "title": info["title"] or query, "tags": info["tags"], "query": query}
                    hit = famous.match(" ".join([info["title"], *info["tags"]]))
                    if hit:
                        meta.update(artist=hit[1], genre=hit[0])
                    for name, blob in files:
                        stats["downloaded"] += 1
                        stored += len(sink.add(blob, name, meta))
                    stats["stored"] += stored
                    ledger.log(url, "ok" if stored else ("empty" if files else "forbidden"), stored=stored, source="tonehunt",
                               reason="" if files else "no download produced (login required, or selectors need updating)")
                    if limit and stats["stored"] >= limit:
                        return stats
        finally:
            await ctx.close()
            await browser.close()
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m patch_scraper.tonehunt")
    ap.add_argument("--query", action="append", help="amp model / gear to search (repeatable); default is a list of classic amps")
    ap.add_argument("--artist", action="append", help="search '<artist> tone' (repeatable)")
    ap.add_argument("--max-pages", type=int, default=1)
    ap.add_argument("--per-query", type=int, default=20, help="tones opened per search")
    ap.add_argument("--limit", type=int, help="stop after this many stored files")
    ap.add_argument("--delay", type=float, default=3.0)
    ap.add_argument("--show", action="store_true", help="show the browser window")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    sink, ledger = Sink(), Ledger()
    stats = asyncio.run(crawl(sink, ledger, queries=queries_for(args.artist, args.query), max_pages=args.max_pages, per_query=args.per_query,
                              limit=args.limit, delay=args.delay, headless=not args.show, storage_state=os.environ.get(STATE_ENV)))
    print(stats)
    print("sink:", sink.stats, "| links:", link_ledger_summary(ledger))


if __name__ == "__main__":
    main()
