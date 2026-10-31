"""Neural DSP Unity forum (a Discourse site) -> user-uploaded ``.xml`` presets and preset-pack links.

    python -m patch_scraper.neural_forum                              # every page of the Plugins + Community categories, preset threads
    python -m patch_scraper.neural_forum --all-topics                 # open every topic, not only preset-looking titles
    python -m patch_scraper.neural_forum --thread 3545                # one thread
    python -m patch_scraper.neural_forum --category 18 --max-pages 2 --limit 200

Uses the forum's own JSON views (``/c/<id>.json``, ``/t/<id>.json``, ``/posts.json``), which robots.txt allows. Each post's HTML is
searched for ``/uploads/...`` attachments and for external links (Dropbox, ``downloads.neuraldsp.com``); everything goes through
``download.harvest`` so zips are unpacked and only valid presets reach ``data/data_scraping/presets/neuraldsp``.

Topic lists are paged until Discourse stops returning ``more_topics_url`` (the Plugins category alone is ~33 pages of 30 topics). By default
only topics whose title looks like preset sharing (``TITLE_RE``) are opened; ``--all-topics`` opens every one. Google Drive links are not
requested: Drive's robots.txt disallows ``/uc`` downloads, so they are logged as ``robots_blocked`` straight away. Download those packs by
hand and run ``python -m patch_scraper.archive ingest-local <folder>``.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import urllib.parse
import urllib.error
from typing import Iterator

from patch_scraper.archive import Sink
from patch_scraper.download import Fetcher, Ledger, classify, extract_urls, harvest, link_ledger_summary
from patch_scraper.http import PoliteClient
from patch_scraper.paths import URLS

log = logging.getLogger(__name__)

BASE = "https://unity.neuraldsp.com"
PLUGINS_CATEGORY = 18
COMMUNITY_CATEGORY = 21
DEFAULT_CATEGORIES = (PLUGINS_CATEGORY, COMMUNITY_CATEGORY)
TITLE_RE = re.compile(r"(?i)preset|patch|tones?|rig|dialed|settings|sounds?|captures?|pack|share|bank")
SKIP_HOSTS = ("drive.google.com", "docs.google.com", "drive.usercontent.google.com")
HREF_RE = re.compile(r'href="([^"]+)"')


def get_json(client: PoliteClient, url: str) -> dict | None:
    """JSON view of a forum page; None when it is missing, private or not JSON (never raises)."""
    try:
        return json.loads(client.get(url, cache=False).decode("utf-8", "replace"))
    except (urllib.error.HTTPError, urllib.error.URLError, PermissionError, ValueError, TimeoutError, OSError) as e:
        log.warning("forum request failed %s: %s", url, e)
        return None


def category_topics(client: PoliteClient, category: int, max_pages: int | None = None) -> Iterator[dict]:
    """Every topic of a category, page by page, until the listing has no ``more_topics_url`` (or ``max_pages`` is reached)."""
    seen: set[int] = set()
    page = 0
    while max_pages is None or page < max_pages:
        doc = get_json(client, f"{BASE}/c/{category}.json?page={page}")
        listing = (doc or {}).get("topic_list") or {}
        fresh = [t for t in listing.get("topics") or [] if t["id"] not in seen]
        if not fresh:
            return
        seen.update(t["id"] for t in fresh)
        yield from fresh
        if not listing.get("more_topics_url"):
            return
        page += 1


def keyword_re(words: list[str]) -> re.Pattern:
    """Case-insensitive whole-word pattern for a keyword list (``neo soul`` also matches ``neo-soul`` / ``neosoul``)."""
    parts = [re.escape(w.strip()).replace(r"\ ", r"[\s\-]?") for w in words if w.strip()]
    return re.compile(r"(?i)(?<![A-Za-z])(?:" + "|".join(parts) + r")(?![A-Za-z])")


def wanted(topic: dict, pattern: re.Pattern | None = None) -> bool:
    """Title (and tags) look like preset sharing, or match ``pattern`` when one is given (then the preset-title test is not applied)."""
    text = " ".join([topic.get("title") or "", *map(str, topic.get("tags") or [])])
    return bool((pattern or TITLE_RE).search(text))


def topic_posts(client: PoliteClient, topic_id: int) -> tuple[dict, list[dict]]:
    """(topic document, every post). ``/t/<id>.json`` carries the first ~20 posts; the rest are fetched by id."""
    doc = get_json(client, f"{BASE}/t/{topic_id}.json") or {}
    posts = list((doc.get("post_stream") or {}).get("posts") or [])
    have = {p["id"] for p in posts}
    missing = [i for i in (doc.get("post_stream") or {}).get("stream") or [] if i not in have]
    for i in range(0, len(missing), 20):
        query = urllib.parse.urlencode([("post_ids[]", pid) for pid in missing[i:i + 20]])
        more = get_json(client, f"{BASE}/t/{topic_id}/posts.json?{query}")
        posts += ((more or {}).get("post_stream") or {}).get("posts") or []
    return doc, posts


def post_links(post: dict) -> list[str]:
    """Absolute URLs of attachments and external links in one post (the ``cooked`` HTML)."""
    cooked = post.get("cooked") or ""
    links = [urllib.parse.urljoin(BASE, h.replace("&amp;", "&")) for h in HREF_RE.findall(cooked)]
    links += extract_urls(re.sub(r"<[^>]+>", " ", cooked))           # links typed as plain text
    out = []
    for u in dict.fromkeys(links):
        host = urllib.parse.urlparse(u).netloc
        if host.endswith("neuraldsp.com") and "/uploads/" not in u and "/downloads." not in u and not classify(u)[1]:
            continue                                                     # forum navigation, user profiles, ...
        if classify(u)[0] != "ignore":
            out.append(u)
    return out


def plugin_from_title(title: str) -> str | None:
    m = re.match(r"(?i)(.*?)\s*(?:-|–)?\s*presets?(?:\s+thread)?\b", title or "")
    return m.group(1).strip() or None if m else None


def all_topics(client: PoliteClient, categories: tuple[int, ...], max_pages: int | None, every: bool, stats: dict,
               pattern: re.Pattern | None = None) -> Iterator[dict]:
    seen: set[int] = set()
    for category in categories:
        for topic in category_topics(client, category, max_pages):
            stats["listed"] += 1
            if topic["id"] in seen:
                continue
            seen.add(topic["id"])
            if every or wanted(topic, pattern):
                yield topic
            else:
                stats["skipped_title"] += 1


def crawl(sink: Sink, fetcher: Fetcher, ledger: Ledger, client: PoliteClient, *, categories: tuple[int, ...] = DEFAULT_CATEGORIES,
          threads: list[int] | None = None, max_pages: int | None = None, limit: int | None = None, all_topics_: bool = False,
          pattern: re.Pattern | None = None) -> dict:
    stats = {"listed": 0, "skipped_title": 0, "threads": 0, "posts": 0, "links": 0, "skipped_drive": 0, "stored": 0}
    topics = [{"id": t, "title": None} for t in threads] if threads else all_topics(client, categories, max_pages, all_topics_, stats, pattern)
    for topic in topics:
        doc, posts = topic_posts(client, topic["id"])
        title = doc.get("title") or topic.get("title") or ""
        stats["threads"] += 1
        for post in posts:
            stats["posts"] += 1
            for url in post_links(post):
                stats["links"] += 1
                if urllib.parse.urlparse(url).netloc.lower() in SKIP_HOSTS:
                    if not ledger.done(url):
                        ledger.log(url, "robots_blocked", reason="Google Drive robots.txt disallows downloads; fetch by hand",
                                   source="neuraldsp_forum", source_url=f"{BASE}/t/{topic['id']}/{post['post_number']}", title=title)
                    stats["skipped_drive"] += 1
                    continue
                meta = {"source": "neuraldsp_forum", "source_url": f"{BASE}/t/{topic['id']}/{post['post_number']}", "title": title,
                        "author": post.get("username"), "post_id": post["id"], "plugin_thread": plugin_from_title(title),
                        "likes": next((a.get("count") for a in post.get("actions_summary") or [] if a.get("id") == 2), None),
                        "date": (post.get("created_at") or "")[:10] or None, "tags": []}
                stats["stored"] += len(harvest(url, meta, fetcher, sink, ledger))
                if limit and stats["stored"] >= limit:
                    return stats
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m patch_scraper.neural_forum")
    ap.add_argument("--category", type=int, action="append", help=f"category id (repeatable; default {DEFAULT_CATEGORIES})")
    ap.add_argument("--thread", type=int, action="append", help="crawl only these topic ids")
    ap.add_argument("--max-pages", type=int, help="category pages of 30 topics (default: all)")
    ap.add_argument("--all-topics", action="store_true", help="open every topic, not only titles that look like preset sharing")
    ap.add_argument("--keyword", action="append", default=[], help="open only topics whose title / tags contain this word (repeatable)")
    ap.add_argument("--limit", type=int, help="stop after this many stored files")
    ap.add_argument("--delay", type=float, default=2.0)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    client = PoliteClient(BASE, None, args.delay, cache_dir=URLS / "_cache" / "neural_forum")
    fetcher = Fetcher(delay_sec=args.delay, clients={BASE: client})
    sink, ledger = Sink(), Ledger()
    print(crawl(sink, fetcher, ledger, client, categories=tuple(args.category or DEFAULT_CATEGORIES), threads=args.thread,
                max_pages=args.max_pages, limit=args.limit, all_topics_=args.all_topics,
                pattern=keyword_re(args.keyword) if args.keyword else None))
    print("sink:", sink.stats, "| links:", link_ledger_summary(ledger))


if __name__ == "__main__":
    main()
