"""Reddit spider: preset-pack links from r/NeuralDSP and r/NAM_NeuralAmpModeler posts.

    $env:REDDIT_TOKEN = "<OAuth bearer token of your own Reddit app>"
    python -m patch_scraper.reddit --limit 100
    python -m patch_scraper.reddit --sub NeuralDSP --query "preset pack" --max-pages 3

Reddit's ``robots.txt`` is ``Disallow: /`` for anonymous crawlers, and ``PoliteClient`` honours robots.txt, so ``www.reddit.com/.../search.json``
is refused. The sanctioned route is the OAuth API (``oauth.reddit.com``, same JSON as the ``.json`` pages): register a "script" app at
https://www.reddit.com/prefs/apps, obtain a bearer token, and put it in ``REDDIT_TOKEN`` (never written to disk). Without the token the spider
stops with that explanation instead of working around robots.txt. Check Reddit's Data API terms before storing posts; only links and ids are kept.

For each post the title, self-text and link are searched for Google Drive / Dropbox / direct ``.xml .nam .wav .zip .rar`` links, which
``download.harvest`` fetches and ``archive.Sink`` unpacks into ``presets/neuraldsp`` and ``presets/nam``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.error
import urllib.parse
from typing import Iterator

from patch_scraper.archive import Sink
from patch_scraper.download import Fetcher, Ledger, classify, extract_urls, harvest, link_ledger_summary
from patch_scraper.http import PoliteClient
from patch_scraper.paths import URLS

log = logging.getLogger(__name__)

API = "https://oauth.reddit.com"
TOKEN_ENV = "REDDIT_TOKEN"
DEFAULT_SUBS = {"NeuralDSP": ["preset", "presets", "preset pack"], "NAM_NeuralAmpModeler": ["preset", "capture", "pack", "ir"]}
PAGE_SIZE = 100


def parse_listing(doc: dict) -> tuple[list[dict], str | None]:
    """Reddit listing JSON -> (posts, ``after`` cursor). Only the fields we keep are copied."""
    data = (doc or {}).get("data") or {}
    posts = []
    for child in data.get("children") or []:
        d = child.get("data") or {}
        if d.get("id"):
            posts.append({"id": d["id"], "title": d.get("title") or "", "selftext": d.get("selftext") or "", "url": d.get("url") or "",
                          "permalink": d.get("permalink") or "", "author": d.get("author"), "subreddit": d.get("subreddit"),
                          "flair": d.get("link_flair_text"), "created": d.get("created_utc"), "removed": bool(d.get("removed_by_category"))})
    return posts, data.get("after")


def post_links(post: dict) -> list[str]:
    candidates = [post["url"], *extract_urls(post["selftext"]), *extract_urls(post["title"])]
    return [u for u in dict.fromkeys(candidates) if u and "reddit.com" not in urllib.parse.urlparse(u).netloc and classify(u)[0] != "ignore"]


class Reddit:
    def __init__(self, client: PoliteClient | None = None, token: str | None = None):
        self.token = token if token is not None else os.environ.get(TOKEN_ENV, "")
        if not self.token and client is None:
            raise SystemExit(f"{__doc__.split(chr(10))[0]}\n\n{TOKEN_ENV} is not set. Reddit's robots.txt disallows anonymous crawling, so this spider "
                             "only uses the OAuth API; see the module docstring for how to get a token.")
        self.client = client or PoliteClient(API, None, 2.0, cache_dir=URLS / "_cache" / "reddit", check_robots=False)   # oauth host serves no robots.txt
        self.headers = {"Authorization": f"bearer {self.token}"} if self.token else {}

    def search(self, sub: str, query: str, max_pages: int) -> Iterator[dict]:
        after = None
        for _ in range(max_pages):
            params = {"q": query, "restrict_sr": "1", "sort": "new", "limit": PAGE_SIZE, "raw_json": 1, **({"after": after} if after else {})}
            url = f"{API}/r/{sub}/search.json?{urllib.parse.urlencode(params)}"
            try:
                doc = json.loads(self.client.get(url, headers=self.headers, cache=False).decode("utf-8", "replace"))
            except urllib.error.HTTPError as e:
                log.error("r/%s search %r failed: HTTP %s%s", sub, query, e.code, " (token expired or invalid)" if e.code in (401, 403) else "")
                return
            except (urllib.error.URLError, ValueError, TimeoutError, OSError) as e:
                log.error("r/%s search %r failed: %s", sub, query, e)
                return
            posts, after = parse_listing(doc)
            yield from posts
            if not after or not posts:
                return


def crawl(reddit: Reddit, fetcher: Fetcher, sink: Sink, ledger: Ledger, *, subs: dict[str, list[str]] | None = None, max_pages: int = 2,
          limit: int | None = None) -> dict:
    stats = {"posts": 0, "links": 0, "stored": 0}
    seen: set[str] = set()
    for sub, queries in (subs or DEFAULT_SUBS).items():
        for query in queries:
            for post in reddit.search(sub, query, max_pages):
                if post["id"] in seen or post["removed"]:
                    continue
                seen.add(post["id"])
                stats["posts"] += 1
                for url in post_links(post):
                    stats["links"] += 1
                    meta = {"source": "reddit", "source_url": "https://www.reddit.com" + post["permalink"], "title": post["title"],
                            "author": post["author"], "post_id": post["id"], "subreddit": post["subreddit"], "tags": [post["flair"]] if post["flair"] else []}
                    stats["stored"] += len(harvest(url, meta, fetcher, sink, ledger))
                    if limit and stats["stored"] >= limit:
                        return stats
    return stats


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="python -m patch_scraper.reddit")
    ap.add_argument("--sub", action="append", help="subreddit (repeatable); default r/NeuralDSP and r/NAM_NeuralAmpModeler")
    ap.add_argument("--query", action="append", help="search term (repeatable)")
    ap.add_argument("--max-pages", type=int, default=2, help="pages of 100 posts per search")
    ap.add_argument("--limit", type=int, help="stop after this many stored files")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    subs = {s: args.query or DEFAULT_SUBS.get(s, ["preset"]) for s in args.sub} if args.sub else (
        {s: args.query for s in DEFAULT_SUBS} if args.query else None)
    sink, ledger = Sink(), Ledger()
    print(crawl(Reddit(), Fetcher(), sink, ledger, subs=subs, max_pages=args.max_pages, limit=args.limit))
    print("sink:", sink.stats, "| links:", link_ledger_summary(ledger))


if __name__ == "__main__":
    main()
