"""Shared downloader for the community spiders: link classification, polite fetching with dead-link handling, link ledger.

``harvest(url, meta, fetcher, sink, ledger)`` is what ``reddit.py`` / ``neural_forum.py`` / ``tonehunt.py`` call for every link they find:

    link -> classify (direct / dropbox / google drive / unsupported host) -> fetch -> ``Sink.add`` (validate, unpack, manifest)

Every attempt is written to ``data/data_scraping/links/links.jsonl`` with its outcome, so a re-run skips links that worked or are dead
and retries the ones that failed for a passing reason (network error, rate limit).

Statuses: ``ok`` | ``empty`` (downloaded but nothing valid inside) | ``dead`` (404/410) | ``forbidden`` (401/403) | ``robots_blocked`` |
``html`` (a page, not a file) | ``unsupported`` (Mega, Drive folders, ...) | ``too_large`` | ``network_error`` | ``rate_limited``.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.error
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from patch_scraper.archive import Sink, sniff_ext
from patch_scraper.http import PoliteClient
from patch_scraper.manifest import append_row, read_rows
from patch_scraper.paths import LINKS, URLS

log = logging.getLogger(__name__)

URL_RE = re.compile(r"https?://[^\s<>\"'\])]+")
FILE_EXTS = (".xml", ".nam", ".wav", ".zip", ".rar")
RETRY_LATER = {"network_error", "rate_limited"}
MAX_DOWNLOAD = 120 * 1024 * 1024


def extract_urls(text: str) -> list[str]:
    """URLs in free text (Reddit self-text, forum posts), trailing punctuation removed, order kept, duplicates dropped."""
    found = [u.rstrip(".,;:!?)*_") for u in URL_RE.findall(html.unescape(text or ""))]
    return list(dict.fromkeys(found))


def classify(url: str) -> tuple[str, str | None]:
    """(kind, direct_url). kind: direct | dropbox | drive | unsupported | ignore. ``direct_url`` is None unless a download URL is known."""
    p = urllib.parse.urlparse(url)
    host, path = p.netloc.lower().removeprefix("www."), p.path
    if host.endswith("dropbox.com"):
        q = urllib.parse.parse_qs(p.query, keep_blank_values=True)
        q["dl"] = ["1"]
        return "dropbox", urllib.parse.urlunparse(p._replace(query=urllib.parse.urlencode(q, doseq=True)))
    if host == "drive.google.com" or host == "docs.google.com":
        if "/folders/" in path:
            return "unsupported", None                      # a folder needs the Drive API; the spider logs it for manual download
        m = re.search(r"/file/d/([\w-]+)", path) or re.search(r"[?&]id=([\w-]+)", url)
        if m:
            return "drive", f"https://drive.google.com/uc?export=download&id={m.group(1)}"
        return "unsupported", None
    if host.endswith(("mega.nz", "mega.io", "mediafire.com", "wetransfer.com", "we.tl", "1drv.ms", "onedrive.live.com", "sendspace.com")):
        return "unsupported", None
    if path.lower().endswith(FILE_EXTS) or host in ("downloads.neuraldsp.com",):
        return "direct", url
    return "ignore", None                                    # an ordinary web page / video / image


@dataclass
class Fetched:
    status: str
    data: bytes = b""
    filename: str = ""
    reason: str = ""


class Fetcher:
    """One ``PoliteClient`` per host (each host's own robots.txt and Crawl-delay), downloads never cached on disk."""

    def __init__(self, delay_sec: float = 3.0, check_robots: bool = True, clients: dict[str, PoliteClient] | None = None,
                 make_client: Callable[[str], PoliteClient] | None = None, max_bytes: int = MAX_DOWNLOAD):
        self.delay_sec, self.check_robots, self.max_bytes = delay_sec, check_robots, max_bytes
        self.clients = clients if clients is not None else {}
        self._make = make_client or (lambda base: PoliteClient(base, None, delay_sec, cache_dir=URLS / "_cache" / "http", check_robots=check_robots))

    def client(self, url: str) -> PoliteClient:
        p = urllib.parse.urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base not in self.clients:
            self.clients[base] = self._make(base)
        return self.clients[base]

    def get(self, url: str, headers: dict | None = None) -> Fetched:
        try:
            data = self.client(url).get(url, headers=headers, cache=False)
        except PermissionError as e:
            return Fetched("robots_blocked", reason=str(e))
        except urllib.error.HTTPError as e:
            status = {404: "dead", 410: "dead", 401: "forbidden", 403: "forbidden", 429: "rate_limited"}.get(e.code, "network_error")
            return Fetched(status, reason=f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, ConnectionError, OSError, ValueError) as e:
            return Fetched("network_error", reason=str(getattr(e, "reason", e)))
        if len(data) > self.max_bytes:
            return Fetched("too_large", reason=f"{len(data)} bytes")
        return Fetched("ok", data=data, filename=Path(urllib.parse.unquote(urllib.parse.urlparse(url).path)).name)

    def download(self, kind: str, url: str) -> Fetched:
        res = self.get(url)
        if res.status != "ok":
            return res
        if kind == "drive" and res.data[:1] == b"<":               # large-file interstitial: confirm form carries id / confirm / uuid
            text = res.data.decode("utf-8", "replace")
            fields = dict(re.findall(r'<input type="hidden" name="([^"]+)" value="([^"]*)"', text))
            if fields.get("id") and fields.get("confirm"):
                res = self.get("https://drive.usercontent.google.com/download?" + urllib.parse.urlencode(fields))
                if res.status != "ok":
                    return res
            elif "quota" in text.lower() or "too many users" in text.lower():
                return Fetched("rate_limited", reason="Google Drive download quota exceeded")
        if sniff_ext(res.data) is None and res.data[:1] == b"<" and b"<html" in res.data[:4096].lower():
            return Fetched("html", reason="page returned instead of a file (private, deleted or login required)")
        return res


class Ledger:
    """``links.jsonl``: one row per link attempt; the newest row for a URL wins."""

    def __init__(self, path: Path = LINKS):
        self.path = Path(path)
        self.latest = {r["url"]: r for r in read_rows(self.path)}

    def done(self, url: str) -> bool:
        row = self.latest.get(url)
        return row is not None and row["status"] not in RETRY_LATER

    def log(self, url: str, status: str, **fields) -> dict:
        row = {"url": url, "status": status, **fields}
        self.latest[url] = row
        append_row(self.path, row)
        return row


def harvest(url: str, meta: dict, fetcher: Fetcher, sink: Sink, ledger: Ledger) -> list[dict]:
    """Download one link into the sink. Returns the manifest rows stored (empty when the link is dead, blocked or has nothing valid)."""
    if ledger.done(url):
        return []
    kind, direct = classify(url)
    if kind == "ignore":
        return []
    origin = {k: meta.get(k) for k in ("source", "source_url", "title", "author", "post_id") if meta.get(k)}
    if direct is None:
        ledger.log(url, "unsupported", host=urllib.parse.urlparse(url).netloc, **origin)
        return []
    res = fetcher.download(kind, direct)
    if res.status != "ok":
        log.info("%s -> %s (%s)", url, res.status, res.reason)
        ledger.log(url, res.status, reason=res.reason, **origin)
        return []
    try:
        rows = sink.add(res.data, res.filename or "download", {**meta, "download_url": url})
    except Exception as e:                                       # a malformed archive must never stop the crawl
        log.exception("could not process %s", url)
        ledger.log(url, "empty", reason=f"{type(e).__name__}: {e}", **origin)
        return []
    ledger.log(url, "ok" if rows else "empty", stored=len(rows), **origin)
    return rows


def link_ledger_summary(ledger: Ledger) -> dict[str, int]:
    out: dict[str, int] = {}
    for r in ledger.latest.values():
        out[r["status"]] = out.get(r["status"], 0) + 1
    return out
