"""Polite HTTP client: robots.txt, per-client delay, on-disk cache, retry with back-off, optional session cookie from the environment."""

from __future__ import annotations

import hashlib
import logging
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import urllib.robotparser
from pathlib import Path

log = logging.getLogger(__name__)

USER_AGENT = "tone-parameter-estimation/0.1 (personal research; contact: see repository)"
_RETRY_CODES = {429, 500, 502, 503, 504}


class PoliteClient:
    def __init__(self, base_url: str, cookie_env: str | None = None, delay_sec: float = 2.0, cache_dir: Path | None = None,
                 check_robots: bool = True, retries: int = 3, max_retry_after: float | None = None):
        self.base_url = base_url
        self.cookie = os.environ.get(cookie_env, "") if cookie_env else ""
        if cookie_env and not self.cookie:
            log.warning("%s is not set: requests are anonymous, downloads that need a login will fail", cookie_env)
        self.delay_sec = delay_sec
        self.cache_dir = cache_dir
        self.retries = retries
        self.max_retry_after = max_retry_after          # a server asking for a longer wait than this gets the error raised instead
        self._last = 0.0
        self.robots: urllib.robotparser.RobotFileParser | None = None
        if check_robots:
            self.robots = urllib.robotparser.RobotFileParser()
            self.robots.set_url(urllib.parse.urljoin(base_url, "/robots.txt"))
            try:
                self.robots.read()
                crawl_delay = self.robots.crawl_delay(USER_AGENT)
                if crawl_delay:
                    self.delay_sec = max(self.delay_sec, float(crawl_delay))
                    log.info("%s asks for Crawl-delay %ss; using %ss", base_url, crawl_delay, self.delay_sec)
            except Exception as e:
                log.warning("could not read robots.txt (%s); proceeding conservatively", e)

    def _cache_path(self, url: str) -> Path | None:
        if self.cache_dir is None:
            return None
        return self.cache_dir / (hashlib.sha1(url.encode()).hexdigest() + ".bin")

    def get(self, url: str, headers: dict | None = None, data: bytes | None = None, cache: bool = True) -> bytes:
        url = urllib.parse.urljoin(self.base_url, url)
        path = self._cache_path(url) if cache and data is None else None
        if path is not None and path.exists():
            return path.read_bytes()
        if self.robots is not None and not self.robots.can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt disallows {url}")
        hdrs = {"User-Agent": USER_AGENT, **(headers or {})}
        if self.cookie:
            hdrs["Cookie"] = self.cookie
        for attempt in range(self.retries + 1):
            wait = self.delay_sec - (time.time() - self._last)
            if wait > 0:
                time.sleep(wait)
            try:
                with urllib.request.urlopen(urllib.request.Request(url, headers=hdrs, data=data), timeout=60) as r:
                    self._last = time.time()
                    body = r.read()
                break
            except urllib.error.HTTPError as e:
                self._last = time.time()
                if e.code not in _RETRY_CODES or attempt == self.retries:
                    raise
                pause = float(e.headers.get("Retry-After", 0) or 0) or self.delay_sec * 2 ** (attempt + 1)
                if self.max_retry_after is not None and pause > self.max_retry_after:
                    log.warning("HTTP %s from %s asks for a %.0fs wait (limit %.0fs); giving up on this request", e.code, url, pause,
                                self.max_retry_after)
                    raise
                log.warning("HTTP %s from %s; retrying in %.0fs", e.code, url, pause)
                time.sleep(pause)
            except urllib.error.URLError:
                self._last = time.time()
                if attempt == self.retries:
                    raise
                time.sleep(self.delay_sec * 2 ** (attempt + 1))
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(body)
        return body
