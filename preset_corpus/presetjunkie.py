"""Logged-in, rate-limited downloader for community presets on presetjunkie.com.

Rules this script follows:
  * the session cookie comes from the environment (PRESETJUNKIE_SESSION, a `name=value; name2=value2` Cookie header value
    copied from your own logged-in browser); it is never stored in code or on disk by this script
  * robots.txt is honoured, one request per `delay_sec` (default 2 s), and everything fetched is cached so a re-run only
    asks for what is missing
  * the site has no published terms in the plan's notes: check them before training on, or redistributing, anything you
    download. Keep the files private.

The site's HTML is not known here. Spike first:

    python -m preset_corpus.presetjunkie inspect "https://www.presetjunkie.com/<a listing page>"

That saves the page under data/presetjunkie/_inspect/ so you can write the three regexes in config/presetjunkie.json
(`preset_link_regex`, `download_link_regex`, `meta_regexes`) and the listing URL template. Defaults below are placeholders
and almost certainly need editing.
"""

from __future__ import annotations

import argparse
import html
import json
import logging
import os
import re
import time
import urllib.parse
import urllib.request
import urllib.robotparser
from dataclasses import dataclass, field
from pathlib import Path

from common.paths import CONFIG_DIR, DATA_DIR, ensure

log = logging.getLogger(__name__)

CONFIG_PATH = CONFIG_DIR / "presetjunkie.json"
OUT_DIR = DATA_DIR / "presetjunkie"
USER_AGENT = "tone-parameter-estimation/0.1 (personal research; contact: see repository)"


@dataclass
class SiteConfig:
    base_url: str = "https://www.presetjunkie.com"
    listing_url: str = "{base}/presets/{plugin_slug}?page={page}"          # PLACEHOLDER
    plugin_slugs: dict[str, str] = field(default_factory=lambda: {
        "Archetype Plini": "archetype-plini", "Archetype Abasi": "archetype-abasi", "Archetype Gojira": "archetype-gojira",
        "Archetype John Mayer X": "archetype-john-mayer-x", "Archetype Misha Mansoor X": "archetype-misha-mansoor-x",
        "Archetype Nolly X": "archetype-nolly-x", "Archetype Petrucci X": "archetype-petrucci-x"})
    preset_link_regex: str = r'href="(/preset/[^"#?]+)"'                    # PLACEHOLDER: link from listing to detail page
    download_link_regex: str = r'href="([^"]+\.(?:xml|zip|preset))"'        # PLACEHOLDER: link from detail page to the file
    meta_regexes: dict[str, str] = field(default_factory=lambda: {          # PLACEHOLDERS; group 1 of each regex
        "title": r"<h1[^>]*>(.*?)</h1>", "author": r'rel="author"[^>]*>(.*?)<', "song": r'data-song="([^"]*)"',
        "artist": r'data-artist="([^"]*)"', "genre": r'data-genre="([^"]*)"', "tags": r'rel="tag"[^>]*>(.*?)<'})
    max_pages: int = 50
    delay_sec: float = 2.0

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "SiteConfig":
        if path.exists():
            return cls(**json.loads(path.read_text(encoding="utf-8")))
        return cls()

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(self.__dict__, indent=2), encoding="utf-8")


class Client:
    def __init__(self, cfg: SiteConfig, cookie: str | None = None):
        self.cfg = cfg
        self.cookie = cookie or os.environ.get("PRESETJUNKIE_SESSION", "")
        if not self.cookie:
            raise RuntimeError("set PRESETJUNKIE_SESSION to the Cookie header of your logged-in session (see module docstring)")
        self._last = 0.0
        self.robots = urllib.robotparser.RobotFileParser()
        self.robots.set_url(urllib.parse.urljoin(cfg.base_url, "/robots.txt"))
        try:
            self.robots.read()
        except Exception as e:
            log.warning("could not read robots.txt (%s); proceeding conservatively", e)

    def get(self, url: str) -> tuple[bytes, str]:
        url = urllib.parse.urljoin(self.cfg.base_url, url)
        if not self.robots.can_fetch(USER_AGENT, url):
            raise PermissionError(f"robots.txt disallows {url}")
        wait = self.cfg.delay_sec - (time.time() - self._last)
        if wait > 0:
            time.sleep(wait)
        req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Cookie": self.cookie})
        with urllib.request.urlopen(req, timeout=60) as r:
            self._last = time.time()
            return r.read(), r.headers.get("Content-Type", "")


def _text(s: str) -> str:
    return html.unescape(re.sub(r"<[^>]+>", "", s)).strip()


def parse_meta(page: str, cfg: SiteConfig) -> dict:
    meta: dict = {}
    for key, rx in cfg.meta_regexes.items():
        found = [_text(m) for m in re.findall(rx, page, flags=re.S | re.I)]
        meta[key] = found if key == "tags" else (found[0] if found else "")
    return meta


def preset_id(link: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", link.strip("/"))[-80:]


def inspect(url: str) -> Path:
    """Spike: fetch one page you can see in your browser and save it for writing the regexes."""
    cfg = SiteConfig.load()
    body, ctype = Client(cfg).get(url)
    out = ensure(OUT_DIR / "_inspect") / "page.html"
    out.write_bytes(body)
    print(f"saved {len(body)} bytes ({ctype}) -> {out}")
    return out


def crawl(plugin: str, cfg: SiteConfig | None = None, limit: int | None = None) -> int:
    """List -> detail -> file download for one plugin, resumable. Returns the number of new presets."""
    cfg = cfg or SiteConfig.load()
    client = Client(cfg)
    slug = cfg.plugin_slugs[plugin]
    root = ensure(OUT_DIR / plugin)
    new = 0
    seen: set[str] = set()
    for page in range(1, cfg.max_pages + 1):
        listing, _ = client.get(cfg.listing_url.format(base=cfg.base_url, plugin_slug=slug, page=page))
        links = [l for l in dict.fromkeys(re.findall(cfg.preset_link_regex, listing.decode("utf-8", "replace"))) if l not in seen]
        if not links:
            break
        seen.update(links)
        for link in links:
            d = root / preset_id(link)
            if (d / "meta.json").exists():
                continue
            detail, _ = client.get(link)
            text = detail.decode("utf-8", "replace")
            m = re.search(cfg.download_link_regex, text)
            if not m:
                log.warning("no download link on %s", link)
                continue
            data, ctype = client.get(m.group(1))
            ensure(d)
            ext = Path(urllib.parse.urlparse(m.group(1)).path).suffix or ".bin"
            (d / f"preset{ext}").write_bytes(data)
            meta = {**parse_meta(text, cfg), "source_url": urllib.parse.urljoin(cfg.base_url, link), "plugin": plugin,
                    "content_type": ctype}
            (d / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
            new += 1
            if limit and new >= limit:
                return new
    return new


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("write-config", help="write config/presetjunkie.json with the placeholder defaults to edit")
    i = sub.add_parser("inspect")
    i.add_argument("url")
    c = sub.add_parser("crawl")
    c.add_argument("plugin")
    c.add_argument("--limit", type=int)
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    if a.cmd == "write-config":
        SiteConfig().save()
        print(f"wrote {CONFIG_PATH}")
    elif a.cmd == "inspect":
        inspect(a.url)
    else:
        print(f"{crawl(a.plugin, limit=a.limit)} new presets")


if __name__ == "__main__":
    main()
