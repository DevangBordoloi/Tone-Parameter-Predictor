"""Famous-artist list (``config/famous_artists.json``: genre -> artists) and a matcher for scraped artist strings.

A name matches anywhere in the artist string as a whole-word sequence ("Metallica - Master of Puppets" matches "Metallica"), except
the names in ``COMMON_WORD``: single words that are also ordinary words ("Yes", "Queen", "Tool", "Rush", "Heart") must equal the
whole artist string, otherwise "Queen of Hearts" or "Tool for the job" would count as the band.
"""

from __future__ import annotations

import json
import re

from common.paths import CONFIG_DIR

PATH = CONFIG_DIR / "famous_artists.json"

COMMON_WORD = {
    "yes", "queen", "tool", "rush", "boston", "kansas", "heart", "cream", "chic", "camel", "sade", "prince", "eagles", "journey",
    "genesis", "alabama", "kiss", "dio", "sia", "cynic", "sting", "parliament", "primus", "placebo", "blur", "muse", "disturbed",
    "korn", "santana", "slash", "who", "police", "doors", "clash", "cure", "damned", "verve", "meters", "offspring", "misfits",
    "killers", "strokes", "smiths", "temptations", "commodores", "distillers", "shakti", "cage the elephant", "1975",
}


def norm(s: str) -> str:
    s = s.lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    toks = [t for t in s.split() if t != "the"]
    return " ".join(toks)


class FamousArtists:
    def __init__(self, table: dict[str, list[str]] | None = None):
        table = table if table is not None else json.loads(PATH.read_text(encoding="utf-8"))
        self.table = table
        self._by_name: dict[str, tuple[str, str]] = {}          # normalised name -> (genre, display name); first genre wins
        for genre, names in table.items():
            for n in names:
                self._by_name.setdefault(norm(n), (genre, n))

    def names(self, genre: str | None = None) -> list[str]:
        if genre:
            return list(self.table.get(genre, []))
        return list(dict.fromkeys(n for names in self.table.values() for n in names))

    def match(self, artist: str | None) -> tuple[str, str] | None:
        """(genre, canonical name) of the famous artist named in ``artist``, or None."""
        if not artist:
            return None
        a = norm(artist)
        if not a:
            return None
        if a in self._by_name:
            return self._by_name[a]
        padded = f" {a} "
        best = None
        for key, val in self._by_name.items():
            if key not in COMMON_WORD and f" {key} " in padded and (best is None or len(key) > len(best[0])):
                best = (key, val)
        return best[1] if best else None
