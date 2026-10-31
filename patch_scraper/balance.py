"""Per-genre counts and a capped copy of the corpus.

    python -m patch_scraper.balance report [presets_dir] --per-genre 300
    python -m patch_scraper.balance cap    [presets_dir] --per-genre 300 --out data/data_scraping/presets_balanced

``report`` shows which genres are short of the quota, so you can run ``crawl --artist ...`` for them. ``cap`` copies, per genre, the
best-rated patches up to the quota into a new folder with its own manifest, which ``patch_corpus.ingest`` reads like any other
folder. The originals are not touched, and an existing manifest in ``--out`` is replaced.
"""

from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from patch_scraper.genre import GENRES
from patch_scraper.manifest import append_row, find_manifests, read_rows, song_key
from patch_scraper.paths import PRESETS


def _all_rows(root: Path) -> list[tuple[Path, dict]]:
    return [(mp.parent, r) for mp in find_manifests(root) for r in read_rows(mp)]


def counts(root: Path) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {g: {"patches": 0, "songs": 0, "prior": 0} for g in GENRES}
    songs: dict[str, set] = defaultdict(set)
    for _, r in _all_rows(Path(root)):
        if not r.get("file"):                       # metadata-only row (Line 6 without a login): there is no preset to train on
            continue
        g = r.get("genre") or "other"
        out.setdefault(g, {"patches": 0, "songs": 0, "prior": 0})["patches"] += 1
        if not r.get("genre"):                      # no evidence: the plugin's usual style is shown separately, never added to ``patches``
            for pg in r.get("genre_prior") or []:
                out.setdefault(pg, {"patches": 0, "songs": 0, "prior": 0})["prior"] += 1
        if (k := song_key(r)):
            songs[g].add(k)
    for g, s in songs.items():
        out[g]["songs"] = len(s)
    return out


def report(root: Path, per_genre: int) -> str:
    """``patches`` counts rows whose genre has evidence (artist list, creator, name, MusicBrainz tags); ``other`` = no genre. ``prior-only`` counts
    untagged presets from a plugin known for that style (``genre_prior``): not evidence, so not part of ``patches`` or ``short by``."""
    lines = [f"{'genre':20s} {'songs':>6s} {'patches':>8s} {'prior-only':>11s} {'short by':>9s}"]
    c_all = counts(root)
    for g, c in sorted(c_all.items(), key=lambda kv: -kv[1]["patches"]):
        lines.append(f"{g:20s} {c['songs']:6d} {c['patches']:8d} {c['prior']:11d} {max(0, per_genre - c['patches']) if g != 'other' else 0:9d}")
    lines.append(f"{'TOTAL':20s} {'':6s} {sum(c['patches'] for c in c_all.values()):8d}")
    return "\n".join(lines)


def cap(root: Path, per_genre: int, out: Path) -> dict[str, int]:
    """Copy at most ``per_genre`` patches per genre, best rating first (unrated last). Returns patches kept per genre."""
    by_genre: dict[str, list[tuple[Path, dict]]] = defaultdict(list)
    for folder, r in _all_rows(Path(root)):
        by_genre[r.get("genre") or "other"].append((folder, r))
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "manifest.jsonl").unlink(missing_ok=True)
    kept: dict[str, int] = {}
    for g, items in by_genre.items():
        items.sort(key=lambda fr: (fr[1].get("rating") is None, -(fr[1].get("rating") or 0), -(fr[1].get("rating_count") or 0)))
        kept[g] = min(per_genre, len(items))
        for folder, r in items[:per_genre]:
            src = folder / r["file"] if r.get("file") else None
            if src is None or not src.exists():                 # metadata-only row (e.g. Line 6 without a login)
                continue
            dst_name = f"{r.get('source', 'src')}__{src.name}"
            shutil.copy2(src, out / dst_name)
            append_row(out / "manifest.jsonl", {**r, "file": dst_name})
    return kept


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("root", nargs="?", default=str(PRESETS))
    r.add_argument("--per-genre", type=int, default=300)
    c = sub.add_parser("cap")
    c.add_argument("root", nargs="?", default=str(PRESETS))
    c.add_argument("--per-genre", type=int, default=300)
    c.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    if a.cmd == "report":
        print(report(Path(a.root), a.per_genre))
    else:
        for g, n in sorted(cap(Path(a.root), a.per_genre, Path(a.out)).items(), key=lambda kv: -kv[1]):
            print(f"{g:20s} {n}")


if __name__ == "__main__":
    main()
