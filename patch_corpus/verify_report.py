"""Corpus verification table: ``python -m patch_corpus.verify_report [--presets data/data_scraping/presets] [--urls data/data_scraping/urls] [--out report.md]``.

For each source folder (katana, boss_gt, fender_mustang, neuraldsp) it runs ``ingest`` over the files and prints

  * valid presets   files read, patches / presets decoded, files that could not be read (``dedupe`` and the acoustic filter are off here: this
                    counts what the parsers can decode, not what training keeps);
  * coverage        katana / boss_gt: of the controls that apply to a patch (amp knobs; drive / tone / level of an enabled booster or drive; time /
                    feedback / mix of an enabled delay; decay / pre-delay / mix of an enabled reverb) the share that decoded to a value.
                    Modulation rate / depth are not decoded for the Katana by design and are left out.
                    neuraldsp: per plugin, the share of stored numeric values that landed on a parameter of the plugin's schema;
                    plugins with no schema keep only switches and 0-1 values and are listed without a percentage;
  * song <-> YouTube pairs   manifest rows with a ``youtube_url``, split by whether a reference duration (Spotify / iTunes / Deezer) backs the
                    match or the title alone does.
"""

from __future__ import annotations

import argparse
import json
import logging
from collections import Counter, defaultdict
from pathlib import Path

from patch_corpus.ingest import ingest
from patch_corpus.mapping.neural import neural_to_vst, schema_for
from patch_corpus.parsers.neural_xml import parse_neural_xml
from patch_corpus.schema import PatchRecord
from patch_scraper.manifest import read_rows

SOURCES = ("katana", "boss_gt", "fender_mustang", "neuraldsp")


def applicable(r: PatchRecord) -> list[float | None]:
    c = r.chain
    vals: list[float | None] = [getattr(c.amp, k) for k in ("gain", "bass", "mid", "treble", "presence", "master")]
    if c.overdrive.enabled:
        vals += [c.overdrive.drive, c.overdrive.tone, c.overdrive.level]
    if c.delay.enabled:
        vals += [c.delay.time_ms, c.delay.feedback, c.delay.mix]
    if c.reverb.enabled:
        vals += [c.reverb.decay, c.reverb.predelay_ms, c.reverb.mix]
    return vals


def device_coverage(folder: Path) -> dict:
    stats: Counter = Counter()
    recs = list(ingest(folder, dedupe=False, skip_acoustic=False, stats=stats))
    by: dict[str, dict] = defaultdict(lambda: {"valid": 0, "got": 0, "total": 0, "unverified": 0, "devices": Counter()})
    for r in recs:                                    # a folder can hold livesets of other units: ingest routes each to its own parser
        v, b = applicable(r), by[r.source]
        b["valid"] += 1
        b["got"], b["total"] = b["got"] + sum(x is not None for x in v), b["total"] + len(v)
        b["unverified"] += any("UNVERIFIED" in w for w in r.warnings)
        b["devices"][r.device] += 1
    files = sum(1 for ext in ("*.tsl", "*.fuse") for _ in folder.glob(ext))
    return {"files": files, "unreadable_files": stats["unreadable"], "valid": len(recs), "by_source": dict(by)}


def neural_coverage(folder: Path) -> dict:
    """Per plugin, read back from the ``ingest`` records: ``native.raw`` = stored values that landed on a schema parameter, ``native.unmatched`` = the rest."""
    per: dict[tuple, list] = defaultdict(lambda: [0, 0, 0])          # (plugin, has schema, container) -> [presets, mapped, total]
    stats: Counter = Counter()
    files = sum(1 for _ in folder.glob("*.xml"))
    recs = list(ingest(folder, dedupe=False, skip_acoustic=False, stats=stats))
    for r in recs:
        if r.native:
            key = (r.native["plugin"], True, r.native["container"].replace("juce_", ""))
            m, u = len(r.native["raw"]), len(r.native["unmatched"])
        else:
            key, m, u = (r.device, False, "-"), 0, 0
        row = per[key]
        row[0] += 1
        row[1], row[2] = row[1] + m, row[2] + m + u
    return {"files": files, "unreadable_files": stats["unreadable"], "valid": len(recs), "per_plugin": per}


def url_pairs(presets: Path, urls: Path) -> dict[str, dict]:
    out = {}
    for s in SOURCES:
        rows = read_rows(presets / s / "manifest.jsonl") if (presets / s / "manifest.jsonl").exists() else []
        with_song = [r for r in rows if r.get("song")]
        yt = [r for r in with_song if r.get("youtube_url")]
        backed = [r for r in yt if r.get("spotify_id") or r.get("duration_source")]
        songs = {(r["artist"].lower(), r["song"].lower()) for r in yt}
        songs_b = {(r["artist"].lower(), r["song"].lower()) for r in backed}
        out[s] = {"rows": len(rows), "with_song": len(with_song), "url": len(yt), "verified": len(backed), "songs": len(songs), "songs_verified": len(songs_b),
                  "sources": Counter(r.get("duration_source") or ("spotify" if r.get("spotify_id") else "none") for r in yt)}
    unique = {}
    f = urls / "song_urls.jsonl"
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                unique[(r["artist"].lower(), r["song"].lower())] = r
    out["_song_urls"] = Counter(r.get("status") for r in unique.values())
    return out


def build(presets: Path, urls: Path) -> str:
    dev = {s: device_coverage(presets / s) for s in ("katana", "boss_gt", "fender_mustang") if (presets / s).exists()}
    nd = neural_coverage(presets / "neuraldsp") if (presets / "neuraldsp").exists() else None
    pairs = url_pairs(presets, urls)
    L = ["## 1. Valid presets per source", "", "| source | files | unreadable files | valid presets / patches |", "|---|---:|---:|---:|"]
    for s, d in dev.items():
        parts = ", ".join(f"{b['valid']} {src}" for src, b in d["by_source"].items())
        L.append(f"| {s} | {d['files']} | {d['unreadable_files']} | {d['valid']} ({parts}) |")
    if nd:
        L.append(f"| neuraldsp | {nd['files']} | {nd['unreadable_files']} | {nd['valid']} |")
    L += ["", "## 2. Parameter mapping coverage", "", "| source | what is measured | coverage |", "|---|---|---:|"]
    for s, d in dev.items():
        for src, b in d["by_source"].items():
            devices = ", ".join(f"{k}: {n}" for k, n in b["devices"].most_common(4))
            L.append(f"| {s} folder -> {src} ({devices}) | applicable controls decoded; {b['unverified']} of {b['valid']} patches use an UNVERIFIED address "
                     f"| {100 * b['got'] / b['total']:.1f}% |")
    L += ["", "| Neural DSP plugin | container | presets | stored values mapped to the schema |", "|---|---|---:|---:|"]
    if nd:
        for (plugin, has_schema, container), (n, m, t) in sorted(nd["per_plugin"].items(), key=lambda kv: (-kv[1][0], kv[0][0])):
            cov = f"{100 * m / t:.1f}%" if has_schema and t else "no schema (only 0-1 values and switches kept)"
            L.append(f"| {plugin} | {container} | {n} | {cov} |")
        with_schema = [(n, m, t) for (_, hs, _), (n, m, t) in nd["per_plugin"].items() if hs]
        tn, tm, tt = sum(x[0] for x in with_schema), sum(x[1] for x in with_schema), sum(x[2] for x in with_schema)
        L.append(f"| **all with a schema** | | {tn} | {100 * tm / tt:.1f}% |" if tt else "")
        L.append(f"| *without a schema* | | {nd['valid'] - tn} | n/a |")
    L += ["", "## 3. Song <-> YouTube URL pairs", "", "| source | manifest rows | rows with a song | rows with a YouTube URL | of those, duration-checked | distinct songs with URL "
          "| distinct songs, duration-checked |", "|---|---:|---:|---:|---:|---:|---:|"]
    tot = Counter()
    for s in SOURCES:
        p = pairs[s]
        L.append(f"| {s} | {p['rows']} | {p['with_song']} | {p['url']} | {p['verified']} | {p['songs']} | {p['songs_verified']} |")
        tot.update({k: p[k] for k in ("rows", "with_song", "url", "verified")})
    L.append(f"| **total** | {tot['rows']} | {tot['with_song']} | {tot['url']} | {tot['verified']} | | |")
    L += ["", f"`song_urls.jsonl` (one latest row per distinct song): {dict(pairs['_song_urls'])}", ""]
    return "\n".join(x for x in L if x is not None)


def main(argv: list[str] | None = None) -> None:
    logging.getLogger("patch_corpus.ingest").setLevel(logging.ERROR)          # one "skip <file>" line per unreadable liveset is noise here
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--presets", default="data/data_scraping/presets")
    ap.add_argument("--urls", default="data/data_scraping/urls")
    ap.add_argument("--out", help="also write the report to this file")
    a = ap.parse_args(argv)
    text = build(Path(a.presets), Path(a.urls))
    print(text)
    if a.out:
        Path(a.out).write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
