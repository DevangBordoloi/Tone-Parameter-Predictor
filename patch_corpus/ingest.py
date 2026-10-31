"""Walk a folder of scraped patch files and turn them into universal records and Neural DSP preset documents.

    python -m patch_corpus.ingest data/data_scraping/presets --records data/patches/records.jsonl --presets data/presets_decoded --min-rating 3.5
    python -m patch_corpus.ingest data/data_scraping/presets/boss_gt --records data/patches/boss_gt_test.jsonl    # unseen-hardware set

Folder layout is free: every ``.hlx`` (Line 6 CustomTone), ``.tsl`` (Boss liveset, may hold many patches: Katana -> ``mapping/katana.py``,
GT / ME / GX -> ``mapping/boss_gt.py``, chosen by the file's ``device``), ``.xml`` (Neural DSP preset -> ``mapping/neural.py``) and
``.fuse`` (Fender Mustang -> ``mapping/fender.py``) below
the root is read. Song / artist / rating come from sidecars or a manifest written by the scraper (see ``metadata.py``); a file that
cannot be parsed is logged and skipped, never fatal.

Outputs
  records.jsonl        one ``PatchRecord`` per patch: metadata + universal chain + warnings (device-agnostic, re-translatable)
  <presets>/<plugin>/<id>.json   ``vst_parameters`` + ``known`` + ``meta`` per Neural DSP plugin, i.e. the same decoded-preset
                       documents ``dataset_builder.render_dataset --preset-dir`` consumes; ``meta.split_group`` = artist|song so
                       the author-level split in ``preset_corpus.splits`` never puts one song on both sides
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
from functools import lru_cache
from collections import Counter
from pathlib import Path
from typing import Iterable, Iterator

from common.paths import DATA_DIR
from common.schema import guitar_plugin_names, load_schema
from patch_corpus.mapping.boss_gt import boss_to_universal
from patch_corpus.mapping.fender import fender_to_universal
from patch_corpus.mapping.katana import Addr, katana_to_universal
from patch_corpus.mapping.line6 import helix_to_universal
from patch_corpus.mapping.neural import neural_to_universal, neural_to_vst
from patch_corpus.metadata import find_meta
from patch_corpus.neural_dsp import Translator
from patch_corpus.parsers.boss_tsl import is_boss_liveset, parse_boss_tsl
from patch_corpus.parsers.fender_fuse import parse_fuse
from patch_corpus.parsers.katana_tsl import parse_tsl
from patch_corpus.parsers.line6_hlx import parse_hlx
from patch_corpus.parsers.neural_xml import parse_neural_xml
from patch_corpus.schema import AmpArchetype, PatchMeta, PatchRecord, UniversalChain

log = logging.getLogger(__name__)

SOURCES = {".hlx": "line6_customtone", ".tsl": "boss_katana", ".xml": "neural_dsp", ".fuse": "fender_mustang"}


@lru_cache(maxsize=4096)
def _tsl_is_gt(path: str) -> bool:
    """Katana and GT/ME livesets share the .tsl extension; the JSON's ``device`` / ``patchList`` tell them apart."""
    try:
        return is_boss_liveset(json.loads(Path(path).read_text(encoding="utf-8-sig")))
    except Exception:
        return False


def iter_patch_files(root: str | Path) -> Iterator[Path]:
    root = Path(root)
    files = [root] if root.is_file() else sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in SOURCES)
    yield from (f for f in files if f.suffix.lower() in SOURCES)


def _record_id(path: Path, root: Path, index: int) -> str:
    rel = path.relative_to(root).as_posix() if path.is_relative_to(root) else path.name
    return hashlib.sha1(f"{rel}#{index}".encode()).hexdigest()[:12]


def read_file(path: Path, root: Path, rating_max: float | None = None, snapshot: int | None = 0,
              katana_overrides: dict[str, Addr] | None = None) -> list[PatchRecord]:
    """One patch file -> its records (a .hlx holds one patch, a .tsl up to a liveset's worth)."""
    source = SOURCES[path.suffix.lower()]
    if source == "boss_katana" and _tsl_is_gt(str(path)):
        source = "boss_gt"
    out: list[PatchRecord] = []
    if source == "neural_dsp":
        pre = parse_neural_xml(path)
        params = neural_to_vst(pre)
        chain, warn = neural_to_universal(pre, params=params)
        meta = find_meta(path, pre.name, root, rating_max)
        native = ({"plugin": params.plugin, "plugin_id": pre.plugin, "vst_parameters": params.vst_parameters, "raw": params.raw, "unmatched": params.unmatched,
                   "container": pre.container}
                  if params.plugin else None)
        out.append(PatchRecord(id=_record_id(path, root, 0), source=source, device=pre.plugin, file=str(path), patch_index=0, meta=meta,
                               chain=chain, warnings=warn, native=native))
    elif source == "fender_mustang":
        fp = parse_fuse(path)
        chain, warn = fender_to_universal(fp)
        meta = find_meta(path, fp.name, root, rating_max)
        out.append(PatchRecord(id=_record_id(path, root, 0), source=source, device=fp.amplifier or "Fender Mustang", file=str(path), patch_index=0,
                               meta=meta, chain=chain, warnings=warn))
    elif source == "boss_gt":
        for bp in parse_boss_tsl(path):
            chain, warn = boss_to_universal(bp)
            meta = find_meta(path, bp.name, root, rating_max)
            out.append(PatchRecord(id=_record_id(path, root, bp.index), source=source, device=bp.device, file=str(path),
                                   patch_index=bp.index, meta=meta, chain=chain, warnings=warn))
    elif source == "line6_customtone":
        hx = parse_hlx(path, snapshot)
        chain, warn = helix_to_universal(hx)
        meta = find_meta(path, hx.name, root, rating_max)
        out.append(PatchRecord(id=_record_id(path, root, 0), source=source, device=hx.device, file=str(path), patch_index=0,
                               meta=meta, chain=chain, warnings=warn))
    else:
        for kp in parse_tsl(path):
            chain, warn = katana_to_universal(kp, katana_overrides)
            meta = find_meta(path, kp.name, root, rating_max)
            if kp.memo and meta.song is None:                                  # the patch memo sometimes names the song
                from patch_corpus.metadata import artist_song_from_text

                a, s = artist_song_from_text(kp.memo)
                if s:
                    meta = meta.model_copy(update={"artist": meta.artist or a, "song": s, "meta_origin": meta.meta_origin if meta.meta_origin != "none" else "memo"})
            out.append(PatchRecord(id=_record_id(path, root, kp.index), source=source, device=kp.device, file=str(path),
                                   patch_index=kp.index, meta=meta, chain=chain, warnings=warn))
    return out


def fingerprint(chain: UniversalChain) -> str:
    """Hash of the chain at 2-decimal precision: CustomTone is full of re-uploads of the same tone."""
    x, k = chain.to_features()
    q = [round(float(v), 2) if kn else None for v, kn in zip(x, k)]
    return hashlib.sha1(json.dumps(q).encode()).hexdigest()


def ingest(root: str | Path, *, min_rating: float | None = None, skip_acoustic: bool = True, require_song: bool = False,
           dedupe: bool = True, rating_max: float | None = None, snapshot: int | None = 0,
           katana_overrides: dict[str, Addr] | None = None, stats: Counter | None = None) -> Iterator[PatchRecord]:
    stats = stats if stats is not None else Counter()
    root = Path(root)
    seen: set[str] = set()
    for f in iter_patch_files(root):
        try:
            recs = read_file(f, root, rating_max, snapshot, katana_overrides)
        except Exception as e:
            log.warning("skip %s: %s", f, e)
            stats["unreadable"] += 1
            continue
        for r in recs:
            stats["read"] += 1
            if skip_acoustic and r.chain.amp.archetype == AmpArchetype.ACOUSTIC:
                stats["acoustic"] += 1
            elif require_song and not (r.meta.song or r.meta.artist):
                stats["no_song"] += 1
            elif min_rating is not None and (r.meta.rating is None or r.meta.rating < min_rating):
                stats["below_rating"] += 1
            elif dedupe and (fp := fingerprint(r.chain)) in seen:
                stats["duplicate"] += 1
            else:
                if dedupe:
                    seen.add(fingerprint(r.chain))
                stats["kept"] += 1
                yield r


def rating_weight(meta: PatchMeta) -> float:
    """Training weight in [0.1, 1]: (stars/5)^2 scaled up by how many people agreed (log of the vote count)."""
    if meta.rating is None:
        return 0.3
    votes = math.log1p(meta.rating_count or 0) / math.log1p(50)
    return float(min(1.0, max(0.1, (meta.rating / 5.0) ** 2 * (0.5 + 0.5 * min(1.0, votes)))))


def preset_meta(r: PatchRecord) -> dict:
    m = r.meta
    group = f"{(m.artist or '').lower()}|{(m.song or '').lower()}" if (m.artist or m.song) else None
    return {"title": m.title, "artist": m.artist, "song": m.song, "author": m.author, "rating": m.rating,
            "rating_count": m.rating_count, "tags": m.tags, "source": r.source, "device": r.device, "source_url": m.source_url,
            "split_group": group, "weight": rating_weight(m), "meta_origin": m.meta_origin, "patch_id": r.id}


def write_records(records: Iterable[PatchRecord], path: str | Path) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(p, "w", encoding="utf-8") as f:
        for r in records:
            f.write(r.model_dump_json() + "\n")
            n += 1
    return n


def read_records(path: str | Path) -> Iterator[PatchRecord]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield PatchRecord.model_validate_json(line)


def export_presets(records: Iterable[PatchRecord], out_dir: str | Path, plugins: list[str] | None = None,
                   channel_overrides: dict | None = None) -> Counter:
    """Translate every record to every plugin and write decoded-preset documents. Returns counts per plugin."""
    out = Path(out_dir)
    translators = {n: Translator(load_schema(n), channel_overrides) for n in (plugins or guitar_plugin_names())}
    counts: Counter = Counter()
    for r in records:
        for name, tr in translators.items():
            if r.native and r.native.get("plugin") == name:                # the preset IS this plugin's state: exact labels, no translation
                doc = native_preset_doc(r, tr.schema)
                d = out / name
                d.mkdir(parents=True, exist_ok=True)
                (d / f"{r.id}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
                counts[f"{name} (native)"] += 1
                continue
            t = tr.translate(r.chain)
            if not t.known - set(tr.section_switches):                    # nothing but section switches decided
                counts[f"{name}: empty"] += 1
                continue
            d = out / name
            d.mkdir(parents=True, exist_ok=True)
            doc = t.to_preset_doc(r.meta.title or r.id, preset_meta(r))
            (d / f"{r.id}.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
            counts[name] += 1
    return counts


def native_preset_doc(r: PatchRecord, schema) -> dict:
    """Decoded-preset document for a Neural DSP preset of the same plugin: every parameter it stores is known, the rest stay default."""
    values = schema.defaults()
    values.update({k: v for k, v in r.native["vst_parameters"].items() if k in values})
    return {"preset_name": r.meta.title or r.id, "target_plugin": schema.plugin_name, "decode_method": "native_preset", "meta": preset_meta(r),
            "vst_parameters": values, "known": sorted(k for k in r.native["vst_parameters"] if k in values),
            "notes": [w for w in r.warnings if "not mapped" in w or "out of range" in w]}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", help="folder of scraped .hlx / .tsl / .xml files (or one file)")
    ap.add_argument("--records", default=str(DATA_DIR / "patches" / "records.jsonl"))
    ap.add_argument("--presets", default=str(DATA_DIR / "presets_decoded"), help="output for per-plugin preset documents")
    ap.add_argument("--plugins", nargs="*", help="default: all guitar plugins with a schema")
    ap.add_argument("--min-rating", type=float, help="drop patches rated below this many stars (0-5); unrated patches are dropped too")
    ap.add_argument("--rating-max", type=float, help="scale of the scraped rating (e.g. 10 or 100); default: guessed")
    ap.add_argument("--require-song", action="store_true")
    ap.add_argument("--keep-duplicates", action="store_true")
    ap.add_argument("--keep-acoustic", action="store_true")
    ap.add_argument("--snapshot", type=int, default=0, help="Helix snapshot whose block on/off states to use; -1 = stored states")
    ap.add_argument("--katana-overrides", help="JSON {name: {group, address, verified, wide}} correcting Katana addresses")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    overrides = None
    if a.katana_overrides:
        overrides = {k: Addr(**v) for k, v in json.loads(Path(a.katana_overrides).read_text(encoding="utf-8")).items()}
    stats: Counter = Counter()
    recs = list(ingest(a.root, min_rating=a.min_rating, skip_acoustic=not a.keep_acoustic, require_song=a.require_song,
                       dedupe=not a.keep_duplicates, rating_max=a.rating_max, snapshot=None if a.snapshot < 0 else a.snapshot,
                       katana_overrides=overrides, stats=stats))
    n = write_records(recs, a.records)
    counts = export_presets(recs, a.presets, a.plugins)
    print(f"patches: {dict(stats)}")
    print(f"wrote {n} records -> {a.records}")
    for k, v in sorted(counts.items()):
        print(f"  {k}: {v}")
    with_song = sum(1 for r in recs if r.meta.song)
    print(f"{with_song}/{n} records have a song title; {sum(1 for r in recs if r.warnings)} carry warnings")


if __name__ == "__main__":
    main()
