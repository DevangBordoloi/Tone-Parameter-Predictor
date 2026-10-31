"""Sink for everything the community spiders download: validates files, unpacks archives, writes manifests.

    sink = Sink()
    rows = sink.add(data, "Gojira Presets.zip", meta={"source": "reddit", "source_url": url, "title": "..."})

``Sink.add`` takes the raw bytes of one download. A plain ``.xml`` / ``.nam`` / ``.wav`` is validated and stored; a ``.zip`` (or ``.rar``
when ``rarfile`` and an ``unrar`` tool are installed) is unpacked into ``urls/_cache/archives/<sha1>/``, junk is dropped
(``__MACOSX``, ``._*``, ``.DS_Store``, ``.txt``, ``.pdf``, anything that is not a preset), and ONLY the valid files are moved to

    data/data_scraping/presets/neuraldsp/   .xml   (Neural DSP presets)
    data/data_scraping/presets/nam/         .nam   (Neural Amp Modeler) and .wav (impulse responses)

Every stored file gets a row in that folder's ``manifest.jsonl`` (the layout ``patch_corpus/metadata.py`` reads). A download that does
not look like what its extension says (an HTML login page saved as ``.xml``) is refused, and byte-identical files are stored once.
"""

from __future__ import annotations

import hashlib
import io
import json
import logging
import re
import shutil
import zipfile
from dataclasses import dataclass
from pathlib import Path

from common.paths import ensure
from patch_scraper.artists import FamousArtists
from patch_scraper.genre import map_tags
from patch_scraper.manifest import append_row, read_rows
from patch_scraper.paths import ARCHIVE_CACHE, NAM, NEURALDSP

log = logging.getLogger(__name__)

KINDS = {".xml": "neuraldsp_xml", ".nam": "nam_model", ".wav": "impulse_response"}
JUNK_PARTS = ("__macosx", ".ds_store", "thumbs.db")
MAX_MEMBERS = 3000
MAX_UNPACKED = 600 * 1024 * 1024          # zip-bomb guard: total uncompressed bytes of the members we read
MAX_FILE = {".xml": 5 * 1024 * 1024, ".nam": 40 * 1024 * 1024, ".wav": 30 * 1024 * 1024}
MAX_DEPTH = 2                              # archives inside archives


# ---------------------------------------------------------------------------------------------------- validation
def sniff_ext(data: bytes) -> str | None:
    """Best guess at a download's type from its first bytes (used when the URL has no useful extension)."""
    head = data[:8]
    if head.startswith(b"PK\x03\x04"):
        return ".zip"
    if head.startswith(b"Rar!"):
        return ".rar"
    if head.startswith(b"RIFF") and data[8:12] == b"WAVE":
        return ".wav"
    if _is_nam(data):
        return ".nam"
    if _is_neural_xml(data):
        return ".xml"
    return None


def _is_nam(data: bytes) -> bool:
    if data[:1] not in (b"{", b"\xef"):
        return False
    try:
        doc = json.loads(data.decode("utf-8-sig"))
    except Exception:
        return False
    return isinstance(doc, dict) and "architecture" in doc and "weights" in doc


def _is_neural_xml(data: bytes) -> bool:
    """Neural DSP plugins save a JUCE binary value tree under a .xml name; some exports are real XML text."""
    if b"<html" in data[:2048].lower() or b"<!doctype" in data[:2048].lower():
        return False
    if b"PARAM\x00" in data and b"value\x00" in data:
        return True
    if re.match(rb"[A-Za-z0-9_\-. ]{3,64}\x00\x01", data) and re.search(rb"[\x00-\xff]parameters\x00", data):      # newer "-X" plugins: string-valued tree (a child such as 'tags' may precede it)
        return True
    head = data[:512].lstrip(b"\xef\xbb\xbf \r\n\t")
    return head.startswith((b"<?xml", b"<")) and b"PARAM" in data


def valid(data: bytes, ext: str) -> bool:
    if ext == ".nam":
        return _is_nam(data)
    if ext == ".xml":
        return _is_neural_xml(data)
    if ext == ".wav":
        return data[:4] == b"RIFF" and data[8:12] == b"WAVE"
    return False


def sniff_plugin(data: bytes) -> str | None:
    """Plugin id stored at the start of a binary Neural DSP preset, e.g. ``neural_dsp_gojira``."""
    m = re.match(rb"([A-Za-z0-9_\-. ]{3,64})\x00", data)
    return m.group(1).decode("ascii") if m and not data.startswith(b"<") else None


def _safe_stem(name: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(name).stem).strip("._")
    return stem[:60] or "file"


def is_junk(member: str) -> bool:
    low = member.replace("\\", "/").lower()
    parts = [p for p in low.split("/") if p]
    return (not parts or any(p in JUNK_PARTS or p.startswith("._") for p in parts) or low.endswith("/")
            or Path(low).suffix not in KINDS)


# ---------------------------------------------------------------------------------------------------- archives
def _zip_members(data: bytes):
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        infos = zf.infolist()
        if len(infos) > MAX_MEMBERS:
            raise ValueError(f"archive has {len(infos)} members (limit {MAX_MEMBERS})")
        budget = MAX_UNPACKED
        for info in infos:
            if info.is_dir():
                continue
            ext = Path(info.filename).suffix.lower()
            nested = ext in (".zip", ".rar")
            if not nested and is_junk(info.filename):
                continue
            if info.file_size > (MAX_FILE.get(ext, MAX_UNPACKED) if not nested else MAX_UNPACKED):
                log.warning("skipping oversize member %s (%d bytes)", info.filename, info.file_size)
                continue
            budget -= info.file_size
            if budget < 0:
                raise ValueError("archive expands past the unpack limit")
            try:
                yield info.filename, zf.read(info)
            except Exception as e:          # encrypted / corrupt member: skip it, keep the rest
                log.warning("cannot read %s in archive: %s", info.filename, e)


def _rar_members(data: bytes, work: Path):
    try:
        import rarfile                                          # optional; needs an `unrar` / `7z` / `bsdtar` binary
    except ImportError as e:
        raise RuntimeError("rar archive: install `rarfile` and an unrar tool") from e
    tmp = work / "in.rar"
    tmp.write_bytes(data)
    try:
        with rarfile.RarFile(tmp) as rf:
            for info in rf.infolist():
                if info.isdir() or is_junk(info.filename) and Path(info.filename).suffix.lower() not in (".zip", ".rar"):
                    continue
                yield info.filename, rf.read(info)
    except rarfile.Error as e:
        raise RuntimeError(f"rar archive could not be read: {e}") from e


# ---------------------------------------------------------------------------------------------------- the sink
@dataclass
class Sink:
    neuraldsp_dir: Path = NEURALDSP
    nam_dir: Path = NAM
    work_dir: Path = ARCHIVE_CACHE
    famous: FamousArtists | None = None

    def __post_init__(self):
        self.neuraldsp_dir, self.nam_dir = ensure(Path(self.neuraldsp_dir)), ensure(Path(self.nam_dir))
        self.work_dir = Path(self.work_dir)
        self.famous = self.famous or FamousArtists()
        self.hashes = {r.get("sha1") for d in (self.neuraldsp_dir, self.nam_dir) for r in read_rows(d / "manifest.jsonl") if r.get("sha1")}
        self.stats = {"stored": 0, "duplicate": 0, "rejected": 0, "junk": 0, "archives": 0, "archive_failed": 0}

    # -- public -------------------------------------------------------------------------------------------
    def add(self, data: bytes, filename: str, meta: dict | None = None, _depth: int = 0) -> list[dict]:
        """Store one download (file or archive). Returns the manifest rows written; raises nothing for bad input."""
        meta = dict(meta or {})
        ext = Path(filename).suffix.lower()
        if ext not in (*KINDS, ".zip", ".rar") or not valid(data, ext) and ext in KINDS:
            ext = sniff_ext(data) or ext                       # extension missing or wrong (Dropbox / Drive names)
        if ext in (".zip", ".rar"):
            return self._add_archive(data, filename, ext, meta, _depth)
        if ext not in KINDS:
            self.stats["junk"] += 1
            return []
        row = self._store(data, ext, filename, meta)
        return [row] if row else []

    # -- internals ----------------------------------------------------------------------------------------
    def _add_archive(self, data: bytes, filename: str, ext: str, meta: dict, depth: int) -> list[dict]:
        if depth >= MAX_DEPTH:
            self.stats["archive_failed"] += 1
            return []
        digest = hashlib.sha1(data).hexdigest()
        work = ensure(self.work_dir / digest[:12])
        rows: list[dict] = []
        try:
            members = _zip_members(data) if ext == ".zip" else _rar_members(data, work)
            for member, blob in members:
                m_ext = Path(member).suffix.lower()
                if m_ext in (".zip", ".rar"):
                    rows += self.add(blob, Path(member).name, {**meta, "archive": filename}, depth + 1)
                    continue
                staged = work / f"{len(rows):04d}{m_ext}"       # extract into the cache folder first, then move out
                staged.write_bytes(blob)
                row = self._store(staged.read_bytes(), m_ext, Path(member).name, {**meta, "archive": filename, "member": member}, staged)
                if row:
                    rows.append(row)
            self.stats["archives"] += 1
        except (zipfile.BadZipFile, ValueError, RuntimeError, OSError) as e:
            self.stats["archive_failed"] += 1
            log.warning("archive %s rejected: %s", filename, e)
        finally:
            shutil.rmtree(work, ignore_errors=True)             # whatever was not moved out is junk
        return rows

    def _store(self, data: bytes, ext: str, filename: str, meta: dict, staged: Path | None = None) -> dict | None:
        if len(data) > MAX_FILE[ext] or not valid(data, ext):
            self.stats["rejected"] += 1
            log.warning("%s is not a valid %s file (html page, truncated or wrong type); skipped", filename, ext)
            return None
        digest = hashlib.sha1(data).hexdigest()
        if digest in self.hashes:
            self.stats["duplicate"] += 1
            return None
        self.hashes.add(digest)
        dest_dir = self.neuraldsp_dir if ext == ".xml" else self.nam_dir
        name = f"{meta.get('source', 'src')}-{digest[:10]}-{_safe_stem(filename)}{ext}"
        if staged is not None and staged.exists():
            shutil.move(str(staged), dest_dir / name)
        else:
            (dest_dir / name).write_bytes(data)
        row = {"source_id": f"{meta.get('source', 'src')}-{digest[:10]}", "file": name, "sha1": digest, "status": "ok",
               "kind": KINDS[ext], "original_name": filename, "bytes": len(data), **meta}
        if ext == ".xml":
            row["plugin"] = sniff_plugin(data)
        row.update(self.tags(row))
        append_row(dest_dir / "manifest.jsonl", row)
        self.stats["stored"] += 1
        return row

    def tags(self, row: dict) -> dict:
        """artist / genre from the famous-artist list (title and tags), else the genre keywords in the tags; never overwrites a given value."""
        text = " ".join(str(x) for x in (row.get("title"), row.get("original_name"), *(row.get("tags") or [])) if x)
        out: dict = {}
        hit = self.famous.match(text)
        if hit and not row.get("artist"):
            out["artist"], out["famous_artist"] = hit[1], hit[1]
        if not row.get("genre"):
            genre = hit[0] if hit else map_tags([(t, 1) for t in row.get("tags") or []])
            if genre != "other":
                out["genre"] = genre
        return out


# ---------------------------------------------------------------------------------------------------- manual packs
LOCAL_EXTS = (*KINDS, ".zip", ".rar")


def ingest_local(folder: Path, sink: Sink | None = None, *, source: str = "manual", artist: str | None = None, genre: str | None = None,
                 tags: list[str] | None = None) -> dict:
    """Unpack / validate / tag every preset file and archive below ``folder`` (packs downloaded by hand from Mega, MediaFire, Drive ...).

    The title is the file's path relative to ``folder`` (pack and sub-folder names often carry the artist or song), so the famous-artist
    and genre taggers see it. Re-running is safe: byte-identical files are skipped."""
    folder = Path(folder)
    sink = sink or Sink()
    files = [folder] if folder.is_file() else sorted(p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in LOCAL_EXTS)
    stats = {"files": 0, "stored": 0, "unreadable": 0}
    for path in files:
        stats["files"] += 1
        rel = path.relative_to(folder).as_posix() if path != folder else path.name
        meta = {"source": source, "source_url": None, "title": rel.rsplit(".", 1)[0].replace("/", " - "), "local_path": str(path),
                "tags": list(tags or [])}
        if artist:
            meta["artist"] = artist
        if genre:
            meta["genre"] = genre
        try:
            data = path.read_bytes()
        except OSError as e:
            log.warning("cannot read %s: %s", path, e)
            stats["unreadable"] += 1
            continue
        stats["stored"] += len(sink.add(data, path.name, meta))
    return {**stats, **{f"sink_{k}": v for k, v in sink.stats.items()}}


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m patch_scraper.archive")
    sub = ap.add_subparsers(dest="cmd", required=True)
    il = sub.add_parser("ingest-local", help="unpack, validate, tag and log presets you downloaded by hand")
    il.add_argument("folder")
    il.add_argument("--source", default="manual", help="value for the manifest's source field (e.g. mega, mediafire, gumroad)")
    il.add_argument("--artist", help="set the artist for every file (else guessed from file / folder names)")
    il.add_argument("--genre", help="set the genre for every file")
    il.add_argument("--tag", action="append", default=[], help="extra tag (repeatable); tags feed the genre keywords")
    il.add_argument("--neuraldsp-dir", default=str(NEURALDSP))
    il.add_argument("--nam-dir", default=str(NAM))
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    sink = Sink(Path(a.neuraldsp_dir), Path(a.nam_dir))
    print(ingest_local(Path(a.folder), sink, source=a.source, artist=a.artist, genre=a.genre, tags=a.tag))


if __name__ == "__main__":
    main()
