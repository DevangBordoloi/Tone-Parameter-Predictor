"""Public DI guitar corpora -> a bank of fixed-length mono clips with a fixed train/val/test split by recording.

Datasets (download them yourself; landing pages below - check the licence of each before use):
  IDMT-SMT-Guitar   https://www.idmt.fraunhofer.de/en/publications/datasets/guitar.html  (also on Zenodo)
  EGDB              https://ss12f32v.github.io/Guitar-Transcription/   use the DI folder (audio_DI)
  GuitarSet         https://guitarset.weebly.com/  /  Zenodo 3371780    use the hex pickup (`*_hex_cln.wav`), summed to mono

Your own `Single Coil.wav` is deliberately excluded from the bank: it is for listening tests only.

Every clip is `PREROLL_SEC + CLIP_SEC` long: the renderer feeds the whole clip through the plugin and discards the pre-roll,
so amp/gate/compressor state has settled before the 4 s that are kept.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from common.paths import CLIP_SECONDS, DATA_DIR, SAMPLE_RATE, ensure

log = logging.getLogger(__name__)

PREROLL_SEC = 0.5
SPLITS = {"train": 0.8, "val": 0.1, "test": 0.1}
SILENCE_DBFS = -50.0
AUDIO_EXT = {".wav", ".flac", ".aif", ".aiff", ".ogg"}


@dataclass
class ClipMeta:
    dataset: str
    recording: str
    split: str
    start_sec: float


def split_of(recording_id: str) -> str:
    """Deterministic split by recording so no recording leaks across splits."""
    h = int(hashlib.md5(recording_id.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    acc = 0.0
    for name, frac in SPLITS.items():
        acc += frac
        if h < acc:
            return name
    return "train"


def find_recordings(dataset: str, root: Path) -> list[Path]:
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXT]
    if dataset == "guitarset":
        return sorted(p for p in files if "hex_cln" in p.name)
    if dataset == "egdb":
        di = [p for p in files if "DI" in {part.upper() for part in p.parts} or "audio_di" in str(p).lower()]
        return sorted(di or files)
    return sorted(files)


def load_mono(path: Path, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Read, sum channels (hex pickups -> one signal) and resample to `sr`."""
    import soundfile as sf
    from scipy.signal import resample_poly

    data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
    y = data.sum(axis=1) if data.shape[1] > 2 else data.mean(axis=1)
    peak = float(np.max(np.abs(y))) + 1e-9
    y = y / peak * 0.5 if peak > 1.0 else y            # hex sums can exceed full scale; keep DI level musical otherwise
    if file_sr != sr:
        from math import gcd

        g = gcd(file_sr, sr)
        y = resample_poly(y, sr // g, file_sr // g).astype(np.float32)
    return y


def slice_clips(y: np.ndarray, sr: int, clip_sec: float, hop_sec: float, max_clips: int) -> list[tuple[float, np.ndarray]]:
    n, hop = int(clip_sec * sr), int(hop_sec * sr)
    out = []
    for s in range(0, max(1, len(y) - n + 1), hop):
        c = y[s : s + n]
        if len(c) < n:
            break
        rms_db = 20 * np.log10(float(np.sqrt(np.mean(c**2))) + 1e-12)
        if rms_db < SILENCE_DBFS + 10:  # skip silent / near-silent windows
            continue
        out.append((s / sr, c.astype(np.float32)))
        if len(out) >= max_clips:
            break
    return out


def build_bank(
    roots: dict[str, str | Path],
    out_dir: str | Path | None = None,
    sr: int = SAMPLE_RATE,
    hop_sec: float = 2.0,
    max_clips_per_recording: int = 20,
    max_clips_per_split: dict[str, int] | None = None,
) -> dict[str, Path]:
    """Slice every dataset in `roots` ({name: folder}) into `<out>/<split>.npy` (float16, N x samples) + `<split>.json`."""
    out = ensure(Path(out_dir) if out_dir else DATA_DIR / "di_bank")
    limits = max_clips_per_split or {"train": 6000, "val": 600, "test": 600}
    clip_sec = PREROLL_SEC + CLIP_SECONDS
    clips: dict[str, list[np.ndarray]] = {s: [] for s in SPLITS}
    meta: dict[str, list[dict]] = {s: [] for s in SPLITS}
    for name, root in roots.items():
        recs = find_recordings(name, Path(root))
        log.info("%s: %d recordings under %s", name, len(recs), root)
        for rec in recs:
            rid = f"{name}/{rec.relative_to(root).as_posix()}"
            split = split_of(rid)
            if len(clips[split]) >= limits[split]:
                continue
            try:
                y = load_mono(rec, sr)
            except Exception as e:  # unreadable file: skip, keep going
                log.warning("skip %s: %s", rec, e)
                continue
            for start, c in slice_clips(y, sr, clip_sec, hop_sec, max_clips_per_recording):
                if len(clips[split]) >= limits[split]:
                    break
                clips[split].append(c.astype(np.float16))
                meta[split].append(asdict(ClipMeta(name, rid, split, start)))
    paths = {}
    for split, arrs in clips.items():
        if not arrs:
            continue
        np.save(out / f"{split}.npy", np.stack(arrs))
        (out / f"{split}.json").write_text(json.dumps(meta[split]), encoding="utf-8")
        paths[split] = out / f"{split}.npy"
        log.info("%s: %d clips", split, len(arrs))
    return paths


class ClipBank:
    """Random access to a split of the bank (memory-mapped)."""

    def __init__(self, split: str, bank_dir: str | Path | None = None):
        d = Path(bank_dir) if bank_dir else DATA_DIR / "di_bank"
        if not (d / f"{split}.npy").exists():  # synthetic / small banks only have `train`
            log.warning("no %s split in %s; using train", split, d)
            split = "train"
        self.clips = np.load(d / f"{split}.npy", mmap_mode="r")
        self.meta = json.loads((d / f"{split}.json").read_text(encoding="utf-8"))

    def __len__(self) -> int:
        return len(self.clips)

    def get(self, i: int) -> np.ndarray:
        return np.asarray(self.clips[i], dtype=np.float32)

    def pick_pair(self, rng: np.random.Generator) -> tuple[int, int]:
        """Two clips from different recordings (so a positive pair shares tone but not content)."""
        a = int(rng.integers(len(self)))
        for _ in range(20):
            b = int(rng.integers(len(self)))
            if self.meta[b]["recording"] != self.meta[a]["recording"]:
                return a, b
        return a, (a + 1) % len(self)


def synthetic_bank(out_dir: str | Path, n: int = 64, sr: int = SAMPLE_RATE, seed: int = 0) -> Path:
    """Plucked-string synthetic DI clips (Karplus-Strong). Lets the whole pipeline run before real datasets are downloaded."""
    rng = np.random.default_rng(seed)
    out = ensure(Path(out_dir))
    length = int((PREROLL_SEC + CLIP_SECONDS) * sr)
    clips, meta = [], []
    for k in range(n):
        y = np.zeros(length, dtype=np.float32)
        t = 0
        while t < length - sr // 4:
            f = 82.41 * 2 ** (rng.integers(0, 30) / 12)
            m = min(int(sr * rng.uniform(0.5, 1.5)), length - t)
            tt = np.arange(m) / sr
            note = np.zeros(m, dtype=np.float32)
            for k in range(1, 9):  # decaying harmonic series, higher partials die faster
                note += (np.sin(2 * np.pi * k * f * tt + rng.uniform(0, 6.28)) * np.exp(-tt * (2.5 + 2.0 * k)) / k).astype(np.float32)
            note *= np.minimum(1.0, tt * 2000)  # 0.5 ms attack ramp
            y[t : t + m] += note * rng.uniform(0.15, 0.4)
            t += int(sr * rng.uniform(0.25, 0.7))
        clips.append(y.astype(np.float16))
        meta.append(asdict(ClipMeta("synthetic", f"synthetic/{k}", "train", 0.0)))
    np.save(out / "train.npy", np.stack(clips))
    (out / "train.json").write_text(json.dumps(meta), encoding="utf-8")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Slice DI datasets into a clip bank")
    ap.add_argument("--idmt"), ap.add_argument("--egdb"), ap.add_argument("--guitarset")
    ap.add_argument("--out", default=str(DATA_DIR / "di_bank"))
    ap.add_argument("--synthetic", type=int, default=0, help="write N synthetic clips instead (smoke tests)")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.synthetic:
        print(synthetic_bank(a.out, a.synthetic))
    else:
        roots = {k: v for k, v in (("idmt", a.idmt), ("egdb", a.egdb), ("guitarset", a.guitarset)) if v}
        if not roots:
            ap.error("give at least one of --idmt/--egdb/--guitarset, or --synthetic N")
        print(build_bank(roots, a.out))
