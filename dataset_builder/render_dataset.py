"""Render DI clips through the real Neural DSP plugins into training shards.

Run locally: the plugins are licensed to this machine. One worker process per plugin loads the plugin once (~20 s) and
renders groups of four (see shards.py). The wet audio is stored as stereo float16 so the model can compute Mid/Side and
its CQT on the GPU; no spectrogram is baked into the data.

    python -m dataset_builder.render_dataset --bank data/di_bank --out data/shards --groups 5000 --split train
"""

from __future__ import annotations

import argparse
import json
import logging
import multiprocessing as mp
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from tqdm import tqdm

from common.dsp import kmeans, mfcc, resample, rms, sanitize_audio, to_mono
from common.paths import DATA_DIR, SAMPLE_RATE
from common.schema import PluginSchema, apply_params, guitar_plugin_names, load_schema
from dataset_builder.augment import augment
from dataset_builder.di_corpus import PREROLL_SEC, ClipBank
from dataset_builder.param_sampler import ParamSampler, Sample
from dataset_builder.shards import ShardWriter

log = logging.getLogger(__name__)

MIN_RENDER_DBFS = -70.0


@dataclass
class RenderConfig:
    bank_dir: str = str(DATA_DIR / "di_bank")
    out_dir: str = str(DATA_DIR / "shards")
    split: str = "train"
    groups: int = 5000                # 4 renders each -> 20k clips per plugin by default
    store_sr: int = 24_000            # the model front-end works below 12 kHz; 24 kHz halves the disk footprint
    seed: int = 0
    preset_dir: str | None = None     # decoded presetjunkie presets: <preset_dir>/<plugin>/*.json
    fx_probs_path: str | None = None  # {plugin: {module: p_on}} written by the trainer (dynamic Fx scheduling)
    diversity_probe: int = 0          # >0: draw this many configs, cluster by MFCC, sample clusters uniformly (ST-ITO)
    diversity_k: int = 10
    augment_p: dict[str, float] = field(default_factory=dict)
    groups_per_shard: int = 500
    vst3_path: dict[str, str] = field(default_factory=dict)
    mode: str = "mixed"               # mixed: 50 % jittered presets / 50 % random | synthetic: random only (source 0, pre-train)
    #                                   | scraped: every train-split preset once per replica (source 1, fine-tune; `groups` ignored)
    sampling: str = "lhs"             # synthetic mode: "lhs" Latin Hypercube over the active knobs | "uniform" iid
    preset_replicas: int = 1          # scraped mode: renders per preset (each with fresh jitter, DI clips and partner preset)
    song_only: int = 0                # scraped mode: 1 = keep only presets whose meta carries a song or artist (fine-tune for song/artist tone)
    shard_offset: int = 0             # first shard number; use a new offset and --seed per round to add data without overwriting

MODES = ("mixed", "synthetic", "scraped")


# ---------------------------------------------------------------------------------------------------------------
# plugin access
# ---------------------------------------------------------------------------------------------------------------
def load_plugin(schema: PluginSchema, override: str | None = None) -> Any:
    import pedalboard

    path = override or schema.vst3_path
    if not Path(path).exists():
        from common.plugin_io import find_vst3

        path = str(find_vst3(schema.plugin_name))
    return pedalboard.load_plugin(path)


def render_clip(plugin: Any, schema: PluginSchema, di: np.ndarray, values: np.ndarray, sr: int = SAMPLE_RATE) -> np.ndarray:
    """DI (N,) mono at `sr` + normalized parameter vector -> wet (2, N) float32."""
    apply_params(plugin, {p.name: float(v) for p, v in zip(schema.parameters, values)})
    x = np.repeat(di[None].astype(np.float32), 2, axis=0)
    out = np.asarray(plugin(x, sr, reset=True), dtype=np.float32)
    if out.shape[0] == 1:
        out = np.repeat(out, 2, axis=0)
    return out[:2]


def _finish(wet: np.ndarray, sr: int, store_sr: int, rng: np.random.Generator, aug: dict[str, float]) -> np.ndarray:
    pre = int(PREROLL_SEC * sr)
    y = wet[:, pre:]                                  # drop the settle-in pre-roll
    y = resample(y, sr, store_sr)
    return augment(y, store_sr, rng, aug)


def _valid(wet: np.ndarray) -> bool:
    return bool(np.isfinite(wet).all()) and 20 * np.log10(rms(wet) + 1e-12) > MIN_RENDER_DBFS


# ---------------------------------------------------------------------------------------------------------------
# perceptually balanced configuration choice (ST-ITO: k-means over MFCC of a probe render)
# ---------------------------------------------------------------------------------------------------------------
def balanced_samples(
    sampler: ParamSampler,
    probe: Callable[[Sample], np.ndarray | None],
    n_probe: int,
    k: int,
    rng: np.random.Generator,
) -> list[Sample]:
    """Draw `n_probe` configs, keep the renderable ones, cluster their MFCC means and return them interleaved by cluster
    so that consuming the list in order covers the perceptual clusters uniformly."""
    samples, feats = [], []
    for _ in range(n_probe):
        s = sampler.sample()
        wet = probe(s)
        if wet is None:
            continue
        m = mfcc(to_mono(wet), SAMPLE_RATE, 13)
        feats.append(np.concatenate([m.mean(1), m.std(1)]))
        samples.append(s)
    if len(samples) <= k:
        return samples
    ids = kmeans(np.stack(feats), k, rng)
    buckets = [[s for s, i in zip(samples, ids) if i == c] for c in range(k)]
    out: list[Sample] = []
    while any(buckets):
        for b in buckets:
            if b:
                out.append(b.pop(int(rng.integers(len(b)))))
    return out


# ---------------------------------------------------------------------------------------------------------------
# per-plugin worker
# ---------------------------------------------------------------------------------------------------------------
def has_song(meta: dict | None) -> bool:
    m = meta or {}
    return bool(str(m.get("song") or "").strip() or str(m.get("artist") or "").strip())


def load_presets(preset_dir: str | None, plugin_name: str, split: str = "train", heldout: bool = False,
                 song_only: bool = False) -> list[dict[str, float]]:
    """Community presets used as a prior. Only the author-level `train` split is used for training renders, so the val/test
    presets stay unseen for the Tier-2 evaluation. `heldout=True` (scraped mode) returns `split`'s own presets, so a
    scraped *val* shard set can be rendered for validation; nothing from it is ever trained on."""
    if not preset_dir or (split != "train" and not heldout):
        return []
    from preset_corpus.splits import load_split

    out = []
    for doc in load_split(preset_dir, plugin_name, split):
        if song_only and not has_song(doc.get("meta")):
            continue
        try:
            vals = {k: float(v) for k, v in doc["vst_parameters"].items()}
            vals["__info__"] = {"id": doc.get("_id"), "preset_name": doc.get("preset_name"), "meta": doc.get("meta", {})}
            if doc.get("known") is not None:               # patch translations: only some parameters are determined
                vals["__known__"] = list(doc["known"])
            out.append(vals)
        except Exception as e:
            log.warning("bad preset %s: %s", doc.get("_id"), e)
    return out


def render_plugin(plugin_name: str, plugin_id: int, cfg: RenderConfig, plugin: Any | None = None, schema: PluginSchema | None = None) -> list[Path]:
    """Render `cfg.groups` groups for one plugin. `plugin`/`schema` can be injected (tests use a fake plugin)."""
    schema = schema or load_schema(plugin_name)
    rng = np.random.default_rng(cfg.seed + 1000 * plugin_id)
    plugin = plugin or load_plugin(schema, cfg.vst3_path.get(plugin_name))
    bank = ClipBank(cfg.split, cfg.bank_dir)
    fx = {}
    if cfg.fx_probs_path and Path(cfg.fx_probs_path).exists():
        fx = json.loads(Path(cfg.fx_probs_path).read_text(encoding="utf-8")).get(plugin_name, {})
    presets = [] if cfg.mode == "synthetic" else load_presets(cfg.preset_dir, plugin_name, cfg.split, cfg.mode == "scraped", bool(cfg.song_only))
    sampler = ParamSampler(schema, presets, rng, fx_probs=fx)
    writer = ShardWriter(cfg.out_dir, plugin_name, cfg.split, plugin_id, cfg.store_sr, cfg.groups_per_shard, cfg.shard_offset)

    pool: list[Sample] = []
    n_groups = cfg.groups
    if cfg.mode == "synthetic":
        pool = sampler.synthetic_batch(2 * cfg.groups, cfg.sampling)       # two configs per group
        if cfg.diversity_probe:
            log.warning("%s: --diversity-probe is ignored in synthetic mode (stratified sampling already spreads the knobs)", plugin_name)
    elif cfg.mode == "scraped":
        if not presets:
            log.warning("%s: no '%s'-split presets in %s: nothing to render", plugin_name, cfg.split, cfg.preset_dir)
            return []
        # the human presets are the labels: rendered exactly as scraped (no knob jitter, no switch flips)
        pool = [sampler.from_preset(p, jitter=0.0, flip_p=0.0) for p in presets for _ in range(cfg.preset_replicas)]
        rng.shuffle(pool)
        n_groups = (len(pool) + 1) // 2                                    # every preset appears; no cap
        log.info("%s: %d presets x %d replica(s) -> %d groups", plugin_name, len(presets), cfg.preset_replicas, n_groups)
    elif cfg.diversity_probe:
        def probe(s: Sample) -> np.ndarray | None:
            a, _ = bank.pick_pair(rng)
            wet = render_clip(plugin, schema, bank.get(a), s.values)
            return wet[:, int(PREROLL_SEC * SAMPLE_RATE):] if _valid(wet) else None

        pool = balanced_samples(sampler, probe, cfg.diversity_probe, cfg.diversity_k, rng)
        log.info("%s: diversity probe kept %d / %d configs", plugin_name, len(pool), cfg.diversity_probe)

    def next_sample() -> Sample:
        if pool:
            return pool.pop(0)
        if cfg.mode == "scraped":                                          # only reached to pair the odd last preset
            return sampler.from_preset(presets[int(rng.integers(len(presets)))], jitter=0.0, flip_p=0.0)
        return sampler.sample()

    t0, done, n_guarded = time.time(), 0, 0
    bar = tqdm(total=n_groups, desc=f"{plugin_name} [{cfg.split}]", unit="grp", dynamic_ncols=True, position=plugin_id)
    while done < n_groups:
        if cfg.mode == "scraped" and not pool:
            break                                                          # presets exhausted (some were dropped as unrenderable)
        t1, t2 = next_sample(), next_sample()
        a, b = bank.pick_pair(rng)
        dis = (bank.get(a), bank.get(b), bank.get(a), bank.get(b))
        vals = (t1.values, t1.values, t2.values, t2.values)
        wets = [render_clip(plugin, schema, di, v) for di, v in zip(dis, vals)]
        if not all(_valid(w) for w in wets):
            continue                                                  # silent / NaN render: resample both configs
        audio = np.stack([_finish(w, SAMPLE_RATE, cfg.store_sr, rng, cfg.augment_p) for w in wets])
        audio, flagged = sanitize_audio(audio)                           # exploded render (peak > 2 / rms > 1): peak-normalise
        n_guarded += int(flagged.sum())
        writer.add_group(
            audio, np.stack([t1.values, t2.values]), np.stack([t1.mask, t2.mask]),
            (a, b, a, b), (int(t1.source == "preset"), int(t2.source == "preset")), (t1.info, t2.info),
        )
        done += 1
        bar.update(1)
        if done % 50 == 0:
            log.info("%s: %d/%d groups (%.1f s/group)", plugin_name, done, n_groups, (time.time() - t0) / done)
    bar.close()
    if n_guarded:
        log.warning("%s: %d / %d rendered clips exceeded the amplitude limits and were peak-normalised", plugin_name, n_guarded, 4 * done)
    return writer.close()


def _worker(args: tuple[str, int, RenderConfig]) -> tuple[str, int]:
    name, pid, cfg = args
    logging.basicConfig(level=logging.INFO, format=f"[{name}] %(message)s")
    paths = render_plugin(name, pid, cfg)
    return name, len(paths)


def estimate_size_gb(groups: int, store_sr: int, n_plugins: int) -> float:
    return groups * 4 * 2 * 4 * store_sr * 2 * n_plugins / 1e9


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plugins", nargs="*", help="plugin names (default: all guitar plugins with a schema)")
    ap.add_argument("--include-vocal", action="store_true", help="also render Mantra (a vocal processor)")
    ap.add_argument("--workers", type=int, default=2)
    for f, default in RenderConfig.__dataclass_fields__.items():
        if f in ("augment_p", "vst3_path"):
            continue
        d = default.default
        kind = type(d) if d is not None and not isinstance(d, str) else str
        ap.add_argument(f"--{f.replace('_', '-')}", type=kind, default=d)
    a = ap.parse_args(argv)
    if a.mode not in MODES or a.sampling not in ("lhs", "uniform"):
        ap.error(f"--mode must be one of {MODES} and --sampling one of lhs|uniform")
    cfg = RenderConfig(**{f: getattr(a, f) for f in RenderConfig.__dataclass_fields__ if f not in ("augment_p", "vst3_path")})
    names = a.plugins or guitar_plugin_names(a.include_vocal)
    print(f"{len(names)} plugins x {cfg.groups} groups x 4 clips ~ {estimate_size_gb(cfg.groups, cfg.store_sr, len(names)):.1f} GB (before gzip)")
    ids = {n: i for i, n in enumerate(guitar_plugin_names(a.include_vocal))}  # stable plugin ids across splits
    jobs = [(n, ids[n], cfg) for n in names]
    ctx = mp.get_context("spawn")
    with ctx.Pool(min(a.workers, len(jobs)), maxtasksperchild=1) as pool:  # fresh process per plugin: VST3 hosts leak memory
        for name, n in pool.imap_unordered(_worker, jobs):
            print(f"{name}: wrote {n} shard(s)")


if __name__ == "__main__":
    main()
