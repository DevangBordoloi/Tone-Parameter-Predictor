"""Choose the plugin to build the tone in.

Order of preference:
  1. the plugin the user asked for, if it is installed
  2. the static substitute for a plugin that is not installed (config/artist_plugins.json: Cory Wong -> John Mayer X, ...)
  3. the artist's signature plugin, if installed (Mayer -> John Mayer X; Prince -> none, so fall through)
  4. the installed plugin the model thinks the stem sounds most like: the plugin-classifier probability averaged over the
     stem windows, with the nearest per-plugin embedding centroid (cosine, Chen et al. ISMIR 2024) as the tie-breaker /
     second opinion when no classifier is available

"Installed" means a .vst3 exists on this machine AND the model was trained for it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common.paths import CONFIG_DIR
from common.plugin_io import installed_plugins
from common.schema import norm_key

CONFIG_PATH = CONFIG_DIR / "artist_plugins.json"


@dataclass
class Selection:
    plugin: str
    reason: str
    scores: dict[str, float] = field(default_factory=dict)
    requested: str | None = None
    substituted: bool = False


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8")) if CONFIG_PATH.exists() else {"artists": {}, "not_installed_fallback": {}}


def _match(name: str | None, candidates: list[str]) -> str | None:
    if not name:
        return None
    want = norm_key(name)
    for c in candidates:
        if norm_key(c) == want:
            return c
    hits = [c for c in candidates if want in norm_key(c)]
    return hits[0] if len(hits) == 1 else None


def compute_centroids(model, loader, device: str = "cpu", max_batches: int | None = None) -> dict[str, np.ndarray]:
    """Mean L2-normalised projection of the training renders per plugin (the "nearest centroid" reference set)."""
    sums = {i: torch.zeros(model.cfg.proj_dim) for i in range(len(model.layouts))}
    counts = {i: 0 for i in sums}
    model.eval()
    with torch.no_grad():
        for b, batch in enumerate(loader):
            if max_batches and b >= max_batches:
                break
            enc = model.encode(batch["audio"].to(device))
            for pid in batch["plugin"].unique().tolist():
                rows = (batch["plugin"] == pid).nonzero(as_tuple=True)[0]
                sums[pid] += enc["proj"][rows.to(device)].sum(0).cpu()
                counts[pid] += len(rows)
    out = {}
    for i, name in enumerate(model.cfg.plugins):
        if counts[i]:
            c = sums[i] / counts[i]
            out[name] = (c / c.norm().clamp(min=1e-8)).numpy()
    return out


def save_centroids(path: str | Path, centroids: dict[str, np.ndarray]) -> None:
    np.savez(path, **{k: v for k, v in centroids.items()})


def load_centroids(path: str | Path) -> dict[str, np.ndarray]:
    p = Path(path)
    if not p.exists():
        return {}
    z = np.load(p)
    return {k: z[k] for k in z.files}


def select_plugin(
    requested: str | None = None,
    artist: str | None = None,
    trained: list[str] | None = None,
    installed: list[str] | None = None,
    plugin_probs: dict[str, float] | None = None,
    embedding: np.ndarray | None = None,
    centroids: dict[str, np.ndarray] | None = None,
) -> Selection:
    """`trained`: plugins the checkpoint has heads for; `installed` defaults to those with a .vst3 on this machine."""
    cfg = _config()
    trained = trained or []
    available = [p for p in trained if p in (installed if installed is not None else installed_plugins(trained))]
    if not available:
        raise RuntimeError("no plugin is both installed and covered by the model")

    if requested:
        hit = _match(requested, available)
        if hit:
            return Selection(hit, "requested plugin is installed", requested=requested)
        sub = _match(cfg.get("not_installed_fallback", {}).get(requested) or next(
            (v for k, v in cfg.get("not_installed_fallback", {}).items() if norm_key(k) == norm_key(requested)), None), available)
        if sub:
            return Selection(sub, f"{requested} is not available; static substitute", requested=requested, substituted=True)

    if artist:
        low = artist.lower()
        for key, plug in sorted(cfg.get("artists", {}).items(), key=lambda kv: -len(kv[0])):
            if key in low:
                hit = _match(plug, available)
                if hit:
                    return Selection(hit, f"artist '{key}' signature plugin", requested=requested)
                break

    scores: dict[str, float] = {}
    if plugin_probs:
        scores = {p: float(plugin_probs.get(p, 0.0)) for p in available}
        best = max(scores, key=scores.get)
        reason = "highest classifier probability among installed plugins"
        if embedding is not None and centroids:
            cos = {p: float(np.dot(embedding, centroids[p])) for p in available if p in centroids}
            if cos and max(cos, key=cos.get) != best and scores[best] < 0.5:
                best, reason = max(cos, key=cos.get), "nearest embedding centroid (classifier was not confident)"
                scores = cos
        return Selection(best, reason, scores, requested, substituted=bool(requested))
    if embedding is not None and centroids:
        cos = {p: float(np.dot(embedding, centroids[p])) for p in available if p in centroids}
        if cos:
            return Selection(max(cos, key=cos.get), "nearest embedding centroid", cos, requested, substituted=bool(requested))
    return Selection(available[0], "default (no audio evidence supplied)", {}, requested, substituted=bool(requested))
