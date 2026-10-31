"""Stage 3: guitar stem -> Neural DSP parameters.

    stem (stereo wav)  ->  focus (Side for double-tracked rhythm, Mid for lead)  ->  4 s windows (50 % overlap)
    -> ToneNet per window  ->  weighted aggregate (sustained, loud windows count most)  ->  snapped to the plugin schema
    -> NeuralDSPPresetJSON

Windows are weighted by their sustain-to-attack energy ratio and level (Stein et al., DAFx 2010), so silence, pick scrape
and a single loud transient do not skew the knob averages. Switch positions are decided on the *averaged probabilities*
(never by averaging class indices).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from common.dsp import resample
from common.paths import OUTPUT_DIR, ensure
from common.schema import NeuralDSPPresetJSON, PluginSchema, TrackConfig, from_array, load_schema, to_array
from dataset_builder.activity import ActivityModel
from guitar_stem_isolator.separator import extract_part_focus
from tone_model.checkpoint import load_checkpoint
from tone_model.dynamics import sustain_weight
from tone_model.net import ToneNet

log = logging.getLogger(__name__)

MIN_WINDOW_DBFS = -55.0


@dataclass
class Prediction:
    plugin: str
    values: np.ndarray                     # (P,) normalized, schema order, snapped
    mask: np.ndarray                       # (P,) bool: parameters that matter for the predicted switch positions
    spread: np.ndarray                     # (P,) weighted std across windows (confidence: lower is better)
    plugin_probs: dict[str, float]
    embedding: np.ndarray                  # (proj_dim,) mean L2-normalised projection of the stem
    window_weights: np.ndarray
    n_windows: int
    focus: str


def load_stem_windows(stem_path: str | Path, focus: str, sr: int, clip_sec: float = 4.0, hop_sec: float = 2.0,
                      max_windows: int = 24) -> np.ndarray:
    """(W, 2, S) windows at the model rate. rhythm -> Side signal, lead -> Mid signal (both fed as dual-mono), full -> stereo."""
    import soundfile as sf

    n = int(clip_sec * sr)
    if focus == "full":
        data, file_sr = sf.read(str(stem_path), dtype="float32", always_2d=True)
        x = data.T if data.shape[1] > 1 else np.repeat(data.T, 2, axis=0)
        x = resample(x[:2], file_sr, sr)
    else:
        mono, file_sr = extract_part_focus(stem_path, focus, gate=(focus == "rhythm"))
        mono = resample(mono, file_sr, sr)
        x = np.stack([mono, mono])
    if x.shape[1] < n:
        x = np.pad(x, ((0, 0), (0, n - x.shape[1])))
    starts = list(range(0, x.shape[1] - n + 1, int(hop_sec * sr)))
    wins = []
    for s in starts:
        w = x[:, s : s + n]
        if 20 * np.log10(np.sqrt(np.mean(w**2)) + 1e-12) > MIN_WINDOW_DBFS:
            wins.append(w)
    if not wins:
        raise ValueError(f"{stem_path}: every window is below {MIN_WINDOW_DBFS} dBFS; is there a guitar in this section?")
    if len(wins) > max_windows:                                      # keep the loudest windows
        order = np.argsort([-np.mean(w**2) for w in wins])[:max_windows]
        wins = [wins[i] for i in sorted(order)]
    return np.stack(wins).astype(np.float32)


def stereo_width(stem_path: str | Path) -> float:
    """Side/Mid RMS ratio of a stereo stem (0 = mono, ~1 = very wide / hard-panned doubles)."""
    import soundfile as sf

    d, _ = sf.read(str(stem_path), dtype="float32", always_2d=True)
    if d.shape[1] < 2:
        return 0.0
    mid, side = (d[:, 0] + d[:, 1]) / 2, (d[:, 0] - d[:, 1]) / 2
    return float(np.sqrt(np.mean(side**2)) / (np.sqrt(np.mean(mid**2)) + 1e-9))


@torch.no_grad()
def embed_windows(model: ToneNet, windows: np.ndarray, device: str, batch: int = 8) -> dict[str, torch.Tensor]:
    model.eval()
    parts: dict[str, list[torch.Tensor]] = {"z": [], "proj": [], "raw_stats": [], "plugin_probs": [], "stats": []}
    for i in range(0, len(windows), batch):
        enc = model.encode(torch.from_numpy(windows[i : i + batch]).to(device))
        parts["z"].append(enc["z"].cpu())
        parts["proj"].append(enc["proj"].cpu())
        parts["raw_stats"].append(enc["raw_stats"].cpu())
        parts["stats"].append(enc["stats"].cpu())
        parts["plugin_probs"].append(torch.softmax(enc["plugin_logits"], -1).cpu())
    return {k: torch.cat(v) for k, v in parts.items()}


def plugin_scores(model: ToneNet, windows: np.ndarray, device: str) -> tuple[dict[str, float], np.ndarray, torch.Tensor]:
    """Weighted mean classifier probabilities over the windows and the mean embedding."""
    e = embed_windows(model, windows, device)
    w = sustain_weight(e["raw_stats"]).clamp(min=1e-3)
    w = w / w.sum()
    probs = (e["plugin_probs"] * w[:, None]).sum(0)
    emb = (e["proj"] * w[:, None]).sum(0)
    emb = emb / emb.norm().clamp(min=1e-8)
    return {n: float(p) for n, p in zip(model.cfg.plugins, probs)}, emb.numpy(), w


@torch.no_grad()
def predict_parameters(model: ToneNet, schema: PluginSchema, windows: np.ndarray, device: str = "cpu", focus: str = "full",
                       batch: int = 8) -> Prediction:
    pid = model.cfg.plugins.index(schema.plugin_name)
    layout = model.layouts[pid]
    e = embed_windows(model, windows, device, batch)
    z, stats = e["z"].to(device), e["stats"].to(device)
    out = model.heads[pid](z, stats)
    numeric, probs = out["numeric"].cpu(), layout.probs(out["logits"]).cpu() if layout.switch_dim else out["logits"].cpu()
    w = sustain_weight(e["raw_stats"]).clamp(min=1e-3)
    w = w / w.sum()
    num_mean = (numeric * w[:, None]).sum(0, keepdim=True)
    prob_mean = (probs * w[:, None]).sum(0, keepdim=True)
    defaults = torch.from_numpy(to_array(schema.defaults(), schema))
    dense = layout.assemble(num_mean, prob_mean, defaults)[0].numpy()
    values = from_array(dense, schema)
    vec = np.array([values[p.name] for p in schema.parameters], dtype=np.float32)

    # spread across windows (numeric knobs only) as a confidence signal
    var = ((numeric - num_mean) ** 2 * w[:, None]).sum(0).sqrt().numpy()
    spread = np.zeros(len(schema.parameters), dtype=np.float32)
    spread[layout.numeric_idx] = var
    act = ActivityModel(schema)
    mask = act.mask({p.name: float(v) for p, v in zip(schema.parameters, vec)}) & act.learnable()
    for n in act.switches():
        if n not in act.ignored:
            mask[[p.name for p in schema.parameters].index(n)] = True
    probs_all, emb, _ = plugin_scores(model, windows, device)
    return Prediction(schema.plugin_name, vec, mask, spread, probs_all, emb, w.numpy(), len(windows), focus)


def describe_chain(schema: PluginSchema, pred: Prediction) -> dict[str, str]:
    """Human-readable summary of the predicted signal chain (amp channel, active modules, uncertain knobs)."""
    act = ActivityModel(schema)
    by = schema.by_name()
    idx = {p.name: i for i, p in enumerate(schema.parameters)}
    values = {p.name: float(pred.values[i]) for i, p in enumerate(schema.parameters)}
    summary: dict[str, str] = {}
    if "amp_type" in by:
        summary["Amp"] = by["amp_type"].label(values["amp_type"]) or ""
    modules = []
    for sw in act.switches():
        spec = by[sw]
        if sw in act.ignored or sw == "amp_type" or sw.startswith("active_") or sw.endswith("_section_active"):
            continue
        if spec.n_options == 2 and spec.normalized_to_class(values[sw]) == 1 and pred.mask[idx[sw]]:
            modules.append(sw.removesuffix("_active"))
    summary["Active modules"] = ", ".join(sorted(modules)) or "none"
    noisy = sorted(((pred.spread[idx[p.name]], p.name) for p in schema.parameters if pred.mask[idx[p.name]]), reverse=True)[:5]
    summary["Least certain knobs"] = ", ".join(f"{n} (±{s:.2f})" for s, n in noisy if s > 0.05) or "none above ±0.05"
    summary["Predictor"] = f"ToneNet, {pred.n_windows} windows, focus={pred.focus}"
    return summary


def to_preset(schema: PluginSchema, pred: Prediction, name: str, focus: str, width: float | None = None,
              extra_summary: dict[str, str] | None = None) -> NeuralDSPPresetJSON:
    """Active parameters (plus every routing switch and pinned value) -> NeuralDSPPresetJSON."""
    act = ActivityModel(schema)
    names = [p.name for p in schema.parameters]
    keep = pred.mask.copy()
    for n, v in act.pinned.items():
        keep[names.index(n)] = True
    params = {n: float(pred.values[i]) for i, n in enumerate(names) if keep[i]}
    double = focus == "rhythm" and (width is None or width > 0.2)
    return NeuralDSPPresetJSON(
        preset_name=name,
        target_plugin=f"{schema.plugin_name}.vst3",
        track_config=TrackConfig(track_name=name, double_tracked=double, panning=[-100.0, 100.0] if double else [0.0]),
        signal_chain_summary={**describe_chain(schema, pred), **(extra_summary or {})},
        vst_parameters=params,
    )


def write_preset(preset: NeuralDSPPresetJSON, out_dir: str | Path | None = None) -> Path:
    out = ensure(Path(out_dir) if out_dir else OUTPUT_DIR)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in preset.preset_name)
    path = out / f"{safe}.json"
    path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")
    return path


class TonePredictor:
    """Loads a checkpoint once and answers many requests."""

    def __init__(self, checkpoint: str | Path, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.schemas, self.ckpt = load_checkpoint(checkpoint, self.device)
        self.schema_by_name = {s.plugin_name: s for s in self.schemas}
        self.sr = self.model.cfg.sr

    def windows(self, stem: str | Path, focus: str) -> np.ndarray:
        return load_stem_windows(stem, focus, self.sr)

    def classify(self, windows: np.ndarray) -> tuple[dict[str, float], np.ndarray]:
        probs, emb, _ = plugin_scores(self.model, windows, self.device)
        return probs, emb

    def predict(self, plugin: str, windows: np.ndarray, focus: str = "full") -> Prediction:
        return predict_parameters(self.model, self.schema_by_name[plugin], windows, self.device, focus)
