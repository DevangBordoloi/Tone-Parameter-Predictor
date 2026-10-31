"""Metrics shared by check.py, evaluate.py and the tests."""

from __future__ import annotations

import numpy as np

from common.dsp import BAND_NAMES, band_energy_db, multires_stft_distance, rms, to_mono
from common.schema import PluginSchema
from dataset_builder.activity import ActivityModel


def param_metrics(pred: np.ndarray, truth: np.ndarray, schema: PluginSchema, mask: np.ndarray | None = None) -> dict[str, float]:
    """Numeric-knob MAE and switch accuracy over the parameters that are audible in the *true* configuration."""
    act = ActivityModel(schema)
    if mask is None:
        mask = act.mask({p.name: float(v) for p, v in zip(schema.parameters, truth)})
    mask = mask & act.learnable()
    cat = np.array([p.is_categorical for p in schema.parameters])
    num_m, cat_m = mask & ~cat, mask & cat
    out = {"n_numeric": float(num_m.sum()), "n_switch": float(cat_m.sum())}
    out["knob_mae"] = float(np.abs(pred - truth)[num_m].mean()) if num_m.any() else float("nan")
    if cat_m.any():
        hits = [schema.parameters[i].normalized_to_class(pred[i]) == schema.parameters[i].normalized_to_class(truth[i])
                for i in np.nonzero(cat_m)[0]]
        out["switch_acc"] = float(np.mean(hits))
    else:
        out["switch_acc"] = float("nan")
    by_mod: dict[str, list[float]] = {}
    for i in np.nonzero(num_m)[0]:
        by_mod.setdefault(act.module_of(schema.parameters[i].name), []).append(abs(float(pred[i] - truth[i])))
    out["module_mae"] = {m: float(np.mean(v)) for m, v in by_mod.items()}
    return out


def band_delta(reference: np.ndarray, rendered: np.ndarray, sr: int) -> tuple[float, dict[str, float]]:
    """Per-band dB error (rendered - reference) over the six guitar bands, level independent. -> (mean |error|, per band)."""
    ref, ren = to_mono(reference), to_mono(rendered)
    for label, y in (("reference", ref), ("rendered", ren)):
        if 20 * np.log10(rms(y) + 1e-12) < -70:
            raise ValueError(f"{label} audio is silent; cannot compare tone")
    a, b = band_energy_db(ref, sr), band_energy_db(ren, sr)
    deltas = {n: b[n] - a[n] for n in BAND_NAMES}
    return float(np.mean(np.abs(list(deltas.values())))), deltas


def tone_distance(reference: np.ndarray, rendered: np.ndarray, sr: int) -> dict[str, float]:
    """Band MAE (dB) and multi-resolution STFT distance between a reference stem and a render (both (C, N) or (N,))."""
    mae, deltas = band_delta(reference, rendered, sr)
    return {"band_mae_db": mae, "mrstft": multires_stft_distance(to_mono(reference), to_mono(rendered)), **{f"band_{k}": v for k, v in deltas.items()}}


def aggregate(rows: list[dict], keys: tuple[str, ...]) -> dict[str, float]:
    out = {}
    for k in keys:
        vals = [r[k] for r in rows if k in r and np.isfinite(r[k])]
        out[k] = float(np.mean(vals)) if vals else float("nan")
    return out
