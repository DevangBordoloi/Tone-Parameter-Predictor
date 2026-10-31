"""The presetjunkie check: how does the model's preset compare with what the community made for the same song?

For one song and one plugin it
  1. finds decoded community presets whose song / artist / tags match,
  2. reports the parameter difference against the model's prediction (on knobs audible in both),
  3. renders the prediction and every matching preset on the same held-out DI and scores each against the song stem:
     band-energy MAE (dB), multi-resolution STFT distance, and cosine similarity in the learned tone-embedding space
     (content-invariant, following Chen et al. ISMIR 2024 / AFx-Rep).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from common.dsp import to_mono
from common.schema import PluginSchema
from evaluation.metrics import param_metrics, tone_distance
from tone_model.net import ToneNet


@dataclass
class CheckRow:
    label: str
    source: str                           # "model" | "community"
    band_mae_db: float = float("nan")
    mrstft: float = float("nan")
    embedding_cos: float = float("nan")
    knob_diff: float = float("nan")       # mean |difference| vs the model's prediction (community rows only)
    n_shared: int = 0
    biggest_diffs: list[tuple[str, float]] = field(default_factory=list)


def find_matching_presets(preset_dir: str | Path, plugin: str, song: str | None = None, artist: str | None = None,
                          limit: int = 10) -> list[dict]:
    """Decoded presets (any split) whose metadata mentions the song or artist."""
    d = Path(preset_dir) / plugin
    needles = [s.lower() for s in (song, artist) if s]
    hits = []
    for f in sorted(d.glob("*.json")) if d.is_dir() else []:
        doc = json.loads(f.read_text(encoding="utf-8"))
        meta = doc.get("meta", {})
        hay = " ".join([str(meta.get(k, "")) for k in ("title", "song", "artist")] + list(meta.get("tags") or [])).lower()
        score = sum(n in hay for n in needles)
        if score:
            doc["_id"], doc["_score"] = f.stem, score
            hits.append(doc)
    hits.sort(key=lambda d: -d["_score"])
    return hits[:limit]


def _embed(model: ToneNet, wet: np.ndarray, device: str) -> np.ndarray:
    with torch.no_grad():
        return model.encode(torch.from_numpy(wet[None].astype(np.float32)).to(device))["proj"][0].cpu().numpy()


def run_check(
    model: ToneNet,
    schema: PluginSchema,
    predicted: np.ndarray,                       # (P,) normalized
    stem: np.ndarray,                            # (2, N) or (N,) target stem excerpt at model.cfg.sr
    render: Callable[[np.ndarray], np.ndarray],  # params -> wet (2, N) on the held-out DI at model.cfg.sr
    presets: list[dict],
    device: str = "cpu",
) -> list[CheckRow]:
    sr = model.cfg.sr
    stem2 = stem if stem.ndim == 2 else np.stack([stem, stem])
    n = min(stem2.shape[1], int(model.cfg.clip_sec * sr))
    stem_emb = _embed(model, stem2[:, :n] if n == int(model.cfg.clip_sec * sr) else np.pad(stem2[:, :n], ((0, 0), (0, int(model.cfg.clip_sec * sr) - n))), device)
    names = [p.name for p in schema.parameters]

    def score(label: str, source: str, vec: np.ndarray) -> CheckRow:
        wet = render(vec)
        row = CheckRow(label, source)
        td = tone_distance(stem2, wet, sr)
        row.band_mae_db, row.mrstft = td["band_mae_db"], td["mrstft"]
        row.embedding_cos = float(np.dot(_embed(model, wet, device), stem_emb))
        return row

    rows = [score("model prediction", "model", predicted)]
    defaults = np.array([p.default_normalized for p in schema.parameters], dtype=np.float32)
    for doc in presets:
        vec = defaults.copy()
        for k, v in doc["vst_parameters"].items():
            if k in names:
                vec[names.index(k)] = schema.parameters[names.index(k)].snap(v)
        row = score(doc.get("preset_name") or doc["_id"], "community", vec)
        pm = param_metrics(predicted, vec, schema)
        row.knob_diff = pm["knob_mae"]
        diffs = sorted(((names[i], float(abs(predicted[i] - vec[i]))) for i in range(len(names))
                        if not schema.parameters[i].is_categorical and abs(predicted[i] - vec[i]) > 0.0), key=lambda t: -t[1])
        row.biggest_diffs, row.n_shared = diffs[:5], int(pm["n_numeric"])
        rows.append(row)
    return rows


def format_check(rows: list[CheckRow]) -> str:
    out = ["| preset | source | band MAE (dB) | MR-STFT | embedding cos | knob diff vs model |", "|---|---|---|---|---|---|"]
    for r in rows:
        out.append(f"| {r.label} | {r.source} | {r.band_mae_db:.2f} | {r.mrstft:.3f} | {r.embedding_cos:.3f} | "
                   f"{'' if r.source == 'model' else f'{r.knob_diff:.3f} ({r.n_shared} knobs)'} |")
    return "\n".join(out)
