"""PyTorch / pedalboard hooks for the patch corpus.

* ``PatchChainDataset``   universal chain features (+ known mask, rating weight) for models that condition on, or are pre-trained
                          to predict, the device-agnostic chain
* ``PatchPresetBank``     Neural DSP parameter vectors with known masks, per plugin, ready for ``dataset_builder`` renders
* ``render_translation``  pedalboard render of a translated patch on a DI clip (reuses the project's plugin renderer)

Typical use in the existing rendering pipeline::

    bank = PatchPresetBank("data/presets_decoded", "Archetype Plini", split="train")
    sampler = ParamSampler(schema, presets=bank.presets, rng=rng)            # 50 % of draws jitter a real patch
    # sample.mask is already restricted to the parameters the translation actually determined
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from common.schema import PluginSchema, load_schema
from patch_corpus.ingest import rating_weight, read_records
from patch_corpus.schema import PatchRecord


class PatchChainDataset(Dataset):
    def __init__(self, records: list[PatchRecord] | str | Path, min_rating: float | None = None):
        recs = list(read_records(records)) if isinstance(records, (str, Path)) else list(records)
        self.records = [r for r in recs if min_rating is None or (r.meta.rating or 0) >= min_rating]

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int) -> dict[str, Any]:
        r = self.records[i]
        x, known = r.chain.to_features()
        return {"x": torch.from_numpy(x), "known": torch.from_numpy(known), "weight": torch.tensor(rating_weight(r.meta)),
                "id": r.id, "song": r.meta.song or "", "artist": r.meta.artist or ""}


class PatchPresetBank:
    """Translated patches for one plugin, in the form ``ParamSampler(presets=...)`` expects."""

    def __init__(self, preset_dir: str | Path, plugin: str | PluginSchema, split: str | None = "train"):
        from preset_corpus.splits import load_split, split_of
        import json

        self.schema = load_schema(plugin) if isinstance(plugin, str) else plugin
        d = Path(preset_dir) / self.schema.plugin_name
        docs = []
        for f in sorted(d.glob("*.json")) if d.is_dir() else []:
            doc = json.loads(f.read_text(encoding="utf-8"))
            doc["_id"] = f.stem
            if split is None or split_of(doc.get("meta", {}), f.stem) == split:
                docs.append(doc)
        self.docs = docs
        self.presets = [{**{k: float(v) for k, v in d["vst_parameters"].items()}, "__known__": d.get("known")} for d in docs]
        self.weights = np.array([d.get("meta", {}).get("weight", 1.0) for d in docs], dtype=np.float32)

    def __len__(self) -> int:
        return len(self.docs)


def render_translation(plugin: Any, schema: PluginSchema, values: np.ndarray, di: np.ndarray, sr: int = 48_000) -> np.ndarray:
    """Normalised parameter vector -> wet stereo (2, N) through a loaded pedalboard plugin (or any object with the same API)."""
    from dataset_builder.render_dataset import render_clip

    return render_clip(plugin, schema, di, values, sr)
