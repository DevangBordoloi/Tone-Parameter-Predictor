"""Line 6 Helix / HX preset (.hlx) reader.

Layout relied on (confirmed against a public .hlx viewer's sample; parameter names are not, see below):

    {"version": ..., "schema": "L6Preset",
     "data": {"meta": {"name": "..."},
              "tone": {"dsp0": {"inputA": {...}, "block0": {"@model": "HD2_AmpPlacaterDirty", "@enabled": true,
                                                              "@position": 1, "Drive": 0.5, ...},
                                "cab0": {"@model": "HD2_Cab4x12UberV30", ...}, "split": {...}, "join": {...}, "outputA": {...}},
                       "dsp1": {...}, "snapshot0": {"@valid": true, "blocks": {"dsp0": {"block0": true, ...}}}, "global": {...}}}}

* any dsp child that is a dict with an ``@model`` and is not an input/output/split/join is a block
* the block category comes from the model ID (``HD2_Amp...``, ``HD2_Dist...``, ``HD2_Delay...``, ``HD2_Reverb...``); the
  per-effect DM4 / DL4 / MM4 / FM4 / M13 families are routed by their remainder
* ``@enabled`` is the stored state; with ``snapshot=N`` the snapshot's per-block flag overrides it when that snapshot is valid
* knob values are taken as stored (Helix stores dial positions as 0-1 floats); the universal mapping decides what to do
  with values outside 0-1

Parameter *names* (Drive, Bass, Mid, Treble, Presence, ChVol, Master, Mix, Decay, Feedback, Time, Speed, Depth ...) are the
ones Helix uses; unknown names are kept verbatim in ``params`` so a mapping can be extended without touching this reader.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SKIP_KEYS = re.compile(r"^(input|output|split|join)", re.I)


@dataclass
class HelixBlock:
    key: str
    dsp: int
    model: str
    category: str
    enabled: bool
    position: int
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class HelixPatch:
    name: str
    device: str
    blocks: list[HelixBlock]
    tempo: float | None = None
    warnings: list[str] = field(default_factory=list)

    def of(self, *categories: str, enabled: bool | None = None) -> list[HelixBlock]:
        return [b for b in self.blocks if b.category in categories and (enabled is None or b.enabled == enabled)]


_CATEGORY_PREFIXES = (
    ("Amp", "amp"), ("Cab", "cab"), ("ImpulseResponse", "ir"), ("Dist", "dist"), ("Delay", "delay"), ("Reverb", "reverb"),
    ("Chorus", "mod"), ("Flanger", "mod"), ("Phaser", "mod"), ("Tremolo", "mod"), ("Rotary", "mod"), ("Vibe", "mod"),
    ("Panner", "mod"), ("EQ", "eq"), ("Compressor", "comp"), ("Gate", "comp"), ("Wah", "wah"), ("Pitch", "pitch"),
    ("Synth", "pitch"), ("Filter", "filter"), ("VolPan", "vol"), ("Looper", "looper"), ("Send", "send"), ("Return", "send"),
)
_FAMILY = {"DM4": "dist", "DL4": "delay", "MM4": "mod", "FM4": "filter", "M13": "mod"}


def categorize(model: str) -> str:
    """Model ID -> amp | cab | ir | dist | delay | reverb | mod | eq | comp | wah | pitch | filter | vol | looper | send | other."""
    rest = model.split("_", 1)[1] if "_" in model else model
    for fam, default in _FAMILY.items():
        if rest.startswith(fam):
            tail = rest[len(fam):]
            if re.search(r"Octav|Pitch|Harmon|Whammy", tail, re.I):
                return "pitch"
            if fam == "M13" and re.search(r"Delay|Echo|Loop", tail, re.I):
                return "delay"
            return default
    for prefix, cat in _CATEGORY_PREFIXES:
        if rest.startswith(prefix):
            return cat
    return "other"


def _snapshot_flags(tone: dict, snapshot: int | None) -> dict[tuple[str, str], bool]:
    if snapshot is None:
        return {}
    snap = tone.get(f"snapshot{snapshot}")
    if not isinstance(snap, dict) or not snap.get("@valid", True):
        return {}
    out = {}
    for dsp, blocks in (snap.get("blocks") or {}).items():
        for k, v in (blocks or {}).items():
            if isinstance(v, bool):
                out[(dsp, k)] = v
    return out


def parse_hlx(path: str | Path, snapshot: int | None = 0) -> HelixPatch:
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    data = doc.get("data", doc)
    tone = data.get("tone")
    if not isinstance(tone, dict):
        raise ValueError(f"{path}: no data.tone object, not a Helix preset")
    meta = data.get("meta", {}) or {}
    flags = _snapshot_flags(tone, snapshot)
    warnings: list[str] = []
    blocks: list[HelixBlock] = []
    for dsp_key in sorted(k for k in tone if re.fullmatch(r"dsp\d+", k)):
        dsp = int(dsp_key[3:])
        for key, node in (tone[dsp_key] or {}).items():
            if not isinstance(node, dict) or "@model" not in node or SKIP_KEYS.match(key):
                continue
            model = str(node["@model"])
            enabled = bool(node.get("@enabled", True))
            if (dsp_key, key) in flags:
                enabled = flags[(dsp_key, key)]
            params = {k: v for k, v in node.items() if not k.startswith("@")}
            blocks.append(HelixBlock(key, dsp, model, categorize(model), enabled, int(node.get("@position", 0)), params))
    blocks.sort(key=lambda b: (b.dsp, b.position))
    if not blocks:
        warnings.append("no effect blocks found")
    tempo = (tone.get("global") or {}).get("@tempo")
    return HelixPatch(
        name=str(meta.get("name") or Path(path).stem), device=str(data.get("device_name") or data.get("device") or "helix"),
        blocks=blocks, tempo=float(tempo) if isinstance(tempo, (int, float)) else None, warnings=warnings)
