"""Human-readable preset report (mode 1: no DAW, no LLM): what the model chose, grouped by module, with display units.

Everything is plain text so you can read it, then edit the preset JSON (or use `python -m pipeline.tune`) and load it into the
plugin / REAPER yourself.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np

from common.schema import NeuralDSPPresetJSON, PluginSchema
from dataset_builder.activity import ActivityModel


def format_report(preset: NeuralDSPPresetJSON, schema: PluginSchema, spread: np.ndarray | None = None) -> str:
    """Markdown: header (plugin, chain summary) + one table per module (knob, value, normalized, +- spread)."""
    act = ActivityModel(schema)
    idx = {p.name: i for i, p in enumerate(schema.parameters)}
    by = schema.by_name()
    groups: dict[str, list[str]] = defaultdict(list)
    for name in preset.vst_parameters:
        if name in by:
            groups[act.module_of(name)].append(name)
    out = [f"# {preset.preset_name}", "", f"**Plugin:** {schema.plugin_name}  ",
           f"**Tracks:** {preset.track_config.track_name}"
           + (f" (double-tracked, pan {preset.track_config.panning})" if preset.track_config.double_tracked else " (single)"), ""]
    for k, v in preset.signal_chain_summary.items():
        out.append(f"- **{k}:** {v}")
    out += ["", "`+-` = disagreement between 4 s windows of the song (high = the model is unsure; tune these by ear).", ""]
    for mod in sorted(groups, key=lambda m: (m != "core", m)):
        out += [f"## {mod.removesuffix('_active').replace('_', ' ') if mod != 'core' else 'global / amp'}", "",
                "| parameter | value | normalized | +- |", "|---|---|---|---|"]
        for name in sorted(groups[mod]):
            p, x = by[name], preset.vst_parameters[name]
            sp = f"{float(spread[idx[name]]):.2f}" if spread is not None and not p.is_categorical else ""
            out.append(f"| {name} | {p.display(x)} | {x:.3f} | {sp} |")
        out.append("")
    return "\n".join(out)


def write_report(preset: NeuralDSPPresetJSON, schema: PluginSchema, path: str | Path, spread: np.ndarray | None = None) -> Path:
    path = Path(path)
    path.write_text(format_report(preset, schema, spread), encoding="utf-8")
    return path

