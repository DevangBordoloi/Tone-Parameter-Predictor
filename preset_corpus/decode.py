"""Turn a downloaded Neural DSP preset file into normalized `vst_parameters`.

The .xml preset format is undocumented (likely a JUCE ValueTree), so this module treats decoding as a spike with ordered
strategies and reports which one worked:

  1. json       the file already is a NeuralDSPPresetJSON / {"vst_parameters": {...}} (our own output)
  2. raw_state  hand the bytes (as-is, then each base64 blob found in the file) to pedalboard's `plugin.raw_state`, the
                VST3 state chunk; if the plugin accepts it, read `parameters[name].raw_value` for every schema name
  3. reaper     load the file through REAPER's plugin and read values back with the bridge (not implemented here: needs the
                outcome of the spike; see `decode_via_reaper`)

A strategy only counts as success if the plugin's parameters actually moved away from the pre-load values, so a plugin that
silently ignores a bad chunk is not mistaken for a decode. `verify_roundtrip` checks the other direction (set -> state ->
decode) and is the unit-test hook the plan asks for.

    python -m preset_corpus.decode "Archetype Plini" path/to/preset.xml        # spike: which strategy works?
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from common.schema import PluginSchema, apply_params, load_schema

log = logging.getLogger(__name__)


@dataclass
class DecodeResult:
    values: dict[str, float]
    method: str
    n_changed: int


class DecodeError(RuntimeError):
    pass


def read_values(plugin: Any, schema: PluginSchema) -> dict[str, float]:
    return {p.name: float(plugin.parameters[p.name].raw_value) for p in schema.parameters if p.name in plugin.parameters}


def _json_values(path: Path) -> dict[str, float] | None:
    try:
        d = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    vp = d.get("vst_parameters") if isinstance(d, dict) else None
    return {k: float(v) for k, v in vp.items()} if isinstance(vp, dict) else None


def candidate_states(raw: bytes) -> list[tuple[str, bytes]]:
    """The byte strings worth offering to `raw_state`: the file itself and every base64 payload inside it."""
    out = [("file bytes", raw)]
    text = raw.decode("utf-8", "ignore")
    for m in re.finditer(r"[A-Za-z0-9+/=\s]{200,}", text):
        blob = re.sub(r"\s+", "", m.group(0))
        try:
            data = base64.b64decode(blob + "=" * (-len(blob) % 4), validate=False)
        except (binascii.Error, ValueError):
            continue
        if len(data) > 64:
            out.append((f"base64 blob @{m.start()}", data))
    return out


def decode_via_raw_state(raw: bytes, plugin: Any, schema: PluginSchema, min_changed: int = 3) -> DecodeResult:
    before = read_values(plugin, schema)
    errors = []
    for label, state in candidate_states(raw):
        try:
            plugin.raw_state = state
        except Exception as e:
            errors.append(f"{label}: {e}")
            continue
        after = read_values(plugin, schema)
        changed = sum(abs(after[k] - before[k]) > 1e-6 for k in after)
        if changed >= min_changed:
            return DecodeResult(after, f"raw_state ({label})", changed)
        errors.append(f"{label}: accepted but only {changed} parameters changed")
        apply_params(plugin, before)  # restore and try the next candidate
    raise DecodeError("raw_state did not decode this file: " + "; ".join(errors[:6]))


def decode_via_reaper(preset_path: Path, plugin_name: str) -> DecodeResult:
    raise NotImplementedError(
        "REAPER fallback: add a track, insert the plugin, load the preset file in the plugin UI / FX preset list and read the "
        "parameters back with the reaper bridge's get_plugin_params. Wire it up once the raw_state spike shows it is needed.")


def decode_preset(preset_path: str | Path, plugin: Any, schema: PluginSchema | str) -> DecodeResult:
    schema = load_schema(schema) if isinstance(schema, str) else schema
    path = Path(preset_path)
    vals = _json_values(path)
    if vals is not None:
        return DecodeResult(vals, "json", len(vals))
    try:
        return decode_via_raw_state(path.read_bytes(), plugin, schema)
    except DecodeError as e:
        log.warning("%s", e)
    return decode_via_reaper(path, schema.plugin_name)


def verify_roundtrip(plugin: Any, schema: PluginSchema, values: dict[str, float], tol: float = 1e-3) -> float:
    """set params -> read raw_state -> reset -> write raw_state back -> read params. Returns the max abs difference."""
    apply_params(plugin, values)
    state = plugin.raw_state
    apply_params(plugin, schema.defaults())
    plugin.raw_state = state
    back = read_values(plugin, schema)
    by = schema.by_name()
    # a real plugin quantises to its own step grid, so allow half a step on top of `tol`
    worst = max(max(0.0, abs(back[k] - v) - (0.5 / (by[k].num_steps - 1) if by[k].num_steps and by[k].num_steps > 1 else 0.0))
                for k, v in values.items() if k in back)
    if worst > tol:
        raise DecodeError(f"state round-trip drifted by {worst:.4f}")
    return worst


def export_corpus(plugin_name: str, raw_dir: Path, out_dir: Path) -> int:
    """Decode every downloaded preset for a plugin into <out_dir>/<plugin>/<id>.json (vst_parameters + meta)."""
    from common.plugin_io import load_plugin

    schema = load_schema(plugin_name)
    plugin = load_plugin(schema.vst3_path or plugin_name)
    dest = out_dir / plugin_name
    dest.mkdir(parents=True, exist_ok=True)
    ok = 0
    for d in sorted((raw_dir / plugin_name).glob("*/")):
        files = [f for f in d.glob("preset.*")]
        if not files:
            continue
        try:
            res = decode_preset(files[0], plugin, schema)
        except Exception as e:
            log.warning("%s: %s", d.name, e)
            continue
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8")) if (d / "meta.json").exists() else {}
        (dest / f"{d.name}.json").write_text(json.dumps({"preset_name": meta.get("title", d.name), "target_plugin": plugin_name,
                                                          "decode_method": res.method, "meta": meta,
                                                          "vst_parameters": res.values}, indent=1), encoding="utf-8")
        ok += 1
    return ok


def main(argv: list[str] | None = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plugin")
    ap.add_argument("preset")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    from common.plugin_io import load_plugin

    schema = load_schema(a.plugin)
    plugin = load_plugin(schema.vst3_path or a.plugin)
    res = decode_preset(a.preset, plugin, schema)
    print(f"decoded with {res.method}; {res.n_changed} parameters changed")
    for k, v in list(res.values.items())[:25]:
        print(f"  {k:32s} {v:.3f}")


if __name__ == "__main__":
    main()
