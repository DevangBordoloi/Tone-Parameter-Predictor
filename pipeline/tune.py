"""Tune a generated preset by hand (no LLM, no MCP): show it, edit values, push it into REAPER, or nudge a live REAPER track.

    python -m pipeline.tune show output/song_lead.json
    python -m pipeline.tune set output/song_lead.json delay_mix=0.3 amp_type=Lead gate=0.4      # edits the JSON in place
    python -m pipeline.tune apply output/song_lead.json [--group "Song name"]                  # create track(s) in REAPER
    python -m pipeline.tune live "Song lead" delay_mix=0.35 reverb_active=1                    # change a track already in REAPER
    python -m pipeline.tune state "Song lead" --filter delay                                    # read knob values back from REAPER

Values are normalized 0-1 (see the report's `normalized` column) or, for enums / switches, the option label.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from common.schema import NeuralDSPPresetJSON, load_schema
from pipeline.reaper_driver import DawError, ReaperDriver
from pipeline.report import format_report


def parse_updates(items: list[str]) -> dict[str, float | str]:
    out: dict[str, float | str] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"expected name=value, got {it!r}")
        k, v = it.split("=", 1)
        try:
            out[k.strip()] = float(v)
        except ValueError:
            out[k.strip()] = v.strip()
    return out


def edit_preset(path: str | Path, updates: dict[str, float | str]) -> NeuralDSPPresetJSON:
    path = Path(path)
    preset = NeuralDSPPresetJSON.model_validate_json(path.read_text(encoding="utf-8"))
    schema = load_schema(preset.target_plugin)
    spec = schema.by_name()
    for k, v in updates.items():
        if k not in spec:
            raise SystemExit(f"{k!r} is not a parameter of {schema.plugin_name}")
        p = spec[k]
        if isinstance(v, str):
            labels = [str(x).lower() for x in (p.valid_values or [])]
            if v.lower() not in labels:
                raise SystemExit(f"{k}: {v!r} is not one of {p.valid_values}")
            v = p.class_to_normalized(labels.index(v.lower()))
        preset.vst_parameters[k] = p.snap(float(v))
    path.write_text(preset.model_dump_json(indent=2), encoding="utf-8")
    md = path.with_suffix(".md")
    if md.exists():
        md.write_text(format_report(preset, schema), encoding="utf-8")
    return preset


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("show"); s.add_argument("preset")
    s = sub.add_parser("set"); s.add_argument("preset"); s.add_argument("updates", nargs="+")
    s = sub.add_parser("apply"); s.add_argument("preset"); s.add_argument("--group")
    s = sub.add_parser("live"); s.add_argument("track"); s.add_argument("updates", nargs="+"); s.add_argument("--plugin")
    s = sub.add_parser("state"); s.add_argument("track"); s.add_argument("--filter"); s.add_argument("--plugin")
    a = ap.parse_args(argv)
    try:
        if a.cmd in ("show", "set"):
            preset = edit_preset(a.preset, parse_updates(a.updates)) if a.cmd == "set" else \
                NeuralDSPPresetJSON.model_validate_json(Path(a.preset).read_text(encoding="utf-8"))
            print(format_report(preset, load_schema(preset.target_plugin)))
        elif a.cmd == "apply":
            print(json.dumps(ReaperDriver().apply_preset_json(a.preset, a.group), indent=2))
        elif a.cmd == "live":
            print(json.dumps(ReaperDriver().tweak(a.track, parse_updates(a.updates), a.plugin), indent=2))
        else:
            st = ReaperDriver().read_state(a.track, a.plugin, a.filter)
            print(st["plugin"])
            for p in st["params"]:
                print(f"  {p['name']:36s} {p['value']:.3f}  {p.get('display', '')}")
    except DawError as e:
        raise SystemExit(f"REAPER: {e}")


if __name__ == "__main__":
    main()

