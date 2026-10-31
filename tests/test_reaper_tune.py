"""REAPER driver, report and tune CLI against a fake bridge (no REAPER, no plugin)."""

from __future__ import annotations

import json

import numpy as np

from common.schema import NeuralDSPPresetJSON, TrackConfig, load_schema
from pipeline import reaper_driver as rd
from pipeline.report import format_report
from pipeline.tune import edit_preset, parse_updates

PLUGIN = "Archetype Plini"


class FakeBridge:
    """Mimics the Lua bridge: tracks, one plugin per track, parameter values by the plugin's host names."""

    def __init__(self):
        self.tracks: dict[str, dict] = {}
        self.calls: list[str] = []

    def call(self, cmd, timeout=None, **a):
        self.calls.append(cmd)
        schema = load_schema(PLUGIN)
        if cmd == "create_track":
            guid = f"{{G{len(self.tracks)}}}"
            self.tracks[guid] = {"name": a["name"], "vals": None}
            return {"guid": guid}
        if cmd == "add_fx":
            self.tracks[a["track"]]["vals"] = {p.host_name or p.name: p.default_normalized for p in schema.parameters}
            return {"slot": 1}
        if cmd == "list_fx":
            t = self.tracks.get(a["track"]) or next(v for v in self.tracks.values() if v["name"] == a["track"])
            return {"fx": [{"slot": 1, "name": f"VST3: {PLUGIN} (Neural DSP)"}]}
        t = self.tracks.get(a["track"]) or next(v for v in self.tracks.values() if v["name"] == a["track"])
        if cmd == "get_params":
            if a.get("names_only"):
                return {"fx": PLUGIN, "params": list(t["vals"])}
            return {"fx": PLUGIN, "params": [{"index": i, "name": k, "value": v, "display": ""} for i, (k, v) in enumerate(t["vals"].items())]}
        if cmd == "set_params":
            names = list(t["vals"])
            for it in a["params"]:
                t["vals"][names[it["index"]]] = it["value"]
            return {"fx": PLUGIN, "results": [{"ok": True} for _ in a["params"]]}
        if cmd == "get_chain":
            return {"chain": dict(t["vals"])}
        if cmd == "set_chain":
            t["vals"] = dict(a["chain"])
            return {}
        return {}


def _preset(double: bool) -> NeuralDSPPresetJSON:
    schema = load_schema(PLUGIN)
    vals = {p.name: p.default_normalized for p in schema.parameters[:10]}
    return NeuralDSPPresetJSON(preset_name="t", target_plugin=f"{PLUGIN}.vst3",
                               track_config=TrackConfig(track_name="t", double_tracked=double, panning=[-100.0, 100.0] if double else [0.0]),
                               vst_parameters=vals)


def _driver(monkeypatch) -> tuple[rd.ReaperDriver, FakeBridge]:
    d = rd.ReaperDriver.__new__(rd.ReaperDriver)
    d.bridge = FakeBridge()
    return d, d.bridge


def test_apply_single_and_double(monkeypatch):
    d, b = _driver(monkeypatch)
    r = d.apply_preset(_preset(False), group="Song")
    assert r["tracks"][0]["applied"] == 10 and not r["tracks"][0]["failed"]
    d2, b2 = _driver(monkeypatch)
    r2 = d2.apply_preset(_preset(True))
    assert [t["track"] for t in r2["tracks"]] == ["t L", "t R"]
    assert list(b2.tracks.values())[1]["vals"] == list(b2.tracks.values())[2]["vals"]


def test_tweak_and_read_state(monkeypatch):
    d, b = _driver(monkeypatch)
    d.apply_preset(_preset(False))
    schema = load_schema(PLUGIN)
    name = next(p.name for p in schema.parameters if not p.is_categorical)
    res = d.tweak("t", {name: 0.8, "nope": 1.0})
    assert res["applied"] == 1 and res["unknown_parameters"] == ["nope"]
    st = d.read_state("t")
    host = schema.by_name()[name].host_name
    assert abs(next(p["value"] for p in st["params"] if p["name"] == host) - schema.by_name()[name].snap(0.8)) < 1e-6


def test_tweak_enum_label(monkeypatch):
    d, b = _driver(monkeypatch)
    d.apply_preset(_preset(False))
    schema = load_schema(PLUGIN)
    enum = next(p for p in schema.parameters if p.valid_values and len(p.valid_values) > 2 and p.name == "amp_type")
    d.tweak("t", {"amp_type": str(enum.valid_values[-1])})
    assert d.read_state("t", name_filter="amp")["params"]


def test_report_and_edit(tmp_path):
    p = tmp_path / "x.json"
    p.write_text(_preset(False).model_dump_json(), encoding="utf-8")
    schema = load_schema(PLUGIN)
    name = next(q.name for q in schema.parameters[:10] if not q.is_categorical)
    preset = edit_preset(p, parse_updates([f"{name}=0.25"]))
    assert preset.vst_parameters[name] == schema.by_name()[name].snap(0.25)
    md = format_report(preset, schema, np.zeros(len(schema.parameters)))
    assert name in md and "| parameter | value |" in md


def test_display_units():
    schema = load_schema(PLUGIN)
    gate = schema.by_name()["gate"]
    assert gate.display(0.0).startswith("-90")

