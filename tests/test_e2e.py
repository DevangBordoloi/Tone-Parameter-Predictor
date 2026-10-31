"""Offline end-to-end: stem file -> plugin choice -> parameters -> preset JSON -> ITO -> community check -> evaluation tiers.

Uses an untrained tiny network and a fake plugin: it proves the plumbing and the file formats, not the accuracy.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import soundfile as sf

from common.dsp import resample
from common.schema import NeuralDSPPresetJSON, load_schema
from dataset_builder.di_corpus import synthetic_bank
from dataset_builder.param_sampler import ParamSampler
from dataset_builder.render_dataset import RenderConfig, render_plugin
from tests.fakes import FakePlugin
from tone_model.checkpoint import build_model, save_checkpoint
from tone_model.net import ToneNetConfig

PLUGINS = ["Archetype Plini", "Archetype Gojira"]


@pytest.fixture(scope="module")
def workspace(tmp_path_factory):
    root = tmp_path_factory.mktemp("e2e")
    bank = synthetic_bank(root / "bank", 6)
    # shards from the fake plugin (train + val so the loaders find both)
    for split in ("train", "val"):
        cfg = RenderConfig(bank_dir=str(bank), out_dir=str(root / "shards"), groups=2, groups_per_shard=2, split=split)
        for i, n in enumerate(PLUGINS):
            s = load_schema(n)
            render_plugin(n, i, cfg, FakePlugin(s), s)
    model, _ = build_model(ToneNetConfig(head_hidden=(64, 32, 16)), PLUGINS)
    ckpt = root / "best.pt"
    save_checkpoint(ckpt, model)
    # a guitar "stem": a DI clip through the fake Plini
    from dataset_builder.di_corpus import ClipBank
    from pipeline.render_utils import held_out_di

    di = held_out_di(bank, 0)
    s = load_schema("Archetype Plini")
    wet = np.concatenate([FakePlugin(s)(np.repeat(di[None], 2, 0), 48_000)] * 2, axis=1)
    stem = root / "stem.wav"
    sf.write(str(stem), wet.T, 48_000, subtype="PCM_24")
    # a decoded "community preset" for the check
    smp = ParamSampler(s, rng=np.random.default_rng(0)).random()
    pdir = root / "presets" / "Archetype Plini"
    pdir.mkdir(parents=True)
    (pdir / "p1.json").write_text(json.dumps({"preset_name": "Cool Song Tone", "meta": {"song": "Cool Song", "artist": "Someone", "author": "a1"},
                                              "vst_parameters": {p.name: float(v) for p, v in zip(s.parameters, smp.values)}}))
    return {"root": root, "bank": bank, "ckpt": ckpt, "stem": stem, "presets": root / "presets", "shards": root / "shards"}


class FakeRenderer:
    def __init__(self, schema, out_sr, **_):
        self.plugin, self.schema, self.out_sr = FakePlugin(schema), schema, out_sr

    def __call__(self, values, di=None):
        from pipeline.render_utils import held_out_di

        from common.schema import apply_params

        apply_params(self.plugin, {p.name: float(v) for p, v in zip(self.schema.parameters, values)})
        wet = self.plugin(np.repeat(held_out_di()[None][:, 24_000:24_000 + 192_000], 2, 0), 48_000)
        return resample(wet, 48_000, self.out_sr)


def test_pipeline_from_stem(workspace, tmp_path, monkeypatch):
    from pipeline import render_utils
    from pipeline.run_pipeline import extract_tone_model

    monkeypatch.setattr(render_utils, "PluginRenderer", FakeRenderer)
    monkeypatch.setattr(render_utils, "held_out_di", lambda *a, **k: np.random.default_rng(0).standard_normal(int(4.5 * 48_000)).astype(np.float32) * 0.1)
    res = extract_tone_model(str(workspace["stem"]), "tight rhythm", "Archetype Plini", None, None, "full", True, None, "Cool Song",
                             workspace["ckpt"], ito=True, run_check=True, preset_dir=workspace["presets"], out_dir=tmp_path,
                             work_dir=tmp_path, installed=PLUGINS)
    assert res["plugin"] == "Archetype Plini"
    preset = NeuralDSPPresetJSON.model_validate_json(open(res["json_path"], encoding="utf-8").read())
    schema = load_schema("Archetype Plini")
    assert preset.target_plugin == "Archetype Plini.vst3" and preset.vst_parameters
    known = schema.by_name()
    for k, v in preset.vst_parameters.items():                       # only real parameter names, on the plugin's step grid
        assert k in known and 0 <= v <= 1 and abs(known[k].snap(v) - v) < 1e-6
    assert "Active modules" in preset.signal_chain_summary and "ITO" in preset.signal_chain_summary
    assert res["ito"]["final"] >= res["ito"]["initial"] - 1e-9       # the search never returns something worse than its start
    sources = [c["source"] for c in res["check"]]
    assert sources[0] == "model" and "community" in sources
    assert "| preset |" in res["check_markdown"]


def test_unavailable_plugin_is_substituted(workspace, tmp_path):
    from pipeline.run_pipeline import extract_tone_model

    res = extract_tone_model(str(workspace["stem"]), "", "Archetype Cory Wong", None, None, "full", True, None, None, workspace["ckpt"],
                             out_dir=tmp_path, work_dir=tmp_path, installed=["Archetype Plini"])
    assert res["plugin"] == "Archetype Plini" or res["selection"]["substituted"]


def test_tier1_runs(workspace):
    from evaluation.evaluate import tier1

    rows = tier1(str(workspace["ckpt"]), str(workspace["shards"]), "val", max_groups=2)
    assert {r["plugin"] for r in rows} >= set(PLUGINS)
    assert all(0 <= r["knob_mae"] <= 1 for r in rows if "knob_mae" in r and np.isfinite(r["knob_mae"]))


def test_parse_songs():
    from evaluation.evaluate import parse_songs

    text = ('Both the guitars are electric guitar\nStupid Song(1 guitar-Rhythm+Lead)\nhttps://www.youtube.com/watch?v=abc&list=RD\n'
            "What I've done(2 guitar- 1 lead and 1 rhythm)\nhttps://www.youtube.com/watch?v=def&list=RD\n"
            'Deva Shree ganesha(2 guitar- 1 lead and 1 rhythm)\nLead Part path in system-"C:\\x\\Lead.mp3"\nRhythm Part path in system-"C:\\x\\Rhythm.mp3"\n'
            "Still Into you(2 guitar- Both rhythm)\nhttps://www.youtube.com/watch?v=ghi\n")
    import tempfile, pathlib

    p = pathlib.Path(tempfile.mkdtemp()) / "Songs.txt"
    p.write_text(text, encoding="utf-8")
    songs = {s["name"]: s for s in parse_songs(p)}
    assert songs["Stupid Song"]["parts"] == ["full"] and songs["What I've done"]["parts"] == ["lead", "rhythm"]
    assert songs["Deva Shree ganesha"]["sources"]["lead"].endswith("Lead.mp3")
    assert songs["Still Into you"]["parts"] == ["rhythm", "rhythm"]
