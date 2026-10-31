"""Offline unit tests (no plugins, no network, no demucs): fakes stand in for the licensed VST3s."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from common.schema import guitar_plugin_names, load_schema, to_array
from dataset_builder.activity import ActivityModel
from dataset_builder.fx_schedule import FxScheduler
from dataset_builder.param_sampler import ParamSampler
from tests.fakes import FakePlugin

PLUGINS = guitar_plugin_names()


@pytest.mark.parametrize("name", PLUGINS)
def test_sampler_masks_follow_switches(name):
    schema = load_schema(name)
    s = ParamSampler(schema, rng=np.random.default_rng(1))
    act = s.activity
    idx = {p.name: i for i, p in enumerate(schema.parameters)}
    for _ in range(40):
        smp = s.random()
        assert smp.values.min() >= 0 and smp.values.max() <= 1
        assert not smp.mask[[idx[n] for n in act.ignored]].any()                     # ignored never learned
        for i, p in enumerate(schema.parameters):                                    # inactive and learnable -> at its default
            if not smp.mask[i] and p.name not in act.ignored:
                assert abs(smp.values[i] - s.defaults[i]) < 1e-6, p.name


def test_plini_amp_channel_gates_knobs():
    schema = load_schema("Archetype Plini")
    act = ActivityModel(schema)
    v = schema.defaults()
    spec = schema.by_name()["amp_type"]
    for cls, active_prefix, other in ((0, "clean_", "lead_"), (2, "lead_", "clean_")):
        v["amp_type"] = spec.class_to_normalized(cls)
        names = [p.name for p in schema.parameters]
        mask = dict(zip(names, act.mask(v)))
        assert mask[f"{active_prefix}gain"] and not mask[f"{other}gain"]


def test_plini_module_switch_gates_delay():
    schema = load_schema("Archetype Plini")
    act = ActivityModel(schema)
    v = schema.defaults()
    names = [p.name for p in schema.parameters]
    by = schema.by_name()
    v["delay_active"], v["active_time"] = by["delay_active"].class_to_normalized(1), by["active_time"].class_to_normalized(1)
    assert dict(zip(names, act.mask(v)))["delay_feedback"]
    v["delay_active"] = by["delay_active"].class_to_normalized(0)
    assert not dict(zip(names, act.mask(v)))["delay_feedback"]
    v["delay_active"], v["active_time"] = by["delay_active"].class_to_normalized(1), by["active_time"].class_to_normalized(0)
    assert not dict(zip(names, act.mask(v)))["delay_feedback"]               # section switch off beats module switch on


def test_nested_modules_resolve_to_most_specific_gate():
    act = ActivityModel(load_schema("Archetype Petrucci X"))
    assert act.module_of("chorus_2_rate") == "chorus_2_active"
    assert act.module_of("chorus_rate") == "chorus_active"


def test_sync_switch_selects_note_or_time():
    schema = load_schema("Archetype Plini")
    act = ActivityModel(schema)
    by = schema.by_name()
    names = [p.name for p in schema.parameters]
    v = schema.defaults()
    v["active_time"] = v["delay_active"] = 1.0
    v["delay_sync"] = by["delay_sync"].class_to_normalized(1)
    m = dict(zip(names, act.mask(v)))
    assert m["delay_sync_note"] and not m["delay_tempo"]
    v["delay_sync"] = by["delay_sync"].class_to_normalized(0)
    m = dict(zip(names, act.mask(v)))
    assert m["delay_tempo"] and not m["delay_sync_note"]


def test_schema_roundtrip_and_categorical_split():
    schema = load_schema("Archetype Nolly X")
    assert any(p.is_categorical for p in schema.parameters) and any(not p.is_categorical for p in schema.parameters)
    sync = schema.by_name()["delay_2_tempo"]               # 1001-option "enum" is regressed, not classified
    assert not sync.is_categorical
    amp = schema.by_name()["amp_type"]
    assert amp.is_categorical and amp.normalized_to_class(amp.class_to_normalized(3)) == 3


def test_fx_scheduler_boosts_weak_modules():
    s = FxScheduler(["a", "b"], smoothing=0.0)
    p = s.update({"a": 0.30, "b": 0.05})
    assert p["a"] > p["b"] and 0.15 <= p["b"] <= p["a"] <= 0.9


# ----------------------------------------------------------------------------------------------------------------
# model pieces
# ----------------------------------------------------------------------------------------------------------------
def test_cqt_peaks_and_harmonic_alignment():
    from tone_model.cqt import HarmonicCQT

    sr = 24_000
    h = HarmonicCQT(sr, hop=256)
    t = torch.arange(sr * 2) / sr
    x = (torch.sin(2 * np.pi * 440 * t) + 0.5 * torch.sin(2 * np.pi * 880 * t))[None, None]
    db = h(x)
    peak = int(db[0, 0].mean(-1).argmax())
    assert abs(h.cqt.freqs[peak] - 440) < 5
    stack = h.stack(db)
    assert stack.shape[1] == 6
    # the fundamental (440 Hz) in the h=1 channel sits on the same row as its 2nd harmonic (880 Hz) in the h=2 channel
    row_f0 = int(stack[0, 1].mean(-1).argmax())
    assert stack[0, 2, row_f0].mean() > stack[0, 2].mean() + 10


def test_modulation_bank_sees_tremolo():
    from tone_model.modulation import GammatoneModulationBank

    sr = 24_000
    m = GammatoneModulationBank(sr)
    t = torch.arange(sr * 4) / sr
    steady = torch.sin(2 * np.pi * 220 * t)
    trem = steady * (1 + 0.8 * torch.sin(2 * np.pi * 5 * t))
    f = m(torch.stack([steady, trem]))[:, :144]
    assert f[1].sum() > 10 * f[0].sum()


def test_info_nce_pulls_positives():
    from tone_model.losses import info_nce

    g = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
    slot = torch.tensor([0, 0, 1, 1] * 2)
    good = torch.nn.functional.normalize(torch.tensor([[1, 0, 0, 0], [1, 0.01, 0, 0], [0, 1, 0, 0], [0, 1, 0.01, 0],
                                                       [0, 0, 1, 0], [0, 0, 1, 0.01], [0, 0, 0, 1], [0, 0.01, 0, 1.0]]).float(), dim=1)
    bad = torch.nn.functional.normalize(torch.randn(8, 4), dim=1)
    assert info_nce(good, g, slot) < info_nce(bad, g, slot)


@pytest.fixture(scope="module")
def tiny_net():
    from tone_model.checkpoint import build_model
    from tone_model.net import ToneNetConfig

    cfg = ToneNetConfig(head_hidden=(64, 32, 16))
    return build_model(cfg, ["Archetype Plini", "Archetype Gojira"])


def test_loss_masking_ignores_inactive_parameters(tiny_net):
    from tone_model.losses import head_losses

    model, schemas = tiny_net
    layout = model.layouts[0]
    b = 3
    h = {"rows": torch.arange(b), "numeric": torch.rand(b, layout.n_numeric), "logits": torch.randn(b, layout.switch_dim)}
    params = torch.rand(b, layout.n_params)
    mask = torch.zeros(b, layout.n_params, dtype=torch.bool)
    r = head_losses(h, layout, params, mask)
    assert r["mse_sum"] == 0 and r["mse_n"] == 0 and r["ce_sum"] == 0
    mask[:, layout.numeric_idx[0]] = True
    r = head_losses(h, layout, params, mask)
    assert r["mse_n"] == b
    # perturbing a masked-out target must not change the loss
    p2 = params.clone()
    p2[:, layout.numeric_idx[5]] += 0.5
    assert torch.isclose(head_losses(h, layout, p2, mask)["mse_sum"], r["mse_sum"])


def test_forward_backward_and_routing(tiny_net):
    from tone_model.losses import LossWeights, total_loss

    model, schemas = tiny_net
    model.train()
    audio = torch.randn(4, 2, 96_000) * 0.1
    pid = torch.tensor([0, 0, 1, 1])
    width = max(len(s.parameters) for s in schemas)
    params = torch.rand(4, width)
    masks = torch.rand(4, width) > 0.5
    out = model(audio, pid, params, tf_prob=1.0)
    loss, logs, _ = total_loss(out, model, params, masks, pid, torch.tensor([0, 0, 0, 0]), torch.tensor([0, 1, 0, 1]), LossWeights())
    loss.backward()
    assert torch.isfinite(loss) and all(np.isfinite(v) for v in logs.values())
    assert model.timbre.multi[0][1].weight.grad is not None                       # gradients reach the shared encoder
    # head 0 must not receive gradient from plugin-1 rows only: its FiLM got some, plugin 1's head too
    assert model.heads[0].out.weight.grad is not None and model.heads[1].out.weight.grad is not None


def test_predict_returns_valid_parameters(tiny_net):
    model, schemas = tiny_net
    schema = schemas[0]
    d = torch.from_numpy(to_array(schema.defaults(), schema))
    res = model.predict(torch.randn(2, 2, 96_000) * 0.1, 0, d)
    assert res["params"].shape == (2, len(schema.parameters))
    assert float(res["params"].min()) >= 0 and float(res["params"].max()) <= 1
    assert np.allclose(float(res["proj"].norm(dim=1).mean()), 1.0, atol=1e-4)


# ----------------------------------------------------------------------------------------------------------------
# plugin selection, decoding, shards
# ----------------------------------------------------------------------------------------------------------------
def test_plugin_fallback_picks_an_installed_plugin():
    from plugin_selector.select_plugin import select_plugin

    trained = ["Archetype Plini", "Archetype Gojira", "Archetype John Mayer X"]
    installed = ["Archetype Plini", "Archetype John Mayer X"]
    assert select_plugin("Archetype Plini", None, trained, installed).plugin == "Archetype Plini"
    s = select_plugin("Archetype Cory Wong", None, trained, installed)
    assert s.plugin == "Archetype John Mayer X" and s.substituted
    assert select_plugin(None, "John Mayer", trained, installed).plugin == "Archetype John Mayer X"
    # Gojira asked for but not installed -> never returned
    assert select_plugin("Archetype Gojira", None, trained, installed, {"Archetype Gojira": .9, "Archetype Plini": .1}).plugin == "Archetype Plini"
    # Prince has no signature plugin: classifier decides among installed
    s = select_plugin(None, "Prince", trained, installed, {"Archetype Plini": .2, "Archetype John Mayer X": .7, "Archetype Gojira": .1})
    assert s.plugin == "Archetype John Mayer X" and "classifier" in s.reason


def test_decode_roundtrip_with_fake_state():
    from preset_corpus.decode import decode_via_raw_state, read_values, verify_roundtrip

    schema = load_schema("Archetype Plini")

    class StatefulFake(FakePlugin):
        @property
        def raw_state(self):
            return np.array([p.raw_value for p in self.parameters.values()], dtype=np.float32).tobytes()

        @raw_state.setter
        def raw_state(self, b):
            for p, v in zip(self.parameters.values(), np.frombuffer(b, dtype=np.float32)):
                p.raw_value = float(v)

    plug = StatefulFake(schema)
    smp = ParamSampler(schema, rng=np.random.default_rng(3)).random()
    vals = {p.name: float(v) for p, v in zip(schema.parameters, smp.values)}
    assert verify_roundtrip(plug, schema, vals) < 1e-3
    from common.schema import apply_params

    apply_params(plug, vals)
    blob = plug.raw_state
    fresh = StatefulFake(schema)
    res = decode_via_raw_state(blob, fresh, schema)
    assert max(abs(res.values[k] - schema.by_name()[k].snap(v)) for k, v in vals.items()) < 1e-3


def test_shard_roundtrip(tmp_path):
    from dataset_builder.shards import Shard, ShardWriter, list_shards

    w = ShardWriter(tmp_path, "Archetype Plini", "train", 0, 24_000, groups_per_shard=2)
    for g in range(3):
        w.add_group(np.random.randn(4, 2, 100).astype(np.float32), np.random.rand(2, 5), np.random.rand(2, 5) > 0.5, (1, 2, 1, 2), (0, 1))
    paths = w.close()
    assert len(paths) == 2 and len(list_shards(tmp_path, "train")) == 2
    s = Shard(paths[0])
    assert s.n_groups == 2 and s.group(1)["slot"].tolist() == [0, 0, 1, 1]
