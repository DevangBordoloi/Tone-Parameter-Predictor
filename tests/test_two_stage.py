"""Smaller heads + dropout, the amplitude guard, synthetic / scraped shard modes, source filter, pre-train -> fine-tune."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from common.dsp import sanitize_audio
from common.schema import load_schema
from dataset_builder.di_corpus import synthetic_bank
from dataset_builder.param_sampler import ParamSampler
from dataset_builder.render_dataset import RenderConfig, render_plugin
from dataset_builder.shards import Shard, ShardWriter, list_shards
from tests.fakes import FakePlugin
from tone_model.checkpoint import build_model, load_pretrained, save_checkpoint
from tone_model.net import ToneNetConfig
from training.data import GroupDataset

PLUGINS = ["Archetype Plini", "Archetype Gojira"]


def test_sanitize_audio_only_touches_exploded_rows():
    rng = np.random.default_rng(0)
    ok = rng.normal(0, 0.05, (2, 4000)).astype(np.float32)
    ok[0, 0] = 1.5                                                    # hot but legitimate (peak <= 2): left alone
    loud = rng.normal(0, 13.0, (2, 4000)).astype(np.float32)          # the Abasi failure: rms ~ 13
    bad = ok.copy()
    bad[0, 5] = np.nan
    out, flagged = sanitize_audio(np.stack([ok, loud, bad]))
    assert flagged.tolist() == [False, True, False]
    assert np.array_equal(out[0], ok)
    assert np.abs(out[1]).max() == pytest.approx(1.0) and np.sqrt((out[1] ** 2).mean()) <= 1.0
    assert np.isfinite(out[2]).all() and out[2][0, 5] == 0.0
    assert np.corrcoef(out[1].ravel(), loud.ravel())[0, 1] == pytest.approx(1.0)       # scaled, not distorted


def test_head_is_small_and_has_dropout():
    model, _ = build_model(ToneNetConfig(), PLUGINS)
    assert model.cfg.head_hidden == (512, 256) and model.cfg.switch_hidden == 256 and model.cfg.head_dropout == 0.25
    n = model.n_params()
    assert n["heads"] / len(PLUGINS) < 1_000_000                                   # was ~4.9M per head
    head = model.heads[0].eval()
    z, st = torch.randn(8, 512), torch.randn(8, 8)
    a, b = head(z, st)["numeric"], head(z, st)["numeric"]
    assert torch.equal(a, b)                                                        # eval: deterministic
    head.train()
    assert not torch.equal(head(z, st)["numeric"], head(z, st)["numeric"])         # train: dropout active
    assert sum(isinstance(m, torch.nn.Dropout) for m in head.stage1) == 1


def test_modulation_survives_bfloat16_autocast():
    from tone_model.modulation import LFONet

    net = LFONet(1, frame_rate=93.75, n_frames=376)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        feats, traj = net(torch.randn(2, 1, 60, 376))
    assert torch.isfinite(feats).all() and traj.dtype == torch.float32


@pytest.mark.parametrize("method", ["lhs", "uniform"])
def test_synthetic_batch_covers_the_range(method):
    schema = load_schema("Archetype Plini")
    s = ParamSampler(schema, rng=np.random.default_rng(3))
    n = 200
    batch = s.synthetic_batch(n, method)
    assert len(batch) == n and all(x.source == "random" for x in batch)
    vals = np.stack([x.values for x in batch])
    assert vals.min() >= 0 and vals.max() <= 1
    # a continuous, always-active knob: LHS puts exactly one draw per stratum; the range is covered either way
    masks = np.stack([x.mask for x in batch])
    for i, p in enumerate(schema.parameters):
        if masks[:, i].all() and not p.is_categorical and "gain" not in p.name:
            col = vals[:, i]
            counts = np.histogram(col, bins=10, range=(0, 1))[0]
            assert counts.min() >= (10 if method == "lhs" else 1), (p.name, counts)
            if method == "lhs":
                assert counts.min() >= 15 and counts.max() <= 25, (p.name, counts)
            break
    else:
        pytest.skip("no always-active continuous knob in this schema")
    assert len({tuple(m) for m in masks}) > 1                                       # routing (hence active blocks) varies


def test_source_filter_and_guard(tmp_path):
    w = ShardWriter(tmp_path, "Archetype Plini", "train", 0, 24_000, groups_per_shard=10)
    p = len(load_schema("Archetype Plini").parameters)
    rng = np.random.default_rng(0)
    for k, src in enumerate([(0, 0), (1, 1), (0, 1)]):                               # pure random, pure preset, mixed
        audio = rng.normal(0, 0.05, (4, 2, 2400)).astype(np.float32)
        if k == 1:
            audio[2] *= 300.0                                                       # one exploded row inside the preset group
        w.add_group(audio, rng.random((2, p)), np.ones((2, p), bool), (0, 1, 0, 1), src)
    w.close()
    paths = list_shards(tmp_path, "train")
    assert len(GroupDataset(paths, ["Archetype Plini"])) == 3
    assert len(GroupDataset(paths, ["Archetype Plini"], source_filter="0")) == 1
    assert len(GroupDataset(paths, ["Archetype Plini"], source_filter="1")) == 1      # the mixed group is in neither
    ds = GroupDataset(paths, ["Archetype Plini"], source_filter="1")
    item = ds[0]
    assert item["audio"].abs().max() <= 1.0 + 1e-6 and ds.n_guarded == 1
    with pytest.raises(ValueError):
        GroupDataset(paths, ["Archetype Plini"], source_filter="2")


def test_load_pretrained_matches_heads_by_plugin_name(tmp_path):
    cfg = dict(head_hidden=(32, 16), switch_hidden=16)
    a, _ = build_model(ToneNetConfig(**cfg), ["Archetype Plini", "Archetype Gojira"])
    save_checkpoint(tmp_path / "a.pt", a)
    b, _ = build_model(ToneNetConfig(**cfg), ["Archetype Gojira", "Archetype Plini"])          # swapped order
    rep = load_pretrained(b, tmp_path / "a.pt")
    assert not rep["skipped"]
    assert torch.equal(b.heads[0].out.weight, a.heads[1].out.weight)                # Gojira's head went to Gojira's slot
    assert torch.equal(b.heads[1].out.weight, a.heads[0].out.weight)
    assert torch.equal(b.plugin_cls.weight[0], a.plugin_cls.weight[1])
    assert torch.equal(b.timbre.multi[0][1].weight, a.timbre.multi[0][1].weight)
    c, _ = build_model(ToneNetConfig(head_hidden=(48, 16), switch_hidden=16), ["Archetype Plini"])   # different head width
    rep = load_pretrained(c, tmp_path / "a.pt")
    assert any(k.startswith("heads.0.") for k in rep["skipped"]) and rep["loaded"]


@pytest.fixture(scope="module")
def two_stage_shards(tmp_path_factory):
    root = tmp_path_factory.mktemp("two_stage")
    bank = synthetic_bank(root / "bank", 6)
    for split in ("train", "val"):                                                   # val for the shared validation directory
        cfg = RenderConfig(bank_dir=str(bank), out_dir=str(root / "base"), groups=2, groups_per_shard=2, split=split)
        for i, n in enumerate(PLUGINS):
            s = load_schema(n)
            render_plugin(n, i, cfg, FakePlugin(s), s)
    return root, bank


def test_synthetic_and_scraped_modes(two_stage_shards, monkeypatch):
    root, bank = two_stage_shards
    pre = RenderConfig(bank_dir=str(bank), out_dir=str(root / "pre"), groups=3, groups_per_shard=3, mode="synthetic", diversity_probe=5)
    s = load_schema("Archetype Plini")
    paths = render_plugin("Archetype Plini", 0, pre, FakePlugin(s), s)
    sh = Shard(paths[0])
    assert sh.n_groups == 3 and set(sh.read("source", slice(None)).tolist()) == {0}
    more = render_plugin("Archetype Plini", 0, RenderConfig(**{**pre.__dict__, "seed": 1, "shard_offset": 1}), FakePlugin(s), s)
    assert more[0].name.endswith("__0001.h5") and len(list_shards(root / "pre", "train")) == 2          # a second round adds a shard

    presets = [{p.name: float(v) for p, v in zip(s.parameters, ParamSampler(s, rng=np.random.default_rng(i)).random().values)}
               for i in range(5)]
    monkeypatch.setattr("dataset_builder.render_dataset.load_presets", lambda *a, **k: presets)
    ft = RenderConfig(bank_dir=str(bank), out_dir=str(root / "ft"), groups=1, groups_per_shard=50, mode="scraped", preset_replicas=2)
    paths = render_plugin("Archetype Plini", 0, ft, FakePlugin(s), s)
    sh = Shard(paths[0])
    assert sh.n_groups == 5 and set(sh.read("source", slice(None)).tolist()) == {1}        # 5 presets x 2 replicas / 2, uncapped
    exact = np.stack([[s.parameters[i].snap(p[s.parameters[i].name]) for i in range(len(s.parameters))] for p in presets])
    rendered = sh.read("params", slice(None))
    assert all(np.abs(exact - r).max(axis=1).min() < 1e-6 for r in rendered)              # every row is one preset, unjittered

    for i, p in enumerate(presets):
        p["__info__"] = {"id": f"p{i}", "preset_name": f"Song {i} tone", "meta": {"song": f"Song {i}", "artist": "A"}}
    monkeypatch.setattr("dataset_builder.render_dataset.load_presets", lambda *a, **k: presets)
    paths = render_plugin("Archetype Plini", 0, RenderConfig(**{**ft.__dict__, "out_dir": str(root / "ft2"), "preset_replicas": 1}),
                          FakePlugin(s), s)
    man = json.loads(paths[0].with_name(paths[0].stem + ".presets.json").read_text(encoding="utf-8"))
    songs = {e[k]["meta"]["song"] for e in man for k in ("slot0", "slot1")}
    assert len(man) == 3 and {f"Song {i}" for i in range(5)} <= songs                         # every song traceable

    monkeypatch.setattr("dataset_builder.render_dataset.load_presets", lambda *a, **k: [])
    assert render_plugin("Archetype Plini", 0, ft, FakePlugin(s), s) == []                # scraped with no presets renders nothing


def test_pretrain_then_finetune_end_to_end(two_stage_shards, tmp_path):
    from training.train import TrainConfig, main, train

    root, bank = two_stage_shards
    for i, n in enumerate(PLUGINS):
        sc = load_schema(n)
        render_plugin(n, i, RenderConfig(bank_dir=str(bank), out_dir=str(root / "pre2"), groups=2, groups_per_shard=2,
                                         mode="synthetic"), FakePlugin(sc), sc)
    common = dict(val_shards=str(root / "base"), epochs=1, groups_per_batch=2, warmup_steps=2, head_hidden=[32, 16], switch_hidden=16)
    best = train(TrainConfig(shards=str(root / "pre2"), out=str(tmp_path / "s1"), **common))
    assert best.exists()
    ft = TrainConfig(shards=str(root / "base"), out=str(tmp_path / "s2"), pretrained=str(best), finetune=True, source_filter="0", **common)
    assert train(ft).exists()
    with pytest.raises(SystemExit):
        main(["--source-filter", "7"])
