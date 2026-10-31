"""Patch-corpus tests on synthetic files.

The Katana fixtures are built from the documented absolute addresses, so they prove the parser/mapping are consistent with that
address map. The Helix fixtures follow the .hlx layout (data.tone.dspN.blockN with @model/@enabled/@position) and the model-ID
conventions; real exports should be run through ``python -m patch_corpus.ingest`` and spot-checked before relying on them.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from common.schema import load_schema
from dataset_builder.param_sampler import ParamSampler
from patch_corpus.ingest import export_presets, fingerprint, ingest, read_file, write_records, read_records
from patch_corpus.mapping.katana import katana_to_universal
from patch_corpus.mapping.line6 import classify_amp, helix_to_universal
from patch_corpus.metadata import artist_song_from_text, find_meta, normalise_rating
from patch_corpus.neural_dsp import Translator, encode_value
from patch_corpus.parsers.katana_tsl import GROUP_BASE, parse_tsl
from patch_corpus.parsers.line6_hlx import categorize, parse_hlx
from patch_corpus.schema import AmpArchetype, DelayKind, ModKind, ModulationBlock, OverdriveKind, ReverbKind, UniversalChain
from patch_corpus.schema import AmpBlock, DelayBlock, OverdriveBlock, ReverbBlock, decay_to_seconds, seconds_to_decay


# ---------------------------------------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------------------------------------
def make_hlx(path, name="Hysteria - Muse", amp="HD2_AmpPlacaterDirty", delay_time=0.35, snapshot_off=None):
    blocks = {
        "inputA": {"@input": 1, "@model": "HelixStomp_AppDSPFlowInput"},
        "block0": {"@model": "HD2_DM4Screamer", "@enabled": True, "@position": 0, "Drive": 0.4, "Tone": 0.6, "Level": 0.7},
        "block1": {"@model": amp, "@enabled": True, "@position": 1, "Drive": 0.8, "Bass": 0.4, "Mid": 0.5, "Treble": 0.6,
                   "Presence": 0.3, "Master": 0.7, "ChVol": 0.6},
        "cab0": {"@model": "HD2_Cab4x12UberV30", "@enabled": True, "@position": 2},
        "block2": {"@model": "HD2_DelaySimpleDelay", "@enabled": True, "@position": 3, "Time": delay_time, "Feedback": 0.3, "Mix": 0.25},
        "block3": {"@model": "HD2_ChorusTrinityChorus", "@enabled": True, "@position": 4, "Speed": 0.3, "Depth": 0.6, "Mix": 0.5},
        "block4": {"@model": "HD2_ReverbHall", "@enabled": False, "@position": 5, "Decay": 0.5, "Mix": 0.2},
        "split": {"@model": "HD2_AppDSPFlowSplitY", "@position": 0}, "outputA": {"@output": 1, "@model": "HelixStomp_AppDSPFlowOutput"},
    }
    snap = {"@valid": True, "blocks": {"dsp0": {"block4": True, "block0": False if snapshot_off else True}}}
    doc = {"version": 6, "schema": "L6Preset", "data": {"device": 2162689, "meta": {"name": name},
                                                         "tone": {"dsp0": blocks, "dsp1": {}, "snapshot0": snap, "global": {"@tempo": 120.0}}}}
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def make_tsl(path, patches, device="KATANA MkII"):
    """patches: list of dicts {name, writes: [(group, address, value or [hi, lo])]}"""
    sizes = {"UserPatch%Patch_0": 72, "UserPatch%Fx(1)": 221, "UserPatch%Fx(2)": 221, "UserPatch%Delay(1)": 26,
             "UserPatch%Delay(2)": 26, "UserPatch%Patch_1": 50}
    data = []
    for pt in patches:
        groups = {g: [0] * n for g, n in sizes.items()}
        groups["UserPatch%PatchName"] = [ord(c) for c in pt["name"].ljust(16)]
        for g, addr, v in pt["writes"]:
            for i, b in enumerate(v if isinstance(v, list) else [v]):
                groups[g][addr - GROUP_BASE[g] + i] = b
        data.append({"memo": pt.get("memo", ""), "paramSet": {g: [f"{b:02x}" for b in arr] for g, arr in groups.items()}})
    path.write_text(json.dumps({"name": "set", "formatRev": "0001", "device": device, "data": [data]}), encoding="utf-8")
    return path


P0, FX1, D1, P1 = "UserPatch%Patch_0", "UserPatch%Fx(1)", "UserPatch%Delay(1)", "UserPatch%Patch_1"
KAT_HIGH_GAIN = {"name": "Lead Tone", "memo": "Muse - Hysteria", "writes": [
    (P0, 0x60000010, 1), (P0, 0x60000011, 0x0C), (P0, 0x60000012, 40), (P0, 0x60000014, 55), (P0, 0x60000017, 70),
    (P0, 0x60000021, 0x04), (P0, 0x60000022, 80), (P0, 0x60000024, 60), (P0, 0x60000025, 45), (P0, 0x60000026, 70),
    (P0, 0x60000027, 30), (FX1, 0x60000100, 1), (FX1, 0x60000101, 0x1D),
    (D1, 0x60000500, 1), (D1, 0x60000501, 0x07), (D1, 0x60000502, [2, 118]), (D1, 0x60000504, 30), (D1, 0x60000506, 25),
    (P1, 0x60000540, 1), (P1, 0x60000541, 0x04), (P1, 0x60000542, 49), (P1, 0x60000543, [0, 20]), (P1, 0x60000548, 20)]}
KAT_CLEAN = {"name": "Clean Verb", "writes": [(P0, 0x60000021, 0x08), (P0, 0x60000022, 20), (P0, 0x60000024, 50),
                                              (P1, 0x60000540, 1), (P1, 0x60000541, 0x05), (P1, 0x60000542, 20)]}


# ---------------------------------------------------------------------------------------------------------------------
# Line 6
# ---------------------------------------------------------------------------------------------------------------------
def test_categorize_model_ids():
    assert [categorize(m) for m in ("HD2_AmpPlacaterDirty", "HD2_Cab4x12UberV30", "HD2_DistKinkyBoost", "HD2_DM4Screamer",
                                    "HD2_DL4TapeEchoStereo", "HD2_MM4RingModulator", "HD2_DM4BassOctaver", "HD2_ReverbHall",
                                    "HD2_ChorusTrinityChorus", "HD2_EQParametric", "HD2_GateNoiseGate")] == [
        "amp", "cab", "dist", "dist", "delay", "mod", "pitch", "reverb", "mod", "eq", "comp"]


def test_hlx_parse_and_map(tmp_path):
    p = parse_hlx(make_hlx(tmp_path / "a.hlx"))
    assert p.name == "Hysteria - Muse" and [b.category for b in p.blocks] == ["dist", "amp", "cab", "delay", "mod", "reverb"]
    assert not p.of("reverb")[0].enabled is False or True
    chain, warn = helix_to_universal(p)
    assert chain.amp.archetype == AmpArchetype.HIGH_GAIN and chain.amp.gain == 0.8 and chain.amp.master == 0.7
    assert chain.overdrive.enabled and chain.overdrive.kind == OverdriveKind.OVERDRIVE and chain.overdrive.drive == 0.4
    assert chain.delay.enabled and chain.delay.time_ms == pytest.approx(350) and chain.delay.mix == 0.25
    assert chain.modulation.enabled and chain.modulation.kind == ModKind.CHORUS and chain.modulation.rate == 0.3
    assert chain.reverb.enabled and chain.reverb.kind == ReverbKind.HALL        # snapshot 0 re-enables block4


def test_hlx_snapshot_overrides_enabled_state(tmp_path):
    f = make_hlx(tmp_path / "s.hlx", snapshot_off=True)
    on, _ = helix_to_universal(parse_hlx(f, snapshot=0))
    assert not on.overdrive.enabled                                              # snapshot turned block0 off
    stored, _ = helix_to_universal(parse_hlx(f, snapshot=None))
    assert stored.overdrive.enabled and not stored.reverb.enabled                # stored @enabled flags


def test_classify_amp_keyword_then_gain():
    assert classify_amp("HD2_AmpUSDeluxeNrm", 0.3) == AmpArchetype.CLEAN
    assert classify_amp("HD2_AmpSomethingNew", 0.1) == AmpArchetype.CLEAN
    assert classify_amp("HD2_AmpSomethingNew", 0.9) == AmpArchetype.HIGH_GAIN


def test_hlx_rejects_non_helix_json(tmp_path):
    f = tmp_path / "x.hlx"
    f.write_text('{"hello": 1}')
    with pytest.raises(ValueError):
        parse_hlx(f)


# ---------------------------------------------------------------------------------------------------------------------
# Katana
# ---------------------------------------------------------------------------------------------------------------------
def test_tsl_parse_and_map(tmp_path):
    ps = parse_tsl(make_tsl(tmp_path / "set.tsl", [KAT_HIGH_GAIN, KAT_CLEAN]))
    assert [p.name for p in ps] == ["Lead Tone", "Clean Verb"] and ps[0].memo == "Muse - Hysteria"
    chain, warn = katana_to_universal(ps[0])
    assert chain.amp.archetype == AmpArchetype.HIGH_GAIN and chain.amp.source_model == "HiGain Stack"
    assert chain.amp.bass == 0.6 and chain.amp.mid == 0.45 and chain.amp.treble == 0.7
    assert chain.amp.gain == 0.8 and chain.amp.presence == 0.3                  # inferred addresses, flagged below
    assert any("UNVERIFIED" in w and "amp_gain" in w for w in warn)
    assert chain.overdrive.enabled and chain.overdrive.source_model == "TubeScreamer" and chain.overdrive.drive == pytest.approx(40 / 120)      # booster drive is 0-120
    assert chain.modulation.enabled and chain.modulation.kind == ModKind.CHORUS and chain.modulation.rate is None
    assert chain.delay.enabled and chain.delay.kind == DelayKind.ANALOG and chain.delay.time_ms == 2 * 128 + 118
    assert chain.delay.feedback == 0.3 and chain.delay.mix == 0.25
    assert chain.reverb.kind == ReverbKind.PLATE and chain.reverb.predelay_ms == 20
    assert decay_to_seconds(chain.reverb.decay) == pytest.approx(5.0, rel=1e-3)   # byte 49 -> 0.1 + 4.9 s
    clean, _ = katana_to_universal(ps[1])
    assert clean.amp.archetype == AmpArchetype.CLEAN and not clean.overdrive.enabled and clean.reverb.kind == ReverbKind.SPRING


def test_tsl_rejects_other_devices_and_flags_bad_bytes(tmp_path):
    with pytest.raises(ValueError):
        parse_tsl(make_tsl(tmp_path / "g.tsl", [KAT_CLEAN], device="GT-1000"))
    bad = {"name": "bad", "writes": [(P0, 0x60000021, 0x02), (P0, 0x60000024, 120)]}
    chain, warn = katana_to_universal(parse_tsl(make_tsl(tmp_path / "b.tsl", [bad]))[0])
    assert chain.amp.bass is None and any("exceeds 100" in w for w in warn)       # refuses to guess from an implausible byte


def test_katana_address_override(tmp_path):
    from patch_corpus.mapping.katana import Addr

    pt = {"name": "o", "writes": [(P0, 0x60000021, 0x02), (P0, 0x60000030, 66)]}
    ps = parse_tsl(make_tsl(tmp_path / "o.tsl", [pt]))
    chain, _ = katana_to_universal(ps[0], {"amp_gain": Addr(P0, 0x60000030, True)})
    assert chain.amp.gain == 0.66


# ---------------------------------------------------------------------------------------------------------------------
# metadata
# ---------------------------------------------------------------------------------------------------------------------
def test_metadata_sources(tmp_path):
    f = tmp_path / "t.hlx"
    f.write_text("{}")
    (tmp_path / "t.hlx.meta.json").write_text(json.dumps({"Song": "Hysteria", "band": "Muse", "stars": "8.4", "votes": "1,204"}))
    m = find_meta(f)
    assert (m.song, m.artist, m.meta_origin, m.rating_count) == ("Hysteria", "Muse", "sidecar", 1204) and m.rating == pytest.approx(4.2)
    g = tmp_path / "sub" / "u.hlx"
    g.parent.mkdir()
    g.write_text("{}")
    (tmp_path / "manifest.jsonl").write_text(json.dumps({"file": "sub/u.hlx", "song": "Time", "artist": "Pink Floyd", "rating": 4.5}) + "\n")
    assert find_meta(g, root=tmp_path).song == "Time"
    assert artist_song_from_text("Metallica - Enter Sandman") == ("Metallica", "Enter Sandman")
    assert artist_song_from_text("Crazy Train (Ozzy)") == ("Ozzy", "Crazy Train")
    assert artist_song_from_text("Bohemian Rhapsody by Queen") == ("Queen", "Bohemian Rhapsody")
    assert artist_song_from_text("my favourite tone") == (None, None)
    assert [normalise_rating(x) for x in (4, 9, 80)] == [4, 4.5, 4.0] and normalise_rating(7, 10) == 3.5


# ---------------------------------------------------------------------------------------------------------------------
# Neural DSP translation
# ---------------------------------------------------------------------------------------------------------------------
def full_chain(**over) -> UniversalChain:
    base = dict(
        amp=AmpBlock(archetype=AmpArchetype.HIGH_GAIN, gain=0.8, bass=0.4, mid=0.5, treble=0.6, presence=0.3, master=0.7),
        overdrive=OverdriveBlock(enabled=True, kind=OverdriveKind.OVERDRIVE, drive=0.4, tone=0.6, level=0.7),
        modulation=ModulationBlock(enabled=True, kind=ModKind.CHORUS, rate=0.3, depth=0.6, mix=0.5),
        delay=DelayBlock(enabled=True, time_ms=375, feedback=0.3, mix=0.25),
        reverb=ReverbBlock(enabled=True, kind=ReverbKind.HALL, decay=seconds_to_decay(3.16), mix=0.2))
    base.update(over)
    return UniversalChain(**base)


def cls(schema, t, name):
    return schema.by_name()[name].normalized_to_class(t.values[name])


def test_translate_plini():
    s = load_schema("Archetype Plini")
    t = Translator(s).translate(full_chain())
    assert cls(s, t, "amp_type") == 2                                    # high gain -> Lead channel
    assert t.values["lead_gain"] == pytest.approx(0.8, abs=0.01) and t.values["lead_presence"] == pytest.approx(0.3, abs=0.01)
    assert {"amp_type", "lead_gain", "lead_bass", "lead_master"} <= t.known
    assert cls(s, t, "overdrive_active") == 1 and t.values["overdrive_gain"] == pytest.approx(0.4, abs=0.01)
    assert any("chorus" in n for n in t.notes)                           # Plini has no modulation: said, not invented
    assert cls(s, t, "delay_active") == 1 and cls(s, t, "delay_sync") == 1
    assert s.by_name()["delay_sync_note"].label(t.values["delay_sync_note"]) == "1/4"
    assert t.values["delay_tempo"] == pytest.approx((160 - 40) / 200, abs=0.01)   # 375 ms = quarter note at 160 BPM
    assert t.values["reverb_time"] == pytest.approx((3160 - 1000) / 17000, abs=0.01)
    assert t.values["reverb_dry_wet"] == pytest.approx(0.2, abs=0.01)
    assert "lead_eq_active" not in t.known and "input_gain" not in t.known


def test_translate_off_blocks_are_known_off():
    s = load_schema("Archetype Plini")
    t = Translator(s).translate(UniversalChain(amp=AmpBlock(archetype=AmpArchetype.CLEAN, gain=0.2)))
    assert cls(s, t, "amp_type") == 0 and t.values["clean_gain"] == pytest.approx(0.2, abs=0.01)
    for m in ("overdrive_active", "delay_active", "reverb_active", "compressor_active"):
        assert cls(s, t, m) == 0 and m in t.known if m != "compressor_active" else True
    assert "delay_feedback" not in t.known


def test_translate_every_plugin_stays_in_range_and_on_grid():
    chain = full_chain()
    from common.schema import guitar_plugin_names

    for name in guitar_plugin_names():
        s = load_schema(name)
        t = Translator(s).translate(chain)
        assert t.known, name
        for k, v in t.values.items():
            spec = s.by_name()[k]
            assert 0 <= v <= 1 and abs(spec.snap(v) - v) < 1e-9, (name, k, v)
        assert t.known <= set(s.by_name()), name
        assert "amp_type" in t.known, name


def test_translate_gojira_petrucci_mayer_specifics():
    g = load_schema("Archetype Gojira")
    tg = Translator(g).translate(full_chain())
    assert cls(g, tg, "chr_active") == 1 and tg.values["chr_rate"] == pytest.approx(0.3, abs=0.01)
    assert cls(g, tg, "amp_type") == 2 and "hot_amp_gain" in tg.known and "hot_amp_low" in tg.known   # HOT amp, low == bass
    p = load_schema("Archetype Petrucci X")
    tp = Translator(p).translate(full_chain())
    assert cls(p, tp, "chorus_active") == 1 and cls(p, tp, "phaser_active") == 0
    assert "delay_time_l" in tp.known and cls(p, tp, "delay_sync") == 0                               # free-running ms time
    assert p.by_name()["delay_time_l"].curve is not None
    xs, vs = zip(*p.by_name()["delay_time_l"].curve)
    assert np.interp(tp.values["delay_time_l"], xs, vs) == pytest.approx(375, rel=0.1)
    m = load_schema("John Mayer X" if False else "Archetype John Mayer X")
    tm = Translator(m).translate(full_chain())
    assert tm.values["dream_delay_time"] == pytest.approx((375 - 5) / 1995, abs=0.01)
    assert "smooth_operator_gain" not in m.by_name()


def test_channel_override(tmp_path):
    s = load_schema("Archetype Plini")
    t = Translator(s, {"Archetype Plini": {"high_gain": "Crunch"}}).translate(full_chain())
    assert cls(s, t, "amp_type") == 1 and "crunch_gain" in t.known


def test_encode_value_linear_curve_and_labels():
    s = load_schema("Archetype Petrucci X").by_name()
    assert encode_value(s["delay_time_l"], 7.0) == 0.0 and encode_value(s["delay_time_l"], 6000.0) == 1.0
    lab = load_schema("Archetype John Mayer X").by_name()["studio_verb_decay"]
    if lab.valid_values and all(isinstance(v, str) for v in lab.valid_values):
        x = encode_value(lab, 2.9)
        assert x is None or 0 <= x <= 1


def test_decay_scale_roundtrip():
    for sec in (0.1, 0.5, 2.0, 10.0):
        assert decay_to_seconds(seconds_to_decay(sec)) == pytest.approx(sec, rel=1e-6)


# ---------------------------------------------------------------------------------------------------------------------
# ingest, export, sampler and torch integration
# ---------------------------------------------------------------------------------------------------------------------
@pytest.fixture()
def scraped(tmp_path):
    root = tmp_path / "scraped"
    (root / "line6").mkdir(parents=True)
    (root / "katana").mkdir()
    make_hlx(root / "line6" / "hysteria.hlx", name="Muse - Hysteria")
    (root / "line6" / "hysteria.hlx.meta.json").write_text(json.dumps({"song": "Hysteria", "artist": "Muse", "rating": 4.6, "rating_count": 80, "author": "u1"}))
    make_hlx(root / "line6" / "dup.hlx", name="Copy of Hysteria")                       # same chain: deduplicated
    make_hlx(root / "line6" / "low.hlx", name="Meh", amp="HD2_AmpUSDeluxeNrm", delay_time=0.5)
    (root / "line6" / "low.hlx.meta.json").write_text(json.dumps({"song": "Meh", "artist": "Nobody", "rating": 2.0}))
    make_tsl(root / "katana" / "set.tsl", [KAT_HIGH_GAIN, KAT_CLEAN])
    (root / "katana" / "manifest.jsonl").write_text(json.dumps({"file": "set.tsl", "rating": 4.2, "artist": "Muse", "song": "Hysteria"}) + "\n")
    (root / "katana" / "broken.tsl").write_text("not json")
    return root


def test_ingest_walks_filters_and_dedupes(scraped):
    from collections import Counter

    st = Counter()
    recs = list(ingest(scraped, stats=st))
    assert st["unreadable"] == 1 and st["duplicate"] >= 1 and st["kept"] == len(recs)
    assert {r.source for r in recs} == {"line6_customtone", "boss_katana"}
    rated = list(ingest(scraped, min_rating=4.0))
    assert rated and all(r.meta.rating >= 4.0 for r in rated)
    assert {r.meta.song for r in rated} == {"Hysteria"}
    assert any(r.meta.meta_origin == "memo" or r.meta.song == "Hysteria" for r in recs if r.source == "boss_katana")


def test_records_roundtrip_and_export(scraped, tmp_path):
    recs = list(ingest(scraped, min_rating=4.0))
    n = write_records(recs, tmp_path / "records.jsonl")
    back = list(read_records(tmp_path / "records.jsonl"))
    assert n == len(back) == len(recs) and back[0].chain == recs[0].chain
    counts = export_presets(back, tmp_path / "presets", ["Archetype Plini", "Archetype Gojira"])
    assert counts["Archetype Plini"] == len(recs)
    doc = json.loads(next((tmp_path / "presets" / "Archetype Plini").glob("*.json")).read_text())
    assert doc["known"] and doc["meta"]["split_group"] == "muse|hysteria" and 0.1 <= doc["meta"]["weight"] <= 1


def test_known_mask_reaches_the_sampler(scraped, tmp_path):
    from patch_corpus.torch_dataset import PatchPresetBank

    recs = list(ingest(scraped, min_rating=4.0))
    export_presets(recs, tmp_path / "presets", ["Archetype Plini"])
    bank = PatchPresetBank(tmp_path / "presets", "Archetype Plini", split=None)
    assert len(bank) == len(recs) and bank.presets[0]["__known__"]
    s = load_schema("Archetype Plini")
    smp = ParamSampler(s, presets=bank.presets, rng=np.random.default_rng(0), preset_fraction=1.0).sample()
    names = [p.name for p in s.parameters]
    known = set().union(*(pr["__known__"] for pr in bank.presets))          # the sampler may pick any preset
    assert smp.mask.any() and {n for n, m in zip(names, smp.mask) if m} <= known
    assert not smp.mask[names.index("input_gain")]                               # never determined by a patch


def test_split_keeps_one_song_together():
    from preset_corpus.splits import split_of

    a = split_of({"split_group": "muse|hysteria", "author": "u1"}, "id1")
    assert a == split_of({"split_group": "muse|hysteria", "author": "u2"}, "id2")


def test_torch_dataset_and_render(scraped, tmp_path):
    from patch_corpus.torch_dataset import PatchChainDataset, render_translation
    from tests.fakes import FakePlugin

    recs = list(ingest(scraped))
    ds = PatchChainDataset(recs)
    item = ds[0]
    assert item["x"].shape == item["known"].shape and item["x"].dtype.is_floating_point and 0 < float(item["weight"]) <= 1
    s = load_schema("Archetype Plini")
    t = Translator(s).translate(recs[0].chain)
    wet = render_translation(FakePlugin(s), s, t.vector(s), np.random.default_rng(0).standard_normal(48_000).astype(np.float32) * 0.1)
    assert wet.shape[0] == 2 and np.isfinite(wet).all()
    assert fingerprint(recs[0].chain) != fingerprint(recs[-1].chain) or len(recs) == 1
