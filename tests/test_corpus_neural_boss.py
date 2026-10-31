"""Offline tests for the Neural DSP value normaliser (mapping/neural.py), the Boss GT/ME liveset reader + mapper, and their wiring into
ingest. Fixtures copy the layouts of real files: a forum Gojira preset (JUCE binary) and GuitarPatches GT-100 / GT-1 / ME-80 livesets."""

from __future__ import annotations

import json
import struct

import pytest

from common.schema import ParameterSpec, load_schema
from patch_corpus.ingest import export_presets, ingest
from patch_corpus.mapping.boss_gt import boss_to_universal
from patch_corpus.mapping.neural import NAME_TABLES, auto_name, neural_to_universal, neural_to_vst, normalise, schema_for
from patch_corpus.parsers.boss_tsl import parse_boss_tsl
from patch_corpus.parsers.neural_xml import NeuralXmlPreset, parse_neural_xml
from patch_corpus.schema import AmpArchetype, DelayKind, OverdriveKind


def param(name: str, value: float) -> bytes:
    return b"PARAM\x00\x01\x02id\x00\x01\x0c\x05" + name.encode() + b"\x00value\x00\x01\t\x04" + struct.pack("<d", value) + b"\x00"


def gojira_xml(**params) -> bytes:
    return (b"neural_dsp_gojira\x00\x01\x07presetNameProp\x00\x01\r\x05Take a Bite\x00presetVersionProp\x00\x01\x07\x051.0.0\x00\x01w"
            + b"".join(param(k, v) for k, v in params.items()))


GOJIRA = dict(ampsActive=1.0, ampType=2.0, leadAmpGain=0.72, leadAmpBass=0.488, leadAmpMid=0.655, leadAmpTreble=0.667, leadAmpPresence=0.692,
              leadAmpVolume=0.673, leadAmpOutput=-2.2, cab1Level=-12.0, cab2MicIR=2.0, pedalsActive=1.0, pedal3Active=1.0, pedal3Distortion=0.181,
              pedal3Filter=1.0, pedal3Volume=0.62, pedal5Active=0.0, fxActive=1.0, delayActive=1.0, delaySyncActive=0.0, delayTempo=120.0,
              delayFeedback=0.232, delayDryWet=0.068, delayPingPong=0.0, delayTapeSaturation=1.0, reverbActive=1.0, reverbMode=0.0,
              reverbTime=4709.32, reverbDryWet=0.172, gateThreshold=-59.9, mysteryKnob=3.0)


# ------------------------------------------------------------------------------------------------ neural normaliser
def test_normalise_rules():
    lin = ParameterSpec(index=0, name="lvl", kind="numeric", default_normalized=0.5, min_value=-48.0, max_value=24.0)
    assert normalise(-12.0, lin) == (0.5, None)
    x, problem = normalise(40.0, lin)
    assert x == 1.0 and "clipped" in problem
    enum = ParameterSpec(index=1, name="t", kind="enum", default_normalized=0, num_steps=3, valid_values=["a", "b", "c"])
    assert normalise(2.0, enum) == (1.0, None) and normalise(5.0, enum)[0] is None
    curve = ParameterSpec(index=2, name="c", kind="numeric", default_normalized=0, min_value=-90, max_value=0, linear=False,
                          curve=[(0.0, -90.0), (0.5, -20.0), (1.0, 0.0)])
    assert normalise(-20.0, curve) == (0.5, None) and normalise(-10.0, curve)[0] == pytest.approx(0.75)


def test_gojira_table_matches_the_schema():
    names = {p.name for p in load_schema("Archetype Gojira").parameters}
    table = NAME_TABLES["Archetype Gojira"]
    assert len(table) >= 115 and set(table.values()) <= names                          # every target exists in the schema


def test_schema_resolution_and_auto_names():
    assert schema_for("neural_dsp_gojira").plugin_name == "Archetype Gojira"
    assert schema_for("gojira-X") is None                                                # a different plugin, no schema yet
    assert schema_for("nolly-X").plugin_name == "Archetype Nolly X"
    names = {"cab_1_level", "dly_feedback", "amp_section_active", "rev_dry_wet"}
    assert auto_name("cab1Level", names) == "cab_1_level" and auto_name("delayFeedback", names) == "dly_feedback"
    assert auto_name("ampsActive", names) == "amp_section_active" and auto_name("reverbDryWet", names) == "rev_dry_wet"
    assert auto_name("nothing", names) is None


def test_gojira_preset_to_vst_and_universal(tmp_path):
    f = tmp_path / "p.xml"
    f.write_bytes(gojira_xml(**GOJIRA))
    pre = parse_neural_xml(f)
    p = neural_to_vst(pre)
    v = p.vst_parameters
    assert p.plugin == "Archetype Gojira" and p.unmatched == ["mysteryKnob"]
    assert v["amp_type"] == 1.0 and v["cab_1_level"] == pytest.approx(0.5, abs=1e-3) and v["cab_2_mic_ir"] == pytest.approx(2 / 6)
    assert v["hot_amp_gain"] == pytest.approx(0.72, abs=1e-3) and v["hot_amp_level"] == pytest.approx((-2.2 + 12) / 24, abs=1e-3)
    assert v["rev_time"] == pytest.approx((4709.32 - 250) / 9750, abs=1e-3) and v["gate_amount"] == pytest.approx((96 - 59.9) / 96, abs=1e-3)
    assert all(0.0 <= x <= 1.0 for x in v.values())
    chain, warn = neural_to_universal(pre, params=p)
    assert chain.amp.archetype == AmpArchetype.HIGH_GAIN and chain.amp.gain == pytest.approx(0.72, abs=1e-3)
    assert chain.amp.bass == pytest.approx(0.488, abs=1e-3) and chain.amp.master == pytest.approx(0.673, abs=1e-3)
    assert chain.overdrive.enabled and chain.overdrive.kind == OverdriveKind.DISTORTION and chain.overdrive.drive == pytest.approx(0.181, abs=1e-3)
    assert not chain.modulation.enabled
    assert chain.delay.enabled and chain.delay.time_ms == 500.0 and chain.delay.kind == DelayKind.TAPE
    assert chain.reverb.enabled and chain.reverb.decay is not None and any("mysteryKnob" in w for w in warn)


def test_no_schema_keeps_only_unit_values():
    pre = NeuralXmlPreset("gojira-X", "x", {"outputGain": 5.0625, "doublerActive": 0.0, "octaverLevel": 0.569}, container="juce_binary_v2")
    p = neural_to_vst(pre)
    assert p.plugin is None and p.vst_parameters == {"doubler_active": 0.0, "octaver_level": 0.569} and p.warnings


# ------------------------------------------------------------------------------------------------ boss GT / ME
def gt_liveset(device="GT", **over) -> dict:
    params = {"divider_ch_select": "0", "preamp_a_on_off": "1", "preamp_a_type": "27", "preamp_a_gain": "90", "preamp_a_bass": "50",
              "preamp_a_middle": "69", "preamp_a_treble": "64", "preamp_a_presence": "51", "preamp_a_level": "45", "preamp_b_gain": "30",
              "preamp_b_bass": "10", "od_ds_on_off": "1", "od_ds_type": "1", "od_ds_drive": "60", "od_ds_tone": "56", "od_ds_effect_level": "50",
              "chorus_on_off": "1", "chorus_rate": "23", "chorus_depth": "40", "chorus_effect_level": "22", "delay_on_off": "1",
              "delay_delay_time": "400", "delay_f_back": "25", "delay_effect_level": "60", "reverb_on_off": "1", "reverb_type": "4",
              "reverb_time": "32", "reverb_pre_delay": "30", "reverb_effect_level": "50",
              **{f"patch_name{i + 1}": str(ord(c)) for i, c in enumerate("HYSTERIA".ljust(16))}, **over}
    return {"device": device, "version": "1", "liveSetData": {"name": "Muse"}, "patchList": [{"name": "", "note": "Muse - Hysteria", "params": params}]}


def test_boss_gt_parse_and_map(tmp_path):
    f = tmp_path / "gt.tsl"
    f.write_text(json.dumps(gt_liveset()))
    (p,) = parse_boss_tsl(f)
    assert (p.name, p.device, p.params["preamp_a_gain"]) == ("HYSTERIA", "GT", 90)
    chain, warn = boss_to_universal(p)
    assert chain.amp.gain == 0.75 and chain.amp.mid == 0.69 and chain.amp.master == 0.45 and chain.amp.source_model == "GT preamp type 27"
    assert chain.overdrive.enabled and chain.overdrive.kind == OverdriveKind.BOOST and chain.overdrive.drive == 0.5
    assert chain.modulation.enabled and chain.modulation.rate == 0.23
    assert chain.delay.time_ms == 400.0 and chain.delay.mix == 0.5 and chain.reverb.predelay_ms == 30.0 and chain.reverb.decay is not None
    assert any("inferred" in w for w in warn)
    f.write_text(json.dumps(gt_liveset(divider_ch_select="1")))                        # channel B selected
    chain_b, _ = boss_to_universal(parse_boss_tsl(f)[0])
    assert chain_b.amp.gain == 0.25 and chain_b.amp.bass == 0.1
    f.write_text(json.dumps(gt_liveset(preamp_a_gain="140")))
    chain_bad, warn_bad = boss_to_universal(parse_boss_tsl(f)[0])
    assert chain_bad.amp.gain is None and any("outside" in w for w in warn_bad)


def test_me80_and_katana_rejected(tmp_path):
    me = {"device": "ME-80", "patchList": [{"name": "", "params": {"patchname": "ComeAsYouAre CL", "amp_sw": "1", "amp_type": "3", "amp1": "64",
                                                                   "amp2": "78", "amp3": "54", "amp4": "74", "amp5": "56", "odds_sw": "1",
                                                                   "odds1": "84", "odds2": "40", "odds3": "42", "rev_sw": "1", "rev": "10"}}]}
    f = tmp_path / "me.tsl"
    f.write_text(json.dumps(me))
    (p,) = parse_boss_tsl(f)
    chain, _ = boss_to_universal(p)
    assert p.name == "ComeAsYouAre CL" and chain.amp.gain == 0.64 and chain.amp.master == 0.56 and chain.overdrive.drive == 0.84
    assert chain.reverb.enabled and chain.reverb.mix == 0.1
    k = tmp_path / "k.tsl"
    k.write_text(json.dumps({"device": "KATANA MkII", "data": [[]]}))
    with pytest.raises(ValueError):
        parse_boss_tsl(k)


# ------------------------------------------------------------------------------------------------ ingest wiring
def test_ingest_dispatches_xml_and_gt(tmp_path):
    root = tmp_path / "presets"
    (root / "neuraldsp").mkdir(parents=True)
    (root / "boss_gt").mkdir()
    (root / "neuraldsp" / "a.xml").write_bytes(gojira_xml(**GOJIRA))
    (root / "neuraldsp" / "manifest.jsonl").write_text(json.dumps({"file": "a.xml", "artist": "Gojira", "song": "Stranded", "title": "Stranded"}) + "\n")
    (root / "boss_gt" / "g.tsl").write_text(json.dumps(gt_liveset()))
    recs = {r.source: r for r in ingest(root, dedupe=False)}
    assert set(recs) == {"neural_dsp", "boss_gt"}
    nd = recs["neural_dsp"]
    assert nd.meta.artist == "Gojira" and nd.native["plugin"] == "Archetype Gojira" and nd.chain.amp.archetype == AmpArchetype.HIGH_GAIN
    counts = export_presets(list(recs.values()), tmp_path / "out", ["Archetype Gojira"])
    assert counts["Archetype Gojira (native)"] == 1 and counts["Archetype Gojira"] == 1
    doc = json.loads((tmp_path / "out" / "Archetype Gojira" / f"{nd.id}.json").read_text())
    assert doc["decode_method"] == "native_preset" and "hot_amp_gain" in doc["known"] and doc["vst_parameters"]["amp_type"] == 1.0


def test_auto_rules_plini_and_abasi():
    plini = [p.name for p in load_schema("Archetype Plini").parameters]
    assert auto_name("leadEQBand0ID", plini) == "lead_eq_65_hz" and auto_name("leadEQBand8ID", plini) == "lead_eq_16_khz"
    assert auto_name("activePedalSectionID", plini) == "active_pedals" and auto_name("overDrivePedalGainID", plini) == "overdrive_gain"
    assert auto_name("reverbHPFID", plini) == "reverb_low_cut" and auto_name("compressorPedalThresID", plini) == "compressor_threshold"
    abasi = [p.name for p in load_schema("Archetype Abasi").parameters]
    al = {"comp": "logos", "drive": "pathos", "bloom": "compression"}
    assert auto_name("compBloom", abasi, al) == "logos_compression" and auto_name("reverbTime", abasi, al) == "reverb_decay"
    assert auto_name("cleanOutput", abasi, al) == "clean_level"


def test_auto_match_rejected_when_value_does_not_fit():
    pre = NeuralXmlPreset("plini", "x", {"ampTypeID": 2.0, "leadGainID": 0.7, "reverbTimeID": 0.5, "cab1MicIRID": 9.0})
    p = neural_to_vst(pre)
    assert p.plugin == "Archetype Plini" and p.vst_parameters["lead_gain"] == pytest.approx(0.7)
    assert set(p.unmatched) == {"reverbTimeID", "cab1MicIRID"} and any("rejected" in w for w in p.warnings)
    chain, _ = neural_to_universal(pre, params=p)
    assert chain.amp.archetype == AmpArchetype.LEAD and chain.amp.gain == pytest.approx(0.7)    # amp chosen by the 'Lead' option label


# ---------------------------------------------------------------------------------------------- X-series (tree reader + scales)
def vprop(name: str, text: str) -> bytes:
    body = b"\x05" + text.encode() + b"\x00"
    return name.encode() + b"\x00\x01" + bytes([len(body)]) + body


def vnode(name: str, props: list[bytes] = (), children: list[bytes] = ()) -> bytes:
    """A node of the X-series container: name, then ``\0`` or ``\x01 n`` + props, then ``\0`` or ``\x01 n`` + children."""
    return (name.encode() + b"\x00" + (b"\x01" + bytes([len(props)]) + b"".join(props) if props else b"\x00")
            + (b"\x01" + bytes([len(children)]) + b"".join(children) if children else b"\x00"))


def x_series(plugin: str, sections: list[bytes], top: list[bytes] = (), app: list[bytes] = ()) -> bytes:
    parameters = vnode("parameters", list(top), [vnode("subModels", [], sections)])
    return vnode(plugin, [vprop("version", "1.0.0"), vprop("name", "T")],
                 [vnode("appModel", list(app), [vnode("subModels", [], [parameters])])])


def test_x_series_tree_reader_keeps_repeated_names_apart(tmp_path):
    long_path = "/" + "x" * 190                                                  # a 192-byte body: the length byte is > 0x7F, not a varint
    data = x_series("nolly-x", [
        vnode("ampParameters", [vprop("sectionActive", "true")], [vnode("leadAmp", [vprop("leadGain", "0.7")])]),
        vnode("eqParameters", [vprop("sectionActive", "false")]),
        vnode("cabParameters", [vprop("sectionActive", "true"), vprop("rightCabPan", "9.5"), vprop("rightChosenIRFilePath", long_path)]),
    ], top=[vprop("gateActive", "true")], app=[vprop("selectedAmp", "2"), vprop("selectedCab", "3")])
    path = tmp_path / "x.xml"
    path.write_bytes(data)
    pre = parse_neural_xml(path)
    assert pre.container == "juce_binary_v2" and pre.plugin == "nolly-x" and not [w for w in pre.warnings if "tree reader" in w]
    assert pre.params == {"amp.selectedAmp": 2.0, "cab.selectedCab": 3.0, "gateActive": 1.0, "amp.sectionActive": 1.0, "amp.leadAmp.leadGain": 0.7,
                          "eq.sectionActive": 0.0, "cab.sectionActive": 1.0, "cab.rightCabPan": 9.5}


def test_x_series_scales_and_names_on_the_nolly_schema():
    pre = NeuralXmlPreset("nolly-x", "t", {
        "amp.selectedAmp": 3.0, "cab.selectedCab": 3.0, "amp.sectionActive": 1.0, "eq.sectionActive": 0.0, "preFX.sectionActive": 1.0,
        "amp.leadAmp.leadGain": 0.8, "amp.rhythmAmp.rhythmGain": 0.2,                       # % knobs: stored as the 0-1 position
        "postFX.reverb.reverbDecay": 4750.0, "postFX.reverb.reverbMix": 0.3,                 # decay: ms for a label in seconds
        "postFX.postDelay.postDelayTime": 765.75, "postFX.postDelay.postDelayTempo": 90.0, "postFX.postDelay.postDelaySyncNote": 10.0,
        "transpose": -12.0, "cab.leftCabPan": -50.0, "cab.rightCabPan": 0.0, "outputGain": 0.0, "gateThreshold": -96.0,
        "eq.leadEQ.leadEQBand3": 6.0, "eq.leadEQ.leadEQHpf": 20.0, "eq.leadEQ.leadEQLpf": 20000.0, "preFX.overdrive1.overdrive1Tone": 0.25,
        "cab.rightCab0MicType": 4.0, "cab.leftCab3MicType": 1.0, "cab.rightRoomMicLevel": -40.0, "cab.leftCabMicLevel": 6.0,
        "cab.rightCabStereo": 0.0,
    }, container="juce_binary_v2")
    p = neural_to_vst(pre)
    v = p.vst_parameters
    assert p.plugin == "Archetype Nolly X" and p.unmatched == ["cab.rightCabStereo"]       # no stereo parameter in the schema
    assert (v["amp_type"], v["cab_type_unlinked"]) == (1.0, 1.0) and p.labels["amp_type"] == "Lead"
    assert (v["active_amp_section"], v["active_eq_section"], v["active_pre_fx_section"]) == (1.0, 0.0, 1.0)      # sectionActive per section
    assert v["lead_amp_gain"] == pytest.approx(0.8, abs=0.01) and v["rhythm_amp_gain"] == pytest.approx(0.2, abs=0.01)
    assert v["reverb_decay"] == pytest.approx((4.75 - 1.0) / 7.5, abs=0.01) and v["reverb_mix"] == pytest.approx(0.3, abs=0.01)
    assert v["delay_2_time"] == pytest.approx((765.75 - 21) / (3000 - 21), abs=0.01) and v["delay_2_tempo"] == pytest.approx(0.25, abs=0.01)
    assert v["delay_2_note"] == pytest.approx(10 / 20)                                       # note index
    assert (v["transpose"], v["cab_l_pan"], v["cab_r_pan"]) == (0.0, 0.0, pytest.approx(0.5, abs=0.01))
    assert v["output_gain"] == pytest.approx(0.5) and v["gate_threshold"] == 0.0
    assert v["lead_amp_eq_band_3"] == pytest.approx(0.75, abs=0.01) and v["lead_amp_eq_high_pass"] == 0.0 and v["lead_amp_eq_low_pass"] == 1.0
    assert v["overdrive_1_tone"] == pytest.approx(0.25, abs=0.01)
    assert v["clean_amp_cab_mic_r_type"] == pytest.approx(4 / 12) and v["lead_amp_cab_mic_l_type"] == pytest.approx(1 / 12)   # cab<N> = the N-th amp
    assert v["room_r_send"] == 0.0 and v["cab_l_level"] == 1.0
    chain, _ = neural_to_universal(pre, params=p)
    assert chain.amp.archetype == AmpArchetype.LEAD and chain.amp.gain == pytest.approx(0.8, abs=0.01)


def test_petrucci_eq_freq_is_a_knob_position():
    pre = NeuralXmlPreset("petrucci-x", "t", {"eq.leadEQ.leadEQBand1Freq": 0.613, "eq.leadEQ.leadEQBand1Gain": 1.5, "eq.leadEQ.leadEQBand4Gain": -12.0,
                                              "eq.leadEQ.leadEQHpf": 20.0, "wahComp.sectionActive": 1.0, "volume.volume.volumeMidPoint": 0.2,
                                              "cab.rightMicType": 2.0, "cab.rightRoomMicLevel": -20.0, "postFX.delay.delayNoteL": 7.0,
                                              "postFX.delay.delayTimeL": 375.0}, container="juce_binary_v2")
    v = neural_to_vst(pre).vst_parameters
    assert v["lead_eq_lo_freq"] == pytest.approx(0.613, abs=0.01) and v["lead_eq_lo_gain"] == pytest.approx(0.5625, abs=0.01)
    assert v["lead_eq_hi_gain"] == 0.0 and v["lead_eq_hpf"] == 0.0 and v["active_wah_comp_section"] == 1.0
    assert v["volume_mid_point"] == pytest.approx(0.2, abs=0.01) and v["cab_mic_r_type"] == pytest.approx(2 / 6)
    assert v["room_r_level"] == pytest.approx((-20 + 60) / 72, abs=0.01) and v["delay_sync_note_l"] > 0 and v["delay_time_l"] > 0


def test_numeric_label_enums_are_looked_up_by_value():
    sp = ParameterSpec(index=0, name="t", kind="enum", default_normalized=0.5, num_steps=25, valid_values=[f"{n} st" for n in range(-12, 13)])
    assert normalise(-12.0, sp) == (0.0, None) and normalise(0.0, sp) == (0.5, None) and normalise(12.0, sp) == (1.0, None)
    assert normalise(40.0, sp)[1] is not None                                                 # outside the labels: clipped and reported
    notes = ParameterSpec(index=0, name="n", kind="enum", default_normalized=0.0, num_steps=3, valid_values=["Clean", "Crunch", "Lead"])
    assert normalise(2.0, notes) == (1.0, None)                                               # a real list of choices stays an index


XSERIES_DIR = pytest.importorskip("pathlib").Path(__file__).resolve().parents[1] / "data" / "data_scraping" / "presets" / "neuraldsp"


@pytest.mark.skipif(not XSERIES_DIR.exists(), reason="scraped Neural DSP presets not present")
@pytest.mark.parametrize("plugin", ["nolly-x", "petrucci-x"])
def test_x_series_coverage_on_scraped_presets(plugin):
    mapped = total = 0
    for f in XSERIES_DIR.glob("*.xml"):
        pre = parse_neural_xml(f)
        if pre.plugin != plugin:
            continue
        assert not [w for w in pre.warnings if "tree reader" in w], f.name
        p = neural_to_vst(pre)
        mapped, total = mapped + len(p.raw), total + len(p.raw) + len(p.unmatched)
    if total:
        assert mapped / total > 0.85
