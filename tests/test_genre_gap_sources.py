"""Offline tests for the genre-gap round: Fender FUSE reader + mapper, the local Neural DSP harvester and its genre-evidence rules, the forum keyword
filter, John Mayer X / Misha Mansoor X name tables and numeric-label scales, and the audio write-back race."""

from __future__ import annotations

import json

import pytest

from patch_corpus.ingest import ingest
from patch_corpus.mapping.fender import fender_to_universal
from patch_corpus.mapping.neural import neural_to_vst, normalise, schema_for, tree_name
from patch_corpus.parsers.fender_fuse import is_fuse, parse_fuse
from patch_corpus.parsers.neural_xml import NeuralXmlPreset
from patch_corpus.schema import AmpArchetype, DelayKind
from patch_scraper import audio
from patch_scraper.archive import Sink, valid
from patch_scraper.artists import FamousArtists
from patch_scraper.exhaustive_scraper import evidence, harvest_local, name_genre, tag_fender
from patch_scraper.manifest import read_rows, write_rows
from patch_scraper.neural_forum import keyword_re, wanted

FUSE = """﻿<?xml version="1.0" encoding="utf-8"?>
<Preset amplifier="Mustang I/II" ProductId="1">
  <Amplifier>
    <Module ID="97" POS="0" BypassState="1">
      <Param ControlIndex="0">51143</Param><Param ControlIndex="1">257</Param><Param ControlIndex="2">32896</Param>
      <Param ControlIndex="3">41634</Param><Param ControlIndex="4">31611</Param><Param ControlIndex="5">59624</Param>
      <Param ControlIndex="6">19532</Param><Param ControlIndex="7">65535</Param><Param ControlIndex="17">1</Param>
    </Module>
  </Amplifier>
  <FX>
    <Stompbox ID="1"><Module ID="60" POS="5" BypassState="1"><Param ControlIndex="0">39321</Param></Module></Stompbox>
    <Modulation ID="2"><Module ID="0" POS="6" BypassState="1"></Module></Modulation>
    <Delay ID="3"><Module ID="69" POS="7" BypassState="1"></Module></Delay>
    <Reverb ID="4"><Module ID="33" POS="4" BypassState="0"></Module></Reverb>
  </FX>
  <FUSE><Info name="Sultan Of Swing" author="x" rating="45" genre1="40" genre2="-1" genre3="-1" tags="Dire Straits, blues rock">notes</Info></FUSE>
</Preset>"""


def x_file(tag: bytes) -> bytes:
    """Smallest thing the validator accepts as an X-series preset (a root, a 'tags' child, then 'parameters'); ``tag`` keeps files distinct."""
    return b"mayer\x00\x01\x01" + tag + b"\x00\x01\x02tags\x00\x00\x00parameters\x00\x00\x00"


# ------------------------------------------------------------------------------------------------------------------ Fender FUSE
def test_fuse_parse_and_map(tmp_path):
    f = tmp_path / "a.fuse"
    f.write_text(FUSE, encoding="utf-8")
    assert is_fuse(f.read_bytes()) and not is_fuse(b"<html></html>") and not is_fuse(b'{"a": 1}')
    p = parse_fuse(f)
    assert (p.name, p.amp_name, p.rating, p.genre_codes, p.tags) == ("Sultan Of Swing", "British '60s", 4.5, (40,), ["Dire Straits", "blues rock"])
    assert p.amp["volume"] == 51143 >> 8 and p.amp["treble"] == 31611 >> 8 and p.amp["cabinet"] == 1
    chain, warn = fender_to_universal(p)
    assert chain.amp.archetype == AmpArchetype.CRUNCH and chain.amp.treble == pytest.approx((31611 >> 8) / 255) and chain.amp.presence == 1.0
    assert chain.overdrive.enabled and chain.overdrive.source_model == "Overdrive"               # module 0x3c = 60
    assert not chain.modulation.enabled                                                           # empty slot
    assert chain.delay.enabled and chain.delay.kind == DelayKind.PING_PONG                       # module id 69 = 0x45 (ids are decimal in the file)
    assert not chain.reverb.enabled and warn                                                      # BypassState 0 = off; the INFERRED note is present


def test_fuse_ingest_dispatch_and_tags(tmp_path):
    root = tmp_path / "fender_mustang"
    root.mkdir()
    (root / "fender-mustang1-1.fuse").write_text(FUSE, encoding="utf-8")
    write_rows(root / "manifest.jsonl", [{"file": "fender-mustang1-1.fuse", "artist": "Dire Straits", "song": "Sultan Of Swing", "title": "Sultan Of Swing"}])
    recs = list(ingest(root, dedupe=False))
    assert len(recs) == 1 and recs[0].source == "fender_mustang" and recs[0].device == "Mustang I/II"
    stats = tag_fender(root, FamousArtists({"blues": ["Dire Straits"]}))
    row = read_rows(root / "manifest.jsonl")[0]
    assert stats["tagged"] == 1 and (row["genre"], row["genre_origin"], row["fuse_genre_codes"], row["rating"]) == ("blues", "famous_artist", [40], 4.5)


# --------------------------------------------------------------------------------------------------------- genre evidence rules
def test_name_genre_is_whole_word_and_narrow():
    assert name_genre("Glassy SRV Tone") == "blues" and name_genre("Nashville Nasty") == "country" and name_genre("DI Funk (MJ Style)") == "funk"
    assert name_genre("Crystal Clear Pop") == "pop" and name_genre("Chug") == "metal"
    assert name_genre("That Bluesy Lead") == "blues"                                          # a word that starts with the keyword counts
    assert name_genre("Popcorn Lead") is None and name_genre("Blue Sky") is None and name_genre("Clean Verb") is None   # a keyword inside another word does not


def test_evidence_order_and_creator_table():
    fam = FamousArtists({"funk": ["Cory Wong"], "progressive": ["Plini"]})
    assert evidence("Cory Wong", None, "Anything", famous=fam) == ("funk", "famous_artist")
    assert evidence("Joey Landreth", None, "Lead", famous=fam) == ("blues", "creator_table")
    assert evidence("Nobody Known", None, "Lead", famous=fam) == (None, None)
    assert evidence("Nobody Known", None, "Texas Blues Lead", famous=fam) == ("blues", "name_keyword")
    assert evidence(None, "Plini", "x", famous=fam) == ("progressive", "famous_artist")


def test_local_harvest_layout_tags_and_no_plugin_prior_as_genre(tmp_path):
    root = tmp_path / "Neural DSP"
    base = root / "Archetype John Mayer X"
    for rel, tag in (("Artists/Cory Wong/Tight Thing.xml", b"a"), ("Artists/Some Guy/Lead.xml", b"b"), ("Neural DSP/Glassy SRV Tone.xml", b"c"),
                     ("John Mayer/Gravity Clean.xml", b"d"), ("User/Mine.xml", b"e"), ("Default.xml", b"f"), ("Artists/Some Guy/Bass - Deep.xml", b"g")):
        (base / rel).parent.mkdir(parents=True, exist_ok=True)
        (base / rel).write_bytes(x_file(tag))
    other = root / "Archetype Nolly X"                        # an installed plugin that is not in the owned list
    other.mkdir(parents=True)
    (other / "Chug.xml").write_bytes(x_file(b"z"))
    sink = Sink(tmp_path / "nd", tmp_path / "nam", tmp_path / "work", famous=FamousArtists({"funk": ["Cory Wong"]}))
    stats = harvest_local([root, tmp_path / "does-not-exist"], sink=sink)
    assert stats["stored"] == 6 and stats["skipped_non_preset"] == 1 and stats["missing_root"] == 1
    rows = {r["title"]: r for r in read_rows(tmp_path / "nd" / "manifest.jsonl")}
    assert set(rows) == {"Tight Thing", "Lead", "Glassy SRV Tone", "Gravity Clean", "Mine", "Bass - Deep"}      # Nolly X is not an owned plugin
    assert (rows["Tight Thing"]["category"], rows["Tight Thing"]["author"], rows["Tight Thing"]["genre"], rows["Tight Thing"]["genre_origin"]) == \
        ("artist", "Cory Wong", "funk", "famous_artist")
    assert rows["Glassy SRV Tone"]["category"] == "factory" and rows["Glassy SRV Tone"]["genre"] == "blues"
    assert rows["Gravity Clean"]["category"] == "signature" and rows["Gravity Clean"]["artist"] == "John Mayer"
    assert rows["Mine"]["category"] == "user"
    assert "genre" not in rows["Lead"] and rows["Lead"]["genre_prior"][0] == "blues"        # the plugin's usual style is a prior, never the genre
    assert "artist" not in rows["Lead"] and rows["Lead"]["author"] == "Some Guy" and rows["Bass - Deep"]["instrument_hint"] == "bass"
    again = harvest_local([root], sink=Sink(tmp_path / "nd", tmp_path / "nam", tmp_path / "work"))
    assert again.get("stored", 0) == 0                                                          # byte-identical files are skipped on a re-run


def test_validator_accepts_a_child_before_parameters():
    assert valid(x_file(b"q"), ".xml") and not valid(b"mayer\x00\x01\x03nothing", ".xml")


def test_balance_counts_evidence_and_shows_priors_separately(tmp_path):
    from patch_scraper.balance import counts

    write_rows(tmp_path / "a" / "manifest.jsonl", [{"file": "1", "genre": "blues"}, {"file": "2", "genre_prior": ["blues", "pop"]}, {"file": "3"}])
    c = counts(tmp_path)
    assert c["blues"]["patches"] == 1 and c["blues"]["prior"] == 1 and c["pop"]["patches"] == 0 and c["pop"]["prior"] == 1 and c["other"]["patches"] == 2


# ---------------------------------------------------------------------------------------------------------------------- forum
def test_forum_keyword_filter_uses_title_and_tags():
    rx = keyword_re(["neo soul", "blues", "John Mayer"])
    assert wanted({"title": "Neo-Soul patches", "tags": []}, rx) and wanted({"title": "Share", "tags": ["blues"]}, rx)
    assert not wanted({"title": "Bluesy thing", "tags": []}, rx) and not wanted({"title": "Metal presets", "tags": []}, rx)
    assert wanted({"title": "Metal presets"})                                                  # the default filter still looks for preset sharing


# --------------------------------------------------------------------------------------------- John Mayer X / Misha Mansoor X mapping
def test_plugin_ids_resolve_to_their_schemas():
    assert schema_for("mayer").plugin_name == "Archetype John Mayer X" and schema_for("misha-x").plugin_name == "Archetype Misha Mansoor X"


def test_mayer_names_scales_and_amps():
    sch = schema_for("mayer")
    order, labels = [p.name for p in sch.parameters], [str(x) for x in sch.by_name()["amp_type"].valid_values]
    want = {"amp.dumbleAmp.dumbleGain": "headroom_hero_gain", "amp.vibroverbAmp.vibroverbVolume": "smooth_operator_volume",
            "amp.twoRockAmp.twoRockPresence": "signature_83_presence", "amp.threeInOneAmp.threeInOneTwoRockOutput": "three_in_one_amp_signature_83_output",
            "preFX.qTron.qTronSweep": "antelope_filter_sweep", "preFX.centaur.centaurLevel": "halfman_od_level", "preFX.overdrive.overdriveTone": "tealbreaker_tone",
            "preFX.katana.katanaBoost": "justa_boost_boost", "preFX.aquapuss.aquapussTime": "millipede_delay_time",
            "verbTrem.reverberato.reverberatoSpeed": "harmonic_tremolo_speed", "verbTrem.springReverb.springReverbDecay": "spring_reverb_decay",
            "verbTrem.sectionActive": "active_gravity_tank_section", "eq.sectionActive": "active_eq_comp_section", "postFX.delay.delayTime": "dream_delay_time",
            "postFX.reverb.reverbDecay": "studio_verb_decay", "eq.dumbleEQ.dumbleLowMidGain": "headroom_hero_eq_lo_mid_gain",
            "eq.twoRockEQ.twoRockHpfFreq": "signature_83_eq_hpf", "eq.vibroverbEQ.vibroverbCompressorInput": "smooth_operator_compressor_input",
            "cab.rightCab1MicType": "cab_headroom_hero_r_mic_type", "cab.leftCabMicLevel": "cab_l_mid_level", "cab.leftCabPan": "cab_l_mic_pan"}
    for key, name in want.items():
        assert tree_name(key, order, sch.plugin_name, labels) == name, key
    pre = NeuralXmlPreset("mayer", "t", {"amp.selectedAmp": 1.0, "amp.dumbleAmp.dumbleGain": 0.6, "preFX.aquapuss.aquapussTime": 0.5,
                                         "eq.dumbleEQ.dumbleHpfFreq": 20.0, "eq.dumbleEQ.dumbleLpfFreq": 20000.0, "eq.dumbleEQ.dumbleLowFreq": 0.4,
                                         "amp.threeInOneAmp.threeInOneRoomSend": -40.0, "postFX.reverb.reverbDecay": 4.0}, container="juce_binary_v2")
    p = neural_to_vst(pre)
    v = p.vst_parameters
    assert p.plugin == "Archetype John Mayer X" and not p.unmatched and p.labels["amp_type"] == "Headroom Hero"
    assert v["headroom_hero_gain"] == pytest.approx(0.6, abs=0.01) and v["millipede_delay_time"] == pytest.approx(0.5, abs=0.01)    # a 0-1 knob position
    hpf, lpf = sch.by_name()["headroom_hero_eq_hpf"], sch.by_name()["headroom_hero_eq_lpf"]
    assert v["headroom_hero_eq_hpf"] == pytest.approx(1 / (hpf.n_options - 1)) and v["headroom_hero_eq_lpf"] == pytest.approx((lpf.n_options - 2) / (lpf.n_options - 1))
    assert v["three_in_one_amp_room_send"] == 0.0 and v["headroom_hero_eq_lo_freq"] == pytest.approx(0.4, abs=0.01)            # OFF first; Freq as a position
    assert v["studio_verb_decay"] > 0


def test_numeric_enum_with_off_at_either_end():
    from common.schema import ParameterSpec

    low = ParameterSpec(index=0, name="hp", kind="enum", default_normalized=0.0, num_steps=5, valid_values=["OFF", "20 Hz", "21 Hz", "22 Hz", "23 Hz"])
    high = ParameterSpec(index=0, name="lp", kind="enum", default_normalized=1.0, num_steps=5, valid_values=["1.00 kHz", "1.01 kHz", "1.02 kHz", "1.03 kHz", "OFF"])
    assert normalise(20.0, low)[0] == pytest.approx(0.25) and normalise(19.0, low)[0] == 0.0 and normalise(23.0, low)[0] == 1.0
    assert normalise(1000.0, high)[0] == 0.0 and normalise(1030.0, high)[0] == pytest.approx(0.75) and normalise(1040.0, high)[0] == 1.0 and normalise(1100.0, high)[1] is not None


# --------------------------------------------------------------------------------------------------------------- audio write-back
def test_audio_writeback_keeps_rows_a_crawler_appended_meanwhile(tmp_path):
    presets, urls = tmp_path / "presets" / "katana", tmp_path / "urls"
    manifest = presets / "manifest.jsonl"
    write_rows(manifest, [{"file": "a.tsl", "artist": "Muse", "song": "Hysteria"}])

    def search(q):                                        # a crawler appends a row while the run is in progress
        if not any(r["file"] == "late.tsl" for r in read_rows(manifest)):
            write_rows(manifest, [*read_rows(manifest), {"file": "late.tsl", "artist": "Muse", "song": "Uprising"}])
        return [{"title": "Muse - Hysteria", "duration": 227, "channel": "Muse - Topic", "url": "https://yt/h"}]

    audio.match(tmp_path / "presets", urls, search=search, use_spotify=False)
    rows = {r["file"]: r for r in read_rows(manifest)}
    assert set(rows) == {"a.tsl", "late.tsl"} and rows["a.tsl"]["youtube_url"] == "https://yt/h"


def test_new_plugins_universal_view():
    from patch_corpus.mapping.neural import neural_to_universal
    from patch_corpus.schema import DelayKind, ModKind, OverdriveKind, ReverbKind

    mayer = NeuralXmlPreset("mayer", "t", {
        "amp.selectedAmp": 3.0, "amp.threeInOneAmp.threeInOneGain": 0.3, "amp.threeInOneAmp.threeInOneOutput": 0.5, "preFX.sectionActive": 1.0,
        "preFX.katana.katanaActive": 1.0, "preFX.katana.katanaOutput": 0.6, "postFX.reverb.reverbActive": 1.0, "postFX.reverb.reverbDecay": 4.0,
        "postFX.reverb.reverbMix": 0.2, "postFX.delay.delayActive": 1.0, "postFX.delay.delayTime": 400.0, "postFX.delay.delayMix": 0.3,
        "verbTrem.sectionActive": 1.0, "verbTrem.reverberato.reverberatoActive": 1.0, "verbTrem.reverberato.reverberatoSpeed": 5.0}, container="juce_binary_v2")
    chain, warn = neural_to_universal(mayer)
    assert chain.amp.archetype == AmpArchetype.UNKNOWN and chain.amp.gain == pytest.approx(0.3, abs=0.01) and any("Three-In-One" in w for w in warn)
    assert chain.overdrive.enabled and chain.overdrive.kind == OverdriveKind.BOOST and chain.overdrive.level == pytest.approx(0.6, abs=0.01)
    assert chain.modulation.enabled and chain.modulation.kind == ModKind.TREMOLO
    assert chain.delay.enabled and chain.delay.time_ms == pytest.approx(400.0, abs=3) and chain.delay.kind == DelayKind.DIGITAL
    assert chain.reverb.enabled and chain.reverb.kind == ReverbKind.HALL and chain.reverb.decay is not None            # 4.0 s, kept in seconds
    misha = NeuralXmlPreset("misha-x", "t", {"amp.selectedAmp": 2.0, "amp.leadAmp.leadGain": 0.7, "postFX.sectionActive": 1.0,
                                             "postFX.reverb.reverbActive": 1.0, "postFX.reverb.reverbDecay": 0.5, "postFX.modulation.modulationActive": 1.0,
                                             "postFX.modulation.modulationMode": 1.0}, container="juce_binary_v2")
    chain, _ = neural_to_universal(misha)
    assert chain.amp.archetype == AmpArchetype.LEAD and chain.amp.gain == pytest.approx(0.7, abs=0.01)
    assert chain.reverb.enabled and chain.reverb.decay is None                                                         # a % knob: no seconds to report
    assert chain.modulation.enabled and chain.modulation.kind == ModKind.FLANGER


def test_plugin_thread_genre_is_demoted_to_a_prior(tmp_path):
    from patch_scraper.exhaustive_scraper import tag_neuraldsp

    write_rows(tmp_path / "manifest.jsonl", [
        {"file": "a.xml", "source": "neuraldsp_forum", "title": "Archetype: Plini Presets Thread", "artist": "Plini", "famous_artist": "Plini", "genre": "progressive"},
        {"file": "b.xml", "source": "neuraldsp_forum", "title": "Andy Timmons Preset", "artist": "Andy Timmons", "famous_artist": "Andy Timmons", "genre": "fusion"},
        {"file": "c.xml", "source": "neuraldsp_forum", "title": "Texas blues pack", "tags": []}])
    stats = tag_neuraldsp(tmp_path, FamousArtists({"progressive": ["Plini"], "fusion": ["Andy Timmons"]}))
    a, b, c = read_rows(tmp_path / "manifest.jsonl")
    assert stats["demoted_plugin_thread"] == 1 and "genre" not in a and a["genre_prior"] == ["progressive"] and a["plugin_artist"] == "Plini" and "artist" not in a
    assert b["genre"] == "fusion" and b["artist"] == "Andy Timmons"                       # a thread about one artist's sound keeps its evidence
    assert (c["genre"], c["genre_origin"]) == ("blues", "name_keyword")
