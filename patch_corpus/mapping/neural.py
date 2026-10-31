"""Neural DSP preset (``parsers/neural_xml.py``)  ->  normalised plugin parameters  +  UniversalChain.

Two outputs, same interface style as ``mapping/katana.py``:

  neural_to_vst(preset)        {schema parameter name: 0-1 value} for the plugin the preset was saved from. These are EXACT labels (the
                               preset is that plugin's own state), so ``ingest.export_presets`` writes them directly instead of translating
                               the universal chain.
  neural_to_universal(preset)  (UniversalChain, warnings): the device-agnostic view, so Neural DSP presets train / evaluate alongside the
                               Katana and Boss GT patches.

Scaling. A preset stores raw display values (``cab1Level = -12.0`` dB, ``reverbTime = 4709`` ms, ``ampType = 2`` = third option). Each one is
rescaled with its ``ParameterSpec`` from the plugin schema (``config/plugin_schemas/*.json``, written by the pedalboard / VST3 inspector):

  enum / boolean   index / (n_options - 1)
  numeric, curve   inverse of the schema's measured (normalised, display) curve, piecewise linear
  numeric, linear  (value - min) / (max - min), clipped to [0, 1] (clipping is reported)

Without a schema for the plugin (e.g. ``gojira-X``: no "Archetype Gojira X" schema yet) only on/off switches and values already inside
[0, 1] are kept, and the record carries a warning; drop a schema JSON into ``config/plugin_schemas`` (or ``schema_dir``) to fix that.

Names. The preset's internal ids differ from the schema's host names. ``NAME_TABLES`` holds explicit tables (Archetype Gojira: written from
a real preset and the schema; ``cleanAmpVolume -> cln_amp_gain`` and ``*Output -> *_level`` are inferred by elimination and by range, see
the table). Other plugins fall back to ``auto_name``: camelCase -> snake_case with ``TOKEN_ALIASES``; unmatched ids are listed in the
warnings rather than guessed. Name mappings are checked against the schema, not against the plugin UI.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

from common.schema import ParameterSpec, PluginSchema, list_schema_files, load_schema
from patch_corpus.parsers.neural_xml import NeuralXmlPreset
from patch_corpus.schema import (AmpArchetype, AmpBlock, DelayBlock, DelayKind, ModKind, ModulationBlock, OverdriveBlock, OverdriveKind,
                                 ReverbBlock, ReverbKind, UniversalChain, seconds_to_decay)

PLUGIN_IDS = {"neural_dsp_gojira": "Archetype Gojira", "neural_dsp_plini": "Archetype Plini", "neural_dsp_abasi": "Archetype Abasi",
              # X-series ids whose words do not spell the plugin name ('mayer', 'misha-x'), read from the files shipped with the plugins
              "mayer": "Archetype John Mayer X", "misha-x": "Archetype Misha Mansoor X"}
STOP_TOKENS = {"neural", "dsp", "archetype", "plugin", "suite"}
TOKEN_ALIASES = {"delay": "dly", "reverb": "rev", "octaver": "oct", "whammy": "wow", "drywet": "dry_wet"}
# word-level synonyms tried after the plain spelling (internal id word -> schema word)
SYNONYMS = {"hpf": "low_cut", "lpf": "high_cut", "thres": "threshold", "sutation": "saturation", "output": "level", "modulation": "mod_on",
            "time": "decay", "mids": "mid", "middle": "mid", "comp": "compressor", "od": "overdrive"}
# per-plugin word renames (pedal brand names in the host names)
PLUGIN_ALIASES: dict[str, dict[str, str]] = {"Archetype Abasi": {"comp": "logos", "drive": "pathos", "bloom": "compression"}}
DROP_WORDS = {"pedal", "id", "section"}


def _gojira_table() -> dict[str, str]:
    t = {"inputGain": "input_gain", "outputGain": "output_gain", "gateThreshold": "gate_amount", "pitchActive": "pitch_section_active",
         "pedalsActive": "pedal_section_active", "ampsActive": "amp_section_active", "eqActive": "eq_section_active",
         "cabActive": "cab_section_active", "fxActive": "fx_section_active", "ampType": "amp_type", "cabType": "cab_type",
         "cabAmpLinked": "cab_amp_linked",
         "whammyActive": "wow_active", "whammyType": "wow_type", "whammyPosition": "wow_position", "whammyMix": "wow_dry_wet",
         "octaverActive": "oct_active", "octaverOct1Level": "oct_oct_1_level", "octaverOct2Level": "oct_oct_2_level",
         "octaverDirectLevel": "oct_direct_level",
         # pedal slots: 2 = overdrive, 3 = dirt/distortion, 4 = phaser, 5 = chorus (matched on the knob names each slot carries)
         "pedal2Active": "od_active", "pedal2Overdrive": "od_dist", "pedal2Tone": "od_tone", "pedal2Level": "od_level",
         "pedal3Active": "drt_active", "pedal3Distortion": "drt_dist", "pedal3Filter": "drt_filter", "pedal3Volume": "drt_vol",
         "pedal4Active": "phsr_active", "pedal4LfoRate": "phsr_rate",
         "pedal5Active": "chr_active", "pedal5Rate": "chr_rate", "pedal5Depth": "chr_depth", "pedal5Feedback": "chr_feedback", "pedal5Mix": "chr_mix",
         # CLN amp: the knob the schema calls gain is stored as Volume (by elimination); Output is the -12..12 dB level (range matches)
         "cleanAmpVolume": "cln_amp_gain", "cleanAmpBright": "cln_amp_bright", "cleanAmpBass": "cln_amp_bass", "cleanAmpMid": "cln_amp_mid",
         "cleanAmpTreble": "cln_amp_treble", "cleanAmpOutput": "cln_amp_level"}
    for internal, host in (("rhythm", "rust"), ("lead", "hot")):          # RUST = rhythm amp, HOT = lead amp (same knob sets)
        t.update({f"{internal}AmpGain": f"{host}_amp_gain", f"{internal}AmpBass": f"{host}_amp_low", f"{internal}AmpMid": f"{host}_amp_mid",
                  f"{internal}AmpTreble": f"{host}_amp_high", f"{internal}AmpVolume": f"{host}_amp_master",
                  f"{internal}AmpPresence": f"{host}_amp_presence", f"{internal}AmpResonance": f"{host}_amp_depth",
                  f"{internal}AmpOutput": f"{host}_amp_level"})
    for internal, host in (("clean", "cln"), ("rhythm", "rust"), ("lead", "hot")):
        t[f"{internal}EqActive"] = f"{host}_eq_active"
        t.update({f"{internal}EqBand{n}": f"{host}_eq_band_{n}_db" for n in range(1, 10)})
    for n in (1, 2):
        t.update({f"cab{n}{k}": f"cab_{n}_{v}" for k, v in (("Active", "active"), ("Position", "position"), ("Distance", "distance"),
                                                             ("Level", "level"), ("Pan", "pan"), ("Phase", "phase"), ("MicIR", "mic_ir"))})
    t.update({f"delay{k}": f"dly_{v}" for k, v in (("Active", "active"), ("SyncActive", "sync_active"), ("PingPong", "ping_pong"),
                                                    ("ModActive", "mod_active"), ("DryWet", "dry_wet"), ("Feedback", "feedback"),
                                                    ("TapeSaturation", "tape_saturation"), ("Tempo", "tempo"), ("SyncNote", "sync_note"),
                                                    ("Tone", "tone"), ("TapTempo", "tap_tempo"))})
    t.update({f"reverb{k}": f"rev_{v}" for k, v in (("Active", "active"), ("Mode", "mode"), ("DryWet", "dry_wet"), ("Time", "time"),
                                                     ("LowCut", "low_cut"), ("HighCut", "high_cut"))})
    return t


NAME_TABLES: dict[str, dict[str, str]] = {
    "Archetype Gojira": _gojira_table(),
    # partial: the rest of Abasi is matched automatically (with PLUGIN_ALIASES); Pathos' drive knob is stored as driveGain
    "Archetype Abasi": {"driveGain": "pathos_drive", "gateThreshold": "gate"},
}


def _tokens(s: str) -> list[str]:
    s = re.sub(r"(?<=[a-z])(?=[A-Z0-9])|(?<=[0-9])(?=[A-Za-z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", s)      # cab1MicIR, EQBand0
    return [t for t in re.split(r"[^a-z0-9]+", s.lower()) if t]


def snake(s: str) -> str:
    return "_".join(_tokens(s))


def _active_forms(words: list[str]) -> list[str]:
    """'<x> active' in any word order -> every way schemas spell a module switch."""
    rest = [w for w in words if w not in ("active", "section")]
    out = []
    for r in ("_".join(rest), "_".join(rest).rstrip("s"), "_".join(rest) + "s"):
        if r:
            out += [f"{r}_active", f"active_{r}", f"{r}_section_active", f"active_{r}_section"]
    return out


def auto_name(internal: str, names, aliases: dict[str, str] | None = None, zero_based_bands: bool = True) -> str | None:
    """camelCase preset id -> schema name, without a table. Tries, in order: the plain snake form; ``TOKEN_ALIASES`` and the plugin's
    ``aliases``; with ``pedal`` / ``ID`` / ``section`` dropped; ``SYNONYMS`` per word; module switches in any spelling; and
    ``<amp>EQBand<n>`` -> the n-th ``<amp>_eq_*`` band of the schema (bands are stored by position, the schema names them by frequency)."""
    order = list(names)
    names = set(order)
    raw = _tokens(re.sub(r"ID$", "", internal).replace("overDrive", "Overdrive"))
    variants: list[list[str]] = []
    for words in (raw, [TOKEN_ALIASES.get(w, w) for w in raw], [(aliases or {}).get(w, TOKEN_ALIASES.get(w, w)) for w in raw]):
        variants.append(words)
        variants.append([w for w in words if w not in DROP_WORDS])
    for words in list(variants):
        variants.append([SYNONYMS.get(w, w) for w in words])
    for words in variants:
        if not words:
            continue
        cand = "_".join(words)
        if cand in names:
            return cand
        if "active" in words:
            hit = next((c for c in _active_forms(words) if c in names), None)
            if hit:
                return hit
        if "eq" in words and "band" in words and words[-1].isdigit():
            prefix = "_".join(words[: words.index("eq")])
            bands = [n for n in order if n.startswith(f"{prefix}_eq_") and not n.endswith("_active")]
            i = int(words[-1]) - (0 if zero_based_bands else 1)
            if 0 <= i < len(bands):
                return bands[i]
    return None


_LABEL_NUMBER = re.compile(r"^([+-]?\d+(?:\.\d+)?)\s*([A-Za-z%]*)$")


@lru_cache(maxsize=None)
def _numeric_labels(labels: tuple[str, ...]) -> tuple[tuple[float, ...], str, int, int] | None:
    """An enum whose options are numbers with a unit (``-12 st``, ``40.0 BPM``, ``1.00s``, ``50 L`` ... ``C`` ... ``50 R``, ``6.12 kHz``) is a dense
    numeric dropdown, not a list of choices: the preset stores the displayed number, not the option index. An ``OFF`` first (high-pass filters, room
    sends) or last (low-pass filters) is allowed. kHz labels are returned in Hz. Returns (values without OFF, unit, leading OFF, trailing OFF) or None."""
    vals, unit, off, off_last = [], "", 0, 0
    for i, lab in enumerate(labels):
        s = lab.strip()
        side = re.match(r"^(\d+(?:\.\d+)?)\s*([LR])$", s)
        if s == "C":
            vals.append(0.0)
        elif side:
            vals.append(-float(side.group(1)) if side.group(2) == "L" else float(side.group(1)))
        elif s.upper() == "OFF" and i in (0, len(labels) - 1):
            off, off_last = (off + 1, off_last) if i == 0 else (off, off_last + 1)
        elif m := _LABEL_NUMBER.match(s):
            u = m.group(2).lower()
            vals.append(float(m.group(1)) * (1000.0 if u == "khz" else 1.0))
            unit = unit or ("hz" if u == "khz" else u)
        else:
            return None
    return (tuple(vals), unit, off, off_last) if len(vals) > 2 and all(a <= b for a, b in zip(vals, vals[1:])) else None


def _numeric_position(value: float, spec: ParameterSpec) -> tuple[float, str | None] | None:
    """(0-1 position, problem) of a displayed number on a numeric-label enum; None when the enum is not one. Decay labels in seconds accept the
    milliseconds the presets store (``reverbDecay = 4225`` for ``4.22s``), ``%`` labels the 0-1 fraction (``0.14`` for ``14 %``), and a value just
    under the first number of a list that starts with ``OFF`` selects OFF (``room send = -40`` for ``OFF, -39.8 dB ...``)."""
    nums = _numeric_labels(tuple(str(x) for x in spec.valid_values)) if spec.valid_values else None
    if nums is None:
        return None
    vals, unit, off, off_last = nums
    n_all = len(vals) + off + off_last
    lo, hi = vals[0], vals[-1]
    tol = (hi - lo) / (len(vals) - 1)
    reach = max(0.01 * (hi - lo), 1.5 * tol)                 # how far past the first / last number still means OFF
    if off and lo - reach <= value < lo - 0.5 * tol:
        return 0.0, None
    if off_last and hi + 0.5 * tol < value <= hi + reach:
        return 1.0, None
    scales = (1.0, 0.001) if unit == "s" else (100.0, 1.0) if unit == "%" and hi >= 50 and 0.0 <= value <= 1.0 else (1.0,)
    problem = None
    scale = next((s for s in scales if lo - tol <= value * s <= hi + tol), None)
    if scale is None:
        scale, problem = 1.0, f"{value} clipped to [{lo}, {hi}]"
    target = min(hi, max(lo, value * scale))
    idx = min(range(len(vals)), key=lambda i: abs(vals[i] - target)) + off
    return idx / (n_all - 1), problem


def fits(value: float, spec: ParameterSpec) -> bool:
    """Plausibility check for an automatic name match: the stored value must be a valid index / lie inside the schema's range."""
    pos = _numeric_position(value, spec) if spec.kind == "enum" else None
    if pos is not None:
        return pos[1] is None
    if spec.kind in ("boolean", "enum"):
        return abs(value - round(value)) < 1e-6 and 0 <= round(value) < max(spec.n_options, 1)
    if spec.min_value is None or spec.max_value is None:
        return 0.0 <= value <= 1.0
    pad = 0.02 * (spec.max_value - spec.min_value)
    return spec.min_value - pad <= value <= spec.max_value + pad


# ----------------------------------------------------------------------------------------- X-series (hierarchical keys)
# ``neural_xml`` keys X-series values by their model path: ``amp.leadAmp.leadMaster``, ``eq.leadEQ.leadEQBand4Gain``, ``cab.rightCabPan``,
# ``eq.sectionActive``. The path decides the schema name, so the repeated ids (sectionActive, selectedAmp ...) no longer collide.
TREE_SPECIAL = {"app.ampCabLinkedState": "amp_cab_linked", "amp.selectedAmp": "amp_type", "cab.selectedCab": "cab_type_unlinked"}
# model snake name -> schema stem, per plugin (Nolly names its two delays by position in the chain, Petrucci has one)
TREE_MODELS: dict[str, dict[str, str]] = {
    "Archetype Nolly X": {"pre_delay": "delay_1", "post_delay": "delay_2"},
    # John Mayer X files use the original amp / pedal names, the plugin (and its schema) the host names. Matched by knob set, e.g. vibroverbAmp
    # (volume, bass, treble, bright, output) = Smooth Operator; dumbleAmp (input jack, mid boost, gain ... master) = Headroom Hero; twoRockAmp
    # (gain, EQ, presence ...) = Signature 83; qTron (gain, range, sweep, peak, response) = Antelope Filter (the envelope filter); centaur
    # (gain, treble, level) = Halfman OD; overdrive (gain, tone, level, mode) = Tealbreaker; katana (boost, output) = Justa Boost;
    # aquapuss (time, feedback, blend) = Millipede Delay; reverberato (speed, intensity, sync) = Harmonic Tremolo
    "Archetype John Mayer X": {
        "dumble_amp": "headroom_hero", "vibroverb_amp": "smooth_operator", "two_rock_amp": "signature_83",
        "dumble_eq": "headroom_hero_eq", "vibroverb_eq": "smooth_operator_eq", "two_rock_eq": "signature_83_eq", "three_in_one_eq": "three_in_one_eq",
        "q_tron": "antelope_filter", "centaur": "halfman_od", "overdrive": "tealbreaker", "katana": "justa_boost", "aquapuss": "millipede_delay",
        "reverberato": "harmonic_tremolo", "delay": "dream_delay", "reverb": "studio_verb"},
    # Misha Mansoor X: pre-FX slots are named by position in the file (preDelay = the Tape Echo, dist = Chaos, drive = Horizon Precision Drive ...)
    "Archetype Misha Mansoor X": {
        "drive": "horizon_devices_precision_drive", "pre_delay": "tape_echo", "dist": "chaos", "octaver": "dual_octaver",
        "comp": "horizon_devices_compressor", "delay": "stereo_delay", "modulation": "modulator"},
}
# '<section>.sectionActive' whose schema switch is named differently
TREE_SECTIONS: dict[str, dict[str, str]] = {"Archetype John Mayer X": {"eq": "active_eq_comp_section", "verbTrem": "active_gravity_tank_section"}}
# phrases inside a knob name that the schema spells differently (the Three-In-One amp lists the other amps' outputs by host name)
TREE_PHRASES: dict[str, list[tuple[tuple[str, ...], tuple[str, ...]]]] = {
    "Archetype John Mayer X": [(("two", "rock"), ("signature", "83")), (("dumble",), ("headroom", "hero")), (("vibroverb",), ("smooth", "operator"))]}
# single keys whose schema name cannot be derived (a typo in the Mayer X schema, renamed knobs)
TREE_EXPLICIT: dict[str, dict[str, str]] = {
    "Archetype John Mayer X": {"cab.leftCabMicLevel": "cab_l_mid_level", "postFX.delay.delayEchoHardness": "dream_delay_hardness"},
    "Archetype Misha Mansoor X": {
        "specialFX.glitch.glitchLength": "glitch_grain_length", "specialFX.glitch.glitchOccurrence": "glitch_grain_occurrence",
        "specialFX.glitch.glitchPitch": "glitch_grain_pitch_amount", "specialFX.laser.laserSpeed": "laser_envelope_speed",
        "specialFX.laser.laserSensitivity": "laser_envelope_sensitivity", "specialFX.laser.laserPedal": "laser_expression_pedal",
        "postFX.modulation.modulationPedal": "modulator_expression_pedal", "postFX.modulation.modulationMixHeel": "modulator_expression_mix_heel",
        "postFX.modulation.modulationMixToe": "modulator_expression_mix_toe", "postFX.modulation.modulationDepthHeel": "modulator_expression_depth_heel",
        "postFX.modulation.modulationDepthToe": "modulator_expression_depth_toe", "postFX.modulation.modulationRateHeel": "modulator_expression_rate_heel",
        "postFX.modulation.modulationRateToe": "modulator_expression_rate_toe", "postFX.modulation.modulationSpeed": "modulator_auto_speed",
        "postFX.modulation.modulationRandom": "modulator_auto_random", "postFX.modulation.modulationAuto": "modulator_auto"}}
KNOB_SYN = {"mid": ("middle",), "middle": ("mid",), "level": ("output", "send"), "output": ("level",), "hpf": ("high_pass",), "lpf": ("low_pass",),
            "modulation": ("mod",), "mod": ("modulation",), "drive": ("gain",), "gain": ("drive",), "dist": ("drive",), "type": ("mode",),
            "low": ("lo",), "high": ("hi",), "lpf": ("low_pass", "lpf"), "freq": ("",)}
KNOB_JOIN = {("sync", "note"): ("syncnote", "note"), ("note",): ("sync_note", "syncnote"), ("mid", "point"): ("mid_point",)}
SIDES = {"left": "l", "right": "r"}


def _variants(words: list[str]) -> list[list[str]]:
    """The word list, then each single synonym substitution and each phrase rewrite (``sync note`` -> ``syncnote`` ...)."""
    out = [words]
    for i, w in enumerate(words):
        for alt in KNOB_SYN.get(w, ()):
            out.append(words[:i] + [alt] + words[i + 1:])
    for phrase, alts in KNOB_JOIN.items():
        n = len(phrase)
        for i in range(len(words) - n + 1):
            if tuple(words[i:i + n]) == phrase:
                out += [words[:i] + [a] + words[i + n:] for a in alts]
    return out


def _amp_order(order: list[str], labels: list[str] | None) -> list[str]:
    """Amp prefixes in the plugin's own order (the option labels of ``amp_type``, else first appearance of ``<amp>_amp_*`` in the schema)."""
    names = [m.group(1) for n in order if (m := re.match(r"^([a-z]+)_amp_(?:gain|bass|mid|middle|treble|master|output)$", n))]
    seen = list(dict.fromkeys(names))
    words = [w for lab in (labels or []) for w in _tokens(lab) if w in seen]
    if words:
        return list(dict.fromkeys(words))
    return [snake(lab) for lab in labels] if labels else seen          # 'Smooth Operator' -> smooth_operator (Mayer X)


def tree_name(key: str, order: list[str], plugin: str = "", amp_labels: list[str] | None = None) -> str | None:
    """X-series ``key`` (``neural_xml.tree_key``) -> schema name, or None. Rules, all checked against ``order`` (the schema's names):
    explicit selectors; ``<section>.sectionActive`` -> ``active_<section>_section``; ``<section>.<model>.<model><Knob>`` -> ``<model>_<knob>``
    (synonyms, ``EQ`` models in either ``lead_eq`` / ``lead_amp_eq`` spelling, 4-band EQ ``BandNGain/Freq`` by position); the cab section's
    ``<left|right><Cab|Room><Knob>`` -> ``cab_l_pan`` / ``room_r_level``, and ``<side>Cab<N>MicType`` -> the N-th amp's mic; bare ids -> ``auto_name``."""
    names = set(order)
    if key in TREE_SPECIAL:
        return TREE_SPECIAL[key] if TREE_SPECIAL[key] in names else None
    if key in TREE_EXPLICIT.get(plugin, {}):
        return TREE_EXPLICIT[plugin][key] if TREE_EXPLICIT[plugin][key] in names else None
    *chain, prop = key.split(".")
    ptok = _tokens(prop)
    if not chain:
        return auto_name(prop, order)
    section = snake(chain[0])
    if prop == "sectionActive":
        own = TREE_SECTIONS.get(plugin, {}).get(chain[0])
        return next((c for c in (own, f"active_{section}_section", f"{section}_section_active", f"active_{section}") if c and c in names), None)
    if len(chain) == 1:                                        # the cab section keeps its knobs directly under it
        return _cab_name(ptok, order, amp_labels)
    model = _tokens(chain[1])
    k = 0
    while k < min(len(model), len(ptok)) and model[k] == ptok[k]:
        k += 1
    knob = ptok[k:] or ["active"]
    for phrase, repl in TREE_PHRASES.get(plugin, []):
        for i in range(len(knob) - len(phrase) + 1):
            if tuple(knob[i:i + len(phrase)]) == phrase:
                knob = [*knob[:i], *repl, *knob[i + len(phrase):]]
                break
    stems = [model]
    if "eq" in model:
        stems.append([*model[:model.index("eq")], "amp", *model[model.index("eq"):]])          # lead_eq vs lead_amp_eq
    alias = TREE_MODELS.get(plugin, {}).get("_".join(model))
    if alias:
        stems.insert(0, alias.split("_"))
    if knob[0] == "compressor":                                # '<amp>_eq' models also carry that amp's compressor: <amp>_compressor_*
        stems += [s[:-1] for s in stems if s[-1] == "eq"]
    if knob[0] == "band" and len(knob) in (2, 3) and knob[1].isdigit():                        # EQ band by number: band_3 / 3rd of 4 (gain|freq)
        for stem in stems:
            pre = "_".join(stem)
            if len(knob) == 2 and f"{pre}_band_{knob[1]}" in names:
                return f"{pre}_band_{knob[1]}"
            if len(knob) == 3:
                bands = [n for n in order if n.startswith(pre + "_") and n.endswith("_" + knob[2]) and "pass" not in n and n != f"{pre}_{knob[2]}"]
                if len(bands) >= int(knob[1]):
                    return bands[int(knob[1]) - 1]
        return None
    for stem in stems:
        for kn in _variants(knob):
            cand = "_".join([*stem, *[w for w in kn if w]])
            if cand in names:
                return cand
    return None


def _cab_name(ptok: list[str], order: list[str], amp_labels: list[str] | None) -> str | None:
    names = set(order)
    side = next((SIDES[t] for t in ptok if t in SIDES), None)
    grp = next((t for t in ptok if t in ("cab", "room")), None)
    num = next((int(t) for t in ptok if t.isdigit()), None)
    rest = [t for t in ptok if t not in SIDES and t not in ("cab", "room") and not t.isdigit()]
    if rest[:2] == ["mic", "type"] and side:
        if num is not None:
            amps = _amp_order(order, amp_labels)
            if num >= len(amps):
                return None
            return next((c for c in (f"{amps[num]}_amp_cab_mic_{side}_type", f"cab_{amps[num]}_{side}_mic_type") if c in names), None)
        return next((c for c in (f"cab_mic_{side}_type",) if c in names), None)
    if rest[:1] == ["mic"]:
        rest = rest[1:]                                       # rightCabMicLevel -> cab_r_level, rightRoomMicDistance -> room_r_distance
    if grp is None or side is None or not rest:
        return None
    for kn in _variants(rest):
        for cand in ("_".join([grp, side, *kn]), "_".join([grp, side, "mic", *kn])):        # cab_l_pan (Nolly) / cab_l_mic_pan (Mayer)
            if cand in names:
                return cand
    return None


def _stored_normalised(key: str, value: float, spec: ParameterSpec) -> bool:
    """X-series presets store ``%`` and unit-less knobs as the 0-1 knob position, and dB / Hz / ms values as the displayed number. Evidence: a
    Nolly X preset at its defaults stores ``reverbMix = 0.288978``, ``reverbLowCut = 0.423153`` = the schema's ``default_normalized`` (0.289, 0.423),
    and a 0-90 % feedback with default 0.16 displays "14 %". Petrucci's EQ ``Band<N>Freq`` is the one Hz parameter stored this way (0.613 = the
    default of 78 Hz). Without this, ``rhythmAmpGain = 0.44`` would be divided by the display range and come out as 0.004."""
    if spec.kind != "numeric" or not 0.0 <= value <= 1.0:
        return False
    return (spec.units or "") in ("%", "") or (key.endswith("Freq") and (spec.min_value or 0) >= 1) or (spec.min_value or 0) > 1.0


def tree_snake(key: str) -> str:
    """Schema-free name for an X-series key: ``<model>_<knob>`` without the repeated model prefix (``reverb_decay``, ``lead_amp_gain``)."""
    *chain, prop = key.split(".")
    if key in TREE_SPECIAL:
        return TREE_SPECIAL[key]
    if prop == "sectionActive":
        return f"active_{snake(chain[0])}_section"
    ptok = _tokens(prop)
    if len(chain) < 2:
        return "_".join(ptok)
    model = _tokens(chain[1])
    k = 0
    while k < min(len(model), len(ptok)) and model[k] == ptok[k]:
        k += 1
    return "_".join([*model, *(ptok[k:] or ["active"])])


def schema_for(plugin_id: str, schema_dir: Path | None = None) -> PluginSchema | None:
    """Schema of the plugin a preset was saved from: ``PLUGIN_IDS``, else equal word sets (``gojira-X`` must not match 'Archetype Gojira')."""
    if plugin_id in PLUGIN_IDS:
        try:
            return load_schema(PLUGIN_IDS[plugin_id], schema_dir)
        except (KeyError, FileNotFoundError):
            return None
    want = set(_tokens(plugin_id)) - STOP_TOKENS
    for f in list_schema_files(schema_dir):
        if want and set(_tokens(f.stem)) - STOP_TOKENS == want:
            return load_schema(f)
    return None


def normalise(value: float, spec: ParameterSpec) -> tuple[float | None, str | None]:
    """Raw display value -> 0-1 position on the plugin's own scale. Returns (value, problem)."""
    pos = _numeric_position(value, spec) if spec.kind == "enum" else None
    if pos is not None:
        return pos
    if spec.kind in ("boolean", "enum"):
        n = spec.n_options
        idx = int(round(value))
        if n < 2:
            return None, "enum without options"
        if not 0 <= idx < n:
            return None, f"index {value} outside 0..{n - 1}"
        return idx / (n - 1), None
    if spec.curve and len(spec.curve) >= 2 and not spec.linear:
        pts = sorted(spec.curve, key=lambda p: p[1])
        lo, hi = pts[0][1], pts[-1][1]
        if value <= lo or value >= hi:
            return (pts[0][0] if value <= lo else pts[-1][0]), (None if lo <= value <= hi else f"{value} clipped to [{lo}, {hi}]")
        for (n0, d0), (n1, d1) in zip(pts, pts[1:]):
            if d0 <= value <= d1:
                return (n0 if d1 == d0 else n0 + (n1 - n0) * (value - d0) / (d1 - d0)), None
    if spec.min_value is None or spec.max_value is None or spec.max_value == spec.min_value:
        return (value, None) if 0.0 <= value <= 1.0 else (None, "no range in schema")
    x = (value - spec.min_value) / (spec.max_value - spec.min_value)
    if x < -1e-6 or x > 1 + 1e-6:
        return min(1.0, max(0.0, x)), f"{value} clipped to [{spec.min_value}, {spec.max_value}]"
    return min(1.0, max(0.0, x)), None


@dataclass
class NeuralParams:
    plugin: str | None                                       # schema name, None when no schema matched
    vst_parameters: dict[str, float] = field(default_factory=dict)   # schema name -> 0-1
    raw: dict[str, float] = field(default_factory=dict)      # schema (or snake_case) name -> raw display value
    unmatched: list[str] = field(default_factory=list)
    order: list[str] = field(default_factory=list)          # schema parameter order (amp knob groups are read in this order)
    labels: dict[str, str] = field(default_factory=dict)    # enum / switch name -> option label of the stored index
    units: dict[str, str] = field(default_factory=dict)     # parameter -> schema unit, lower case ('%', 'ms', 'hz' ...), when it has one
    seconds: dict[str, float] = field(default_factory=dict) # parameters labelled in seconds -> the stored value in seconds (some presets keep ms)
    warnings: list[str] = field(default_factory=list)

    @property
    def coverage(self) -> float:
        """Share of the preset's stored numeric values that landed on a schema parameter."""
        n = len(self.raw) + len(self.unmatched)
        return len(self.raw) / n if n else 0.0


def neural_to_vst(preset: NeuralXmlPreset, schema: PluginSchema | None = None, schema_dir: Path | None = None) -> NeuralParams:
    schema = schema or schema_for(preset.plugin, schema_dir)
    if schema is None:
        out = NeuralParams(None, warnings=[f"no plugin schema for {preset.plugin!r}: only switches and 0-1 values kept, not rescaled"])
        for k, v in preset.params.items():
            name = tree_snake(k) if preset.container == "juce_binary_v2" else snake(re.sub(r"ID$", "", k))
            out.raw[name] = v
            out.order.append(name)
            if 0.0 <= v <= 1.0:
                out.vst_parameters[name] = v
        return out
    specs = schema.by_name()
    order = [p.name for p in schema.parameters]
    table = NAME_TABLES.get(schema.plugin_name, {})
    aliases = PLUGIN_ALIASES.get(schema.plugin_name, {})
    zero_based = any(re.search(r"Band0(ID)?$", k) for k in preset.params)
    tree = preset.container == "juce_binary_v2"
    amp_labels = [str(x) for x in (specs["amp_type"].valid_values or [])] if "amp_type" in specs else None
    out = NeuralParams(schema.plugin_name, order=order)
    clipped, rejected = [], []
    taken: set[str] = set()
    normalised_keys: set[str] = set()
    for k, v in preset.params.items():
        name = table.get(k)
        if name is None:
            name = tree_name(k, order, schema.plugin_name, amp_labels) if tree else auto_name(k, order, aliases, zero_based)
            if name is None and tree and (m := re.fullmatch(r"cab\.(left|right)Cab(\d)MicType", k)):
                # one mic per amp in the file, one 'cab_mic_<side>_type' in the schema: it holds the selected cab's mic
                sel = preset.params.get("cab.selectedCab")
                if sel is not None and int(sel) == int(m.group(2)) and f"cab_mic_{SIDES[m.group(1)]}_type" in specs:
                    name = f"cab_mic_{SIDES[m.group(1)]}_type"
            if name is not None and tree and _stored_normalised(k, v, specs[name]):
                normalised_keys.add(name)
            elif name is not None and (name in taken or not fits(v, specs[name])):
                rejected.append(f"{k}->{name}")                # an automatic guess whose value does not fit is not used
                name = None
        if name is None or name not in specs:
            out.unmatched.append(k)
            continue
        taken.add(name)
        out.raw[name] = v
        if specs[name].units:
            out.units[name] = specs[name].units.lower()
        if specs[name].valid_values:
            out.labels[name] = specs[name].label(normalise(v, specs[name])[0] or 0.0) or ""
            nums = _numeric_labels(tuple(str(x) for x in specs[name].valid_values))
            if nums and nums[1] == "s":                           # Mayer X keeps seconds, Nolly X milliseconds, both under a "1.00 s" label
                out.seconds[name] = v if v <= nums[0][-1] * 1.05 else v / 1000.0
        x, problem = (v, None) if name in normalised_keys else normalise(v, specs[name])
        if problem:
            clipped.append(f"{k}: {problem}")
        if x is not None:
            out.vst_parameters[name] = specs[name].snap(x)
    if normalised_keys:
        out.warnings.append(f"{len(normalised_keys)} X-series values (%, unit-less knobs, EQ Freq) are stored as 0-1 knob positions and used as is")
    if out.unmatched:
        out.warnings.append(f"{len(out.unmatched)} preset ids not mapped to {schema.plugin_name}: {', '.join(out.unmatched[:12])}"
                            + (" ..." if len(out.unmatched) > 12 else ""))
    if clipped:
        out.warnings.append("out of range: " + "; ".join(clipped[:8]))
    if rejected:
        out.warnings.append("automatic matches rejected (value outside the parameter's range, or name already used): " + ", ".join(rejected[:8]))
    if len(table) < len(preset.params) // 2:
        out.warnings.append(f"{schema.plugin_name}: names mostly matched automatically (rules + range check), check before trusting")
    return out


# ---------------------------------------------------------------------------------------------------- universal view
_AMP_KNOB = re.compile(r"^(?P<p>[a-z0-9_]+?)(?:_amp)?_(?P<k>gain|volume|bass|low|mid|middle|treble|high|presence|master)$")
AMP_WORDS = ("cln", "clean", "crunch", "rust", "rhythm", "hot", "lead", "piezo", "acoustic", "smooth_operator", "headroom_hero", "signature_83")
ARCHETYPE_WORDS = [("operator", AmpArchetype.CLEAN), ("hero", AmpArchetype.CRUNCH), ("83", AmpArchetype.CRUNCH), ("piezo", AmpArchetype.ACOUSTIC), ("acoustic", AmpArchetype.ACOUSTIC), ("cln", AmpArchetype.CLEAN), ("clean", AmpArchetype.CLEAN),
                   ("crunch", AmpArchetype.CRUNCH), ("rust", AmpArchetype.CRUNCH), ("rhythm", AmpArchetype.HIGH_GAIN), ("hot", AmpArchetype.HIGH_GAIN),
                   ("lead", AmpArchetype.LEAD)]
KNOB_ALIASES = {"gain": ("gain",), "bass": ("bass", "low"), "mid": ("mid", "middle"), "treble": ("treble", "high"), "presence": ("presence",),
                "master": ("master", "volume")}


def _on(p: NeuralParams, *names: str) -> bool:
    return any(p.vst_parameters.get(n, 0.0) >= 0.5 for n in names)


def _first(p: NeuralParams, *names: str) -> float | None:
    return next((p.vst_parameters[n] for n in names if n in p.vst_parameters), None)


def neural_to_universal(preset: NeuralXmlPreset, schema_dir: Path | None = None, params: NeuralParams | None = None) -> tuple[UniversalChain, list[str]]:
    p = params or neural_to_vst(preset, schema_dir=schema_dir)
    warn = [*preset.warnings, *p.warnings]
    v = p.vst_parameters

    # amp: the amp_type option selects one of the amp-prefixed knob groups, in schema order
    prefixes: list[str] = []
    for name in (p.order or list(v)):                               # schema order, so the n-th group is the n-th amp
        m = _AMP_KNOB.match(name)
        if m and name in v and m.group("p") in AMP_WORDS and m.group("p") not in prefixes:
            prefixes.append(m.group("p"))
    amp = AmpBlock()
    label = (p.labels.get("amp_type") or "").lower()
    if snake(label) == "three_in_one" and "three_in_one_amp_gain" in v:       # John Mayer X's blend of all three amps: no single amp group applies
        amp = AmpBlock(archetype=AmpArchetype.UNKNOWN, source_model=f"{p.plugin}: three-in-one blend", gain=_first(p, "three_in_one_amp_gain"),
                       master=_first(p, "three_in_one_amp_output"))
        warn.append("Three-In-One blends the three amps: only its gain and output are used, not the single amps' tone knobs")
    elif prefixes:
        pre = next((x for x in prefixes if x in label.replace("-", " ").split() or x == snake(label)), None)   # 'HOT Amp' -> hot, 'Smooth Operator' -> smooth_operator
        if pre is None:
            idx = int(round(p.raw.get("amp_type", 0)))
            if not 0 <= idx < len(prefixes):
                warn.append(f"amp_type {idx} has no matching amp knob group {prefixes}; using {prefixes[0]}")
                idx = 0
            pre = prefixes[idx]
        if pre:
            arch = next((a for w, a in ARCHETYPE_WORDS if w in pre.split("_")), AmpArchetype.UNKNOWN)
            knobs = {k: _first(p, *(f"{pre}{s}_{a}" for a in names for s in ("_amp", ""))) for k, names in KNOB_ALIASES.items()}
            amp = AmpBlock(archetype=arch, source_model=f"{p.plugin or preset.plugin}: {pre} amp", **knobs)
        if p.vst_parameters.get("amp_section_active", 1.0) < 0.5:
            warn.append("amp section bypassed in the preset")

    od = OverdriveBlock(enabled=False)
    pedals_on = all(p.vst_parameters.get(n, 1.0) >= 0.5 for n in ("pedal_section_active", "pedals_active", "active_pedals", "active_pre_fx_section"))
    for pre, kind, drive, tone, level in (("od", OverdriveKind.OVERDRIVE, ("dist",), ("tone",), ("level",)),
                                          ("tealbreaker", OverdriveKind.OVERDRIVE, ("gain",), ("tone",), ("level",)),                 # John Mayer X
                                          ("halfman_od", OverdriveKind.OVERDRIVE, ("gain",), ("treble",), ("level",)),
                                          ("horizon_devices_precision_drive", OverdriveKind.DISTORTION, ("drive",), ("bright",), ("volume",)),   # Misha Mansoor X
                                          ("overdrive_1", OverdriveKind.OVERDRIVE, ("drive", "gain"), ("tone",), ("level",)),          # Nolly X
                                          ("overdrive_2", OverdriveKind.OVERDRIVE, ("gain", "drive"), ("treble", "tone"), ("level",)),
                                          ("justa_boost", OverdriveKind.BOOST, (), (), ("output",)),
                                          ("overdrive", OverdriveKind.OVERDRIVE, ("gain", "drive"), ("tone",), ("level",)),
                                          ("drt", OverdriveKind.DISTORTION, ("dist",), ("filter",), ("vol",)),
                                          ("pathos", OverdriveKind.OVERDRIVE, ("drive",), ("mids", "treble"), ("level",))):
        if _on(p, f"{pre}_active") and pedals_on:
            od = OverdriveBlock(enabled=True, kind=kind, source_model=pre, drive=_first(p, *(f"{pre}_{x}" for x in drive)),
                                tone=_first(p, *(f"{pre}_{x}" for x in tone)), level=_first(p, *(f"{pre}_{x}" for x in level)))
            break

    mod = ModulationBlock(enabled=False)
    for pre, kind in (("chr", ModKind.CHORUS), ("chorus", ModKind.CHORUS), ("phsr", ModKind.PHASER), ("phaser", ModKind.PHASER),
                      ("flanger", ModKind.FLANGER), ("tremolo", ModKind.TREMOLO), ("harmonic_tremolo", ModKind.TREMOLO),
                      ("antelope_filter", ModKind.OTHER), ("chorus_2", ModKind.CHORUS)):
        if _on(p, f"{pre}_active"):
            mod = ModulationBlock(enabled=True, kind=kind, source_model=pre, rate=_first(p, f"{pre}_rate", f"{pre}_speed"),
                                  depth=_first(p, f"{pre}_depth", f"{pre}_intensity"), mix=_first(p, f"{pre}_mix", f"{pre}_dry_wet"))
            break
    if not mod.enabled and _on(p, "modulator_active"):                # Misha Mansoor X: one modulator, chorus or flanger by its mode
        mod = ModulationBlock(enabled=True, kind=ModKind.FLANGER if "flanger" in p.labels.get("modulator_mode", "").lower() else ModKind.CHORUS,
                              source_model="modulator", rate=_first(p, "modulator_auto_speed"))

    delay = DelayBlock(enabled=False)
    for pre in ("dly", "delay", "dream_delay", "stereo_delay", "delay_2", "tape_echo", "delay_1", "millipede_delay"):
        if _on(p, f"{pre}_active"):
            time_ms = p.raw.get(f"{pre}_time") if p.raw.get(f"{pre}_time", 0) > 20 and p.units.get(f"{pre}_time", "ms") == "ms" else None   # <20: a note index
            tempo = p.raw.get(f"{pre}_tempo")
            if time_ms is None and tempo and not _on(p, f"{pre}_sync_active", f"{pre}_sync"):
                time_ms = 60000.0 / tempo                                   # free-running tempo knob in BPM = quarter-note delay
                warn.append("delay time derived from the BPM knob (quarter note): inferred")
            kind = (DelayKind.PING_PONG if _on(p, f"{pre}_ping_pong") else DelayKind.TAPE if _on(p, f"{pre}_tape_saturation") or pre == "tape_echo"
                    else DelayKind.DIGITAL)
            delay = DelayBlock(enabled=True, kind=kind, source_model=pre, time_ms=time_ms if time_ms and 0 < time_ms <= 4000 else None,
                               feedback=_first(p, f"{pre}_feedback"), mix=_first(p, f"{pre}_dry_wet", f"{pre}_mix", f"{pre}_blend"))
            break

    reverb = ReverbBlock(enabled=False)
    for pre in ("rev", "reverb", "studio_verb", "spring_reverb"):
        if _on(p, f"{pre}_active"):
            key = next((k for k in (f"{pre}_time", f"{pre}_decay") if k in p.raw), None)
            t = p.raw.get(key) if key else None
            sec = p.seconds.get(key) if key in p.seconds else (t / 1000.0 if t and t > 0 and p.units.get(key, "") != "%" else None)     # '%' = a knob, no seconds
            reverb = ReverbBlock(enabled=True, kind=ReverbKind.SPRING if pre == "spring_reverb" else ReverbKind.OTHER if _on(p, f"{pre}_mode") else ReverbKind.HALL,
                                 source_model=pre, decay=seconds_to_decay(sec) if sec and sec > 0 else None, mix=_first(p, f"{pre}_dry_wet", f"{pre}_mix"))
            break

    if any(p.vst_parameters.get(n, 1.0) < 0.5 for n in ("fx_section_active", "time_active", "active_time")):
        delay, reverb = DelayBlock(enabled=False), ReverbBlock(enabled=False)
    return UniversalChain(amp=amp, overdrive=od, modulation=mod, delay=delay, reverb=reverb), warn
