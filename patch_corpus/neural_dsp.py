"""UniversalChain  ->  Neural DSP plugin parameters (normalised 0-1 knobs and discrete switch states).

Nothing here hard-codes a plugin's parameter names. For each plugin the translator reads the inspected schema
(``config/plugin_schemas``), asks ``ActivityModel`` which switches gate which parameters, and resolves the *roles* in the
lookup tables below (gain, bass, delay mix, reverb decay, ...) by pattern against the real names. A role that a plugin does not
have (the Smooth Operator amp has no Mid knob; Plini has no modulation) is reported in ``Translation.notes`` and is simply not
set.

``Translation.known`` is the set of parameters the translation actually decided. Everything else stays at the plugin default and
must be masked out of any loss (``ParamSampler.from_preset`` does this when the preset carries a ``known`` list). A block that
the source patch does not have is *known to be off*, so its module switch is set and marked known.

Scaling rules
  * dial positions (gain, bass, drive, mix, feedback ...)  fraction of travel -> the same fraction of the plugin range
  * delay time        ms -> a ms parameter via the schema's min/max or measured curve; tempo-only plugins get a (BPM, note
                       division) pair whose product reproduces the time, with sync switched on
  * reverb decay      log-fraction of 0.1-10 s -> seconds -> ms / s / enum-label units, or a plain fraction for '%' knobs
  * switches / enums  class index / (n - 1), chosen by label match (amp channel) or the "on" option (module switches)
Every value is snapped to the parameter's own step grid.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from common.paths import CONFIG_DIR
from common.schema import ParameterSpec, PluginSchema, load_schema, to_array
from dataset_builder.activity import ActivityModel, _on_class
from patch_corpus.schema import (AmpArchetype, DelayKind, ModKind, OverdriveKind, ReverbKind, UniversalChain, decay_to_seconds)

# ---------------------------------------------------------------------------------------------------------------------
# lookup tables (edit here, or override per plugin in config/neuraldsp_channel_map.json)
# ---------------------------------------------------------------------------------------------------------------------
# amp archetype -> option-label keywords of the plugin's `amp_type` selector, in order of preference
ARCHETYPE_CHANNEL_KEYWORDS: dict[AmpArchetype, tuple[str, ...]] = {
    AmpArchetype.CLEAN: ("clean", "cln", "smooth", "piezo"),
    AmpArchetype.CRUNCH: ("crunch", "rhythm", "rust", "headroom", "three"),
    AmpArchetype.LEAD: ("lead", "hot", "signature", "rhythm", "three"),
    AmpArchetype.HIGH_GAIN: ("hot", "lead", "rhythm", "rust", "signature"),
    AmpArchetype.UNKNOWN: ("crunch", "rhythm", "three"),
}
# role -> regexes over a parameter name, tried in order, first hit wins (within the selected amp channel)
AMP_ROLES: dict[str, tuple[str, ...]] = {
    "gain": (r"_gain$", r"_drive$", r"_volume$"),
    "bass": (r"_(bass|low)$",),
    "mid": (r"_(mid|middle)$",),
    "treble": (r"_(treble|high)$",),
    "presence": (r"_presence$",),
    "master": (r"_master$", r"_(output|level)$"),
}
AMP_EXCLUDE = re.compile(r"eq|cab|room|compressor|boost|bright|voice|depth|resonance|tight|bite|soar")

PEDAL_MODULE = re.compile(r"(^|_)(od|overdrive|drt|dist|tealbreaker|halfman_od|precision_drive|pathos|boost)(_|\d|$)")
PEDAL_KIND_PREF: dict[OverdriveKind, tuple[str, ...]] = {
    OverdriveKind.BOOST: ("boost", "od", "overdrive", "pathos", "precision"),
    OverdriveKind.OVERDRIVE: ("od", "overdrive", "tealbreaker", "halfman", "pathos", "precision", "boost"),
    OverdriveKind.DISTORTION: ("drt", "dist", "precision", "pathos", "od", "overdrive"),
    OverdriveKind.FUZZ: ("fuzz", "drt", "dist", "precision", "od"),
}
PEDAL_ROLES = {"drive": (r"_(drive|dist|gain)$",), "tone": (r"_(tone|filter|treble|bright)$",),
               "level": (r"_(level|volume|vol|output)$",)}

MOD_MODULE = re.compile(r"(^|_)(chr|chorus|phsr|phaser|flanger|tremolo|modulator|rotary|vibe)(_|\d|$)")
MOD_KIND_PREF: dict[ModKind, tuple[str, ...]] = {
    ModKind.CHORUS: ("chr", "chorus", "modulator"), ModKind.PHASER: ("phsr", "phaser"),
    ModKind.FLANGER: ("flanger", "modulator"), ModKind.TREMOLO: ("tremolo",), ModKind.ROTARY: ("rotary",),
    ModKind.VIBRATO: ("vibe", "chorus"), ModKind.OTHER: (),
}
MOD_MODE_KEYWORD = {ModKind.CHORUS: "chorus", ModKind.FLANGER: "flanger", ModKind.PHASER: "phaser", ModKind.TREMOLO: "trem"}
MOD_ROLES = {"rate": (r"_(rate|speed)$",), "depth": (r"_(depth|intensity)$",), "mix": (r"_(mix|level)$",)}

DELAY_MODULE = re.compile(r"(^|_)(delay|dly|tape_echo|echo)(_|\d|$)")
DELAY_PRIORITY = ("^delay$", "^dly$", "dream_delay", "stereo_delay", "delay_2", "delay_1", "tape_echo", "millipede")
DELAY_TAPE_PRIORITY = ("tape_echo", "millipede", "dly", "dream_delay")
DELAY_ROLES = {"mix": (r"_(dry_wet|mix|blend)$",), "feedback": (r"_feedback$",)}

REVERB_MODULE = re.compile(r"(^|_)(reverb|rev|verb)(_|\d|$)")
REVERB_ROLES = {"mix": (r"_(dry_wet|mix|amount)$",), "decay": (r"_(decay|time)$",), "predelay": (r"_(pre_delay|predelay)$",)}

NOTE_BEATS = {"1/64": 1 / 16, "1/32": 1 / 8, "1/16": 1 / 4, "1/8": 1 / 2, "1/4": 1.0, "1/2": 2.0, "1/1": 4.0,
              "1/16D": 3 / 8, "1/8D": 3 / 4, "1/4D": 1.5, "1/2D": 3.0, "1/8T": 1 / 3, "1/4T": 2 / 3, "1/16T": 1 / 6}


def load_channel_overrides(path: Path | None = None) -> dict[str, dict[str, str]]:
    """{plugin: {archetype: amp_type option label}}, e.g. {"Archetype Plini": {"high_gain": "Lead"}}."""
    p = path or CONFIG_DIR / "neuraldsp_channel_map.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


@dataclass
class Translation:
    plugin: str
    values: dict[str, float]
    known: set[str]
    notes: list[str] = field(default_factory=list)

    def vector(self, schema: PluginSchema) -> np.ndarray:
        return to_array(self.values, schema)

    def known_mask(self, schema: PluginSchema) -> np.ndarray:
        return np.array([p.name in self.known for p in schema.parameters], dtype=bool)

    def to_preset_doc(self, name: str, meta: dict) -> dict:
        """The decoded-preset document that dataset_builder.render_dataset.load_presets / preset_corpus.splits read."""
        return {"preset_name": name, "target_plugin": self.plugin, "decode_method": "patch_translation", "meta": meta,
                "vst_parameters": {k: self.values[k] for k in self.values}, "known": sorted(self.known), "notes": self.notes}


# ---------------------------------------------------------------------------------------------------------------------
# value encoding
# ---------------------------------------------------------------------------------------------------------------------
def _first_float(label: str) -> float | None:
    m = re.search(r"-?\d+(?:\.\d+)?", str(label))
    return float(m.group(0)) if m else None


def encode_value(spec: ParameterSpec, value: float) -> float | None:
    """Physical value in the parameter's display unit -> normalised 0-1, or None when the schema cannot say."""
    if spec.kind == "numeric" and spec.min_value is not None and spec.max_value is not None:
        if spec.linear is False and spec.curve:
            xs, vs = zip(*spec.curve)
            return spec.snap(float(np.interp(value, vs, xs)))
        if spec.max_value != spec.min_value:
            return spec.snap((value - spec.min_value) / (spec.max_value - spec.min_value))
    if spec.valid_values and all(isinstance(v, str) for v in spec.valid_values):
        nums = [_first_float(v) for v in spec.valid_values]
        if all(n is not None for n in nums):
            return spec.class_to_normalized(int(np.argmin(np.abs(np.array(nums) - value))))
    return None


def _unit_of(spec: ParameterSpec) -> str:
    """'ms' | 's' | '%' | 'bpm' | '' from the schema's units, falling back on the label text / value range."""
    u = (spec.units or "").strip().lower()
    if u in ("ms", "s", "%", "bpm"):
        return u
    if spec.valid_values and isinstance(spec.valid_values[0], str):
        lab = spec.valid_values[0].lower()
        for needle, unit in ((" ms", "ms"), (" s", "s"), ("%", "%"), ("bpm", "bpm")):
            if lab.endswith(needle):
                return unit
    if spec.default_display:
        d = spec.default_display.lower()
        if d.endswith("bpm"):
            return "bpm"
        if d.endswith(" ms"):
            return "ms"
        if d.endswith(" s"):
            return "s"
    return ""


class Translator:
    def __init__(self, schema: PluginSchema | str, channel_overrides: dict[str, dict[str, str]] | None = None):
        self.schema = load_schema(schema) if isinstance(schema, str) else schema
        self.act = ActivityModel(self.schema)
        self.by = self.schema.by_name()
        self.names = [p.name for p in self.schema.parameters]
        self.overrides = (channel_overrides if channel_overrides is not None else load_channel_overrides()).get(self.schema.plugin_name, {})
        self.module_switches = [n for n in self.act.switches() if n.endswith("_active") and not n.endswith("_section_active")
                                and not n.startswith("active_") and n not in ("pedals_active", "time_active", "cab_active")
                                and n not in self.act.ignored]
        self.section_switches = [n for n in self.act.switches() if n not in self.act.ignored and (
            n.startswith("active_") or n.endswith("_section_active") or n in ("pedals_active", "time_active", "cab_active"))]

    # ---- small helpers -------------------------------------------------------------------------------------------
    def _set_class(self, t: "_State", name: str, cls: int) -> None:
        spec = self.by[name]
        t.values[name] = spec.class_to_normalized(max(0, min(cls, spec.n_options - 1)))
        t.known.add(name)

    def _switch(self, t: "_State", name: str, on: bool) -> None:
        spec = self.by[name]
        on_cls = _on_class(spec)
        self._set_class(t, name, on_cls if on else (0 if on_cls != 0 else 1))

    def _set_norm(self, t: "_State", name: str | None, x: float | None, note: str = "") -> bool:
        if name is None or x is None:
            return False
        t.values[name] = self.by[name].snap(x)
        t.known.add(name)
        return True

    def _find(self, candidates: list[str], patterns: tuple[str, ...], used: set[str] | None = None) -> str | None:
        for rx in patterns:
            for n in candidates:
                if re.search(rx, n) and not self.by[n].is_categorical and (used is None or n not in used):
                    return n
        return None

    def _module_params(self, base: str) -> list[str]:
        return [n for n in self.names if n.startswith(base + "_") and n not in self.act.ignored]

    def _pick_module(self, rx: re.Pattern, prefs: tuple[str, ...]) -> tuple[str | None, list[str]]:
        mods = [m for m in self.module_switches if rx.search(m[: -len("_active")])]
        for p in prefs:
            for m in mods:
                if re.search(p, m[: -len("_active")]):
                    return m, mods
        return (mods[0] if mods else None), mods

    # ---- blocks --------------------------------------------------------------------------------------------------
    def _amp(self, c: UniversalChain, t: "_State") -> None:
        sel = self.by.get("amp_type")
        if sel is None or not sel.valid_values:
            t.notes.append("plugin has no amp_type selector")
            return
        a = c.amp
        if a.archetype == AmpArchetype.ACOUSTIC:
            t.notes.append("acoustic amp type: no electric-guitar equivalent, amp left at default")
            return
        labels = [str(v) for v in sel.valid_values]
        pick = None
        forced = self.overrides.get(a.archetype.value)
        if forced and forced in labels:
            pick = labels.index(forced)
        else:
            for kw in ARCHETYPE_CHANNEL_KEYWORDS[a.archetype]:
                hit = [i for i, l in enumerate(labels) if kw in l.lower()]
                if hit:
                    pick = hit[0]
                    break
        if pick is None:
            t.notes.append(f"no amp channel found for archetype {a.archetype.value}")
            return
        self._set_class(t, "amp_type", pick)
        targets = sorted({n for g in self.act.gates if g.param == "amp_type" and pick in g.on_classes for n in g.targets})
        cand = [n for n in targets if not AMP_EXCLUDE.search(n)]
        used: set[str] = set()
        for role in ("gain", "bass", "mid", "treble", "presence", "master"):
            val = getattr(a, role)
            name = self._find(cand, AMP_ROLES[role], used)
            if name is None:
                if val is not None:
                    t.notes.append(f"{labels[pick]} has no '{role}' knob")
                continue
            used.add(name)
            self._set_norm(t, name, val)

    def _pedal(self, c: UniversalChain, t: "_State") -> None:
        od = c.overdrive
        mods = [m for m in self.module_switches if PEDAL_MODULE.search(m[: -len("_active")])]
        if not od.enabled:
            for m in mods:
                self._switch(t, m, False)
            return
        chosen = None
        for pref in PEDAL_KIND_PREF[od.kind]:
            hit = [m for m in mods if pref in m]
            if hit:
                chosen = min(hit, key=len)                       # chorus_active before chorus_2_active
                break
        if chosen is None:
            t.notes.append(f"no {od.kind.value} pedal module in this plugin")
            return
        for m in mods:
            self._switch(t, m, m == chosen)
        cand = self._module_params(chosen[: -len("_active")])
        used: set[str] = set()
        for role, val in (("drive", od.drive), ("tone", od.tone), ("level", od.level)):
            name = self._find(cand, PEDAL_ROLES[role], used)
            if name:
                used.add(name)
                self._set_norm(t, name, val)

    def _modulation(self, c: UniversalChain, t: "_State") -> None:
        md = c.modulation
        mods = [m for m in self.module_switches if MOD_MODULE.search(m[: -len("_active")])]
        if not md.enabled:
            for m in mods:
                self._switch(t, m, False)
            return
        chosen = None
        for pref in MOD_KIND_PREF[md.kind]:
            hit = [m for m in mods if pref in m]
            if hit:
                chosen = min(hit, key=len)                       # chorus_active before chorus_2_active
                break
        if chosen is None:
            t.notes.append(f"no {md.kind.value} module in this plugin; modulation left unspecified")
            return
        for m in mods:
            self._switch(t, m, m == chosen)
        base = chosen[: -len("_active")]
        cand = self._module_params(base)
        for n in cand:                                           # e.g. modulator_mode: Chorus | Flanger
            spec = self.by[n]
            if re.search(r"_(mode|type)$", n) and spec.is_categorical and md.kind in MOD_MODE_KEYWORD:
                kw = MOD_MODE_KEYWORD[md.kind]
                hit = [i for i, v in enumerate(spec.valid_values or []) if kw in str(v).lower()]
                if hit:
                    self._set_class(t, n, hit[0])
        used: set[str] = set()
        for role, val in (("rate", md.rate), ("depth", md.depth), ("mix", md.mix)):
            name = self._find(cand, MOD_ROLES[role], used)
            if name:
                used.add(name)
                self._set_norm(t, name, val)

    def _delay(self, c: UniversalChain, t: "_State") -> None:
        d = c.delay
        mods = [m for m in self.module_switches if DELAY_MODULE.search(m[: -len("_active")])]
        if not d.enabled:
            for m in mods:
                self._switch(t, m, False)
            return
        prefs = DELAY_TAPE_PRIORITY if d.kind == DelayKind.TAPE else DELAY_PRIORITY
        chosen = next((m for p in prefs for m in mods if re.search(p, m[: -len("_active")])), mods[0] if mods else None)
        if chosen is None:
            t.notes.append("plugin has no delay")
            return
        for m in mods:
            self._switch(t, m, m == chosen)
        base = chosen[: -len("_active")]
        cand = self._module_params(base)
        used: set[str] = set()
        for role, val in (("mix", d.mix), ("feedback", d.feedback)):
            name = self._find(cand, DELAY_ROLES[role], used)
            if name:
                used.add(name)
                self._set_norm(t, name, val)
        pp = next((n for n in cand if n.endswith("ping_pong") and self.by[n].is_categorical), None)
        if pp:
            self._switch(t, pp, d.kind == DelayKind.PING_PONG)
        if d.time_ms is None:
            return
        sync = next((n for n in cand if re.search(r"_sync(_active)?$", n) and self.by[n].is_categorical), None)
        time_p = next((n for n in cand if re.search(r"_time(_l)?$|_time$", n) and not self.by[n].is_categorical
                       and _unit_of(self.by[n]) in ("ms", "s", "")), None)
        tempo_p = next((n for n in cand if n.endswith("_tempo")), None)
        note_p = next((n for n in cand if re.search(r"(sync_?note|_note)$", n) and self.by[n].kind == "enum"), None)   # 21 options: ordinal
        if time_p:
            unit = _unit_of(self.by[time_p])
            x = encode_value(self.by[time_p], d.time_ms / (1000.0 if unit == "s" else 1.0))
            if self._set_norm(t, time_p, x) and sync:
                spec = self.by[sync]
                labels = [str(v).lower() for v in (spec.valid_values or [False, True])]
                free = next((i for i, l in enumerate(labels) if l in ("free", "ms", "inactive", "false", "off")), 0)
                self._set_class(t, sync, free)
        elif tempo_p and note_p and sync:
            self._tempo_sync(t, d.time_ms, tempo_p, note_p, sync)
        else:
            t.notes.append("delay time has no direct or tempo/note encoding here")

    def _tempo_sync(self, t: "_State", time_ms: float, tempo_p: str, note_p: str, sync_p: str) -> None:
        tspec, nspec = self.by[tempo_p], self.by[note_p]
        lo, hi = tspec.min_value or 40.0, tspec.max_value or 240.0
        if tspec.valid_values:                                    # enum of "135.0 BPM" labels
            nums = [_first_float(v) for v in tspec.valid_values]
            lo, hi = min(n for n in nums if n is not None), max(n for n in nums if n is not None)
        best = None
        for note, beats in NOTE_BEATS.items():
            if note not in [str(v) for v in (nspec.valid_values or [])]:
                continue
            bpm = 60000.0 * beats / time_ms
            if lo <= bpm <= hi:
                score = abs(np.log(bpm / 120.0)) + (0.0 if note in ("1/4", "1/8", "1/2") else 0.3)
                if best is None or score < best[0]:
                    best = (score, note, bpm)
        if best is None:
            t.notes.append(f"{time_ms:.0f} ms cannot be expressed as tempo x note in {lo:.0f}-{hi:.0f} BPM")
            return
        _, note, bpm = best
        sspec = self.by[sync_p]
        labels = [str(v).lower() for v in (sspec.valid_values or [False, True])]
        on = next((i for i, l in enumerate(labels) if l in ("active", "on", "true", "note", "daw")), len(labels) - 1)
        self._set_class(t, sync_p, on)
        self._set_class(t, note_p, [str(v) for v in nspec.valid_values].index(note))
        x = encode_value(tspec, bpm)
        if x is not None:
            self._set_norm(t, tempo_p, x)

    def _reverb(self, c: UniversalChain, t: "_State") -> None:
        r = c.reverb
        mods = [m for m in self.module_switches if REVERB_MODULE.search(m[: -len("_active")])]
        if not r.enabled:
            for m in mods:
                self._switch(t, m, False)
            return
        spring = r.kind == ReverbKind.SPRING
        order = [m for m in mods if ("spring" in m) == spring] + [m for m in mods if ("spring" in m) != spring]
        if not order:
            t.notes.append("plugin has no reverb")
            return
        chosen = order[0]
        for m in mods:
            self._switch(t, m, m == chosen)
        cand = self._module_params(chosen[: -len("_active")])
        used: set[str] = set()
        name = self._find(cand, REVERB_ROLES["mix"], used)
        if name:
            used.add(name)
            self._set_norm(t, name, r.mix)
        dn = self._find(cand + [n for n in self.names if n.endswith("_decay") and n.startswith(chosen[: -len("_active")])], REVERB_ROLES["decay"], used) \
            or next((n for n in cand if re.search(r"_(decay|time)$", n)), None)
        if dn and r.decay is not None:
            spec = self.by[dn]
            sec = decay_to_seconds(r.decay)
            unit = _unit_of(spec)
            if unit == "%":
                x = r.decay
            elif unit == "s":
                x = encode_value(spec, sec)
            elif unit == "ms" or (spec.max_value or 0) >= 100:
                x = encode_value(spec, sec * 1000.0)
            else:
                x = encode_value(spec, sec)
            if x is None and spec.kind == "enum":                 # unlabeled enum: fall back to the log fraction
                x = r.decay
            if not self._set_norm(t, dn, x):
                t.notes.append(f"could not scale reverb decay for {dn}")
        pn = self._find(cand, REVERB_ROLES["predelay"], used)
        if pn and r.predelay_ms is not None and _unit_of(self.by[pn]) in ("ms", ""):
            self._set_norm(t, pn, encode_value(self.by[pn], r.predelay_ms))

    # ---- entry point ---------------------------------------------------------------------------------------------
    def translate(self, chain: UniversalChain) -> Translation:
        t = _State(values=dict(self.schema.defaults()), known=set(), notes=[])
        for n, v in self.act.pinned.items():
            t.values[n] = v
        for sec in self.section_switches:                         # sections on, so module switches decide what is heard
            self._switch(t, sec, True)
        self._amp(chain, t)
        self._pedal(chain, t)
        self._modulation(chain, t)
        self._delay(chain, t)
        self._reverb(chain, t)
        t.values = {k: self.by[k].snap(v) for k, v in t.values.items()}
        return Translation(self.schema.plugin_name, t.values, t.known, t.notes)


@dataclass
class _State:
    values: dict[str, float]
    known: set[str]
    notes: list[str]


_CACHE: dict[str, Translator] = {}


def translate_chain(chain: UniversalChain, plugin: str | PluginSchema) -> Translation:
    key = plugin if isinstance(plugin, str) else plugin.plugin_name
    if key not in _CACHE:
        _CACHE[key] = Translator(plugin)
    return _CACHE[key].translate(chain)
