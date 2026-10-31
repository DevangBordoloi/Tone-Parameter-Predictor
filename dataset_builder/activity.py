"""Which parameters of a Neural DSP plugin actually matter for a given set of switch positions.

Each plugin is a fixed chain (pedals -> amp -> EQ -> cab -> time effects). A knob only changes the sound when every switch
on its path is on: the section switch, the module's own switch, and (for amp channels) the amp-type selector. This
module derives that routing from the parameter names in the plugin schema:

  * section gates     a per-plugin table (SECTION_GATES) of "this switch enables these parameters"
  * module gates      `<prefix>_active` / `active_<prefix>` style switches enable every `<prefix>_*` parameter; the most
                      specific prefix wins (chorus_2_* belongs to chorus_2_active, not chorus_active)
  * amp selector      `amp_type` options map to parameter groups by name prefix (clean_*, rhythm_*, lead_*, ...)
  * sync switches     tempo-synced delays use the note parameter, free-running ones use the time parameter
  * ignored           UI-only and linking parameters are never learned and stay at a pinned value

The activity model drives two things: the sampler (randomise only the active branch) and the loss mask (inactive
parameters do not contribute). A wrong guess is soft: an extra active parameter just gets supervised on its default.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np

from common.schema import ParameterSpec, PluginSchema, norm_key

IGNORED_RX = re.compile(
    r"^bypass$|preset_(previous|next)|tap_tempo|expression|_trigger|latch|selected_band|solo|visualization|auditioning|"
    r"settings_locked|^cab_type|_cab_type|cab_type_unlinked|amp_cab_linked|cab_amp_linked|sculpt_state|_auto$|_auto_|"
    r"^key_root_note$|^scale$|^reference_pitch$|^limiter_|^active_correction$|^tune_|^sculpt_|^de_esser|^harmonies_|^voice_|"
    r"^link_voice"
)
PINNED: dict[str, float] = {"bypass": 0.0, "amp_cab_linked": 1.0, "cab_amp_linked": 1.0}
CORE_RX = re.compile(r"^(input_gain|output_gain|gate|gate_amount|gate_threshold|transpose|doubler_spread)$")

_AMP_RX = (
    r"^(clean|crunch|rhythm|lead|piezo|cln_amp|rust_amp|hot_amp|smooth_operator|headroom_hero|signature_83|three_in_one)_"
)
_NOT_AMP_CORE = r"_eq|compressor|cab_|_cab|room"

# gate parameter -> regexes (matched with re.search on the parameter name) of parameters it enables
SECTION_GATES: dict[str, dict[str, list[str]]] = {
    "archetypeplini": {
        "active_pedals": [r"^overdrive_", r"^compressor_"],
        "active_amp": [r"^amp_type$", _AMP_RX + r"(?!eq)(gain|bass|mid|treble|master|presence|bright)"],
        "active_eq": [r"_eq_"],
        "active_cab": [r"^cab_"],
        "active_time": [r"^delay_", r"^reverb_"],
    },
    "archetypeabasi": {
        "pedals_active": [r"^logos_", r"^pathos_"],
        "time_active": [r"^delay_", r"^reverb_"],
        "active_amp": [r"^amp_type$", _AMP_RX + r"(?!eq)(gain|bass|mid|treble|master|presence|bright|low_boost|attack|tight|blend|level)"],
        "active_eq": [r"_eq_"],
        "cab_active": [r"^cab_"],
    },
    "archetypegojira": {
        "pitch_section_active": [r"^wow_", r"^oct_"],
        "pedal_section_active": [r"^od_", r"^drt_", r"^phsr_", r"^chr_"],
        "amp_section_active": [r"^amp_type$", r"_amp_(?!.*eq)"],
        "eq_section_active": [r"_eq_"],
        "cab_section_active": [r"^cab_"],
        "fx_section_active": [r"^dly_", r"^rev_"],
    },
    "archetypejohnmayerx": {
        "active_pre_fx_section": [r"^justa_boost_", r"^antelope_filter_", r"^halfman_od_", r"^tealbreaker_", r"^millipede_delay_"],
        "active_gravity_tank_section": [r"^spring_reverb_", r"^harmonic_tremolo_"],
        "active_amp_section": [r"^amp_type$"],
        "active_cab_section": [r"^cab_", r"^room_"],
        "active_eq_comp_section": [r"_eq_", r"_compressor_"],
        "active_post_fx_section": [r"^dream_delay_", r"^studio_verb_"],
    },
    "archetypemishamansoorx": {
        "active_special_fx_section": [r"^laser_", r"^glitch_"],
        "active_pre_fx_section": [r"^horizon_devices_", r"^tape_echo_", r"^dual_octaver_", r"^chaos_"],
        "active_amp_section": [r"^amp_type$"],
        "active_cab_section": [r"^cab_", r"^room_"],
        "active_eq_section": [r"_eq_"],
        "active_post_fx_section": [r"^modulator_", r"^stereo_delay_", r"^reverb_"],
    },
    "archetypenollyx": {
        "active_pre_fx_section": [r"^compressor_", r"^overdrive_", r"^delay_1_"],
        "active_amp_section": [r"^amp_type$"],
        "active_cab_section": [r"^cab_", r"^room_", r"_cab_mic_"],
        "active_eq_section": [r"_eq_"],
        "active_post_fx_section": [r"^delay_2_", r"^reverb_"],
    },
    "archetypepetruccix": {
        "active_wah_comp_section": [r"^wah_", r"^compressor_"],
        "active_pre_fx_section": [r"^overdrive_", r"^phaser_", r"^chorus_(?!2)", r"^flanger_"],
        "active_amp_section": [r"^amp_type$"],
        "active_cab_section": [r"^cab_", r"^room_"],
        "active_eq_section": [r"_eq_"],
        "active_volume_section": [r"^volume_"],
        "active_post_fx_section": [r"^chorus_2_", r"^delay_", r"^reverb_"],
    },
}
# extra (gate, regex) pairs the prefix rule cannot see
EXTRA_GATES: dict[str, list[tuple[str, str]]] = {
    "archetypejohnmayerx": [("cab_l_mic_active", r"^cab_.*_l_mic_type$|^cab_l_mid_level$"), ("cab_r_mic_active", r"^cab_.*_r_mic_type$")],
    "archetypenollyx": [("cab_l_active", r"_cab_mic_l_type$"), ("cab_r_active", r"_cab_mic_r_type$")],
    "archetypepetruccix": [("cab_l_active", r"^cab_mic_l_type$"), ("cab_r_active", r"^cab_mic_r_type$")],
    "archetypemishamansoorx": [("cab_l_active", r"^cab_mic_l_type$"), ("cab_r_active", r"^cab_mic_r_type$")],
    "archetypeabasi": [("cab_1_active", r"^cab_1_"), ("cab_2_active", r"^cab_2_")],
}

_ON_LABELS = {"active", "on", "true", "app", "daw", "note", "bypassed"}
_NOTE_RX = r"(sync_?note|_note)"
_TIME_RX = r"(_time|_tempo)"


def _on_class(spec: ParameterSpec) -> int:
    """Class index meaning "switched on" (the last option unless a label says otherwise)."""
    vals = spec.valid_values or [False, True]
    for i, v in enumerate(vals):
        if str(v).lower() in {"active", "on", "true"}:
            return i
    return len(vals) - 1


def _slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(label).lower()).strip("_")


@dataclass
class Gate:
    """`param` enables `targets` when its class is in `on_classes`."""

    param: str
    on_classes: frozenset[int]
    targets: set[str] = field(default_factory=set)


class ActivityModel:
    def __init__(self, schema: PluginSchema):
        self.schema = schema
        self.specs = schema.by_name()
        self.names = [p.name for p in schema.parameters]
        self.key = norm_key(schema.plugin_name)
        self.ignored: set[str] = {n for n in self.names if IGNORED_RX.search(n)} - set(PINNED)
        self.pinned = {n: v for n, v in PINNED.items() if n in self.specs}
        self.gates: list[Gate] = []
        self._build()
        self.parents: dict[str, list[Gate]] = {}
        for g in self.gates:
            for t in g.targets:
                self.parents.setdefault(t, []).append(g)

    # ---- construction -------------------------------------------------------------------------------------------
    def _is_switch(self, name: str) -> bool:
        s = self.specs.get(name)
        return bool(s and s.kind in ("boolean", "enum") and 2 <= s.n_options <= 3)

    def _gate_for(self, name: str, targets: list[str]) -> None:
        targets = [t for t in targets if t != name and t in self.specs]
        if name in self.specs and targets:
            self.gates.append(Gate(name, frozenset({_on_class(self.specs[name])}), set(targets)))

    def _match(self, rx: str, exclude: set[str] = frozenset()) -> list[str]:
        return [n for n in self.names if re.search(rx, n) and n not in exclude]

    def _build(self) -> None:
        # 1. section gates
        for gate, rxs in SECTION_GATES.get(self.key, {}).items():
            self._gate_for(gate, [n for rx in rxs for n in self._match(rx)])
        for gate, rx in EXTRA_GATES.get(self.key, []):
            self._gate_for(gate, self._match(rx))

        # 2. module gates: <prefix>_active / active_<prefix>; the longest matching prefix owns a parameter
        prefixes: dict[str, str] = {
            n: n[: -len("_active")] + "_"
            for n in self.names
            if self._is_switch(n) and n.endswith("_active") and not n.endswith("_section_active")
        }
        owner: dict[str, str] = {}
        for gate, pre in prefixes.items():
            for n in self.names:
                if n != gate and n.startswith(pre) and (n not in owner or len(prefixes[owner[n]]) < len(pre)):
                    owner[n] = gate
        by_gate: dict[str, list[str]] = {}
        for n, g in owner.items():
            by_gate.setdefault(g, []).append(n)
        for g, ts in by_gate.items():
            self._gate_for(g, ts)

        # 3. modulation sub-switches (delay_mod_on -> delay_mod)
        for n in self.names:
            m = re.match(r"^(.+)_mod(?:_on|_active)$", n)
            if m and self._is_switch(n):
                self._gate_for(n, self._match(rf"^{re.escape(m.group(1))}_mod(ulation)?$"))

        # 4. amp selector -> parameter groups by name prefix
        sel = self.specs.get("amp_type")
        if sel and sel.valid_values:
            slugs = [_slug(v) for v in sel.valid_values]
            for cls, slug in enumerate(slugs):
                grp = [n for n in self.names if n.startswith(slug + "_") and n not in self.ignored]
                grp += [n for n in self.names if re.match(rf"^cab_{re.escape(slug)}_", n)]
                self.gates.append(Gate("amp_type", frozenset({cls}), set(grp)))

        # 5. tempo-sync switches: note parameter when synced, time/tempo parameter when free-running
        for n in self.names:
            m = re.match(r"^(.*)_sync(?:_active)?$", n)
            if not m or not self._is_switch(n):
                continue
            pre = m.group(1)
            spec = self.specs[n]
            vals = [str(v).lower() for v in (spec.valid_values or [False, True])]
            note_cls = frozenset(i for i, v in enumerate(vals) if v in _ON_LABELS)
            time_cls = frozenset(set(range(len(vals))) - note_cls)
            notes = [x for x in self.names if x.startswith(pre + "_") and re.search(_NOTE_RX, x) and x != n]
            times = [x for x in self.names if x.startswith(pre + "_") and re.search(_TIME_RX, x) and x != n
                     and x not in notes]
            if note_cls and time_cls and notes:
                self.gates.append(Gate(n, note_cls, set(notes)))
            if note_cls and time_cls and times:
                self.gates.append(Gate(n, time_cls, set(times)))

    # ---- evaluation ---------------------------------------------------------------------------------------------
    def is_active(self, name: str, classes: dict[str, int], _seen: frozenset[str] = frozenset()) -> bool:
        if name in self.ignored or name in _seen:
            return False
        for gate in self.parents.get(name, ()):
            if gate.param not in classes or classes[gate.param] not in gate.on_classes:
                return False
            if not self.is_active(gate.param, classes, _seen | {name}):
                return False
        return True

    def classes_of(self, values: dict[str, float]) -> dict[str, int]:
        return {
            n: self.specs[n].normalized_to_class(values.get(n, self.specs[n].default_normalized))
            for n in self.names if self._is_switch(n) or n == "amp_type"
        }

    def mask(self, values: dict[str, float]) -> np.ndarray:
        """Boolean vector in schema order: True where the parameter affects the sound for these switch positions."""
        classes = self.classes_of(values)
        return np.array([self.is_active(n, classes) for n in self.names], dtype=bool)

    def learnable(self) -> np.ndarray:
        return np.array([n not in self.ignored for n in self.names], dtype=bool)

    def module_of(self, name: str) -> str:
        """The switch that owns `name` (its most specific gate, i.e. the one with the fewest targets), else "core"."""
        own = [g for g in self.parents.get(name, ()) if g.param != "amp_type" and not g.param.startswith("active_")]
        own = [g for g in own if not g.param.endswith("_section_active")]
        if not own:
            return "core"
        return min(own, key=lambda g: len(g.targets)).param

    def modules(self) -> list[str]:
        return sorted({self.module_of(n) for n in self.names})

    def switches(self) -> list[str]:
        """Parameters that gate others (the roots the sampler must draw first)."""
        return sorted({g.param for g in self.gates})

    def describe(self) -> str:
        lines = [f"{self.schema.plugin_name}: {len(self.names)} params, {len(self.ignored)} ignored, {len(self.gates)} gates"]
        for g in sorted(self.gates, key=lambda g: g.param):
            lines.append(f"  {g.param} in {sorted(g.on_classes)} -> {len(g.targets)} params")
        return "\n".join(lines)
