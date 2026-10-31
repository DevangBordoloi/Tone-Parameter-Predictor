"""Boss GT / ME / GX patch (``parsers/boss_tsl.py``)  ->  UniversalChain. Same interface as ``mapping/katana.py``.

The liveset already names every parameter, so the work here is choosing the active preamp channel and scaling knobs:

  GT-style units (GT-100, GT-1, GT-1000, GX-100, ME-90 ... any patch with ``preamp_a_*`` keys)
      channel        ``divider_ch_select`` 0 = A, 1 = B -> ``preamp_a_*`` / ``preamp_b_*`` (``preamp_*`` when the unit has one channel)
      amp            gain 0-120, bass / middle / treble / presence / level 0-100
      od/ds          ``od_ds_*``: drive 0-120, tone 0-100, effect level 0-100; type -> kind by the Boss OD/DS list (boost / OD / DS / fuzz)
      chorus         ``chorus_*`` rate / depth / effect level 0-100
      delay          ``delay_delay_time`` (ms, precomputed by the editor), feedback 0-100, effect level 0-120
      reverb         ``reverb_time`` in 0.1 s steps (32 = 3.2 s), ``reverb_pre_delay`` ms, effect level 0-100
  ME-80 (compact knobs)
      ``amp1..5`` = gain / bass / middle / treble / level, ``odds1..3`` = drive / tone / level, ``rev`` = reverb level

INFERRED, not verified against hardware: the ranges above (taken from the Boss owner's manuals' parameter tables: an out-of-range raw
value is reported and dropped), the OD/DS type -> kind grouping, and the ME-80 knob order. Preamp type numbers are kept as
``source_model`` ("GT preamp type 27"); no archetype is guessed from them. Every record carries a warning naming what was inferred.
"""

from __future__ import annotations

from patch_corpus.parsers.boss_tsl import BossPatch
from patch_corpus.schema import (AmpArchetype, AmpBlock, DelayBlock, DelayKind, ModKind, ModulationBlock, OverdriveBlock, OverdriveKind,
                                 ReverbBlock, ReverbKind, UniversalChain, seconds_to_decay)

_O = OverdriveKind
OD_KIND = {**{i: _O.BOOST for i in (0, 1, 2)}, **{i: _O.OVERDRIVE for i in (3, 4, 5, 10, 11, 12, 13)}, **{i: _O.FUZZ for i in (9, 19, 20)}}
_R = ReverbKind
REVERB_KIND = {0: _R.AMBIENCE, 1: _R.ROOM, 2: _R.HALL, 3: _R.HALL, 4: _R.PLATE, 5: _R.SPRING, 6: _R.OTHER}
INFERRED = "Boss GT/ME mapping: knob ranges and type groupings are inferred from the owner's manuals, not checked on hardware"


class _Reader:
    def __init__(self, patch: BossPatch):
        self.p, self.warn = patch, list(patch.warnings)

    def knob(self, top: int, *names: str) -> float | None:
        v = self.p.get(*names)
        if v is None:
            return None
        if not 0 <= v <= top:
            self.warn.append(f"{names[0]}={v} outside 0..{top}: dropped")
            return None
        return v / top

    def on(self, *names: str) -> bool:
        return bool(self.p.get(*names))


def _gt(r: _Reader) -> UniversalChain:
    p = r.p
    ch = "b" if p.get("divider_ch_select") == 1 and p.get("preamp_b_gain") is not None else "a"
    pre = f"preamp_{ch}" if p.get(f"preamp_{ch}_gain") is not None else "preamp"
    amp = AmpBlock()
    if p.get(f"{pre}_gain") is not None:
        t = p.get(f"{pre}_type")
        amp = AmpBlock(archetype=AmpArchetype.UNKNOWN, source_model=f"{p.device} preamp type {t}" if t is not None else None,
                       gain=r.knob(120, f"{pre}_gain"), bass=r.knob(100, f"{pre}_bass"), mid=r.knob(100, f"{pre}_middle", f"{pre}_mid"),
                       treble=r.knob(100, f"{pre}_treble"), presence=r.knob(100, f"{pre}_presence"), master=r.knob(100, f"{pre}_level"))
        if not r.on(f"{pre}_on_off") and p.get(f"{pre}_on_off") is not None:
            r.warn.append(f"{pre} switched off in the patch")

    od = OverdriveBlock(enabled=False)
    if p.get("od_ds_type") is not None:
        t = p.get("od_ds_type")
        od = OverdriveBlock(enabled=r.on("od_ds_on_off"), kind=OD_KIND.get(t, _O.DISTORTION), source_model=f"{p.device} OD/DS type {t}",
                            drive=r.knob(120, "od_ds_drive"), tone=r.knob(100, "od_ds_tone"), level=r.knob(100, "od_ds_effect_level"))

    mod = ModulationBlock(enabled=False)
    if r.on("chorus_on_off"):
        mod = ModulationBlock(enabled=True, kind=ModKind.CHORUS, source_model="chorus", rate=r.knob(100, "chorus_rate"),
                              depth=r.knob(100, "chorus_depth"), mix=r.knob(100, "chorus_effect_level"))

    delay = DelayBlock(enabled=False)
    if r.on("delay_on_off"):
        ms = p.get("delay_delay_time")
        if ms is None and p.get("delay_delay_time_h") is not None:
            ms = p.get("delay_delay_time_h") * 128 + (p.get("delay_delay_time_l") or 0)
        delay = DelayBlock(enabled=True, kind=DelayKind.DIGITAL, source_model=f"delay type {p.get('delay_type')}",
                           time_ms=float(ms) if ms and 0 < ms <= 4000 else None, feedback=r.knob(100, "delay_f_back", "delay_feedback"),
                           mix=r.knob(120, "delay_effect_level"))

    reverb = ReverbBlock(enabled=False)
    if r.on("reverb_on_off"):
        t, tm, pre_ms = p.get("reverb_type"), p.get("reverb_time"), p.get("reverb_pre_delay")
        reverb = ReverbBlock(enabled=True, kind=REVERB_KIND.get(t, _R.OTHER), source_model=f"reverb type {t}",
                             decay=seconds_to_decay(tm / 10.0) if tm and 1 <= tm <= 100 else None,
                             predelay_ms=float(pre_ms) if pre_ms is not None and 0 <= pre_ms <= 500 else None,
                             mix=r.knob(100, "reverb_effect_level"))
    return UniversalChain(amp=amp, overdrive=od, modulation=mod, delay=delay, reverb=reverb)


def _me80(r: _Reader) -> UniversalChain:
    p = r.p
    amp = AmpBlock()
    if p.get("amp1") is not None:
        amp = AmpBlock(source_model=f"ME-80 amp type {p.get('amp_type')}", gain=r.knob(100, "amp1"), bass=r.knob(100, "amp2"),
                       mid=r.knob(100, "amp3"), treble=r.knob(100, "amp4"), master=r.knob(100, "amp5"))
    od = OverdriveBlock(enabled=False)
    if p.get("odds1") is not None:
        od = OverdriveBlock(enabled=r.on("odds_sw"), kind=OverdriveKind.OVERDRIVE, source_model=f"ME-80 OD/DS type {p.get('odds_type')}",
                            drive=r.knob(100, "odds1"), tone=r.knob(100, "odds2"), level=r.knob(100, "odds3"))
    mod = ModulationBlock(enabled=r.on("mod_sw"), kind=ModKind.OTHER, source_model=f"ME-80 mod type {p.get('mod_type')}")
    delay = DelayBlock(enabled=r.on("dly_sw"), source_model=f"ME-80 delay type {p.get('dly_type')}")
    reverb = ReverbBlock(enabled=r.on("rev_sw"), kind=REVERB_KIND.get(p.get("rev_type"), _R.OTHER), mix=r.knob(100, "rev"))
    return UniversalChain(amp=amp, overdrive=od, modulation=mod, delay=delay, reverb=reverb)


def boss_to_universal(patch: BossPatch) -> tuple[UniversalChain, list[str]]:
    r = _Reader(patch)
    if any(k.startswith(("preamp_a_", "preamp_gain")) for k in patch.params):
        chain = _gt(r)
    elif "amp1" in patch.params:
        chain = _me80(r)
    else:
        r.warn.append(f"device {patch.device!r}: no known parameter names (preamp_a_* / amp1..5); nothing decoded")
        chain = UniversalChain()
    r.warn.append(INFERRED)
    return chain, r.warn
