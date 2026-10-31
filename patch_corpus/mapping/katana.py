"""Boss Katana (MkII) patch  ->  UniversalChain.

Addresses are the absolute Katana memory addresses from TuxKatana's ``params/*.yaml``. Each entry carries a ``verified`` flag:

  verified     named explicitly in TuxKatana (booster, amp type/bass/middle/treble, mod and delay and reverb switches / types,
               delay time, feedback and effect level, reverb time, pre-delay, effect level)
  inferred     NOT named there. Amp gain (0x22) and presence (0x27) are placed by continuing the type / ... / bass / middle / treble
               run (TuxKatana lists 0x22 and 0x28 only as "unknown levels"); the FX(2) block is assumed to share the MOD block's
               switch/type layout. Check these on real files before trusting them:
               ``python -m patch_corpus.inspect_tsl file.tsl`` prints the raw bytes, and ``KATANA_OVERRIDES`` / the ``overrides``
               argument lets you correct an address without editing code.

Checked on all 603 scraped files (742 MkII patches; ``python -m patch_corpus.verify_katana``): type 0x21 takes 21 distinct values, all in
TuxKatana's type table; bass / middle / treble 0x24-0x26 are 0-100 knobs; gain 0x22 is a 15-100 knob that follows the patch character
(names with "clean": mean 49, "lead/solo": 68), still inferred. The old ``amp_volume`` address 0x23 is the constant 10 in every patch, so
master is left ``None``. TuxKatana's live-edit addresses for gain / volume (60 00 06 51 / 52) do not line up with the ``Patch_Mk2V2`` group
of these files (am_num agrees with the Patch_0 type in 3 of 299 patches), so they cannot confirm 0x22 / 0x27 / 0x28.

Not decoded: per-type modulation rate/depth (their offsets depend on the effect type, TuxKatana keeps them per type), delay
and reverb high/low cut. Those universal fields stay ``None`` rather than guessed.

Scaling: knob bytes are 0-100 (booster drive 0-120; reported when larger); delay time and pre-delay are two 7-bit bytes (hi*128+lo, ms); reverb time
is 0-99 = 0.1-10 s.
"""

from __future__ import annotations

from dataclasses import dataclass

from patch_corpus.parsers.katana_tsl import KatanaPatch
from patch_corpus.schema import (AmpArchetype, AmpBlock, DelayBlock, DelayKind, ModKind, ModulationBlock, OverdriveBlock,
                                 OverdriveKind, ReverbBlock, ReverbKind, UniversalChain, seconds_to_decay)

P0, FX1, FX2, D1, D2, P1 = ("UserPatch%Patch_0", "UserPatch%Fx(1)", "UserPatch%Fx(2)", "UserPatch%Delay(1)",
                            "UserPatch%Delay(2)", "UserPatch%Patch_1")


@dataclass(frozen=True)
class Addr:
    group: str
    address: int
    verified: bool = True
    wide: bool = False                      # two 7-bit bytes
    max: int = 100                          # top of the knob's range (the booster's drive goes to 120)


KATANA_ADDRESSES: dict[str, Addr] = {
    # booster (Patch_0 @ 60000010)
    "booster_sw": Addr(P0, 0x60000010), "booster_type": Addr(P0, 0x60000011), "booster_drive": Addr(P0, 0x60000012, max=120),
    "booster_bottom": Addr(P0, 0x60000013), "booster_tone": Addr(P0, 0x60000014), "booster_level": Addr(P0, 0x60000017),
    # preamp
    "amp_type": Addr(P0, 0x60000021), "amp_bass": Addr(P0, 0x60000024), "amp_mid": Addr(P0, 0x60000025),
    "amp_treble": Addr(P0, 0x60000026), "amp_gain": Addr(P0, 0x60000022, False), "amp_presence": Addr(P0, 0x60000027, False),
    # 0x60000023 was mapped to amp_volume; it is the constant 10 in all 742 MkII patches (not a knob), so it is no longer read.
    # MOD (Fx(1) @ 60000100) and FX (Fx(2) @ 60000300)
    "mod_sw": Addr(FX1, 0x60000100), "mod_type": Addr(FX1, 0x60000101),
    "fx_sw": Addr(FX2, 0x60000300, False), "fx_type": Addr(FX2, 0x60000301, False),
    # delay (Delay(1) @ 60000500)
    "delay_sw": Addr(D1, 0x60000500), "delay_type": Addr(D1, 0x60000501), "delay_time": Addr(D1, 0x60000502, wide=True),
    "delay_feedback": Addr(D1, 0x60000504), "delay_effect": Addr(D1, 0x60000506),
    "delay2_sw": Addr(D2, 0x60000520, False), "delay2_type": Addr(D2, 0x60000521, False),
    "delay2_time": Addr(D2, 0x60000522, False, wide=True), "delay2_feedback": Addr(D2, 0x60000524, False),
    "delay2_effect": Addr(D2, 0x60000526, False),
    # reverb (Patch_1 @ 60000540)
    "reverb_sw": Addr(P1, 0x60000540), "reverb_type": Addr(P1, 0x60000541), "reverb_time": Addr(P1, 0x60000542),
    "reverb_predelay": Addr(P1, 0x60000543, wide=True), "reverb_effect": Addr(P1, 0x60000548),
}
KATANA_OVERRIDES: dict[str, Addr] = {}

AMP_TYPES = {0x00: "Natural Clean", 0x01: "Acoustic", 0x02: "Combo Crunch", 0x03: "Stack Crunch", 0x04: "HiGain Stack",
             0x05: "Power Drive", 0x06: "Extreme Lead", 0x07: "Core Metal", 0x08: "Clean", 0x09: "Clean Twin", 0x0A: "Pro Crunch",
             0x0B: "Crunch", 0x0C: "Delux Crunch", 0x0D: "VO Drive", 0x0E: "VO Lead", 0x0F: "Match Drive", 0x11: "BG Drive",
             0x12: "MS 1959 I", 0x13: "MS 1959 I+II", 0x14: "Fier Vintage", 0x15: "Fier Modern", 0x16: "T-Amp Lead", 0x17: "Brown",
             0x18: "Lead", 0x1C: "Acoustic Variation", 0x1D: "Clean Variation", 0x1E: "Crunch Variation", 0x1F: "Lead Variation",
             0x20: "Brown Variation"}
_A = AmpArchetype
AMP_ARCHETYPE = {"Natural Clean": _A.CLEAN, "Acoustic": _A.ACOUSTIC, "Acoustic Variation": _A.ACOUSTIC, "Clean": _A.CLEAN,
                 "Clean Twin": _A.CLEAN, "Clean Variation": _A.CLEAN, "Combo Crunch": _A.CRUNCH, "Stack Crunch": _A.CRUNCH,
                 "Pro Crunch": _A.CRUNCH, "Crunch": _A.CRUNCH, "Delux Crunch": _A.CRUNCH, "Crunch Variation": _A.CRUNCH,
                 "VO Drive": _A.CRUNCH, "BG Drive": _A.CRUNCH, "MS 1959 I": _A.CRUNCH, "MS 1959 I+II": _A.CRUNCH,
                 "Match Drive": _A.CRUNCH, "Power Drive": _A.LEAD, "VO Lead": _A.LEAD, "T-Amp Lead": _A.LEAD, "Lead": _A.LEAD,
                 "Lead Variation": _A.LEAD, "Brown": _A.LEAD, "Brown Variation": _A.LEAD, "Fier Vintage": _A.LEAD,
                 "HiGain Stack": _A.HIGH_GAIN, "Extreme Lead": _A.HIGH_GAIN, "Core Metal": _A.HIGH_GAIN, "Fier Modern": _A.HIGH_GAIN}

_O = OverdriveKind
BOOSTER_TYPES = {0x00: ("Mid Boost", _O.BOOST), 0x01: ("Clean Boost", _O.BOOST), 0x02: ("Treble Boost", _O.BOOST),
                 0x03: ("Crunch OD", _O.OVERDRIVE), 0x04: ("Natural OD", _O.OVERDRIVE), 0x05: ("Warm OD", _O.OVERDRIVE),
                 0x06: ("Fat DS", _O.DISTORTION), 0x08: ("Metal DS", _O.DISTORTION), 0x09: ("Oct Fuzz", _O.FUZZ),
                 0x0A: ("Blues Drive", _O.OVERDRIVE), 0x0B: ("Over Drive", _O.OVERDRIVE), 0x0C: ("TubeScreamer", _O.OVERDRIVE),
                 0x0D: ("Turbo OD", _O.OVERDRIVE), 0x0E: ("Distortion", _O.DISTORTION), 0x0F: ("Rat", _O.DISTORTION),
                 0x10: ("GuV DS", _O.DISTORTION), 0x11: ("DTS+", _O.DISTORTION), 0x12: ("Metal Zone", _O.DISTORTION),
                 0x13: ("60s Fuzz", _O.FUZZ), 0x14: ("Muff Fuzz", _O.FUZZ), 0x15: ("HM-2", _O.DISTORTION),
                 0x16: ("Metal Core", _O.DISTORTION), 0x17: ("Centa OD", _O.OVERDRIVE)}

_M = ModKind
MOD_TYPES = {0x13: ("Phaser", _M.PHASER), 0x14: ("Flanger", _M.FLANGER), 0x15: ("Tremolo", _M.TREMOLO), 0x16: ("Rotary", _M.ROTARY),
             0x17: ("Uni-V", _M.VIBRATO), 0x1A: ("Vibrato", _M.VIBRATO), 0x1C: ("Humanizer", _M.OTHER), 0x1D: ("Chorus", _M.CHORUS),
             0x23: ("Phaser 90E", _M.PHASER), 0x24: ("Flanger 117E", _M.FLANGER), 0x26: ("DC30", _M.CHORUS)}

_D = DelayKind
DELAY_TYPES = {0x00: _D.DIGITAL, 0x01: _D.PING_PONG, 0x02: _D.DIGITAL, 0x03: _D.DIGITAL, 0x04: _D.DIGITAL, 0x05: _D.DIGITAL,
               0x06: _D.REVERSE, 0x07: _D.ANALOG, 0x08: _D.OTHER, 0x09: _D.MODULATED, 0x0A: _D.ANALOG}
_R = ReverbKind
REVERB_TYPES = {0x00: _R.AMBIENCE, 0x01: _R.ROOM, 0x02: _R.HALL, 0x03: _R.HALL, 0x04: _R.PLATE, 0x05: _R.SPRING, 0x06: _R.OTHER}


class _Reader:
    def __init__(self, patch: KatanaPatch, overrides: dict[str, Addr] | None):
        self.p, self.map, self.warn = patch, {**KATANA_ADDRESSES, **KATANA_OVERRIDES, **(overrides or {})}, []
        self.used_inferred: set[str] = set()

    def raw(self, key: str) -> int | None:
        a = self.map[key]
        v = self.p.word7(a.group, a.address) if a.wide else self.p.byte(a.group, a.address)
        if v is not None and not a.verified:
            self.used_inferred.add(key)
        return v

    def knob(self, key: str) -> float | None:
        v = self.raw(key)
        if v is None:
            return None
        top = self.map[key].max
        if v > top:
            self.warn.append(f"{key}={v} exceeds {top}: address {self.map[key].address:08X} is probably not this parameter")
            return None
        return v / top

    def on(self, key: str) -> bool:
        return bool(self.raw(key))


def katana_to_universal(patch: KatanaPatch, overrides: dict[str, Addr] | None = None) -> tuple[UniversalChain, list[str]]:
    r = _Reader(patch, overrides)
    r.warn.extend(patch.warnings)

    t = r.raw("amp_type")
    amp = AmpBlock()
    if t is not None:
        name = AMP_TYPES.get(t, f"type 0x{t:02X}")
        arch = AMP_ARCHETYPE.get(name, AmpArchetype.UNKNOWN)
        if name.startswith("type"):
            r.warn.append(f"unknown amp type byte 0x{t:02X}")
        amp = AmpBlock(archetype=arch, source_model=name, gain=r.knob("amp_gain"), bass=r.knob("amp_bass"), mid=r.knob("amp_mid"),
                       treble=r.knob("amp_treble"), presence=r.knob("amp_presence"), master=None)
    else:
        r.warn.append("Patch_0 group missing or short: no preamp data")

    od = OverdriveBlock(enabled=False)
    bt = r.raw("booster_type")
    if bt is not None:
        name, kind = BOOSTER_TYPES.get(bt, (f"type 0x{bt:02X}", OverdriveKind.OVERDRIVE))
        od = OverdriveBlock(enabled=r.on("booster_sw"), kind=kind, source_model=name, drive=r.knob("booster_drive"),
                            tone=r.knob("booster_tone"), level=r.knob("booster_level"))

    mod = ModulationBlock(enabled=False)
    for sw, ty in (("mod_sw", "mod_type"), ("fx_sw", "fx_type")):
        mt = r.raw(ty)
        if mt is not None and r.on(sw) and mt in MOD_TYPES:
            name, kind = MOD_TYPES[mt]
            mod = ModulationBlock(enabled=True, kind=kind, source_model=name)      # rate / depth: per-type offsets, not decoded
            break

    delay = DelayBlock(enabled=False)
    for pre in ("delay", "delay2"):
        if r.raw(f"{pre}_sw") is not None and r.on(f"{pre}_sw"):
            dt = r.raw(f"{pre}_type") or 0
            time = r.raw(f"{pre}_time")
            delay = DelayBlock(enabled=True, kind=DELAY_TYPES.get(dt, DelayKind.OTHER), source_model=f"type 0x{dt:02X}",
                               time_ms=float(time) if time and time <= 2000 else None, feedback=r.knob(f"{pre}_feedback"),
                               mix=r.knob(f"{pre}_effect"))
            break

    reverb = ReverbBlock(enabled=False)
    if r.raw("reverb_sw") is not None and r.on("reverb_sw"):
        rt, tm, pre = r.raw("reverb_type") or 0, r.raw("reverb_time"), r.raw("reverb_predelay")
        reverb = ReverbBlock(enabled=True, kind=REVERB_TYPES.get(rt, ReverbKind.OTHER), source_model=f"type 0x{rt:02X}",
                             decay=None if tm is None or tm > 99 else seconds_to_decay(0.1 + 0.1 * tm),
                             predelay_ms=float(pre) if pre is not None and pre <= 500 else None, mix=r.knob("reverb_effect"))

    if r.used_inferred:
        r.warn.append("used UNVERIFIED addresses: " + ", ".join(sorted(r.used_inferred)))
    return UniversalChain(amp=amp, overdrive=od, modulation=mod, delay=delay, reverb=reverb), r.warn
