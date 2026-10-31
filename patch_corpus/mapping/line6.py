"""Line 6 Helix / HX preset  ->  UniversalChain.

What is measured and what is guessed (also recorded in the returned warnings):
  * block category and enabled state            from the model ID prefix and ``@enabled`` / snapshot flag: reliable
  * amp archetype                                keyword table on the model ID, else the Drive position: a heuristic
  * dial positions (Drive, Bass, Mid, ...)        Helix stores dial positions as 0-1 floats: used as stored; values above 1 are
                                                  clipped and reported
  * delay time                                    seconds (or ms when > 20); a tempo-synced delay keeps ``time_ms=None`` because the
                                                  note-division index is not decoded
  * modulation rate / depth                       0-1 as stored; > 1 treated as Hz / percent (log-scaled), flagged low-confidence

Only one block of each kind is kept (the first enabled one in signal order); extra ones are mentioned in the warnings.
"""

from __future__ import annotations

import math
import re
from typing import Any

from patch_corpus.parsers.line6_hlx import HelixBlock, HelixPatch
from patch_corpus.schema import (AmpArchetype, AmpBlock, DelayBlock, DelayKind, ModKind, ModulationBlock, OverdriveBlock,
                                 OverdriveKind, ReverbBlock, ReverbKind, UniversalChain)

_CLEAN = re.compile(r"deluxe|twin|doubleverb|double|jazz|rivet|clean|fullerton|vibroking|vibro|essex|mailordertwin|fender|tweed", re.I)
_HIGH = re.compile(r"placater|revv|cali|recti|treadplate|panama|badonk|fatality|litigator|das ?benzin|angel|mandarin|dirty|soup|"
                   r"herbrand|mesa|5150|invader|hot|savage|metal", re.I)
_LEAD = re.compile(r"lead|solo|jvm|plexi|brit2204|j800|jcm", re.I)


def _ci(params: dict[str, Any], names: tuple[str, ...]) -> Any:
    low = {k.lower(): v for k, v in params.items()}
    for n in names:
        if n.lower() in low and low[n.lower()] is not None:
            return low[n.lower()]
    return None


def _frac(v: Any, warn: list[str], what: str) -> float | None:
    if v is None or isinstance(v, bool):
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not 0.0 <= x <= 1.0:
        warn.append(f"{what}={x:g} outside 0-1, clipped")
    return min(1.0, max(0.0, x))


def _rate_frac(v: Any, warn: list[str]) -> float | None:
    """0-1 as stored; larger values are read as Hz on a 0.1-10 Hz log axis (low confidence)."""
    if v is None:
        return None
    x = float(v)
    if 0.0 <= x <= 1.0:
        return x
    warn.append(f"modulation rate {x:g} read as Hz")
    return min(1.0, max(0.0, math.log10(max(x, 0.1) / 0.1) / 2.0))


def classify_amp(model: str, gain: float | None) -> AmpArchetype:
    rest = model.split("_", 1)[-1]
    if _HIGH.search(rest):
        return AmpArchetype.HIGH_GAIN
    if _CLEAN.search(rest) and (gain is None or gain < 0.6):
        return AmpArchetype.CLEAN
    if _LEAD.search(rest):
        return AmpArchetype.LEAD if (gain is None or gain >= 0.5) else AmpArchetype.CRUNCH
    if gain is None:
        return AmpArchetype.UNKNOWN
    return (AmpArchetype.CLEAN if gain < 0.3 else AmpArchetype.CRUNCH if gain < 0.55 else
            AmpArchetype.LEAD if gain < 0.8 else AmpArchetype.HIGH_GAIN)


def _amp(blocks: list[HelixBlock], warn: list[str]) -> AmpBlock:
    amps = [b for b in blocks if b.category == "amp" and b.enabled]
    if not amps:
        if any(b.category == "amp" for b in blocks):
            warn.append("amp block is bypassed")
        return AmpBlock()
    if len(amps) > 1:
        warn.append(f"{len(amps)} enabled amps; using the first ({amps[0].model})")
    a = amps[0]
    p = a.params
    gain = _frac(_ci(p, ("Drive", "Gain")), warn, "Drive")
    master = _ci(p, ("Master",))
    return AmpBlock(
        archetype=classify_amp(a.model, gain), source_model=a.model, gain=gain,
        bass=_frac(_ci(p, ("Bass",)), warn, "Bass"), mid=_frac(_ci(p, ("Mid", "Middle")), warn, "Mid"),
        treble=_frac(_ci(p, ("Treble",)), warn, "Treble"), presence=_frac(_ci(p, ("Presence",)), warn, "Presence"),
        master=_frac(master if master is not None else _ci(p, ("ChVol",)), warn, "Master"))


def overdrive_kind(model: str) -> OverdriveKind:
    rest = model.split("_", 1)[-1]
    if re.search(r"fuzz|muff|bender|mule|arbitrator|thrifter", rest, re.I):
        return OverdriveKind.FUZZ
    if re.search(r"boost|treble|mini", rest, re.I):
        return OverdriveKind.BOOST
    if re.search(r"distortion|rodent|rat|ds1|metal|hm2|deezmo|dist\b|kwb|bitcrush", rest, re.I):
        return OverdriveKind.DISTORTION
    return OverdriveKind.OVERDRIVE


def _overdrive(blocks: list[HelixBlock], warn: list[str]) -> OverdriveBlock:
    amp_pos = next(((b.dsp, b.position) for b in blocks if b.category == "amp"), None)
    dists = [b for b in blocks if b.category == "dist"]
    pre = [b for b in dists if b.enabled and (amp_pos is None or (b.dsp, b.position) <= amp_pos)]
    on = pre or [b for b in dists if b.enabled]
    if not on:
        return OverdriveBlock(enabled=False)
    if len(on) > 1:
        warn.append(f"{len(on)} enabled drive pedals; using the first ({on[0].model})")
    b = on[0]
    return OverdriveBlock(enabled=True, kind=overdrive_kind(b.model), source_model=b.model,
                          drive=_frac(_ci(b.params, ("Drive", "Gain", "Dist", "Sustain")), warn, "pedal Drive"),
                          tone=_frac(_ci(b.params, ("Tone", "Treble", "Filter")), warn, "pedal Tone"),
                          level=_frac(_ci(b.params, ("Level", "Output", "Vol", "Volume")), warn, "pedal Level"))


def mod_kind(model: str) -> ModKind:
    rest = model.split("_", 1)[-1]
    for rx, kind in ((r"rotary", ModKind.ROTARY), (r"flang", ModKind.FLANGER), (r"phas|barberpole|script", ModKind.PHASER),
                     (r"chorus|dimension|ensemble", ModKind.CHORUS), (r"trem", ModKind.TREMOLO), (r"vibe|vibrato", ModKind.VIBRATO)):
        if re.search(rx, rest, re.I):
            return kind
    return ModKind.OTHER


def _modulation(blocks: list[HelixBlock], warn: list[str]) -> ModulationBlock:
    on = [b for b in blocks if b.category == "mod" and b.enabled]
    if not on:
        return ModulationBlock(enabled=False)
    if len(on) > 1:
        warn.append(f"{len(on)} enabled modulation blocks; using the first ({on[0].model})")
    b = on[0]
    depth = _ci(b.params, ("Depth", "Intensity", "Width"))
    return ModulationBlock(
        enabled=True, kind=mod_kind(b.model), source_model=b.model,
        rate=_rate_frac(_ci(b.params, ("Speed", "Rate", "Freq", "Frequency")), warn),
        depth=_frac(depth if depth is None or float(depth) <= 1 else float(depth) / 100.0, warn, "mod Depth"),
        mix=_frac(_ci(b.params, ("Mix", "Level")), warn, "mod Mix"))


def delay_kind(model: str) -> DelayKind:
    rest = model.split("_", 1)[-1]
    for rx, kind in ((r"ping", DelayKind.PING_PONG), (r"reverse", DelayKind.REVERSE), (r"tape|echo(?!plex)|space", DelayKind.TAPE),
                     (r"analog|bucket|transistor|adriatic|mod", DelayKind.ANALOG), (r"multi|pitch|swell", DelayKind.OTHER)):
        if re.search(rx, rest, re.I):
            return kind
    return DelayKind.DIGITAL


def _delay(blocks: list[HelixBlock], warn: list[str]) -> DelayBlock:
    on = [b for b in blocks if b.category == "delay" and b.enabled]
    if not on:
        return DelayBlock(enabled=False)
    if len(on) > 1:
        warn.append(f"{len(on)} enabled delays; using the first ({on[0].model})")
    b = on[0]
    synced = any(str(k).lower().startswith("tempo") and bool(v) for k, v in b.params.items())
    t = _ci(b.params, ("Time", "Time1", "DelayTime"))
    time_ms = None
    if t is not None and not synced:
        x = float(t)
        time_ms = x * 1000.0 if x <= 20 else x
    elif synced:
        warn.append("delay is tempo-synced; time left unknown")
    return DelayBlock(enabled=True, kind=delay_kind(b.model), source_model=b.model, time_ms=time_ms,
                      feedback=_frac(_ci(b.params, ("Feedback", "Repeats", "Feedback1")), warn, "Feedback"),
                      mix=_frac(_ci(b.params, ("Mix", "Level")), warn, "delay Mix"))


def reverb_kind(model: str) -> ReverbKind:
    rest = model.split("_", 1)[-1]
    for rx, kind in ((r"spring", ReverbKind.SPRING), (r"plate", ReverbKind.PLATE), (r"hall|cave|glitz|ganymede", ReverbKind.HALL),
                     (r"room|chamber", ReverbKind.ROOM), (r"ambien|particle|shimmer", ReverbKind.AMBIENCE)):
        if re.search(rx, rest, re.I):
            return kind
    return ReverbKind.OTHER


def _reverb(blocks: list[HelixBlock], warn: list[str]) -> ReverbBlock:
    on = [b for b in blocks if b.category == "reverb" and b.enabled]
    if not on:
        return ReverbBlock(enabled=False)
    if len(on) > 1:
        warn.append(f"{len(on)} enabled reverbs; using the first ({on[0].model})")
    b = on[0]
    return ReverbBlock(enabled=True, kind=reverb_kind(b.model), source_model=b.model,
                       decay=_frac(_ci(b.params, ("Decay", "Time")), warn, "Decay"),
                       mix=_frac(_ci(b.params, ("Mix", "Level")), warn, "reverb Mix"))


def helix_to_universal(patch: HelixPatch) -> tuple[UniversalChain, list[str]]:
    warn = list(patch.warnings)
    chain = UniversalChain(amp=_amp(patch.blocks, warn), overdrive=_overdrive(patch.blocks, warn),
                           modulation=_modulation(patch.blocks, warn), delay=_delay(patch.blocks, warn),
                           reverb=_reverb(patch.blocks, warn))
    return chain, warn
