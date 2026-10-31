"""Fender Mustang patch (``parsers/fender_fuse.py``)  ->  UniversalChain.

Same interface as ``mapping/katana.py``: ``fender_to_universal(patch) -> (UniversalChain, warnings)``.

Scaling: the amp's knob bytes are 0-255 (``value >> 8`` of the file's 16-bit number, as the PLUG editor reads them), so ``/ 255``. The amp
channel volume and ``master`` are separate knobs; ``master`` is the master volume (ControlIndex 3). The amp archetype per model is INFERRED
from the model it emulates. Effects are mapped by type and on/off only: their knobs differ per effect and are not decoded, so rate / depth /
time / decay / mix stay ``None`` rather than guessed. A noise gate and the cabinet are not part of the universal chain.
"""

from __future__ import annotations

from patch_corpus.parsers.fender_fuse import FusePatch
from patch_corpus.schema import (AmpArchetype, AmpBlock, DelayBlock, DelayKind, ModKind, ModulationBlock, OverdriveBlock, OverdriveKind, ReverbBlock,
                                 ReverbKind, UniversalChain)

_A = AmpArchetype
AMP_ARCHETYPE = {"'57 Deluxe": _A.CRUNCH, "'59 Bassman": _A.CRUNCH, "'57 Champ": _A.CRUNCH, "'65 Deluxe Reverb": _A.CLEAN, "'65 Princeton": _A.CLEAN,
                 "'65 Twin Reverb": _A.CLEAN, "Super-Sonic": _A.CRUNCH, "British '60s": _A.CRUNCH, "British '70s": _A.CRUNCH, "British '80s": _A.HIGH_GAIN,
                 "American '90s": _A.HIGH_GAIN, "Metal 2000": _A.HIGH_GAIN, "Studio Preamp": _A.CLEAN, "'57 Twin": _A.CLEAN, "'60s Thrift": _A.CLEAN,
                 "British Colour": _A.CRUNCH, "British Watts": _A.CRUNCH}
STOMP_KIND = {"Overdrive": OverdriveKind.OVERDRIVE, "Green Box": OverdriveKind.OVERDRIVE, "Orange Box": OverdriveKind.DISTORTION,
              "Black Box": OverdriveKind.DISTORTION, "Ranger Boost": OverdriveKind.BOOST, "Fuzz": OverdriveKind.FUZZ, "Big Fuzz": OverdriveKind.FUZZ,
              "Fuzz Touch Wah": OverdriveKind.FUZZ}
_M = ModKind
MOD_KIND = {"Sine Chorus": _M.CHORUS, "Triangle Chorus": _M.CHORUS, "Sine Flanger": _M.FLANGER, "Triangle Flanger": _M.FLANGER, "Vibratone": _M.ROTARY,
            "Vintage Tremolo": _M.TREMOLO, "Sine Tremolo": _M.TREMOLO, "Phaser": _M.PHASER}
_D = DelayKind
DELAY_KIND = {"Mono Delay": _D.DIGITAL, "Ping Pong Delay": _D.PING_PONG, "Reverse Delay": _D.REVERSE, "Tape Delay": _D.TAPE, "Stereo Tape Delay": _D.TAPE,
              "Ducking Delay": _D.DIGITAL, "Multitap Delay": _D.DIGITAL}
_R = ReverbKind
REVERB_KIND = {"Small Hall": _R.HALL, "Large Hall": _R.HALL, "Small Room": _R.ROOM, "Large Room": _R.ROOM, "Small Plate": _R.PLATE, "Large Plate": _R.PLATE,
               "Ambient": _R.AMBIENCE, "Arena": _R.HALL, "'63 Spring": _R.SPRING, "'65 Spring": _R.SPRING}


def _knob(patch: FusePatch, key: str) -> float | None:
    v = patch.amp.get(key)
    return None if v is None else min(1.0, max(0.0, v / 255.0))


def fender_to_universal(patch: FusePatch) -> tuple[UniversalChain, list[str]]:
    warn = [*patch.warnings, "INFERRED: amp archetypes, and BypassState = 1 meaning 'on'; effect knobs are not decoded (type and on/off only)"]
    amp = AmpBlock(archetype=AMP_ARCHETYPE.get(patch.amp_name, AmpArchetype.UNKNOWN), source_model=f"{patch.amplifier}: {patch.amp_name}",
                   gain=_knob(patch, "gain"), bass=_knob(patch, "bass"), mid=_knob(patch, "middle"), treble=_knob(patch, "treble"),
                   presence=_knob(patch, "presence"), master=_knob(patch, "master"))
    od = OverdriveBlock(enabled=False)
    if (m := next((m for m in patch.effects if m.kind == "stompbox" and m.on and m.name in STOMP_KIND), None)):
        od = OverdriveBlock(enabled=True, kind=STOMP_KIND[m.name], source_model=m.name)
    mod = ModulationBlock(enabled=False)
    if (m := patch.active("modulation")) is not None:
        mod = ModulationBlock(enabled=True, kind=MOD_KIND.get(m.name, ModKind.OTHER), source_model=m.name)
    delay = DelayBlock(enabled=False)
    if (m := patch.active("delay")) is not None:
        delay = DelayBlock(enabled=True, kind=DELAY_KIND.get(m.name, DelayKind.OTHER), source_model=m.name)
    reverb = ReverbBlock(enabled=False)
    if (m := patch.active("reverb")) is not None:
        reverb = ReverbBlock(enabled=True, kind=REVERB_KIND.get(m.name, ReverbKind.OTHER), source_model=m.name)
    return UniversalChain(amp=amp, overdrive=od, modulation=mod, delay=delay, reverb=reverb), warn
