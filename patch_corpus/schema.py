"""Device-agnostic guitar signal chain and the record that carries it through the corpus.

Conventions
  * every knob is a *fraction of its dial travel* in [0, 1] (0 = fully counter-clockwise, 1 = fully clockwise)
  * ``None`` means "this device/patch does not tell us": it is never silently replaced by a default, so downstream code can
    mask it out of the training loss instead of teaching the network a made-up value
  * ``enabled`` is always known (a block that is absent from a patch is disabled, which is information)
  * reverb ``decay`` is a fraction on a log scale between ``DECAY_MIN_S`` (0.1 s) and ``DECAY_MAX_S`` (10 s)
"""

from __future__ import annotations

import math
from enum import Enum
from typing import Annotated, Optional

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field

DECAY_MIN_S, DECAY_MAX_S = 0.1, 10.0


def _clip01(v):
    if v is None:
        return None
    return min(1.0, max(0.0, float(v)))


Unit = Annotated[Optional[float], BeforeValidator(_clip01)]


def decay_to_seconds(frac: float) -> float:
    return DECAY_MIN_S * (DECAY_MAX_S / DECAY_MIN_S) ** frac


def seconds_to_decay(sec: float) -> float:
    return _clip01(math.log(max(sec, DECAY_MIN_S) / DECAY_MIN_S) / math.log(DECAY_MAX_S / DECAY_MIN_S))


class AmpArchetype(str, Enum):
    CLEAN = "clean"
    CRUNCH = "crunch"
    LEAD = "lead"
    HIGH_GAIN = "high_gain"
    ACOUSTIC = "acoustic"
    UNKNOWN = "unknown"


class OverdriveKind(str, Enum):
    BOOST = "boost"
    OVERDRIVE = "overdrive"
    DISTORTION = "distortion"
    FUZZ = "fuzz"


class ModKind(str, Enum):
    CHORUS = "chorus"
    PHASER = "phaser"
    FLANGER = "flanger"
    TREMOLO = "tremolo"
    ROTARY = "rotary"
    VIBRATO = "vibrato"
    OTHER = "other"


class DelayKind(str, Enum):
    DIGITAL = "digital"
    ANALOG = "analog"
    TAPE = "tape"
    PING_PONG = "ping_pong"
    REVERSE = "reverse"
    MODULATED = "modulated"
    OTHER = "other"


class ReverbKind(str, Enum):
    AMBIENCE = "ambience"
    ROOM = "room"
    HALL = "hall"
    PLATE = "plate"
    SPRING = "spring"
    OTHER = "other"


class _Block(BaseModel):
    model_config = ConfigDict(extra="forbid", use_enum_values=False)


class AmpBlock(_Block):
    archetype: AmpArchetype = AmpArchetype.UNKNOWN
    source_model: Optional[str] = Field(None, description="the device's own amp name, kept for provenance")
    gain: Unit = None
    bass: Unit = None
    mid: Unit = None
    treble: Unit = None
    presence: Unit = None
    master: Unit = Field(None, description="master / channel volume")


class OverdriveBlock(_Block):
    enabled: bool = False
    kind: OverdriveKind = OverdriveKind.OVERDRIVE
    source_model: Optional[str] = None
    drive: Unit = None
    tone: Unit = None
    level: Unit = None


class ModulationBlock(_Block):
    enabled: bool = False
    kind: ModKind = ModKind.OTHER
    source_model: Optional[str] = None
    rate: Unit = None
    depth: Unit = None
    mix: Unit = None


class DelayBlock(_Block):
    enabled: bool = False
    kind: DelayKind = DelayKind.DIGITAL
    source_model: Optional[str] = None
    time_ms: Optional[float] = Field(None, ge=0)
    feedback: Unit = None
    mix: Unit = None


class ReverbBlock(_Block):
    enabled: bool = False
    kind: ReverbKind = ReverbKind.HALL
    source_model: Optional[str] = None
    decay: Unit = Field(None, description="log-scaled fraction of 0.1-10 s")
    predelay_ms: Optional[float] = Field(None, ge=0)
    mix: Unit = None


class UniversalChain(_Block):
    amp: AmpBlock = Field(default_factory=AmpBlock)
    overdrive: OverdriveBlock = Field(default_factory=OverdriveBlock)
    modulation: ModulationBlock = Field(default_factory=ModulationBlock)
    delay: DelayBlock = Field(default_factory=DelayBlock)
    reverb: ReverbBlock = Field(default_factory=ReverbBlock)

    def to_features(self) -> tuple[np.ndarray, np.ndarray]:
        """Fixed-length numeric view for models: (values, known) with unknown knobs set to 0 and known=False."""
        vals: list[float] = []
        known: list[bool] = []

        def add(v: Optional[float], scale: float = 1.0):
            vals.append(0.0 if v is None else float(v) / scale)
            known.append(v is not None)

        def onehot(e: Enum, enum_cls):
            for m in enum_cls:
                vals.append(1.0 if e == m else 0.0)
                known.append(True)

        onehot(self.amp.archetype, AmpArchetype)
        for k in ("gain", "bass", "mid", "treble", "presence", "master"):
            add(getattr(self.amp, k))
        for name, blk, kind_cls, extra in (
            ("overdrive", self.overdrive, OverdriveKind, ("drive", "tone", "level")),
            ("modulation", self.modulation, ModKind, ("rate", "depth", "mix")),
            ("delay", self.delay, DelayKind, ("feedback", "mix")),
            ("reverb", self.reverb, ReverbKind, ("decay", "mix")),
        ):
            vals.append(float(blk.enabled))
            known.append(True)
            onehot(blk.kind, kind_cls)
            for k in extra:
                add(getattr(blk, k))
        add(self.delay.time_ms, 2000.0)
        add(self.reverb.predelay_ms, 500.0)
        return np.array(vals, dtype=np.float32), np.array(known, dtype=bool)


class PatchMeta(BaseModel):
    """What the website knew about the patch (the preset file itself rarely carries it)."""

    title: Optional[str] = None
    artist: Optional[str] = None
    song: Optional[str] = None
    author: Optional[str] = None
    rating: Optional[float] = Field(None, description="normalised to 0-5 stars")
    rating_count: Optional[int] = None
    downloads: Optional[int] = None
    source_url: Optional[str] = None
    tags: list[str] = Field(default_factory=list)
    meta_origin: str = Field("none", description="sidecar | manifest | filename | patch_name | none")


class PatchRecord(BaseModel):
    id: str
    source: str = Field(description="line6_customtone | boss_katana | boss_gt | neural_dsp | fender_mustang")
    device: str
    file: str
    patch_index: int = 0
    meta: PatchMeta = Field(default_factory=PatchMeta)
    chain: UniversalChain = Field(default_factory=UniversalChain)
    warnings: list[str] = Field(default_factory=list)
    native: Optional[dict] = Field(None, description="Neural DSP presets only: {plugin, vst_parameters (0-1, exact), raw, unmatched} of the plugin "
                                                      "the preset was saved from")
