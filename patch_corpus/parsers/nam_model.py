"""Neural Amp Modeler capture (``.nam``) and impulse-response (``.wav``) reader. SKELETON: container and metadata only.

A ``.nam`` file is JSON::

    {"version": "0.5.4", "architecture": "WaveNet" | "LSTM" | ..., "config": {...layer sizes...},
     "metadata": {"name", "modeled_by", "gear_make", "gear_model", "gear_type", "tone_type", "input_level_dbu", "output_level_dbu",
                  "loudness", "date": {...}, "training": {...}},       # every metadata key is optional
     "weights": [float, ...], "sample_rate": 48000}

A capture is a neural model of one rig at fixed knob positions: it carries NO knob values, so it is not a source of parameter labels. What it
gives you is the amp's sound (usable for audio-matching or as a reference tone) plus whatever the uploader wrote in ``metadata``. ``weights`` is
the bulk of the file and is only loaded on request.

``parse_ir`` reads the header of a WAV impulse response (channels, sample rate, length) so IRs can be filtered (cab IRs are usually under ~1 s).
"""

from __future__ import annotations

import json
import struct
import wave
from dataclasses import dataclass, field
from pathlib import Path

META_KEYS = ("name", "modeled_by", "gear_make", "gear_model", "gear_type", "tone_type", "input_level_dbu", "output_level_dbu", "loudness")


@dataclass
class NamModel:
    version: str
    architecture: str
    config: dict
    metadata: dict
    sample_rate: int | None
    n_weights: int
    weights: list[float] | None = None
    warnings: list[str] = field(default_factory=list)

    @property
    def name(self) -> str:
        return str(self.metadata.get("name") or "")

    @property
    def gear(self) -> str:
        return " ".join(str(self.metadata[k]) for k in ("gear_make", "gear_model") if self.metadata.get(k))


def parse_nam(path: str | Path, load_weights: bool = False) -> NamModel:
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(doc, dict) or "architecture" not in doc or "weights" not in doc:
        raise ValueError(f"{path}: not a Neural Amp Modeler file (needs 'architecture' and 'weights')")
    weights = doc.get("weights") or []
    meta = doc.get("metadata") if isinstance(doc.get("metadata"), dict) else {}
    model = NamModel(str(doc.get("version", "")), str(doc["architecture"]), doc.get("config") or {}, meta,
                     int(doc["sample_rate"]) if doc.get("sample_rate") else None, len(weights), list(weights) if load_weights else None)
    if not meta:
        model.warnings.append("no metadata block: gear and tone type unknown")
    if model.sample_rate is None:
        model.warnings.append("no sample_rate: older files default to 48000 Hz")
    return model


@dataclass
class ImpulseResponse:
    channels: int
    sample_rate: int
    frames: int
    sample_bits: int
    warnings: list[str] = field(default_factory=list)

    @property
    def seconds(self) -> float:
        return self.frames / self.sample_rate if self.sample_rate else 0.0


def parse_ir(path: str | Path) -> ImpulseResponse:
    try:
        with wave.open(str(path), "rb") as w:                                      # integer PCM
            return ImpulseResponse(w.getnchannels(), w.getframerate(), w.getnframes(), w.getsampwidth() * 8)
    except wave.Error:
        pass
    data = Path(path).read_bytes()                                                 # IEEE-float WAV (format 3): read the chunks by hand
    if data[:4] != b"RIFF" or data[8:12] != b"WAVE":
        raise ValueError(f"{path}: not a WAV file")
    pos, fmt, size = 12, None, 0
    while pos + 8 <= len(data):
        tag, n = data[pos:pos + 4], struct.unpack("<I", data[pos + 4:pos + 8])[0]
        if tag == b"fmt ":
            fmt = struct.unpack("<HHIIHH", data[pos + 8:pos + 24])
        elif tag == b"data":
            size = n
        pos += 8 + n + (n & 1)
    if fmt is None:
        raise ValueError(f"{path}: WAV has no fmt chunk")
    _, channels, rate, _, align, bits = fmt
    ir = ImpulseResponse(channels, rate, size // align if align else 0, bits)
    ir.warnings.append(f"WAV format tag {fmt[0]} (float or extensible), read from the header only")
    return ir
