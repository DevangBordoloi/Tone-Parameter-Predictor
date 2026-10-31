"""Fender FUSE preset (``.fuse``) reader for the Mustang amps (GuitarPatches ``mustang1`` / ``mustang3``).

A ``.fuse`` file is plain UTF-8 XML (with a BOM), checked on a real GuitarPatches download::

    <Preset amplifier="Mustang I/II" ProductId="1">
      <Amplifier><Module ID="97" POS="0" BypassState="1"><Param ControlIndex="0">51143</Param> ... </Module></Amplifier>
      <FX><Stompbox ID="1"><Module ID="7" POS="5" BypassState="1"><Param ControlIndex="0">39321</Param> ...
          <Modulation ID="2">...  <Delay ID="3">...  <Reverb ID="4"><Module ID="36" POS="4" BypassState="0">...
      <FUSE><Info name="Day Tripper Beatles" author="..." rating="45" genre1="40" genre2="-1" genre3="-1" tags="Beatles, Day Tripper">notes</Info>

The module ids and the meaning of the amp's ``ControlIndex`` values are not in the file. They are taken from the open-source PLUG editor
(github.com/offa/plug, ``src/ui/loadfromfile.cpp``, GPL), which reads exactly these files: amp knobs 0 volume, 1 gain, 2 gain2, 3 master, 4 treble,
5 middle, 6 bass, 7 presence, 9 depth, 10 bias (16-bit values; PLUG keeps ``value >> 8``, the amp's 0-255 scale), 15 noise gate, 16 threshold,
17 cabinet, 19 sag, 20 brightness. Effect knobs are per-effect and are not decoded here, only the effect and whether it is switched on.

NOT VERIFIED here: ``BypassState`` (1 is read as "on": the amp module and the compressor of the sample carry 1, its reverb 0, and the author's note
says "add Reverb"); the ``genre1..3`` codes are Fender FUSE's own numbering and are kept raw (no public table was found).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from xml.etree import ElementTree

AMPS = {0x67: "'57 Deluxe", 0x64: "'59 Bassman", 0x7C: "'57 Champ", 0x53: "'65 Deluxe Reverb", 0x6A: "'65 Princeton", 0x75: "'65 Twin Reverb",
        0x72: "Super-Sonic", 0x61: "British '60s", 0x79: "British '70s", 0x5E: "British '80s", 0x5D: "American '90s", 0x6D: "Metal 2000",
        0xF1: "Studio Preamp", 0xF6: "'57 Twin", 0xF9: "'60s Thrift", 0xFC: "British Colour", 0xFF: "British Watts"}
STOMPBOX = {0x3C: "Overdrive", 0x49: "Wah", 0x4A: "Touch Wah", 0x1A: "Fuzz", 0x1C: "Fuzz Touch Wah", 0x88: "Simple Comp", 0x07: "Compressor",
            0x103: "Ranger Boost", 0xBA: "Green Box", 0x110: "Orange Box", 0x111: "Black Box", 0x10F: "Big Fuzz", 0xF4: "Wah", 0xF5: "Touch Wah"}
MODULATION = {0x12: "Sine Chorus", 0x13: "Triangle Chorus", 0x18: "Sine Flanger", 0x19: "Triangle Flanger", 0x2D: "Vibratone", 0x40: "Vintage Tremolo",
              0x41: "Sine Tremolo", 0x22: "Ring Modulator", 0x29: "Step Filter", 0x4F: "Phaser", 0x1F: "Pitch Shifter", 0x11F: "Diatonic Pitch Shifter"}
DELAY = {0x16: "Mono Delay", 0x43: "Mono Echo Filter", 0x48: "Stereo Echo Filter", 0x44: "Multitap Delay", 0x45: "Ping Pong Delay", 0x15: "Ducking Delay",
         0x46: "Reverse Delay", 0x2B: "Tape Delay", 0x2A: "Stereo Tape Delay"}
REVERB = {0x24: "Small Hall", 0x3A: "Large Hall", 0x26: "Small Room", 0x3B: "Large Room", 0x4E: "Small Plate", 0x4B: "Large Plate", 0x4C: "Ambient",
          0x4D: "Arena", 0x21: "'63 Spring", 0x0B: "'65 Spring"}
AMP_KNOBS = {0: "volume", 1: "gain", 2: "gain2", 3: "master", 4: "treble", 5: "middle", 6: "bass", 7: "presence", 9: "depth", 10: "bias"}
AMP_SMALL = {15: "noise_gate", 16: "threshold", 17: "cabinet", 19: "sag", 20: "brightness"}


@dataclass
class FuseModule:
    kind: str                          # stompbox / modulation / delay / reverb
    module_id: int
    name: str                          # "" for an empty slot
    position: int
    on: bool
    params: dict[int, int] = field(default_factory=dict)


@dataclass
class FusePatch:
    name: str
    amplifier: str                     # "Mustang I/II"
    amp_id: int | None
    amp_name: str
    amp: dict[str, int]                # volume / gain / ... -> 0-255 ; noise_gate / cabinet ... -> raw
    effects: list[FuseModule] = field(default_factory=list)
    author: str = ""
    tags: list[str] = field(default_factory=list)
    rating: float | None = None
    genre_codes: tuple[int, ...] = ()
    note: str = ""
    warnings: list[str] = field(default_factory=list)

    def active(self, kind: str) -> FuseModule | None:
        return next((m for m in self.effects if m.kind == kind and m.name and m.on), None)


def is_fuse(data: bytes) -> bool:
    """Plain-XML check used by the crawler: a <Preset> root with an <Amplifier> section."""
    try:
        root = ElementTree.fromstring(data.decode("utf-8-sig"))
    except (ElementTree.ParseError, UnicodeDecodeError):
        return False
    return root.tag == "Preset" and root.find("Amplifier") is not None


def _int(s, default=0) -> int:
    try:
        return int(str(s).strip())
    except (TypeError, ValueError):
        return default


def parse_fuse(path: str | Path) -> FusePatch:
    root = ElementTree.fromstring(Path(path).read_text(encoding="utf-8-sig"))
    if root.tag != "Preset" or root.find("Amplifier") is None:
        raise ValueError(f"{path}: not a Fender FUSE preset")
    info = root.find("FUSE/Info")
    mod = root.find("Amplifier/Module")
    amp_id = _int(mod.get("ID"), -1) if mod is not None else -1
    amp: dict[str, int] = {}
    if mod is not None:
        for p in mod.findall("Param"):
            i, v = _int(p.get("ControlIndex"), -1), _int(p.text)
            if i in AMP_KNOBS:
                amp[AMP_KNOBS[i]] = v >> 8                                  # 16-bit -> the amp's 0-255 scale (as PLUG does)
            elif i in AMP_SMALL:
                amp[AMP_SMALL[i]] = v
    patch = FusePatch(name=(info.get("name") if info is not None else "") or Path(path).stem, amplifier=root.get("amplifier") or "",
                      amp_id=amp_id if amp_id >= 0 else None, amp_name=AMPS.get(amp_id, f"amp 0x{amp_id:02X}"), amp=amp)
    if amp_id not in AMPS:
        patch.warnings.append(f"unknown amp module id {amp_id}")
    if info is not None:
        patch.author = info.get("author") or ""
        patch.tags = [t.strip() for t in (info.get("tags") or "").split(",") if t.strip()]
        patch.rating = _int(info.get("rating"), -1) / 10.0 if _int(info.get("rating"), -1) >= 0 else None        # 45 -> 4.5 stars (assumed)
        patch.genre_codes = tuple(c for c in (_int(info.get(f"genre{i}"), -1) for i in (1, 2, 3)) if c >= 0)
        patch.note = (info.text or "").strip()
    tables = {"Stompbox": ("stompbox", STOMPBOX), "Modulation": ("modulation", MODULATION), "Delay": ("delay", DELAY), "Reverb": ("reverb", REVERB)}
    for tag, (kind, table) in tables.items():
        for slot in root.findall(f"FX/{tag}"):
            for m in slot.findall("Module"):
                mid = _int(m.get("ID"))
                patch.effects.append(FuseModule(kind, mid, table.get(mid, "" if mid == 0 else f"effect 0x{mid:02X}"), _int(m.get("POS")),
                                                m.get("BypassState") == "1", {_int(p.get("ControlIndex")): _int(p.text) for p in m.findall("Param")}))
    return patch
