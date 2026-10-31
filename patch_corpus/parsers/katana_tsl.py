"""Boss Tone Studio liveset (.tsl) reader for the Katana.

File layout (verified against the open-source TuxKatana parser):

    {"name": "...", "formatRev": "0001", "device": "KATANA MkII",
     "data": [[ {"memo": "...", "paramSet": {"UserPatch%PatchName": ["4d","79", ...16 hex bytes],
                                              "UserPatch%Patch_0":  [72 hex bytes],     # booster + preamp
                                              "UserPatch%Fx(1)":    [221 bytes],        # MOD
                                              "UserPatch%Fx(2)":    [221 bytes],        # FX
                                              "UserPatch%Delay(1)": [26 bytes], "UserPatch%Delay(2)": [26 bytes],
                                              "UserPatch%Patch_1":  [50 bytes],         # reverb
                                              ...}}, ... ]]}

Each ``paramSet`` value is a list of hex-byte strings. The group's base address (``GROUP_BASE``, from TuxKatana's
``presets_addrs.yaml``) plus an index into the list gives the parameter's Katana memory address, so a parameter at
``60 00 00 24`` is byte ``0x24 - 0x10 = 20`` of ``UserPatch%Patch_0``.

``Patch`` exposes ``byte(group, address)`` / ``word7(group, address)`` so mappings are written with the documented absolute
addresses, not with hand-counted offsets. The 7-bit pair encoding (``0f 50`` = 15*128+80 = 2000 ms) matches the ranges
given for delay time and reverb pre-delay.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

GROUP_BASE: dict[str, int] = {
    "UserPatch%PatchName": 0x60000000,
    "UserPatch%Patch_0": 0x60000010,
    "UserPatch%Eq(2)": 0x60000060,
    "UserPatch%Fx(1)": 0x60000100,
    "UserPatch%Fx(2)": 0x60000300,
    "UserPatch%Delay(1)": 0x60000500,
    "UserPatch%Delay(2)": 0x60000520,
    "UserPatch%Patch_1": 0x60000540,
    "UserPatch%Patch_2": 0x60000620,
}


def _to_int(tok) -> int:
    if isinstance(tok, int):
        return tok
    return int(str(tok).strip(), 16)


@dataclass
class KatanaPatch:
    name: str
    memo: str
    groups: dict[str, list[int]]
    index: int = 0
    device: str = ""
    format_rev: str = ""
    warnings: list[str] = field(default_factory=list)

    def byte(self, group: str, address: int) -> int | None:
        """Value at an absolute Katana address inside `group`, or None when the group is short / missing."""
        data = self.groups.get(group)
        off = address - GROUP_BASE[group]
        if data is None or not 0 <= off < len(data):
            return None
        return data[off]

    def word7(self, group: str, address: int) -> int | None:
        """Two 7-bit bytes (hi, lo) -> hi*128 + lo."""
        hi, lo = self.byte(group, address), self.byte(group, address + 1)
        return None if hi is None or lo is None else hi * 128 + lo


def decode_name(tokens: list[int]) -> str:
    return "".join(chr(b) if 32 <= b < 127 else " " for b in tokens).strip()


def parse_tsl(path: str | Path) -> list[KatanaPatch]:
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    device = str(doc.get("device", ""))
    if not device.upper().startswith("KATANA"):
        raise ValueError(f"{path}: device {device!r} is not a Katana liveset")
    rev = str(doc.get("formatRev", ""))
    rows = doc.get("data") or []
    patches_raw = rows[0] if rows and isinstance(rows[0], list) else rows       # data is [[patch, patch, ...]]
    out: list[KatanaPatch] = []
    for i, entry in enumerate(patches_raw):
        ps = entry.get("paramSet") if isinstance(entry, dict) else None
        if not isinstance(ps, dict):
            continue
        groups = {k: [_to_int(t) for t in v] for k, v in ps.items() if isinstance(v, list)}
        p = KatanaPatch(decode_name(groups.get("UserPatch%PatchName", [])) or f"patch{i}", str(entry.get("memo", "") or ""),
                        groups, i, device, rev)
        if "mk2" not in device.lower().replace(" ", ""):
            p.warnings.append(f"device {device!r}: address map was written for the KATANA MkII")
        out.append(p)
    return out
