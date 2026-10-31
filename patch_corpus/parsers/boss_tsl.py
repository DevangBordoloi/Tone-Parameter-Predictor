"""Boss Tone Studio liveset (.tsl) reader for the GT / ME / GX multi-effects (GT-100, GT-1, GT-1000, ME-80, ME-90, GX-100 ...).

File layout (checked on real GuitarPatches downloads for GT-100, GT-1 and ME-80)::

    {"device": "GT" | "GT-1" | "ME-80" | ..., "version": ..., "liveSetData": {"name": ..., "image": ...},
     "patchList": [ {"name": "...", "note": ..., "category": ..., "params": {"preamp_a_gain": "97", "delay_f_back": "25", ...}}, ... ]}

Unlike the Katana liveset (hex bytes at memory addresses, ``katana_tsl.py``), these units store NAMED parameters as decimal strings, so no
address table is needed: ``params`` maps the editor's parameter name to an int (``"06"``-style hex-looking strings are decimal too). Some
units spell the patch name as character codes (``patch_name1..16`` on GT-1, ``name1..16`` on ME-80); ``name`` / ``patchname`` win when present.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

NAME_CODE_PREFIXES = ("patch_name", "name", "gt100Name")


@dataclass
class BossPatch:
    name: str
    device: str
    params: dict[str, int]
    index: int = 0
    note: str = ""
    category: str = ""
    liveset: str = ""
    warnings: list[str] = field(default_factory=list)

    def get(self, *names: str) -> int | None:
        """First of ``names`` that the patch has (units name the same knob differently)."""
        return next((self.params[n] for n in names if n in self.params), None)


def _int(v) -> int | None:
    if isinstance(v, bool):
        return int(v)
    if isinstance(v, (int, float)):
        return int(v)
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return None


def _name_from_codes(params: dict, prefix: str) -> str:
    codes = [params.get(f"{prefix}{i}") for i in range(1, 17)]
    chars = [chr(c) for c in (_int(x) for x in codes) if c is not None and 32 <= c < 127]
    return "".join(chars).strip()


def is_boss_liveset(doc: dict) -> bool:
    return isinstance(doc, dict) and isinstance(doc.get("patchList"), list) and not str(doc.get("device", "")).upper().startswith("KATANA")


def parse_boss_tsl(path: str | Path) -> list[BossPatch]:
    doc = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not is_boss_liveset(doc):
        raise ValueError(f"{path}: not a Boss GT/ME liveset (device {doc.get('device')!r})")
    device = str(doc.get("device", ""))
    liveset = str((doc.get("liveSetData") or {}).get("name") or "")
    out: list[BossPatch] = []
    for i, entry in enumerate(doc["patchList"]):
        raw = entry.get("params") if isinstance(entry, dict) else None
        if not isinstance(raw, dict):
            continue
        params = {k: n for k, v in raw.items() if (n := _int(v)) is not None}
        name = str(entry.get("name") or raw.get("patchname") or "").strip()
        if not name:
            name = next((s for p in NAME_CODE_PREFIXES if (s := _name_from_codes(raw, p))), "") or f"patch{i}"
        p = BossPatch(name, device, params, i, str(entry.get("note") or ""), str(entry.get("category") or ""), liveset)
        if not params:
            p.warnings.append("patch has no numeric parameters")
        out.append(p)
    return out
