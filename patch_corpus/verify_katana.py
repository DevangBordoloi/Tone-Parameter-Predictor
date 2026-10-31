"""Check the Katana address map against real files: ``python -m patch_corpus.verify_katana [presets/katana]``.

For every byte of ``UserPatch%Patch_0`` over all MkII patches it prints the number of distinct values, the range, the mean and the most
common values, then for the addresses ``mapping/katana.py`` reads it says what the distribution looks like:

  knob      many distinct values inside 0-100                       (a real control)
  constant  one value in every patch                                (NOT a control: reading it gives every record the same number)
  switch    values 0 / 1 only
  type      few distinct values, as expected for an enumeration

and how the gain candidate follows the patch character (names containing clean / lead / metal ...). It cannot say which knob is
which: the names come from TuxKatana's address files, the distributions only rule addresses out.
"""

from __future__ import annotations

import argparse
import re
import statistics as st
from collections import Counter
from pathlib import Path

from patch_corpus.mapping.katana import KATANA_ADDRESSES
from patch_corpus.parsers.katana_tsl import GROUP_BASE, parse_tsl

P0 = "UserPatch%Patch_0"
KEYWORDS = {"clean": r"\bclean|\bcln|\bsparkle|\bchime", "crunch": r"crunch|blues|\bedge", "lead/solo": r"\blead|\bsolo",
            "high gain / metal": r"metal|high ?gain|\bhg\b|\bheavy|\bbrutal|djent|\bdist"}


def load(root: Path) -> list[tuple[str, list[int]]]:
    rows = []
    for f in sorted(root.glob("*.tsl")):
        try:
            for p in parse_tsl(f):
                if p.device == "KATANA MkII" and len(p.groups.get(P0, [])) >= 0x48:
                    rows.append((f"{p.name} {p.memo}".lower(), p.groups[P0]))
        except Exception:
            continue
    return rows


def classify(col: list[int]) -> str:
    uniq = set(col)
    if len(uniq) == 1:
        return f"constant ({col[0]})"
    if uniq <= {0, 1}:
        return "switch"
    if len(uniq) <= 40 and max(col) <= 100 and st.mean(col) < 30:
        return "type / enumeration"
    return "knob" if max(col) <= 100 else "out of 0-100"


def report(root: Path) -> dict[str, str]:
    rows = load(root)
    print(f"{len(rows)} MkII patches in {root}")
    base = GROUP_BASE[P0]
    out = {}
    print(f"\n{'name':16s} {'address':>9} {'uniq':>4} {'min':>3} {'max':>3} {'mean':>5}  verdict")
    for name, a in KATANA_ADDRESSES.items():
        if a.group != P0:
            continue
        col = [r[a.address - base] for _, r in rows]
        out[name] = classify(col)
        print(f"{name:16s} {a.address:08X} {len(set(col)):4d} {min(col):3d} {max(col):3d} {st.mean(col):5.1f}  {out[name]}"
              f"{'' if a.verified else '   [inferred in the address map]'}")
    cand = [0x22, 0x27, 0x28]
    print(f"\ncandidate bytes by patch name:   " + "  ".join(f"{c:#04x}" for c in cand))
    for k, rx in KEYWORDS.items():
        sel = [r for n, r in rows if re.search(rx, n)]
        if sel:
            print(f"  {k:20s} n={len(sel):3d}   " + "  ".join(f"{st.mean(r[c - 0x10] for r in sel):5.1f}" for c in cand))
    const = [f"{base + i:08X}" for i in range(0x48) if len({r[i] for _, r in rows}) == 1]
    print(f"\n{len(const)} of 72 bytes of Patch_0 are constant across all patches (not controls): {' '.join(const[:12])} ...")
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", nargs="?", default="data/data_scraping/presets/katana")
    report(Path(ap.parse_args(argv).root))


if __name__ == "__main__":
    main()
