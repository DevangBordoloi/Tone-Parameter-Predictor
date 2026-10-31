"""Dump the raw bytes of a Katana .tsl so the address map can be checked against real files.

    python -m patch_corpus.inspect_tsl liveset.tsl [--patch 0]

Prints, per memory group, its length and the bytes at the addresses patch_corpus uses (marking the UNVERIFIED ones), so you can
turn a knob on the amp, export, and see which byte moved.
"""

from __future__ import annotations

import argparse

from patch_corpus.mapping.katana import KATANA_ADDRESSES
from patch_corpus.parsers.katana_tsl import GROUP_BASE, parse_tsl


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file")
    ap.add_argument("--patch", type=int, default=0)
    a = ap.parse_args(argv)
    patches = parse_tsl(a.file)
    p = patches[a.patch]
    print(f"{len(patches)} patches; showing #{a.patch}: {p.name!r}  memo={p.memo!r}  formatRev={p.format_rev}")
    for g, data in p.groups.items():
        base = GROUP_BASE.get(g)
        print(f"{g:28s} {len(data):4d} bytes" + (f"  base {base:08X}" if base else ""))
    print("\nparameters used:")
    for name, a_ in KATANA_ADDRESSES.items():
        v = p.word7(a_.group, a_.address) if a_.wide else p.byte(a_.group, a_.address)
        print(f"  {name:18s} {a_.address:08X}  {'' if v is None else v:>5}  {'' if a_.verified else 'UNVERIFIED'}")


if __name__ == "__main__":
    main()
