"""Locate and load the licensed Neural DSP VST3 plugins (headless, through Spotify's pedalboard)."""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path


def vst3_search_dirs() -> list[Path]:
    dirs: list[Path] = []
    if os.environ.get("NEURAL_DSP_VST3_DIR"):
        dirs.append(Path(os.environ["NEURAL_DSP_VST3_DIR"]).expanduser())
    if sys.platform == "win32":
        for var in ("COMMONPROGRAMFILES", "COMMONPROGRAMFILES(X86)"):
            if os.environ.get(var):
                dirs.append(Path(os.environ[var]) / "VST3")
        dirs += [Path(r"C:\Program Files\Common Files\VST3"), Path(r"C:\Program Files (x86)\Common Files\VST3")]
    elif sys.platform == "darwin":
        dirs += [Path("/Library/Audio/Plug-Ins/VST3"), Path.home() / "Library/Audio/Plug-Ins/VST3"]
    else:
        dirs += [Path("/usr/lib/vst3"), Path("/usr/local/lib/vst3"), Path.home() / ".vst3"]
    return [d for d in dict.fromkeys(dirs) if d.is_dir()]


def _key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().removesuffix(".vst3"))


def find_vst3(name_or_path: str | os.PathLike[str]) -> Path:
    """Resolve a path or a (partial) plugin name such as "Nolly X" to an installed .vst3."""
    p = Path(name_or_path).expanduser()
    if p.exists():
        return p
    want = _key(str(name_or_path))
    found = {c.resolve(): c for d in vst3_search_dirs() for c in d.rglob("*.vst3") if want and want in _key(c.stem)}
    cands = list(found.values())
    exact = [c for c in cands if _key(c.stem) == want]
    cands = exact or cands
    if len(cands) != 1:
        raise FileNotFoundError(f"{name_or_path!r}: {len(cands)} .vst3 matches in {[str(d) for d in vst3_search_dirs()]}")
    return cands[0]


def installed_plugins(names: list[str]) -> list[str]:
    """The subset of plugin names that have a .vst3 on this machine (all of them if TONE_ASSUME_INSTALLED=1, e.g. in
    Docker, where presets can still be predicted but not rendered)."""
    if os.environ.get("TONE_ASSUME_INSTALLED") == "1":
        return list(names)
    out = []
    for n in names:
        try:
            find_vst3(n)
            out.append(n)
        except FileNotFoundError:
            pass
    return out


def load_plugin(name_or_path: str | os.PathLike[str]):
    import pedalboard

    return pedalboard.load_plugin(str(find_vst3(name_or_path)))
