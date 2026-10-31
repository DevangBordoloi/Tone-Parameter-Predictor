"""Stage 1 of the pipeline: song (URL or file) -> isolated guitar stem WAV.

    python -m guitar_stem_isolator.isolate "https://youtu.be/..." --start 30 --duration 25 --focus lead
    python -m guitar_stem_isolator.isolate my_song.wav --is-stem          # input is already a guitar track
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

from common.paths import OUTPUT_DIR, ensure
from guitar_stem_isolator.downloader import resolve_source
from guitar_stem_isolator.separator import Focus, assess_stem, separate_guitar

log = logging.getLogger(__name__)


@dataclass
class IsolatedStem:
    source: str
    audio_path: str          # the (cut) mix
    stem_path: str           # stereo guitar stem, input sample rate
    focus: str
    strategy: str | None
    report: dict


def isolate(source: str, start_sec: float | None = 30, duration_sec: float | None = 25, focus: Focus = "rhythm",
            is_guitar_stem: bool = False, strategy: str | None = None, work_dir: str | Path | None = None) -> IsolatedStem:
    """Resolve the source, cut the window, and (unless it already is a guitar track) run Demucs.

    `focus` is carried along for the later stages: "rhythm" analyses the Side channel of a double-tracked part, "lead" the Mid
    channel. The stem written here is always the full stereo guitar so either can be taken afterwards.
    """
    work = ensure(Path(work_dir) if work_dir else OUTPUT_DIR / "audio")
    audio = resolve_source(source, start_sec, duration_sec, work)
    if is_guitar_stem:
        stem = audio
    else:
        stem = separate_guitar(audio, out_dir=work / "stems", strategy=strategy)
    return IsolatedStem(source, str(audio), str(stem), focus, None if is_guitar_stem else (strategy or "cascade"),
                        assess_stem(audio, stem))


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--start", type=float, default=30)
    ap.add_argument("--duration", type=float, default=25)
    ap.add_argument("--focus", choices=["rhythm", "lead", "full"], default="rhythm")
    ap.add_argument("--strategy", choices=["single", "cascade"])
    ap.add_argument("--is-stem", action="store_true")
    ap.add_argument("--work-dir")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    res = isolate(a.source, a.start, a.duration, a.focus, a.is_stem, a.strategy, a.work_dir)
    print(json.dumps(asdict(res), indent=2))


if __name__ == "__main__":
    main()
