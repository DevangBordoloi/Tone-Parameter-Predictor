"""Headless rendering of a parameter vector through a plugin on a DI clip (for ITO, the check and previews)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from common.dsp import resample
from common.paths import DATA_DIR, SAMPLE_RATE, ensure
from common.schema import PluginSchema
from dataset_builder.di_corpus import PREROLL_SEC, ClipBank, synthetic_bank
from dataset_builder.render_dataset import load_plugin, render_clip


def held_out_di(bank_dir: str | Path | None = None, index: int = 0) -> np.ndarray:
    """A DI clip (PREROLL + 4 s at 48 kHz) from the *test* split, or a synthetic one if no bank has been built."""
    d = Path(bank_dir) if bank_dir else DATA_DIR / "di_bank"
    if not (d / "test.npy").exists() and not (d / "train.npy").exists():
        synthetic_bank(ensure(d), 4)
    bank = ClipBank("test", d)
    return bank.get(index % len(bank))


class PluginRenderer:
    def __init__(self, schema: PluginSchema, out_sr: int, plugin: Any | None = None, di: np.ndarray | None = None):
        self.schema, self.out_sr = schema, out_sr
        self.plugin = plugin or load_plugin(schema)
        self.di = di if di is not None else held_out_di()

    def __call__(self, values: np.ndarray, di: np.ndarray | None = None) -> np.ndarray:
        """normalized vector -> wet (2, out_sr * 4) with the pre-roll removed."""
        wet = render_clip(self.plugin, self.schema, self.di if di is None else di, values, SAMPLE_RATE)
        wet = wet[:, int(PREROLL_SEC * SAMPLE_RATE):]
        return resample(wet, SAMPLE_RATE, self.out_sr)

    def write_preview(self, values: np.ndarray, path: str | Path) -> Path:
        import soundfile as sf

        ensure(Path(path).parent)
        sf.write(str(path), self(values).T, self.out_sr, subtype="PCM_24")
        return Path(path)
