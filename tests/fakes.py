"""A stand-in for a pedalboard plugin so the data pipeline and training can be exercised without the licensed VST3s."""

from __future__ import annotations

import numpy as np
from scipy import signal

from common.schema import PluginSchema


class FakeParam:
    def __init__(self, default: float):
        self.raw_value = default


class FakePlugin:
    """Parameter-dependent distortion + tone filter + feedback delay + tremolo, so a model has something to learn."""

    def __init__(self, schema: PluginSchema):
        self.schema = schema
        self.parameters = {p.name: FakeParam(p.default_normalized) for p in schema.parameters}

    def _mean(self, *needles: str) -> float:
        vals = [v.raw_value for k, v in self.parameters.items() if any(n in k for n in needles)]
        return float(np.mean(vals)) if vals else 0.5

    def __call__(self, audio: np.ndarray, sr: int, reset: bool = True) -> np.ndarray:
        drive = 1.0 + 25.0 * self._mean("gain")
        tone = 0.1 + 0.8 * self._mean("treble", "tone", "presence")
        wet = self._mean("dry_wet", "_mix")
        rate = 0.5 + 8.0 * self._mean("rate", "speed")
        y = np.tanh(drive * audio) / np.tanh(drive)
        b, a = signal.butter(2, min(0.45, 0.02 + tone * 0.4))
        y = signal.lfilter(b, a, y, axis=-1)
        d = int(sr * 0.18)
        y[:, d:] += wet * 0.6 * y[:, :-d]
        t = np.arange(y.shape[1]) / sr
        y = y * (1 - 0.4 * self._mean("depth", "mod") * (0.5 + 0.5 * np.sin(2 * np.pi * rate * t)))
        y[1] = np.roll(y[1], int(self._mean("pan", "spread") * 40))  # a little stereo difference for the Side channel
        return y.astype(np.float32)
