"""Augmentations that narrow the gap between a clean plugin render and a Demucs guitar stem from a real song.

Applied to the wet (stereo) render: random EQ, room reverb, bus compression, noise, a random level. An optional
`stem_roundtrip` mixes the render with a backing track and runs it through the real separator, so the model also sees
separation artefacts (slow, needs demucs; used on a small subset).
"""

from __future__ import annotations

import numpy as np
from scipy import signal


def _peaking_biquad(f0: float, gain_db: float, q: float, sr: int) -> tuple[np.ndarray, np.ndarray]:
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * a, -2 * np.cos(w0), 1 - alpha * a])
    den = np.array([1 + alpha / a, -2 * np.cos(w0), 1 - alpha / a])
    return b / den[0], den / den[0]


def random_eq(x: np.ndarray, sr: int, rng: np.random.Generator, max_gain_db: float = 6.0) -> np.ndarray:
    """2-3 random peaking filters, applied identically to every channel of x (C, N)."""
    y = x
    for _ in range(int(rng.integers(2, 4))):
        f0 = float(np.exp(rng.uniform(np.log(120), np.log(min(8000, sr * 0.4)))))
        b, a = _peaking_biquad(f0, float(rng.uniform(-max_gain_db, max_gain_db)), float(rng.uniform(0.5, 2.5)), sr)
        y = signal.lfilter(b, a, y, axis=-1)
    return y.astype(np.float32)


def room_reverb(x: np.ndarray, sr: int, rng: np.random.Generator, wet_db_range: tuple[float, float] = (-30, -14)) -> np.ndarray:
    """Convolve with a synthetic exponentially decaying noise IR (RT60 0.2-0.9 s), mixed in below the dry signal."""
    rt60 = rng.uniform(0.2, 0.9)
    n = int(sr * min(rt60 * 1.2, 1.0))
    t = np.arange(n) / sr
    ir = rng.standard_normal((x.shape[0], n)) * np.exp(-6.9 * t / rt60)
    ir[:, 0] = 0
    ir /= np.sqrt((ir**2).sum(axis=-1, keepdims=True)) + 1e-9
    wet = np.stack([signal.fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(x.shape[0])])
    return (x + wet * 10 ** (rng.uniform(*wet_db_range) / 20)).astype(np.float32)


def bus_compress(x: np.ndarray, sr: int, rng: np.random.Generator) -> np.ndarray:
    """Feed-forward peak compressor on the linked stereo envelope (mix-bus style, gentle)."""
    thr_db, ratio = rng.uniform(-24, -10), rng.uniform(1.5, 4.0)
    att, rel = np.exp(-1 / (sr * rng.uniform(0.005, 0.03))), np.exp(-1 / (sr * rng.uniform(0.05, 0.3)))
    level = np.abs(x).max(axis=0)
    env = np.empty_like(level)
    e = 0.0
    # block-wise one-pole follower (vectorised enough for 4 s clips)
    for i in range(0, len(level), 64):
        peak = float(level[i : i + 64].max())
        coef = att if peak > e else rel
        e = coef * e + (1 - coef) * peak
        env[i : i + 64] = e
    over = np.maximum(20 * np.log10(env + 1e-9) - thr_db, 0.0)
    gain = 10 ** (-(over * (1 - 1 / ratio)) / 20)
    return (x * gain[None]).astype(np.float32)


def add_noise(x: np.ndarray, rng: np.random.Generator, snr_db_range: tuple[float, float] = (35, 70)) -> np.ndarray:
    rms = float(np.sqrt(np.mean(x**2))) + 1e-9
    noise = rng.standard_normal(x.shape)
    if rng.random() < 0.5:  # pink-ish
        noise = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786], [1, -2.494956002, 2.017265875, -0.522189400], noise, axis=-1)
    noise *= rms * 10 ** (-rng.uniform(*snr_db_range) / 20) / (np.sqrt(np.mean(noise**2)) + 1e-9)
    return (x + noise).astype(np.float32)


def random_level(x: np.ndarray, rng: np.random.Generator, db_range: tuple[float, float] = (-12.0, 6.0)) -> np.ndarray:
    return (x * 10 ** (rng.uniform(*db_range) / 20)).astype(np.float32)


def augment(x: np.ndarray, sr: int, rng: np.random.Generator, p: dict[str, float] | None = None) -> np.ndarray:
    """x: (C, N) float32 wet render. Each stage fires with its own probability."""
    p = {"eq": 0.7, "reverb": 0.4, "comp": 0.4, "noise": 0.5, "level": 1.0, **(p or {})}
    y = x
    if rng.random() < p["eq"]:
        y = random_eq(y, sr, rng)
    if rng.random() < p["reverb"]:
        y = room_reverb(y, sr, rng)
    if rng.random() < p["comp"]:
        y = bus_compress(y, sr, rng)
    if rng.random() < p["noise"]:
        y = add_noise(y, rng)
    if rng.random() < p["level"]:
        y = random_level(y, rng)
    return np.nan_to_num(y).astype(np.float32)


def stem_roundtrip(x: np.ndarray, sr: int, backing: np.ndarray, rng: np.random.Generator, work_dir: str) -> np.ndarray:
    """Mix `x` (C, N) with a backing-track excerpt (C, N) at -6..-18 dB and re-extract the guitar with the project's
    separator. Returns a (2, N) array. Needs demucs; callers should run it on a small fraction of the data."""
    import os

    import soundfile as sf

    from guitar_stem_isolator.separator import separate_guitar

    os.makedirs(work_dir, exist_ok=True)
    n = x.shape[1]
    mix = x[:2] + backing[:2, :n] * 10 ** (-rng.uniform(6, 18) / 20)
    src = os.path.join(work_dir, "rt_mix.wav")
    sf.write(src, mix.T, sr, subtype="PCM_24")
    stem = separate_guitar(src, out_dir=work_dir)
    data, _ = sf.read(str(stem), dtype="float32", always_2d=True)
    return data.T[:, :n]
