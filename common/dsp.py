"""Small numpy/scipy DSP helpers shared by the feature extractor, sampler diversity probe and the evaluation metrics."""

from __future__ import annotations

import numpy as np
from scipy import fft as sfft
from scipy import signal

BAND_NAMES = ("sub_thump", "low_body", "low_mid_boxiness", "mid_bark", "high_mid_bite", "presence_air")
BAND_EDGES_HZ = ((60, 120), (120, 250), (250, 600), (600, 1600), (1600, 4000), (4000, 10000))


def db(x: float, floor: float = 1e-12) -> float:
    return float(20 * np.log10(max(float(x), floor)))


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


PEAK_LIMIT = 2.0     # a rendered row beyond either limit is an exploded / mis-gained render, not a guitar signal
RMS_LIMIT = 1.0


def sanitize_audio(x: np.ndarray, peak_limit: float = PEAK_LIMIT, rms_limit: float = RMS_LIMIT) -> tuple[np.ndarray, np.ndarray]:
    """Peak-normalise every row of `x` (rows, ..., S) whose peak or RMS exceeds the limits to a peak of 1.0 (so it lies in
    [-1, 1]). Peak-normalising keeps the spectral shape, where clamping would add distortion; rows within the limits are left
    untouched. Non-finite samples become 0. Returns (audio float32, flagged bool (rows,))."""
    x = np.nan_to_num(np.asarray(x, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    axes = tuple(range(1, x.ndim))
    peak = np.abs(x).max(axis=axes)
    level = np.sqrt(np.square(x).mean(axis=axes))
    flagged = (peak > peak_limit) | (level > rms_limit)
    gain = np.where(flagged, 1.0 / np.maximum(peak, 1e-12), 1.0).astype(np.float32)
    return x * gain.reshape((-1,) + (1,) * (x.ndim - 1)), flagged


def to_mono(x: np.ndarray) -> np.ndarray:
    """(C, N) or (N,) -> (N,)."""
    return x if x.ndim == 1 else x.mean(axis=0)


def resample(x: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return x
    from math import gcd

    g = gcd(sr_from, sr_to)
    return signal.resample_poly(x, sr_to // g, sr_from // g, axis=-1).astype(np.float32)


def stft_mag(y: np.ndarray, n_fft: int, hop: int) -> np.ndarray:
    _, _, z = signal.stft(y, nperseg=n_fft, noverlap=n_fft - hop, nfft=n_fft, boundary="even", padded=True)
    return np.abs(z)


def band_energy_db(y: np.ndarray, sr: int) -> dict[str, float]:
    """Energy of each guitar band in dB relative to the 60-10000 Hz total (level independent)."""
    f, p = signal.welch(y, sr, nperseg=min(8192, len(y)))
    total = p[(f >= 60) & (f < 10000)].sum() + 1e-20
    return {
        n: float(10 * np.log10(p[(f >= lo) & (f < hi)].sum() / total + 1e-12))
        for n, (lo, hi) in zip(BAND_NAMES, BAND_EDGES_HZ)
    }


def mel_filterbank(sr: int, n_fft: int, n_mels: int = 40, fmin: float = 30.0, fmax: float | None = None) -> np.ndarray:
    fmax = fmax or sr / 2
    mel = lambda f: 2595 * np.log10(1 + f / 700)
    inv = lambda m: 700 * (10 ** (m / 2595) - 1)
    pts = inv(np.linspace(mel(fmin), mel(fmax), n_mels + 2))
    bins = np.fft.rfftfreq(n_fft, 1 / sr)
    fb = np.zeros((n_mels, len(bins)), dtype=np.float32)
    for i in range(n_mels):
        lo, c, hi = pts[i], pts[i + 1], pts[i + 2]
        up = (bins - lo) / max(c - lo, 1e-9)
        dn = (hi - bins) / max(hi - c, 1e-9)
        fb[i] = np.maximum(0, np.minimum(up, dn))
    return fb


def mfcc(y: np.ndarray, sr: int, n_mfcc: int = 13, n_fft: int = 2048, hop: int = 512, n_mels: int = 40) -> np.ndarray:
    """(n_mfcc, frames). Plain log-mel + DCT-II, no librosa."""
    mag = stft_mag(y, n_fft, hop) ** 2
    mel = np.log(mel_filterbank(sr, n_fft, n_mels) @ mag + 1e-10)
    return sfft.dct(mel, type=2, axis=0, norm="ortho")[:n_mfcc]


def multires_stft_distance(a: np.ndarray, b: np.ndarray, sizes: tuple[int, ...] = (512, 1024, 2048, 4096)) -> float:
    """Multi-resolution STFT loss (spectral convergence + log-magnitude L1), RMS-normalised so level does not count."""
    n = min(len(a), len(b))
    a, b = a[:n] / (rms(a[:n]) + 1e-9), b[:n] / (rms(b[:n]) + 1e-9)
    total = 0.0
    for s in sizes:
        A, B = stft_mag(a, s, s // 4), stft_mag(b, s, s // 4)
        sc = np.linalg.norm(A - B) / (np.linalg.norm(A) + 1e-9)
        lm = np.mean(np.abs(np.log(A + 1e-5) - np.log(B + 1e-5)))
        total += sc + lm
    return float(total / len(sizes))


def kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    """Plain k-means; returns the cluster id per row."""
    x = (x - x.mean(0)) / (x.std(0) + 1e-9)
    centers = x[rng.choice(len(x), size=min(k, len(x)), replace=False)]
    for _ in range(iters):
        ids = np.argmin(((x[:, None] - centers[None]) ** 2).sum(-1), axis=1)
        for j in range(len(centers)):
            if (ids == j).any():
                centers[j] = x[ids == j].mean(0)
    return ids
