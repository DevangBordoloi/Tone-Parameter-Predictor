"""GPU constant-Q transform (octave-by-octave, in the style of nnAudio's CQT2010) with no extra dependency.

Why not mel: on a CQT the distance between a note's fundamental and its k-th harmonic is Q*log2(k) bins whatever the
note, so a single convolution template can measure harmonic ratios (the signature of drive, bias and clipping hardness)
for every string and fret.

How it works: the top `bins_per_octave` bins are computed at the full rate with complex Hann-windowed kernels (as 1-D
convolutions); the signal is then low-passed and decimated by 2 and the next octave down is computed with the *same*
kernel shapes, so cost stays flat. Window lengths below `1/ (2^(1/b) - 1) * sr / f` samples are capped at
`max_window_sec` (bins under ~90 Hz therefore get a slightly broader filter than the ideal Q, which a 4 s clip cannot
resolve anyway). Octaves are computed at different frame rates and resampled in time to the top-octave frame grid.

The result matches nnAudio's CQT semantics (magnitude, centre-padded frames); `HarmonicCQT` slices one wide CQT into the
Bittner-style harmonic stack, so only one transform is computed for all six harmonic channels.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from scipy.signal import firwin
from torch import nn


class CQT(nn.Module):
    def __init__(
        self,
        sr: int,
        fmin: float,
        n_bins: int,
        bins_per_octave: int = 60,
        hop: int = 256,
        filter_scale: float = 1.0,
        max_window_sec: float = 1.0,
        min_stride: int = 4,
    ):
        super().__init__()
        self.sr, self.fmin, self.n_bins, self.bpo, self.hop = sr, fmin, n_bins, bins_per_octave, hop
        self.freqs = fmin * 2.0 ** (np.arange(n_bins) / bins_per_octave)
        fmax = float(self.freqs[-1])
        if fmax >= 0.45 * sr:
            raise ValueError(f"top bin {fmax:.0f} Hz is too close to Nyquist for sr={sr}")
        base = 1
        while fmax < 0.45 * sr / (base * 2):
            base *= 2
        self.base = base
        q = filter_scale / (2.0 ** (1.0 / bins_per_octave) - 1.0)

        self.groups: list[tuple[int, int]] = []
        top = n_bins
        while top > 0:
            lo = max(0, top - bins_per_octave)
            self.groups.append((lo, top))
            top = lo

        self.strides: list[int] = []
        self.spacing: list[int] = []  # frame spacing in input samples
        for j, (lo, hi) in enumerate(self.groups):
            rate = sr / (base * 2**j)
            f = self.freqs[lo:hi]
            n_k = np.minimum(q * rate / f, max_window_sec * rate)
            length = int(math.ceil(n_k.max()))
            length += 1 - length % 2
            n = np.arange(length) - (length - 1) / 2
            win = np.where(np.abs(n)[None] < n_k[:, None] / 2, 0.5 + 0.5 * np.cos(2 * np.pi * n[None] / n_k[:, None]), 0.0)
            win /= win.sum(axis=1, keepdims=True)
            ph = -2j * np.pi * f[:, None] * n[None] / rate
            ker = win * np.exp(ph)
            self.register_buffer(f"k_re_{j}", torch.tensor(ker.real, dtype=torch.float32)[:, None])
            self.register_buffer(f"k_im_{j}", torch.tensor(ker.imag, dtype=torch.float32)[:, None])
            s = max(min_stride, int(round(hop / (base * 2**j)))) if hop / (base * 2**j) < min_stride else int(round(hop / (base * 2**j)))
            self.strides.append(s)
            self.spacing.append(s * base * 2**j)
        self.register_buffer("lp", torch.tensor(firwin(61, 0.45), dtype=torch.float32)[None, None])

    def _halve(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        y = F.conv1d(x.reshape(b * c, 1, t), self.lp, stride=2, padding=self.lp.shape[-1] // 2)
        return y.reshape(b, c, -1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, C, S) -> magnitude (B, C, n_bins, T)."""
        b, c, _ = x.shape
        for _ in range(int(math.log2(self.base))):
            x = self._halve(x)
        outs: list[torch.Tensor] = []
        t_ref = None
        for j, (lo, hi) in enumerate(self.groups):
            kr, ki = getattr(self, f"k_re_{j}"), getattr(self, f"k_im_{j}")
            xin = x.reshape(b * c, 1, -1)
            re = F.conv1d(xin, kr, stride=self.strides[j], padding=kr.shape[-1] // 2)
            im = F.conv1d(xin, ki, stride=self.strides[j], padding=ki.shape[-1] // 2)
            mag = torch.sqrt(re * re + im * im + 1e-12)                       # (B*C, nb, T_j)
            if t_ref is None:
                t_ref = mag.shape[-1]
            elif mag.shape[-1] != t_ref:
                mag = F.interpolate(mag, size=t_ref, mode="linear", align_corners=True)
            outs.append(mag.reshape(b, c, hi - lo, t_ref))
            if j < len(self.groups) - 1:
                x = self._halve(x)
        return torch.cat(outs[::-1], dim=2)                                   # ascending frequency


def amplitude_to_db(mag: torch.Tensor, floor_db: float = -100.0) -> torch.Tensor:
    return torch.clamp(20.0 * torch.log10(mag + 1e-7), min=floor_db)


class HarmonicCQT(nn.Module):
    """Stack of CQTs whose channel h starts at h * fmin (Bittner et al., ISMIR 2017).

    One wide CQT starting at min(h) * fmin is computed once; channel h is the slice starting
    round(bpo * log2(h / min(h))) bins up, so the fundamental, sub-octave and four overtones of every time-frequency
    pixel line up across the channels.
    """

    def __init__(self, sr: int, fmin: float = 32.7, bins_per_octave: int = 60, n_bins: int = 360, hop: int = 256,
                 harmonics: tuple[float, ...] = (0.5, 1, 2, 3, 4, 5)):
        super().__init__()
        self.harmonics, self.n_bins, self.bpo = tuple(harmonics), n_bins, bins_per_octave
        h0 = min(harmonics)
        self.offsets = [int(round(bins_per_octave * math.log2(h / h0))) for h in harmonics]
        self.fundamental_offset = self.offsets[self.harmonics.index(1)] if 1 in self.harmonics else 0
        self.cqt = CQT(sr, fmin * h0, max(self.offsets) + n_bins, bins_per_octave, hop)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x (B, C, S) -> (db of the full wide CQT (B, C, F_wide, T)); use `stack`/`fundamental` to slice."""
        return amplitude_to_db(self.cqt(x))

    def stack(self, db: torch.Tensor, channel: int = 0) -> torch.Tensor:
        """(B, H, n_bins, T): harmonic stack of one input channel."""
        return torch.stack([db[:, channel, o : o + self.n_bins] for o in self.offsets], dim=1)

    def fundamental(self, db: torch.Tensor) -> torch.Tensor:
        """(B, C, n_bins, T): the h=1 slice of every input channel."""
        o = self.fundamental_offset
        return db[:, :, o : o + self.n_bins]
