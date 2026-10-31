"""Modulation (chorus / phaser / flanger / tremolo) feature extractors.

* GammatoneModulationBank - fixed DSP (Martinez Ramirez & Reiss, DAFx 2019): 12 log-spaced gammatone bands
  (26-6950 Hz) -> Hilbert envelope -> 400 Hz -> 12-filter modulation bank (0.5-100 Hz); the energy in every
  (audio band x modulation band) cell, normalised by the envelope's DC energy, is a shift-invariant vector that tracks LFO
  speed and depth. Runs on the GPU in the frequency domain, no parameters.
* LFONet - learned (Mitcheltree et al., DAFx 2023): a stack of conv blocks that max-pool over frequency while the temporal
  dilation doubles, ending in a time-distributed linear layer that emits `n_traj` latent modulation trajectories s(t).
  The trajectories' FFT magnitudes (rate) and spread (depth) are returned as features. There is no ground-truth LFO in a
  plugin render, so the trajectories are trained end-to-end through the parameter loss, plus a multi-derivative
  smoothness prior (Mitcheltree's L_S without the L1 term to a reference).
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class GammatoneModulationBank(nn.Module):
    def __init__(self, sr: int, n_audio_bands: int = 12, n_mod_bands: int = 12, fmin: float = 26.0, fmax: float = 6950.0,
                 env_rate: int = 400, mod_fmin: float = 0.5, mod_fmax: float = 100.0, clip_sec: float = 4.0):
        super().__init__()
        self.sr, self.env_rate = sr, env_rate
        self.pool = max(1, sr // env_rate)
        self.n_audio, self.n_mod = n_audio_bands, n_mod_bands
        fmax = min(fmax, 0.45 * sr)
        centres = np.geomspace(fmin, fmax, n_audio_bands)
        self.centres = centres
        self.out_dim = n_audio_bands * n_mod_bands + 4 * n_audio_bands
        self._cache: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
        self.register_buffer("_c", torch.tensor(centres, dtype=torch.float32), persistent=False)
        mod_centres = np.geomspace(mod_fmin, mod_fmax, n_mod_bands)
        self.mod_centres = mod_centres
        self.mod_edges = np.geomspace(mod_fmin, mod_fmax * 1.25, n_mod_bands + 2)

    def _gamma_filters(self, n: int, device) -> torch.Tensor:
        """Magnitude response of 4th-order gammatones on the rfft grid, doubled so that ifft gives the analytic signal."""
        f = torch.fft.rfftfreq(n, 1.0 / self.sr).to(device)
        fc = self._c.to(device)[:, None]
        erb = 24.7 * (4.37 * fc / 1000.0 + 1.0)
        b = 1.019 * erb
        h = (1.0 + ((f[None] - fc) / b) ** 2) ** (-2.0)
        h = h / h.max(dim=1, keepdim=True).values
        return 2.0 * h                                                         # one-sided doubling -> analytic signal

    def _mod_filters(self, n: int, device) -> torch.Tensor:
        f = torch.fft.rfftfreq(n, 1.0 / self.env_rate).to(device)
        e = torch.tensor(self.mod_edges, dtype=torch.float32, device=device)
        lo, c, hi = e[:-2, None], e[1:-1, None], e[2:, None]                     # triangular filters, log-spaced
        up = (f[None] - lo) / (c - lo)
        dn = (hi - f[None]) / (hi - c)
        return torch.clamp(torch.minimum(up, dn), min=0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, S) mono, RMS-normalised -> (B, out_dim)."""
        b, s = x.shape
        spec = torch.fft.fft(x, dim=-1)
        g = self._gamma_filters(s, x.device)                                    # (A, s//2+1), already doubled
        g = torch.cat([g, torch.zeros(g.shape[0], s - g.shape[1], device=x.device)], dim=1)
        bands = []
        for a in range(g.shape[0]):                                             # one band at a time keeps memory at B x S
            env = torch.fft.ifft(spec * g[a], dim=-1).abs()                      # Hilbert envelope of the band
            bands.append(F.avg_pool1d(env[:, None], self.pool, self.pool))      # -> ~400 Hz
        env = torch.cat(bands, dim=1)                                           # (B, A, N)
        n = env.shape[-1]
        dc = env.mean(dim=-1, keepdim=True)
        spec_e = torch.fft.rfft(env - dc, dim=-1).abs() ** 2                    # (B, A, F_m)
        mf = self._mod_filters(n, x.device)                                      # (M, F_m)
        cell = torch.einsum("baf,mf->bam", spec_e, mf) / (n * dc.squeeze(-1)[..., None] ** 2 + 1e-8)
        cell = torch.log1p(cell * 1e3)                                          # (B, A, M)
        centres = torch.tensor(self.mod_centres, device=x.device)
        groups = [(0.5, 4.0), (4.5, 10.0), (10.5, 20.0), (20.5, 100.0)]
        canon = torch.stack([cell[..., (centres >= lo) & (centres <= hi)].mean(-1) if ((centres >= lo) & (centres <= hi)).any()
                             else torch.zeros_like(cell[..., 0]) for lo, hi in groups], dim=-1)   # (B, A, 4)
        return torch.cat([cell.flatten(1), canon.flatten(1)], dim=1)


class LFONet(nn.Module):
    def __init__(self, in_ch: int = 1, widths: tuple[int, ...] = (24, 32, 48, 48, 64), n_traj: int = 4, frame_rate: float = 93.75,
                 max_mod_hz: float = 20.0, out_dim: int = 128, n_frames: int = 376):
        super().__init__()
        blocks, c = [], in_ch
        for i, w in enumerate(widths):
            blocks += [
                nn.Conv2d(c, w, 3, padding=(1, 2**i), dilation=(1, 2**i)),
                nn.BatchNorm2d(w), nn.ReLU(inplace=True), nn.MaxPool2d((3, 1), ceil_mode=True),
            ]
            c = w
        self.blocks = nn.Sequential(*blocks)
        self.time_linear = nn.Conv1d(c, n_traj, 1)                              # time-distributed linear -> s(t)
        self.n_traj, self.frame_rate, self.max_mod_hz = n_traj, frame_rate, max_mod_hz
        self.freq_pool = nn.AdaptiveMaxPool2d((1, None))
        self.out_dim = out_dim
        freqs = np.fft.rfftfreq(n_frames, 1.0 / frame_rate)
        n_keep = int(((freqs >= 0.25) & (freqs <= max_mod_hz)).sum())
        self.proj = nn.Linear(n_traj * n_keep + 2 * n_traj, out_dim)
        self.act = nn.PReLU()

    def trajectories(self, cqt_db: torch.Tensor) -> torch.Tensor:
        h = self.freq_pool(self.blocks(cqt_db)).squeeze(2)                        # (B, C, T)
        return torch.sigmoid(self.time_linear(h))                                 # (B, n_traj, T) in [0, 1]

    def forward(self, cqt_db: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """cqt_db (B, 1, F, T) -> (features (B, out_dim), trajectories (B, n_traj, T))."""
        s = self.trajectories(cqt_db)
        t = s.shape[-1]
        s = s.float()                                                             # rfft has no bfloat16 kernel (--amp)
        spec = torch.fft.rfft(s - s.mean(-1, keepdim=True), dim=-1).abs() / t
        freqs = torch.fft.rfftfreq(t, 1.0 / self.frame_rate).to(s.device)
        keep = (freqs >= 0.25) & (freqs <= self.max_mod_hz)
        mag = spec[..., keep]                                                     # (B, n_traj, K)
        w = F.softmax(mag * 50.0, dim=-1)
        peak_hz = (w * freqs[keep]).sum(-1)                                       # soft-argmax LFO rate
        depth = s.std(-1)                                                         # amplitude spread = depth
        feats = torch.cat([torch.log1p(mag * 100).flatten(1), peak_hz / self.max_mod_hz, depth * 4], dim=1)
        return self.act(self.proj(feats)), s


def smoothness_loss(traj: torch.Tensor, weights: tuple[float, float, float] = (0.0, 1.0, 2.0)) -> torch.Tensor:
    """Multi-derivative prior on the latent trajectories: penalise 1st and 2nd differences (alpha=0: no reference L1)."""
    d1 = traj[..., 1:] - traj[..., :-1]
    d2 = d1[..., 1:] - d1[..., :-1]
    return weights[1] * d1.abs().mean() + weights[2] * d2.abs().mean()
