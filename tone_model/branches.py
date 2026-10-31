"""Convolutional branches over the CQT: timbre (tall vertical kernels) and time+frequency (wide horizontal kernels).

TimbreEncoder (TE, Pons et al. EUSIPCO 2017 / Peladeau & Peeters ICASSP 2024)
    The first layer runs parallel kernels of very different heights on the harmonic CQT stack:
    80x1, 80x3 (almost the whole spectrum: tone-stack tilt, cab/mic curve), 25x1, 25x3 (EQ band shape), 5x3 (resonant
    peaks). Kernels are (frequency x time). Deeper layers are conventional 3x3 blocks, then frequency is squeezed.

TemporalEncoder (TFE, Pons & Serra ICASSP 2019)
    Mid/Side CQT through wide horizontal kernels (3x15, 3x31: a few bins by hundreds of ms) to catch delay taps,
    slapback and reverb decay, with stereo width available because Side is a separate channel.

HDConvStem (HPPNet, Wei et al. ISMIR 2022)
    Alternative to the six-channel HCQT: one 48-bin/octave CQT, six parallel branches whose frequency offsets are the
    harmonic distances round(Q*log2 k) = 0, 48, 76, 96, 111, 124 bins, summed. Same job (aligning overtones with their
    fundamental), one sixth of the CQT bandwidth.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


def conv_bn(cin: int, cout: int, k: tuple[int, int], stride: tuple[int, int] = (1, 1), dilation: tuple[int, int] = (1, 1)) -> nn.Sequential:
    """Conv + BN + ReLU with 'same' output size for any kernel (even sizes such as 80x1 are padded asymmetrically)."""
    span = (dilation[0] * (k[0] - 1), dilation[1] * (k[1] - 1))
    pad = nn.ZeroPad2d((span[1] // 2, span[1] - span[1] // 2, span[0] // 2, span[0] - span[0] // 2))
    return nn.Sequential(pad, nn.Conv2d(cin, cout, k, stride, 0, dilation, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class HDConvStem(nn.Module):
    def __init__(self, in_ch: int = 1, out_ch: int = 6, bpo: int = 48, ks: tuple[int, ...] = (1, 2, 3, 4, 5, 6)):
        super().__init__()
        self.shifts = [int(round(bpo * math.log2(k))) for k in ks]        # 0, 48, 76, 96, 111, 124 for bpo=48
        self.branches = nn.ModuleList(nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False) for _ in ks)
        self.bn = nn.BatchNorm2d(out_ch)
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = x.shape[2]
        y = 0
        for s, conv in zip(self.shifts, self.branches):
            xs = F.pad(x[:, :, s:], (0, 0, 0, min(s, f)), value=-1.0) if s else x   # out[f] looks at bin f + shift
            y = y + conv(xs)
        return F.relu(self.bn(y))


class TimbreEncoder(nn.Module):
    def __init__(self, in_ch: int = 6, out_dim: int = 256, first: tuple[tuple[int, int, int], ...] = (
            (80, 1, 48), (80, 3, 32), (25, 1, 48), (25, 3, 32), (5, 3, 32))):
        super().__init__()
        self.multi = nn.ModuleList(conv_bn(in_ch, c, (m, n)) for m, n, c in first)
        c0 = sum(c for _, _, c in first)
        self.body = nn.Sequential(
            nn.MaxPool2d((4, 1)),
            conv_bn(c0, 160, (3, 3)), nn.MaxPool2d((2, 2)),
            conv_bn(160, 192, (3, 3)), nn.MaxPool2d((2, 2)),
            conv_bn(192, 256, (3, 3)), nn.MaxPool2d((2, 2)),
            conv_bn(256, 256, (3, 3)),
            nn.AdaptiveMaxPool2d((4, None)),                              # keep 4 coarse frequency regions
        )
        self.out = nn.Sequential(nn.Linear(256 * 4 * 2, out_dim), nn.PReLU())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.cat([m(x) for m in self.multi], dim=1)
        h = self.body(h)                                                   # (B, 256, 4, T')
        return self.out(torch.cat([h.mean(-1), h.amax(-1)], dim=1).flatten(1))


class TemporalEncoder(nn.Module):
    def __init__(self, in_ch: int = 2, out_dim: int = 192):
        super().__init__()
        self.net = nn.Sequential(
            conv_bn(in_ch, 32, (3, 15)), nn.MaxPool2d((4, 1)),
            conv_bn(32, 64, (3, 31)), nn.MaxPool2d((4, 2)),
            conv_bn(64, 96, (3, 15), dilation=(1, 2)), nn.MaxPool2d((2, 2)),
            conv_bn(96, 128, (3, 7), dilation=(1, 4)), nn.MaxPool2d((2, 1)),
            nn.AdaptiveMaxPool2d((3, None)),
        )
        self.out = nn.Sequential(nn.Linear(128 * 3 * 2, out_dim), nn.PReLU())
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.net(x)
        return self.out(torch.cat([h.mean(-1), h.amax(-1)], dim=1).flatten(1))
