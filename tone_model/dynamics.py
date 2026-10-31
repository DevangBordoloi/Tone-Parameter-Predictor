"""Time-domain dynamics: compressor / gate / amp-sag cues that a 11-46 ms CQT frame smears out.

* `envelope_features`  RMS and peak envelopes in dB with first-order derivatives (Stein et al., DAFx 2010)
* `level_stats`        *unnormalised* level statistics (log-RMS, crest, dynamic range, noise floor, attack/sustain ratio,
                       onset rate, stereo width). They go straight into the parameter heads, because Neural DSP plugins
                       have explicit gain / gate / master knobs and RMS-normalising the input would erase those cues.
* `TFiLM`              temporal feature-wise linear modulation (Comunita et al., ICASSP 2023): a recurrent controller over
                       block-pooled activations produces time-varying scale and shift
* `DynamicsEncoder`    small 1-D residual cascade with TFiLM over the envelope features
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

STAT_NAMES = ("rms_db", "peak_db", "crest_db", "dyn_range_db", "noise_floor_db", "attack_sustain_db", "onsets_per_sec", "width_db")
N_STATS = len(STAT_NAMES)
_STAT_SHIFT = torch.tensor([-30.0, -15.0, 10.0, 20.0, -70.0, 0.0, 2.0, -20.0])
_STAT_SCALE = torch.tensor([15.0, 10.0, 5.0, 10.0, 20.0, 6.0, 2.0, 10.0])


def _db(x: torch.Tensor, floor: float = -120.0) -> torch.Tensor:
    return torch.clamp(20.0 * torch.log10(x + 1e-9), min=floor)


def frame_levels(x: torch.Tensor, frame: int, hop: int) -> tuple[torch.Tensor, torch.Tensor]:
    """x (B, S) -> (rms (B, T), peak (B, T)) with centre padding so T = S // hop + 1."""
    pad = frame // 2
    xp = F.pad(x, (pad, pad), mode="reflect" if x.shape[-1] > pad else "constant")
    rms = torch.sqrt(F.avg_pool1d((xp**2)[:, None], frame, hop).squeeze(1) + 1e-12)
    peak = F.max_pool1d(xp.abs()[:, None], frame, hop).squeeze(1)
    return rms, peak


def envelope_features(x: torch.Tensor, hop: int) -> torch.Tensor:
    """x (B, S) raw mid signal -> (B, 5, T): rms dB, peak dB, crest dB, d(rms), d(peak), roughly unit scale."""
    rms, peak = frame_levels(x, hop * 2, hop)
    r, p = _db(rms), _db(peak)
    dr = F.pad(r[:, 1:] - r[:, :-1], (1, 0))
    dp = F.pad(p[:, 1:] - p[:, :-1], (1, 0))
    return torch.stack([(r + 40) / 20, (p + 30) / 20, (p - r - 10) / 6, dr / 6, dp / 6], dim=1)


def level_stats(mid: torch.Tensor, side: torch.Tensor, hop: int, frame_rate: float) -> tuple[torch.Tensor, torch.Tensor]:
    """Returns (raw stats (B, N_STATS) in physical units, normalised stats for the network)."""
    rms, peak = frame_levels(mid, hop * 2, hop)
    r = _db(rms)
    pk = _db(peak.max(dim=1).values)
    glob_rms = _db(torch.sqrt((mid**2).mean(-1) + 1e-12))
    p95 = torch.quantile(r, 0.95, dim=1)
    p05 = torch.quantile(r, 0.05, dim=1)
    # Stein et al.: attack = from an onset to the following local maximum; approximated as +-4 frames around a >3 dB rise
    rise = (F.pad(r[:, 1:] - r[:, :-1], (1, 0)) > 3.0).float()
    attack = F.max_pool1d(rise[:, None], 9, 1, 4).squeeze(1)
    active = (r > (p95[:, None] - 40.0)).float()
    sustain = active * (1.0 - attack)
    power = 10 ** (r / 10)
    att_db = 10 * torch.log10((power * attack).sum(1) / attack.sum(1).clamp(min=1) + 1e-12)
    sus_db = 10 * torch.log10((power * sustain).sum(1) / sustain.sum(1).clamp(min=1) + 1e-12)
    onsets = (rise * (1.0 - F.pad(rise[:, :-1], (1, 0)))).sum(1) / (r.shape[1] / frame_rate)
    side_rms = torch.sqrt((side**2).mean(-1) + 1e-12)
    width = _db(side_rms) - glob_rms
    raw = torch.stack([glob_rms, pk, pk - glob_rms, p95 - p05, p05, att_db - sus_db, onsets, width], dim=1)
    norm = (raw - _STAT_SHIFT.to(raw)) / _STAT_SCALE.to(raw)
    return raw, torch.clamp(norm, -5, 5)


def sustain_weight(raw_stats: torch.Tensor) -> torch.Tensor:
    """Per-window reliability weight for averaging predictions: loud, sustained windows count, silence and pick scrape do not.
    Uses the sustain-to-attack energy ratio (higher = steadier) gated by absolute level."""
    ratio = torch.sigmoid(-raw_stats[:, 5] / 4.0 + 1.0)                  # attack_sustain_db low -> sustained
    loud = torch.sigmoid((raw_stats[:, 0] + 50.0) / 4.0)                 # rms above ~ -50 dBFS
    return ratio * loud


class TFiLM(nn.Module):
    """y = gamma(t) * x + beta(t), gamma/beta from a GRU run over block-max-pooled activations."""

    def __init__(self, channels: int, block: int = 16):
        super().__init__()
        self.block = block
        self.rnn = nn.GRU(channels, channels, batch_first=True)
        self.to_gb = nn.Linear(channels, 2 * channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, t = x.shape
        pad = (-t) % self.block
        xp = F.pad(x, (0, pad), mode="replicate") if pad else x
        blocks = F.max_pool1d(xp, self.block)                              # (B, C, T/b)
        h, _ = self.rnn(blocks.transpose(1, 2))
        gamma, beta = self.to_gb(h).chunk(2, dim=-1)
        gamma = 1.0 + gamma.transpose(1, 2).repeat_interleave(self.block, dim=-1)[..., :t]
        beta = beta.transpose(1, 2).repeat_interleave(self.block, dim=-1)[..., :t]
        return gamma * x + beta


class _ResBlock(nn.Module):
    def __init__(self, ch: int, dilation: int, block: int):
        super().__init__()
        self.conv1 = nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(ch)
        self.conv2 = nn.Conv1d(ch, ch, 5, padding=2 * dilation, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(ch)
        self.tfilm = TFiLM(ch, block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        return F.relu(x + self.tfilm(y))


class DynamicsEncoder(nn.Module):
    def __init__(self, in_ch: int = 5, ch: int = 64, out_dim: int = 128, dilations: tuple[int, ...] = (1, 2, 4, 8)):
        super().__init__()
        self.inp = nn.Sequential(nn.Conv1d(in_ch, ch, 5, padding=2), nn.BatchNorm1d(ch), nn.ReLU(inplace=True))
        self.blocks = nn.Sequential(*[_ResBlock(ch, d, 16) for d in dilations])
        self.out = nn.Sequential(nn.Linear(2 * ch, out_dim), nn.PReLU())
        self.out_dim = out_dim

    def forward(self, env: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.inp(env))
        return self.out(torch.cat([h.mean(-1), h.amax(-1)], dim=1))
