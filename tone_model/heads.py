"""Per-plugin parameter heads: routing/switch classifier first, then FiLM-gated knob regression.

Neural DSP plugins are fixed chains (pedals -> amp channel -> EQ -> cab -> time effects). A flat head predicts every knob
independently of the amp channel / pedal switches, so it averages knob values across incompatible channels. Here:

  stage 1  2-layer MLP on [z, level stats] -> one softmax per boolean / small enum (amp_type, *_active, mic types, ...)
  stage 2  MLP (default 512 -> 256 -> C, BatchNorm1d + PReLU, sigmoid) over the numeric knobs; every hidden layer
           is modulated by FiLM (gamma, beta) computed from the stage-1 probabilities (the ground-truth one-hots with
           probability `tf_prob` during early epochs: teacher forcing). FiLM is initialised near identity with 10x smaller
           LeCun variance (DX7-GNN, DAFx 2026) so the conditioning is learned gradually. Dropout follows stage 1's hidden
           layer and every FiLM-modulated block: the heads are the over-parameterised part when the data is small.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from common.schema import PluginSchema


@dataclass
class PluginLayout:
    """Index bookkeeping between the dense schema vector and the head outputs."""

    name: str
    n_params: int
    numeric_idx: np.ndarray            # (C,) schema indices regressed with a sigmoid
    cat_idx: np.ndarray                # (S,) schema indices classified with a softmax
    cat_sizes: np.ndarray              # (S,) options per categorical parameter
    cat_offsets: np.ndarray            # (S+1,) offsets into the concatenated logits

    @property
    def n_numeric(self) -> int:
        return len(self.numeric_idx)

    @property
    def switch_dim(self) -> int:
        return int(self.cat_offsets[-1])

    @classmethod
    def from_schema(cls, schema: PluginSchema, ignored: set[str] | None = None) -> "PluginLayout":
        ignored = ignored or set()
        num, cat, sizes = [], [], []
        for i, p in enumerate(schema.parameters):
            if p.name in ignored:
                continue
            if p.is_categorical:
                cat.append(i)
                sizes.append(p.n_options)
            else:
                num.append(i)
        offs = np.concatenate([[0], np.cumsum(sizes)]).astype(np.int64)
        return cls(schema.plugin_name, len(schema.parameters), np.array(num, np.int64), np.array(cat, np.int64),
                   np.array(sizes, np.int64), offs)

    def targets_to_classes(self, params: torch.Tensor) -> torch.Tensor:
        """(B, P) normalized -> (B, S) integer class per categorical parameter."""
        x = params[:, torch.as_tensor(self.cat_idx, device=params.device)]
        n = torch.as_tensor(self.cat_sizes, device=params.device, dtype=x.dtype)
        return torch.round(x.clamp(0, 1) * (n - 1)).long()

    def one_hot(self, classes: torch.Tensor) -> torch.Tensor:
        """(B, S) classes -> (B, switch_dim) concatenated one-hots."""
        parts = [F.one_hot(classes[:, j], int(n)).float() for j, n in enumerate(self.cat_sizes)]
        return torch.cat(parts, dim=1) if parts else classes.new_zeros((classes.shape[0], 0)).float()

    def probs(self, logits: torch.Tensor) -> torch.Tensor:
        parts = [F.softmax(logits[:, int(a):int(b)], dim=1) for a, b in zip(self.cat_offsets[:-1], self.cat_offsets[1:])]
        return torch.cat(parts, dim=1) if parts else logits

    def assemble(self, numeric: torch.Tensor, logits: torch.Tensor, defaults: torch.Tensor) -> torch.Tensor:
        """Head outputs -> dense normalized vector (B, P); un-modelled parameters keep `defaults` (P,)."""
        out = defaults.to(numeric)[None].repeat(numeric.shape[0], 1)
        if self.n_numeric:
            out[:, torch.as_tensor(self.numeric_idx, device=out.device)] = numeric
        for j, (a, b, n) in enumerate(zip(self.cat_offsets[:-1], self.cat_offsets[1:], self.cat_sizes)):
            cls = logits[:, int(a):int(b)].argmax(1).float()
            out[:, int(self.cat_idx[j])] = cls / (int(n) - 1) if n > 1 else 0.0
        return out


class SafeBN(nn.BatchNorm1d):
    """BatchNorm1d that falls back to running statistics when a (per-plugin) sub-batch has a single sample."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and x.shape[0] < 2:
            return F.batch_norm(x, self.running_mean, self.running_var, self.weight, self.bias, False, 0.0, self.eps)
        return super().forward(x)


class FiLM(nn.Module):
    def __init__(self, cond_dim: int, features: int):
        super().__init__()
        self.lin = nn.Linear(cond_dim, 2 * features)
        nn.init.normal_(self.lin.weight, 0.0, math.sqrt(1.0 / (10.0 * max(cond_dim, 1))))   # 10x reduced LeCun variance
        nn.init.zeros_(self.lin.bias)

    def forward(self, h: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        gamma, beta = self.lin(cond).chunk(2, dim=-1)
        return (1.0 + gamma) * h + beta


class TwoStageHead(nn.Module):
    def __init__(self, layout: PluginLayout, z_dim: int = 512, stat_dim: int = 8, hidden: tuple[int, ...] = (512, 256),
                 switch_hidden: int = 256, dropout: float = 0.25):
        super().__init__()
        self.layout = layout
        d_in = z_dim + stat_dim
        sd = layout.switch_dim
        self.stage1 = nn.Sequential(nn.Linear(d_in, switch_hidden), nn.PReLU(), nn.Dropout(dropout),
                                    nn.Linear(switch_hidden, max(sd, 1)))
        self.drop = nn.Dropout(dropout)
        dims = [d_in, *hidden]
        self.fc = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.bn = nn.ModuleList(SafeBN(b) for b in hidden)
        self.act = nn.ModuleList(nn.PReLU() for _ in hidden)
        self.film = nn.ModuleList(FiLM(max(sd, 1), b) for b in hidden)
        self.out = nn.Linear(hidden[-1], max(layout.n_numeric, 1))

    def forward(self, z: torch.Tensor, stats: torch.Tensor, gt_classes: torch.Tensor | None = None, tf_prob: float = 0.0
                ) -> dict[str, torch.Tensor]:
        x = torch.cat([z, stats], dim=1)
        logits = self.stage1(x)[:, : self.layout.switch_dim] if self.layout.switch_dim else x.new_zeros((x.shape[0], 0))
        cond = self.layout.probs(logits).detach() if self.layout.switch_dim else x.new_zeros((x.shape[0], 1))
        if gt_classes is not None and tf_prob > 0 and self.layout.switch_dim:
            use_gt = (torch.rand(x.shape[0], 1, device=x.device) < tf_prob).float()
            cond = use_gt * self.layout.one_hot(gt_classes) + (1 - use_gt) * cond
        h = x
        for fc, bn, act, film in zip(self.fc, self.bn, self.act, self.film):
            h = self.drop(film(act(bn(fc(h))), cond))
        numeric = torch.sigmoid(self.out(h))[:, : self.layout.n_numeric]
        return {"numeric": numeric, "logits": logits}
