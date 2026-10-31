"""Training objectives.

L = w_mse * masked MSE (numeric knobs)  +  w_ce * masked CE (switches / enums)  +  w_plugin * CE (which plugin)
    + w_contrast * InfoNCE (same params, different DI = positive; same DI, different params = hard negative)
    + w_smooth * trajectory smoothness (LFO prior)

The mask removes parameters that cannot be heard for the given switch positions (inactive amp channels, bypassed pedals),
so they never push gradients into the shared encoder.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from tone_model.modulation import smoothness_loss


@dataclass
class LossWeights:
    mse: float = 1.0
    ce: float = 0.5
    plugin: float = 0.2
    contrast: float = 0.3
    smooth: float = 0.05
    tau: float = 0.1
    hard_negative_boost: float = 2.0


def head_losses(h: dict, layout, params: torch.Tensor, masks: torch.Tensor) -> dict[str, torch.Tensor]:
    """One plugin's head output vs targets. Returns summed errors and counts so batches can be pooled exactly.

    `err_numeric` / `err_switch` are per-(row, parameter) errors in schema order slices, used for per-module reporting.
    """
    rows = h["rows"]
    tgt, msk = params[rows], masks[rows].float()
    out: dict[str, torch.Tensor] = {}
    if layout.n_numeric:
        idx = torch.as_tensor(layout.numeric_idx, device=tgt.device)
        m = msk[:, idx]
        se = (h["numeric"] - tgt[:, idx]) ** 2 * m
        out.update(mse_sum=se.sum(), mse_n=m.sum(), abs_sum=((h["numeric"] - tgt[:, idx]).abs() * m).sum(),
                   err_numeric=se.detach().sum(0), cnt_numeric=m.detach().sum(0))
    if layout.switch_dim:
        cls = layout.targets_to_classes(tgt)
        cidx = torch.as_tensor(layout.cat_idx, device=tgt.device)
        m = msk[:, cidx]
        ces, hits = [], []
        for j, (a, b) in enumerate(zip(layout.cat_offsets[:-1], layout.cat_offsets[1:])):
            lg = h["logits"][:, int(a):int(b)]
            ces.append(F.cross_entropy(lg, cls[:, j], reduction="none"))
            hits.append((lg.argmax(1) == cls[:, j]).float())
        ce = torch.stack(ces, 1) * m
        out.update(ce_sum=ce.sum(), ce_n=m.sum(), acc_sum=(torch.stack(hits, 1) * m).sum(),
                   err_switch=ce.detach().sum(0), cnt_switch=m.detach().sum(0))
    return out


def info_nce(proj: torch.Tensor, group: torch.Tensor, slot: torch.Tensor, tau: float = 0.1, hard_boost: float = 2.0) -> torch.Tensor:
    """Multi-positive InfoNCE over a batch of L2-normalised projections.

    positives  same group, same slot (identical parameters, a different DI clip)
    hard negs  same group, other slot (identical DI clip, other parameters); their logits are up-weighted by log(hard_boost)
    other negs everything else in the batch (other groups / plugins)
    """
    n = proj.shape[0]
    sim = proj @ proj.T / tau
    eye = torch.eye(n, dtype=torch.bool, device=proj.device)
    same_g = group[:, None] == group[None]
    pos = same_g & (slot[:, None] == slot[None]) & ~eye
    hard = same_g & (slot[:, None] != slot[None])
    sim = sim + hard.float() * math.log(max(hard_boost, 1.0))
    sim = sim.masked_fill(eye, float("-inf"))
    log_prob = sim - torch.logsumexp(sim, dim=1, keepdim=True)
    has_pos = pos.any(1)
    if not has_pos.any():
        return proj.new_zeros(())
    return -(log_prob.masked_fill(~pos, 0).sum(1)[has_pos] / pos.sum(1)[has_pos]).mean()


def total_loss(out: dict, model, params, masks, plugin_ids, group, slot, w: LossWeights) -> tuple[torch.Tensor, dict[str, float], dict]:
    mse_s = mse_n = ce_s = ce_n = acc_s = abs_s = torch.zeros((), device=params.device)
    per_plugin: dict[int, dict[str, torch.Tensor]] = {}
    for h in out["heads"]:
        r = head_losses(h, model.layouts[h["plugin"]], params, masks)
        per_plugin[h["plugin"]] = r
        mse_s, mse_n = mse_s + r.get("mse_sum", 0), mse_n + r.get("mse_n", 0)
        abs_s = abs_s + r.get("abs_sum", 0)
        ce_s, ce_n, acc_s = ce_s + r.get("ce_sum", 0), ce_n + r.get("ce_n", 0), acc_s + r.get("acc_sum", 0)
    l_mse = mse_s / mse_n.clamp(min=1)
    l_ce = ce_s / ce_n.clamp(min=1)
    l_plugin = F.cross_entropy(out["plugin_logits"], plugin_ids)
    l_con = info_nce(out["proj"], group, slot, w.tau, w.hard_negative_boost)
    l_smooth = smoothness_loss(out["traj"])
    loss = w.mse * l_mse + w.ce * l_ce + w.plugin * l_plugin + w.contrast * l_con + w.smooth * l_smooth
    logs = {
        "loss": loss.item(), "mse": l_mse.item(), "mae": (abs_s / mse_n.clamp(min=1)).item(), "ce": l_ce.item(),
        "switch_acc": (acc_s / ce_n.clamp(min=1)).item(), "plugin": l_plugin.item(), "contrast": l_con.item(),
        "plugin_acc": (out["plugin_logits"].argmax(1) == plugin_ids).float().mean().item(), "smooth": l_smooth.item(),
    }
    return loss, logs, per_plugin
