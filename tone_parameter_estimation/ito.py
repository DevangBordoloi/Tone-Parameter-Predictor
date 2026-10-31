"""Optional inference-time optimisation (ST-ITO, Steinmetz et al., ISMIR 2024).

The network's single-pass prediction warm-starts a small gradient-free search (a (mu, lambda) evolution strategy) over the
*active numeric* parameters. Each candidate is rendered through the real plugin on a reference DI clip and scored by how
close its tone embedding is to the target stem's embedding (cosine in the learned, content-invariant space) plus a
multi-resolution STFT term against the stem. 10-20 generations are usually enough; each generation costs `population`
plugin renders.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch

from common.dsp import multires_stft_distance, resample, to_mono
from common.schema import PluginSchema
from tone_model.net import ToneNet

log = logging.getLogger(__name__)


@dataclass
class ItoResult:
    values: np.ndarray
    score: float
    initial_score: float
    history: list[float]


@torch.no_grad()
def embed_audio(model: ToneNet, wet: np.ndarray, device: str) -> np.ndarray:
    """wet (2, S) at the model rate -> L2-normalised projection (proj_dim,)."""
    return model.encode(torch.from_numpy(wet[None].astype(np.float32)).to(device))["proj"][0].cpu().numpy()


def ito_refine(
    model: ToneNet,
    schema: PluginSchema,
    render: Callable[[np.ndarray], np.ndarray],   # normalized parameter vector -> wet (2, S) at model.cfg.sr
    target_embedding: np.ndarray,
    target_audio: np.ndarray | None,              # mono excerpt at model.cfg.sr for the STFT term, or None
    init: np.ndarray,
    free: np.ndarray,                             # bool (P,): numeric active parameters the search may move
    generations: int = 12,
    population: int = 8,
    parents: int = 3,
    sigma: float = 0.10,
    stft_weight: float = 0.15,
    seed: int = 0,
    device: str = "cpu",
) -> ItoResult:
    rng = np.random.default_rng(seed)
    specs = schema.parameters
    idx = np.nonzero(free)[0]

    def score(vec: np.ndarray) -> float:
        wet = render(vec)
        s = float(np.dot(embed_audio(model, wet, device), target_embedding))           # cosine (both unit norm)
        loss = 1.0 - s
        if target_audio is not None and stft_weight:
            loss += stft_weight * multires_stft_distance(to_mono(wet), target_audio) / 4.0
        return -loss

    best_vec, best = init.copy(), score(init)
    initial = best
    mean = init[idx].copy()
    history = [best]
    for g in range(generations):
        sd = sigma * (0.85**g)
        cands = []
        for _ in range(population):
            v = init.copy() if g == 0 else best_vec.copy()
            v[idx] = np.clip(mean + rng.normal(0, sd, len(idx)), 0, 1)
            for i in idx:
                v[i] = specs[i].snap(v[i])
            cands.append((score(v), v))
        cands.sort(key=lambda t: -t[0])
        elite = cands[:parents]
        mean = np.mean([v[idx] for _, v in elite], axis=0)
        if elite[0][0] > best:
            best, best_vec = elite[0]
        history.append(best)
        log.info("ITO gen %d  best score %.4f", g, best)
    return ItoResult(best_vec, best, initial, history)
