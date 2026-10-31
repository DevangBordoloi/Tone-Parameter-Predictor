"""ToneNet: the 3-stream hybrid tone encoder with per-plugin two-stage heads.

audio (B, 2, S) wet stereo  ->  Mid/Side  ->  [front end]
    stream 1  Timbre branch        HCQT of Mid (6 harmonic channels, 60 bins/oct)  or  CQT + HD-Conv stem
    stream 2  Temporal branch      Mid/Side CQT, wide horizontal kernels
    stream 3  Modulation+Dynamics  LFONet + gammatone modulation filterbank + envelope ResNet with TFiLM
  -> z (512) -> contrastive projection (128, L2-normalised), plugin classifier, and per-plugin heads fed with
     [z, unnormalised level stats].
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import torch
import torch.nn.functional as F
from torch import nn

from common.schema import PluginSchema
from tone_model.branches import HDConvStem, TemporalEncoder, TimbreEncoder
from tone_model.cqt import CQT, HarmonicCQT, amplitude_to_db
from tone_model.dynamics import N_STATS, DynamicsEncoder, envelope_features, level_stats
from tone_model.heads import PluginLayout, TwoStageHead
from tone_model.modulation import GammatoneModulationBank, LFONet


@dataclass
class ToneNetConfig:
    sr: int = 24_000
    clip_sec: float = 4.0
    frontend: str = "hcqt"            # "hcqt" (6-channel harmonic stack) | "hdconv" (single 48 bpo CQT + HD-Conv stem)
    fmin: float = 32.7
    bins_per_octave: int = 60
    n_bins: int = 360
    z_dim: int = 512
    proj_dim: int = 128
    timbre_dim: int = 256
    temporal_dim: int = 192
    moddyn_dim: int = 192
    head_hidden: tuple[int, ...] = (512, 256)
    switch_hidden: int = 256
    head_dropout: float = 0.25
    plugins: list[str] = field(default_factory=list)

    @property
    def hop(self) -> int:
        return int(round(self.sr * 0.01067))            # ~10.7 ms

    @property
    def frame_rate(self) -> float:
        return self.sr / self.hop

    @property
    def n_frames(self) -> int:
        return (int(self.clip_sec * self.sr) - 1) // self.hop + 1        # frames the CQT emits


class ToneNet(nn.Module):
    def __init__(self, cfg: ToneNetConfig, schemas: list[PluginSchema], ignored: dict[str, set[str]] | None = None):
        super().__init__()
        self.cfg = cfg
        cfg.plugins = [s.plugin_name for s in schemas]
        self.layouts = [PluginLayout.from_schema(s, (ignored or {}).get(s.plugin_name)) for s in schemas]

        if cfg.frontend == "hcqt":
            self.hcqt = HarmonicCQT(cfg.sr, cfg.fmin, cfg.bins_per_octave, cfg.n_bins, cfg.hop)
            self.stem = None
            timbre_in, self.n_freq = 6, cfg.n_bins
        elif cfg.frontend == "hdconv":
            n_bins = 352
            self.cqt48 = CQT(cfg.sr, cfg.fmin, n_bins, 48, cfg.hop)
            self.stem = HDConvStem(1, 6, 48)
            timbre_in, self.n_freq = 6, n_bins
        else:
            raise ValueError(cfg.frontend)

        self.timbre = TimbreEncoder(timbre_in, cfg.timbre_dim)
        self.temporal = TemporalEncoder(2, cfg.temporal_dim)
        self.lfo = LFONet(1, frame_rate=cfg.frame_rate, n_frames=cfg.n_frames)
        self.modfb = GammatoneModulationBank(cfg.sr)
        self.modfb_mlp = nn.Sequential(nn.Linear(self.modfb.out_dim, 128), nn.PReLU())
        self.dyn = DynamicsEncoder(5, 64, 128)
        self.moddyn = nn.Sequential(nn.Linear(self.lfo.out_dim + 128 + self.dyn.out_dim, cfg.moddyn_dim), nn.PReLU())

        self.fuse = nn.Sequential(
            nn.Linear(cfg.timbre_dim + cfg.temporal_dim + cfg.moddyn_dim, cfg.z_dim), nn.BatchNorm1d(cfg.z_dim), nn.PReLU(),
            nn.Dropout(0.1),
        )
        self.proj = nn.Sequential(nn.Linear(cfg.z_dim, 256), nn.PReLU(), nn.Linear(256, cfg.proj_dim))
        self.plugin_cls = nn.Linear(cfg.z_dim, len(schemas))
        self.heads = nn.ModuleList(
            TwoStageHead(l, cfg.z_dim, N_STATS, tuple(cfg.head_hidden), cfg.switch_hidden, cfg.head_dropout) for l in self.layouts)
        self.register_buffer("_defaults", torch.zeros(1), persistent=False)

    # ---- feature path ----------------------------------------------------------------------------------------
    def _dsp(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """Everything that must stay in float32 even under autocast: level statistics, envelopes, CQT, modulation bank."""
        mid, side = (audio[:, 0] + audio[:, 1]) / 2, (audio[:, 0] - audio[:, 1]) / 2
        raw_stats, stats = level_stats(mid, side, self.cfg.hop, self.cfg.frame_rate)
        env = envelope_features(mid, self.cfg.hop)

        scale = 1.0 / (mid.pow(2).mean(-1, keepdim=True).sqrt() + 1e-6)         # RMS-normalise the CQT path only
        ms = torch.stack([mid * scale, side * scale], dim=1)
        if self.cfg.frontend == "hcqt":
            db = self.hcqt(ms)
            timbre_in = (self.hcqt.stack(db, 0) + 60.0) / 30.0
            ms_in = (self.hcqt.fundamental(db) + 60.0) / 30.0
        else:
            db = amplitude_to_db(self.cqt48(ms))
            timbre_in = (db[:, :1] + 60.0) / 30.0                              # the HD-Conv stem runs in the network part
            ms_in = (db + 60.0) / 30.0
        return {"timbre_in": timbre_in, "ms_in": ms_in, "env": env, "modfb": self.modfb(mid * scale),
                "stats": stats, "raw_stats": raw_stats}

    def encode(self, audio: torch.Tensor) -> dict[str, torch.Tensor]:
        """audio (B, 2, S) float at cfg.sr -> embeddings."""
        with torch.autocast(audio.device.type, enabled=False):
            d = self._dsp(audio.float())
        timbre_in = self.stem(d["timbre_in"]) if self.stem is not None else d["timbre_in"]
        t_emb = self.timbre(timbre_in)
        f_emb = self.temporal(d["ms_in"])
        lfo_emb, traj = self.lfo(d["ms_in"][:, :1])
        m_emb = self.modfb_mlp(d["modfb"])
        d_emb = self.dyn(d["env"])
        md = self.moddyn(torch.cat([lfo_emb, m_emb, d_emb], dim=1))
        z = self.fuse(torch.cat([t_emb, f_emb, md], dim=1))
        return {"z": z, "proj": F.normalize(self.proj(z).float(), dim=-1), "stats": d["stats"], "raw_stats": d["raw_stats"],
                "plugin_logits": self.plugin_cls(z), "traj": traj}

    # ---- heads -----------------------------------------------------------------------------------------------
    def run_heads(self, enc: dict[str, torch.Tensor], plugin_ids: torch.Tensor, params: torch.Tensor | None = None,
                  tf_prob: float = 0.0) -> list[dict]:
        """Run each plugin's head on the rows that belong to it. Returns [{plugin, rows, numeric, logits}]."""
        out = []
        for pid in torch.unique(plugin_ids).tolist():
            rows = (plugin_ids == pid).nonzero(as_tuple=True)[0]
            layout = self.layouts[pid]
            gt = layout.targets_to_classes(params[rows]) if params is not None and layout.switch_dim else None
            res = self.heads[pid](enc["z"][rows], enc["stats"][rows], gt, tf_prob)
            out.append({"plugin": pid, "rows": rows, **res})
        return out

    def forward(self, audio: torch.Tensor, plugin_ids: torch.Tensor, params: torch.Tensor | None = None, tf_prob: float = 0.0) -> dict:
        enc = self.encode(audio)
        enc["heads"] = self.run_heads(enc, plugin_ids, params, tf_prob)
        return enc

    @torch.no_grad()
    def predict(self, audio: torch.Tensor, plugin_id: int, defaults: torch.Tensor) -> dict[str, torch.Tensor]:
        """Single-plugin inference: audio (B, 2, S) -> dense normalized parameters (B, P) + switch probabilities."""
        self.eval()
        enc = self.encode(audio)
        ids = torch.full((audio.shape[0],), plugin_id, dtype=torch.long, device=audio.device)
        h = self.run_heads(enc, ids)[0]
        layout = self.layouts[plugin_id]
        return {"params": layout.assemble(h["numeric"], h["logits"], defaults), "switch_probs": layout.probs(h["logits"]),
                "z": enc["z"], "proj": enc["proj"], "raw_stats": enc["raw_stats"],
                "plugin_probs": F.softmax(enc["plugin_logits"], dim=-1)}

    def n_params(self) -> dict[str, int]:
        count = lambda m: sum(p.numel() for p in m.parameters())
        enc = [self.timbre, self.temporal, self.lfo, self.modfb_mlp, self.dyn, self.moddyn, self.fuse, self.proj]
        if self.stem is not None:
            enc.append(self.stem)
        return {"encoder": sum(count(m) for m in enc), "timbre": count(self.timbre), "temporal": count(self.temporal),
                "modulation_dynamics": count(self.lfo) + count(self.modfb_mlp) + count(self.dyn) + count(self.moddyn),
                "heads": count(self.heads), "total": count(self)}

    def config_dict(self) -> dict:
        return asdict(self.cfg)
