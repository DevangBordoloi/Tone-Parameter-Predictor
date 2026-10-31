"""Draw plugin parameter vectors: jittered community presets 50 %, structured random 50 %.

Structured random = draw the routing switches first (amp channel, which pedals / time effects are on), derive the
active-parameter mask from them, randomise only the active branch and leave every other parameter at its default.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from common.schema import PluginSchema, to_array
from dataset_builder.activity import CORE_RX, ActivityModel


@dataclass
class Sample:
    values: np.ndarray  # (P,) normalized, schema order
    mask: np.ndarray    # (P,) bool: parameter affects the sound and is learnable
    source: str         # "preset" | "random"
    info: dict | None = None   # presets: {"id", "preset_name", "meta"} (song / artist / author) for the shard manifest


class ParamSampler:
    def __init__(
        self,
        schema: PluginSchema,
        presets: list[dict[str, float]] | None = None,
        rng: np.random.Generator | None = None,
        preset_fraction: float = 0.5,
        fx_probs: dict[str, float] | None = None,
        default_switch_p: float = 0.5,
    ):
        self.schema = schema
        self.activity = ActivityModel(schema)
        self.specs = schema.parameters
        self.index = {p.name: i for i, p in enumerate(self.specs)}
        self.presets = presets or []
        self.rng = rng or np.random.default_rng()
        self.preset_fraction = preset_fraction
        self.fx_probs = fx_probs or {}
        self.default_switch_p = default_switch_p
        self.defaults = to_array(schema.defaults(), schema)
        for name, v in self.activity.pinned.items():
            self.defaults[self.index[name]] = v
        self.learnable = self.activity.learnable()
        self._switches = [n for n in self.activity.switches() if n not in self.activity.ignored]

    # -- helpers -------------------------------------------------------------------------------------------------
    def _p_on(self, switch: str) -> float:
        if switch.startswith("active_") or switch.endswith("_section_active") or switch in ("pedals_active", "time_active", "cab_active"):
            return 0.9
        return self.fx_probs.get(switch, self.default_switch_p)

    def _draw_switch(self, spec) -> float:
        n = spec.n_options
        if n == 2 and spec.name != "amp_type":
            return spec.class_to_normalized(int(self.rng.random() < self._p_on(spec.name)))
        return spec.class_to_normalized(int(self.rng.integers(n)))

    def _draw_numeric(self, spec, u: float | None = None) -> float:
        if CORE_RX.match(spec.name) and "gain" in spec.name:
            return spec.snap(float(np.clip(self.rng.normal(0.5, 0.08), 0.3, 0.7)))   # keep the level near unity
        if u is not None:                                                            # pre-drawn stratified / uniform value
            return spec.snap(float(np.clip(u, 0.0, 1.0)))
        if self.rng.random() < 0.65:
            x = self.rng.random()
        else:  # cluster near the factory value so "musical" mid-range settings are well covered
            x = self.rng.normal(spec.default_normalized, 0.18)
        return spec.snap(float(np.clip(x, 0.0, 1.0)))

    def _values(self, v: np.ndarray) -> dict[str, float]:
        return {s.name: float(v[i]) for i, s in enumerate(self.specs)}

    def _finish_mask(self, v: np.ndarray) -> np.ndarray:
        mask = self.activity.mask(self._values(v)) & self.learnable
        for name in self._switches:  # switches that gate something are always supervised (they define the routing)
            mask[self.index[name]] = True
        return mask

    # -- sampling ------------------------------------------------------------------------------------------------
    def random(self, strata: dict[str, float] | None = None) -> Sample:
        """Structured random sample. `strata` optionally maps a numeric parameter name to a pre-drawn value in [0, 1] (used by
        `synthetic_batch`); parameters it does not cover, and inactive ones, are drawn / left as usual."""
        v = self.defaults.copy()
        for name in self._switches:
            spec = self.specs[self.index[name]]
            if spec.is_categorical:
                v[self.index[name]] = self._draw_switch(spec)
        mask = self.activity.mask(self._values(v)) & self.learnable
        for i, spec in enumerate(self.specs):
            if not mask[i] or spec.name in self._switches:
                continue
            if spec.is_categorical:
                v[i] = spec.class_to_normalized(int(self.rng.integers(spec.n_options)))
            else:
                v[i] = self._draw_numeric(spec, (strata or {}).get(spec.name))
        return Sample(v, self._finish_mask(v), "random")

    def synthetic_batch(self, n: int, method: str = "lhs") -> list[Sample]:
        """`n` samples for synthetic pre-training: routing switches drawn at random per sample (so each sample's active
        blocks differ), numeric knobs spread over [0, 1] either by Latin Hypercube ("lhs": per knob, one value in each of the
        n equal strata, shuffled independently) or iid uniform ("uniform"). A knob only takes its value where it is active."""
        if method not in ("lhs", "uniform"):
            raise ValueError(f"method must be 'lhs' or 'uniform', got {method!r}")
        cols = {s.name: ((self.rng.permutation(n) + self.rng.random(n)) / n) if method == "lhs" else self.rng.random(n)
                for s in self.specs if not s.is_categorical and s.name not in self.activity.ignored}
        return [self.random({k: float(c[i]) for k, c in cols.items()}) for i in range(n)]

    def from_preset(self, preset: dict, jitter: float = 0.03, flip_p: float = 0.02) -> Sample:
        """Jitter a community preset. A preset may carry ``"__known__"``: the names its source actually determined (patch
        translations only know some knobs). Everything else is masked out of the loss instead of being taught as a default."""
        known = preset.get("__known__")
        info = preset.get("__info__")
        preset = {k: v for k, v in preset.items() if not k.startswith("__")}
        if known is not None:
            flip_p = 0.0                                   # a flipped switch would contradict the known labels
        v = self.defaults.copy()
        for name, x in preset.items():
            if name in self.index and name not in self.activity.ignored:
                v[self.index[name]] = self.specs[self.index[name]].snap(x)
        for name in self._switches:  # rarely flip a switch to widen the prior
            spec = self.specs[self.index[name]]
            if spec.is_categorical and self.rng.random() < flip_p:
                v[self.index[name]] = spec.class_to_normalized(int(self.rng.integers(spec.n_options)))
        mask = self.activity.mask(self._values(v)) & self.learnable
        if jitter > 0:
            for i, spec in enumerate(self.specs):
                if mask[i] and not spec.is_categorical and (known is None or spec.name in known):
                    v[i] = spec.snap(float(np.clip(v[i] + self.rng.normal(0, jitter), 0.0, 1.0)))
        mask = self._finish_mask(v)
        if known is not None:
            ks = set(known)
            mask &= np.array([s.name in ks for s in self.specs], dtype=bool)
        return Sample(v, mask, "preset", info)

    def sample(self) -> Sample:
        if self.presets and self.rng.random() < self.preset_fraction:
            return self.from_preset(self.presets[int(self.rng.integers(len(self.presets)))])
        return self.random()
