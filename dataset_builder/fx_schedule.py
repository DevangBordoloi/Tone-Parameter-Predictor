"""Dynamic Fx probability scheduling (Fx-Encoder++ / RelFx).

After each epoch the trainer reports the mean parameter loss per module (a module is the switch that owns a group of
knobs, e.g. `delay_active`). Modules with a higher residual error are switched on more often in the next rendering round,
so the dataset spends its budget where the model is weakest.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


class FxScheduler:
    def __init__(self, modules: list[str], base_p: float = 0.5, p_min: float = 0.15, p_max: float = 0.9, smoothing: float = 0.5):
        self.modules = list(modules)
        self.base_p, self.p_min, self.p_max, self.smoothing = base_p, p_min, p_max, smoothing
        self.p = {m: base_p for m in self.modules}

    def update(self, module_loss: dict[str, float]) -> dict[str, float]:
        """Renormalise: p_m = base_p * loss_m / mean(loss), clipped, then blended with the previous value."""
        known = {m: float(module_loss[m]) for m in self.modules if m in module_loss and np.isfinite(module_loss[m])}
        if not known:
            return self.p
        mean = float(np.mean(list(known.values()))) or 1.0
        for m, loss in known.items():
            target = float(np.clip(self.base_p * loss / mean, self.p_min, self.p_max))
            self.p[m] = self.smoothing * self.p[m] + (1 - self.smoothing) * target
        return self.p

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.p, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path, modules: list[str]) -> "FxScheduler":
        s = cls(modules)
        p = Path(path)
        if p.exists():
            saved = json.loads(p.read_text(encoding="utf-8"))
            s.p.update({m: float(v) for m, v in saved.items() if m in s.p})
        return s
