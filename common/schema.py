"""Plugin parameter schema (same JSON as neural-tone-mcp's vst_inspector) and the preset JSON it consumes."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, Field

from common.paths import SCHEMA_DIR

ParamKind = Literal["boolean", "enum", "numeric"]

# Enums with more options than this (delay sync notes, EQ filter sweeps, "transpose") behave like ordered knobs:
# they are regressed on the normalized 0-1 position instead of classified.
MAX_CATEGORICAL_OPTIONS = 16


class ParameterSpec(BaseModel):
    index: int
    name: str
    host_name: str | None = None
    kind: ParamKind
    default_normalized: float
    default_display: str = ""
    units: str | None = None
    min_value: float | None = None
    max_value: float | None = None
    step_size: float | None = None
    num_steps: int | None = None
    valid_values: list[bool | str] | None = None
    linear: bool | None = None
    curve: list[tuple[float, float]] | None = None
    options_verified: bool | None = None
    samples: list[dict] | None = None      # measured {"normalized", "display"} points of non-linear knobs

    def display(self, normalized: float) -> str:
        """Human-readable value of a normalized position: enum label, measured-curve interpolation, or min/max span."""
        x = min(1.0, max(0.0, float(normalized)))
        if self.valid_values:
            return str(self.label(x))
        unit = f" {self.units}" if self.units else ""
        pts = []
        for s in self.samples or []:
            try:
                pts.append((float(s["normalized"]), float(str(s["display"]).split()[0])))
            except (KeyError, ValueError, IndexError):
                pts = []
                break
        if len(pts) >= 2:
            pts.sort()
            return f"{float(np.interp(x, [p[0] for p in pts], [p[1] for p in pts])):.1f}{unit}"
        if self.min_value is not None and self.max_value is not None:
            return f"{self.min_value + x * (self.max_value - self.min_value):.1f}{unit}"
        return f"{x:.3f}"

    @property
    def n_options(self) -> int:
        if self.valid_values:
            return len(self.valid_values)
        return self.num_steps or 0

    @property
    def is_categorical(self) -> bool:
        """Boolean / small enum: predicted with a softmax head. Everything else is regressed."""
        return self.kind in ("boolean", "enum") and 2 <= self.n_options <= MAX_CATEGORICAL_OPTIONS

    def snap(self, normalized: float) -> float:
        x = min(1.0, max(0.0, float(normalized)))
        n = self.num_steps
        if n and n > 1:
            x = round(x * (n - 1)) / (n - 1)
        return x

    def class_to_normalized(self, index: int) -> float:
        n = self.n_options
        return index / (n - 1) if n > 1 else 0.0

    def normalized_to_class(self, x: float) -> int:
        n = self.n_options
        return int(round(min(1.0, max(0.0, x)) * (n - 1))) if n > 1 else 0

    def label(self, normalized: float) -> str | None:
        if self.valid_values:
            return str(self.valid_values[self.normalized_to_class(normalized)])
        return None


class PluginSchema(BaseModel):
    plugin_name: str
    vst3_path: str = ""
    is_instrument: bool = False
    parameter_count: int
    inspected_with: str = ""
    parameters: list[ParameterSpec]

    def by_name(self) -> dict[str, ParameterSpec]:
        return {p.name: p for p in self.parameters}

    def defaults(self) -> dict[str, float]:
        return {p.name: p.default_normalized for p in self.parameters}


class TrackConfig(BaseModel):
    track_name: str
    double_tracked: bool = False
    panning: list[float] = Field(default_factory=lambda: [0.0])
    input_gain_db: float = 0.0


class NeuralDSPPresetJSON(BaseModel):
    """Identical layout to neural-tone-mcp's preset JSON, so apply_tone_json_to_daw can load our output."""

    preset_name: str
    target_plugin: str
    track_config: TrackConfig
    signal_chain_summary: dict[str, str] = Field(default_factory=dict)
    vst_parameters: dict[str, float] = Field(default_factory=dict)

    def snapped(self, schema: PluginSchema) -> dict[str, float]:
        known = schema.by_name()
        return {k: known[k].snap(v) for k, v in self.vst_parameters.items() if k in known}


def norm_key(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower().removesuffix(".vst3"))


def list_schema_files(schema_dir: Path | None = None) -> list[Path]:
    return sorted((schema_dir or SCHEMA_DIR).glob("*.json"))


def load_schema(name_or_path: str | Path, schema_dir: Path | None = None) -> PluginSchema:
    """Load by file path or by (partial, case-insensitive) plugin name, e.g. "Nolly X"."""
    p = Path(name_or_path)
    if p.suffix == ".json" and p.exists():
        return PluginSchema.model_validate_json(p.read_text(encoding="utf-8"))
    want = norm_key(str(name_or_path))
    files = list_schema_files(schema_dir)
    exact = [f for f in files if norm_key(f.stem) == want]
    hits = exact or [f for f in files if want and want in norm_key(f.stem)]
    if len(hits) != 1:
        raise KeyError(f"{name_or_path!r} matches {[h.stem for h in hits]} in {schema_dir or SCHEMA_DIR}")
    return PluginSchema.model_validate_json(hits[0].read_text(encoding="utf-8"))


def guitar_plugin_names(include_vocal: bool = False, schema_dir: Path | None = None) -> list[str]:
    """Plugin names with a schema on disk. Mantra is a vocal processor, so it is opt-in."""
    names = [f.stem for f in list_schema_files(schema_dir)]
    return names if include_vocal else [n for n in names if "mantra" not in n.lower()]


def to_array(values: dict[str, float], schema: PluginSchema) -> np.ndarray:
    """Dense normalized vector in schema order; missing names take the plugin default."""
    return np.array([values.get(p.name, p.default_normalized) for p in schema.parameters], dtype=np.float32)


def from_array(vec: np.ndarray, schema: PluginSchema) -> dict[str, float]:
    return {p.name: p.snap(float(v)) for p, v in zip(schema.parameters, vec)}


def apply_params(plugin: Any, values: dict[str, float]) -> list[str]:
    """Write normalized values to a loaded pedalboard plugin; returns the names the plugin did not have."""
    missing = []
    for k, v in values.items():
        if k in plugin.parameters:
            plugin.parameters[k].raw_value = min(1.0, max(0.0, float(v)))
        else:
            missing.append(k)
    return missing
