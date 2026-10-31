"""Project-relative locations (override with TONE_DATA_DIR / TONE_PLUGIN_SCHEMA_DIR)."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PROJECT_ROOT / "config"
SCHEMA_DIR = Path(os.environ.get("TONE_PLUGIN_SCHEMA_DIR", CONFIG_DIR / "plugin_schemas"))
DATA_DIR = Path(os.environ.get("TONE_DATA_DIR", PROJECT_ROOT / "data"))
OUTPUT_DIR = PROJECT_ROOT / "output"

SAMPLE_RATE = 48_000
CLIP_SECONDS = 4.0
CLIP_SAMPLES = int(SAMPLE_RATE * CLIP_SECONDS)


def ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path
