"""Optional MCP server (mode 2): the same local pipeline as `python -m pipeline.run_pipeline`, plus live tweaking in REAPER.

No LLM runs in here: the host (Antigravity / Claude / ...) only decides which tool to call. For the plain no-agent workflow use
`python -m pipeline.run_pipeline` and `python -m pipeline.tune` instead.

Register (example, see mcp_config.example.json):
    command  <repo>\\.venv\\Scripts\\python.exe     args  -m pipeline.mcp_server     cwd  <repo>
    env      TONE_CHECKPOINT=<repo>\\runs\\finetune\\best.pt
"""

from __future__ import annotations

import os
from typing import Any

try:
    from mcp.server import MCPServer as FastMCP
except ImportError:
    from mcp.server.fastmcp import FastMCP

from pipeline.run_pipeline import DEFAULT_CHECKPOINT, extract_tone_model

mcp = FastMCP("tone-model")
_PREDICTOR = None


def _ckpt() -> str:
    return os.environ.get("TONE_CHECKPOINT", str(DEFAULT_CHECKPOINT))


def _predictor():
    global _PREDICTOR
    if _PREDICTOR is None:
        from tone_parameter_estimation.infer import TonePredictor

        _PREDICTOR = TonePredictor(_ckpt())
    return _PREDICTOR


def _brief(res: dict[str, Any]) -> dict[str, Any]:
    out = {"json_path": res["json_path"], "report_path": res.get("report_path"), "plugin": res["plugin"],
           "plugin_choice": res["selection"], "signal_chain_summary": res["preset"].signal_chain_summary,
           "parameters": res["preset"].vst_parameters, "guitar_stem": res["stem"]["stem_path"]}
    if "reaper" in res:
        out["reaper"] = res["reaper"]
    return out


@mcp.tool()
def match_url_tone_model(url: str, target_plugin: str | None = None, artist: str | None = None, start_sec: int = 30,
                         duration_sec: int = 25, focus: str | None = None, is_guitar_stem: bool = False,
                         refine_with_ito: bool = False) -> dict[str, Any]:
    """URL or audio file -> guitar stem -> trained model -> preset JSON + readable report. Does NOT touch REAPER.

    focus: "rhythm" (Side channel of double-tracked parts) or "lead" (Mid channel). refine_with_ito renders candidates through the
    real plugin for a few generations (slower, closer match). The result is a model estimate: `signal_chain_summary` lists
    the least certain knobs.
    """
    return _brief(extract_tone_model(url, "", target_plugin, start_sec, duration_sec, focus, is_guitar_stem, artist,
                                     checkpoint=_ckpt(), ito=refine_with_ito, predictor=_predictor()))


@mcp.tool()
def match_url_to_reaper(url: str, target_plugin: str | None = None, artist: str | None = None, start_sec: int = 30,
                        duration_sec: int = 25, focus: str | None = None, is_guitar_stem: bool = False,
                        refine_with_ito: bool = False, group: str | None = None) -> dict[str, Any]:
    """Same as match_url_tone_model, then creates the track(s) in the running REAPER with every parameter set.
    `group` wraps them in a folder (e.g. the song name). Returns the track GUIDs for follow-up tweaks."""
    return _brief(extract_tone_model(url, "", target_plugin, start_sec, duration_sec, focus, is_guitar_stem, artist,
                                     checkpoint=_ckpt(), ito=refine_with_ito, predictor=_predictor(), apply_reaper=True, group=group))


@mcp.tool()
def apply_preset_to_reaper(json_path: str, group: str | None = None) -> dict[str, Any]:
    """Load a saved preset JSON (possibly hand-edited) into REAPER as new track(s)."""
    from pipeline.reaper_driver import ReaperDriver

    return ReaperDriver().apply_preset_json(json_path, group)


@mcp.tool()
def adjust_reaper_params(track: str, updates: dict[str, float | str], plugin: str | None = None) -> dict[str, Any]:
    """Change parameters of the Neural DSP plugin on a REAPER track (name, 1-based number or GUID), live.
    updates: {schema parameter name: normalized 0-1, or an option label for enums/switches e.g. {"amp_type": "Lead"}}.
    Read the names and current values first with get_reaper_track_state."""
    from pipeline.reaper_driver import ReaperDriver

    return ReaperDriver().tweak(track, updates, plugin)


@mcp.tool()
def get_reaper_track_state(track: str, name_filter: str | None = None, plugin: str | None = None) -> dict[str, Any]:
    """Current parameter values (normalized + display text) of the plugin on a REAPER track; optional name filter."""
    from pipeline.reaper_driver import ReaperDriver

    return ReaperDriver().read_state(track, plugin, name_filter)


@mcp.tool()
def edit_preset_file(json_path: str, updates: dict[str, float | str]) -> dict[str, Any]:
    """Edit values in a saved preset JSON without opening REAPER; returns the readable report."""
    from common.schema import load_schema
    from pipeline.report import format_report
    from pipeline.tune import edit_preset

    preset = edit_preset(json_path, updates)
    return {"json_path": json_path, "report": format_report(preset, load_schema(preset.target_plugin))}


def main() -> None:
    mcp.run()


if __name__ == "__main__":
    main()
