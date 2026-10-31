"""REAPER control through the file-based Lua bridge (`reaper_mcp_bridge.lua`, polling ~/.reaper-mcp). No LLM, no network.

Setup once: in REAPER, Actions > Show action list > ReaScript: Load... > %APPDATA%\\REAPER\\Scripts\\reaper_mcp_bridge.lua > Run.
The bridge answers one request at a time, so do not run two clients (or another REAPER MCP server) simultaneously.

    apply_preset_json(path)                      create the track(s), insert the plugin, write every parameter
    read_state(track)                            current knob values + display strings of a plugin on a track
    tweak(track, {"delay_mix": 0.3, ...})        change individual parameters live (schema names, normalized 0-1)
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from pathlib import Path
from typing import Any

from common.schema import NeuralDSPPresetJSON, PluginSchema, load_schema


class DawError(RuntimeError):
    pass


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


def match_params(schema: PluginSchema, reaper_names: list[str], wanted: list[str]) -> tuple[dict[str, int], list[str]]:
    """schema parameter name -> REAPER parameter index by host display name. Returns (found, missing)."""
    by_norm: dict[str, list[int]] = {}
    for i, n in enumerate(reaper_names):
        by_norm.setdefault(_norm(n), []).append(i)
    spec = schema.by_name()
    found: dict[str, int] = {}
    missing: list[str] = []
    for name in wanted:
        p = spec.get(name)
        cands = by_norm.get(_norm(p.host_name or name), []) if p else []
        cands = cands or by_norm.get(_norm(name), [])
        if not cands:
            missing.append(name)
        else:                                        # duplicate labels: the one nearest the plugin's own order
            found[name] = min(cands, key=lambda i: abs(i - (p.index if p else i)))
    return found, missing


class Bridge:
    def __init__(self, bridge_dir: str | Path | None = None, timeout: float = 30.0):
        self.dir = Path(bridge_dir or os.environ.get("REAPER_MCP_BRIDGE_DIR") or Path.home() / ".reaper-mcp").expanduser()
        self.dir.mkdir(parents=True, exist_ok=True)
        self.timeout = float(os.environ.get("REAPER_MCP_TIMEOUT", timeout))

    def call(self, cmd: str, timeout: float | None = None, **args: Any) -> Any:
        args = {k: v for k, v in args.items() if v is not None}
        req, resp = self.dir / "request.json", self.dir / "response.json"
        rid = uuid.uuid4().hex
        resp.unlink(missing_ok=True)
        req.unlink(missing_ok=True)
        tmp = self.dir / f"request.{rid}.tmp"
        tmp.write_text(json.dumps({"id": rid, "cmd": cmd, "args": args}), encoding="utf-8")
        os.replace(tmp, req)
        wait = timeout or self.timeout
        deadline = time.monotonic() + wait
        while time.monotonic() < deadline:
            try:
                data = json.loads(resp.read_text(encoding="utf-8"))
            except (FileNotFoundError, PermissionError, json.JSONDecodeError):
                data = None
            if data is not None:
                resp.unlink(missing_ok=True)
                if data.get("id") == rid:
                    if data.get("ok"):
                        return data.get("result")
                    raise DawError(str(data.get("error") or "unknown REAPER error"))
            time.sleep(0.02)
        req.unlink(missing_ok=True)
        raise DawError(f"REAPER did not answer within {wait:.0f}s. Open REAPER and run reaper_mcp_bridge.lua "
                       f"(Actions > Show action list > Load ReaScript); it watches {self.dir}.")


class ReaperDriver:
    def __init__(self, bridge_dir: str | Path | None = None):
        self.bridge = Bridge(bridge_dir)

    def ping(self) -> dict:
        return self.bridge.call("ping", timeout=5)

    def _track(self, name: str, pan: float = 0.0, folder_start: bool = False) -> str:
        r = self.bridge.call("create_track", name=name, pan=pan or None, folder_depth=1 if folder_start else None)
        return r["guid"]

    def _add_plugin(self, track: str, plugin: str) -> Any:
        return self.bridge.call("add_fx", track=track, plugin=plugin, timeout=120)["slot"]

    def _set(self, track: str, slot: Any, values: dict[str, float], schema: PluginSchema) -> dict[str, Any]:
        info = self.bridge.call("get_params", track=track, fx=slot, names_only=True, limit=5000)
        idx, missing = match_params(schema, info["params"], list(values))
        items = [{"index": i, "value": float(values[n]), "mode": "normalized"} for n, i in idx.items()]
        failed = [{"param": n, "error": "no matching parameter in REAPER"} for n in missing]
        applied = 0
        if items:
            res = self.bridge.call("set_params", track=track, fx=slot, params=items, timeout=120)
            for n, r in zip(idx, res["results"]):
                if r.get("ok"):
                    applied += 1
                else:
                    failed.append({"param": n, "error": r.get("error")})
        return {"plugin": info["fx"], "applied": applied, "failed": failed}

    # ------------------------------------------------------------------------------------------------------
    def apply_preset(self, preset: NeuralDSPPresetJSON, group: str | None = None) -> dict[str, Any]:
        """Create the track(s) for a preset. `group` wraps them in a parent folder (e.g. the song name)."""
        schema = load_schema(preset.target_plugin)
        cfg = preset.track_config
        values = preset.snapped(schema)
        report: dict[str, Any] = {"preset": preset.preset_name, "plugin": schema.plugin_name, "tracks": []}
        if group:
            self._track(group, folder_start=True)
        if cfg.double_tracked:
            pans = cfg.panning[:2] if len(cfg.panning) >= 2 else [-100.0, 100.0]
            self._track(cfg.track_name, folder_start=True)
            left = self._track(f"{cfg.track_name} L", pan=pans[0])
            slot = self._add_plugin(left, schema.plugin_name)
            res = self._set(left, slot, values, schema)
            right = self._track(f"{cfg.track_name} R", pan=pans[1])
            chain = self.bridge.call("get_chain", track=left, snapshot=False)
            self.bridge.call("set_chain", track=right, chain=chain["chain"], mode="replace", timeout=120)
            self.bridge.call("adjust_folder_depth", track=right, delta=-1)
            report["tracks"] = [{"track": f"{cfg.track_name} L", "guid": left, **res},
                                {"track": f"{cfg.track_name} R", "guid": right, "copied_from": "L"}]
        else:
            t = self._track(cfg.track_name, pan=cfg.panning[0] if cfg.panning else 0.0)
            slot = self._add_plugin(t, schema.plugin_name)
            report["tracks"] = [{"track": cfg.track_name, "guid": t, **self._set(t, slot, values, schema)}]
        if group:                                    # close the group folder on the last created track
            self.bridge.call("adjust_folder_depth", track=report["tracks"][-1]["guid"], delta=-1)
        return report

    def apply_preset_json(self, path: str | Path, group: str | None = None) -> dict[str, Any]:
        return self.apply_preset(NeuralDSPPresetJSON.model_validate_json(Path(path).read_text(encoding="utf-8")), group)

    def _find_slot(self, track: str, plugin: str | None) -> tuple[Any, PluginSchema]:
        from common.schema import guitar_plugin_names

        names = guitar_plugin_names(True)
        for f in self.bridge.call("list_fx", track=track)["fx"]:
            key = _norm(str(f.get("name", "")))
            for n in names:
                if _norm(n) in key and (plugin is None or _norm(plugin) in _norm(n)):
                    return f["slot"], load_schema(n)
        raise DawError(f"no Neural DSP plugin{f' matching {plugin!r}' if plugin else ''} found on track {track!r}")

    def read_state(self, track: str, plugin: str | None = None, name_filter: str | None = None) -> dict[str, Any]:
        slot, schema = self._find_slot(track, plugin)
        res = self.bridge.call("get_params", track=track, fx=slot, filter=name_filter, limit=5000)
        return {"plugin": schema.plugin_name, "slot": slot, "params": res["params"]}

    def tweak(self, track: str, updates: dict[str, float], plugin: str | None = None) -> dict[str, Any]:
        """updates: {schema name: normalized 0-1}. Enum / switch values may be given as the option label (\"Lead\")."""
        slot, schema = self._find_slot(track, plugin)
        spec = schema.by_name()
        values: dict[str, float] = {}
        unknown = []
        for k, v in updates.items():
            p = spec.get(k)
            if p is None:
                unknown.append(k)
            elif isinstance(v, str) and p.valid_values:
                labels = [str(x).lower() for x in p.valid_values]
                if v.lower() not in labels:
                    raise DawError(f"{k}: {v!r} is not one of {p.valid_values}")
                values[k] = p.class_to_normalized(labels.index(v.lower()))
            else:
                values[k] = p.snap(float(v))
        res = self._set(track, slot, values, schema)
        res["unknown_parameters"] = unknown
        return res
