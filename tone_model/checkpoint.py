"""Save / load a trained ToneNet together with everything needed to rebuild it."""

from __future__ import annotations

from pathlib import Path

import torch

from common.schema import PluginSchema, load_schema
from dataset_builder.activity import ActivityModel
from tone_model.net import ToneNet, ToneNetConfig


def build_model(cfg: ToneNetConfig, plugin_names: list[str]) -> tuple[ToneNet, list[PluginSchema]]:
    schemas = [load_schema(n) for n in plugin_names]
    ignored = {s.plugin_name: ActivityModel(s).ignored for s in schemas}
    return ToneNet(cfg, schemas, ignored), schemas


def save_checkpoint(path: str | Path, model: ToneNet, extra: dict | None = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state": model.state_dict(), "cfg": model.config_dict(), "plugins": list(model.cfg.plugins), **(extra or {})}, path)


def load_checkpoint(path: str | Path, device: str = "cpu") -> tuple[ToneNet, list[PluginSchema], dict]:
    ck = torch.load(path, map_location=device, weights_only=False)
    cfg = ToneNetConfig(**_cfg_from_checkpoint(ck))
    model, schemas = build_model(cfg, ck["plugins"])
    model.load_state_dict(ck["state"])
    return model.to(device).eval(), schemas, ck


def _cfg_from_checkpoint(ck: dict) -> dict:
    cfg = {k: (tuple(v) if k == "head_hidden" else v) for k, v in ck["cfg"].items()}
    if "switch_hidden" not in cfg:                      # checkpoints from before the head shrink: 512 wide, no dropout
        cfg.update(switch_hidden=512, head_dropout=0.0)
    return cfg


def load_pretrained(model: ToneNet, path: str | Path, device: str = "cpu") -> dict[str, list[str]]:
    """Initialise `model` from a checkpoint, strict=False. The encoder is copied key-for-key. Per-plugin tensors (`heads.<i>.*`,
    `plugin_cls` rows) are matched by plugin *name*, so a different plugin subset or order never loads the wrong head. Tensors
    whose shape differs (e.g. a head trained at another width) are skipped. Returns {"loaded", "skipped", "missing"} key lists."""
    ck = torch.load(path, map_location=device, weights_only=False)
    src_plugins = list(ck["plugins"])
    own = model.state_dict()
    new: dict[str, torch.Tensor] = {}
    skipped: list[str] = []
    for k, v in ck["state"].items():
        if k.startswith("heads."):
            _, i, rest = k.split(".", 2)
            name = src_plugins[int(i)]
            if name not in model.cfg.plugins:
                continue
            k = f"heads.{model.cfg.plugins.index(name)}.{rest}"
        elif k.startswith("plugin_cls."):                # (n_plugins, z) / (n_plugins,): copy the rows of the shared plugins
            if k not in own:
                continue
            t = own[k].clone()
            for j, name in enumerate(model.cfg.plugins):
                if name in src_plugins and t.shape[1:] == v.shape[1:]:
                    t[j] = v[src_plugins.index(name)]
            new[k] = t
            continue
        if k in own and own[k].shape == v.shape:
            new[k] = v
        else:
            skipped.append(k)
    model.load_state_dict(new, strict=False)
    return {"loaded": sorted(new), "skipped": skipped, "missing": sorted(set(own) - set(new))}
