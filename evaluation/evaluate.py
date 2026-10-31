"""Three test tiers -> CSV + markdown report.

  tier1  synthetic held-out   unseen DI recordings and unseen random parameters (the val/test shards).
                              Per-knob MAE, switch accuracy, plugin accuracy; with --render also the spectral distance of the
                              re-render (predicted vs true parameters through the real plugin).
  tier2  unseen community     presets from the author-level test split, rendered on test DI, parameters predicted back.
         presets              Same metrics, against two baselines: plugin defaults, and (optional) the heuristic
                              tone_mapper.build_preset from neural-tone-mcp when NEURAL_TONE_MCP points at that repo.
  tier3  real songs           Songs.txt + any extra stems: full pipeline, then the presetjunkie check (model preset vs matching
                              community presets, both scored against the stem), plus a manifest for a side-by-side REAPER session.

    python -m evaluation.evaluate tier1 --checkpoint runs/exp/best.pt --shards data/shards --split test
    python -m evaluation.evaluate tier2 --checkpoint runs/exp/best.pt --presets data/presets_decoded
    python -m evaluation.evaluate tier3 --checkpoint runs/exp/best.pt --songs "C:/.../Songs.txt"
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from common.paths import DATA_DIR, OUTPUT_DIR, ensure
from common.schema import PluginSchema, to_array
from evaluation.metrics import aggregate, param_metrics, tone_distance
from tone_model.checkpoint import load_checkpoint
from tone_parameter_estimation.infer import TonePredictor, load_stem_windows

log = logging.getLogger(__name__)


def write_csv(rows: list[dict], path: Path) -> None:
    ensure(path.parent)
    keys = sorted({k for r in rows for k, v in r.items() if not isinstance(v, (dict, list))})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def md_table(rows: list[dict], cols: list[str]) -> str:
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        out.append("| " + " | ".join(f"{r[c]:.3f}" if isinstance(r.get(c), float) else str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------------------------------------------------------
# tier 1
# ---------------------------------------------------------------------------------------------------------------
@torch.no_grad()
def tier1(checkpoint: str, shards: str, split: str = "test", max_groups: int | None = 400, device: str = "cpu",
          renderers: dict | None = None) -> list[dict]:
    from training.data import make_loader

    model, schemas, _ = load_checkpoint(checkpoint, device)
    loader = make_loader(shards, split, model.cfg.plugins, 4, False, 0, max_groups)
    by_plugin: dict[int, list[dict]] = defaultdict(list)
    plug_hits, plug_n = 0, 0
    for batch in loader:
        enc = model.encode(batch["audio"].to(device))
        heads = model.run_heads(enc, batch["plugin"].to(device))
        plug_hits += int((enc["plugin_logits"].argmax(1).cpu() == batch["plugin"]).sum())
        plug_n += len(batch["plugin"])
        for h in heads:
            layout, schema = model.layouts[h["plugin"]], schemas[h["plugin"]]
            dense = layout.assemble(h["numeric"], h["logits"], torch.from_numpy(to_array(schema.defaults(), schema)).to(device)).cpu().numpy()
            for k, row in enumerate(h["rows"].cpu().tolist()):
                truth = batch["params"][row, : len(schema.parameters)].numpy()
                m = param_metrics(dense[k], truth, schema, batch["mask"][row, : len(schema.parameters)].numpy())
                rec = {"tier": 1, "plugin": schema.plugin_name, **{k2: v for k2, v in m.items() if k2 != "module_mae"}}
                if renderers and schema.plugin_name in renderers:
                    r = renderers[schema.plugin_name]
                    rec.update({k2: v for k2, v in tone_distance(r(truth), r(dense[k]), model.cfg.sr).items() if not k2.startswith("band_") or k2 == "band_mae_db"})
                by_plugin[h["plugin"]].append(rec)
    rows = []
    for pid, recs in by_plugin.items():
        agg = aggregate(recs, ("knob_mae", "switch_acc", "band_mae_db", "mrstft"))
        rows.append({"tier": 1, "plugin": model.cfg.plugins[pid], "n": len(recs), **agg})
    rows.append({"tier": 1, "plugin": "(plugin id accuracy)", "n": plug_n, "switch_acc": plug_hits / max(plug_n, 1)})
    return rows


# ---------------------------------------------------------------------------------------------------------------
# tier 2
# ---------------------------------------------------------------------------------------------------------------
def _heuristic_baseline(schema: PluginSchema, wet: np.ndarray, sr: int, workdir: Path):
    """tone_mapper.build_preset from neural-tone-mcp, if available (NEURAL_TONE_MCP=<repo path>)."""
    repo = os.environ.get("NEURAL_TONE_MCP")
    if not repo or not Path(repo).exists():
        return None
    import importlib
    import soundfile as sf

    sys.path.insert(0, repo)
    try:
        extractor = importlib.import_module("src.dsp.extractor")
        mapper = importlib.import_module("src.llm.tone_mapper")
        ms = importlib.import_module("src.dsp.vst_inspector")
        wav = workdir / "baseline_stem.wav"
        sf.write(str(wav), wet.T, sr)
        feats = extractor.extract_features(wav, "full")
        preset = mapper.build_preset(feats, "", schema.plugin_name, schema=ms.load_schema(schema.plugin_name))
        return to_array(preset.vst_parameters, schema)
    except Exception as e:
        log.warning("heuristic baseline failed: %s", e)
        return None
    finally:
        sys.path.remove(repo)


def tier2(checkpoint: str, presets_dir: str, plugins: list[str] | None = None, n_di: int = 2, device: str = "cpu",
          workdir: Path | None = None) -> list[dict]:
    from pipeline.render_utils import PluginRenderer, held_out_di
    from preset_corpus.splits import load_split

    pred = TonePredictor(checkpoint, device)
    work = ensure(workdir or OUTPUT_DIR / "eval_tier2")
    rows = []
    for name in plugins or pred.model.cfg.plugins:
        schema = pred.schema_by_name[name]
        docs = load_split(presets_dir, name, "test")
        if not docs:
            log.warning("%s: no test-split presets in %s", name, presets_dir)
            continue
        rend = PluginRenderer(schema, pred.sr)
        defaults = to_array(schema.defaults(), schema)
        for doc in docs:
            truth = to_array(doc["vst_parameters"], schema)
            for k in range(n_di):
                wet = rend(truth, held_out_di(index=k + 1))
                windows = wet[None].astype(np.float32)
                p = pred.predict(name, windows)
                rec = {"tier": 2, "plugin": name, "preset": doc["_id"], "di": k}
                rec.update({f"model_{a}": b for a, b in param_metrics(p.values, truth, schema).items() if a != "module_mae"})
                rec.update({f"default_{a}": b for a, b in param_metrics(defaults, truth, schema).items() if a != "module_mae"})
                heur = _heuristic_baseline(schema, wet, pred.sr, work)
                if heur is not None:
                    rec.update({f"heuristic_{a}": b for a, b in param_metrics(heur, truth, schema).items() if a != "module_mae"})
                for label, vec in (("model", p.values), ("default", defaults)):
                    d = tone_distance(wet, rend(vec, held_out_di(index=k + 1)), pred.sr)
                    rec[f"{label}_band_mae_db"], rec[f"{label}_mrstft"] = d["band_mae_db"], d["mrstft"]
                rows.append(rec)
    return rows


# ---------------------------------------------------------------------------------------------------------------
# tier 3
# ---------------------------------------------------------------------------------------------------------------
def parse_songs(path: str | Path) -> list[dict]:
    """Songs.txt: `Name(2 guitar- 1 lead and 1 rhythm)` followed by a URL or `... path in system-"C:\\..."` line(s)."""
    songs: list[dict] = []
    cur = None
    for raw in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        m = re.match(r"^(.*?)\s*\((.*(?:guitar|rhythm|lead).*)\)\s*$", line, re.I)
        if m and not line.lower().startswith(("both", "for each")):
            cur = {"name": m.group(1).strip(), "desc": m.group(2), "sources": {}}
            songs.append(cur)
            continue
        if cur is None:
            continue
        if line.startswith("http"):
            cur["sources"]["mix"] = line.split("&list=")[0]
        else:
            p = re.search(r'(lead|rhythm)\s+part\s+path.*?"([^"]+)"', line, re.I)
            if p:
                cur["sources"][p.group(1).lower()] = p.group(2)
    for s in songs:
        d = s["desc"].lower()
        s["parts"] = (["lead", "rhythm"] if "lead" in d and "rhythm" in d and "+" not in d else
                      ["full"] if "+" in d else ["rhythm", "rhythm"] if "both rhythm" in d else ["lead"] if "lead" in d else ["rhythm"])
    return songs


def tier3(checkpoint: str, songs_file: str, presets_dir: str | None, out_dir: Path | None = None, ito: bool = False,
          device: str = "cpu") -> list[dict]:
    from pipeline.run_pipeline import extract_tone_model

    out = ensure(out_dir or OUTPUT_DIR / "eval_tier3")
    pred = TonePredictor(checkpoint, device)
    rows, manifest = [], []
    for song in parse_songs(songs_file):
        for part in dict.fromkeys(song["parts"]):
            source = song["sources"].get(part) or song["sources"].get("mix")
            if not source:
                log.warning("no source for %s / %s", song["name"], part)
                continue
            is_stem = part in song["sources"]
            try:
                res = extract_tone_model(source, "", None, 30, 25, "full" if part == "full" else part, is_stem, song=song["name"],
                                         checkpoint=checkpoint, ito=ito, run_check=bool(presets_dir), preview=False,
                                         preset_dir=presets_dir, out_dir=out, predictor=pred)
            except Exception as e:
                log.error("%s / %s failed: %s", song["name"], part, e)
                rows.append({"tier": 3, "song": song["name"], "part": part, "error": str(e)})
                continue
            model_row = next((c for c in res.get("check", []) if c["source"] == "model"), {})
            comm = [c for c in res.get("check", []) if c["source"] == "community"]
            rows.append({"tier": 3, "song": song["name"], "part": part, "plugin": res["plugin"], "json": res["json_path"],
                         "model_band_mae_db": model_row.get("band_mae_db"), "model_mrstft": model_row.get("mrstft"),
                         "model_embedding_cos": model_row.get("embedding_cos"), "n_community": len(comm),
                         "best_community_band_mae_db": min((c["band_mae_db"] for c in comm), default=None)})
            manifest.append({"track_name": f"{song['name']} {part}", "group": song["name"], "preset_json": res["json_path"]})
    (out / "reaper_session.json").write_text(json.dumps({"tracks": manifest}, indent=2), encoding="utf-8")
    return rows


# ---------------------------------------------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tier", choices=["tier1", "tier2", "tier3"])
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--shards", default=str(DATA_DIR / "shards"))
    ap.add_argument("--split", default="test")
    ap.add_argument("--presets", default=str(DATA_DIR / "presets_decoded"))
    ap.add_argument("--songs")
    ap.add_argument("--ito", action="store_true")
    ap.add_argument("--out", default=str(OUTPUT_DIR / "eval"))
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    out = ensure(Path(a.out))
    if a.tier == "tier1":
        rows = tier1(a.checkpoint, a.shards, a.split, device=a.device)
        cols = ["plugin", "n", "knob_mae", "switch_acc", "band_mae_db", "mrstft"]
    elif a.tier == "tier2":
        rows = tier2(a.checkpoint, a.presets, device=a.device)
        cols = ["plugin", "preset", "model_knob_mae", "default_knob_mae", "heuristic_knob_mae", "model_switch_acc", "default_switch_acc",
                "model_band_mae_db", "default_band_mae_db"]
    else:
        if not a.songs:
            ap.error("tier3 needs --songs")
        rows = tier3(a.checkpoint, a.songs, a.presets if Path(a.presets).exists() else None, device=a.device, ito=a.ito)
        cols = ["song", "part", "plugin", "model_band_mae_db", "model_mrstft", "model_embedding_cos", "n_community", "best_community_band_mae_db"]
    write_csv(rows, out / f"{a.tier}.csv")
    (out / f"{a.tier}.md").write_text(f"# {a.tier}\n\n" + md_table(rows, cols) + "\n", encoding="utf-8")
    print(md_table(rows, cols))
    print(f"\nwrote {out / (a.tier + '.csv')} and .md")


if __name__ == "__main__":
    main()
