"""End to end: song -> guitar stem -> plugin choice -> parameter estimation -> preset JSON (-> ITO, check, preview).

    python -m pipeline.run_pipeline "https://youtu.be/..." --checkpoint runs/exp1/best.pt --focus lead --artist "John Mayer"
    python -m pipeline.run_pipeline my_stem.wav --is-stem --plugin "Archetype Plini" --ito --check

The JSON has the same layout as neural-tone-mcp's presets, so its `apply_tone_json_to_daw` tool (REAPER / Ableton) can load it.
`extract_tone_model` mirrors the signature of that repo's `extract_tone` so it can be pasted into src/pipeline.py as the plan's
`extract_tone_model()` (see pipeline/mcp_server.py for the matching MCP tool).
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from common.paths import DATA_DIR, OUTPUT_DIR, ensure
from common.plugin_io import installed_plugins
from guitar_stem_isolator.isolate import isolate
from plugin_selector.select_plugin import Selection, load_centroids, select_plugin
from tone_parameter_estimation.infer import TonePredictor, stereo_width, to_preset, write_preset

log = logging.getLogger(__name__)

DEFAULT_CHECKPOINT = Path(__file__).resolve().parents[1] / "runs" / "finetune" / "best.pt"


def extract_tone_model(
    source: str,
    user_prompt: str = "",
    target_plugin: str | None = None,
    start_sec: float | None = 30,
    duration_sec: float | None = 25,
    focus: str | None = None,
    is_guitar_stem: bool = False,
    artist: str | None = None,
    song: str | None = None,
    checkpoint: str | Path = DEFAULT_CHECKPOINT,
    ito: bool = False,
    run_check: bool = False,
    preview: bool = False,
    preset_dir: str | Path | None = DATA_DIR / "presets_decoded",
    out_dir: str | Path | None = None,
    work_dir: str | Path | None = None,
    installed: list[str] | None = None,
    predictor: TonePredictor | None = None,
    renderer_factory: Any = None,
    apply_reaper: bool = False,
    group: str | None = None,
) -> dict[str, Any]:
    """Returns {json_path, preset, plugin, selection, prediction, stem, ...}. `focus` may also be inferred from the prompt
    ("lead" / "rhythm" in `user_prompt`); default rhythm."""
    out = ensure(Path(out_dir) if out_dir else OUTPUT_DIR)
    low = (user_prompt or "").lower()
    focus = focus or ("lead" if "lead" in low else "rhythm" if "rhythm" in low else "rhythm")

    stem = isolate(source, start_sec, duration_sec, focus, is_guitar_stem, work_dir=work_dir)
    predictor = predictor or TonePredictor(checkpoint)
    windows = predictor.windows(stem.stem_path, focus)

    probs, emb = predictor.classify(windows)
    centroids = load_centroids(Path(checkpoint).with_name("centroids.npz"))
    sel: Selection = select_plugin(target_plugin, artist, predictor.model.cfg.plugins, installed, probs, emb, centroids)
    log.info("plugin: %s (%s)", sel.plugin, sel.reason)

    pred = predictor.predict(sel.plugin, windows, focus)
    schema = predictor.schema_by_name[sel.plugin]
    name = f"{Path(source).stem if Path(source).exists() else (song or 'song')}_{focus}".replace(" ", "_")
    extra = {"Plugin choice": sel.reason + (f" (asked for {sel.requested})" if sel.substituted else "")}

    result: dict[str, Any] = {"stem": asdict(stem), "selection": asdict(sel), "plugin": sel.plugin, "prediction": pred}
    renderer = None
    if ito or run_check or preview:
        try:
            from pipeline.render_utils import PluginRenderer

            renderer = (renderer_factory or PluginRenderer)(schema, predictor.sr)
        except Exception as e:
            log.warning("plugin rendering unavailable (%s); skipping ITO / check / preview", e)

    if ito and renderer is not None:
        from tone_parameter_estimation.ito import ito_refine

        target_emb = pred.embedding
        mono = windows[int(np.argmax(pred.window_weights))].mean(0)
        free = pred.mask & ~np.array([p.is_categorical for p in schema.parameters])
        res = ito_refine(predictor.model, schema, renderer, target_emb, mono, pred.values, free, device=predictor.device)
        extra["ITO"] = f"score {res.initial_score:.3f} -> {res.score:.3f} in {len(res.history) - 1} generations"
        pred.values = res.values
        result["ito"] = {"initial": res.initial_score, "final": res.score, "history": res.history}

    preset = to_preset(schema, pred, name, focus, width=stereo_width(stem.stem_path), extra_summary=extra)
    path = write_preset(preset, out)
    from pipeline.report import write_report

    report_path = write_report(preset, schema, path.with_suffix(".md"), pred.spread)
    result.update(json_path=str(path), preset=preset, report_path=str(report_path))
    if apply_reaper:                                  # mode 2: push straight into the running REAPER
        from pipeline.reaper_driver import ReaperDriver

        result["reaper"] = ReaperDriver().apply_preset(preset, group)
    if renderer is not None and preview:
        result["preview"] = str(renderer.write_preview(pred.values, out / f"{path.stem}_preview.wav"))
    if renderer is not None and run_check and preset_dir:
        from evaluation.check import find_matching_presets, format_check, run_check as _run_check

        matches = find_matching_presets(preset_dir, sel.plugin, song, artist)
        target = windows[int(np.argmax(pred.window_weights))]
        rows = _run_check(predictor.model, schema, pred.values, target, renderer, matches, predictor.device)
        md = format_check(rows)
        (out / f"{path.stem}_check.md").write_text(md, encoding="utf-8")
        result["check"] = [asdict(r) for r in rows]
        result["check_markdown"] = md
    return result


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", help="URL or audio file")
    ap.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    ap.add_argument("--plugin", help="requested plugin (substituted if not installed)")
    ap.add_argument("--artist")
    ap.add_argument("--song")
    ap.add_argument("--prompt", default="")
    ap.add_argument("--start", type=float, default=None, help="seconds into a URL (default 30); a local file is used whole unless given")
    ap.add_argument("--duration", type=float, default=None, help="seconds to analyse (default 25 for URLs)")
    ap.add_argument("--focus", choices=["rhythm", "lead", "full"])
    ap.add_argument("--is-stem", action="store_true")
    ap.add_argument("--ito", action="store_true", help="refine with inference-time optimisation (renders through the plugin)")
    ap.add_argument("--check", action="store_true", help="compare with matching community presets")
    ap.add_argument("--preview", action="store_true")
    ap.add_argument("--apply-reaper", action="store_true", help="create the track(s) in the running REAPER (needs reaper_mcp_bridge.lua)")
    ap.add_argument("--group", help="REAPER folder to wrap the tracks in (e.g. the song name)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if Path(a.source).exists():                       # local audio / stem: analyse all of it unless a window is given
        start, dur = a.start, a.duration
    else:
        start, dur = (30 if a.start is None else a.start), (25 if a.duration is None else a.duration)
    res = extract_tone_model(a.source, a.prompt, a.plugin, start, dur, a.focus, a.is_stem, a.artist, a.song, a.checkpoint,
                             a.ito, a.check, a.preview, out_dir=a.out, apply_reaper=a.apply_reaper, group=a.group)
    print(json.dumps({k: v for k, v in res.items() if k in ("json_path", "report_path", "plugin", "preview", "ito", "reaper")}, indent=2, default=str))
    print(json.dumps(res["preset"].signal_chain_summary, indent=2))
    if "check_markdown" in res:
        print(res["check_markdown"])


if __name__ == "__main__":
    main()
