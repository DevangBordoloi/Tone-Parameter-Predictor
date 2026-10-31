"""Plain PyTorch training loop: AdamW, warm-up + cosine schedule, teacher-forced routing, early stopping on validation.

    python -m training.train --shards data/shards --out runs/exp1 --epochs 40

Two stages (synthetic pre-train, then fine-tune on community presets; see dataset_builder.render_dataset --mode):

    python -m training.train --shards data/shards_pretrain --val-shards data/shards --out runs/pretrain --epochs 40
    python -m training.train --shards data/shards_finetune --val-shards data/shards --out runs/finetune \
        --pretrained runs/pretrain/best.pt --finetune --epochs 30

After every epoch the per-module parameter error is turned into new switch-on probabilities (Dynamic Fx Scheduling) and
written to `<out>/fx_probs.json`; pass that file to `dataset_builder.render_dataset --fx-probs-path` for the next
rendering round.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch

from common.paths import DATA_DIR
from dataset_builder.activity import ActivityModel
from dataset_builder.fx_schedule import FxScheduler
from tone_model.checkpoint import build_model, load_pretrained, save_checkpoint
from tone_model.losses import LossWeights, total_loss
from tone_model.net import ToneNetConfig
from training.data import make_loader

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    shards: str = str(DATA_DIR / "shards")
    out: str = "runs/exp"
    plugins: list[str] = field(default_factory=list)       # default: every plugin present in the shards
    frontend: str = "hcqt"
    epochs: int = 40
    groups_per_batch: int = 8                              # x4 renders = 32 clips per step
    lr: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 300
    grad_clip: float = 1.0
    patience: int = 6
    tf_epochs: int = 8                                     # teacher forcing probability decays 1 -> 0 over these epochs
    amp: bool = False
    workers: int = 0
    max_groups: int | None = None
    seed: int = 0
    head_hidden: list[int] = field(default_factory=lambda: [512, 256])
    switch_hidden: int = 256
    head_dropout: float = 0.25
    val_shards: str = ""                                   # directory holding the val split (default: `shards`)
    source_filter: str = "all"                             # train groups to use: all | 0 (random renders) | 1 (presets)
    pretrained: str = ""                                   # checkpoint to initialise from (strict=False, matched by plugin name)
    finetune: bool = False                                 # differential LRs + reduced teacher forcing (use with --pretrained)
    encoder_lr: float = 1e-5                               # finetune: every non-head parameter
    head_lr: float = 1e-4                                  # finetune: the per-plugin heads
    finetune_tf_start: float = 0.0                         # finetune: initial teacher-forcing probability (decays over tf_epochs)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    weights: LossWeights = field(default_factory=LossWeights)


def shard_plugins(shard_dir: str) -> list[str]:
    from dataset_builder.shards import Shard, list_shards

    names = set()
    for p in list_shards(shard_dir, "train"):
        s = Shard(p)
        names.add(str(s.attrs["plugin"]))
        s.close()
    return sorted(names)


def module_book(model) -> dict[int, tuple[list[str], list[str]]]:
    """plugin id -> (module of each numeric head output, module of each switch)."""
    book = {}
    for pid, layout in enumerate(model.layouts):
        act = ActivityModel(_schema_of(model, pid))
        names = [p.name for p in _schema_of(model, pid).parameters]
        book[pid] = ([act.module_of(names[i]) for i in layout.numeric_idx], [act.module_of(names[i]) for i in layout.cat_idx])
    return book


_SCHEMAS: dict[int, object] = {}


def _schema_of(model, pid):
    if pid not in _SCHEMAS:
        from common.schema import load_schema

        _SCHEMAS[pid] = load_schema(model.cfg.plugins[pid])
    return _SCHEMAS[pid]


def run_epoch(model, loader, opt, sched, scaler, cfg: TrainConfig, epoch: int, train: bool, book) -> tuple[dict, dict]:
    model.train(train)
    dev = cfg.device
    tf_start = cfg.finetune_tf_start if cfg.finetune else 1.0
    tf = tf_start * max(0.0, 1.0 - epoch / max(cfg.tf_epochs, 1)) if train else 0.0
    sums: dict[str, float] = defaultdict(float)
    n_batches = 0
    mod_err: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch in loader:
            audio = batch["audio"].to(dev)
            params, masks, pid = batch["params"].to(dev), batch["mask"].to(dev), batch["plugin"].to(dev)
            gid, slot = batch["gid"].to(dev), batch["slot"].to(dev)
            with torch.autocast(dev if dev != "cpu" else "cpu", dtype=torch.bfloat16, enabled=cfg.amp):
                out = model(audio, pid, params if train else None, tf)
            loss, logs, per_plugin = total_loss(out, model, params, masks, pid, gid, slot, cfg.weights)
            if train:
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
                opt.step()
                sched.step()
            for k, v in logs.items():
                sums[k] += v
            n_batches += 1
            for p, r in per_plugin.items():
                if "err_numeric" in r:
                    for m, e, c in zip(book[p][0], r["err_numeric"].tolist(), r["cnt_numeric"].tolist()):
                        mod_err[m][0] += e
                        mod_err[m][1] += c
    means = {k: v / max(n_batches, 1) for k, v in sums.items()}
    module_loss = {m: e / c for m, (e, c) in mod_err.items() if c > 0}
    return means, module_loss


def train(cfg: TrainConfig) -> Path:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    plugins = cfg.plugins or shard_plugins(cfg.shards)
    model, schemas = build_model(ToneNetConfig(frontend=cfg.frontend, head_hidden=tuple(cfg.head_hidden),
                                               switch_hidden=cfg.switch_hidden, head_dropout=cfg.head_dropout), plugins)
    if cfg.pretrained:
        rep = load_pretrained(model, cfg.pretrained)
        log.info("pretrained %s: %d tensors loaded, %d skipped (shape), %d left at init", cfg.pretrained, len(rep["loaded"]),
                 len(rep["skipped"]), len(rep["missing"]))
        if rep["skipped"]:
            log.warning("shape-mismatched tensors not loaded: %s", rep["skipped"][:6])
    elif cfg.finetune:
        log.warning("--finetune without --pretrained: differential learning rates on a randomly initialised model")
    model.to(cfg.device)
    log.info("plugins=%s params=%s", plugins, model.n_params())
    train_dl = make_loader(cfg.shards, "train", plugins, cfg.groups_per_batch, True, cfg.workers, cfg.max_groups, cfg.source_filter)
    val_dir = cfg.val_shards or cfg.shards
    try:
        val_dl = make_loader(val_dir, "val", plugins, cfg.groups_per_batch, False, cfg.workers, cfg.max_groups, cfg.source_filter)
    except FileNotFoundError:
        try:                                               # the val shards may hold no groups of this source: validate on all
            val_dl = make_loader(val_dir, "val", plugins, cfg.groups_per_batch, False, cfg.workers, cfg.max_groups)
            log.warning("no val groups with source=%s: validating on all val groups", cfg.source_filter)
        except FileNotFoundError:
            log.warning("no validation shards: validating on a slice of train (early stopping will be optimistic)")
            val_dl = make_loader(cfg.shards, "train", plugins, cfg.groups_per_batch, False, 0, min(64, len(train_dl.dataset)),
                                 cfg.source_filter)

    if cfg.finetune:
        heads = [p for n, p in model.named_parameters() if n.startswith("heads.")]
        shared = [p for n, p in model.named_parameters() if not n.startswith("heads.")]
        opt = torch.optim.AdamW([{"params": shared, "lr": cfg.encoder_lr}, {"params": heads, "lr": cfg.head_lr}],
                                weight_decay=cfg.weight_decay)
        log.info("finetune: encoder lr %g (%d tensors), heads lr %g (%d tensors), tf start %g", cfg.encoder_lr, len(shared),
                 cfg.head_lr, len(heads), cfg.finetune_tf_start)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    total_steps = cfg.epochs * len(train_dl)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / cfg.warmup_steps) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / max(total_steps, 1)))))
    book = module_book(model)
    schedulers = {schemas[i].plugin_name: FxScheduler(sorted({m for m in book[i][0]})) for i in range(len(schemas))}

    best, bad, history = float("inf"), 0, []
    for epoch in range(cfg.epochs):
        t0 = time.time()
        tr, mod_loss = run_epoch(model, train_dl, opt, sched, None, cfg, epoch, True, book)
        va, _ = run_epoch(model, val_dl, opt, sched, None, cfg, epoch, False, book)
        # dynamic Fx scheduling: weak modules get switched on more often in the next rendering round
        per_plugin_loss = _split_by_plugin(mod_loss, book, schemas)
        for name, sch in schedulers.items():
            sch.update(per_plugin_loss.get(name, {}))
        (out / "fx_probs.json").write_text(json.dumps({n: s.p for n, s in schedulers.items()}, indent=1), encoding="utf-8")
        row = {"epoch": epoch, "sec": round(time.time() - t0, 1), "train": tr, "val": va, "lr": sched.get_last_lr()[0]}
        history.append(row)
        log.info("epoch %d  train loss %.4f mae %.4f | val loss %.4f mae %.4f sw-acc %.3f plug-acc %.3f", epoch, tr["loss"], tr["mae"],
                 va["loss"], va["mae"], va["switch_acc"], va["plugin_acc"])
        save_checkpoint(out / "last.pt", model, {"epoch": epoch, "val": va})
        score = va["mse"] + 0.5 * va["ce"]
        if score < best - 1e-5:
            best, bad = score, 0
            save_checkpoint(out / "best.pt", model, {"epoch": epoch, "val": va, "train_cfg": _jsonable(cfg)})
        else:
            bad += 1
            if bad >= cfg.patience:
                log.info("early stop at epoch %d", epoch)
                break
        (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    (out / "history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    _write_centroids(out, cfg, plugins)
    return out / "best.pt"


def _write_centroids(out: Path, cfg: TrainConfig, plugins: list[str]) -> None:
    """Per-plugin mean embedding of the training renders (for nearest-centroid plugin selection), from the best weights."""
    from plugin_selector.select_plugin import compute_centroids, save_centroids
    from tone_model.checkpoint import load_checkpoint

    model, _, _ = load_checkpoint(out / "best.pt", cfg.device)
    loader = make_loader(cfg.shards, "train", plugins, cfg.groups_per_batch, False, cfg.workers, cfg.max_groups, cfg.source_filter)
    save_centroids(out / "centroids.npz", compute_centroids(model, loader, cfg.device, max_batches=60))


def _split_by_plugin(mod_loss: dict[str, float], book, schemas) -> dict[str, dict[str, float]]:
    out = {}
    for pid, (num_mods, _) in book.items():
        out[schemas[pid].plugin_name] = {m: mod_loss[m] for m in set(num_mods) if m in mod_loss}
    return out


def _jsonable(cfg: TrainConfig) -> dict:
    d = asdict(cfg)
    return d


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    d = TrainConfig()
    for f, v in asdict(d).items():
        if f in ("weights", "plugins", "head_hidden", "source_filter"):
            continue
        t = type(v) if v is not None and not isinstance(v, bool) else (str if v is None else bool)
        if isinstance(v, bool):
            ap.add_argument(f"--{f.replace('_', '-')}", action="store_true", default=v)
        elif v is None:
            ap.add_argument(f"--{f.replace('_', '-')}", type=int, default=None)
        else:
            ap.add_argument(f"--{f.replace('_', '-')}", type=t, default=v)
    ap.add_argument("--plugins", nargs="*", default=[])
    ap.add_argument("--head-hidden", type=int, nargs="+", default=d.head_hidden, help="widths of the stage-2 hidden layers")
    ap.add_argument("--source-filter", choices=["all", "0", "1"], default="all", help="0 random renders, 1 community presets")
    a = ap.parse_args(argv)
    kw = {k: v for k, v in vars(a).items()}
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    print(train(TrainConfig(**kw)))


if __name__ == "__main__":
    main()
