"""Shard storage for rendered training groups (HDF5 when h5py is installed, compressed npz otherwise).

A *group* is four renders from two DI clips (a, b) and two parameter sets (t1, t2), stored in this order:
    0: (a, t1)  1: (b, t1)  2: (a, t2)  3: (b, t2)
Same parameters on different content (0-1, 2-3) are contrastive positives; same content with different parameters
(0-2, 1-3) are the hard negatives (Fx-Encoder++ / RelFx recipe).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

try:
    import h5py
except ImportError:  # npz fallback keeps the pipeline dependency-light
    h5py = None

GROUP_SIZE = 4
FIELDS = ("audio", "params", "mask", "group", "slot", "clip", "source")


class ShardWriter:
    """Accumulates groups for one plugin/split and writes a shard every `groups_per_shard` groups."""

    def __init__(self, out_dir: str | Path, plugin_name: str, split: str, plugin_id: int, store_sr: int, groups_per_shard: int = 500,
                 first_shard: int = 0):
        self.dir = Path(out_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.plugin_name, self.split, self.plugin_id, self.store_sr = plugin_name, split, plugin_id, store_sr
        self.groups_per_shard = groups_per_shard
        self.buf: dict[str, list] = {k: [] for k in FIELDS}
        self.manifest: list[dict] = []
        self.n_groups_buffered = 0
        self.shard_idx = first_shard                       # >0 lets several render rounds share one directory
        self.total_groups = 0
        self.paths: list[Path] = []

    def add_group(self, audio: np.ndarray, params: np.ndarray, mask: np.ndarray, clips: tuple[int, int, int, int], sources: tuple[int, int],
                  infos: tuple[dict | None, dict | None] = (None, None)) -> None:
        """audio (4, 2, S); params (2, P) = [t1, t2]; mask (2, P). `infos`: preset identity (id / name / song meta) of t1, t2,
        written to a `<shard>.presets.json` manifest next to the shard so every preset group can be traced to its song."""
        gid = self.total_groups
        if any(infos):
            self.manifest.append({"group": gid, "slot0": infos[0], "slot1": infos[1]})
        slots = (0, 0, 1, 1)
        for i in range(GROUP_SIZE):
            self.buf["audio"].append(audio[i].astype(np.float16))
            self.buf["params"].append(params[slots[i]].astype(np.float32))
            self.buf["mask"].append(mask[slots[i]].astype(bool))
            self.buf["group"].append(gid)
            self.buf["slot"].append(slots[i])
            self.buf["clip"].append(clips[i])
            self.buf["source"].append(sources[slots[i]])
        self.total_groups += 1
        self.n_groups_buffered += 1
        if self.n_groups_buffered >= self.groups_per_shard:
            self.flush()

    def flush(self) -> None:
        if not self.n_groups_buffered:
            return
        stem = f"{self.plugin_name.replace(' ', '_')}__{self.split}__{self.shard_idx:04d}"
        arrays = {k: np.stack(v) if k in ("audio", "params", "mask") else np.asarray(v) for k, v in self.buf.items()}
        attrs = {"plugin": self.plugin_name, "plugin_id": self.plugin_id, "split": self.split, "store_sr": self.store_sr}
        if h5py is not None:
            path = self.dir / f"{stem}.h5"
            with h5py.File(path, "w") as f:
                for k, v in arrays.items():
                    f.create_dataset(k, data=v, compression="gzip" if k == "audio" else None, chunks=(1,) + v.shape[1:] if k == "audio" else None)
                f.attrs.update(attrs)
        else:
            path = self.dir / f"{stem}.npz"
            np.savez_compressed(path, **arrays, attrs=json.dumps(attrs))
        if self.manifest:
            (self.dir / f"{stem}.presets.json").write_text(json.dumps(self.manifest, indent=1, ensure_ascii=False), encoding="utf-8")
        self.paths.append(path)
        self.shard_idx += 1
        self.buf = {k: [] for k in FIELDS}
        self.manifest = []
        self.n_groups_buffered = 0

    def close(self) -> list[Path]:
        self.flush()
        return self.paths


class Shard:
    """Read-only view of one shard file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        if self.path.suffix == ".h5":
            self._h5 = h5py.File(self.path, "r")
            self.attrs = dict(self._h5.attrs)
            self.n_records = len(self._h5["group"])
            self._npz = None
        else:
            self._h5 = None
            self._npz = np.load(self.path)
            self.attrs = json.loads(str(self._npz["attrs"]))
            self.n_records = len(self._npz["group"])
            self._cache = {k: self._npz[k] for k in FIELDS}
        self.n_groups = self.n_records // GROUP_SIZE

    def read(self, key: str, sl) -> np.ndarray:
        return np.asarray(self._h5[key][sl]) if self._h5 is not None else self._cache[key][sl]

    def group(self, g: int) -> dict[str, np.ndarray]:
        sl = slice(g * GROUP_SIZE, (g + 1) * GROUP_SIZE)
        return {k: self.read(k, sl) for k in FIELDS}

    def close(self) -> None:
        if self._h5 is not None:
            self._h5.close()


def list_shards(data_dir: str | Path, split: str, plugin: str | None = None) -> list[Path]:
    d = Path(data_dir)
    pat = f"{plugin.replace(' ', '_')}__{split}__*" if plugin else f"*__{split}__*"
    return sorted(p for p in d.glob(pat) if p.suffix in (".h5", ".npz"))
