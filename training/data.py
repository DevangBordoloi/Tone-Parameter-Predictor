"""Torch dataset over rendered shards. One item = one group of four renders (see dataset_builder/shards.py)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from common.dsp import sanitize_audio
from dataset_builder.shards import GROUP_SIZE, Shard, list_shards


SOURCE_FILTERS = ("all", "0", "1")      # shard `source` field: 0 = structured-random render, 1 = jittered community preset


class GroupDataset(Dataset):
    def __init__(self, shard_paths: list[Path], plugin_names: list[str], max_groups: int | None = None, source_filter: str = "all"):
        """max_groups: optional cap on groups taken from each shard (keeps every plugin represented in quick runs).
        source_filter: keep only groups whose four rows all have this `source` ("0" random, "1" preset, "all" no filter)."""
        if source_filter not in SOURCE_FILTERS:
            raise ValueError(f"source_filter must be one of {SOURCE_FILTERS}, got {source_filter!r}")
        self.paths = [Path(p) for p in shard_paths]
        self.plugin_index = {n: i for i, n in enumerate(plugin_names)}
        self._open: dict[int, Shard] = {}
        self.index: list[tuple[int, int]] = []
        self.shard_plugin: list[int] = []
        self.n_guarded = 0                                # rows peak-normalised by the amplitude guard (counted per worker)
        for si, p in enumerate(self.paths):
            s = Shard(p)
            name = str(s.attrs["plugin"])
            if name not in self.plugin_index:
                s.close()
                self.shard_plugin.append(-1)
                continue
            self.shard_plugin.append(self.plugin_index[name])
            groups = range(s.n_groups)
            if source_filter != "all":
                src = s.read("source", slice(0, s.n_groups * GROUP_SIZE)).reshape(s.n_groups, GROUP_SIZE)
                groups = np.nonzero((src == int(source_filter)).all(axis=1))[0].tolist()
            self.index += [(si, g) for g in groups[: max_groups or len(groups)]]       # cap applies per shard
            s.close()

    def __len__(self) -> int:
        return len(self.index)

    def _shard(self, si: int) -> Shard:
        if si not in self._open:                      # opened lazily per worker process (h5py handles are not fork-safe)
            self._open[si] = Shard(self.paths[si])
        return self._open[si]

    def __getitem__(self, i: int) -> dict[str, torch.Tensor]:
        si, g = self.index[i]
        d = self._shard(si).group(g)
        audio, flagged = sanitize_audio(d["audio"])       # exploded renders (peak > 2 or rms > 1) must not reach the encoder
        self.n_guarded += int(flagged.sum())
        return {
            "audio": torch.from_numpy(audio),
            "params": torch.from_numpy(d["params"].astype(np.float32)),
            "mask": torch.from_numpy(d["mask"].astype(bool)),
            "slot": torch.from_numpy(d["slot"].astype(np.int64)),
            "plugin": torch.full((GROUP_SIZE,), self.shard_plugin[si], dtype=torch.long),
            "gid": torch.full((GROUP_SIZE,), i, dtype=torch.long),
        }


def collate(items: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    """Groups from different plugins have different parameter counts, so params/mask are right-padded to the widest."""
    width = max(it["params"].shape[1] for it in items)
    pad = lambda t, v: torch.nn.functional.pad(t, (0, width - t.shape[1]), value=v)
    return {
        "audio": torch.cat([it["audio"] for it in items]),
        "params": torch.cat([pad(it["params"], 0.0) for it in items]),
        "mask": torch.cat([pad(it["mask"].float(), 0.0).bool() for it in items]),
        "slot": torch.cat([it["slot"] for it in items]),
        "plugin": torch.cat([it["plugin"] for it in items]),
        "gid": torch.cat([it["gid"] for it in items]),
    }


def make_loader(data_dir: str | Path, split: str, plugin_names: list[str], groups_per_batch: int = 8, shuffle: bool = True,
                workers: int = 0, max_groups: int | None = None, source_filter: str = "all") -> DataLoader:
    paths = list_shards(data_dir, split)
    ds = GroupDataset(paths, plugin_names, max_groups, source_filter)
    if len(ds) == 0:
        raise FileNotFoundError(f"no '{split}' groups (source={source_filter}) for {plugin_names} in {data_dir}")
    return DataLoader(ds, batch_size=groups_per_batch, shuffle=shuffle, num_workers=workers, collate_fn=collate,
                      drop_last=shuffle and len(ds) > groups_per_batch, persistent_workers=workers > 0)
