"""Train on a cloud GPU with Modal. Rendering stays local (the plugins are licensed to this machine); only the shards go up.

    pip install modal && modal setup
    modal run training/modal_train.py::upload --shards data/shards          # one-off: shards -> Modal volume
    modal run training/modal_train.py::train_remote --epochs 40              # trains on a GPU, writes to the volume
    modal run training/modal_train.py::download --out runs/exp1              # best.pt, centroids.npz, fx_probs.json, history.json

The same `training.train.train()` runs remotely, so a Colab notebook is equivalent: mount Drive, `pip install -r
requirements-train.txt`, upload the shards and call `python -m training.train --shards ... --out ...`.
"""

from __future__ import annotations

from pathlib import Path

try:
    import modal
except ImportError:  # the module is importable for inspection without modal installed
    modal = None

ROOT = Path(__file__).resolve().parents[1]

if modal is not None:
    app = modal.App("guitar-tone-estimation")
    volume = modal.Volume.from_name("guitar-tone-data", create_if_missing=True)
    image = (
        modal.Image.debian_slim(python_version="3.11")
        .pip_install("torch", "numpy", "scipy", "pydantic>=2", "h5py", "soundfile")
        .add_local_dir(str(ROOT), remote_path="/app", ignore=["data", "runs", "output", ".git", "__pycache__", "*.pt"])
    )

    @app.function(image=image, volumes={"/vol": volume}, gpu="A10G", timeout=60 * 60 * 12)
    def train_remote(epochs: int = 40, groups_per_batch: int = 16, frontend: str = "hcqt", out: str = "exp1", amp: bool = True) -> str:
        import sys

        sys.path.insert(0, "/app")
        from training.train import TrainConfig, train

        best = train(TrainConfig(shards="/vol/shards", out=f"/vol/runs/{out}", epochs=epochs, groups_per_batch=groups_per_batch,
                                 frontend=frontend, amp=amp, workers=4, device="cuda"))
        volume.commit()
        return str(best)

    @app.local_entrypoint()
    def upload(shards: str = "data/shards"):
        with volume.batch_upload(force=True) as up:
            up.put_directory(shards, "/shards")
        print(f"uploaded {shards} -> volume guitar-tone-data:/shards")

    @app.local_entrypoint()
    def download(out: str = "runs/exp1", name: str = "exp1"):
        dest = Path(out)
        dest.mkdir(parents=True, exist_ok=True)
        for fname in ("best.pt", "centroids.npz", "fx_probs.json", "history.json"):
            try:
                data = b"".join(volume.read_file(f"runs/{name}/{fname}"))
            except Exception as e:
                print(f"skip {fname}: {e}")
                continue
            (dest / fname).write_bytes(data)
            print(f"wrote {dest / fname}")
