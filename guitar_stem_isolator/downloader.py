"""URL (yt-dlp) or local file -> 48 kHz stereo WAV of the requested slice."""

from __future__ import annotations

import logging
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from common.paths import OUTPUT_DIR, SAMPLE_RATE, ensure

log = logging.getLogger(__name__)

DEFAULT_WINDOW_SEC = 30.0
DEFAULT_WORK_DIR = OUTPUT_DIR / "audio"


class DownloadError(RuntimeError):
    pass


def use_system_trust_store() -> None:
    """Make TLS use the OS certificate store (antivirus that re-signs HTTPS breaks Python's bundled CA list)."""
    try:
        import truststore

        truststore.inject_into_ssl()
    except Exception as e:
        log.debug("truststore unavailable: %s", e)


def find_ffmpeg() -> str:
    env = os.environ.get("FFMPEG_PATH")
    if env and Path(env).exists():
        return env
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg

        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception as e:
        raise DownloadError("ffmpeg not found: install it, set FFMPEG_PATH, or pip install imageio-ffmpeg") from e


def slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", text).strip("_")[:60] or "audio"


def to_wav_48k(src: Path, dst: Path, start_sec: float | None = None, duration_sec: float | None = None) -> Path:
    """Any audio file -> 48 kHz stereo 24-bit WAV, optionally cutting a segment."""
    args = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-y"]
    if start_sec is not None:
        args += ["-ss", f"{start_sec:.3f}"]
    args += ["-i", str(src)]
    if duration_sec is not None:
        args += ["-t", f"{duration_sec:.3f}"]
    args += ["-ar", str(SAMPLE_RATE), "-ac", "2", "-c:a", "pcm_s24le", str(dst)]
    dst.parent.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(args, capture_output=True, text=True)
    if proc.returncode != 0:
        raise DownloadError(f"ffmpeg failed: {proc.stderr.strip()[-500:]}")
    return dst


def find_energetic_window(wav: Path, window_sec: float = DEFAULT_WINDOW_SEC) -> float:
    """Start (s) of the loudest `window_sec` stretch (1 s hop)."""
    import soundfile as sf

    data, sr = sf.read(str(wav), dtype="float32", always_2d=True)
    mono = data.mean(axis=1)
    if len(mono) / sr <= window_sec:
        return 0.0
    per_sec = np.array([np.mean(mono[i * sr:(i + 1) * sr] ** 2) for i in range(len(mono) // sr)])
    return float(np.argmax(np.convolve(per_sec, np.ones(int(window_sec)), mode="valid")))


def download_audio(url: str, start_sec: float | None = None, end_sec: float | None = None,
                   out_dir: str | os.PathLike[str] | None = None, name: str | None = None) -> Path:
    """Download `url`, write a 48 kHz stereo WAV of [start, end]; the default is the most energetic 30 s."""
    use_system_trust_store()
    try:
        import yt_dlp
    except ImportError as e:
        raise DownloadError("yt-dlp is not installed (pip install yt-dlp)") from e
    out = ensure(Path(out_dir) if out_dir else DEFAULT_WORK_DIR)
    with tempfile.TemporaryDirectory(prefix="tone_dl_") as tmp:
        opts = {"format": "bestaudio/best", "outtmpl": str(Path(tmp) / "src.%(ext)s"), "quiet": True, "no_warnings": True,
                "noplaylist": True, "noprogress": True, "ffmpeg_location": str(Path(find_ffmpeg()).parent)}
        try:
            with yt_dlp.YoutubeDL(opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except Exception as e:
            raise DownloadError(f"yt-dlp could not download {url}: {e}") from e
        files = [p for p in Path(tmp).iterdir() if p.stem == "src"]
        if not files:
            raise DownloadError("yt-dlp produced no audio file")
        title = name or slug(str(info.get("title") or info.get("id") or "audio"))
        full = to_wav_48k(files[0], Path(tmp) / "full.wav")
        if start_sec is None:
            start = find_energetic_window(full) if end_sec is None else max(0.0, end_sec - DEFAULT_WINDOW_SEC)
            end = end_sec if end_sec is not None else start + DEFAULT_WINDOW_SEC
        else:
            start, end = float(start_sec), (end_sec if end_sec is not None else float(start_sec) + DEFAULT_WINDOW_SEC)
        if end <= start:
            raise DownloadError(f"end_sec ({end}) must be after start_sec ({start})")
        dst = out / f"{title}_{int(start)}-{int(end)}s.wav"
        to_wav_48k(full, dst, start, end - start)
    log.info("saved %s", dst)
    return dst


def resolve_source(source: str, start_sec: float | None = None, duration_sec: float | None = None,
                   work_dir: Path | None = None) -> Path:
    """A local file is used as-is (cut to the window if one is given); anything else is treated as a URL."""
    p = Path(source).expanduser()
    if p.exists():
        if start_sec is None and duration_sec is None:
            return p
        dst = ensure(work_dir or DEFAULT_WORK_DIR) / f"{slug(p.stem)}_{int(start_sec or 0)}-{int((start_sec or 0) + (duration_sec or DEFAULT_WINDOW_SEC))}s.wav"
        return to_wav_48k(p, dst, start_sec, duration_sec or DEFAULT_WINDOW_SEC)
    end = None if start_sec is None or duration_sec is None else start_sec + duration_sec
    return download_audio(source, start_sec=start_sec, end_sec=end, out_dir=work_dir)
