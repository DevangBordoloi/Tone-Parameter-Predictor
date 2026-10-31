# Tone Parameter Estimation: song / URL / guitar stem -> Neural DSP preset JSON + Markdown report.
#
#   docker build -t tone-estimation .                                   # CPU
#   docker build -t tone-estimation --build-arg TORCH_INDEX=cu128 .     # NVIDIA GPU (run with --gpus all)
#
#   docker run --rm -v "$PWD/output:/app/output" tone-estimation pipeline.run_pipeline "https://youtu.be/..." --focus rhythm
#
# The Neural DSP plugins are Windows/macOS only, so inside the container presets are predicted but not rendered:
# --ito, --check, --preview and --apply-reaper need a local install with the licensed .vst3 plugins.

FROM denoland/deno:bin-2.1.4 AS deno

FROM python:3.12-slim

ARG TORCH_INDEX=cpu
ARG TORCH_VERSION=2.11.0

RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*
# yt-dlp needs a JavaScript runtime for YouTube
COPY --from=deno /deno /usr/local/bin/deno

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TONE_ASSUME_INSTALLED=1 \
    OMP_NUM_THREADS=1

WORKDIR /app

RUN pip install torch==${TORCH_VERSION} torchaudio==${TORCH_VERSION} --index-url https://download.pytorch.org/whl/${TORCH_INDEX}
COPY requirements.txt .
RUN pip install -r requirements.txt "yt-dlp[default]"

# bake the Demucs weights (cascade separation uses both) into the image so runs work offline
RUN python -c "from demucs.pretrained import get_model; get_model('htdemucs'); get_model('htdemucs_6s')"

COPY . .

ENTRYPOINT ["python", "-m"]
CMD ["pipeline.run_pipeline", "--help"]
