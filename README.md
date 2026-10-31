# Tone Parameter Estimation

An end-to-end neural audio system that estimates **Neural DSP plugin parameters** directly from music audio. Given a YouTube URL, full mix, or isolated guitar track, the system isolates the guitar using Demucs, extracts harmonic and temporal timbre features via **ToneNet**, predicts all amplifier and FX parameters, and generates both machine-readable preset JSON and human-readable Markdown reports—with optional direct integration into REAPER DAW.

Runs **100% locally** on GPU/CPU with **zero LLM dependencies** in the estimation engine.

---

## Directory Structure

```
Tone Parameter Estimation/
├── guitar_stem_isolator/       # Stage 1: Audio ingestion & stem isolation
│   ├── downloader.py          # yt-dlp wrapper with time-range slicing
│   ├── separator.py           # Demucs guitar isolation (CUDA/CPU, cascade/single)
│   └── isolate.py             # Mid/Side channel extraction and sanity checks
│
├── tone_model/                # Stage 2: Neural network architecture
│   ├── cqt.py                 # GPU CQT / HCQT (Harmonic Constant-Q Transform)
│   ├── branches.py            # Timbre, temporal, and HD-Conv feature branches
│   ├── modulation.py          # LFO-net & gammatone modulation filterbank
│   ├── dynamics.py            # Envelope followers & Temporal FiLM (TFiLM)
│   ├── heads.py               # Two-stage FiLM parameter prediction heads
│   ├── net.py                 # ToneNet top-level model
│   └── losses.py              # Active-parameter masked loss functions
│
├── tone_parameter_estimation/ # Stage 3: Inference & parameter estimation
│   ├── infer.py               # TonePredictor: windowing, centroid matching, inference
│   └── ito.py                 # Optional Inference-Time Optimization via plugin render
│
├── plugin_selector/           # Plugin routing & classifier
│   └── select_plugin.py       # Plugin substitution & embedding centroid matching
│
├── pipeline/                  # End-to-end execution & integrations
│   ├── run_pipeline.py        # Master pipeline CLI (URL/Stem -> Preset + Report)
│   ├── report.py              # Human-readable Markdown preset report generator
│   ├── tune.py                # Local CLI to inspect, tune, and push presets to REAPER
│   ├── reaper_driver.py       # Standalone REAPER bridge driver (~/.reaper-mcp)
│   ├── mcp_server.py          # Local MCP server exposing 6 tools (Zero LLMs)
│   └── mcp_config.example.json# Template configuration for MCP clients
│
├── dataset_builder/           # Training dataset generation
│   ├── di_corpus.py           # Clean DI audio clip management
│   ├── activity.py            # Active parameter dependency graphs per plugin
│   ├── param_sampler.py       # Physically plausible parameter sampling
│   ├── render_dataset.py      # VST3 plugin rendering via Pedalboard
│   ├── augment.py             # Room impulse responses, EQ, noise augmentations
│   └── shards.py              # HDF5 / NPZ dataset sharding
│
├── training/                  # Model training loops
│   ├── train.py               # Local PyTorch training with AMP and cosine annealing
│   ├── data.py                # Contrastive dataset loader (hard negatives)
│   └── modal_train.py         # Modal cloud-GPU runner
│
├── evaluation/                # Benchmarking & metrics
│   ├── evaluate.py            # Multi-tier evaluation runner
│   ├── metrics.py             # Knob MAE, switch accuracy, spectral distance
│   └── check.py               # Comparison against community reference presets
│
├── common/                    # Shared utilities
│   ├── schema.py              # Plugin parameter schemas & physical unit conversions
│   ├── paths.py               # Project workspace paths
│   └── dsp.py                 # Audio DSP helper functions
│
├── patch_scraper/             # Community preset collection (Katana, Line 6, ...)
├── patch_corpus/              # Decodes scraped patches and translates them to Neural DSP params
├── preset_corpus/             # Neural DSP preset corpus handling
│
├── config/
│   ├── plugin_schemas/        # JSON parameter schemas for the Neural DSP plugins
│   ├── artist_plugins.json    # Fallback artist-to-plugin mapping table
│   └── famous_artists.json    # Artist/genre list used by the scraper
│
├── runs/                      # Model checkpoints
│   ├── pretrain/best.pt       # Stage 1: Synthetic pretraining weights
│   └── finetune/best.pt       # Stage 2: Fine-tuned weights (default)
│
├── eval/                      # Tier-1 evaluation results (finetune_val, pretrain_test)
├── output/                    # Example preset JSON + Markdown reports
├── docs/
│   ├── EVALUATION.md          # Evaluation audit and next steps
├── Dockerfile                 # Containerised inference / training / tests
│
└── tests/                     # Comprehensive test suite (120+ unit tests)
```

`data/` (≈13 GB of rendered shards, DI audio and scraped presets) is not in the repository.

---

## Required Modules & Libraries

### 1. Requirements File (`requirements.txt`)

```text
# Core deep learning & audio processing
torch
torchaudio
numpy
scipy
pydantic>=2
soundfile

# Local VST3 plugin rendering (dataset generation & ITO)
pedalboard>=0.9

# Storage
h5py

# Audio download & stem isolation
yt-dlp
demucs
imageio-ffmpeg
truststore

# MCP Server integration
mcp>=1.2
modal

# Testing
pytest
```

### 2. Environment Setup

It is strongly recommended to use a virtual environment with CUDA-enabled PyTorch:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install PyTorch with CUDA support (e.g. CUDA 12.8 / 12.x)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# Install remaining dependencies
pip install -r requirements.txt
```

### 3. Docker

The image bundles the trained checkpoints and the Demucs weights, so no Python setup is needed:

```bash
docker build -t tone-estimation .                                  # CPU
docker build -t tone-estimation --build-arg TORCH_INDEX=cu128 .    # NVIDIA GPU

# Estimate a preset from a YouTube URL (results land in ./output)
docker run --rm -v "$PWD/output:/app/output" tone-estimation \
  pipeline.run_pipeline "https://www.youtube.com/watch?v=eIMb2lDIaYM" --start 30 --duration 20 --focus rhythm --song "What I've Done"

# From a local audio file
docker run --rm -v "$PWD/output:/app/output" -v "$PWD/audio:/audio" tone-estimation \
  pipeline.run_pipeline /audio/song.wav --focus lead

# GPU: add --gpus all.  Other entry points: pipeline.tune, evaluation.evaluate, training.train, pytest tests/
docker run --rm tone-estimation pytest tests/ -q
```

Neural DSP plugins only exist for Windows and macOS, so the container predicts presets but cannot render them:
`--ito`, `--check`, `--preview` and `--apply-reaper` need a local install with the licensed `.vst3` plugins
(and REAPER). Inside the container `TONE_ASSUME_INSTALLED=1` lets the model pick any of the plugins it was trained on.
For training, mount the shards: `-v /path/to/shards_v2:/app/shards_v2 -v "$PWD/runs:/app/runs"`.

### Module Roles:
- **`torch` & `torchaudio`**: Powers ToneNet's multi-branch neural network, GPU CQT transforms, and CUDA-accelerated Demucs separation.
- **`demucs`**: High-fidelity music source separation isolating guitar from full mixes (`htdemucs` / `htdemucs_6s`).
- **`yt-dlp` & `imageio-ffmpeg`**: Fetches and slices audio directly from YouTube URLs into 48 kHz WAV.
- **`pedalboard`**: Loads and renders VST3 audio plugins locally for dataset creation and parameter validation.
- **`pydantic`**: Validates preset JSON schemas and parameter bounds.
- **`mcp`**: Provides the Model Context Protocol interface for optional agent-based workflows.

---

## Supported Plugins

The model is trained on 7 Neural DSP guitar plugins (schemas in `config/plugin_schemas/`):
1. **Archetype Abasi**
2. **Archetype Gojira**
3. **Archetype John Mayer X**
4. **Archetype Misha Mansoor X**
5. **Archetype Nolly X**
6. **Archetype Petrucci X**
7. **Archetype Plini**

(`Mantra.json` is also in the schema folder but is excluded from training: it is a vocal plugin.)

---

## Mode 1: Local CLI & Inspectable Preset Reports (No LLM, No Server)

Mode 1 is completely standalone. It runs directly from the terminal, extracts tones, outputs inspectable preset files with human-readable Markdown reports, and allows you to manually or programmatically adjust knobs and push them into REAPER.

### Step 1: Run Tone Estimation
Run `pipeline.run_pipeline` with a YouTube URL or an existing audio file:

```powershell
# From a YouTube URL:
python -m pipeline.run_pipeline "https://www.youtube.com/watch?v=eIMb2lDIaYM" --start 30 --duration 20 --focus rhythm --song "What I've Done" --out output/my_tones

# Directly push into a running REAPER session:
python -m pipeline.run_pipeline "https://www.youtube.com/watch?v=eIMb2lDIaYM" --start 30 --duration 20 --focus rhythm --song "What I've Done" --group "What I've Done" --apply-reaper

# From an existing audio stem:
python -m pipeline.run_pipeline "path/to/guitar_stem.wav" --is-stem --focus lead --song "Solo Section" --out output/my_tones
```

#### CLI Options:
- `source`: YouTube URL or path to local audio file / stem.
- `--start`: Start offset in seconds (default: 30 for URLs; whole file for local audio).
- `--duration`: Number of seconds to analyze (default: 25 for URLs).
- `--focus`: Stereo focus channel: `rhythm` (Side channel, rhythm guitar), `lead` (Mid channel, centered lead guitar), or `full` (stereo).
- `--is-stem`: Treat the input as an already-isolated guitar stem (skips Demucs).
- `--plugin`: Force a specific plugin (e.g. `--plugin "Archetype Gojira"`); otherwise ToneNet auto-selects via embedding distance.
- `--apply-reaper`: Automatically sends the preset to the running REAPER instance via `reaper_mcp_bridge.lua`.
- `--group`: Creates a REAPER folder track named after the song to group left/right tracks.
- `--out`: Destination directory for output files.

---

### Step 2: Inspect and Tweak Presets Locally (`pipeline.tune`)

The `pipeline.tune` CLI tool lets you view and edit presets offline, or adjust live REAPER tracks in real-time.

```powershell
# 1. View the full formatted Markdown report in the terminal:
python -m pipeline.tune show output/my_tones/What_I_ve_Done_rhythm.json

# 2. Modify parameters (accepts raw numbers or textual switch labels; auto-updates .json and .md):
python -m pipeline.tune set output/my_tones/What_I_ve_Done_rhythm.json lead_gain=0.65 logos_active=Inactive

# 3. Apply an existing preset JSON into REAPER:
python -m pipeline.tune apply output/my_tones/What_I_ve_Done_rhythm.json --group "Linkin Park"

# 4. Live knob tweak on an active REAPER track:
python -m pipeline.tune live "What_I_ve_Done_rhythm" lead_gain=0.75

# 5. Read back current live parameter values from a REAPER track:
python -m pipeline.tune state "What_I_ve_Done_rhythm" --filter "gain"
```

---

### What Mode 1 Outputs

For every analysis run, Mode 1 generates two primary files:

#### 1. Machine-Readable Preset JSON (`<name>.json`)
A structured JSON file storing normalized `0.0` to `1.0` dial values, switch booleans, and track metadata compatible with DAW automation bridges:
```json
{
  "target_plugin": "Archetype Abasi",
  "vst_parameters": {
    "gate": 0.478,
    "lead_gain": 0.450,
    "lead_bass": 0.490,
    "lead_mid": 0.260,
    "lead_treble": 0.700,
    "logos_compression": 0.630,
    "logos_active": 1.0,
    "delay_active": 0.0
  },
  "daw_meta": {
    "double_track": true,
    "pan_lr": [-100.0, 100.0]
  }
}
```

#### 2. Human-Readable Markdown Report (`<name>.md`)
A formatted markdown document organized by amplifier and pedal modules. It converts normalized numbers into **real physical units** (`dB`, `Hz`, `ms`, enum labels) and displays the model's cross-window standard deviation (`±spread`):

```markdown
# What_I've_Done_rhythm

**Plugin:** Archetype Abasi  
**Tracks:** What_I_ve_Done_rhythm (double-tracked, pan [-100.0, 100.0])

- **Amp:** Lead
- **Active modules:** cab, clean_eq, lead_eq, logos, pedals, time
- **Least certain knobs:** lead_gain (±0.05)

## global / amp
| parameter | value | normalized | +- |
|---|---|---|---|
| gate | -56.4 dB | 0.478 | 0.01 |
| lead_gain | 0.4 | 0.450 | 0.05 |
| lead_bass | 0.5 | 0.490 | 0.03 |
| lead_mid | 0.3 | 0.260 | 0.02 |
| lead_treble | 0.7 | 0.700 | 0.03 |

## logos (compressor)
| parameter | value | normalized | +- |
|---|---|---|---|
| logos_active | Active | 1.000 |  |
| logos_compression | 0.6 | 0.630 | 0.02 |
| logos_level | 0.4 | 0.430 | 0.00 |
```
- **`+-` spread**: Quantifies disagreement across 4-second audio windows. Knobs with high variance indicate parameters that vary across the performance or where the model has lower confidence, highlighting exactly what to fine-tune by ear.

#### 3. DAW Automation (when `--apply-reaper` is used)
- Creates track(s) in REAPER matching the song and focus name.
- Automatically handles **double-tracking**: creates Left (-100% Pan) and Right (+100% Pan) tracks with identical FX chains.
- Wraps tracks inside a parent folder track named via `--group`.
- Instantiates the target Neural DSP VST3 and writes all parameter automation values.

---

## Mode 2: Local MCP Server (Agent Tooling with Zero Internal LLM)

Mode 2 exposes the pipeline via the **Model Context Protocol (MCP)**. This allows external coding assistants or agent interfaces to execute tone extraction and REAPER manipulation as callable tools, while the server itself contains zero internal LLM logic.

### MCP Server Location
[`pipeline/mcp_server.py`](file:///c:/Users/devan/Desktop/IIIT/Research/Tone%20Parameter%20Estimation/pipeline/mcp_server.py)

### Starting the Server
```powershell
python -m pipeline.mcp_server
```

### Configuration Snippet
Add the server to your MCP client configuration (e.g., Claude Desktop, IDE settings):

```json
{
  "mcpServers": {
    "neural-tone": {
      "command": "c:\\Users\\devan\\Desktop\\IIIT\\Research\\Tone Parameter Estimation\\.venv\\Scripts\\python.exe",
      "args": ["-m", "pipeline.mcp_server"],
      "cwd": "c:\\Users\\devan\\Desktop\\IIIT\\Research\\Tone Parameter Estimation"
    }
  }
}
```

---

### Exposed MCP Tools

The server registers 6 distinct tools:

| Tool Name | Parameters | Description |
|---|---|---|
| `match_url_tone_model` | `source`, `focus`, `start`, `duration`, `plugin`, `song` | Extracts guitar from audio/URL, predicts tone parameters, and writes preset JSON and Markdown report. |
| `match_url_to_reaper` | `source`, `focus`, `start`, `duration`, `plugin`, `song`, `group` | Runs full tone extraction and immediately pushes the track and VST3 settings into REAPER. |
| `apply_preset_to_reaper` | `json_path`, `group` | Applies an existing preset JSON into the running REAPER session. |
| `adjust_reaper_params` | `track`, `updates`, `plugin` | Modifies active parameters on a running REAPER track in real time. |
| `get_reaper_track_state` | `track`, `plugin`, `filter` | Reads current live parameter values from a REAPER track. |
| `edit_preset_file` | `json_path`, `updates` | Updates an offline preset `.json` file and regenerates its `.md` report without opening REAPER. |

---

### What Mode 2 Outputs

When an external tool call is made, the MCP server returns structured JSON responses:

#### 1. From `match_url_tone_model`:
```json
{
  "plugin": "Archetype Abasi",
  "json_path": "output/url_test/What_I_ve_Done_rhythm.json",
  "report_path": "output/url_test/What_I_ve_Done_rhythm.md",
  "audio_path": "output/audio/What_I_ve_Done_Live_30-50s.wav",
  "stem_path": "output/audio/stems/What_I_ve_Done_Live_30-50s_guitar.wav",
  "report": "# What_I've_Done_rhythm\n\n**Plugin:** Archetype Abasi..."
}
```

#### 2. From `match_url_to_reaper` / `apply_preset_to_reaper`:
```json
{
  "status": "applied",
  "tracks": ["What_I_ve_Done_rhythm_L", "What_I_ve_Done_rhythm_R"],
  "group": "What I've Done",
  "plugin": "Archetype Abasi",
  "double_tracked": true,
  "parameters_written": 42
}
```

#### 3. From `get_reaper_track_state`:
```json
{
  "track": "What_I_ve_Done_rhythm",
  "plugin": "Archetype Abasi",
  "parameters": {
    "gate": 0.478,
    "lead_gain": 0.450,
    "lead_master": 0.670
  }
}
```

---

## REAPER Bridge Setup

The connection between Python and REAPER runs through a single lightweight file-based IPC protocol using `reaper_mcp_bridge.lua`.

1. Ensure the bridge script is installed in REAPER:
   `%APPDATA%\REAPER\Scripts\reaper_mcp_bridge.lua`
2. In REAPER, start the script via **Actions > Show action list > Run reaper_mcp_bridge.lua**.
3. The bridge monitors `~/.reaper-mcp/` for command requests and executes them with REAPER's native ReaScript API (`TrackFX_SetParamNormalized`, `InsertTrackAtIndex`, etc.).

---

## Running Verification Tests

To verify that the environment, models, schemas, and REAPER drivers are functioning correctly:

```powershell
# Run the complete test suite (120+ tests)
$env:OMP_NUM_THREADS="1"
pytest tests/ -v

# Run REAPER driver and tuning unit tests specifically
pytest tests/test_reaper_tune.py -v
```
