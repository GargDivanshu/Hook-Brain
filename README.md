# HookBrain

Run any short-form video hook through Meta's TRIBE v2 fMRI model to predict cortical brain activation. Scores hooks on 5 viral mechanics and uses Claude to generate brain-optimized rewrites. Runs fully local on CPU (MacBook Apple Silicon).

---

## What it does

- Converts hook text to speech via gTTS, then word-level transcription via WhisperX
- Runs through TRIBE v2 to predict activation across 20,484 cortical vertices
- Scores on 5 viral mechanics: watch signal, emotional onset, right hemisphere dominance, drop-off risk, sustained engagement
- Sends brain data to Claude API to generate 5 rewrites targeting different neural mechanics
- Flask web UI at http://127.0.0.1:5050 with scan history stored in SQLite

---

## Requirements

- MacBook Apple Silicon (M1/M2/M3/M4/M5) — CPU-only, no GPU needed
- macOS with Homebrew installed
- Python 3.12 specifically (not 3.13 or 3.14 — both will fail)
- HuggingFace account with access approved for both gated models below
- Anthropic API key

HuggingFace gated models — request access before starting:
- https://huggingface.co/facebook/tribev2
- https://huggingface.co/meta-llama/Llama-3.2-3B

Approval for both is usually granted within a few minutes.

---

## First-run model downloads (~9GB total)

The first time you run a scan, these will download automatically:

| Model | Size |
|---|---|
| facebook/tribev2 (best.ckpt) | 709 MB |
| meta-llama/Llama-3.2-3B | ~6.4 GB |
| facebook/w2v-bert-2.0 | 2.3 GB |
| en-core-web-lg (spaCy) | 400 MB |

This only happens once. All models are cached after the first run.

---

## Setup

### 1. Add Homebrew to your PATH permanently
```bash
echo 'export PATH="/opt/homebrew/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc
```

Without this, uvx won't be found when WhisperX tries to use it internally.

### 2. Install pyenv and uv
```bash
brew install pyenv uv
```

### 3. Install Python 3.12
```bash
pyenv install 3.12.0
```

Do not use the system Python. macOS ships with Python 3.14 which cannot install torch<2.7 and will fail immediately.

### 4. Clone this repo and set local Python version
```bash
git clone https://github.com/subkap88-coder/Hook-Brain.git
cd Hook-Brain
pyenv local 3.12.0
```

### 5. Create the virtual environment using Python 3.12 explicitly
```bash
~/.pyenv/versions/3.12.0/bin/python3 -m venv venv
source venv/bin/activate
```

Do not use python3 -m venv venv directly — that picks up system Python 3.14 and breaks the torch install.

### 6. Install TRIBE v2 and its dependencies
```bash
pip install -e .
```

This installs torch==2.6.0 and all neuralset/neuraltrain dependencies.

### 7. Install WhisperX
```bash
pip install whisperx
```

This upgrades torch to 2.8.0 as a side effect. You will see warnings like:
tribev2 0.1.0 requires torch<2.7,>=2.5.1, but you have torch 2.8.0
This is expected. Continue.

### 8. Pin torch back to 2.6.0
```bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0
```

WhisperX pulls torch to 2.8.0 which breaks TRIBE v2. This pins it back. Conflict warnings are expected — ignore them.

### 9. Install Flask and Anthropic
```bash
pip install flask anthropic
```

### 10. Log in to HuggingFace
```bash
python -c "from huggingface_hub import login; login(token='YOUR_HF_TOKEN')"
```

Use a token with read scope from https://huggingface.co/settings/tokens. Your account must already have approved access to both gated models.

### 11. Apply CPU patches (all 6 required)

TRIBE v2 was written for CUDA. These patches force everything to run on CPU:
```bash
sed -i '' 's/model.to(self.device)/model.to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/text.py
sed -i '' 's/).to(device)/).to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/text.py
sed -i '' 's/device = "cuda" if torch.cuda.is_available() else "cpu"/device = "cpu"/' venv/lib/python3.12/site-packages/neuralset/extractors/text.py
sed -i '' 's/_model.to(self.device)/_model.to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/audio.py
sed -i '' 's/features.to(self.device)/features.to("cpu")/' venv/lib/python3.12/site-packages/neuralset/extractors/audio.py
sed -i '' 's/compute_type = "float16"/compute_type = "int8"/' tribev2/eventstransforms.py
```

### 12. Apply the cache config patch (after first successful run only)

On the very first successful run, TRIBE v2 creates a config file that hardcodes device: cuda. After your first scan completes, run:
```bash
sed -i '' 's/device: cuda/device: cpu/' "./cache/neuralset.extractors.text.HuggingFaceText._get_data,release/"*"/config.yaml"
```

If this returns no matches found, the cache hasn't been created yet — run a scan first then come back to this step.

---

## Running the app

Every new terminal session:
```bash
source venv/bin/activate
export ANTHROPIC_API_KEY="your-key-here"
./hookbrain/run.sh
```

Open http://127.0.0.1:5050

---

## Run in a container (Docker / Portainer stack)

### Option A: Plain Docker

```bash
docker build -t hookbrain:cpu .
docker run --rm -it \
  -p 5050:5050 \
  -e ANTHROPIC_API_KEY="your-key" \
  -e HF_TOKEN="your-hf-token" \
  -v hookbrain_cache:/app/cache \
  -v hookbrain_db:/app/hookbrain/data \
  hookbrain:cpu
```

Then open: http://127.0.0.1:5050

### Option B: Portainer stack (`stack.yml`)

Use `deploy/hookbrain.stack.yml` as a **separate stack** from your existing services and set environment variables in Portainer:

- `ANTHROPIC_API_KEY`
- `LLM_PROVIDER` (`anthropic` or `gemini`)
- `GEMINI_API_KEY` (required only if using `LLM_PROVIDER=gemini`)
- `HF_TOKEN` (HuggingFace token with read access to gated models)
- `HOOKBRAIN_TAG` (optional; defaults to `main`, set to branch name like `divanshu`)

Notes:
- This stack is CPU-only.
- This stack deploys from Docker Hub image `gargdivanshu/hookbrain:${HOOKBRAIN_TAG}`.
- The GitHub Actions workflow publishes tags on every push:
  - pointer tag: `<branch>`
  - retention tag: `<branch>-<shortsha>`
- It publishes on host port `5051` (container `5050`) to avoid common `5050` conflicts.
- First run can be slow due to model download; volumes keep cache/db across restarts.

### API provider support

- Rewrites endpoint supports provider fallback via `LLM_PROVIDER`:
  - `anthropic` (default): set `ANTHROPIC_API_KEY` (+ optional `ANTHROPIC_MODEL`)
  - `gemini`: set `GEMINI_API_KEY` (+ optional `GEMINI_MODEL`)
- If provider output is not valid JSON, API returns a clear error payload.

---

## Resource planning (local vs AWS EC2)

This repository is configured to run on **CPU only**. You do **not** need a GPU for inference, and the project README/setup is optimized for Apple Silicon laptops.

If your local machine is resource-constrained (low RAM / weak CPU), using EC2 can be more stable. Recommended starting points:

- **t3.large / t3.xlarge** (budget testing): can run setup and light scans, but slower and may hit memory pressure during first-time downloads and preprocessing.
- **m7i.xlarge or m6i.xlarge** (recommended baseline): better sustained CPU performance and memory headroom for model loading + WhisperX + Flask.
- **c7i.xlarge** (CPU-speed prioritized): useful if you want faster scan turnaround and don't mind less RAM-per-vCPU than memory-oriented families.

Practical baseline for smoother runs:
- **4 vCPU / 16 GB RAM minimum**
- **50+ GB disk** (models/cache, virtualenv, logs, temporary files)

Notes:
- First run downloads ~9 GB of models; this is the heaviest step.
- This app currently processes **hook text** through TRIBE v2 (with internal TTS/transcription), not direct video-file upload in the Flask flow.
- You can start on CPU EC2 first; move to larger CPU instances only if scan latency is too high for your workflow.

### GPU precision (important)

- For the current codebase and documented setup, **GPU is not required**.
- Treat GPU as **optional acceleration only** if you later modify runtime patches and dependency behavior.
- Out-of-the-box path in this repo is CPU-first, including explicit CPU patch steps.

### Cheapest non-always-on deployment options

If your goal is **not paying for an always-running server**, these are practical options:

1. **AWS Batch on Fargate (CPU)** — best AWS-native pay-per-job option  
   - Submit one scan job per request, container starts, runs inference, exits.  
   - You pay compute only while the job is active.  
   - Better fit than Lambda for long-running startup + model load.

2. **Cloud Run Jobs (GCP) / Azure Container Apps Jobs** — similar pay-per-execution model  
   - Good for containerized CPU workloads with occasional traffic.

3. **Replicate-style serverless model endpoint**  
   - Works for this use case if you package the full environment and cache strategy.  
   - Expect cold starts and longer first-token latency due to large model initialization/download.

4. **Spot EC2 + queue-triggered worker (intermittent)**  
   - Very low cost if you can tolerate interruptions and occasional retries.  
   - Not fully serverless, but cheaper than 24/7 on-demand EC2.

### Why Lambda is usually a poor fit here

- This workflow includes large dependencies, heavyweight model startup, and multi-minute inference windows.
- Even if technically possible with container images and external storage, Lambda limits and cold starts make it operationally awkward and often not the cheapest in practice for this profile.

### Practical recommendation

- If you want lowest ongoing idle cost: start with **job-based containers** (AWS Batch on Fargate CPU).
- If you want simplest setup right now: use **EC2 CPU instance** and stop/start it as needed.
- If you want marketplace-style endpoint billing: use **Replicate-style deployment**, accepting slower cold starts.

### Can I run this on older Xeon CPU machines?

Yes—if you have enough RAM and disk, older multi-core Xeon systems can run this CPU pipeline.

Example class that should work (with slower latency): Intel Xeon E5 v3 generation with 12 physical cores / 24 threads.

Minimum practical checks before you commit to that machine:
- **RAM:** 16 GB minimum, 32 GB preferred for smoother preprocessing/inference.
- **Disk:** 50+ GB free (models + cache + venv + temp files).
- **Python:** 3.12 environment available and isolated.
- **Patience on first run:** initial model download/startup is heavy and can take significantly longer on older CPUs.

If your machine passes those checks, you can run locally and avoid cloud costs entirely; expect slower per-scan turnaround versus modern CPUs.

#### Go / no-go quick decision

- **GO (run locally now):** your machine has **12c/24t-class CPU**, **>=16 GB RAM** (prefer 32 GB), and **>=50 GB free disk**.
- **NO-GO (use cloud job runner):** RAM < 16 GB, disk < 50 GB free, or you need low latency.

For Linux servers, add swap if it is currently zero to reduce crash risk during heavy first-run preprocessing/model loading.

---

## CLI usage

Score a hook without the web UI:
```bash
source venv/bin/activate
echo "Your hook text here" > hook.txt
python test_hook.py
```

Batch test 5 rewrites with a comparison table:
```bash
./run_hooks.sh
```

---

## Troubleshooting

**No such file or directory: uvx**
Homebrew bin is not in your PATH. Run:
```bash
export PATH="/opt/homebrew/bin:$PATH"
```
Add to ~/.zshrc to make it permanent.

**Requested float16 compute type, but the target device does not support efficient float16**
The eventstransforms.py patch did not apply. Check:
```bash
grep "compute_type" tribev2/eventstransforms.py
```
Should say int8. If it says float16, re-run the patch from step 11.

**Torch not compiled with CUDA enabled**
One of the text.py or audio.py patches did not apply. Re-run all 5 sed commands from step 11.

**Model loading went wrong**
The cache config still has device: cuda. Run the cache patch from step 12.

**mktemp: mkstemp failed on /tmp/tribe_hook_XXXXXX.py: File exists**
A crashed run left a temp file behind:
```bash
rm -f /tmp/tribe_hook_XXXXXX.py
```

**No matching distribution found for torch<2.7,>=2.5.1**
You are using system Python instead of pyenv 3.12:
```bash
deactivate
rm -rf venv
~/.pyenv/versions/3.12.0/bin/python3 -m venv venv
source venv/bin/activate
```
Then restart from step 6.

**DataLoader worker exited unexpectedly**
Missing if __name__ == '__main__': guard in your script. See test_hook.py for the correct structure.

**First scan takes more than 10 minutes**
Normal — Llama-3.2-3B (6.4GB) is downloading. Subsequent scans take 3-5 minutes on Apple Silicon CPU.

---

## Warnings you can safely ignore

- LabelEncoder: event_types has not been set — TRIBE v2 uses a default fallback
- Missing events will be encoded using the default all-zero value — same as above
- FutureWarning: torch.cuda.amp.autocast is deprecated — upstream issue, does not affect results
- This DataLoader will create 20 worker processes, suggested max is 10 — performance warning only
- torchcodec is not installed correctly so built-in audio decoding will fail — WhisperX uses its own av library instead, works fine
- WARNING: There was an error checking the latest version of pip — network timeout, safe to ignore

---

## Built on

- [TRIBE v2](https://github.com/facebookresearch/tribev2) by Meta FAIR
- [WhisperX](https://github.com/m-bain/whisperX) by Max Bain
- [Claude](https://anthropic.com) by Anthropic
