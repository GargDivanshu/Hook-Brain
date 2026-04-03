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
