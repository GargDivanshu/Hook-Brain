# HookBrain

Run any short-form video hook through Meta's TRIBE v2 fMRI model to predict cortical brain activation. Scores hooks on virality mechanics and uses Claude to generate brain-optimized rewrites. Runs fully local on CPU (MacBook M-series).

## What it does

- Converts hook text → speech → word-level transcription (WhisperX)
- Runs through TRIBE v2 to predict activation across 20,484 cortical vertices
- Scores on 5 viral mechanics: watch signal, emotional onset, right hemisphere dominance, drop-off risk, sustained engagement
- Sends brain data to Claude API to generate 5 rewrites targeting different neural mechanics
- Flask web UI at http://127.0.0.1:5050

## Requirements

- MacBook Apple Silicon (M1/M2/M3/M4)
- Python 3.12 via pyenv
- HuggingFace account with access to:
  - https://huggingface.co/facebook/tribev2
  - https://huggingface.co/meta-llama/Llama-3.2-3B
- Anthropic API key
