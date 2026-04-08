# syntax=docker/dockerfile:1.7
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Base OS deps commonly needed by audio/text preprocessing toolchain.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml LICENSE README.md /app/
COPY tribev2 /app/tribev2

# Install project + runtime deps.
# Keep pip cache between builds when this layer is re-executed.
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --upgrade pip && \
    pip install -e . && \
    pip install whisperx flask anthropic google-genai && \
    pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0

# CPU-forcing patches required by this repo's workflow.
RUN python - <<'PY'
from pathlib import Path
import site

site_packages = Path(site.getsitepackages()[0])
text_py = site_packages / "neuralset" / "extractors" / "text.py"
audio_py = site_packages / "neuralset" / "extractors" / "audio.py"
events_py = Path("/app/tribev2/eventstransforms.py")

def rewrite(path, old, new):
    s = path.read_text()
    if old in s:
        s = s.replace(old, new)
        path.write_text(s)

rewrite(text_py, 'model.to(self.device)', 'model.to("cpu")')
rewrite(text_py, ').to(device)', ').to("cpu")')
rewrite(text_py, 'device = "cuda" if torch.cuda.is_available() else "cpu"', 'device = "cpu"')
rewrite(audio_py, '_model.to(self.device)', '_model.to("cpu")')
rewrite(audio_py, 'features.to(self.device)', 'features.to("cpu")')
rewrite(events_py, 'compute_type = "float16"', 'compute_type = "int8"')
PY

# Copy frequently-changing app files after heavy dependency layers
# so small app edits don't force full dependency reinstall.
COPY hookbrain /app/hookbrain
COPY test_hook.py /app/test_hook.py
COPY run_hooks.sh /app/run_hooks.sh

EXPOSE 5050

CMD ["python", "hookbrain/app.py"]
