import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from flask import Flask, jsonify, render_template, request

from cache_store import cache_rewrites, cache_scan, get_cached_scan
from db import (
    attach_rewrite_scan,
    get_history,
    get_rewrites_for_scan,
    get_scan,
    init_db,
    save_rewrites,
    save_scan,
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(SCRIPT_DIR)
_venv_python = os.path.join(REPO_ROOT, "venv", "bin", "python")
VENV_PYTHON = _venv_python if os.path.exists(_venv_python) else sys.executable
SCANNER = os.path.join(SCRIPT_DIR, "scanner.py")
UPLOAD_ROOT = Path(
    os.environ.get("HOOKBRAIN_UPLOAD_DIR", os.path.join(REPO_ROOT, "upload_cache"))
).resolve()
UPLOAD_TMP_DIR = UPLOAD_ROOT / ".chunks"
ALLOWED_VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".avi", ".webm"}
CHUNK_SIZE = 8 * 1024 * 1024

# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = CHUNK_SIZE + 1024 * 1024

# In-memory job store  { job_id: {status, result, error} }
_jobs: dict = {}
_lock = threading.Lock()

EXPECTED_MECHANICS = [
    "watch_signal",
    "self_relevance",
    "emotional_salience",
    "share_signal",
    "dropoff_prevention",
]

GENERIC_BANNED_PHRASES = [
    "this changes everything",
    "nobody talks about this",
    "you need to hear this",
    "this is crazy",
    "this is wild",
    "here's why",
    "did you know",
    "have you ever",
    "what happened next",
    "and it's insane",
]


def _dedupe_keep_order(items):
    out = []
    seen = set()
    for item in items:
        key = item.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def ensure_upload_dirs():
    UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)
    UPLOAD_TMP_DIR.mkdir(parents=True, exist_ok=True)


def safe_filename(name):
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name or "")
    cleaned = cleaned.strip("._")
    return cleaned or f"upload_{uuid.uuid4().hex}.mp4"


def ensure_within_upload_root(path):
    path = Path(path).resolve()
    try:
        path.relative_to(UPLOAD_ROOT)
    except ValueError as exc:
        raise ValueError("Path is outside configured upload root") from exc
    return path


def extract_concrete_anchors(hook_text: str):
    numeric = re.findall(r"\$?\d[\d,\.]*\s*[A-Za-z%]*", hook_text)
    title_case = re.findall(r"\b[A-Z][A-Za-z0-9&\-\+]{2,}\b", hook_text)
    quoted = re.findall(r'"([^"]+)"', hook_text)

    anchors = []
    anchors.extend(numeric)
    anchors.extend(title_case)
    anchors.extend(quoted)

    if not anchors:
        strong_words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9'\-]{4,}\b", hook_text)
        anchors.extend(strong_words[:4])

    return _dedupe_keep_order([a.strip() for a in anchors if a.strip()])[:6]


def derive_rewrite_goals(brain_data):
    viral = brain_data.get("viral", {})
    goals = []

    watch_signal = viral.get("watch_signal", 0)
    emotional_onset = viral.get("emotional_onset", 0)
    right_dom_t0 = viral.get("right_dom_t0", 0)
    dropoff_risk = viral.get("dropoff_risk", 0)
    mean_sustained = viral.get("mean_sustained", 0)

    if watch_signal < 0.01:
        goals.append(
            "Low watch signal: make the first 3 to 5 words hit faster with conflict, motion, or contrast."
        )
    else:
        goals.append(
            "Watch signal is acceptable: keep the opening immediate and avoid slow setup."
        )

    if emotional_onset < 0.3:
        goals.append(
            "Weak emotional onset: increase tension, novelty, status, money, fear, relief, or surprise."
        )

    if right_dom_t0 < 45:
        goals.append(
            "Opening reads too analytical: use more visceral wording and less explanation in the first beat."
        )
    elif right_dom_t0 <= 55:
        goals.append(
            "Opening is balanced: preserve clarity but add a sharper emotional edge."
        )
    else:
        goals.append("Opening already has emotional pull: do not over-explain it.")

    if dropoff_risk > 55:
        goals.append(
            "High drop-off risk: remove second-clause explanation and keep the hook to one clean thought."
        )
    elif dropoff_risk > 45:
        goals.append(
            "Moderate drop-off risk: tighten the back half and reduce analytical wording."
        )

    if mean_sustained < 0:
        goals.append(
            "Sustained engagement is weak: improve rhythm and keep the idea unresolved enough to pull forward."
        )

    return goals


def normalize_rewrites(raw_rewrites):
    if not isinstance(raw_rewrites, list):
        raise ValueError("LLM did not return a JSON array")

    by_mechanic = {}
    for item in raw_rewrites:
        if not isinstance(item, dict):
            continue

        mechanic = str(item.get("mechanic", "")).strip()
        hook = str(item.get("hook", "")).strip()
        why = str(item.get("why", "")).strip()
        if mechanic and hook:
            by_mechanic[mechanic] = {
                "mechanic": mechanic,
                "hook": hook,
                "why": why,
            }

    missing = [m for m in EXPECTED_MECHANICS if m not in by_mechanic]
    if missing:
        raise ValueError(f"Missing rewrite mechanics: {', '.join(missing)}")

    return [by_mechanic[m] for m in EXPECTED_MECHANICS]


def build_rewrite_prompt(hook_text, brain_data):
    anchors = extract_concrete_anchors(hook_text)
    rewrite_goals = derive_rewrite_goals(brain_data)
    anchor_text = ", ".join(anchors) if anchors else "No strong anchors detected"
    banned_text = ", ".join(f'"{p}"' for p in GENERIC_BANNED_PHRASES)

    return f"""You are rewriting a short-form video hook.

Your job is not to make it prettier. Your job is to preserve the original idea while making the opening more brain-grabbing and less generic.

First, do this analysis internally before you write:
1. Identify the factual spine of the hook: topic, claim, speaker or subject, audience, proof or detail, and emotional angle.
2. Identify the concrete anchors that must be preserved when possible.
3. Apply the rewrite goals from the brain scan.
4. Draft each rewrite, then silently reject any version that sounds generic, templated, or detached from the original claim.

Non-negotiable constraints:
- Same topic as the source hook.
- Same core claim as the source hook.
- Do not invent brands, names, people, numbers, outcomes, or facts.
- If the source hook contains a concrete anchor, preserve at least one anchor in every rewrite.
- If the source hook is vague, sharpen framing and contrast. Do not fabricate specifics.
- Each rewrite must feel spoken, compressed, and immediate.
- Each rewrite must take a different editorial angle, not just paraphrase the same line.
- Avoid these generic phrases and close variants: {banned_text}

Voice rules:
- Start with "I", "You", a name, a brand, a number, or a sharp claim.
- Prefer present tense and immediate verbs.
- Use fragments when useful.
- Land one contrast, tension point, or curiosity gap fast.
- Under 15 words is ideal. Never bloat the opening.
- Spoken, not essay-like.

Source hook:
"{hook_text}"

Concrete anchors to preserve when natural:
{anchor_text}

Brain scan summary:
{json.dumps(brain_data.get("viral", {}), indent=2)}

Rewrite goals:
{json.dumps(rewrite_goals, indent=2)}

Mechanic targets:
- watch_signal: strongest first 3 to 5 words, fast commitment
- self_relevance: make the viewer feel implicated or directly addressed
- emotional_salience: more tension, contrast, stakes, novelty, or felt emotion
- share_signal: make it feel like something worth repeating to another person
- dropoff_prevention: simplest phrasing, least analytical load, one clear thought

Return only a valid JSON array, no markdown fences, no extra text:
[
  {{"mechanic": "watch_signal", "hook": "...", "why": "one short sentence explaining the specific change"}},
  {{"mechanic": "self_relevance", "hook": "...", "why": "..."}},
  {{"mechanic": "emotional_salience", "hook": "...", "why": "..."}},
  {{"mechanic": "share_signal", "hook": "...", "why": "..."}},
  {{"mechanic": "dropoff_prevention", "hook": "...", "why": "..."}}
]"""


# ---------------------------------------------------------------------------
# Background scan worker
# ---------------------------------------------------------------------------
def _run_scan(
    job_id: str,
    scan_label: str,
    parent_scan_id=None,
    mechanic=None,
    rewrite_id=None,
    input_mode="text",
    input_value=None,
):
    tmp = tempfile.mktemp(suffix=".json")
    try:
        with _lock:
            _jobs[job_id]["status"] = "running"

        proc = subprocess.run(
            [
                VENV_PYTHON,
                SCANNER,
                "--mode",
                input_mode,
                "--input",
                input_value if input_value is not None else scan_label,
                "--output",
                tmp,
            ],
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            timeout=900,
        )

        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            if "Cannot access gated repo" in stderr or "gated repo" in stderr:
                raise RuntimeError(
                    "HuggingFace gated model access missing. "
                    "Approve access for facebook/tribev2 and meta-llama/Llama-3.2-3B, "
                    "then re-login with HF_TOKEN and retry."
                )
            raise RuntimeError(stderr[-3000:] if stderr else "Scanner exited non-zero")

        with open(tmp) as f:
            data = json.load(f)

        if input_mode == "video":
            data.setdefault("metadata", {})
            data["metadata"]["input_mode"] = "video"
            data["metadata"]["source_path"] = input_value
            data["metadata"]["source_name"] = os.path.basename(input_value)
            data["hook"] = scan_label

        scan_record = save_scan(
            scan_label,
            data,
            parent_scan_id=parent_scan_id,
            mechanic=mechanic,
        )
        cache_scan(scan_record)
        if rewrite_id:
            attach_rewrite_scan(rewrite_id, scan_record["id"])

        with _lock:
            _jobs[job_id]["status"] = "done"
            _jobs[job_id]["result"] = {**data, "scan_id": scan_record["id"]}

    except Exception as exc:
        with _lock:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["error"] = str(exc)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _start_job(
    scan_label: str,
    parent_scan_id=None,
    mechanic=None,
    rewrite_id=None,
    input_mode="text",
    input_value=None,
) -> str:
    job_id = str(uuid.uuid4())
    with _lock:
        _jobs[job_id] = {"status": "queued", "result": None, "error": None}
    threading.Thread(
        target=_run_scan,
        args=(
            job_id,
            scan_label,
            parent_scan_id,
            mechanic,
            rewrite_id,
            input_mode,
            input_value,
        ),
        daemon=True,
    ).start()
    return job_id


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    hook_text = (request.json or {}).get("hook", "").strip()
    if not hook_text:
        return jsonify({"error": "No hook text"}), 400
    job_id = _start_job(hook_text)
    return jsonify({"job_id": job_id})


@app.route("/api/upload/init", methods=["POST"])
def api_upload_init():
    body = request.json or {}
    filename = safe_filename(body.get("filename", ""))
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_VIDEO_SUFFIXES:
        return jsonify({"error": "Unsupported video format"}), 400

    ensure_upload_dirs()
    upload_id = uuid.uuid4().hex
    staging_dir = UPLOAD_TMP_DIR / upload_id
    staging_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "upload_id": upload_id,
        "filename": filename,
        "total_size": int(body.get("size") or 0),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (staging_dir / "manifest.json").write_text(json.dumps(manifest))
    return jsonify({
        "upload_id": upload_id,
        "chunk_size": CHUNK_SIZE,
        "filename": filename,
    })


@app.route("/api/upload/chunk", methods=["POST"])
def api_upload_chunk():
    ensure_upload_dirs()
    upload_id = (request.form.get("upload_id") or "").strip()
    chunk_index = request.form.get("chunk_index")
    file = request.files.get("chunk")

    if not upload_id or chunk_index is None or file is None:
        return jsonify({"error": "Missing upload chunk fields"}), 400

    staging_dir = ensure_within_upload_root(UPLOAD_TMP_DIR / upload_id)
    if not staging_dir.exists():
        return jsonify({"error": "Upload session not found"}), 404

    chunk_path = staging_dir / f"{int(chunk_index):08d}.part"
    file.save(str(chunk_path))
    return jsonify({"ok": True})


@app.route("/api/upload/complete", methods=["POST"])
def api_upload_complete():
    body = request.json or {}
    upload_id = (body.get("upload_id") or "").strip()
    total_chunks = int(body.get("total_chunks") or 0)
    if not upload_id or total_chunks <= 0:
        return jsonify({"error": "Invalid upload completion payload"}), 400

    staging_dir = ensure_within_upload_root(UPLOAD_TMP_DIR / upload_id)
    manifest_path = staging_dir / "manifest.json"
    if not manifest_path.exists():
        return jsonify({"error": "Upload session not found"}), 404

    manifest = json.loads(manifest_path.read_text())
    final_name = f"{upload_id}_{manifest['filename']}"
    final_path = ensure_within_upload_root(UPLOAD_ROOT / final_name)

    with open(final_path, "wb") as out:
        for idx in range(total_chunks):
            chunk_path = staging_dir / f"{idx:08d}.part"
            if not chunk_path.exists():
                return jsonify({"error": f"Missing chunk {idx}"}), 400
            with open(chunk_path, "rb") as src:
                shutil.copyfileobj(src, out)

    shutil.rmtree(staging_dir, ignore_errors=True)
    return jsonify({
        "video_path": str(final_path),
        "filename": manifest["filename"],
    })


@app.route("/api/scan_video", methods=["POST"])
def api_scan_video():
    video_path = (request.json or {}).get("video_path", "").strip()
    if not video_path:
        return jsonify({"error": "No video path"}), 400

    try:
        video_file = ensure_within_upload_root(video_path)
    except ValueError:
        return jsonify({"error": "Video path is outside configured upload root"}), 400

    if not video_file.exists() or not video_file.is_file():
        return jsonify({"error": "Video file not found"}), 404

    label = f"[VIDEO] {video_file.name}"
    job_id = _start_job(
        label,
        input_mode="video",
        input_value=str(video_file),
    )
    return jsonify({"job_id": job_id, "label": label})


@app.route("/api/scan/<job_id>")
def api_scan_status(job_id):
    with _lock:
        job = dict(_jobs.get(job_id) or {})
    if not job:
        return jsonify({"error": "Job not found"}), 404
    return jsonify(job)


@app.route("/api/rewrites", methods=["POST"])
def api_rewrites():
    body = request.json or {}
    hook_text = body.get("hook", "")
    brain_data = body.get("brain_data", {})
    scan_id = body.get("scan_id")
    provider = os.environ.get("LLM_PROVIDER", "anthropic").strip().lower()
    prompt = build_rewrite_prompt(hook_text, brain_data)

    try:
        if provider == "gemini":
            from google import genai

            api_key = os.environ.get("GEMINI_API_KEY", "")
            if not api_key:
                return jsonify({"error": "GEMINI_API_KEY not set"}), 500

            model_name = os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")
            client = genai.Client(api_key=api_key)
            resp = client.models.generate_content(
                model=model_name,
                contents=prompt,
            )
            text = (resp.text or "").strip()
        else:
            import anthropic

            api_key = os.environ.get("ANTHROPIC_API_KEY", "")
            if not api_key:
                return jsonify({"error": "ANTHROPIC_API_KEY not set"}), 500

            model_name = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-6")
            client = anthropic.Anthropic(api_key=api_key)
            msg = client.messages.create(
                model=model_name,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            text = msg.content[0].text.strip()
    except Exception as exc:
        return jsonify({
            "error": "Rewrite provider request failed",
            "provider": provider,
            "detail": str(exc)[:500],
        }), 500

    if "```" in text:
        parts = text.split("```")
        text = parts[1]
        if text.startswith("json"):
            text = text[4:]
    text = text.strip()

    try:
        rewrite_list = normalize_rewrites(json.loads(text))
    except (json.JSONDecodeError, ValueError) as exc:
        return jsonify({
            "error": "LLM did not return valid rewrite payload",
            "provider": provider,
            "detail": str(exc),
            "preview": text[:500],
        }), 500

    if scan_id:
        stored_rewrites = save_rewrites(scan_id, rewrite_list, provider=provider)
        cache_rewrites(scan_id, {
            "scan_id": scan_id,
            "source_hook": hook_text,
            "rewrites": stored_rewrites,
            "provider": provider,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        })
        rewrite_list = stored_rewrites

    return jsonify({"rewrites": rewrite_list})


@app.route("/api/scan_rewrites", methods=["POST"])
def api_scan_rewrites():
    body = request.json or {}
    rewrites = body.get("rewrites", [])
    parent_id = body.get("scan_id")

    jobs_out = []
    for r in rewrites:
        job_id = _start_job(
            r["hook"],
            parent_scan_id=parent_id,
            mechanic=r.get("mechanic"),
            rewrite_id=r.get("id"),
        )
        jobs_out.append({
            "job_id": job_id,
            "rewrite_id": r.get("id"),
            "mechanic": r.get("mechanic", ""),
            "hook": r["hook"],
        })
    return jsonify({"jobs": jobs_out})


@app.route("/api/history")
def api_history():
    return jsonify({"history": get_history()})


@app.route("/api/history/<int:scan_id>")
def api_history_detail(scan_id):
    scan = get_cached_scan(scan_id) or get_scan(scan_id)
    if not scan:
        return jsonify({"error": "Not found"}), 404
    scan["rewrites"] = get_rewrites_for_scan(scan_id)
    return jsonify(scan)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    ensure_upload_dirs()
    host = os.environ.get("HOOKBRAIN_HOST", "0.0.0.0")
    port = int(os.environ.get("HOOKBRAIN_PORT", "5050"))
    app.run(host=host, port=port, debug=False, use_reloader=False)
