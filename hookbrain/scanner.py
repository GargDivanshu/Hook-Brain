"""
TRIBE v2 wrapper.
Import for compute_metrics() / compute_viral_score().
Run as a script:
  python scanner.py --mode text --input "hook text" --output output.json
  python scanner.py --mode video --input /path/to/video.mp4 --output output.json
"""

import argparse
import json
import os
import sys
import tempfile

import numpy as np

N_TOP = 100


def compute_metrics(preds):
    """Raw TRIBE predictions (T x 20484) -> list of per-second dicts."""
    n_hemi = preds.shape[1] // 2
    seconds = []
    for i in range(preds.shape[0]):
        row = preds[i]
        top_idx = np.argsort(row)[-N_TOP:]
        seconds.append({
            "t": i,
            "mean": round(float(row.mean()), 4),
            "max": round(float(row.max()), 4),
            "min": round(float(row.min()), 4),
            "left_mean": round(float(row[:n_hemi].mean()), 4),
            "right_mean": round(float(row[n_hemi:].mean()), 4),
            "top100_mean": round(float(np.sort(row)[-N_TOP:].mean()), 4),
            "top100_left_pct": int(sum(1 for idx in top_idx if idx < n_hemi)),
        })
    return seconds


def compute_viral_score(seconds):
    """Viral score breakdown from per-second brain data."""

    def get(t, key):
        return seconds[t].get(key, 0) if t < len(seconds) else 0

    watch_signal = get(0, "mean") + get(1, "mean")
    emotional_onset = get(0, "top100_mean")
    right_dom_t0 = 100 - get(0, "top100_left_pct")
    dropoff_risk = get(3, "top100_left_pct")
    mean_sustained = sum(s["mean"] for s in seconds) / len(seconds) if seconds else 0

    viral = (
        watch_signal * 4.0
        + emotional_onset * 2.0
        + right_dom_t0 * 0.02
        - dropoff_risk * 0.04
        + mean_sustained * 3.0
    )
    return {
        "watch_signal": round(watch_signal, 4),
        "emotional_onset": round(emotional_onset, 4),
        "right_dom_t0": round(right_dom_t0, 1),
        "dropoff_risk": round(dropoff_risk, 1),
        "mean_sustained": round(mean_sustained, 4),
        "viral_score": round(viral, 3),
    }


def scan_input(input_mode, input_value):
    from tribev2 import TribeModel

    model = TribeModel.from_pretrained("facebook/tribev2", cache_folder="./cache")

    if input_mode == "text":
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(input_value)
            tmp_path = f.name
        try:
            df = model.get_events_dataframe(text_path=tmp_path)
        finally:
            os.unlink(tmp_path)
        result_hook = input_value
        metadata = {"input_mode": "text"}
    elif input_mode == "video":
        df = model.get_events_dataframe(video_path=input_value)
        result_hook = f"[VIDEO] {os.path.basename(input_value)}"
        metadata = {
            "input_mode": "video",
            "source_path": input_value,
            "source_name": os.path.basename(input_value),
        }
    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    preds, segments = model.predict(events=df)
    seconds = compute_metrics(preds)
    viral = compute_viral_score(seconds)
    return {
        "hook": result_hook,
        "seconds": seconds,
        "viral": viral,
        "metadata": metadata,
    }


def parse_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["text", "video"], default="text")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args(argv)


if __name__ == "__main__":
    args = parse_args(sys.argv[1:])
    result = scan_input(args.mode, args.input)
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved to {args.output}", flush=True)
