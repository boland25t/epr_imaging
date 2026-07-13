"""
frame_stats.py — Per-frame statistical analysis of extracted video frames.

Combines two things per frame:
  • Image-quality metrics (from the image itself): sharpness (variance of the
    Laplacian), brightness, contrast, clipped-highlight/shadow fraction, and
    mean R/G/B — to QC frames before photogrammetry and flag blurry / dark /
    over-exposed ones.
  • Sensor + navigation values attached to that frame (joined from the
    sampling set's interp.csv by frame_filename).

Outputs (written into a frame_stats/run_NNN directory):
  frame_stats.csv          — one row per frame: image metrics + sensor/nav cols
  frame_stats_summary.json — aggregate stats per metric/channel + flag counts
  hist_<metric>.png        — histograms for image metrics and sensor channels
  flagged_frames.txt       — frames failing the quality thresholds

Reuses qc_report.channel_stats and qc_report.write_histogram_png so the report
style matches the Data QC Report.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Callable, Optional

import numpy as np

import qc_report  # channel_stats, write_histogram_png

_IMG_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_IMG_METRICS = ["sharpness", "brightness", "contrast",
                "pct_clipped_high", "pct_clipped_low"]


def image_quality(path: str) -> dict:
    """Compute image-quality metrics for one frame.  {} if unreadable."""
    import cv2
    img = cv2.imread(path)
    if img is None:
        return {}
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total = gray.size or 1
    b, g, r = (float(img[:, :, i].mean()) for i in range(3))
    return {
        "sharpness":        float(cv2.Laplacian(gray, cv2.CV_64F).var()),
        "brightness":       float(gray.mean()),
        "contrast":         float(gray.std()),
        "pct_clipped_high": float((gray >= 250).sum()) / total * 100.0,
        "pct_clipped_low":  float((gray <= 5).sum()) / total * 100.0,
        "mean_r": r, "mean_g": g, "mean_b": b,
    }


def _safe(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]", "_", str(name))


def analyze_frame_set(
    frame_dir,
    run_dir,
    interp_csv: Optional[str] = None,
    *,
    sharpness_min: float = 100.0,
    brightness_min: float = 20.0,
    brightness_max: float = 235.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Analyse every frame in frame_dir; write the report into run_dir.

    interp_csv (the sampling set's interp.csv) supplies per-frame sensor/nav
    values, joined by frame_filename.  Returns the list of written file paths.
    """
    import pandas as pd

    log = log_fn or (lambda m: None)
    frame_dir = Path(frame_dir)
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    frames = sorted(f for f in frame_dir.iterdir() if f.suffix.lower() in _IMG_EXTS)
    if not frames:
        raise FileNotFoundError(f"No images found in {frame_dir}")
    log(f"Frame statistics: {len(frames)} frame(s) in {frame_dir}")

    rows = []
    for i, f in enumerate(frames, start=1):
        q = image_quality(str(f))
        q["frame_filename"] = f.name
        rows.append(q)
        if i % 50 == 0 or i == len(frames):
            log(f"      analysed {i}/{len(frames)} frames")
    df = pd.DataFrame(rows)

    # ── Join per-frame sensor/nav values from interp.csv ─────────────────────
    sensor_cols: list[str] = []
    if interp_csv and Path(interp_csv).exists():
        interp = pd.read_csv(interp_csv)
        if "frame_filename" in interp.columns:
            df = df.merge(interp, on="frame_filename", how="left")
            exclude = {"unix_time", "frame_index", "utm_zone",
                       "timestamp_iso", "video_filename"}
            sensor_cols = [
                c for c in interp.columns
                if c != "frame_filename" and c not in exclude
                and c in df.columns and pd.api.types.is_numeric_dtype(df[c])
            ]
            log(f"      joined sensor/nav values ({len(sensor_cols)} numeric columns)")
        else:
            log("      interp.csv has no frame_filename column — sensor stats skipped")

    # ── Quality flags ────────────────────────────────────────────────────────
    df["flag_blurry"] = df["sharpness"]  < sharpness_min
    df["flag_dark"]   = df["brightness"] < brightness_min
    df["flag_bright"] = df["brightness"] > brightness_max
    df["flagged"]     = df[["flag_blurry", "flag_dark", "flag_bright"]].any(axis=1)

    products: list[str] = []

    csv_path = run_dir / "frame_stats.csv"
    df.to_csv(csv_path, index=False)
    products.append(str(csv_path))
    log(f"      per-frame CSV: {csv_path}")

    # ── Summary JSON ─────────────────────────────────────────────────────────
    summary = {
        "n_frames":  int(len(df)),
        "n_flagged": int(df["flagged"].sum()),
        "n_blurry":  int(df["flag_blurry"].sum()),
        "n_dark":    int(df["flag_dark"].sum()),
        "n_bright":  int(df["flag_bright"].sum()),
        "thresholds": {
            "sharpness_min":  sharpness_min,
            "brightness_min": brightness_min,
            "brightness_max": brightness_max,
        },
        "image_quality": {
            m: qc_report.channel_stats(df[m].to_numpy())
            for m in _IMG_METRICS if m in df.columns
        },
        "sensor": {
            c: qc_report.channel_stats(df[c].to_numpy()) for c in sensor_cols
        },
    }
    json_path = run_dir / "frame_stats_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    products.append(str(json_path))

    # ── Histograms ───────────────────────────────────────────────────────────
    for m in ("sharpness", "brightness", "contrast"):
        if m in df.columns:
            p = run_dir / f"hist_{m}.png"
            if qc_report.write_histogram_png(str(p), df[m].to_numpy(), title=m):
                products.append(str(p))
    for c in sensor_cols:
        p = run_dir / f"hist_{_safe(c)}.png"
        if qc_report.write_histogram_png(str(p), df[c].to_numpy(), title=c):
            products.append(str(p))

    # ── Flagged-frames list ──────────────────────────────────────────────────
    flagged = df.loc[df["flagged"], "frame_filename"].tolist()
    fl_path = run_dir / "flagged_frames.txt"
    fl_path.write_text("\n".join(flagged) + ("\n" if flagged else ""), encoding="utf-8")
    products.append(str(fl_path))

    log(f"Frame statistics complete: {summary['n_flagged']}/{summary['n_frames']} flagged "
        f"({summary['n_blurry']} blurry, {summary['n_dark']} dark, "
        f"{summary['n_bright']} over-bright). Report in {run_dir}")
    return products
