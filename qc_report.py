"""qc_report.py — Per-run data quality / quality-control reporting.

Produces a human-readable summary of how trustworthy a run's per-frame
sensor record is, addressing the fact that the pipeline interpolates sensor
values onto frame timestamps with *clamped extrapolation* (numpy.interp with
left/right bounds).  Clamping silently fabricates a boundary value for frames
that fall outside a sensor's coverage window, and linearly bridges multi-minute
dropouts as if data were continuous — neither is visible in the output CSV.

This module re-examines the source sensor timestamps to surface:
  • coverage window per channel,
  • frames falling outside that window (values were clamped),
  • gaps in the source series larger than a threshold, and how many frames
    landed inside them (values were bridged across the gap),
  • basic value statistics + a histogram per channel.

Rendering deliberately avoids matplotlib.pyplot (not thread-safe with Qt);
histograms are drawn by hand with PIL, consistent with point_cloud_pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw


# ---------------------------------------------------------------------------
# Computational core (no Qt, no file I/O beyond the PNG/txt writers)
# ---------------------------------------------------------------------------

def channel_stats(values: np.ndarray) -> dict:
    """Return summary statistics for one channel's per-frame values.

    NaN entries are treated as missing and excluded from the numeric stats but
    counted in n_missing.
    """
    values = np.asarray(values, dtype=float)
    n_total   = int(values.size)
    finite    = values[np.isfinite(values)]
    n_valid   = int(finite.size)
    n_missing = n_total - n_valid

    if n_valid == 0:
        return {
            "n_total": n_total, "n_valid": 0, "n_missing": n_missing,
            "min": None, "max": None, "mean": None, "median": None, "std": None,
        }
    return {
        "n_total":  n_total,
        "n_valid":  n_valid,
        "n_missing": n_missing,
        "min":    float(np.min(finite)),
        "max":    float(np.max(finite)),
        "mean":   float(np.mean(finite)),
        "median": float(np.median(finite)),
        "std":    float(np.std(finite)),
    }


def detect_gaps(source_times: np.ndarray, max_gap_s: float) -> list[tuple[float, float]]:
    """Return [(gap_start, gap_end), ...] for consecutive source samples whose
    spacing exceeds max_gap_s.  source_times need not be sorted.
    """
    t = np.asarray(source_times, dtype=float)
    t = np.sort(t[np.isfinite(t)])
    if t.size < 2 or max_gap_s <= 0:
        return []
    dt = np.diff(t)
    idx = np.where(dt > max_gap_s)[0]
    return [(float(t[i]), float(t[i + 1])) for i in idx]


def coverage_analysis(
    frame_times: np.ndarray,
    source_times: np.ndarray,
    max_gap_s: float,
) -> dict:
    """Quantify how many frame timestamps fall outside source coverage or
    inside source gaps.

    Args:
        frame_times:  per-frame unix times (the interp.csv time axis).
        source_times: source sensor sample unix times (channel present).
        max_gap_s:    spacing above which a source interval counts as a gap.

    Returns dict with coverage window, counts, and the gap list.
    """
    ft = np.asarray(frame_times, dtype=float)
    ft = ft[np.isfinite(ft)]
    st = np.sort(np.asarray(source_times, dtype=float))
    st = st[np.isfinite(st)]

    if st.size == 0:
        return {
            "src_start": None, "src_end": None,
            "n_frames": int(ft.size), "n_outside": int(ft.size),
            "n_in_gaps": 0, "gaps": [],
        }

    src_start, src_end = float(st[0]), float(st[-1])
    n_outside = int(np.count_nonzero((ft < src_start) | (ft > src_end)))

    gaps = detect_gaps(st, max_gap_s)
    n_in_gaps = 0
    for g0, g1 in gaps:
        n_in_gaps += int(np.count_nonzero((ft > g0) & (ft < g1)))

    return {
        "src_start": src_start,
        "src_end":   src_end,
        "n_frames":  int(ft.size),
        "n_outside": n_outside,
        "n_in_gaps": n_in_gaps,
        "gaps":      gaps,
    }


# ---------------------------------------------------------------------------
# PIL histogram (no pyplot)
# ---------------------------------------------------------------------------

def write_histogram_png(
    path: str,
    values: np.ndarray,
    *,
    bins: int = 40,
    title: str = "",
    width: int = 460,
    height: int = 260,
    bar_color: tuple = (70, 130, 180),
) -> bool:
    """Draw a value histogram to a PNG using PIL.  Returns False if there is
    no finite data to plot.
    """
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return False

    lo, hi = float(vals.min()), float(vals.max())
    if hi <= lo:
        hi = lo + 1.0  # avoid zero-width range for a constant channel

    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    cmax = int(counts.max()) or 1

    pad_l, pad_r, pad_t, pad_b = 48, 12, 26 if title else 10, 34
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    img  = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    # Axes.
    draw.line([(pad_l, pad_t), (pad_l, pad_t + plot_h)], fill=(0, 0, 0))
    draw.line([(pad_l, pad_t + plot_h), (pad_l + plot_w, pad_t + plot_h)], fill=(0, 0, 0))

    # Bars.
    n = len(counts)
    bar_w = plot_w / n
    for i, c in enumerate(counts):
        if c <= 0:
            continue
        bh = max(1, int(plot_h * c / cmax))  # keep tiny-but-nonzero bins visible
        x0 = pad_l + int(i * bar_w)
        x1 = pad_l + int((i + 1) * bar_w) - 1
        y0 = pad_t + plot_h - bh
        y1 = pad_t + plot_h - 1
        draw.rectangle([x0, y0, max(x1, x0), y1], fill=bar_color)

    # Labels.
    if title:
        draw.text((pad_l, 6), title, fill=(20, 20, 20))
    draw.text((pad_l, pad_t + plot_h + 6), f"{lo:.3g}", fill=(20, 20, 20))
    draw.text((pad_l + plot_w - 36, pad_t + plot_h + 6), f"{hi:.3g}", fill=(20, 20, 20))
    draw.text((4, pad_t - 2), f"{cmax}", fill=(20, 20, 20))
    draw.text((4, pad_t + plot_h - 8), "0", fill=(20, 20, 20))

    img.save(path)
    return True


# ---------------------------------------------------------------------------
# Report assembly
# ---------------------------------------------------------------------------

def _fmt_dt(unix_time: Optional[float]) -> str:
    if unix_time is None or not np.isfinite(unix_time):
        return "n/a"
    from datetime import datetime, timezone
    return datetime.fromtimestamp(unix_time, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def generate_report(
    interp_df: pd.DataFrame,
    channel_sources: dict[str, Optional[np.ndarray]],
    output_dir: str,
    *,
    max_gap_s: float = 60.0,
    log_fn: Optional[Callable[[str], None]] = None,
) -> list[str]:
    """Write qc_report.txt, qc_summary.json, and one histogram PNG per channel.

    Args:
        interp_df:       per-frame record; must contain "unix_time" plus one
                         column per channel in channel_sources.
        channel_sources: {channel_name: source_unix_times | None}.  When the
                         source times are provided, coverage/gap analysis is
                         included; when None, only value statistics are.
        output_dir:      directory to write the report files into.
        max_gap_s:       source spacing above which a span counts as a gap.
        log_fn:          optional progress sink.

    Returns:
        List of written file paths (report first, then histograms).
    """
    log = log_fn or (lambda _m: None)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    frame_times = (
        interp_df["unix_time"].to_numpy(dtype=float)
        if "unix_time" in interp_df.columns
        else np.array([])
    )
    n_frames = int(frame_times.size)

    lines: list[str] = []
    lines.append("=" * 64)
    lines.append("DATA QUALITY / QC REPORT")
    lines.append("=" * 64)
    lines.append(f"Frames in record:      {n_frames}")
    if n_frames:
        lines.append(f"Frame time span:       {_fmt_dt(np.nanmin(frame_times))} "
                     f"→ {_fmt_dt(np.nanmax(frame_times))} (UTC)")
    lines.append(f"Gap threshold:         {max_gap_s:g} s")
    lines.append("")

    summary: dict = {
        "n_frames": n_frames,
        "max_gap_s": max_gap_s,
        "channels": {},
    }
    written: list[str] = []

    for channel, src_times in channel_sources.items():
        if channel not in interp_df.columns:
            continue
        values = interp_df[channel].to_numpy(dtype=float)
        stats = channel_stats(values)

        lines.append("-" * 64)
        lines.append(f"CHANNEL: {channel}")
        lines.append("-" * 64)
        if stats["n_valid"] == 0:
            lines.append("  No valid values in record.")
            summary["channels"][channel] = {"stats": stats, "coverage": None}
            lines.append("")
            continue

        miss_pct = 100.0 * stats["n_missing"] / max(stats["n_total"], 1)
        lines.append(f"  Values: {stats['n_valid']} valid, "
                     f"{stats['n_missing']} missing ({miss_pct:.1f}%)")
        lines.append(f"  Range:  min={stats['min']:.4g}  max={stats['max']:.4g}  "
                     f"mean={stats['mean']:.4g}  median={stats['median']:.4g}  "
                     f"std={stats['std']:.4g}")

        cov = None
        if src_times is not None and len(src_times) > 0:
            cov = coverage_analysis(frame_times, src_times, max_gap_s)
            cov_pct = 100.0 * (cov["n_frames"] - cov["n_outside"]) / max(cov["n_frames"], 1)
            lines.append(f"  Source coverage: {_fmt_dt(cov['src_start'])} "
                         f"→ {_fmt_dt(cov['src_end'])} (UTC)")
            lines.append(f"  Frames within coverage: {cov_pct:.1f}%  "
                         f"({cov['n_outside']} frame(s) outside → clamped value)")
            if cov["gaps"]:
                lines.append(f"  Source gaps > {max_gap_s:g}s: {len(cov['gaps'])}  "
                             f"({cov['n_in_gaps']} frame(s) inside gaps → bridged value)")
                for g0, g1 in cov["gaps"][:10]:
                    lines.append(f"     gap {_fmt_dt(g0)} → {_fmt_dt(g1)}  "
                                 f"({g1 - g0:.0f} s)")
                if len(cov["gaps"]) > 10:
                    lines.append(f"     … and {len(cov['gaps']) - 10} more")
            else:
                lines.append(f"  Source gaps > {max_gap_s:g}s: none")
        else:
            lines.append("  Source coverage: (source timestamps unavailable)")

        # Histogram PNG.
        safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in channel)
        hist_path = out / f"{safe}_hist.png"
        if write_histogram_png(str(hist_path), values, title=f"{channel} distribution"):
            written.append(str(hist_path))
            log(f"  histogram: {hist_path.name}")

        summary["channels"][channel] = {"stats": stats, "coverage": cov}
        lines.append("")

    report_path = out / "qc_report.txt"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    written.insert(0, str(report_path))

    json_path = out / "qc_summary.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    written.insert(1, str(json_path))

    log(f"QC report: {report_path}")
    return written
