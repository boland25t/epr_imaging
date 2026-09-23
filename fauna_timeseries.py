#!/usr/bin/env python3
"""Per-frame megafauna areal density and its time series.

Owns the footprint model and the density computation (previously ad-hoc —
critique_fauna.md MOD-6): appends area/density columns to
survey/fauna/fauna_density.csv and regenerates the time-series figure from
this committed code.

Footprint model: a 5312x2988 frame at ~1 mm/px near 5 m altitude spans
roughly 5.3 x 3.0 m, i.e. width ~ K_W * alt with K_W ~= 1.06 and aspect
2988/5312, giving area ~= AREA_K * alt^2.  MOD-1 of the critique measured
K_W ~= 0.86-0.88 from the chunk orthomosaics themselves; both constants are
exposed here so the choice is explicit and revisable.

Qt-free.  build(workspace_dir) -> figure path.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_W = 1.06           # frame ground width / altitude (nominal; ortho-measured ~0.87)
ASPECT = 2988 / 5312
AREA_K = K_W * K_W * ASPECT     # ~0.63 * alt^2 m^2 per frame
GAP_S = 120          # inter-frame gap that breaks an imaged span
ROLL_N = 12          # rolling-median window (frames, ~1 min at 5 s cadence)
YCAP_Q = 0.99        # display cap quantile (disclosed on the figure)


def add_density_columns(ws: str) -> pd.DataFrame:
    """Append area_m2 + per-bucket dens_* columns to fauna_density.csv."""
    path = Path(ws) / "survey" / "fauna" / "fauna_density.csv"
    d = pd.read_csv(path)
    d["area_m2"] = (AREA_K * d.alt ** 2).round(3)
    for b in ("crustacean", "worm", "fish", "anemone", "unknown", "total"):
        d[f"dens_{b}"] = (d[f"n_{b}"] / d.area_m2).round(4)
    d.to_csv(path, index=False)
    return d


def build(ws: str, log=print) -> str:
    d = add_density_columns(ws).dropna(subset=["unix_time", "alt"])
    d = d.sort_values("unix_time").reset_index(drop=True)
    t0 = d.unix_time.min()
    d["hrs"] = (d.unix_time - t0) / 3600
    gap = d.unix_time.diff().fillna(0) > GAP_S
    spans = d.groupby(gap.cumsum()).unix_time.agg(["min", "max"])

    def broken(series):
        r = series.rolling(ROLL_N, center=True, min_periods=3).median().copy()
        r[gap] = np.nan
        return r

    ycap = max(d.dens_total.quantile(YCAP_Q), 0.5)
    win = pd.read_csv(Path(ws) / "survey" / "anomaly" / "window_context.csv")
    # the full window inventory is deliberately high-recall and blankets the
    # dive; shade only HIGH-tier transit windows so shading stays legible
    win = win[(win.confidence_tier == "HIGH") & (win.context == "transit")]
    wt0 = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    wt1 = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()

    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=150)
    fig.patch.set_facecolor("#0e1620")
    ax.set_facecolor("#0e1620")
    for a, b in zip(wt0, wt1):                # shade only over imaged spans
        for _, s in spans.iterrows():
            lo, hi = max(a, s["min"]), min(b, s["max"])
            if hi > lo:
                ax.axvspan((lo - t0) / 3600, (hi - t0) / 3600,
                           color="#ff5d6e", alpha=0.22, lw=0)
    prev = None
    for _, s in spans.iterrows():             # grey no-imagery blocks
        if prev is not None and s["min"] - prev > GAP_S:
            ax.axvspan((prev - t0) / 3600, (s["min"] - t0) / 3600,
                       color="#1c2833", alpha=0.9, lw=0, zorder=0.5)
        prev = s["max"]
    ax.scatter(d.hrs, d.dens_total.clip(upper=ycap), s=3, color="#3d5666",
               alpha=0.55, lw=0)
    ax.plot(d.hrs, broken(d.dens_total), color="#6ec6e6", lw=1.7,
            label="all fauna (rolling median)")
    ax.plot(d.hrs, broken(d.dens_worm), color="#e05c62", lw=1.5,
            label="worms (Riftia-dominated)")
    imaged_h = (spans["max"] - spans["min"]).sum() / 3600
    ax.set_xlim(d.hrs.min() - 0.1, d.hrs.max() + 0.1)
    ax.set_ylim(0, ycap * 1.04)
    ax.set_xlabel("hours (grey = no imagery; lines break across gaps)",
                  color="#93a3af", fontsize=10)
    ax.set_ylabel("animals / m²", color="#93a3af", fontsize=10)
    dive = Path(ws).name.split("_")[0]
    ax.set_title(f"{dive} — per-frame megafauna density ({imaged_h:.1f} h imaged; "
                 f"red = HIGH-tier transit windows over imaged spans; "
                 f"footprint {AREA_K:.2f}·alt²; y cap p99 = {ycap:.2f}/m²)",
                 color="#dce4eb", fontsize=11)
    ax.tick_params(colors="#6b7b87", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#263442")
    ax.legend(fontsize=8.5, framealpha=0.9, facecolor="#16222e",
              edgecolor="#2a3846", labelcolor="#dbe4ec")
    fig.tight_layout()
    out = Path(ws) / "survey" / "fauna" / "fauna_density_timeseries_v2.png"
    fig.savefig(out, dpi=150, facecolor="#0e1620", bbox_inches="tight")
    plt.close(fig)
    log(f"{dive}: {imaged_h:.1f} h imaged -> {out}")
    return str(out)


if __name__ == "__main__":
    for dv in (sys.argv[1:] or ["J1754", "J1756"]):
        build(f"/mnt/f/EPR_2026_PROCESSED/{dv}_down.eprproj")
