#!/usr/bin/env python3
"""Cross-dive anomaly overview map: every dive's trackline with tier-coloured
anomaly windows and ranked sites overlaid in one UTM frame.

Station-context windows (deliberate on-station measurements, from
window_context.csv) can be de-emphasised so the map shows the *detection*
population — anomalies encountered while genuinely surveying.

build_map(dives, out_path, transit_only=False) -> path

Qt-free.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
TIER_COL = {"HIGH": "#d7263d", "MODERATE": "#e8871e", "SCREEN": "#e0a800"}
DIVE_COL = {"J1754": "#6ec6e6", "J1755": "#8fd18a", "J1756": "#e6a3e0",
            "J1757": "#c9c9c9", "J1758": "#f2c94c", "J1759": "#9fb0ff",
            "J1760": "#ffb08a", "J1761": "#7fe0d0"}


def _load(dive):
    ws = ROOT / f"{dive}_down.eprproj"
    trk = np.array(json.load(open(ws / "survey/nav_trackline/trackline.geojson"))
                   ["features"][0]["geometry"]["coordinates"])
    seg = json.load(open(ws / "survey/anomaly/anomaly_segments_utm.geojson"))
    sites = json.load(open(ws / "survey/anomaly/anomalous_sites_utm.geojson"))
    ctx_path = ws / "survey/anomaly/window_context.csv"
    ctx = pd.read_csv(ctx_path) if ctx_path.is_file() else None
    return trk, seg, sites, ctx


def build_map(dives, out_path=None, transit_only=False, log=print) -> str:
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(12.5, 13), dpi=150)
    fig.patch.set_facecolor("#0e1620")
    ax.set_facecolor("#0e1620")
    n_win = 0
    for dive in dives:
        try:
            trk, seg, sites, ctx = _load(dive)
        except FileNotFoundError as e:
            log(f"{dive}: skipped ({e})")
            continue
        ax.plot(trk[:, 0], trk[:, 1], color=DIVE_COL.get(dive, "#9fb3c8"),
                lw=0.7, alpha=0.75, zorder=2)
        station_ids = set()
        if transit_only and ctx is not None:
            station_ids = set(ctx.loc[ctx.context == "station", "window_id"])
        for i, tier in enumerate(("SCREEN", "MODERATE", "HIGH")):
            s = [np.array(f["geometry"]["coordinates"]) for f in seg["features"]
                 if f["properties"].get("confidence") == tier
                 and f["properties"].get("window_id") not in station_ids]
            if s:
                ax.add_collection(LineCollection(
                    s, colors=TIER_COL[tier], linewidths=2.2, zorder=3 + i))
                n_win += len(s)
        for f in sites["features"]:
            x, y = f["geometry"]["coordinates"]
            ax.plot(x, y, marker="o", ms=5, mfc="none",
                    mec="#ffffff", mew=0.8, alpha=0.8, zorder=7)
    ax.set_aspect("equal")
    ax.autoscale()
    ax.tick_params(colors="#6b7d8f", labelsize=7)
    for sp in ax.spines.values():
        sp.set_color("#2a3846")
    xl, yl = ax.get_xlim(), ax.get_ylim()
    x0, y0 = xl[0] + (xl[1] - xl[0]) * 0.04, yl[0] + (yl[1] - yl[0]) * 0.03
    ax.plot([x0, x0 + 500], [y0, y0], color="w", lw=3)
    ax.text(x0 + 250, y0 + (yl[1] - yl[0]) * 0.008, "500 m", color="w",
            ha="center", fontsize=9)
    leg = ([Line2D([0], [0], color=DIVE_COL.get(d, "#9fb3c8"), lw=2, label=d)
            for d in dives] +
           [Line2D([0], [0], color=TIER_COL[t], lw=3, label=f"{t.title()} window")
            for t in ("HIGH", "MODERATE", "SCREEN")] +
           [Line2D([0], [0], marker="o", ls="", mfc="none", mec="#fff", ms=6,
                   label="ranked site")])
    ax.legend(handles=leg, loc="upper right", fontsize=8, framealpha=0.92,
              facecolor="#16222e", edgecolor="#2a3846", labelcolor="#dbe4ec",
              ncol=2)
    sub = " — transit (survey) detections only" if transit_only else ""
    ax.set_title(f"EPR 9°N — gas anomaly windows across {len(dives)} Jason dives{sub}",
                 color="#e8eef3", fontsize=13, pad=10)
    out = Path(out_path or ROOT / ("cross_dive_anomalies_transit.png" if transit_only
                                   else "cross_dive_anomalies.png"))
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0e1620")
    plt.close(fig)
    log(f"wrote {out} ({n_win} windows)")
    return str(out)


if __name__ == "__main__":
    dives = sys.argv[1:] or ["J1754", "J1755", "J1756", "J1758", "J1759", "J1760", "J1761"]
    build_map(dives)
    build_map(dives, transit_only=True)
