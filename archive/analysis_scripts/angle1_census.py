#!/usr/bin/env python3
"""Statistical hardening of the un-targeted detection census (methods-paper
Angle 1): how robust is the transit anomaly-encounter rate?

The census counts anomaly windows encountered while the vehicle was genuinely
surveying (transit, median smoothed speed >= V_STATION) per kilometre of
transit track.  Three hardening exercises across the seven dives:

  1. Threshold sensitivity — recompute transit windows, transit km and rate
     for a sweep of station/transit speed thresholds (windows re-split from
     the median_speed already in window_context.csv; km re-masked from the
     smoothed 1 Hz speed).  A window exactly at a threshold goes to transit
     (median_speed >= threshold), mirroring window_context's station rule
     (median < threshold).
  2. Tier-filtered rates at the canonical threshold — all windows,
     HIGH+MODERATE, HIGH only.  Rates count windows once each regardless of
     how many channel families they span.
  3. Uncertainty — per-dive block bootstrap over contiguous transit segments
     (runs of smoothed speed >= V_STATION lasting >= SEG_MIN_S; each segment
     carries its km and the windows whose midpoints fall in it), 1000
     resamples of the segment set; plus an exact (Garwood chi-square) Poisson
     95% CI for comparison, and a fleet-pooled bootstrap over all segments.

Outputs under /mnt/f/EPR_2026_PROCESSED/paper/:
  angle1_threshold_sweep.csv, angle1_tier_rates.csv, angle1_rate_ci.csv,
  angle1_threshold_fig.png, angle1_forest_fig.png, angle1_summary.json

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
from matplotlib.lines import Line2D
from scipy.stats import chi2

from window_context import ROOT, V_STATION, _speed_series
from multi_dive_map import DIVE_COL

DIVES = ["J1754", "J1755", "J1756", "J1758", "J1759", "J1760", "J1761"]
THRESHOLDS = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.15]
PAPER = ROOT / "paper"
SEG_MIN_S = 60.0          # s — minimum contiguous transit run for bootstrap blocks
MAX_STEP_DT = 5.0         # s — nav steps across larger gaps carry no distance
N_BOOT = 1000
SEED = 42

BG, PANEL, FG, MUTED, GRID = "#0e1620", "#16222e", "#dbe4ec", "#6b7d8f", "#2a3846"
TIER_MARK = {"all": "o", "high_mod": "s", "high": "^"}


# ---------------------------------------------------------------- loading

def _load_dive(dive):
    """Read one dive's windows + 1 Hz nav once; everything else derives."""
    ws = ROOT / f"{dive}_down.eprproj"
    win = pd.read_csv(ws / "survey" / "anomaly" / "window_context.csv")
    t, sm = _speed_series(ws)
    ip = pd.read_csv(ws / "inputs" / "interp_full.csv",
                     usecols=["unix_time", "easting", "northing"]).dropna()
    ip = ip.sort_values("unix_time")
    ti = ip.unix_time.to_numpy()
    if len(ti) != len(t) or not np.allclose(ti, t):
        raise RuntimeError(f"{dive}: nav re-read does not align with _speed_series")
    dt = np.diff(t)
    step = np.hypot(np.diff(ip.easting.to_numpy()), np.diff(ip.northing.to_numpy()))
    step[(dt <= 0) | (dt > MAX_STEP_DT)] = 0.0     # no distance across gaps
    t0 = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    t1 = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    return dict(dive=dive, win=win, t=t, sm=sm, step=step,
                med=win.median_speed.to_numpy(),
                tier=win.confidence_tier.astype(str).to_numpy(),
                mid=(t0 + t1) / 2.0)


def _transit_km(d, v):
    """km of track whose smoothed speed is >= v (step i gated by sm[i])."""
    return float(d["step"][d["sm"][1:] >= v].sum()) / 1000.0


def _segments(d, v=V_STATION, min_s=SEG_MIN_S):
    """Contiguous transit runs >= min_s: list of (km, n_windows)."""
    mask = d["sm"] >= v
    edges = np.flatnonzero(np.diff(mask.astype(np.int8)))
    starts = np.r_[0, edges + 1]
    ends = np.r_[edges, len(mask) - 1]
    segs = []
    for a, b in zip(starts, ends):
        if not mask[a] or d["t"][b] - d["t"][a] < min_s:
            continue
        km = float(d["step"][a:b].sum()) / 1000.0   # steps a+1..b → step[a:b]
        n = int(((d["mid"] >= d["t"][a]) & (d["mid"] <= d["t"][b])).sum())
        segs.append((km, n))
    return segs


# ------------------------------------------------------------- statistics

def _bootstrap(segs, n_boot=N_BOOT, rng=None):
    """Resample segments with replacement; rate* = Σwindows*/Σkm*."""
    rng = rng or np.random.default_rng(SEED)
    km = np.array([s[0] for s in segs])
    n = np.array([s[1] for s in segs], dtype=float)
    idx = rng.integers(0, len(segs), size=(n_boot, len(segs)))
    tot_km = km[idx].sum(axis=1)
    tot_n = n[idx].sum(axis=1)
    rates = np.divide(tot_n, tot_km, out=np.zeros(n_boot), where=tot_km > 0)
    lo, med, hi = np.percentile(rates, [2.5, 50.0, 97.5])
    return float(lo), float(med), float(hi)


def _poisson_ci(k, km):
    """Exact (Garwood) 95% CI for a Poisson rate of k events over km."""
    lo = chi2.ppf(0.025, 2 * k) / 2.0 / km if k > 0 else 0.0
    hi = chi2.ppf(0.975, 2 * (k + 1)) / 2.0 / km
    return float(lo), float(hi)


# ---------------------------------------------------------------- figures

def _style_ax(ax):
    ax.set_facecolor(BG)
    ax.tick_params(colors=MUTED, labelsize=8)
    ax.grid(True, color=GRID, lw=0.5, alpha=0.6)
    for sp in ax.spines.values():
        sp.set_color(GRID)
    ax.xaxis.label.set_color(FG)
    ax.yaxis.label.set_color(FG)


def _fig_threshold(sweep, out):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    for dive, g in sweep.groupby("dive"):
        g = g.sort_values("v_station")
        ax.plot(g.v_station, g.rate, marker="o", ms=4, lw=1.4,
                color=DIVE_COL.get(dive, "#9fb3c8"), label=dive)
    ax.axvline(V_STATION, color=FG, lw=0.9, ls="--", alpha=0.55)
    ax.text(V_STATION, ax.get_ylim()[1], " canonical 0.08 m/s",
            color=MUTED, fontsize=7.5, va="top")
    ax.set_xlabel("station/transit speed threshold (m/s)")
    ax.set_ylabel("transit anomaly windows per km")
    ax.set_title("Detection-census rate vs. speed threshold",
                 color="#e8eef3", fontsize=12, pad=10)
    ax.legend(fontsize=8, framealpha=0.92, facecolor=PANEL,
              edgecolor=GRID, labelcolor=FG, ncol=2)
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


def _fig_forest(ci, tiers, fleet, out):
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.5, 5), dpi=150)
    fig.patch.set_facecolor(BG)
    _style_ax(ax)
    ax.grid(True, axis="x", color=GRID, lw=0.5, alpha=0.6)
    ax.grid(False, axis="y")
    dives = list(ci.dive)
    ys = np.arange(len(dives))[::-1]
    ax.axvspan(fleet["boot_lo"], fleet["boot_hi"], color="#0e7c86", alpha=0.14)
    ax.axvline(fleet["rate"], color="#0e7c86", lw=1.6,
               label=f"fleet pooled {fleet['rate']:.1f}/km")
    for y, (_, r) in zip(ys, ci.iterrows()):
        col = DIVE_COL.get(r.dive, "#9fb3c8")
        ax.plot([r.boot_lo, r.boot_hi], [y, y], color=col, lw=2.2,
                solid_capstyle="butt")
        ax.plot([r.pois_lo, r.pois_hi], [y - 0.22, y - 0.22], color=col,
                lw=0.9, alpha=0.55)
        ax.plot(r.rate, y, "o", ms=7, color=col, mec=BG, mew=0.8, zorder=5)
        tr = tiers[tiers.dive == r.dive].iloc[0]
        ax.plot(tr.rate_high_mod, y + 0.22, TIER_MARK["high_mod"], ms=4.5,
                color=col, alpha=0.55, zorder=4)
        ax.plot(tr.rate_high, y + 0.22, TIER_MARK["high"], ms=4.5,
                color=col, alpha=0.35, zorder=4)
    ax.set_yticks(ys, dives)
    ax.tick_params(axis="y", colors=FG)
    ax.set_xlim(left=0)
    ax.set_xlabel("transit anomaly windows per km (v_station = 0.08 m/s)")
    ax.set_title("Per-dive detection rate — block-bootstrap 95% CI",
                 color="#e8eef3", fontsize=12, pad=10)
    extra = [Line2D([0], [0], marker="o", ls="", color=FG, ms=7, mec=BG,
                    label="all tiers (thick bar: bootstrap; thin: Poisson)"),
             Line2D([0], [0], marker="s", ls="", color=FG, alpha=0.55, ms=5,
                    label="HIGH+MODERATE"),
             Line2D([0], [0], marker="^", ls="", color=FG, alpha=0.35, ms=5,
                    label="HIGH only")]
    h, _l = ax.get_legend_handles_labels()
    ax.legend(handles=h + extra, fontsize=7.5, framealpha=0.92,
              facecolor=PANEL, edgecolor=GRID, labelcolor=FG, loc="lower right")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig)


# -------------------------------------------------------------------- run

def run(dives=DIVES, log=print):
    PAPER.mkdir(parents=True, exist_ok=True)
    data = []
    for dive in dives:
        try:
            data.append(_load_dive(dive))
        except FileNotFoundError as e:
            log(f"{dive}: skipped ({e})")
    rng = np.random.default_rng(SEED)

    # 1 — threshold sweep -------------------------------------------------
    sweep_rows = []
    for d in data:
        for v in THRESHOLDS:
            n = int((d["med"] >= v).sum())
            km = _transit_km(d, v)
            sweep_rows.append(dict(dive=d["dive"], v_station=v,
                                   transit_windows=n, transit_km=round(km, 3),
                                   rate=round(n / km, 3) if km > 0 else np.nan))
    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(PAPER / "angle1_threshold_sweep.csv", index=False)

    # 2 — tier-filtered rates at the canonical threshold ------------------
    tier_rows = []
    for d in data:
        km = _transit_km(d, V_STATION)
        tmask = d["med"] >= V_STATION
        n_all = int(tmask.sum())
        n_hm = int((tmask & np.isin(d["tier"], ["HIGH", "MODERATE"])).sum())
        n_hi = int((tmask & (d["tier"] == "HIGH")).sum())
        tier_rows.append(dict(
            dive=d["dive"], transit_km=round(km, 3),
            windows_all=n_all, rate_all=round(n_all / km, 3),
            windows_high_mod=n_hm, rate_high_mod=round(n_hm / km, 3),
            windows_high=n_hi, rate_high=round(n_hi / km, 3)))
    tiers = pd.DataFrame(tier_rows)
    tiers.to_csv(PAPER / "angle1_tier_rates.csv", index=False)

    # 3 — block bootstrap + Poisson CI ------------------------------------
    ci_rows, all_segs = [], []
    for d in data:
        segs = _segments(d)
        all_segs.extend(segs)
        km = sum(s[0] for s in segs)
        n = sum(s[1] for s in segs)
        rate = n / km if km > 0 else np.nan
        blo, bmed, bhi = _bootstrap(segs, rng=rng)
        plo, phi = _poisson_ci(n, km)
        ci_rows.append(dict(dive=d["dive"], n_segments=len(segs),
                            transit_km=round(km, 3), windows=n,
                            rate=round(rate, 3),
                            boot_lo=round(blo, 3), boot_med=round(bmed, 3),
                            boot_hi=round(bhi, 3),
                            pois_lo=round(plo, 3), pois_hi=round(phi, 3)))
    ci = pd.DataFrame(ci_rows)
    ci.to_csv(PAPER / "angle1_rate_ci.csv", index=False)

    # fleet pooled --------------------------------------------------------
    pool_km = sum(s[0] for s in all_segs)
    pool_n = sum(s[1] for s in all_segs)
    flo, fmed, fhi = _bootstrap(all_segs, rng=rng)
    fleet = dict(n_segments=len(all_segs), transit_km=round(pool_km, 3),
                 windows=pool_n, rate=round(pool_n / pool_km, 3),
                 boot_lo=round(flo, 3), boot_med=round(fmed, 3),
                 boot_hi=round(fhi, 3))

    # threshold-sensitivity verdict (fleet rate = Σwindows / Σkm per v) ---
    fleet_by_v = {f"{v:.2f}": round(float(sweep[sweep.v_station == v].transit_windows.sum())
                                    / float(sweep[sweep.v_station == v].transit_km.sum()), 3)
                  for v in THRESHOLDS}
    r0 = fleet_by_v[f"{V_STATION:.2f}"]
    rel = {k: (v - r0) / r0 for k, v in fleet_by_v.items()}
    max_rel = max(rel.values(), key=abs)
    verdict = (f"fleet transit rate changes at most "
               f"{100 * abs(max_rel):.1f}% relative to the canonical "
               f"{V_STATION} m/s across thresholds {THRESHOLDS[0]}-{THRESHOLDS[-1]} m/s")

    # tier shares of the fleet transit population at the canonical v -----
    tot = int(tiers.windows_all.sum())
    hm, hi = int(tiers.windows_high_mod.sum()), int(tiers.windows_high.sum())
    tier_shares = dict(HIGH=round(hi / tot, 3),
                       MODERATE=round((hm - hi) / tot, 3),
                       SCREEN=round((tot - hm) / tot, 3))

    # figures -------------------------------------------------------------
    _fig_threshold(sweep, PAPER / "angle1_threshold_fig.png")
    _fig_forest(ci, tiers, fleet, PAPER / "angle1_forest_fig.png")

    # summary JSON ---------------------------------------------------------
    per_dive = {}
    for d in data:
        tr = tiers[tiers.dive == d["dive"]].iloc[0]
        cr = ci[ci.dive == d["dive"]].iloc[0]
        per_dive[d["dive"]] = dict(
            transit_windows=int(tr.windows_all), transit_km=float(tr.transit_km),
            rate=float(tr.rate_all), rate_high_mod=float(tr.rate_high_mod),
            rate_high=float(tr.rate_high),
            segment_rate=float(cr.rate), n_segments=int(cr.n_segments),
            boot_ci95=[float(cr.boot_lo), float(cr.boot_hi)],
            poisson_ci95=[float(cr.pois_lo), float(cr.pois_hi)])
    summary = dict(
        canonical_v_station=V_STATION, n_bootstrap=N_BOOT,
        segment_min_s=SEG_MIN_S, fleet=fleet, per_dive=per_dive,
        threshold_sensitivity=dict(fleet_rate_by_threshold=fleet_by_v,
                                   max_rel_change_vs_canonical=round(max_rel, 4),
                                   verdict=verdict),
        tier_shares=tier_shares)
    out = PAPER / "angle1_summary.json"
    out.write_text(json.dumps(summary, indent=2))

    log(f"fleet pooled rate {fleet['rate']}/km "
        f"(bootstrap 95% CI {fleet['boot_lo']}-{fleet['boot_hi']}, "
        f"{fleet['windows']} windows over {fleet['transit_km']} km, "
        f"{fleet['n_segments']} segments)")
    log(f"threshold sensitivity: {verdict}")
    log(f"tier shares of transit windows: {tier_shares}")
    log(ci.to_string(index=False))
    log(tiers.to_string(index=False))
    log(f"wrote {PAPER}/angle1_[threshold_sweep|tier_rates|rate_ci].csv, "
        f"angle1_[threshold|forest]_fig.png, {out.name}")
    return summary


if __name__ == "__main__":
    run(sys.argv[1:] or DIVES)
