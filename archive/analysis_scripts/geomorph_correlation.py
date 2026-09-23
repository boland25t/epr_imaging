#!/usr/bin/env python3
"""Geomorphometric correlation — do sensor anomalies preferentially occur over
particular seafloor terrain?  For every trackline sample with DEM coverage we
extract local slope, roughness and relief, label it anomaly/background by whether
its timestamp lies inside a detected anomaly window, and test the two populations.
Qt-free.  run(workspace_dir) -> dict of results (+ writes a figure)."""
from __future__ import annotations
import io, base64
from pathlib import Path

import numpy as np, pandas as pd, rasterio
from scipy import ndimage
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _cliffs_delta(a, b):
    """Rank-based effect size in [-1,1]; >0 means group a tends larger."""
    a = np.asarray(a); b = np.asarray(b)
    # via Mann-Whitney U: delta = 2U/(n*m) - 1
    u, _ = mannwhitneyu(a, b, alternative="two-sided")
    return 2 * u / (len(a) * len(b)) - 1


def run(workspace_dir, fig_path=None) -> dict:
    B = str(workspace_dir)
    ip = pd.read_csv(f"{B}/inputs/interp_full.csv")
    ip = ip.dropna(subset=["easting", "northing", "unix_time"]).sort_values("unix_time")
    # --- anomaly time intervals ---
    win = pd.read_csv(f"{B}/survey/anomaly/anomaly_windows_all.csv")
    # .timestamp() is resolution-agnostic (pandas may parse as us or ns).
    ws = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    we = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    iv = sorted(zip(ws, we))
    t = ip.unix_time.to_numpy()
    is_anom = np.zeros(len(t), bool)
    for a, b in iv:
        is_anom |= (t >= a) & (t <= b)
    # --- precompute terrain rasters from merged DEM ---
    with rasterio.open(f"{B}/survey/photogrammetry/merged/dem_merged.tif") as ds:
        z = ds.read(1).astype(float); nod = ds.nodata; tr = ds.transform; res = ds.res[0]
        inv = ~ds.transform
    valid = np.isfinite(z) & (z != nod if nod is not None else True)
    zf = np.where(valid, z, np.nan)
    # slope (deg)
    gy, gx = np.gradient(np.where(valid, z, np.nanmean(z[valid])), res)
    slope = np.degrees(np.arctan(np.hypot(gx, gy)))
    # roughness: local std over ~1 m; relief: local (max-min) over ~3 m
    w1 = max(3, int(1.0 / res)); w3 = max(5, int(3.0 / res))
    zz = np.where(valid, z, np.nanmean(z[valid]))
    m1 = ndimage.uniform_filter(zz, w1)
    rough = np.sqrt(np.maximum(ndimage.uniform_filter(zz * zz, w1) - m1 * m1, 0))
    relief = ndimage.maximum_filter(zz, w3) - ndimage.minimum_filter(zz, w3)
    # --- sample at trackline points with coverage, thinned to ~1 / 2 m ---
    e = ip.easting.to_numpy(); n = ip.northing.to_numpy()
    cols, rows = (inv * (e, n))
    rows = np.round(rows).astype(int); cols = np.round(cols).astype(int)
    inb = (rows >= 0) & (rows < z.shape[0]) & (cols >= 0) & (cols < z.shape[1])
    cov = np.zeros(len(t), bool)
    cov[inb] = valid[rows[inb], cols[inb]]
    # thin along track by 2 m of horizontal travel
    keep = np.zeros(len(t), bool); last = None; acc = 0.0
    for i in range(len(t)):
        if not cov[i]:
            continue
        if last is None:
            keep[i] = True; last = i
        else:
            acc += np.hypot(e[i] - e[last], n[i] - n[last])
            if acc >= 2.0:
                keep[i] = True; last = i; acc = 0.0
    idx = np.where(keep)[0]
    r_, c_ = rows[idx], cols[idx]
    df = pd.DataFrame(dict(anom=is_anom[idx],
                           slope=slope[r_, c_], rough=rough[r_, c_], relief=relief[r_, c_]))
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    A = df[df.anom]; Bg = df[~df.anom]
    metrics = {"slope": ("Slope", "°"), "rough": ("Roughness", "m"), "relief": ("Local relief", "m")}
    res_out = {"n_anom": int(len(A)), "n_bg": int(len(Bg)), "metrics": {}}
    for k, (label, unit) in metrics.items():
        u, p = mannwhitneyu(A[k], Bg[k], alternative="two-sided")
        res_out["metrics"][k] = dict(
            label=label, unit=unit,
            anom_median=float(A[k].median()), bg_median=float(Bg[k].median()),
            anom_mean=float(A[k].mean()), bg_mean=float(Bg[k].mean()),
            p=float(p), delta=float(_cliffs_delta(A[k], Bg[k])))
    # --- figure: box comparison per metric ---
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(9.6, 3.4), dpi=140)
    fig.patch.set_alpha(0)
    for ax, (k, (label, unit)) in zip(axes, metrics.items()):
        data = [Bg[k].values, A[k].values]
        bp = ax.boxplot(data, positions=[0, 1], widths=0.6, showfliers=False, patch_artist=True,
                        medianprops=dict(color="#16222b", lw=1.6))
        for patch, col in zip(bp["boxes"], ["#9fb3c8", "#c1272d"]):
            patch.set_facecolor(col); patch.set_alpha(.75); patch.set_edgecolor("#5a6b78")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["background", "anomaly"], fontsize=9, color="#33424f")
        ax.set_title(f"{label} ({unit})", fontsize=10.5, color="#16222b")
        ax.tick_params(colors="#8a97a3", labelsize=8)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
        for s in ("left", "bottom"): ax.spines[s].set_color("#cdd6df")
        d = res_out["metrics"][k]["delta"]; pp = res_out["metrics"][k]["p"]
        ax.text(0.5, 0.97, f"δ={d:+.2f}  p={'<0.001' if pp<1e-3 else f'{pp:.3f}'}",
                transform=ax.transAxes, ha="center", va="top", fontsize=8.5, color="#5a6b78")
    plt.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=140, bbox_inches="tight", facecolor="none")
    plt.close(fig)
    res_out["fig_b64"] = "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()
    if fig_path:
        Path(fig_path).write_bytes(base64.b64decode(res_out["fig_b64"].split(",")[1]))
    return res_out


if __name__ == "__main__":
    import sys, json
    r = run(sys.argv[1] if len(sys.argv) > 1 else ".",
            fig_path=(sys.argv[2] if len(sys.argv) > 2 else None))
    print(f"anomaly samples: {r['n_anom']}   background samples: {r['n_bg']}\n")
    print(f"{'metric':14}{'anom med':>10}{'bg med':>10}{'Cliff δ':>10}{'p-value':>12}")
    for k, m in r["metrics"].items():
        pp = "<0.001" if m["p"] < 1e-3 else f"{m['p']:.4f}"
        print(f"{m['label']:14}{m['anom_median']:>9.2f}{m['unit']:1}{m['bg_median']:>9.2f}{m['unit']:1}"
              f"{m['delta']:>+10.3f}{pp:>12}")
