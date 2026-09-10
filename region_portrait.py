#!/usr/bin/env python3
"""Region portrait — a fused visual of one anomalous region: photogrammetry as
the base (orthomosaic where imaged, DEM hillshade filling the gaps), anomaly
windows as translucent strokes under the track, per-frame corrected brightness
as coloured dots along every pass, bright-pixel-cover frames ringed, ranked
sites labelled, with a survey locator inset.

build_portrait(workspace_dir, bbox=None, out_path=None) -> str
bbox = (e0, e1, n0, n1) in UTM; default frames the ranked-site cluster.
"""
from __future__ import annotations
import json, sys
from pathlib import Path

import numpy as np, pandas as pd, rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D

TIER_COL = {"HIGH": "#d7263d", "MODERATE": "#e8871e", "SCREEN": "#e0a800"}


def _crop(path, bbox, bands, res_t):
    e0, e1, n0, n1 = bbox
    with rasterio.open(path) as ds:
        win = from_bounds(e0, n0, e1, n1, ds.transform)
        W = max(1, int((e1 - e0) / res_t)); H = max(1, int((n1 - n0) / res_t))
        data = ds.read(bands, window=win, out_shape=(len(bands), H, W),
                       boundless=True, fill_value=0, resampling=Resampling.bilinear)
        nod = ds.nodata
    return data, nod


def build_portrait(workspace_dir, bbox=None, out_path=None) -> str:
    B = str(workspace_dir)
    sites = json.load(open(f"{B}/survey/anomaly/anomalous_sites_utm.geojson"))
    seg = json.load(open(f"{B}/survey/anomaly/anomaly_segments_utm.geojson"))
    d = pd.read_csv(f"{B}/survey/frame_color_metrics.csv")
    trk = np.array(json.load(open(f"{B}/survey/nav_trackline/trackline.geojson"))
                   ["features"][0]["geometry"]["coordinates"])
    if bbox is None:
        # frame the densest cluster of ranked sites, with margin: centre on the
        # site with the most neighbours (a corridor dive's median can fall in
        # empty water between clusters)
        pts = np.array([f["geometry"]["coordinates"] for f in sites["features"]])
        dist = np.hypot(pts[:, 0, None] - pts[None, :, 0],
                        pts[:, 1, None] - pts[None, :, 1])
        cx, cy = pts[int(np.argmax((dist < 80).sum(1)))]
        keep = pts[np.hypot(pts[:, 0] - cx, pts[:, 1] - cy) < 80]
        if len(keep) < 2:
            keep = pts
        e0, e1 = keep[:, 0].min() - 25, keep[:, 0].max() + 25
        n0, n1 = keep[:, 1].min() - 22, keep[:, 1].max() + 22
        bbox = (e0, e1, n0, n1)
    e0, e1, n0, n1 = bbox
    res_t = max((e1 - e0), (n1 - n0)) / 2300
    # --- base: ortho over hillshade ---
    rgb, _ = _crop(f"{B}/survey/photogrammetry/merged/ortho_merged.tif",
                   bbox, [1, 2, 3], res_t)
    rgb = np.transpose(rgb, (1, 2, 0)).astype(float) / 255.0
    has_img = rgb.sum(2) > 0
    (zed, nod) = _crop(f"{B}/survey/photogrammetry/merged/dem_merged.tif",
                       bbox, [1], res_t * 1.6)
    z = zed[0].astype(float)
    if nod is not None: z[z == nod] = np.nan
    zvalid = np.isfinite(z) & (z != 0)
    zf = np.where(zvalid, z, np.nanmedian(z[zvalid]) if zvalid.any() else 0)
    gy, gx = np.gradient(zf, res_t * 1.6)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy) * 1.2)
    az, alt = np.radians(315), np.radians(45)
    asp = np.arctan2(-gx, gy)
    hs = np.clip(np.sin(alt) * np.sin(slope) +
                 np.cos(alt) * np.cos(slope) * np.cos(az - asp), 0, 1)
    # upsample hillshade to rgb grid
    from PIL import Image as PILImage
    hs_img = np.array(PILImage.fromarray((hs * 255).astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]))) / 255.0
    zv_img = np.array(PILImage.fromarray((zvalid * 255).astype(np.uint8)).resize(
        (rgb.shape[1], rgb.shape[0]))) > 128
    base = np.zeros_like(rgb)
    # imaged ground: ortho modulated slightly by relief; gaps: dim grey hillshade
    mod = (0.65 + 0.35 * hs_img)[..., None]
    base[has_img] = (rgb * mod)[has_img]
    grey = np.dstack([hs_img * 0.32 + 0.10] * 3)
    base[~has_img & zv_img] = grey[~has_img & zv_img]
    base[~has_img & ~zv_img] = 0.045
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11.4, 10.2), dpi=150)
    fig.patch.set_facecolor("#0b1119"); ax.set_facecolor("#0b1119")
    ax.imshow(base, extent=[e0, e1, n0, n1], origin="upper", interpolation="bilinear")
    # --- anomaly windows: wide translucent strokes under the track ---
    for tier, lw, al in (("SCREEN", 7, .4), ("MODERATE", 7, .5), ("HIGH", 7, .55)):
        s = [np.array(f["geometry"]["coordinates"]) for f in seg["features"]
             if f["properties"].get("confidence") == tier]
        if s: ax.add_collection(LineCollection(s, colors=TIER_COL[tier],
                                               linewidths=lw, alpha=al, zorder=3,
                                               capstyle="round"))
    ax.plot(trk[:, 0], trk[:, 1], color="#c9d4de", lw=0.55, alpha=0.65, zorder=4)
    # --- per-frame brightness dots + cover rings ---
    m = d[(d.E >= e0) & (d.E <= e1) & (d.N >= n0) & (d.N <= n1)]
    sc = ax.scatter(m.E, m.N, c=m.bright, cmap="cividis", s=22, lw=0.4,
                    edgecolor="#0b1119", zorder=5)
    hp = m[m.white > 0.02]
    ax.scatter(hp.E, hp.N, s=95, facecolor="none", edgecolor="#f2c94c",
               lw=1.1, zorder=6)
    # --- sites ---
    for f in sites["features"]:
        x, y = f["geometry"]["coordinates"]
        if not (e0 <= x <= e1 and n0 <= y <= n1): continue
        ax.plot(x, y, marker="o", ms=13, mfc="none", mec="#ffffff", mew=1.5, zorder=7)
        ax.annotate(f["properties"]["site_id"].replace("SITE-", "S"),
                    (x, y), xytext=(9, 7), textcoords="offset points",
                    fontsize=8.5, color="#ffffff", zorder=7,
                    path_effects=None)
    ax.set_xlim(e0, e1); ax.set_ylim(n0, n1); ax.set_aspect("equal")
    ax.tick_params(colors="#5c6b7a", labelsize=7)
    for s_ in ax.spines.values(): s_.set_color("#2a3846")
    # scale bar
    ax.plot([e0 + 6, e0 + 26], [n0 + 6, n0 + 6], color="w", lw=3, zorder=8)
    ax.text(e0 + 16, n0 + 9, "20 m", color="w", ha="center", fontsize=9, zorder=8)
    # colourbar + legend
    cb = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.02)
    cb.set_label("corrected scene brightness (per frame)", fontsize=9, color="#9fb0bd")
    cb.ax.tick_params(colors="#71828f", labelsize=7)
    leg = [Line2D([0], [0], color=TIER_COL["HIGH"], lw=6, alpha=.55, label="anomaly window — high"),
           Line2D([0], [0], color=TIER_COL["MODERATE"], lw=6, alpha=.5, label="anomaly window — moderate"),
           Line2D([0], [0], color=TIER_COL["SCREEN"], lw=6, alpha=.4, label="anomaly window — screen"),
           Line2D([0], [0], marker="o", ls="", mfc="#8f9c48", mec="none", ms=6,
                  label="frame (colour = brightness)"),
           Line2D([0], [0], marker="o", ls="", mfc="none", mec="#f2c94c", ms=9,
                  label="bright-pixel cover >2%"),
           Line2D([0], [0], marker="o", ls="", mfc="none", mec="#fff", ms=9,
                  label="ranked anomaly site")]
    ax.legend(handles=leg, loc="upper left", fontsize=8, framealpha=0.92,
              facecolor="#131f2b", edgecolor="#2a3846", labelcolor="#dbe4ec")
    # locator inset
    axi = fig.add_axes([0.685, 0.055, 0.24, 0.20])
    axi.set_facecolor("#0e1620")
    axi.plot(trk[:, 0], trk[:, 1], color="#5f7182", lw=0.6)
    axi.add_patch(plt.Rectangle((e0, n0), e1 - e0, n1 - n0, fill=False,
                                edgecolor="#f2c94c", lw=1.3))
    axi.set_xticks([]); axi.set_yticks([]); axi.set_aspect("equal")
    for s_ in axi.spines.values(): s_.set_color("#3a4b5c")
    axi.set_title("survey context", fontsize=7.5, color="#9fb0bd", pad=3)
    ax.set_title("Vent-field core — photogrammetry, brightness and anomaly windows fused",
                 fontsize=13, color="#e8eef3", pad=10)
    out = out_path or f"{B}/survey/analysis_figs/region_portrait_core.png"
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="#0b1119")
    plt.close(fig)
    return out


if __name__ == "__main__":
    print(build_portrait(sys.argv[1] if len(sys.argv) > 1 else "."))
