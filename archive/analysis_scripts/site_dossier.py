#!/usr/bin/env python3
"""Anomalous-site correlation dossier — one panel per ranked anomaly site pairing
the sensor evidence with the co-located photogrammetry (ortho patch + DEM terrain).

Bridges the two primary products: for each site it locates the reconstruction that
covers it, crops the orthomosaic and DEM, derives terrain metrics, and lays them
beside the anomaly evidence.  Qt-free.  build_dossier(workspace_dir, out_path=None).
"""
from __future__ import annotations
import base64, glob, io, os
from datetime import datetime
from pathlib import Path

import numpy as np, pandas as pd, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from pyproj import Transformer

TIER = {"HIGH": "#c1272d", "MODERATE": "#d97316", "SCREEN": "#b8890a"}
R = 12.0  # crop half-width, metres


def _b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _fig_b64(fig, fmt="png", dpi=140, fc="none") -> str:
    buf = io.BytesIO(); fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight",
                                    pad_inches=0.02, facecolor=fc); plt.close(fig)
    return _b64(buf.getvalue(), f"image/{'jpeg' if fmt=='jpg' else fmt}")


def _find_chunk(PG, e, n, kind):
    """Return (path, has_data) for the chunk raster of `kind` covering (e,n)."""
    best = None
    for p in glob.glob(f"{PG}/seg*/chunk_*/{kind}"):
        if os.path.getsize(p) < 1e5: continue
        try:
            with rasterio.open(p) as ds:
                b = ds.bounds
                if not (b.left <= e <= b.right and b.bottom <= n <= b.top): continue
                row, col = ds.index(e, n)
                val = ds.read(boundless=True, window=((row, row + 1), (col, col + 1)))
                has = bool(np.any(val[:3] > 0)) if ds.count >= 3 else bool(np.isfinite(val).any())
                if has: return p
                best = best or p
        except Exception:
            continue
    return best


def _crop(ds, e, n, r, bands, res_target):
    win = from_bounds(e - r, n - r, e + r, n + r, ds.transform)
    ow = max(1, int(2 * r / res_target)); oh = ow
    data = ds.read(bands, window=win, out_shape=(len(bands), oh, ow),
                   boundless=True, fill_value=(0 if ds.count >= 3 else (ds.nodata or 0)),
                   resampling=Resampling.bilinear)
    return data


def render_ortho(path, e, n):
    with rasterio.open(path) as ds:
        d = _crop(ds, e, n, R, [1, 2, 3, 4] if ds.count >= 4 else [1, 2, 3], 0.02)
    rgb = np.transpose(d[:3], (1, 2, 0)).astype(float) / 255.0
    if d.shape[0] == 4:
        a = d[3] / 255.0
    else:
        a = (d[:3].sum(0) > 0).astype(float)
    rgba = np.dstack([rgb, a])
    fig, ax = plt.subplots(figsize=(3.1, 3.1), dpi=140)
    fig.patch.set_facecolor("#0e1620"); ax.set_facecolor("#0e1620")
    ax.imshow(rgba, extent=[-R, R, -R, R], origin="upper", interpolation="bilinear")
    ax.plot(0, 0, marker="+", ms=15, mew=1.6, color="#ffd24a")
    ax.plot([R - 8, R - 3], [-R + 2, -R + 2], color="w", lw=2.4)
    ax.text(R - 5.5, -R + 3.2, "5 m", color="w", ha="center", fontsize=7)
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    for s in ax.spines.values(): s.set_color("#24313d")
    return _fig_b64(fig, "jpg", 140, "#0e1620")


def render_dem(path, e, n):
    with rasterio.open(path) as ds:
        d = _crop(ds, e, n, R, [1], 0.04)[0].astype(float)
        nod = ds.nodata
    if nod is not None: d[d == nod] = np.nan
    valid = np.isfinite(d)
    stats = None
    if valid.sum() > 30:
        yy, xx = np.mgrid[0:d.shape[0], 0:d.shape[1]] * 0.04
        A = np.c_[xx[valid], yy[valid], np.ones(valid.sum())]
        coef, *_ = np.linalg.lstsq(A, d[valid], rcond=None)
        res = d[valid] - A @ coef
        gy, gx = np.gradient(np.where(valid, d, np.nanmean(d[valid])), 0.04)
        slope = np.degrees(np.arctan(np.hypot(gx, gy)))
        stats = dict(relief=float(np.nanmax(d[valid]) - np.nanmin(d[valid])),
                     rough=float(res.std()), slope=float(np.nanmedian(slope[valid])),
                     cover=float(valid.mean()))
    fill = np.where(valid, d, np.nanmedian(d[valid]) if valid.any() else 0)
    gy, gx = np.gradient(fill, 0.04)
    slope = np.pi / 2 - np.arctan(np.hypot(gx, gy))
    az, alt = np.radians(315), np.radians(45)
    asp = np.arctan2(-gx, gy)
    hs = np.sin(alt) * np.sin(slope) + np.cos(alt) * np.cos(slope) * np.cos(az - asp)
    hs = np.clip(hs, 0, 1)
    img = np.dstack([hs, hs, hs, valid.astype(float)])
    fig, ax = plt.subplots(figsize=(3.1, 3.1), dpi=140)
    fig.patch.set_facecolor("#12202b"); ax.set_facecolor("#12202b")
    ax.imshow(img, extent=[-R, R, -R, R], origin="upper", interpolation="bilinear")
    ax.plot(0, 0, marker="+", ms=15, mew=1.6, color="#ffd24a")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-R, R); ax.set_ylim(-R, R)
    for s in ax.spines.values(): s.set_color("#24313d")
    return _fig_b64(fig, "png", 140, "#12202b"), stats


def build_dossier(workspace_dir, out_path=None) -> str:
    B = str(workspace_dir); PG = f"{B}/survey/photogrammetry"
    sites = pd.read_csv(f"{B}/survey/anomaly/anomalous_sites.csv")
    clips = pd.read_csv(f"{B}/survey/anomaly/video_review_clips.csv")
    tf = Transformer.from_crs("EPSG:4326", "EPSG:32613", always_xy=True)
    sites["E"], sites["N"] = tf.transform(sites.lon.values, sites.lat.values)
    sites = sites.sort_values(["window_count", "max_evidence_score"], ascending=False)
    cards = []
    covered = 0
    for _, s in sites.iterrows():
        e, n = s["E"], s["N"]
        op = _find_chunk(PG, e, n, "orthomosaic_lite.tif")
        dp = _find_chunk(PG, e, n, "dem_lite.tif")
        ortho_img = render_ortho(op, e, n) if op else None
        dem_img, terr = (render_dem(dp, e, n) if dp else (None, None))
        if ortho_img: covered += 1
        nclip = int((clips.site_id == s["site_id"]).sum())
        rc = clips[clips.site_id == s["site_id"]]
        span = ""
        if len(rc):
            span = f"{pd.to_datetime(rc.review_start).min():%H:%M} – {pd.to_datetime(rc.review_end).max():%H:%M} UTC"
        cards.append(dict(
            id=s["site_id"], tier=s["best_tier"], wins=int(s["window_count"]),
            high=int(s["high_windows"]), mod=int(s["moderate_windows"]),
            ev=float(s["max_evidence_score"]), chans=s["channels"],
            first=str(s["first_time"])[:16].replace("T", " "),
            last=str(s["last_time"])[:16].replace("T", " "),
            e=e, n=n, ortho=ortho_img, dem=dem_img, terr=terr, nclip=nclip, span=span))
    html = _render(cards, covered, len(sites))
    out = out_path or f"{B}/ANOMALY_SITE_DOSSIER.html"
    Path(out).write_text(html, encoding="utf-8")
    return out


def _render(cards, covered, total) -> str:
    def card(c):
        chans = ", ".join(str(c["chans"]).split(","))
        if c["ortho"]:
            t = c["terr"] or {}
            figs = (f'<div class="pair">'
                    f'<figure><img alt="orthomosaic at {c["id"]}" src="{c["ortho"]}">'
                    f'<figcaption>Orthomosaic &middot; 24&times;24 m</figcaption></figure>'
                    f'<figure><img alt="terrain at {c["id"]}" src="{c["dem"]}">'
                    f'<figcaption>DEM hillshade &middot; 24&times;24 m</figcaption></figure></div>')
            terr = (f'<div class="metrics">'
                    f'<div><span class="mk">Median slope</span><span class="mv">{t.get("slope",0):.0f}&deg;</span></div>'
                    f'<div><span class="mk">Local relief</span><span class="mv">{t.get("relief",0):.1f} m</span></div>'
                    f'<div><span class="mk">Roughness (RMS)</span><span class="mv">{t.get("rough",0):.2f} m</span></div>'
                    f'<div><span class="mk">Imagery cover</span><span class="mv">{t.get("cover",0)*100:.0f}%</span></div>'
                    f'</div>') if t else '<p class="nodata">Terrain metrics unavailable.</p>'
        else:
            figs = ('<div class="pair nocov"><div class="ph">No co-located photogrammetry'
                    '<span>site lies outside the imaged traverse corridors</span></div></div>')
            terr = ""
        return f"""<article class="card">
  <header class="ch">
    <div class="cid">{c['id']}</div>
    <span class="pill {str(c['tier']).lower()}">{c['tier']}</span>
    <div class="cev">evidence {c['ev']:.0f}</div>
  </header>
  {figs}
  {terr}
  <div class="ev">
    <div class="evrow"><span class="ek">Windows</span><span class="ev-v">{c['wins']}<span class="sub"> &nbsp;{c['high']} high &middot; {c['mod']} moderate</span></span></div>
    <div class="evrow"><span class="ek">Channels</span><span class="ev-v">{chans}</span></div>
    <div class="evrow"><span class="ek">Active</span><span class="ev-v mono">{c['first']} &rarr; {c['last']}</span></div>
    <div class="evrow"><span class="ek">Review clips</span><span class="ev-v">{c['nclip']}{f" &middot; <span class='mono'>{c['span']}</span>" if c['span'] else ""}</span></div>
    <div class="evrow"><span class="ek">Position</span><span class="ev-v mono">E {c['e']:.0f} &nbsp; N {c['n']:.0f}</span></div>
  </div>
</article>"""
    cards_html = "\n".join(card(c) for c in cards)
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>J1756 Anomaly Site Dossier</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#fff;--ground:#eef1f4;--ink:#16222b;--muted:#5a6b78;--dim:#8493a0;
  --line:#dde4ea;--line2:#eaeff3;--accent:#0e6d78;--accent-soft:#e3eef0;
  --high:#c1272d;--moderate:#d97316;--screen:#b8890a;--shadow:0 1px 2px rgba(20,40,55,.05),0 8px 30px rgba(20,40,55,.06);}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#131c24;--ground:#0d151b;--ink:#e8eef3;--muted:#9fb0bd;--dim:#71828f;--line:#26333d;--line2:#1c272f;--accent:#3fb6c2;--accent-soft:#152a2e;--high:#f26571;--moderate:#f0a04b;--screen:#e0bb4f;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 34px rgba(0,0,0,.35);}}}}
:root[data-theme="dark"]{{--paper:#131c24;--ground:#0d151b;--ink:#e8eef3;--muted:#9fb0bd;--dim:#71828f;--line:#26333d;--line2:#1c272f;--accent:#3fb6c2;--accent-soft:#152a2e;--high:#f26571;--moderate:#f0a04b;--screen:#e0bb4f;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 34px rgba(0,0,0,.35);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6;-webkit-font-smoothing:antialiased}}
.wrap{{max-width:1120px;margin:0 auto;padding:0 24px}}
.mono{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums}}
header.top{{padding:52px 8px 26px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.16em;text-transform:uppercase;color:var(--accent);margin:0 0 12px}}
h1{{font-family:"Spectral",Georgia,serif;font-weight:600;font-size:40px;line-height:1.08;margin:0 0 12px;letter-spacing:-.01em;text-wrap:balance}}
.lede{{font-family:"Spectral",serif;font-size:18px;font-style:italic;color:var(--muted);margin:0;max-width:60ch}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(330px,1fr));gap:20px;padding:8px 8px 60px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:8px;overflow:hidden;box-shadow:var(--shadow);display:flex;flex-direction:column}}
.ch{{display:flex;align-items:center;gap:10px;padding:14px 16px;border-bottom:1px solid var(--line2)}}
.cid{{font-family:"IBM Plex Mono",monospace;font-weight:500;font-size:15px;letter-spacing:.02em}}
.cev{{margin-left:auto;font-family:"IBM Plex Mono",monospace;font-size:12px;color:var(--dim)}}
.pill{{font-family:"IBM Plex Mono",monospace;font-size:10.5px;font-weight:500;padding:2px 9px;border-radius:20px;letter-spacing:.03em}}
.pill.high{{background:color-mix(in srgb,var(--high) 16%,transparent);color:var(--high)}}
.pill.moderate{{background:color-mix(in srgb,var(--moderate) 18%,transparent);color:var(--moderate)}}
.pill.screen{{background:color-mix(in srgb,var(--screen) 20%,transparent);color:var(--screen)}}
.pair{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line2)}}
.pair figure{{margin:0;background:#0e1620}}
.pair img{{display:block;width:100%;height:auto}}
.pair figcaption{{font-size:10.5px;color:var(--dim);padding:5px 8px;background:var(--paper);text-align:center;font-family:"IBM Plex Mono",monospace}}
.nocov{{grid-template-columns:1fr;background:var(--ground)}}
.ph{{aspect-ratio:2/1;display:flex;flex-direction:column;align-items:center;justify-content:center;color:var(--muted);font-size:13px;text-align:center;padding:16px}}
.ph span{{color:var(--dim);font-size:11.5px;margin-top:5px;max-width:26ch}}
.metrics{{display:grid;grid-template-columns:1fr 1fr;gap:1px;background:var(--line2);border-bottom:1px solid var(--line2)}}
.metrics>div{{background:var(--paper);padding:9px 14px;display:flex;flex-direction:column}}
.mk{{font-size:11px;color:var(--muted)}}
.mv{{font-family:"IBM Plex Mono",monospace;font-size:15px;font-weight:500;font-variant-numeric:tabular-nums}}
.ev{{padding:12px 16px 15px;display:flex;flex-direction:column;gap:7px}}
.evrow{{display:flex;gap:12px;font-size:13px;align-items:baseline}}
.ek{{color:var(--muted);min-width:82px;flex-shrink:0}}
.ev-v{{color:var(--ink)}}
.ev-v .sub{{color:var(--dim);font-size:12px}}
.nodata{{padding:12px 16px;color:var(--dim);font-size:12.5px;margin:0}}
footer{{padding:0 8px 60px;color:var(--dim);font-size:13px;max-width:70ch}}
footer .mono{{color:var(--muted)}}
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);padding:13px 18px;border-radius:0 6px 6px 0;margin:0 8px 30px;color:var(--muted);font-size:14px;max-width:75ch}}
.note b{{color:var(--ink)}}
@media(max-width:520px){{.grid{{grid-template-columns:1fr}} h1{{font-size:31px}}}}
</style></head><body><div class="wrap">
<header class="top">
  <p class="eyebrow">Jason Dive J1756 &middot; Anomaly &times; Photogrammetry</p>
  <h1>Anomalous Site Dossier</h1>
  <p class="lede">The {total} ranked sensor-anomaly sites, each paired with the co-located
  seafloor reconstruction &mdash; orthomosaic, terrain and the anomaly evidence in one view.</p>
</header>
<div class="note"><b>Reading a card.</b> The crosshair marks the site centroid on a
24&times;24 m window. Orthomosaic (left) shows the seafloor imagery; DEM hillshade (right)
shows the micro-terrain. Sensor anomalies are advected plumes, so a site marks where the
signal was sensed &mdash; recurring high-window sites are the strongest candidates for a
source directly beneath. {covered} of {total} sites fall within the imaged corridors.</div>
<div class="grid">
{cards_html}
</div>
<footer>
  <div class="mono">Jason Dive J1756 &middot; East Pacific Rise 9&deg;N &middot; UTM 13N (EPSG:32613)</div>
  Generated {datetime.now():%Y-%m-%d} &middot; sites ranked by anomaly-window count; terrain
  metrics computed over each 24&times;24 m DEM window.
</footer>
</div></body></html>"""


if __name__ == "__main__":
    import sys
    print(build_dossier(sys.argv[1] if len(sys.argv) > 1 else "."))
