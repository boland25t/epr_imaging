#!/usr/bin/env python3
"""Slide deck builder for the J1754/J1756 imagery x chemistry campaign.

16:9 HTML slides (1280x720) printed to PDF via headless Chrome.  The core
content: per-dive overview slides, the J1756 vent-field-core fused portrait as
a featured slide, and one full-bleed slide per "biology-heavy" mosaic — chunks
ranked by bright-pixel-cover frame density — rendered as fused overlays
(ortho/hillshade base + trackline + anomaly windows + brightness dots +
cover rings) by region_portrait over each chunk's bounding box.

build_deck(out_html) -> path.  Qt-free.
"""
from __future__ import annotations
import base64
import glob
import json
import sys
from pathlib import Path

import pandas as pd
import rasterio

from region_portrait import build_portrait

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
DECK_DIR = ROOT / "slides"


def _b64(path):
    mime = "image/jpeg" if str(path).endswith(".jpg") else "image/png"
    return f"data:{mime};base64," + base64.b64encode(Path(path).read_bytes()).decode()


def pick_biology_chunks(workspace_dir, n=4, min_cover=6):
    """Rank chunk mosaics by bright-cover frame count within their bounds."""
    B = str(workspace_dir)
    d = pd.read_csv(f"{B}/survey/frame_color_metrics.csv")
    hp = d[d.white > 0.02]
    ranked = []
    for p in sorted(glob.glob(f"{B}/survey/photogrammetry/seg*/chunk_*/orthomosaic.tif")):
        try:
            with rasterio.open(p) as ds:
                bb = ds.bounds
        except Exception:
            continue
        inside = hp[(hp.E >= bb.left) & (hp.E <= bb.right) &
                    (hp.N >= bb.bottom) & (hp.N <= bb.top)]
        if len(inside) >= min_cover:
            label = "/".join(Path(p).parts[-3:-1])
            ranked.append((len(inside), label,
                           (bb.left - 8, bb.right + 8, bb.bottom - 8, bb.top + 8)))
    ranked.sort(reverse=True)
    # avoid near-duplicate extents (same seg neighbouring chunks): keep first per seg
    out, segs = [], set()
    for cnt, label, bbox in ranked:
        seg = label.split("/")[0]
        if seg in segs:
            continue
        segs.add(seg)
        out.append((cnt, label, bbox))
        if len(out) >= n:
            break
    return out


# Explicit user-directed featured chunks per dive; falls back to the
# bright-cover ranking when a dive has no explicit list.
FEATURED_CHUNKS = {"J1756": ["seg10/chunk_01", "seg11/chunk_01"],
                   "J1754": ["seg13/chunk_01", "seg06/chunk_01"]}


def render_native_fused(dive, label, bbox, cnt, max_px=10000, log=print):
    """Fused overlay on the chunk's NATIVE orthomosaic (no merged product, no
    contrast alteration): mosaic as backdrop, trackline + anomaly windows +
    bright-cover rings + sites on top.  Saved as high-quality JPEG so the
    original resolution survives into the printed PDF for zooming."""
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from rasterio.enums import Resampling
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    ws = ROOT / f"{dive}_down.eprproj"
    tif = ws / "survey" / "photogrammetry" / label / "orthomosaic.tif"
    with rasterio.open(tif) as ds:
        sc = max(1.0, max(ds.width, ds.height) / max_px)
        shape = (int(ds.height / sc), int(ds.width / sc))
        img = ds.read([1, 2, 3], out_shape=(3, *shape), resampling=Resampling.average)
        alpha = (ds.read(4, out_shape=shape, resampling=Resampling.nearest)
                 if ds.count >= 4 else None)
        bb = ds.bounds
        px_mm = ds.res[0] * sc * 1000
    rgb = np.transpose(img, (1, 2, 0)).astype(float) / 255.0
    if alpha is not None:
        rgb[alpha == 0] = 0.04   # some exports bake white under alpha=0
    e0, e1, n0, n1 = bb.left, bb.right, bb.bottom, bb.top
    h_px, w_px = rgb.shape[:2]
    dpi = 150
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, extent=[e0, e1, n0, n1], origin="upper", interpolation="none")
    seg = json.load(open(ws / "survey/anomaly/anomaly_segments_utm.geojson"))
    for tier, col, al in (("SCREEN", "#e0a800", .4), ("MODERATE", "#e8871e", .5),
                          ("HIGH", "#d7263d", .55)):
        segs = [np.array(f["geometry"]["coordinates"]) for f in seg["features"]
                if f["properties"].get("confidence") == tier]
        if segs:
            ax.add_collection(LineCollection(segs, colors=col, linewidths=10, alpha=al,
                                             zorder=3, capstyle="round"))
    trk = np.array(json.load(open(ws / "survey/nav_trackline/trackline.geojson"))
                   ["features"][0]["geometry"]["coordinates"])
    ax.plot(trk[:, 0], trk[:, 1], color="#c9d4de", lw=1.2, alpha=0.8, zorder=4)
    d = pd.read_csv(ws / "survey/frame_color_metrics.csv")
    hp = d[(d.white > 0.02) & (d.E >= e0) & (d.E <= e1) & (d.N >= n0) & (d.N <= n1)]
    ax.scatter(hp.E, hp.N, s=550, facecolor="none", edgecolor="#f2c94c", lw=2.2, zorder=6)
    sites = json.load(open(ws / "survey/anomaly/anomalous_sites_utm.geojson"))
    for f in sites["features"]:
        x, y = f["geometry"]["coordinates"]
        if e0 <= x <= e1 and n0 <= y <= n1:
            ax.plot(x, y, marker="o", ms=26, mfc="none", mec="#ffffff", mew=2.5, zorder=7)
    ax.set_xlim(e0, e1); ax.set_ylim(n0, n1)
    ax.plot([e0 + 1, e0 + 6], [n0 + 1.2, n0 + 1.2], color="w", lw=5, zorder=8)
    ax.text(e0 + 3.5, n0 + 1.9, "5 m", color="w", ha="center", fontsize=26, zorder=8)
    ax.set_axis_off()
    tmp = DECK_DIR / f"{dive}_{label.replace('/', '_')}_native.png"
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    out = tmp.with_suffix(".jpg")
    im = Image.open(tmp).convert("RGB")
    if im.height > 1.6 * im.width:      # tall ribbon -> landscape for the 16:9 slide
        im = im.rotate(90, expand=True)
    im.save(out, quality=92)
    tmp.unlink()
    log(f"  {dive} {label}: native fused {w_px}x{h_px}px ({px_mm:.1f} mm/px) -> {out.name}")
    return str(out)


def _hotspots(ws, bb, k=2, cell=6.0):
    """Densest bright-cover clusters inside a chunk's bounds (UTM centers)."""
    import numpy as np
    d = pd.read_csv(ws / "survey/frame_color_metrics.csv")
    hp = d[(d.white > 0.02) & (d.E >= bb.left) & (d.E <= bb.right) &
           (d.N >= bb.bottom) & (d.N <= bb.top)]
    if not len(hp):
        return [((bb.left + bb.right) / 2, (bb.bottom + bb.top) / 2)]
    gx = ((hp.E - bb.left) // cell).astype(int)
    gy = ((hp.N - bb.bottom) // cell).astype(int)
    counts = hp.groupby([gx, gy]).size().sort_values(ascending=False)
    out = []
    for (ix, iy), _ in counts.items():
        cx = bb.left + (ix + 0.5) * cell
        cy = bb.bottom + (iy + 0.5) * cell
        if all(abs(cx - a) + abs(cy - b) > 2 * cell for a, b in out):
            out.append((cx, cy))
        if len(out) >= k:
            break
    return out


def render_native_detail(dive, label, center, half_w=3.5, log=print):
    """True-native 1:1 crop around a hotspot: organisms at full mosaic
    resolution with thin overlay context.  half_w in metres."""
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from rasterio.windows import from_bounds
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None

    ws = ROOT / f"{dive}_down.eprproj"
    tif = ws / "survey" / "photogrammetry" / label / "orthomosaic.tif"
    cx, cy = center
    with rasterio.open(tif) as ds:
        e0, e1 = cx - half_w, cx + half_w
        n0, n1 = cy - half_w * 0.62, cy + half_w * 0.62
        win = from_bounds(e0, n0, e1, n1, ds.transform)
        img = ds.read([1, 2, 3], window=win, boundless=True, fill_value=0)
        alpha = (ds.read(4, window=win, boundless=True, fill_value=0)
                 if ds.count >= 4 else None)
        px_mm = ds.res[0] * 1000
    rgb = np.transpose(img, (1, 2, 0)).astype(float) / 255.0
    if alpha is not None:
        valid = alpha > 0
        rgb[~valid] = 0.04
    else:
        valid = rgb.sum(2) > 0
    if valid.mean() < 0.25:
        log(f"  {dive} {label} detail @({cx:.0f},{cy:.0f}): <25% imaged — skipped")
        return None
    h_px, w_px = rgb.shape[:2]
    dpi = 150
    fig = plt.figure(figsize=(w_px / dpi, h_px / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(rgb, extent=[e0, e1, n0, n1], origin="upper", interpolation="none")
    seg = json.load(open(ws / "survey/anomaly/anomaly_segments_utm.geojson"))
    for tier, col, al in (("SCREEN", "#e0a800", .35), ("MODERATE", "#e8871e", .4),
                          ("HIGH", "#d7263d", .45)):
        segs = [np.array(f["geometry"]["coordinates"]) for f in seg["features"]
                if f["properties"].get("confidence") == tier]
        if segs:
            ax.add_collection(LineCollection(segs, colors=col, linewidths=14, alpha=al,
                                             zorder=3, capstyle="round"))
    ax.set_xlim(e0, e1); ax.set_ylim(n0, n1)
    ax.plot([e0 + 0.25, e0 + 1.25], [n0 + 0.3, n0 + 0.3], color="w", lw=6, zorder=8)
    ax.text(e0 + 0.75, n0 + 0.45, "1 m", color="w", ha="center", fontsize=34, zorder=8)
    ax.set_axis_off()
    tmp = DECK_DIR / f"{dive}_{label.replace('/', '_')}_det_{int(cx)}_{int(cy)}.png"
    fig.savefig(tmp, dpi=dpi)
    plt.close(fig)
    out = tmp.with_suffix(".jpg")
    Image.open(tmp).convert("RGB").save(out, quality=90)
    tmp.unlink()
    log(f"  {dive} {label} detail @({cx:.0f},{cy:.0f}): {w_px}x{h_px}px native {px_mm:.1f} mm/px")
    return str(out)


def render_biology_portraits(dive, n=4, log=print):
    ws = ROOT / f"{dive}_down.eprproj"
    DECK_DIR.mkdir(exist_ok=True)
    explicit = FEATURED_CHUNKS.get(dive)
    if explicit:
        import rasterio as _rio
        picks = []
        for label in explicit:
            tif = ws / "survey" / "photogrammetry" / label / "orthomosaic.tif"
            with _rio.open(tif) as ds:
                bb = ds.bounds
            d = pd.read_csv(ws / "survey" / "frame_color_metrics.csv")
            hp = d[d.white > 0.02]
            cnt = int(((hp.E >= bb.left) & (hp.E <= bb.right) &
                       (hp.N >= bb.bottom) & (hp.N <= bb.top)).sum())
            picks.append((cnt, label,
                          (bb.left - 8, bb.right + 8, bb.bottom - 8, bb.top + 8)))
    else:
        picks = pick_biology_chunks(ws, n=n)
    outs = []
    for cnt, label, bbox in picks:
        fp = render_native_fused(dive, label, bbox, cnt, log=log)
        outs.append((label, cnt, fp))
    return outs


DIVES_ALL = ["J1754", "J1755", "J1756", "J1758", "J1759", "J1760", "J1761"]


def render_log_trackline(dive, log_fn=print):
    """Per-dive track comparison row, all panels on one identical UTM frame:
    log-spectrum tracklines (raw CH4/CO2, turbo on log10 — the 3D viewer's
    log scalar colouring), the anomaly-window trackline, and, for imaged
    dives, the bright-spot trackline."""
    import json
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from matplotlib.collections import LineCollection

    ws = ROOT / f"{dive}_down.eprproj"
    ip = pd.read_csv(ws / "inputs" / "interp_full.csv",
                     usecols=["easting", "northing", "CH4 Concentration",
                              "CO2 Concentration"]).dropna()
    seg = json.load(open(ws / "survey/anomaly/anomaly_segments_utm.geojson"))
    fb = ws / "survey" / "frame_color_metrics.csv"
    has_bright = fb.is_file()
    npanels = 4 if has_bright else 3

    e0, e1 = ip.easting.min(), ip.easting.max()
    n0, n1 = ip.northing.min(), ip.northing.max()
    me = max((e1 - e0) * 0.06, 20); mn = max((n1 - n0) * 0.04, 20)
    lims = (e0 - me, e1 + me, n0 - mn, n1 + mn)

    fig, axes = plt.subplots(1, npanels, figsize=(3.2 * npanels + 1.5, 6.8), dpi=150)
    fig.patch.set_facecolor("#0e1620")

    def frame(ax, title):
        ax.set_facecolor("#0e1620")
        ax.set_xlim(lims[0], lims[1]); ax.set_ylim(lims[2], lims[3])
        ax.set_aspect("equal")
        ax.set_title(title, fontsize=10, color="#dbe4ec")
        ax.set_xticks([]); ax.set_yticks([])
        for s_ in ax.spines.values():
            s_.set_color("#2a3846")

    for k, (col, pretty) in enumerate((("CH4 Concentration", "CH$_4$ (log)"),
                                       ("CO2 Concentration", "CO$_2$ (log)"))):
        ax = axes[k]
        v = ip[col].clip(lower=max(ip[col][ip[col] > 0].min(), 1e-3))
        sc = ax.scatter(ip.easting, ip.northing, c=v, s=2.5, lw=0, cmap="turbo",
                        norm=LogNorm(vmin=v.quantile(0.02), vmax=v.quantile(0.995)))
        cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02)
        cb.ax.tick_params(colors="#71828f", labelsize=6)
        frame(ax, f"{pretty} spectrum")

    ax = axes[2]
    ax.plot(ip.easting, ip.northing, color="#41505d", lw=0.6, zorder=1)
    for tier, colr in (("SCREEN", "#e0a800"), ("MODERATE", "#e8871e"), ("HIGH", "#d7263d")):
        segs = [np.array(f["geometry"]["coordinates"]) for f in seg["features"]
                if f["properties"].get("confidence") == tier]
        if segs:
            ax.add_collection(LineCollection(segs, colors=colr, linewidths=2.4, zorder=3))
    frame(ax, "anomaly windows")

    if has_bright:
        d = pd.read_csv(fb)
        ax = axes[3]
        ax.plot(ip.easting, ip.northing, color="#41505d", lw=0.6, zorder=1)
        ax.scatter(d.E, d.N, c=d.bright, cmap="cividis", s=4, lw=0, zorder=3)
        hp = d[d.white > 0.02]
        ax.scatter(hp.E, hp.N, s=26, facecolor="none", edgecolor="#f2c94c",
                   lw=0.7, zorder=4)
        frame(ax, "bright spots")

    fig.suptitle(dive, fontsize=13, color="#e8eef3")
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fp = DECK_DIR / f"{dive}_log_trackline.png"
    fig.savefig(fp, dpi=150, facecolor="#0e1620", bbox_inches="tight")
    plt.close(fig)
    log_fn(f"  {dive}: track comparison row ({npanels} panels) -> {fp.name}")
    return str(fp)


def _dive_stats(dive):
    ws = ROOT / f"{dive}_down.eprproj"
    a = json.loads((ws / "survey/analysis_figs/analysis_stats.json").read_text())
    win = pd.read_csv(ws / "survey/anomaly/anomaly_windows_all.csv")
    sites = pd.read_csv(ws / "survey/anomaly/anomalous_sites.csv")
    return dict(
        frames=a.get("n_frames", 0), cover=a.get("n_cover", 0),
        windows=len(win), sites=len(sites),
        band_rho=a.get("conc", {}).get("band", {}).get("rho_bright_ch4", 0),
        cover_fold=a.get("conc", {}).get("band", {}).get("cover_fold", 0),
        or_any=a.get("bool", {}).get("any", {}).get("odds_ratio", 0),
    )


def _slide(body, cls=""):
    return f'<section class="slide {cls}">{body}</section>'


def _img_slide(img_path, title, note=""):
    cap = f'<div class="cap"><b>{title}</b>{" — " + note if note else ""}</div>'
    return _slide(f'<img class="full" src="{_b64(img_path)}">{cap}', "dark")


def build_deck(out_html=None, log=print) -> str:
    slides = []
    slides.append(_slide(
        '<div class="titlebox"><p class="eyebrow">EPR 9&deg;N &middot; East Pacific Rise '
        '&middot; ROV Jason downward traversals</p>'
        '<h1>Seafloor Imagery &times; Gas Chemistry</h1>'
        '<p class="sub">Dives J1754 &amp; J1756 — photogrammetry, altitude-corrected '
        'brightness, and multi-channel anomaly detection, fused.</p></div>', "title"))

    for dive in ("J1756", "J1754"):
        ws = ROOT / f"{dive}_down.eprproj"
        st = _dive_stats(dive)
        stat_html = "".join(
            f'<div class="stat"><div class="v">{v}</div><div class="k">{k}</div></div>'
            for v, k in ((f"{st['frames']:,}", "transit frames analysed"),
                         (f"{st['windows']}", "anomaly windows"),
                         (f"{st['sites']}", "ranked sites"),
                         (f"{st['cover']}", "bright-cover frames")))
        bmap = ws / "survey/analysis_figs/brightness_map.png"
        slides.append(_slide(
            f'<div class="split"><div class="left"><p class="eyebrow">{dive}</p>'
            f'<h2>Dive overview</h2><div class="stats">{stat_html}</div>'
            f'<p class="note">Bright-pixel cover rides inside anomaly windows at '
            f'{st["or_any"]:.1f}&times; the odds of bare seafloor; in the fixed altitude '
            f'band, cover frames carry {st["cover_fold"]:.1f}&times; the CH&#8324;.</p></div>'
            f'<div class="right"><img src="{_b64(bmap)}"></div></div>'))

        if dive == "J1756":
            core = ws / "survey/analysis_figs/region_portrait_core.png"
            slides.append(_img_slide(
                core, "J1756 vent-field core — the anomaly cluster",
                "multiple intersecting tracklines over the same ground; each pass an "
                "independent optical record; windows and brightness fused on the mosaic"))

        n_det = 2 if dive in FEATURED_CHUNKS else 1
        for label, cnt, fp in render_biology_portraits(dive, n=4, log=log):
            slides.append(_img_slide(
                fp, f"{dive} {label}",
                f"biology-dense mosaic ({cnt} bright-cover frames) with trackline, "
                "anomaly windows and bright spots overlaid"))
            tifp = ROOT / f"{dive}_down.eprproj" / "survey" / "photogrammetry" / label / "orthomosaic.tif"
            with rasterio.open(tifp) as _ds:
                _bb = _ds.bounds
            got = 0
            for ctr in _hotspots(ROOT / f"{dive}_down.eprproj", _bb, k=n_det + 4):
                dfp = render_native_detail(dive, label, ctr, log=log)
                if dfp is None:
                    continue
                got += 1
                slides.append(_img_slide(
                    dfp, f"{dive} {label} — detail {got}, native resolution",
                    "1:1 mosaic pixels at a bright-cover / anomaly hotspot"))
                if got >= n_det:
                    break

        if dive == "J1754":
            core = ws / "survey/analysis_figs/region_portrait_core.png"
            slides.append(_img_slide(
                core, "J1754 densest site cluster — fused view",
                "the corridor dive's strongest anomaly neighbourhood"))

    slides.append(_slide(
        '<div class="titlebox"><p class="eyebrow">All gas-equipped dives</p>'
        '<h2>Log-scaled sensor spectrum tracklines</h2>'
        '<p class="sub">Per dive, on one identical frame: the raw log&#8321;&#8320; sensor record, '
        'the detector windows, and the bright-spot record where imagery exists.</p></div>', "title"))
    for dv in DIVES_ALL:
        try:
            fp = render_log_trackline(dv, log_fn=log)
            slides.append(_img_slide(fp, f"{dv} — spectrum, anomaly and bright-spot tracklines",
                                     "identical frame per panel: raw log-scaled sensor record, "
                                     "detector windows, and (imaged dives) bright-pixel cover"))
        except Exception as e:
            log(f"  {dv}: log trackline failed ({e})")

    slides.append(_slide(
        '<div class="titlebox"><h2>Next: the cross-dive picture</h2>'
        '<p class="sub">Anomaly detection across all gas-equipped dives (J1754&ndash;J1761), '
        'station vs transit spike separation, and sensor time-delay correction — '
        'in preparation.</p></div>', "title"))

    css = """
<style>
@page { size: 1280px 720px; margin: 0; }
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif; background: #0b1119; }
.slide { width: 1280px; height: 720px; page-break-after: always; position: relative;
         background: #0e1620; color: #dbe4ec; overflow: hidden; display: flex;
         align-items: center; justify-content: center; }
.slide.title { background: linear-gradient(135deg, #0b1119 0%, #122733 100%); }
.titlebox { max-width: 900px; padding: 60px; }
.eyebrow { font-family: 'IBM Plex Mono', monospace; font-size: 13px; letter-spacing: 2px;
           text-transform: uppercase; color: #0e7c86; margin-bottom: 18px; }
h1 { font-size: 52px; font-weight: 600; color: #eef3f7; margin-bottom: 18px; }
h2 { font-size: 34px; font-weight: 600; color: #eef3f7; margin-bottom: 20px; }
.sub { font-size: 20px; color: #9fb0bd; line-height: 1.5; }
img.full { max-width: 1280px; max-height: 660px; object-fit: contain; }
.cap { position: absolute; left: 0; right: 0; bottom: 0; padding: 10px 24px;
       background: rgba(11,17,25,0.85); font-size: 15px; color: #c8d3dc; }
.split { display: flex; width: 100%; height: 100%; }
.split .left { width: 420px; padding: 56px 36px; }
.split .right { flex: 1; display: flex; align-items: center; justify-content: center; }
.split .right img { max-width: 100%; max-height: 700px; object-fit: contain; }
.stats { display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin: 22px 0; }
.stat .v { font-family: 'IBM Plex Mono', monospace; font-size: 30px; color: #6ec6e6; }
.stat .k { font-size: 12.5px; color: #8a97a3; margin-top: 2px; }
.note { font-size: 15px; color: #9fb0bd; line-height: 1.5; }
</style>"""
    html = ("<!doctype html><html><head><meta charset='utf-8'>"
            "<title>EPR J1754 + J1756 Imagery x Chemistry Deck</title>"
            + css + "</head><body>" + "".join(slides) + "</body></html>")
    out = Path(out_html or DECK_DIR / "J1754_J1756_deck.html")
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    log(f"deck html: {out} ({len(slides)} slides)")
    return str(out)


if __name__ == "__main__":
    build_deck()
    print("DECK_DONE")
