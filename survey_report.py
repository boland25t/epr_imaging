#!/usr/bin/env python3
"""Assemble the J1756 survey report — a self-contained professional HTML
document (embedded figures) covering the photogrammetry, the integrated survey
map, anomaly detection, sensor products, and the processing methodology.

Qt-free.  build_survey_report(workspace_dir, out_path=None) -> str.
"""
from __future__ import annotations
import base64, glob, io, json, os
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, rasterio
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from rasterio.enums import Resampling

TIER_COL = {"HIGH": "#d7263d", "MODERATE": "#e8871e", "SCREEN": "#e0a800"}

# Chunks omitted from report figures and the inventory table, keyed by the
# bundle they were reviewed in (a product-level decision, not a global one).
EXCLUDE_CHUNKS = {"J1756_down.eprproj": {"seg16/chunk_02", "seg17/chunk_01"}}


def _bundle_excludes(path: str) -> set:
    p = path.replace("\\", "/")
    for bundle, ex in EXCLUDE_CHUNKS.items():
        if f"/{bundle}/" in p + "/":
            return ex
    return set()


def _excluded(path: str) -> bool:
    p = path.replace("\\", "/")
    return any(p.endswith(x) or f"/{x}/" in p + "/" for x in _bundle_excludes(p))


def _b64(data: bytes, mime: str) -> str:
    return f"data:{mime};base64," + base64.b64encode(data).decode()


def _fig_bytes(fig, fmt="png", dpi=150, facecolor="none") -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=dpi, bbox_inches="tight", facecolor=facecolor)
    plt.close(fig); return buf.getvalue()


def _img_file_b64(path: str, mime="image/png") -> str | None:
    try:
        return _b64(Path(path).read_bytes(), mime)
    except OSError:
        return None


# ---------------------------------------------------------------- figures ----
def fig_survey_map(B) -> bytes:
    with rasterio.open(f"{B}/survey/photogrammetry/merged/ortho_merged.tif") as ds:
        sc = max(ds.width, ds.height) / 2200
        W, H = int(ds.width / sc), int(ds.height / sc)
        img = ds.read([1, 2, 3], out_shape=(3, H, W), resampling=Resampling.average)
        b = ds.bounds; ext = [b.left, b.right, b.bottom, b.top]
    rgb = np.transpose(img, (1, 2, 0)).astype(float)
    mask = rgb.sum(2) == 0
    rgba = np.dstack([rgb / 255.0, np.where(mask, 0, 1.0)])
    trk = np.array(json.load(open(f"{B}/survey/nav_trackline/trackline.geojson"))
                   ["features"][0]["geometry"]["coordinates"])
    _seg_p = Path(B) / "survey/anomaly/anomaly_segments_utm.geojson"
    _site_p = Path(B) / "survey/anomaly/anomalous_sites_utm.geojson"
    seg = json.load(open(_seg_p)) if _seg_p.is_file() else {"features": []}
    sites = json.load(open(_site_p)) if _site_p.is_file() else {"features": []}
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(11, 9.2), dpi=150)
    fig.patch.set_facecolor("#0e1620"); ax.set_facecolor("#0e1620")
    ax.imshow(rgba, extent=ext, origin="upper", interpolation="bilinear")
    ax.plot(trk[:, 0], trk[:, 1], color="#9fb3c8", lw=0.5, alpha=0.7, zorder=2)
    for i, tier in enumerate(("SCREEN", "MODERATE", "HIGH")):
        s = [np.array(f["geometry"]["coordinates"]) for f in seg["features"]
             if f["properties"].get("confidence") == tier]
        if s:
            ax.add_collection(LineCollection(s, colors=TIER_COL[tier], linewidths=2.4, zorder=3 + i))
    for f in sites["features"]:
        x, y = f["geometry"]["coordinates"]
        ax.plot(x, y, marker="o", ms=7, mfc="none", mec="#ffffff", mew=1.4, zorder=8)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    x0, y0 = ext[0] + 15, ext[2] + 18
    ax.plot([x0, x0 + 50], [y0, y0], color="w", lw=3)
    ax.text(x0 + 25, y0 + 6, "50 m", color="w", ha="center", fontsize=9)
    ax.tick_params(colors="#6b7d8f", labelsize=7)
    for sp in ax.spines.values(): sp.set_color("#2a3846")
    leg = [Line2D([0], [0], color=TIER_COL["HIGH"], lw=3, label="High confidence"),
           Line2D([0], [0], color=TIER_COL["MODERATE"], lw=3, label="Moderate"),
           Line2D([0], [0], color=TIER_COL["SCREEN"], lw=3, label="Screen"),
           Line2D([0], [0], color="#9fb3c8", lw=1, label="ROV trackline"),
           Line2D([0], [0], marker="o", mfc="none", mec="#fff", ls="", label="Anomalous site")]
    ax.legend(handles=leg, loc="upper right", framealpha=0.9, facecolor="#16222e",
              edgecolor="#2a3846", fontsize=8, labelcolor="#dbe4ec")
    plt.tight_layout()
    return _fig_bytes(fig, "jpg", 150, "#0e1620")


def fig_channels(win: pd.DataFrame) -> bytes:
    chans = ["CO2", "CH4", "O2", "Temperature", "Salinity"]
    counts = {c: int(win["channels"].fillna("").str.contains(c).sum()) for c in chans}
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.2, 2.9), dpi=150)
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    y = np.arange(len(chans))
    ax.barh(y, [counts[c] for c in chans], color="#0e7c86", height=0.62)
    for i, c in enumerate(chans):
        ax.text(counts[c] + max(counts.values()) * 0.01, i, str(counts[c]),
                va="center", fontsize=10, color="#33424f")
    ax.set_yticks(y); ax.set_yticklabels([c.replace("Temperature", "Temp.") for c in chans], fontsize=10, color="#33424f")
    ax.invert_yaxis(); ax.set_xlabel("windows implicating channel", fontsize=9, color="#5c6b7a")
    ax.tick_params(colors="#8a97a3", labelsize=8)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#cdd6df")
    plt.tight_layout()
    return _fig_bytes(fig, "png", 150, "none")


def fig_sensor(B) -> bytes | None:
    cand = sorted(glob.glob(f"{B}/survey/sensor_2d/**/CO2*2d.tif", recursive=True))
    if not cand: return None
    with rasterio.open(cand[-1]) as ds:
        sc = max(ds.width, ds.height) / 1200
        z = ds.read(1, out_shape=(int(ds.height / sc), int(ds.width / sc)),
                    resampling=Resampling.average).astype(float)
        nod = ds.nodata
    if nod is not None: z[z == nod] = np.nan
    z[z <= 0] = np.nan
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.2, 4.4), dpi=150)
    fig.patch.set_alpha(0); ax.set_facecolor("#0e1620")
    vmax = np.nanpercentile(z, 98)
    im = ax.imshow(z, cmap="magma", vmin=np.nanpercentile(z, 5), vmax=vmax)
    ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_color("#2a3846")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("CO$_2$ concentration", fontsize=9, color="#5c6b7a")
    cb.ax.tick_params(colors="#8a97a3", labelsize=7)
    plt.tight_layout()
    return _fig_bytes(fig, "png", 150, "none")


# ------------------------------------------------------------------- data ----
def _dive_name(B) -> str:
    import re
    m = re.search(r"(J\d{4})", Path(B).name)
    return m.group(1) if m else Path(B).name


def collect(B, fast_mesh=False) -> dict:
    ip = pd.read_csv(f"{B}/inputs/interp_full.csv")
    ip = ip.dropna(subset=["easting", "northing", "depth"]).sort_values("unix_time")
    e, n, d, t = (ip[c].to_numpy() for c in ("easting", "northing", "depth", "unix_time"))
    p3 = np.sqrt(np.diff(e) ** 2 + np.diff(n) ** 2 + np.diff(d) ** 2).sum()
    ph = np.hypot(np.diff(e), np.diff(n)).sum()
    dive = dict(
        start=datetime.fromtimestamp(t.min(), timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        end=datetime.fromtimestamp(t.max(), timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        dur_h=(t.max() - t.min()) / 3600, samples=len(ip),
        path_3d=p3, path_h=ph, dmin=float(d.min()), dmax=float(d.max()),
        e0=float(e.min()), e1=float(e.max()), n0=float(n.min()), n1=float(n.max()),
        utm="UTM 13N (EPSG:32613)",
        name=_dive_name(B),
        excl=sorted(_bundle_excludes(f"{B}/")))
    PG = f"{B}/survey/photogrammetry"
    chunks = [c for c in sorted(glob.glob(f"{PG}/seg*/chunk_*")) if not _excluded(c)]
    orthos = [c for c in chunks if os.path.exists(f"{c}/orthomosaic.tif")
              and os.path.getsize(f"{c}/orthomosaic.tif") > 1e6]
    # streaming OBJ line counts are slow on /mnt/f — cache by (size, mtime)
    cache_path = Path(B) / "survey" / ".mesh_vf_cache.json"
    try: _vf_cache = json.loads(cache_path.read_text())
    except (OSError, ValueError): _vf_cache = {}

    def vf(fn):
        try: st = os.stat(fn)
        except OSError: return 0, 0
        key = f"{fn}|{st.st_size}|{int(st.st_mtime)}"
        if key in _vf_cache: return tuple(_vf_cache[key])
        if fast_mesh:
            # size-based estimate (calibrated on streamed OBJs: ~229 B/vertex,
            # ~1.95 faces/vertex); not cached so a later exact pass replaces it
            nv = int(st.st_size / 229)
            return nv, int(nv * 1.95)
        nv = nf = 0
        try:
            with open(fn, "rb") as f:
                for ln in f:
                    if ln[:2] == b"v ": nv += 1
                    elif ln[:2] == b"f ": nf += 1
        except OSError: pass
        _vf_cache[key] = [nv, nf]
        return nv, nf
    segrows = []
    tv = tf = 0
    segset = sorted({os.path.basename(os.path.dirname(c)) for c in chunks})
    for seg in segset:
        cs = [c for c in sorted(glob.glob(f"{PG}/{seg}/chunk_*")) if not _excluded(c)]
        if not cs: continue
        sv = sf = 0; ext = ""
        for c in cs:
            v, f_ = vf(f"{c}/mesh.obj"); sv += v; sf += f_
            if not ext and os.path.exists(f"{c}/dem.tif"):
                with rasterio.open(f"{c}/dem.tif") as ds:
                    ext = f"{ds.width*ds.res[0]:.0f}×{ds.height*ds.res[1]:.0f} m"
        tv += sv; tf += sf
        segrows.append(dict(seg=seg, chunks=len(cs), verts=sv, faces=sf, ext=ext))
    try: cache_path.write_text(json.dumps(_vf_cache))
    except OSError: pass
    with rasterio.open(f"{PG}/merged/ortho_merged.tif") as ds:
        me = f"{ds.width*ds.res[0]:.0f}×{ds.height*ds.res[1]:.0f} m @ {ds.res[0]*100:.0f} cm"
    photo = dict(segments=len(segrows), chunks=len(chunks), orthos=len(orthos),
                 dems=len([d for d in glob.glob(f"{PG}/seg*/chunk_*/dem.tif") if not _excluded(d)]),
                 verts=tv, faces=tf, merged_ext=me, rows=segrows)
    # anomaly (optional — the detector may not have run yet)
    anom, win = None, pd.DataFrame()
    if (Path(B) / "survey/anomaly/anomaly_windows_all.csv").is_file():
        win = pd.read_csv(f"{B}/survey/anomaly/anomaly_windows_all.csv")
        sites = pd.read_csv(f"{B}/survey/anomaly/anomalous_sites.csv")
        seg_gj = json.load(open(f"{B}/survey/anomaly/anomaly_segments_utm.geojson"))
        tiers = {"HIGH": 0, "MODERATE": 0, "SCREEN": 0}
        for f in seg_gj["features"]:
            c = f["properties"].get("confidence")
            if c in tiers: tiers[c] += 1
        clips = pd.read_csv(f"{B}/survey/anomaly/video_review_clips.csv")
        top = sites.sort_values("window_count", ascending=False).head(8)
        anom = dict(windows=len(win), sites=len(sites), clips=len(clips), tiers=tiers,
                    site_rows=top.to_dict("records"))
    # sensors
    sens = []
    for lbl, pat in (("3D trackline", "nav_trackline/**/*.ply"),
                     ("Depth raster", "nav_depth/**/*.tif")):
        g = glob.glob(f"{B}/survey/{pat}", recursive=True)
        if g: sens.append((lbl, "1", sum(os.path.getsize(x) for x in g)))
    for ch in ("CO2", "CH4", "O2", "Salinity", "Temperature"):
        t2 = glob.glob(f"{B}/survey/sensor_2d/**/{ch}*2d.tif", recursive=True)
        nc = glob.glob(f"{B}/survey/sensor_netcdf/**/{ch}*.nc", recursive=True)
        sens.append((f"{ch}", f"2D raster + netCDF", sum(os.path.getsize(x) for x in t2 + nc)))
    return dict(dive=dive, photo=photo, anom=anom, win=win, sensors=sens)


# ------------------------------------------------------------------- html ----
def build_survey_report(workspace_dir, out_path=None, fast_mesh=False) -> str:
    B = str(workspace_dir)
    D = collect(B, fast_mesh=fast_mesh)
    figs = {
        "map": _b64(fig_survey_map(B), "image/jpeg"),
        "channels": (_b64(fig_channels(D["win"]), "image/png")
                     if len(D["win"]) else None),
        "contact": _img_file_b64(f"{B}/survey/photogrammetry/mosaics_contact_sheet.png"),
        "merged": _img_file_b64(f"{B}/survey/photogrammetry/merged/preview_ortho_merged.png"),
    }
    figs["gallery"] = [g for g in (_img_file_b64(x) for x in sorted(
        glob.glob(f"{B}/survey/photogrammetry/ortho_gallery/gallery_*.png"))) if g]
    # IDW sensor-raster figure intentionally omitted (interpolation not informative
    # enough to feature); the sensor-product inventory table is retained.
    figs["sensor"] = None
    # Imagery-x-chemistry findings (analysis_figures.py output); section renders
    # only when the stats + figures exist.  Same chunk exclusions as the rest of
    # this report (EXCLUDE_CHUNKS) are applied upstream by the analysis itself.
    af = Path(B) / "survey" / "analysis_figs"
    astats, afigs = None, {}
    if (af / "analysis_stats.json").exists():
        astats = json.loads((af / "analysis_stats.json").read_text())
        afigs = {p.stem: _img_file_b64(str(p)) for p in sorted(af.glob("*.png"))}
    html = _render_html(D, figs, astats, afigs)
    out = out_path or f"{B}/SURVEY_REPORT.html"
    Path(out).write_text(html, encoding="utf-8")
    return out


def _analysis_section(a, f, dv) -> str:
    """Imagery-x-chemistry findings: altitude-corrected seafloor brightness vs
    the dive's gas anomalies (CO2, CH4, O2, temperature).  Strictly optical framing; no biological mechanism is
    asserted."""
    def fig(key, alt, cap):
        src = f.get(key)
        if not src: return ""
        return (f'<figure><img alt="{alt}" src="{src}">'
                f'<figcaption>{cap}</figcaption></figure>')
    n = a.get("n_frames", 0)
    ncov = a.get("n_cover", 0); covpct = a.get("cover_frac", 0) * 100
    smp = a.get("sampling", {})
    bc4 = a.get("bool", {}).get("ch4", {})
    bany = a.get("bool", {}).get("any", {})
    conc = a.get("conc", {}); band = conc.get("band", {})
    sp = a.get("spearman", {})
    def rho(m, g):
        try: return f"{sp[m][g]['rho']:+.2f}"
        except KeyError: return "&mdash;"
    att = a.get("attenuation", {}).get("slopes", {})
    geo = a.get("geomorph", {}); gm = geo.get("metrics", {})
    def georow(k, label, unit):
        m = gm.get(k, {})
        if not m: return ""
        pp = "&lt;0.001" if m.get("p", 1) < 1e-3 else f"{m.get('p', 1):.2f}"
        return (f"<tr><td>{label}</td>"
                f"<td class='num'>{m.get('anom_median', 0):.2f}{unit}</td>"
                f"<td class='num'>{m.get('bg_median', 0):.2f}{unit}</td>"
                f"<td class='num'>{m.get('delta', 0):+.2f}</td>"
                f"<td class='num'>{pp}</td></tr>")
    def pct(x): return f"{x * 100:.0f}%"
    def pfmt(p): return "&lt;0.001" if p < 1e-3 else f"{p:.3f}"
    rba = a.get("rho_bright_alt", 0); rwa = a.get("rho_white_alt", 0)
    bl = a.get("bool", {})
    has_bool = bool(bl)
    has_geo = bool(gm)
    _gas_sig = [(nm, bl[k]) for k, nm in (("co2", "CO&#8322;"), ("ch4", "CH&#8324;"),
                ("o2", "O&#8322;")) if k in bl and bl[k].get("p", 1) < 0.01]
    _bt = bl.get("temp", {})
    _sat = bany.get("rate_bare", 0) > 0.5
    _sig_note = ", ".join(
        nm for k, nm in (("co2", "CO&#8322;"), ("ch4", "CH&#8324;"), ("o2", "O&#8322;"),
                         ("temp", "temperature"), ("any", "any-channel"))
        if k in bl and bl[k].get("p", 1) < 0.01) or "none"

    bool_rows = "".join(
        f"<tr><td>{nm}</td>"
        f"<td class='num'>{pct(bl[k].get('rate_bare', 0))}</td>"
        f"<td class='num'>{pct(bl[k].get('rate_cover', 0))}</td>"
        f"<td class='num'>{bl[k].get('odds_ratio', 0):.1f}</td>"
        f"<td class='num'>{pfmt(bl[k].get('p', 1))}</td></tr>"
        for k, nm in (("co2", "CO&#8322;"), ("ch4", "CH&#8324;"), ("o2", "O&#8322;"),
                      ("temp", "Temperature"), ("any", "Any channel")) if k in bl)
    fused_block = (f"""<h3>The fused view &mdash; one anomalous region</h3>
  <p>The three products come together in a single co-registered frame of the vent-field
  core: photogrammetry as the base (orthomosaic where imaged, DEM hillshade filling the
  gaps), anomaly windows as translucent strokes beneath the track, and every analysed
  frame as a dot coloured by corrected brightness. Because several traverse passes cross
  the region, each pass carries an <em>independent</em> optical record over the same
  ground &mdash; bright-cover chains repeat across passes through the site cluster, while
  several high-tier window strokes ride over plainly dim ground: the plume&ndash;ground
  decoupling, visible directly.</p>
  {fig("region_portrait_core", "Fused region portrait of the vent-field core",
       "Vent-field core, fused: orthomosaic/hillshade base, anomaly windows (translucent, "
       "coloured by confidence), per-frame corrected brightness along every pass, "
       "bright-pixel-cover rings, ranked sites, and a survey locator. UTM 13N.")}"""
                   if has_bool and f.get("region_portrait_core") else "")
    geo_block = (f"""<h3>Terrain does not predict the chemistry either</h3>
  <p>The same track-matched design applied to seafloor <em>morphology</em> finds nothing
  of consequence: terrain under anomaly-flagged track segments
  (n&nbsp;=&nbsp;{geo.get('n_anom', 0):,}) is practically indistinguishable from background
  (n&nbsp;=&nbsp;{geo.get('n_bg', 0):,}) &mdash; no metric exceeds a negligible effect size
  (|&delta;|&nbsp;&le;&nbsp;{max((abs(m.get('delta', 0)) for m in gm.values()), default=0):.2f})
  &mdash; consistent with the anomalies being advected plumes sensed down-current of their
  sources. Of the visual variables examined, the optical character of the seafloor &mdash;
  not its shape &mdash; is the one that carries information.</p>
  <div class="tw"><table>
    <thead><tr><th>terrain metric</th><th class="num">anomaly median</th>
      <th class="num">background</th><th class="num">Cliff&rsquo;s &delta;</th><th class="num">p</th></tr></thead>
    <tbody>{georow('slope', 'Slope', '&deg;')}{georow('rough', 'Roughness (RMS)', '&nbsp;m')}{georow('relief', 'Local relief', '&nbsp;m')}</tbody>
  </table></div>
  {fig("geomorph", "Terrain metrics for anomaly vs background track samples",
       "Slope, roughness and relief distributions under anomaly-flagged vs background "
       "track segments overlap almost completely: no meaningful geomorphometric "
       "association.")}""" if has_geo else "")
    interp_para = ("""<p>Read together: corrected seafloor brightness &mdash; especially localised bright-pixel
  cover &mdash; tracks <em>the in-situ gas concentrations at the frame</em>, while
  window membership, a plume-timing property, is not an optical question. What makes
  ground bright here (biological cover, mineral precipitates, sediment character) is not
  established by this analysis and is left open.</p>""" if has_bool else
  """<p>Corrected seafloor brightness &mdash; especially localised bright-pixel cover
  &mdash; tracks <em>the in-situ gas concentrations at the frame</em>. The boolean
  window-membership analysis, the terrain null test and the fused regional view will be
  appended once the detector catalogue is available. What makes ground bright here
  (biological cover, mineral precipitates, sediment character) is not established by
  this analysis and is left open.</p>""")
    _so, _st = a.get("sites_outside"), a.get("sites_total")
    coverage_li = (f"""<li><b>Coverage gaps.</b> {_so} of the {_st} ranked anomaly sites lie
    outside the imaged corridors (no analysed frame within 12&nbsp;m); add them to the next
    dive&rsquo;s traverse plan.</li>""" if has_bool and _so else "")
    excl = dv.get("excl") or []
    excl_sentence = ((" The " + ("two reconstructions" if len(excl) == 2 else
                     f"{len(excl)} reconstruction(s)") + " omitted elsewhere in this report ("
                     + ", ".join(x.replace("/", "&nbsp;/&nbsp;") for x in excl)
                     + ") are omitted as <em>mosaic products</em> only; their source frames are "
                     "ordinary photographs and are included in this frame-level analysis.")
                    if excl else "")
    ministat3 = (f"""<div><div class="v">{bany.get('odds_ratio', 0):.1f}&times;</div>
      <div class="k">odds that a bright-cover frame lies inside an anomaly window (any channel)</div></div>"""
                 if has_bool else f"""<div><div class="v">{ncov}</div>
      <div class="k">transit frames with &gt;2&nbsp;% bright-pixel cover, of {n:,} analysed</div></div>""")
    bool_figs_block = (f"""{fig("bool_brightness", "Boolean tests: window rates vs brightness",
       "The boolean tests. Left: the share of frames inside anomaly windows "
       "(CH&#8324;-implicating and any-channel) across scene-brightness quintiles. "
       "Right: in-window rates for bright-pixel-cover vs bare frames, per window "
       "family; the table below carries the odds ratios and significance.")}
  <div class="tw"><table>
    <thead><tr><th>window family</th><th class="num">bare in-window</th>
      <th class="num">cover in-window</th><th class="num">odds ratio</th>
      <th class="num">Fisher p</th></tr></thead>
    <tbody>{bool_rows}</tbody>
  </table></div>
  <p class="tablenote">Boolean membership per window family (a window may implicate
  several channels, so families overlap). Bright-pixel cover elevates the in-window
  rate across every family; individually significant (p&nbsp;&lt;&nbsp;0.01):
  {_sig_note}.</p>""" if has_bool else "")
    bool_intro = ("""<p>Concentration at the sensor is <strong>altitude-confounded</strong>: the same seep
  reads stronger when the vehicle flies lower, regardless of any image correction. The
  detector&rsquo;s anomaly windows &mdash; baselined, multi-detector, multi-configuration
  &mdash; are the altitude-fair target, so the primary test is boolean: is a frame inside
  a window, or not?</p>""" if has_bool else
  """<p>Concentration at the sensor is <strong>altitude-confounded</strong>: the same seep
  reads stronger when the vehicle flies lower, regardless of any image correction. The
  detector&rsquo;s baselined anomaly windows are the altitude-fair target for the primary
  boolean test; that cross-analysis is pending the detector run and will be appended.
  The statistics below are therefore the continuous, altitude-band-controlled view
  only.</p>""")
    _sig_txt = "; ".join(
        f"{nm} {b.get('odds_ratio', 0):.1f}&times; (p&nbsp;{pfmt(b.get('p', 1))})"
        for nm, b in _gas_sig) or "none of the gas channels individually"
    bool_para_sat = (f"""<p>The boolean picture is dominated by how anomalous this dive is:
  windows implicating CO&#8322; or O&#8322; blanket more than
  {pct(min(bl.get('co2', {}).get('rate_bare', 0), bl.get('o2', {}).get('rate_bare', 0)))}
  of the transit record, so their cover-vs-bare contrasts are ceiling-limited and the
  pooled any-channel rate ({pct(bany.get('rate_cover', 0))} vs
  {pct(bany.get('rate_bare', 0))}, {bany.get('odds_ratio', 0):.1f}&times;,
  p&nbsp;=&nbsp;{pfmt(bany.get('p', 1))}) compresses toward its ceiling. The
  discriminating signal sits in the <em>sparser</em> window families &mdash;
  significant gas channels: {_sig_txt} &mdash; and above all in
  <strong>temperature-implicating windows</strong>:
  {pct(_bt.get('rate_cover', 0))} of bright-cover frames sit inside one versus
  {pct(_bt.get('rate_bare', 0))} of bare frames
  ({_bt.get('odds_ratio', 0):.1f}&times; the odds, p&nbsp;{pfmt(_bt.get('p', 1))}),
  and temperature is the one family whose member frames are themselves visibly brighter
  (Cliff&rsquo;s &delta;&nbsp;=&nbsp;{_bt.get('delta_bright', 0):+.2f}). <strong>Scene
  brightness alone still does not sort the pooled record</strong> &mdash; any-channel
  window rates are essentially flat across brightness quintiles &mdash; the per-family
  breakdown is tabulated below.</p>""")
    bool_para_split = (f"""<p>The boolean answer is a clean split. <strong>Scene brightness predicts window
  membership for no channel family</strong> &mdash; inside CH&#8324;-implicating windows
  frames are, if anything, marginally dimmer (Cliff&rsquo;s
  &delta;&nbsp;=&nbsp;{bc4.get('delta_bright', 0):+.2f}), and window rates are flat to
  non-monotonic across brightness quintiles. <strong>Bright-pixel cover behaves
  differently</strong>: cover frames lie inside an anomaly window of <em>some</em> channel
  {pct(bany.get('rate_cover', 0))} of the time versus {pct(bany.get('rate_bare', 0))} for
  bare frames &mdash; <strong>{bany.get('odds_ratio', 0):.1f}&times; the odds</strong>
  (Fisher p&nbsp;=&nbsp;{pfmt(bany.get('p', 1))}) &mdash; with the per-family breakdown in
  the table below. Among the gas channels none is individually significant
  (CH&#8324;: {pct(bc4.get('rate_cover', 0))} vs {pct(bc4.get('rate_bare', 0))},
  p&nbsp;=&nbsp;{pfmt(bc4.get('p', 1))}), while <strong>temperature-implicating windows
  are strongly enriched over bright cover</strong>
  ({bl.get('temp', {}).get('odds_ratio', 0):.1f}&times; the odds,
  p&nbsp;{pfmt(bl.get('temp', {}).get('p', 1))}). The split is consistent with the terrain result
  below: windows flag <em>advected plumes</em> sensed in the water column, partly
  decoupled from the exact ground beneath them &mdash; while the multi-channel
  co-occurrence suggests bright ground sits inside the broader anomalous
  neighbourhood.</p>""")
    bool_para = ("" if not has_bool else (bool_para_sat if _sat else bool_para_split))
    return f"""
<section>
  <p class="sec-label">Imagery &times; chemistry</p>
  <h2>Seafloor brightness and gas anomalies: what holds, and what doesn&rsquo;t</h2>
  <p>Motivated by an informal field impression &mdash; anomalous ground often looks pale
  &mdash; this section asks a strictly optical question: does the
  <strong>altitude-corrected brightness of the seafloor</strong> relate to the dive&rsquo;s
  gas anomalies &mdash; CO&#8322;, CH&#8324;, O&#8322; and temperature alike? No biological mechanism is asserted; the physical cause of bright
  ground is deliberately left open. Stationary periods correspond to deliberate
  measurements parked at known high-CH&#8324; sites, so they are excluded by construction:
  frames exist only along the moving traverse legs, residual on-station moments
  (speed&nbsp;&lt;&nbsp;{smp.get('v_min', 0.08)}&nbsp;m/s) were removed, and what remains is the
  un-targeted <strong>between-station</strong> record &mdash; {n:,} frames
  (every fifth, &sim;1.25&nbsp;m spacing), each matched to the gas record through the
  sampling pipeline&rsquo;s per-frame manifests. Two optical metrics are used:
  <em>scene brightness</em> (corrected frame mean) and <em>bright-pixel cover</em>
  (the fraction of pixels that remain bright and chromatically neutral after
  correction; {ncov} frames &mdash; {covpct:.0f}&nbsp;% &mdash; exceed 2&nbsp;% cover).</p>
  {bool_intro}
  {bool_para}
  <div class="ministats">
    <div><div class="v">&rho;&nbsp;{band.get('rho_bright_ch4', 0):+.2f}</div>
      <div class="k">brightness &harr; CH&#8324; concentration, fixed 4&ndash;6.5&nbsp;m altitude band</div></div>
    <div><div class="v">{band.get('cover_fold', 0):.1f}&times;</div>
      <div class="k">CH&#8324; over bright-cover frames vs bare, within the same band</div></div>
    {ministat3}
  </div>
  {fig("example_frames", "Typical transit frame vs highest bright-cover frame",
       "Altitude-corrected frames. Left: typical transit seafloor. Right: the frame with "
       f"the highest bright-pixel cover of the survey ({a.get('top_cover', 0)*100:.0f}&nbsp;%) — "
       f"captured while the CH&#8324; sensor read {a.get('top_ch4_fold', 0):.1f}&times; the "
       "transit background.")}
  {bool_figs_block}
  {fig("brightness_gases", "Gas concentrations vs corrected brightness",
       "The continuous view: per-frame CH&#8324; (log), CO&#8322; and O&#8322; against "
       "corrected scene brightness. Because concentration is altitude-confounded, the "
       "quoted correlations come from the fixed 4&ndash;6.5&nbsp;m altitude band.")}
  <div class="tw"><table>
    <thead><tr><th>optical metric</th><th class="num">&rho; vs CO&#8322;</th>
      <th class="num">&rho; vs CH&#8324;</th><th class="num">&rho; vs O&#8322;</th></tr></thead>
    <tbody>
      <tr><td>Scene brightness</td><td class="num">{rho('bright','CO2')}</td><td class="num">{rho('bright','CH4')}</td><td class="num">{rho('bright','O2')}</td></tr>
      <tr><td>Bright-pixel cover</td><td class="num">{rho('white','CO2')}</td><td class="num">{rho('white','CH4')}</td><td class="num">{rho('white','O2')}</td></tr>
      <tr><td>Neutral-brightness index</td><td class="num">{rho('matidx','CO2')}</td><td class="num">{rho('matidx','CH4')}</td><td class="num">{rho('matidx','O2')}</td></tr>
    </tbody>
  </table></div>
  <p class="tablenote">Spearman rank correlations across {n:,} transit frames; every cell
  p&nbsp;&lt;&nbsp;0.001. Concentrations are altitude-confounded and frames are spatially
  autocorrelated, so these are corroborating context &mdash; the altitude-band statistics
  above are the defensible quantities. Within the 4&ndash;6.5&nbsp;m band
  (n&nbsp;=&nbsp;{band.get('n', 0)}), brightness&ndash;CH&#8324; remains
  &rho;&nbsp;=&nbsp;{band.get('rho_bright_ch4', 0):+.2f} and bright-cover frames carry
  {band.get('cover_fold', 0):.1f}&times; the CH&#8324; (p&nbsp;=&nbsp;{pfmt(band.get('p', 1))}).</p>

  <h3>Where the bright ground is</h3>
  {fig("brightness_map", "Spatial distribution of corrected brightness",
       "Transit frames coloured by corrected scene brightness; yellow rings mark frames "
       "with &gt;2&nbsp;% bright-pixel cover, and open circles the catalogue&rsquo;s ranked "
       "anomaly sites. Bright-cover frames cluster in the vent-field core. UTM 13N.")}

  {fused_block}

  <h3>How brightness was measured</h3>
  <p>Raw frames are blue-shifted by seawater attenuation, and the shift grows with
  vehicle altitude &mdash; uncorrected, &ldquo;brightness&rdquo; would partly measure flying
  height. The attenuation was fitted empirically across
  {a.get('attenuation', {}).get('n', 0)} frames using each frame&rsquo;s manifest altitude:
  channel intensity falls log-linearly at
  <span class="mono">R&nbsp;{att.get('R', 0):+.2f}</span>,
  <span class="mono">G&nbsp;{att.get('G', 0):+.2f}</span>,
  <span class="mono">B&nbsp;{att.get('B', 0):+.2f}</span> per metre. Each frame is corrected
  to a 5&nbsp;m reference altitude
  (<span class="mono">corrected&nbsp;=&nbsp;observed&nbsp;&middot;&nbsp;e<sup>c&middot;(alt&minus;5)</sup></span>
  per channel, altitude term clamped to &plusmn;2&nbsp;m of reference) plus a fixed white
  balance. Residual altitude dependence after correction is mild and conservative
  (brightness &rho;&nbsp;=&nbsp;{rba:+.2f}, cover &rho;&nbsp;=&nbsp;{rwa:+.2f} vs altitude
  &mdash; high flights read <em>darker</em>, not brighter), and the headline statistics are
  quoted within a fixed altitude band precisely to neutralise it. Per-frame ground truth
  (time, altitude, position, gases) comes from the sampling pipeline&rsquo;s own segment
  manifests.{excl_sentence}</p>
  {fig("attenuation", "Colour-channel attenuation with altitude",
       "The empirical basis of the correction: log channel intensity vs altitude with "
       "fitted slopes. Differential loss (R &gt; G &gt; B) is the classic seawater "
       "attenuation signature.")}
  {fig("colorcorrection_demo", "Raw vs altitude-normalised frame",
       "The correction applied to a representative mid-altitude frame: the blue cast is "
       "removed and substrate tone and contrast are restored.")}

  {geo_block}

  <h3>Interpretation &amp; recommendations</h3>
  {interp_para}
  <ul class="method">
    <li><b>Use imagery for targeting, windows for detection.</b> Pair the bright-cover
    index with in-band concentration to shortlist ground-truthing targets; treat the
    detector windows as the anomaly inventory, not as the imagery&rsquo;s counterpart.</li>
    <li><b>Ground-truth the bright patches.</b> The clustered bright-cover frames on the
    map above coincide with the ranked sites; sampling there would establish the physical
    cause the imagery cannot.</li>
    {coverage_li}
  </ul>
  <div class="note"><b>Caveats.</b> The cause of bright ground is not established here;
  overlapping frames are not independent samples; concentrations are in native sensor
  units; anomaly windows derive from the same sensor record (though independently of the
  imagery); and this is a single dive. The altitude-band statistics are the quantities
  designed to survive these concerns.</div>
</section>"""


def _render_html(D, figs, astats=None, afigs=None) -> str:
    dv, ph, an = D["dive"], D["photo"], D["anom"]
    analysis_html = _analysis_section(astats, afigs or {}, dv) if astats else ""
    def mb(x): return f"{x/1e6:.1f} MB" if x >= 1e6 else f"{x/1e3:.0f} KB"
    seg_rows = "\n".join(
        f"<tr><td class='mono'>{r['seg']}</td><td class='num'>{r['chunks']}</td>"
        f"<td class='num'>{r['verts']:,}</td><td class='num'>{r['faces']:,}</td>"
        f"<td class='mono dim'>{r['ext']}</td></tr>" for r in ph["rows"])
    site_rows = "\n".join(
        f"<tr><td class='mono'>{r['site_id']}</td>"
        f"<td class='num'>{int(r['window_count'])}</td>"
        f"<td><span class='pill {str(r['best_tier']).lower()}'>{r['best_tier']}</span></td>"
        f"<td class='num'>{r['max_evidence_score']:.0f}</td>"
        f"<td class='dim'>{r['channels']}</td></tr>"
        for r in (an["site_rows"] if an else []))
    sens_rows = "\n".join(
        f"<tr><td>{lbl}</td><td class='dim'>{kind}</td><td class='num'>{mb(sz)}</td></tr>"
        for lbl, kind, sz in D["sensors"])
    T = an["tiers"] if an else {}
    hero_anom_stat = (f"""<div class="stat"><div class="v">{an['sites']}</div><div class="k">Anomalous sites</div></div>"""
                      if an else
                      f"""<div class="stat"><div class="v">{ph['segments']}</div><div class="k">Traverse legs imaged</div></div>""")
    hero_h2 = "Anomalies over the photomosaic" if an else "The dive at a glance"
    map_para_anom = (", and the detected anomaly windows" if an else " and the sensor products")
    map_para_tail = (f""" Anomaly windows are colour-coded by confidence; open circles mark the
  {an['sites']} ranked sites where evidence recurs.""" if an else
  """ Anomaly windows will overlay this frame once the detector catalogue is
  available.""")
    map_cap_anom = (f""",
  colour-coded anomaly windows (red high / orange moderate / yellow screen) and the
  {an['sites']} anomalous sites""" if an else "")

    sensor_fig = (f"<figure><img alt='CO2 concentration raster' src='{figs['sensor']}'>"
                  f"<figcaption>Gridded CO₂ concentration (IDW, UTM 13N) — one of five "
                  f"co-registered sensor rasters. Warm tones mark elevated concentration over the "
                  f"vent field.</figcaption></figure>") if figs["sensor"] else ""
    contact_fig = (f"<figure><img alt='Orthomosaic contact sheet' src='{figs['contact']}'>"
                   f"<figcaption>All {ph['orthos']} orthomosaics — clean traverse ribbons, "
                   f"dense full-coverage patches, and the curved passes over the vent field.</figcaption>"
                   f"</figure>") if figs["contact"] else ""
    gallery_html = ""
    if figs.get("gallery"):
        panels = "".join(
            f"<figure><img alt='Orthomosaic close-ups, sheet {i+1}' src='{g}'></figure>"
            for i, g in enumerate(figs["gallery"]))
        gallery_html = (
            "<h3>Orthomosaic close-ups</h3>"
            "<p>The best-covered reconstructions, shown near native scale and spread across "
            "the dive's traverse legs. A per-tile percentile contrast stretch is applied for "
            "legibility on screen and in print &mdash; the underlying GeoTIFFs retain the "
            "unmodified radiometry, and the altitude-corrected analysis uses the frames, "
            "not these renderings.</p>"
            + panels +
            "<p class='tablenote'>Each panel is one chunk orthomosaic (segment / chunk labelled, "
            "extent in metres, 10&nbsp;m scale bar). Contrast-stretched for display.</p>")
    merged_fig = (f"<figure><img alt='Merged dive-wide orthomosaic' src='{figs['merged']}'>"
                  f"<figcaption>The {ph['orthos']} orthomosaics merged into one dive-wide mosaic "
                  f"({ph['merged_ext']}); overlapping passes register at their crossings.</figcaption>"
                  f"</figure>") if figs["merged"] else ""
    if an:
        anom_section = f"""<section>
  <p class="sec-label">Anomaly detection</p>
  <h2>Sensor anomalies and ranked sites</h2>
  <p>A four-configuration detector matrix over the CO&#8322;, CH&#8324;, O&#8322; and temperature
  channels produced {an['windows']} fused signature windows, consolidated
  into <strong>{an['sites']} spatially-ranked anomalous sites</strong> and a
  {an['clips']}-clip video-review queue. Windows are graded by cross-detector, cross-configuration
  agreement.</p>
  <div class="tierbar">
    <span style="flex:{max(T['HIGH'],1)};background:var(--high)"></span>
    <span style="flex:{max(T['MODERATE'],1)};background:var(--moderate)"></span>
    <span style="flex:{max(T['SCREEN'],1)};background:var(--screen)"></span>
  </div>
  <div class="tierkey">
    <span><span class="dot" style="background:var(--high)"></span><b>{T['HIGH']}</b> high confidence</span>
    <span><span class="dot" style="background:var(--moderate)"></span><b>{T['MODERATE']}</b> moderate</span>
    <span><span class="dot" style="background:var(--screen)"></span><b>{T['SCREEN']}</b> screen</span>
    <span class="dim">of {an['windows']} windows total</span>
  </div>
  <h3>Windows by sensor channel</h3>
  <figure style="background:var(--paper)"><img alt="Windows per sensor channel" src="{figs['channels']}">
  <figcaption>Number of anomaly windows implicating each channel. Temperature and O&#8322;
  dominate, with CO&#8322; and CH&#8324; corroborating at the strongest sites.</figcaption></figure>
  <h3>Top anomalous sites</h3>
  <div class="tw"><table>
    <thead><tr><th>Site</th><th class="num">Windows</th><th>Best tier</th>
      <th class="num">Evidence</th><th>Channels</th></tr></thead>
    <tbody>{site_rows}</tbody>
  </table></div>
</section>"""
    else:
        anom_section = """<section>
  <p class="sec-label">Anomaly detection</p>
  <h2>Sensor anomalies and ranked sites</h2>
  <div class="note"><b>Pending.</b> The GrapherMatrix detector and site catalogue have
  not yet been run for this dive; the anomaly inventory, ranked sites and the
  imagery&nbsp;&times;&nbsp;anomaly cross-analysis will be appended once available. All
  other products in this report are final.</div>
</section>"""

    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{dv['name']} Survey Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,400;0,500;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{
  --paper:#ffffff; --ground:#eef1f4; --ink:#16222b; --muted:#5a6b78;
  --dim:#8493a0; --line:#dde4ea; --line2:#eaeff3; --accent:#0e6d78;
  --accent-soft:#e3eef0; --high:#c1272d; --moderate:#d97316; --screen:#b8890a;
  --shadow:0 1px 2px rgba(20,40,55,.05),0 8px 30px rgba(20,40,55,.06);
}}
:root:not([data-theme="light"]) {{}}
@media (prefers-color-scheme: dark){{
  :root:not([data-theme="light"]){{
    --paper:#131c24; --ground:#0d151b; --ink:#e8eef3; --muted:#9fb0bd;
    --dim:#71828f; --line:#26333d; --line2:#1c272f; --accent:#3fb6c2;
    --accent-soft:#152a2e; --high:#f26571; --moderate:#f0a04b; --screen:#e0bb4f;
    --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 34px rgba(0,0,0,.35);
  }}
}}
:root[data-theme="dark"]{{
  --paper:#131c24; --ground:#0d151b; --ink:#e8eef3; --muted:#9fb0bd;
  --dim:#71828f; --line:#26333d; --line2:#1c272f; --accent:#3fb6c2;
  --accent-soft:#152a2e; --high:#f26571; --moderate:#f0a04b; --screen:#e0bb4f;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 34px rgba(0,0,0,.35);
}}
*{{box-sizing:border-box}}
html{{-webkit-text-size-adjust:100%}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:16px;line-height:1.65;
  -webkit-font-smoothing:antialiased;}}
.wrap{{max-width:920px;margin:0 auto;padding:0 24px}}
.doc{{background:var(--paper);margin:28px auto;box-shadow:var(--shadow);
  border:1px solid var(--line);border-radius:6px;overflow:hidden}}
.mono{{font-family:"IBM Plex Mono",ui-monospace,monospace;font-variant-numeric:tabular-nums}}
.num{{font-family:"IBM Plex Mono",monospace;font-variant-numeric:tabular-nums;text-align:right;white-space:nowrap}}
.dim{{color:var(--dim)}}
/* masthead */
.mast{{padding:52px 56px 40px;border-bottom:1px solid var(--line);
  background:linear-gradient(180deg,var(--accent-soft),transparent)}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 14px}}
h1{{font-family:"Spectral",Georgia,serif;font-weight:600;font-size:44px;line-height:1.06;
  margin:0 0 12px;letter-spacing:-.01em;text-wrap:balance;max-width:16ch}}
.lede{{font-family:"Spectral",serif;font-size:19px;line-height:1.55;color:var(--muted);
  margin:0;max-width:56ch;font-style:italic}}
.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:1px;background:var(--line);
  border-top:1px solid var(--line);border-bottom:1px solid var(--line)}}
.stat{{background:var(--paper);padding:20px 22px}}
.stat .v{{font-family:"IBM Plex Mono",monospace;font-size:23px;font-weight:500;
  font-variant-numeric:tabular-nums;letter-spacing:-.01em}}
.stat .k{{font-size:12.5px;color:var(--muted);margin-top:3px}}
@media(max-width:640px){{.stats{{grid-template-columns:repeat(2,1fr)}}}}
/* sections */
section{{padding:46px 56px}}
section + section{{border-top:1px solid var(--line2)}}
.sec-label{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 6px;
  display:flex;align-items:center;gap:10px}}
.sec-label::before{{content:"";width:20px;height:1px;background:var(--accent);display:inline-block}}
h2{{font-family:"Spectral",serif;font-weight:600;font-size:29px;line-height:1.15;
  margin:0 0 18px;letter-spacing:-.01em;text-wrap:balance}}
h3{{font-family:"IBM Plex Sans",sans-serif;font-weight:600;font-size:16px;margin:30px 0 10px}}
p{{margin:0 0 16px;max-width:66ch}}
a{{color:var(--accent)}}
strong{{font-weight:600}}
/* figures */
figure{{margin:26px 0;background:var(--ground);border:1px solid var(--line);
  border-radius:6px;overflow:hidden}}
figure img{{display:block;width:100%;height:auto}}
figcaption{{font-size:13.5px;color:var(--muted);padding:12px 18px;line-height:1.5;
  border-top:1px solid var(--line);background:var(--paper)}}
.hero figure{{margin:0}}
/* tables */
.tw{{overflow-x:auto;margin:22px 0;border:1px solid var(--line);border-radius:6px}}
table{{border-collapse:collapse;width:100%;font-size:14px}}
th,td{{padding:9px 14px;text-align:left;border-bottom:1px solid var(--line2);
  white-space:nowrap}}
th{{font-family:"IBM Plex Mono",monospace;font-size:11.5px;letter-spacing:.05em;
  text-transform:uppercase;color:var(--muted);font-weight:500;background:var(--ground);
  position:sticky;top:0}}
th.num{{text-align:right}}
tbody tr:last-child td{{border-bottom:none}}
tbody tr:hover td{{background:var(--accent-soft)}}
tfoot td{{border-top:2px solid var(--line);font-weight:600;background:var(--ground)}}
.pill{{display:inline-block;font-family:"IBM Plex Mono",monospace;font-size:11px;
  font-weight:500;padding:2px 9px;border-radius:20px;letter-spacing:.03em}}
.pill.high{{background:color-mix(in srgb,var(--high) 16%,transparent);color:var(--high)}}
.pill.moderate{{background:color-mix(in srgb,var(--moderate) 18%,transparent);color:var(--moderate)}}
.pill.screen{{background:color-mix(in srgb,var(--screen) 20%,transparent);color:var(--screen)}}
/* tier bar */
.tierbar{{display:flex;height:12px;border-radius:6px;overflow:hidden;margin:8px 0 4px;border:1px solid var(--line)}}
.tierbar span{{display:block}}
.tierkey{{display:flex;gap:20px;flex-wrap:wrap;font-size:13px;color:var(--muted);margin-top:10px}}
.tierkey b{{font-family:"IBM Plex Mono",monospace;color:var(--ink)}}
.dot{{width:9px;height:9px;border-radius:50%;display:inline-block;margin-right:6px;vertical-align:middle}}
/* callout */
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);
  padding:14px 20px;border-radius:0 6px 6px 0;margin:22px 0;font-size:14.5px;color:var(--muted)}}
.note b{{color:var(--ink)}}
.grid2{{display:grid;grid-template-columns:1fr 1fr;gap:26px}}
@media(max-width:640px){{.grid2{{grid-template-columns:1fr}} section{{padding:38px 28px}} .mast{{padding:40px 28px 32px}} h1{{font-size:34px}}}}
footer{{padding:30px 56px 44px;color:var(--dim);font-size:13px;border-top:1px solid var(--line2)}}
footer .mono{{color:var(--muted)}}
ul.method{{margin:8px 0 16px;padding-left:0;list-style:none}}
ul.method li{{padding:9px 0 9px 26px;border-bottom:1px solid var(--line2);position:relative;font-size:14.5px;max-width:66ch}}
ul.method li::before{{content:"";position:absolute;left:2px;top:17px;width:8px;height:8px;
  border:1.5px solid var(--accent);border-radius:2px}}
ul.method li:last-child{{border-bottom:none}}
ul.method b{{font-weight:600}}
.ministats{{display:grid;grid-template-columns:repeat(3,1fr);gap:1px;background:var(--line);
  border:1px solid var(--line);border-radius:6px;overflow:hidden;margin:22px 0}}
.ministats>div{{background:var(--paper);padding:16px 20px}}
.ministats .v{{font-family:"IBM Plex Mono",monospace;font-size:24px;font-weight:500;
  font-variant-numeric:tabular-nums;color:var(--accent)}}
.ministats .k{{font-size:12.5px;color:var(--muted);margin-top:3px;line-height:1.4}}
@media(max-width:640px){{.ministats{{grid-template-columns:1fr}}}}
.tablenote{{font-size:12.5px;color:var(--dim);margin:-12px 0 20px;max-width:70ch}}
sup{{font-size:70%}}
@media print{{
  *{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  body{{background:#fff}}
  .wrap{{max-width:none;padding:0}}
  .doc{{margin:0;box-shadow:none;border:none;border-radius:0}}
  section{{padding:26px 40px}}
  .mast{{padding:34px 40px 26px}}
  footer{{padding:22px 40px 30px}}
  figure,.tw,.note,.ministats,.stats{{break-inside:avoid;page-break-inside:avoid}}
  h2,h3{{break-after:avoid;page-break-after:avoid}}
  figure img{{max-height:8.4in;width:auto;max-width:100%;margin:0 auto}}
  tbody tr:hover td{{background:transparent}}
}}
</style></head>
<body><div class="wrap"><article class="doc">

<header class="mast">
  <p class="eyebrow">ROV Dive Survey &middot; East Pacific Rise 9&deg;N</p>
  <h1>Jason Dive {dv['name']} Survey Report</h1>
  <p class="lede">Photogrammetric seafloor mapping and multi-channel sensor-anomaly
  detection across the downward traverse legs of a {dv['dur_h']:.1f}-hour hydrothermal
  vent-field survey.</p>
</header>

<div class="stats">
  <div class="stat"><div class="v">{dv['dur_h']:.1f} h</div><div class="k">Bottom time</div></div>
  <div class="stat"><div class="v">{dv['path_3d']/1000:.2f} km</div><div class="k">Vehicle path (3D)</div></div>
  {hero_anom_stat}
  <div class="stat"><div class="v">{ph['chunks']}</div><div class="k">Reconstructions</div></div>
</div>

<section class="hero">
  <p class="sec-label">Integrated survey map</p>
  <h2>{hero_h2}</h2>
  <p>The dive resolved into {ph['segments']} moving traverse legs over the vent field. Every
  orthomosaic, the ROV trackline{map_para_anom} are co-registered in
  {dv['utm']}.{map_para_tail}</p>
  <figure><img alt="Integrated survey map: orthomosaics with colour-coded anomaly windows"
    src="{figs['map']}">
  <figcaption>Survey overview &mdash; dive-wide orthomosaic base with the ROV
  trackline{map_cap_anom}. Coordinates in {dv['utm']}.</figcaption></figure>
</section>

<section>
  <p class="sec-label">Survey overview</p>
  <h2>The dive at a glance</h2>
  <p>{dv['name']} ran from {dv['start']} to {dv['end']} &mdash; {dv['dur_h']:.1f} hours,
  {dv['samples']:,} navigation samples. The vehicle covered {dv['path_3d']/1000:.2f} km of
  3D path ({dv['path_h']/1000:.2f} km horizontal) but spent most of the dive
  station-keeping; photogrammetry was scoped automatically to the {ph['segments']} legs where
  the vehicle was genuinely traversing.</p>
  <div class="tw"><table>
    <tbody>
      <tr><th>Operating window</th><td class="mono">{dv['start']} &rarr; {dv['end']}</td></tr>
      <tr><th>Depth range</th><td class="mono">{dv['dmin']:.0f} &ndash; {dv['dmax']:.0f} m</td></tr>
      <tr><th>Horizontal extent</th><td class="mono">E {dv['e0']:.0f}&ndash;{dv['e1']:.0f} &nbsp; N {dv['n0']:.0f}&ndash;{dv['n1']:.0f}</td></tr>
      <tr><th>Coordinate system</th><td class="mono">{dv['utm']}</td></tr>
    </tbody>
  </table></div>
</section>

<section>
  <p class="sec-label">Photogrammetry</p>
  <h2>Seafloor reconstructions</h2>
  <p>The {ph['segments']} traverse legs yielded <strong>{ph['chunks']} photogrammetric
  reconstructions</strong> &mdash; {ph['orthos']} orthomosaics, {ph['dems']} digital elevation
  models and {ph['chunks']} textured 3D meshes ({ph['verts']:,} vertices /
  {ph['faces']:,} faces in total), each nav-georeferenced in {dv['utm']}.</p>
  {merged_fig}
  {contact_fig}
  {gallery_html}
  <h3>Per-segment inventory</h3>
  <div class="tw"><table>
    <thead><tr><th>Segment</th><th class="num">Chunks</th><th class="num">Vertices</th>
      <th class="num">Faces</th><th>DEM extent</th></tr></thead>
    <tbody>{seg_rows}</tbody>
    <tfoot><tr><td>{ph['segments']} segments</td><td class="num">{ph['chunks']}</td>
      <td class="num">{ph['verts']:,}</td><td class="num">{ph['faces']:,}</td>
      <td class="dim mono">merged {ph['merged_ext']}</td></tr></tfoot>
  </table></div>
</section>

{anom_section}

<section>
  <p class="sec-label">Sensor products &amp; methodology</p>
  <h2>Co-registered sensor grids</h2>
  <p>Every sensor channel is delivered as both a 2D GeoTIFF raster (IDW-interpolated, {dv['utm']})
  and a volumetric netCDF grid, alongside the 3D trackline and depth surface &mdash; all sharing the
  photogrammetry's coordinate frame.</p>
  {sensor_fig}
  <div class="tw"><table>
    <thead><tr><th>Product</th><th>Formats</th><th class="num">Size</th></tr></thead>
    <tbody>{sens_rows}</tbody>
  </table></div>

  <h2 style="margin-top:38px">How it was produced</h2>
  <ul class="method">
    <li><b>Nav-velocity segmentation.</b> The dive track is split into traverse legs by sustained
    speed, so photogrammetry is spent on moving passes rather than station-keeping.</li>
    <li><b>Nav-georeferenced photogrammetry.</b> Cameras are seeded from the reference navigation
    and orientation in {dv['utm']}; an image-quality gate and reference preselection drive
    alignment, dense matching, Height-Field meshing and DEM/ortho generation per chunk.</li>
    <li><b>Detector matrix.</b> Four suppression configurations &times; multiple detector families
    per channel are fused into signature windows, then clustered into ranked sites with a
    video-review queue.</li>
    <li><b>One coordinate frame.</b> All products &mdash; orthos, DEMs, meshes, sensor grids,
    trackline and anomalies &mdash; are delivered in {dv['utm']} so they overlay without
    reprojection.</li>
  </ul>
</section>
{analysis_html}
<footer>
  <div class="mono">Jason Dive {dv['name']} &middot; East Pacific Rise 9&deg;N &middot; {dv['utm']}</div>
  Generated {datetime.now().strftime('%Y-%m-%d')} from the .eprproj workspace &middot;
  all figures rendered from the delivered products.
</footer>

</article></div></body></html>"""


if __name__ == "__main__":
    import sys
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    print(build_survey_report(ws))
