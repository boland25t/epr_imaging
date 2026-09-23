#!/usr/bin/env python3
"""
project_report.py — assemble a single, self-contained HTML report that
summarises an entire EPR imaging dive project (an ``.eprproj`` bundle).

The report inventories everything a dive produced:

  1. Dive overview        — timing, path length, depth, UTM extent (from interp).
  2. Photogrammetry       — per-chunk cameras / mesh / DEM geometry / ortho.
  3. Sensor products      — trackline, depth raster, per-channel 2D + netCDF.
  4. Anomaly detection    — event counts per channel per matrix config.
  5. Ortho previews       — any preview_ortho.png embedded inline.

Design constraints (all honoured here):

  * Qt-free and importable.  ``build_report(workspace_dir, out_path=None) -> str``
    returns the path of the HTML file it wrote.
  * Paths come from ``workspace_paths.PathResolver`` wherever it exposes them.
  * OBJ vertex/face counts are STREAMED — meshes reach ~750 MB and are never
    read whole into memory.
  * Every file access is guarded; a missing/optional product renders as "—"
    instead of raising.

CLI:  ``python3 project_report.py <workspace_dir> [out_path]``
"""

from __future__ import annotations

import html
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

# Optional heavy deps — imported lazily-ish but at module load so failures are
# obvious.  Each use site still guards against per-file errors.
try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

try:
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:
    import rasterio
except Exception:  # pragma: no cover
    rasterio = None

from workspace_paths import PathResolver

CHANNELS_2D = ["CO2 Concentration", "CH4 Concentration", "O2 Concentration",
               "Salinity", "Temperature"]

ANOMALY_CONFIGS = ["masked", "nomask_log", "nomask_ceiling", "nomask_both"]
ANOMALY_CHANNELS = ["CO2", "CH4", "O2", "temp"]
ANOMALY_KINDS = [
    ("anomaly_events", "Anomaly events"),
    ("fine_core_events", "Fine core events"),
    ("fine_method_events", "Fine method events"),
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _human_size(n: Optional[int]) -> str:
    if n is None:
        return "—"
    size = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{size:.1f} TB"


def _size_of(path: Optional[Path]) -> Optional[int]:
    try:
        if path and Path(path).is_file():
            return Path(path).stat().st_size
    except OSError:
        pass
    return None


def _fmt(v, nd=1, dash="—"):
    if v is None:
        return dash
    try:
        if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
            return dash
    except (TypeError, ValueError):
        pass
    if isinstance(v, float):
        return f"{v:,.{nd}f}"
    if isinstance(v, int):
        return f"{v:,}"
    return str(v)


def _esc(s) -> str:
    return html.escape("" if s is None else str(s))


# ---------------------------------------------------------------------------
# 1. Dive overview (interp_full.csv)
# ---------------------------------------------------------------------------
def summarize_interp(interp_path: Path) -> dict:
    """Timing, path length, depth and UTM extent from the interp CSV."""
    out: dict = {"ok": False, "path": str(interp_path)}
    if pd is None or not Path(interp_path).is_file():
        return out
    try:
        df = pd.read_csv(interp_path)
    except Exception as exc:  # noqa: BLE001 - report, don't crash
        out["error"] = str(exc)
        return out

    out["ok"] = True
    out["n_samples"] = int(len(df))

    # -- timing -----------------------------------------------------------
    start = end = dur_h = None
    if "timestamp_iso" in df.columns and len(df):
        try:
            ts = pd.to_datetime(df["timestamp_iso"], errors="coerce").dropna()
            if len(ts):
                start, end = ts.iloc[0], ts.iloc[-1]
                dur_h = (end - start).total_seconds() / 3600.0
        except Exception:
            pass
    if start is None and "unix_time" in df.columns and len(df):
        try:
            ut = pd.to_numeric(df["unix_time"], errors="coerce").dropna()
            if len(ut):
                start = datetime.fromtimestamp(ut.iloc[0], tz=timezone.utc)
                end = datetime.fromtimestamp(ut.iloc[-1], tz=timezone.utc)
                dur_h = (ut.iloc[-1] - ut.iloc[0]) / 3600.0
        except Exception:
            pass
    out["start"] = str(start) if start is not None else None
    out["end"] = str(end) if end is not None else None
    out["duration_h"] = float(dur_h) if dur_h is not None else None

    # -- path length ------------------------------------------------------
    def _num(col):
        return pd.to_numeric(df[col], errors="coerce") if col in df.columns else None

    e, n = _num("easting"), _num("northing")
    d = _num("depth")
    len_3d = len_h = None
    if e is not None and n is not None and len(df) > 1:
        de = e.diff()
        dn = n.diff()
        horiz = np.sqrt(de ** 2 + dn ** 2)
        len_h = float(np.nansum(horiz))
        if d is not None:
            dd = d.diff()
            len_3d = float(np.nansum(np.sqrt(de ** 2 + dn ** 2 + dd ** 2)))
        else:
            len_3d = len_h
    out["path_len_3d_m"] = len_3d
    out["path_len_h_m"] = len_h

    # -- depth ------------------------------------------------------------
    if d is not None and d.notna().any():
        out["depth_min"] = float(d.min())
        out["depth_max"] = float(d.max())
    else:
        out["depth_min"] = out["depth_max"] = None

    # -- UTM zone + bounding box -----------------------------------------
    zone = None
    if "utm_zone" in df.columns and df["utm_zone"].notna().any():
        try:
            zone = str(df["utm_zone"].dropna().mode().iloc[0])
        except Exception:
            zone = str(df["utm_zone"].dropna().iloc[0])
    out["utm_zone"] = zone
    if e is not None and e.notna().any():
        out["easting_min"] = float(e.min())
        out["easting_max"] = float(e.max())
    if n is not None and n.notna().any():
        out["northing_min"] = float(n.min())
        out["northing_max"] = float(n.max())
    return out


# ---------------------------------------------------------------------------
# 2. Photogrammetry inventory
# ---------------------------------------------------------------------------
def count_obj_verts_faces(obj_path: Path) -> tuple[Optional[int], Optional[int]]:
    """Stream an OBJ counting ``v `` (vertex) and ``f `` (face) lines.

    Never reads the whole file — meshes reach ~750 MB.  Works on binary blocks,
    carrying a remainder across block boundaries, and matches only lines that
    begin exactly with ``v ``/``f `` (so ``vn ``/``vt ``/``vp `` are excluded).
    """
    if not Path(obj_path).is_file():
        return None, None
    verts = faces = 0
    remainder = b""            # partial line carried across block boundaries
    block = 4 * 1024 * 1024
    try:
        with open(obj_path, "rb") as fh:
            while True:
                data = fh.read(block)
                if not data:
                    break
                buf = remainder + data
                start = 0
                nl = buf.find(b"\n")
                while nl != -1:
                    # A line split on "\n" always begins at a real line start.
                    two = buf[start:start + 2]
                    if two == b"v ":
                        verts += 1
                    elif two == b"f ":
                        faces += 1
                    start = nl + 1
                    nl = buf.find(b"\n", start)
                remainder = buf[start:]
            # final line if the file did not end in a newline
            if remainder[:2] == b"v ":
                verts += 1
            elif remainder[:2] == b"f ":
                faces += 1
    except OSError:
        return None, None
    return verts, faces


def dem_geometry(dem_path: Path, max_samples: int = 300) -> dict:
    """Real-world DEM extent + plane-fit tilt and roughness (RMS to plane)."""
    res: dict = {"ok": False}
    if rasterio is None or np is None or not Path(dem_path).is_file():
        return res
    try:
        with rasterio.open(dem_path) as ds:
            resx, resy = ds.res
            res["extent_x_m"] = float(ds.width * resx)
            res["extent_y_m"] = float(ds.height * resy)
            res["width_px"] = int(ds.width)
            res["height_px"] = int(ds.height)

            # decimated read for the plane fit (keeps memory tiny)
            oh = min(max_samples, ds.height)
            ow = min(max_samples, ds.width)
            arr = ds.read(1, out_shape=(oh, ow), masked=True)
            left, bottom, right, top = ds.bounds
            xs = np.linspace(left, right, ow)
            ys = np.linspace(top, bottom, oh)
            gx, gy = np.meshgrid(xs, ys)

            z = np.ma.getdata(arr).astype("float64")
            mask = np.ma.getmaskarray(arr)
            nod = ds.nodata
            valid = ~mask & np.isfinite(z)
            if nod is not None:
                valid &= (z != nod)
            xf = gx[valid]
            yf = gy[valid]
            zf = z[valid]
            res["ok"] = True
            if zf.size >= 8:
                # centre coords so the fit is well-conditioned
                x0, y0 = xf.mean(), yf.mean()
                A = np.column_stack([xf - x0, yf - y0, np.ones_like(zf)])
                coef, *_ = np.linalg.lstsq(A, zf, rcond=None)
                a, b, _c = coef
                res["tilt_deg"] = float(math.degrees(math.atan(math.hypot(a, b))))
                resid = zf - A @ coef
                res["rms_m"] = float(np.std(resid))
                res["z_min"] = float(zf.min())
                res["z_max"] = float(zf.max())
    except Exception as exc:  # noqa: BLE001
        res["error"] = str(exc)
    return res


def _cameras_count(cam_path: Path) -> Optional[int]:
    import json
    if not Path(cam_path).is_file():
        return None
    try:
        d = json.loads(Path(cam_path).read_text(encoding="utf-8"))
    except Exception:
        return None
    if isinstance(d, dict):
        # top-level dict of camera-id -> pose, or {"cameras": [...]}
        if "cameras" in d and isinstance(d["cameras"], (list, dict)):
            return len(d["cameras"])
        return len(d)
    if isinstance(d, list):
        return len(d)
    return None


def inventory_photogrammetry(photo_root: Path) -> dict:
    """Walk segNN/chunk_CC, gathering per-chunk product geometry."""
    out: dict = {"root": str(photo_root), "segments": [], "chunks": [],
                 "missing_ortho": []}
    if not Path(photo_root).is_dir():
        return out
    segs = sorted([p for p in Path(photo_root).iterdir()
                   if p.is_dir() and p.name.startswith("seg")])
    out["segments"] = [s.name for s in segs]
    for seg in segs:
        chunks = sorted([p for p in seg.iterdir()
                         if p.is_dir() and p.name.startswith("chunk_")])
        for ch in chunks:
            rec: dict = {"segment": seg.name, "chunk": ch.name}
            rec["cameras"] = _cameras_count(ch / "cameras.json")

            v, f = count_obj_verts_faces(ch / "mesh.obj")
            rec["verts"], rec["faces"] = v, f

            rec["dem"] = dem_geometry(ch / "dem.tif")

            ortho = ch / "orthomosaic.tif"
            osize = _size_of(ortho)
            rec["ortho_present"] = bool(osize and osize > 0)
            rec["ortho_size"] = osize
            if not rec["ortho_present"]:
                out["missing_ortho"].append(f"{seg.name}/{ch.name}")

            # a preview image if the pipeline dropped one
            preview = ch / "preview_ortho.png"
            rec["preview"] = str(preview) if preview.is_file() else None
            out["chunks"].append(rec)
    return out


# ---------------------------------------------------------------------------
# 3. Sensor products
# ---------------------------------------------------------------------------
def _first_run_file(base: Path, filename_glob: str) -> Optional[Path]:
    """Find run_*/<glob> under base, newest run first."""
    if not Path(base).is_dir():
        return None
    runs = sorted([p for p in Path(base).iterdir()
                   if p.is_dir() and p.name.startswith("run_")], reverse=True)
    for r in runs:
        hits = sorted(r.glob(filename_glob))
        if hits:
            return hits[0]
    # some products keep the file directly under base
    hits = sorted(Path(base).glob(filename_glob))
    return hits[0] if hits else None


def inventory_sensors(survey_root: Path) -> dict:
    out: dict = {}
    survey = Path(survey_root)

    trk = _first_run_file(survey / "nav_trackline" / "nav_trackline",
                          "nav_trackline.ply")
    out["trackline"] = {"path": str(trk) if trk else None,
                        "size": _size_of(trk)}

    depth = _first_run_file(survey / "nav_depth" / "nav_depth", "nav_depth.tif")
    out["nav_depth"] = {"path": str(depth) if depth else None,
                        "size": _size_of(depth)}

    channels = []
    for ch in CHANNELS_2D:
        r2d = _first_run_file(survey / "sensor_2d" / "sensor_2d" / ch, "*_2d.tif")
        nc = _first_run_file(survey / "sensor_netcdf" / "sensor_netcdf" / ch, "*.nc")
        channels.append({
            "channel": ch,
            "raster_2d": str(r2d) if r2d else None,
            "raster_2d_size": _size_of(r2d),
            "netcdf": str(nc) if nc else None,
            "netcdf_size": _size_of(nc),
        })
    out["channels"] = channels
    return out


# ---------------------------------------------------------------------------
# 4. Anomaly detection
# ---------------------------------------------------------------------------
def _csv_event_count(path: Path) -> Optional[int]:
    """Number of data rows in an events CSV (header excluded); None if absent."""
    if not Path(path).is_file():
        return None
    try:
        n = 0
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh):
                if i == 0:
                    continue  # header
                if line.strip():
                    n += 1
        return n
    except OSError:
        return None


def inventory_anomalies(anomaly_figs_dir: Path) -> dict:
    """Event counts per config / channel / kind, plus a grand total."""
    out: dict = {"root": str(anomaly_figs_dir), "configs": {}, "total": 0,
                 "by_config": {}}
    base = Path(anomaly_figs_dir)
    total = 0
    for cfg in ANOMALY_CONFIGS:
        cfg_dir = base / cfg
        if not cfg_dir.is_dir():
            continue
        cfg_rec: dict = {}
        cfg_total = 0
        for kind_key, _label in ANOMALY_KINDS:
            row = {}
            for ch in ANOMALY_CHANNELS:
                cnt = _csv_event_count(cfg_dir / f"{kind_key}_{ch}.csv")
                row[ch] = cnt
                if cnt:
                    cfg_total += cnt
                    total += cnt
            cfg_rec[kind_key] = row
        out["configs"][cfg] = cfg_rec
        out["by_config"][cfg] = cfg_total
    out["total"] = total
    return out


# ---------------------------------------------------------------------------
# HTML rendering
# ---------------------------------------------------------------------------
CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body { margin: 0; background: #f4f6f8; color: #1c2833;
       font: 14px/1.5 -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif; }
.wrap { max-width: 1160px; margin: 0 auto; padding: 28px 22px 80px; }
h1 { font-size: 26px; margin: 0 0 4px; }
h2 { font-size: 19px; margin: 38px 0 12px; padding-bottom: 6px;
     border-bottom: 2px solid #d6dde3; }
.sub { color: #6b7a86; margin: 0 0 22px; font-size: 13px; }
.cards { display: flex; flex-wrap: wrap; gap: 14px; margin: 18px 0 6px; }
.card { background: #fff; border: 1px solid #e2e8ee; border-radius: 10px;
        padding: 14px 18px; min-width: 150px; flex: 1 1 150px;
        box-shadow: 0 1px 2px rgba(20,40,60,.04); }
.card .big { font-size: 24px; font-weight: 700; color: #14507a; }
.card .lbl { font-size: 12px; color: #6b7a86; text-transform: uppercase;
             letter-spacing: .04em; margin-top: 3px; }
table { border-collapse: collapse; width: 100%; background: #fff;
        border: 1px solid #e2e8ee; border-radius: 8px; overflow: hidden;
        font-size: 13px; }
.scroll { overflow-x: auto; }
th, td { padding: 7px 10px; text-align: right; border-bottom: 1px solid #eef2f5;
         white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
thead th { background: #eef3f7; color: #33505f; font-weight: 600;
           position: sticky; top: 0; }
tbody tr:nth-child(even) { background: #fafcfd; }
tbody tr:hover { background: #f0f7ff; }
.miss { color: #c0392b; font-weight: 600; }
.ok { color: #1e8449; }
.dim { color: #9aa7b1; }
.flag { display: inline-block; background: #fdecea; color: #c0392b;
        border: 1px solid #f5b7b1; border-radius: 6px; padding: 8px 12px;
        margin: 8px 0; font-size: 13px; }
.note { background: #fff8e1; border: 1px solid #ffe082; border-radius: 6px;
        padding: 8px 12px; font-size: 13px; margin: 10px 0; }
.gallery { display: flex; flex-wrap: wrap; gap: 16px; }
.gallery figure { margin: 0; background: #fff; border: 1px solid #e2e8ee;
                  border-radius: 8px; padding: 8px; max-width: 340px; }
.gallery img { display: block; border-radius: 4px; }
.gallery figcaption { font-size: 12px; color: #6b7a86; margin-top: 6px; }
footer { margin-top: 50px; color: #9aa7b1; font-size: 12px;
         border-top: 1px solid #d6dde3; padding-top: 14px; }
code { background: #eef2f5; padding: 1px 5px; border-radius: 4px; font-size: 12px; }
"""


def _card(big, lbl) -> str:
    return f'<div class="card"><div class="big">{big}</div><div class="lbl">{_esc(lbl)}</div></div>'


def _render_overview(o: dict) -> str:
    if not o.get("ok"):
        return ('<h2>1 · Dive overview</h2>'
                f'<div class="flag">interp not available: '
                f'{_esc(o.get("error") or o.get("path"))}</div>')
    cards = "".join([
        _card(_fmt(o.get("duration_h"), 2) + " h", "Duration"),
        _card(_fmt(o.get("n_samples"), 0), "Samples"),
        _card(_fmt(o.get("path_len_3d_m"), 0) + " m", "3D path length"),
        _card(_fmt(o.get("path_len_h_m"), 0) + " m", "Horizontal path"),
        _card(_esc(o.get("utm_zone") or "—"), "UTM zone"),
    ])
    rows = [
        ("UTC start", _esc(o.get("start"))),
        ("UTC end", _esc(o.get("end"))),
        ("Duration (h)", _fmt(o.get("duration_h"), 3)),
        ("Samples", _fmt(o.get("n_samples"), 0)),
        ("3D path length (m)", _fmt(o.get("path_len_3d_m"), 1)),
        ("Horizontal path length (m)", _fmt(o.get("path_len_h_m"), 1)),
        ("Depth range (m)",
         f'{_fmt(o.get("depth_min"),1)} – {_fmt(o.get("depth_max"),1)}'),
        ("UTM zone", _esc(o.get("utm_zone") or "—")),
        ("Easting bbox (m)",
         f'{_fmt(o.get("easting_min"),1)} – {_fmt(o.get("easting_max"),1)}'),
        ("Northing bbox (m)",
         f'{_fmt(o.get("northing_min"),1)} – {_fmt(o.get("northing_max"),1)}'),
    ]
    body = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>" for k, v in rows)
    return (f'<h2>1 · Dive overview</h2><div class="cards">{cards}</div>'
            f'<div class="scroll"><table><tbody>{body}</tbody></table></div>')


def _render_photogrammetry(p: dict) -> str:
    chunks = p.get("chunks", [])
    n_seg = len(p.get("segments", []))
    n_chunk = len(chunks)
    tot_v = sum(c["verts"] for c in chunks if c.get("verts"))
    tot_f = sum(c["faces"] for c in chunks if c.get("faces"))
    tot_cam = sum(c["cameras"] for c in chunks if c.get("cameras"))
    missing = p.get("missing_ortho", [])

    cards = "".join([
        _card(_fmt(n_seg, 0), "Segments"),
        _card(_fmt(n_chunk, 0), "Chunks"),
        _card(_fmt(tot_cam, 0), "Cameras/frames"),
        _card(_fmt(tot_v, 0), "Mesh vertices"),
        _card(_fmt(tot_f, 0), "Mesh faces"),
    ])
    flag = ""
    if missing:
        flag = (f'<div class="flag">⚑ {len(missing)} chunk(s) with missing/empty '
                f'orthomosaic: {_esc(", ".join(missing))}</div>')

    head = ("<tr><th>Segment</th><th>Chunk</th><th>Cameras</th><th>Vertices</th>"
            "<th>Faces</th><th>DEM extent (m)</th><th>Tilt (°)</th>"
            "<th>Roughness RMS (m)</th><th>Ortho</th><th>Ortho size</th></tr>")
    body_rows = []
    for c in chunks:
        dem = c.get("dem", {}) or {}
        ext = "—"
        if dem.get("ok") and dem.get("extent_x_m") is not None:
            ext = f'{_fmt(dem.get("extent_x_m"),1)} × {_fmt(dem.get("extent_y_m"),1)}'
        ortho = ('<span class="ok">yes</span>' if c.get("ortho_present")
                 else '<span class="miss">missing</span>')
        body_rows.append(
            "<tr>"
            f"<td>{_esc(c.get('segment'))}</td>"
            f"<td>{_esc(c.get('chunk'))}</td>"
            f"<td>{_fmt(c.get('cameras'),0)}</td>"
            f"<td>{_fmt(c.get('verts'),0)}</td>"
            f"<td>{_fmt(c.get('faces'),0)}</td>"
            f"<td>{ext}</td>"
            f"<td>{_fmt(dem.get('tilt_deg'),2)}</td>"
            f"<td>{_fmt(dem.get('rms_m'),3)}</td>"
            f"<td>{ortho}</td>"
            f"<td>{_human_size(c.get('ortho_size'))}</td>"
            "</tr>")
    body = "".join(body_rows)
    return (f'<h2>2 · Photogrammetry inventory</h2><div class="cards">{cards}</div>'
            f'{flag}<div class="scroll"><table><thead>{head}</thead>'
            f'<tbody>{body}</tbody></table></div>')


def _render_sensors(s: dict) -> str:
    def _line(label, rec):
        path = rec.get("path")
        val = (f'{_human_size(rec.get("size"))} '
               f'<span class="dim">{_esc(path)}</span>' if path
               else '<span class="miss">—</span>')
        return f"<tr><td>{_esc(label)}</td><td>{val}</td></tr>"

    top = _line("Nav trackline (.ply)", s.get("trackline", {}))
    top += _line("Nav depth raster (.tif)", s.get("nav_depth", {}))

    head = ("<tr><th>Channel</th><th>2D raster</th><th>2D size</th>"
            "<th>NetCDF</th><th>NetCDF size</th></tr>")
    rows = []
    for c in s.get("channels", []):
        r2d = '<span class="ok">yes</span>' if c.get("raster_2d") else '<span class="miss">—</span>'
        nc = '<span class="ok">yes</span>' if c.get("netcdf") else '<span class="miss">—</span>'
        rows.append(
            "<tr>"
            f"<td>{_esc(c.get('channel'))}</td>"
            f"<td>{r2d}</td><td>{_human_size(c.get('raster_2d_size'))}</td>"
            f"<td>{nc}</td><td>{_human_size(c.get('netcdf_size'))}</td>"
            "</tr>")
    return (f'<h2>3 · Sensor products</h2>'
            f'<div class="scroll"><table><tbody>{top}</tbody></table></div>'
            f'<div class="scroll" style="margin-top:12px"><table>'
            f'<thead>{head}</thead><tbody>{"".join(rows)}</tbody></table></div>')


def _render_anomalies(a: dict) -> str:
    total = a.get("total", 0)
    cards = _card(_fmt(total, 0), "Total events")
    for cfg in ANOMALY_CONFIGS:
        if cfg in a.get("by_config", {}):
            cards += _card(_fmt(a["by_config"][cfg], 0), cfg)

    head = ("<tr><th>Config</th><th>Kind</th>"
            + "".join(f"<th>{ch}</th>" for ch in ANOMALY_CHANNELS)
            + "<th>Subtotal</th></tr>")
    rows = []
    for cfg in ANOMALY_CONFIGS:
        cfg_rec = a.get("configs", {}).get(cfg)
        if not cfg_rec:
            continue
        for kind_key, label in ANOMALY_KINDS:
            row = cfg_rec.get(kind_key, {})
            sub = sum(v for v in row.values() if v)
            cells = "".join(
                f"<td>{_fmt(row.get(ch), 0)}</td>" for ch in ANOMALY_CHANNELS)
            rows.append(
                f"<tr><td>{_esc(cfg)}</td><td>{_esc(label)}</td>{cells}"
                f"<td><b>{_fmt(sub,0)}</b></td></tr>")
    note = ('<div class="note">The anomaly site-catalog / packaging step '
            '(<code>anomaly/</code> product) is not yet produced; the counts '
            'above come directly from the Grapher matrix event CSVs.</div>')
    if not rows:
        return (f'<h2>4 · Anomaly detection</h2>'
                f'<div class="flag">No anomaly event CSVs found under '
                f'{_esc(a.get("root"))}</div>{note}')
    return (f'<h2>4 · Anomaly detection</h2><div class="cards">{cards}</div>'
            f'<div class="scroll"><table><thead>{head}</thead>'
            f'<tbody>{"".join(rows)}</tbody></table></div>{note}')


def _render_previews(p: dict, out_dir: Path) -> str:
    previews = [c for c in p.get("chunks", []) if c.get("preview")]
    if not previews:
        return ('<h2>5 · Ortho previews</h2>'
                '<p class="dim">No <code>preview_ortho.png</code> files found '
                'under the chunk directories.</p>')
    figs = []
    for c in previews:
        try:
            rel = os.path.relpath(c["preview"], out_dir)
        except ValueError:
            rel = c["preview"]
        src = "file:" + rel if os.path.isabs(rel) else rel
        figs.append(
            f'<figure><img src="{_esc(rel)}" alt="preview" loading="lazy">'
            f'<figcaption>{_esc(c.get("segment"))}/{_esc(c.get("chunk"))}'
            f'</figcaption></figure>')
    return (f'<h2>5 · Ortho previews</h2>'
            f'<div class="gallery">{"".join(figs)}</div>')


# ---------------------------------------------------------------------------
# top-level assembly
# ---------------------------------------------------------------------------
def build_report(workspace_dir, out_path: Optional[str] = None) -> str:
    """Assemble the HTML project report; return the path written."""
    ws = Path(workspace_dir)
    resolver = PathResolver(ws)

    interp_path = resolver.interp_full()
    survey_root = resolver.survey_products()
    photo_root = resolver.photogrammetry_root(survey_root)
    # anomaly event CSVs live under inputs/grapher_matrix_figs (not the resolver's
    # anomaly_dir, which is the not-yet-produced site-catalog output).
    anomaly_figs = resolver.inputs_dir() / "grapher_matrix_figs"

    out = Path(out_path) if out_path else ws / "PROJECT_REPORT.html"
    out_dir = out.parent

    overview = summarize_interp(interp_path)
    photo = inventory_photogrammetry(photo_root)
    sensors = inventory_sensors(survey_root)
    anomalies = inventory_anomalies(anomaly_figs)

    generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    proj_name = ws.name

    html_doc = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Project Report — {_esc(proj_name)}</title>
<style>{CSS}</style></head>
<body><div class="wrap">
<h1>EPR Dive Project Report</h1>
<p class="sub"><b>{_esc(proj_name)}</b> · {_esc(str(ws))} · generated {_esc(generated)}</p>
{_render_overview(overview)}
{_render_photogrammetry(photo)}
{_render_sensors(sensors)}
{_render_anomalies(anomalies)}
{_render_previews(photo, out_dir)}
<footer>Generated by <code>project_report.py</code> — a read-only inventory of
the <code>.eprproj</code> bundle. No workspace files were modified.</footer>
</div></body></html>"""

    out.write_text(html_doc, encoding="utf-8")
    return str(out)


def _cli(argv) -> int:
    if len(argv) < 2:
        print("usage: python3 project_report.py <workspace_dir> [out_path]",
              file=sys.stderr)
        return 2
    ws = argv[1]
    out = argv[2] if len(argv) > 2 else None
    path = build_report(ws, out)
    size = _size_of(Path(path))
    print(f"wrote {path} ({_human_size(size)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv))
