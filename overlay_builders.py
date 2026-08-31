"""
overlay_builders.py — build 3D overlay meshes for the viewer.

Two overlays the 3D viewer draws on top of meshes / photogrammetry models:

  * a TRACKLINE — the vehicle's nav path as a 3D polyline in UTM world
    coordinates (easting, northing, Z = -depth), optionally coloured by a sensor
    channel so you can see, say, CO2 rising along the dive; and
  * a DRAPED SURFACE — a DEM GeoTIFF as an elevation grid with the matching
    orthomosaic painted onto it as per-vertex RGB, so the map reads as real
    terrain in the same world frame as everything else.

The heavy lifting (reading CSV/GeoTIFF, gridding, resampling) is pure numpy /
pandas / rasterio and is unit-tested.  Turning the resulting arrays into
pyvista meshes is isolated in the ``make_*`` functions and guarded so this module
imports fine in environments without pyvista (e.g. CI).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np

try:
    import pyvista as pv
    _PV_OK = True
except Exception:  # noqa: BLE001
    _PV_OK = False


# ---------------------------------------------------------------------------
# trackline (pure data prep)
# ---------------------------------------------------------------------------
def load_trackline(interp_csv: str | Path, channel: Optional[str] = None,
                   max_points: int = 300_000) -> dict:
    """Points + optional scalar for the nav trackline.

    Returns ``{points:(N,3) float32, scalars:(N,) float32|None, scalar_name,
    n, dropped}``.  Coordinates are UTM easting/northing with Z = -depth (depth
    in interp_full.csv is a positive distance below the surface), matching the
    convention every other 3D product uses.  Rows missing easting/northing/depth
    are dropped; if a channel is requested, rows missing it are dropped too.
    """
    import pandas as pd
    df = pd.read_csv(interp_csv)
    for col in ("easting", "northing", "depth"):
        if col not in df.columns:
            raise ValueError(
                f"interp CSV is missing '{col}' — a trackline needs UTM "
                f"easting/northing/depth columns.")
    cols = ["easting", "northing", "depth"]
    scalar_name = None
    if channel and channel in df.columns:
        cols.append(channel)
        scalar_name = channel
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    before = len(sub)
    sub = sub.dropna()
    dropped = before - len(sub)

    pts = np.empty((len(sub), 3), dtype=np.float32)
    pts[:, 0] = sub["easting"].to_numpy()
    pts[:, 1] = sub["northing"].to_numpy()
    pts[:, 2] = -sub["depth"].to_numpy()          # below-surface → negative Z
    scalars = (sub[scalar_name].to_numpy(dtype=np.float32)
               if scalar_name else None)

    # Subsample uniformly (keep order) if the path is enormous.
    n = len(pts)
    if n > max_points:
        idx = np.linspace(0, n - 1, max_points).astype(int)
        pts = pts[idx]
        scalars = scalars[idx] if scalars is not None else None

    return {"points": pts, "scalars": scalars, "scalar_name": scalar_name,
            "n": len(pts), "dropped": int(dropped)}


def make_trackline_mesh(data: dict):
    """A pyvista polyline (with an optional scalar field) from load_trackline()."""
    if not _PV_OK:
        raise RuntimeError("pyvista is not installed.")
    pts = data["points"]
    if len(pts) < 2:
        raise ValueError("trackline needs at least two points.")
    n = len(pts)
    # VTK polyline connectivity: [n, 0, 1, 2, …, n-1].
    lines = np.empty(n + 1, dtype=np.int64)
    lines[0] = n
    lines[1:] = np.arange(n, dtype=np.int64)
    poly = pv.PolyData(pts.astype(np.float32))
    poly.lines = lines
    if data.get("scalars") is not None:
        poly[data["scalar_name"] or "value"] = data["scalars"]
        poly.set_active_scalars(data["scalar_name"] or "value")
    return poly


# ---------------------------------------------------------------------------
# draped DEM + orthomosaic (pure data prep)
# ---------------------------------------------------------------------------
def load_draped_grid(dem_tif: str | Path, ortho_tif: str | Path,
                     max_cells: int = 750_000) -> dict:
    """A DEM elevation grid with the orthomosaic resampled onto it as RGB.

    Returns ``{x,y,z:(H,W) float32, rgb:(H,W,3) uint8, mask:(H,W) bool, shape}``.
    ``mask`` marks valid (non-nodata) DEM cells.  The DEM defines the grid; the
    ortho is reprojected onto that exact grid so the two align cell-for-cell even
    if they were exported at different resolutions.  Large DEMs are decimated to
    ``max_cells`` for interactive display.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    with rasterio.open(str(dem_tif)) as dem:
        # Decimate on read so the working grid stays interactive.
        H0, W0 = dem.height, dem.width
        step = max(1, int(np.ceil(np.sqrt((H0 * W0) / float(max_cells)))))
        out_h, out_w = max(2, H0 // step), max(2, W0 // step)
        z = dem.read(1, out_shape=(out_h, out_w),
                     resampling=Resampling.bilinear).astype(np.float32)
        # Build the scaled transform for the decimated grid.
        transform = dem.transform * dem.transform.scale(W0 / out_w, H0 / out_h)
        nodata = dem.nodata
        dem_crs = dem.crs

    mask = np.isfinite(z)
    if nodata is not None:
        mask &= (z != nodata)

    # Cell-centre world coordinates for the decimated grid.
    rows, cols = np.mgrid[0:out_h, 0:out_w]
    xs = transform.c + (cols + 0.5) * transform.a + (rows + 0.5) * transform.b
    ys = transform.f + (cols + 0.5) * transform.d + (rows + 0.5) * transform.e
    x = xs.astype(np.float32)
    y = ys.astype(np.float32)

    # Resample the ortho onto the DEM grid.
    rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
    try:
        with rasterio.open(str(ortho_tif)) as ort:
            nbands = min(3, ort.count)
            for b in range(nbands):
                dst = np.zeros((out_h, out_w), dtype=np.uint8)
                reproject(
                    source=rasterio.band(ort, b + 1),
                    destination=dst,
                    src_transform=ort.transform, src_crs=ort.crs,
                    dst_transform=transform, dst_crs=dem_crs or ort.crs,
                    resampling=Resampling.bilinear)
                rgb[:, :, b] = dst
            if nbands == 1:                     # single-band ortho → greyscale
                rgb[:, :, 1] = rgb[:, :, 0]
                rgb[:, :, 2] = rgb[:, :, 0]
    except Exception:  # noqa: BLE001 — ortho is optional; fall back to a flat tint
        rgb[mask] = (120, 140, 160)

    # Fill nodata elevation with the grid mean so the surface has no spikes.
    if not mask.all():
        fill = float(np.nanmean(z[mask])) if mask.any() else 0.0
        z = np.where(mask, z, fill)

    return {"x": x, "y": y, "z": z, "rgb": rgb, "mask": mask,
            "shape": (out_h, out_w)}


def make_draped_mesh(data: dict):
    """A pyvista StructuredGrid with per-vertex RGB from load_draped_grid()."""
    if not _PV_OK:
        raise RuntimeError("pyvista is not installed.")
    x, y, z = data["x"], data["y"], data["z"]
    grid = pv.StructuredGrid(x, y, z)
    # pyvista ravels each (H,W) coordinate array in Fortran order to build points,
    # so per-vertex RGB must be raveled the same way, per channel, to line up.
    rgb = data["rgb"]
    rgb_flat = np.stack([rgb[:, :, c].ravel(order="F") for c in range(3)], axis=1)
    if rgb_flat.shape[0] != grid.n_points:        # defensive: never mis-map colours
        rgb_flat = rgb_flat[:grid.n_points]
    grid["RGB"] = rgb_flat.astype(np.uint8)
    grid.set_active_scalars("RGB")
    return grid
