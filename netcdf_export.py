"""netcdf_export.py — CF-compliant gridded NetCDF writer for sensor fields.

Oceanographic analysis pipelines typically consume NetCDF rather than GeoTIFF
or PLY.  This module writes a 3-D gridded sensor field (depth × northing ×
easting) following the CF (Climate and Forecast) metadata conventions, so the
output drops directly into xarray / Panoply / Ferret / QGIS mesh layers.

The vertical axis is stored as ``depth`` (positive down, CF standard_name
"depth"); horizontal axes are UTM ``easting`` / ``northing`` with a CF
``grid_mapping`` variable carrying the projection so georeferencing survives.

netCDF4 is imported lazily; callers get a clear, actionable error if the
package is not installed rather than an import failure at module load.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import numpy as np


def _require_netcdf4():
    try:
        import netCDF4  # noqa: F401
        return netCDF4
    except ImportError as exc:
        raise ImportError(
            "NetCDF export needs the 'netCDF4' package. Install it with:\n"
            "    pip install netCDF4"
        ) from exc


def _utm_grid_mapping_name(epsg: Optional[int]) -> str:
    # UTM (and most projected EPSG codes used here) is Transverse Mercator.
    return "transverse_mercator"


def write_gridded_netcdf(
    path: str,
    *,
    easting: np.ndarray,
    northing: np.ndarray,
    depth: np.ndarray,
    data: np.ndarray,
    channel: str,
    units: str = "",
    epsg: Optional[int] = None,
    title: str = "",
    log_fn=None,
) -> str:
    """Write a CF-1.8 gridded NetCDF for one sensor channel.

    Args:
        path:     Output .nc file path.
        easting:  1-D UTM easting coordinate (length nx), ascending.
        northing: 1-D UTM northing coordinate (length ny), ascending.
        depth:    1-D depth coordinate (length nz), metres positive-down.
        data:     3-D array shaped (nz, ny, nx); NaN marks empty cells.
        channel:  Sensor channel name (becomes the data variable name).
        units:    Physical units of the channel (e.g. "degC", "PSU").
        epsg:     UTM EPSG code for the grid_mapping (None ⇒ omit projection).
        title:    Global 'title' attribute; defaults to the channel name.
        log_fn:   Optional progress sink.

    Returns:
        The written file path.
    """
    log = log_fn or (lambda _m: None)
    netCDF4 = _require_netcdf4()

    data = np.asarray(data, dtype=np.float32)
    nz, ny, nx = data.shape
    if (len(depth), len(northing), len(easting)) != (nz, ny, nx):
        raise ValueError(
            f"Coordinate lengths {(len(depth), len(northing), len(easting))} "
            f"do not match data shape {(nz, ny, nx)}."
        )

    # A CF-safe variable name (no spaces / special chars).
    safe = "".join(c if c.isalnum() or c in "_" else "_" for c in channel) or "value"

    ds = netCDF4.Dataset(path, "w", format="NETCDF4")
    try:
        ds.createDimension("depth", nz)
        ds.createDimension("northing", ny)
        ds.createDimension("easting", nx)

        v_e = ds.createVariable("easting", "f8", ("easting",))
        v_e[:] = easting
        v_e.units = "m"
        v_e.standard_name = "projection_x_coordinate"
        v_e.long_name = "UTM easting"
        v_e.axis = "X"

        v_n = ds.createVariable("northing", "f8", ("northing",))
        v_n[:] = northing
        v_n.units = "m"
        v_n.standard_name = "projection_y_coordinate"
        v_n.long_name = "UTM northing"
        v_n.axis = "Y"

        v_d = ds.createVariable("depth", "f8", ("depth",))
        v_d[:] = depth
        v_d.units = "m"
        v_d.standard_name = "depth"
        v_d.long_name = "depth below surface"
        v_d.positive = "down"
        v_d.axis = "Z"

        var = ds.createVariable(
            safe, "f4", ("depth", "northing", "easting"),
            fill_value=np.float32(np.nan), zlib=True, complevel=4,
        )
        var[:] = data
        if units:
            var.units = units
        var.long_name = channel
        var.coordinates = "depth northing easting"

        if epsg is not None:
            crs = ds.createVariable("crs", "i4")
            crs.grid_mapping_name = _utm_grid_mapping_name(epsg)
            crs.epsg_code = f"EPSG:{epsg}"
            try:
                from rasterio.crs import CRS
                crs.crs_wkt = CRS.from_epsg(epsg).to_wkt()
                crs.spatial_ref = crs.crs_wkt  # GDAL/QGIS compatibility
            except Exception:
                pass
            var.grid_mapping = "crs"

        n_valid = int(np.isfinite(data).sum())
        ds.Conventions = "CF-1.8"
        ds.title = title or f"{channel} gridded field"
        ds.institution = "Woods Hole Oceanographic Institution"
        ds.source = "EPR Sampling Tool — gridded sensor field"
        ds.history = (
            f"{datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} "
            f"created by EPR Sampling Tool"
        )
        ds.summary = (
            f"{channel} interpolated onto a {nx}×{ny}×{nz} (E×N×depth) grid; "
            f"{n_valid} populated cells."
        )
    finally:
        ds.close()

    log(f"NetCDF: {path}  ({nx}×{ny}×{nz}, {safe})")
    return path


def grid_to_arrays(df_grid_reset, scalar_col: str = "scalar"):
    """Pivot a PointCloudPipeline grid DataFrame into dense CF arrays.

    Args:
        df_grid_reset: pipeline.df_grid.reset_index() — must have columns
                       x, y, z, ix, iy, iz, and the scalar column.  z is the
                       negated depth used internally (Z-up, negative below
                       surface), so depth = -z.
        scalar_col:    Name of the value column.

    Returns:
        (easting, northing, depth, data) where data is shaped
        (nz, ny, nx) with NaN in empty cells, and all three coordinate
        vectors are ascending.
    """
    df = df_grid_reset
    nx = int(df["ix"].max()) + 1
    ny = int(df["iy"].max()) + 1
    nz = int(df["iz"].max()) + 1

    # Coordinate axes: one representative value per integer index.
    easting  = df.groupby("ix")["x"].first().reindex(range(nx)).to_numpy()
    northing = df.groupby("iy")["y"].first().reindex(range(ny)).to_numpy()
    z_axis   = df.groupby("iz")["z"].first().reindex(range(nz)).to_numpy()
    depth    = -z_axis  # internal z is -depth

    data = np.full((nz, ny, nx), np.nan, dtype=np.float32)
    iz = df["iz"].to_numpy(dtype=int)
    iy = df["iy"].to_numpy(dtype=int)
    ix = df["ix"].to_numpy(dtype=int)
    data[iz, iy, ix] = df[scalar_col].to_numpy(dtype=float)

    # CF prefers ascending coordinates; depth (= -z) ends up descending in iz
    # order, so flip the depth axis (and the data along it) to ascend.
    if nz > 1 and depth[0] > depth[-1]:
        depth = depth[::-1]
        data = data[::-1, :, :]

    return easting, northing, depth, data
