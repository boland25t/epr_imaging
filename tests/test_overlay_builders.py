"""Tests for overlay_builders — the pure data-prep half (no pyvista needed).

Pins: the trackline uses UTM easting/northing with Z = -depth, drops bad rows,
subsamples huge paths, and carries a channel scalar; the drape aligns the ortho
to the DEM grid cell-for-cell even at different source resolutions.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import overlay_builders as ob


# --------------------------------------------------------------------------
# trackline
# --------------------------------------------------------------------------
def _write_interp(tmp_path, rows):
    import pandas as pd
    p = tmp_path / "interp_full.csv"
    pd.DataFrame(rows).to_csv(p, index=False)
    return p


def test_trackline_uses_utm_and_negates_depth(tmp_path):
    p = _write_interp(tmp_path, [
        {"easting": 500000.0, "northing": 6000000.0, "depth": 10.0, "co2": 400},
        {"easting": 500010.0, "northing": 6000005.0, "depth": 12.0, "co2": 420},
        {"easting": 500020.0, "northing": 6000010.0, "depth": 15.0, "co2": 500},
    ])
    d = ob.load_trackline(p, channel="co2")
    assert d["n"] == 3
    assert d["points"].shape == (3, 3)
    assert d["points"][0].tolist() == [500000.0, 6000000.0, -10.0]   # Z = -depth
    assert d["scalar_name"] == "co2"
    assert d["scalars"].tolist() == [400.0, 420.0, 500.0]


def test_trackline_drops_nan_rows(tmp_path):
    p = _write_interp(tmp_path, [
        {"easting": 1.0, "northing": 2.0, "depth": 3.0},
        {"easting": np.nan, "northing": 2.0, "depth": 3.0},   # bad coord
        {"easting": 4.0, "northing": 5.0, "depth": 6.0},
    ])
    d = ob.load_trackline(p)
    assert d["n"] == 2 and d["dropped"] == 1
    assert d["scalars"] is None


def test_trackline_missing_channel_is_ignored(tmp_path):
    p = _write_interp(tmp_path, [
        {"easting": 1.0, "northing": 2.0, "depth": 3.0},
        {"easting": 4.0, "northing": 5.0, "depth": 6.0},
    ])
    d = ob.load_trackline(p, channel="nonexistent")
    assert d["scalar_name"] is None and d["n"] == 2


def test_trackline_requires_utm_columns(tmp_path):
    p = _write_interp(tmp_path, [{"lat": 1.0, "lon": 2.0, "depth": 3.0}])
    with pytest.raises(ValueError):
        ob.load_trackline(p)


def test_trackline_subsamples_large_paths(tmp_path):
    n = 5000
    rows = [{"easting": float(i), "northing": float(i), "depth": 1.0} for i in range(n)]
    p = _write_interp(tmp_path, rows)
    d = ob.load_trackline(p, max_points=1000)
    assert d["n"] == 1000
    assert d["points"][0][0] == 0.0 and d["points"][-1][0] == float(n - 1)  # endpoints kept


# --------------------------------------------------------------------------
# draped DEM + ortho
# --------------------------------------------------------------------------
def _write_geotiff(path, array, *, transform, count=1, dtype="float32", nodata=None):
    rasterio = pytest.importorskip("rasterio")
    from rasterio.crs import CRS
    if array.ndim == 2:
        array = array[np.newaxis, :, :]
    with rasterio.open(
        str(path), "w", driver="GTiff", height=array.shape[1], width=array.shape[2],
        count=array.shape[0], dtype=dtype, crs=CRS.from_epsg(32610),
        transform=transform, nodata=nodata,
    ) as dst:
        dst.write(array)


def test_drape_aligns_ortho_to_dem_grid(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    # DEM: 20x20 at 1 m; ortho: 40x40 at 0.5 m covering the same 20x20 m extent.
    dem = (np.arange(400, dtype="float32").reshape(20, 20))
    dem_tr = Affine(1.0, 0, 500000.0, 0, -1.0, 6000000.0)
    _write_geotiff(tmp_path / "dem.tif", dem, transform=dem_tr, dtype="float32")

    ortho = np.zeros((3, 40, 40), dtype="uint8")
    ortho[0] = 200  # red channel
    ort_tr = Affine(0.5, 0, 500000.0, 0, -0.5, 6000000.0)
    _write_geotiff(tmp_path / "ortho.tif", ortho, transform=ort_tr, count=3, dtype="uint8")

    d = ob.load_draped_grid(tmp_path / "dem.tif", tmp_path / "ortho.tif")
    H, W = d["shape"]
    assert d["z"].shape == (H, W)
    assert d["rgb"].shape == (H, W, 3)
    assert d["x"].shape == (H, W) and d["y"].shape == (H, W)
    # ortho red channel dominates after resampling onto the DEM grid
    assert d["rgb"][..., 0].mean() > 150
    # world coordinates land inside the DEM extent
    assert 500000.0 <= d["x"].min() and d["x"].max() <= 500020.0


def test_drape_decimates_large_dem(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    dem = np.random.rand(400, 400).astype("float32")
    tr = Affine(1.0, 0, 0.0, 0, -1.0, 0.0)
    _write_geotiff(tmp_path / "dem.tif", dem, transform=tr, dtype="float32")
    ortho = np.zeros((3, 400, 400), dtype="uint8")
    _write_geotiff(tmp_path / "ortho.tif", ortho, transform=tr, count=3, dtype="uint8")
    d = ob.load_draped_grid(tmp_path / "dem.tif", tmp_path / "ortho.tif", max_cells=10_000)
    H, W = d["shape"]
    assert H * W <= 10_000 * 1.2         # decimated near the budget


def test_drape_handles_nodata(tmp_path):
    rasterio = pytest.importorskip("rasterio")
    from affine import Affine
    dem = np.ones((10, 10), dtype="float32")
    dem[0, 0] = -9999.0
    tr = Affine(1.0, 0, 0.0, 0, -1.0, 0.0)
    _write_geotiff(tmp_path / "dem.tif", dem, transform=tr, dtype="float32", nodata=-9999.0)
    ortho = np.zeros((3, 10, 10), dtype="uint8")
    _write_geotiff(tmp_path / "ortho.tif", ortho, transform=tr, count=3, dtype="uint8")
    d = ob.load_draped_grid(tmp_path / "dem.tif", tmp_path / "ortho.tif", max_cells=10_000)
    assert d["mask"].dtype == bool
    assert not d["mask"][0, 0]            # nodata cell flagged
    assert np.isfinite(d["z"]).all()      # but filled so the surface has no spikes
