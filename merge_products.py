#!/usr/bin/env python3
"""Merge per-chunk photogrammetry rasters into survey-wide mosaics.

Builds survey/photogrammetry/merged/{ortho_merged.tif, dem_merged.tif} from
every chunk orthomosaic/DEM under survey/photogrammetry/seg*/chunk_*/,
honouring the report's product-level chunk exclusions.  Output resolution is
chosen so the longest mosaic axis stays within ``max_dim`` pixels (full-res
chunk products remain the authoritative data; the merge is an overview
product for maps, report figures and region crops).

Qt-free.  merge_survey(workspace_dir, exclude=(), max_dim=30000) -> dict
"""
from __future__ import annotations
import glob
import math
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.merge import merge as rio_merge
from rasterio.vrt import WarpedVRT


def _chunk_rasters(workspace_dir: str, name: str, exclude) -> list[str]:
    pat = f"{workspace_dir}/survey/photogrammetry/seg*/chunk_*/{name}"
    out = []
    for p in sorted(glob.glob(pat)):
        norm = p.replace("\\", "/")
        if any(f"/{x}/" in norm + "/" for x in exclude):
            continue
        try:
            if Path(p).stat().st_size > 1e5:
                out.append(p)
        except OSError:
            pass
    return out


def _target_res(paths: list[str], max_dim: int) -> float:
    e0 = n0 = math.inf
    e1 = n1 = -math.inf
    native = math.inf
    for p in paths:
        with rasterio.open(p) as ds:
            b = ds.bounds
            e0, e1 = min(e0, b.left), max(e1, b.right)
            n0, n1 = min(n0, b.bottom), max(n1, b.top)
            native = min(native, max(ds.res))
    span = max(e1 - e0, n1 - n0)
    return max(native, span / max_dim)


def _merge_one(paths: list[str], out_path: Path, res: float, bands: int,
               dtype, nodata, resampling, log) -> None:
    log(f"  merging {len(paths)} rasters at {res:.3f} m -> {out_path.name}")
    srcs = [rasterio.open(p) for p in paths]
    try:
        arr, transform = rio_merge(srcs, res=res, nodata=nodata,
                                   resampling=resampling)
    finally:
        for s in srcs:
            s.close()
    arr = arr[:bands]
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2],
                   count=bands, dtype=dtype, crs="EPSG:32613",
                   transform=transform, nodata=nodata,
                   compress="deflate", tiled=True, bigtiff="IF_SAFER")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
    log(f"  wrote {out_path.name}: {arr.shape[2]}x{arr.shape[1]} px")


def merge_survey(workspace_dir, exclude=(), max_dim=30000, log_fn=print) -> dict:
    B = str(workspace_dir)
    out_dir = Path(B) / "survey" / "photogrammetry" / "merged"
    result = {}

    orthos = _chunk_rasters(B, "orthomosaic.tif", exclude)
    if orthos:
        res = _target_res(orthos, max_dim)
        out = out_dir / "ortho_merged.tif"
        _merge_one(orthos, out, res, bands=3, dtype="uint8", nodata=0,
                   resampling=Resampling.average, log=log_fn)
        result["ortho"] = str(out)

    dems = _chunk_rasters(B, "dem.tif", exclude)
    if dems:
        res = _target_res(dems, max_dim) * 1.6
        out = out_dir / "dem_merged.tif"
        _merge_one(dems, out, res, bands=1, dtype="float32", nodata=-32767.0,
                   resampling=Resampling.bilinear, log=log_fn)
        result["dem"] = str(out)
    return result


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    print(merge_survey(ws))


def build_previews(workspace_dir, exclude=(), log_fn=print) -> dict:
    """Render preview_ortho_merged.png and mosaics_contact_sheet.png for the
    report's photogrammetry gallery."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from rasterio.enums import Resampling as RS

    from ortho_gallery import stretch
    B = str(workspace_dir)
    PG = Path(B) / "survey" / "photogrammetry"
    out = {}

    merged = PG / "merged" / "ortho_merged.tif"
    if merged.is_file():
        with rasterio.open(merged) as ds:
            sc = max(ds.width, ds.height) / 2600
            img = ds.read([1, 2, 3], out_shape=(3, int(ds.height / sc), int(ds.width / sc)),
                          resampling=RS.average)
        rgb = np.transpose(img, (1, 2, 0)).astype(float)
        valid = rgb.sum(2) > 0
        rgb = stretch(rgb, valid)
        alpha = np.where(valid, 1.0, 0.0)
        h, w = rgb.shape[:2]
        fig, ax = plt.subplots(figsize=(w / 150, h / 150), dpi=150)
        fig.patch.set_facecolor("#0e1620")
        ax.imshow(np.dstack([rgb / 255.0, alpha]))
        ax.set_axis_off()
        fig.subplots_adjust(0, 0, 1, 1)
        fp = PG / "merged" / "preview_ortho_merged.png"
        fig.savefig(fp, dpi=150, facecolor="#0e1620")
        plt.close(fig)
        out["preview"] = str(fp)
        log_fn(f"  wrote {fp.name}")

    orthos = _chunk_rasters(B, "orthomosaic.tif", exclude)
    if orthos:
        cols = max(4, int(math.ceil(math.sqrt(len(orthos) * 1.4))))
        rows = int(math.ceil(len(orthos) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 2.1, rows * 2.1), dpi=110)
        fig.patch.set_facecolor("#0e1620")
        axf = np.atleast_1d(axes).ravel()
        for ax in axf:
            ax.set_axis_off()
        for ax, p in zip(axf, orthos):
            with rasterio.open(p) as ds:
                sc = max(ds.width, ds.height) / 320
                img = ds.read([1, 2, 3], out_shape=(3, int(ds.height / sc), int(ds.width / sc)),
                              resampling=RS.average)
            rgb = np.transpose(img, (1, 2, 0)).astype(float)
            valid = rgb.sum(2) > 0
            rgb = stretch(rgb, valid)
            alpha = np.where(valid, 1.0, 0.0)
            ax.imshow(np.dstack([rgb / 255.0, alpha]))
            seg_chunk = "/".join(Path(p).parts[-3:-1])
            ax.set_title(seg_chunk, fontsize=6.5, color="#9fb0bd", pad=2)
        fp = PG / "mosaics_contact_sheet.png"
        fig.savefig(fp, dpi=110, facecolor="#0e1620", bbox_inches="tight")
        plt.close(fig)
        out["contact"] = str(fp)
        log_fn(f"  wrote {fp.name} ({len(orthos)} orthos)")
    return out
