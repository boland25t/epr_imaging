#!/usr/bin/env python3
"""Stage a dive workspace's imagery products for BIIGLE annotation.

Everything a BIIGLE volume needs is collected under <workspace>/survey/biigle/:

  frames.csv           transit frame inventory (stride-sampled, speed-filtered
                       with the same V_MIN transit rule as frame_color_analysis)
  metadata.csv         BIIGLE volume-metadata CSV (filename,taken_at,lng,lat,
                       gps_altitude,distance_to_ground)
  orthos/              per-chunk orthomosaics re-encoded as plain 8-bit RGB
                       TIFFs (BIIGLE ignores geo-referencing, so it is
                       stripped) — each with a <name>.georef.json sidecar
                       recording the original affine transform + CRS so pixel
                       annotations can be mapped back to UTM by the ingest
                       module.  The sidecar is the round-trip contract.
  seed_candidates.csv  white-mat blob detections (frame px coords) consumed by
                       biigle_bridge.push_seed_annotations as seed points.

Frame ground truth comes from the per-segment manifests via
frame_color_analysis.load_frame_manifest; the per-dive colour-correction
coefficients fitted by that pipeline are adopted from
survey/frame_color_meta.json (same pattern as analysis_figures).

Qt-free.  CLI: python3 biigle_export_prep.py <workspace> [--orthos-max-px N]
"""
from __future__ import annotations
import glob
import json
import math
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import Affine
from rasterio.windows import Window

import frame_color_analysis as fca
from frame_color_analysis import V_MIN, WHITE_T, correct, load_frame_manifest, \
    vehicle_speed_at

WORKERS = 10          # parallel image I/O (slow /mnt/f mount)
STRIDE = 5            # every 5th transit frame (~1.25 m spacing)
MIN_ORTHO_BYTES = 1e6  # skip stub orthomosaics below 1 MB
ORTHO_CRS = "EPSG:32613"


def _adopt_dive_coeffs(workspace_dir) -> None:
    """Adopt the dive's fitted attenuation coefficients (frame_color_meta.json
    'coeffs') into frame_color_analysis.C so correct() matches the pipeline."""
    meta = Path(workspace_dir) / "survey" / "frame_color_meta.json"
    if meta.exists():
        fca.C.update(json.loads(meta.read_text()).get("coeffs", {}))


def _manifest_latlon(workspace_dir) -> pd.DataFrame:
    """lat/lon/depth per frame path, from the same interp.csv manifests that
    load_frame_manifest reads (it drops those columns, so re-key them here)."""
    rows = []
    for mc in sorted(glob.glob(
            f"{workspace_dir}/survey/photogrammetry/seg*/segment_*/interp.csv")):
        m = pd.read_csv(mc)
        keep = [c for c in ("frame_filename", "lat", "lon", "depth") if c in m]
        m = m[keep]
        base = Path(mc).parent / "frames"
        m["fn"] = [str(base / f) for f in m.frame_filename]
        rows.append(m.drop(columns=["frame_filename"]))
    return pd.concat(rows, ignore_index=True).drop_duplicates("fn")


def build_frame_listing(workspace_dir, stride=STRIDE, log=print) -> pd.DataFrame:
    """Transit frame inventory for a BIIGLE volume -> survey/biigle/frames.csv.

    Same filtering as frame_color_analysis.build_table: drop on-station frames
    (speed < V_MIN from the global interp) and stride-sample the remainder.
    Columns: frame_filename, src_path, seg, unix_time, lat, lon, alt
    (+ depth, carried for write_metadata_csv's gps_altitude).
    """
    B = str(workspace_dir)
    mf = load_frame_manifest(B)
    n_manifest = len(mf)
    mf["speed"] = vehicle_speed_at(B, mf.unix_time)
    mf = mf[mf.speed >= V_MIN].reset_index(drop=True)
    n_moving = len(mf)
    mf = mf.iloc[::stride].reset_index(drop=True)
    mf = mf.merge(_manifest_latlon(B), on="fn", how="left")
    for c in ("lat", "lon", "depth"):
        if c not in mf:
            mf[c] = np.nan
    df = pd.DataFrame(dict(
        frame_filename=mf.fn.map(os.path.basename), src_path=mf.fn,
        seg=mf.seg, unix_time=mf.unix_time, lat=mf.lat, lon=mf.lon,
        alt=mf.alt, depth=mf.depth))
    out = Path(B) / "survey" / "biigle" / "frames.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    log(f"frames: {n_manifest} manifest -> {n_moving} transit "
        f"(speed >= {V_MIN} m/s) -> {len(df)} listed (every {stride}th)")
    return df


def write_metadata_csv(df, out_path) -> Path:
    """BIIGLE volume-metadata CSV from a build_frame_listing frame.

    Header (exact): filename,taken_at,lng,lat,gps_altitude,distance_to_ground
    taken_at is UTC 'YYYY-MM-DD HH:MM:SS'; gps_altitude is the negative depth
    (metres below sea surface) when a depth column is present, else empty;
    distance_to_ground is the vehicle altitude above the seafloor.
    """
    taken = pd.to_datetime(df.unix_time, unit="s", utc=True)
    depth = df["depth"] if "depth" in df else pd.Series(np.nan, index=df.index)
    md = pd.DataFrame(dict(
        filename=df.frame_filename,
        taken_at=taken.dt.strftime("%Y-%m-%d %H:%M:%S"),
        lng=df.lon, lat=df.lat,
        gps_altitude=(-depth.abs()).round(3),
        distance_to_ground=df.alt))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    md.to_csv(out_path, index=False)  # delimiter ',', quotechar '"' (defaults)
    return out_path


def _export_one_ortho(src, dst, max_px, log=print) -> dict:
    with rasterio.open(src) as ds:
        w, h = ds.width, ds.height
        transform = ds.transform
        if max_px and max(w, h) > max_px:
            s = max_px / max(w, h)
            ow, oh = max(1, round(w * s)), max(1, round(h * s))
            transform = transform * Affine.scale(w / ow, h / oh)
        else:
            ow, oh = w, h
        prof = dict(driver="GTiff", dtype="uint8", count=3, width=ow,
                    height=oh, compress="deflate", tiled=True,
                    blockxsize=256, blockysize=256, photometric="RGB",
                    BIGTIFF="IF_SAFER")  # no crs/transform: georef stripped
        with rasterio.open(dst, "w", **prof) as out:
            if (ow, oh) == (w, h):
                for _, win in out.block_windows(1):  # stream, RAM-safe
                    out.write(ds.read((1, 2, 3), window=win), window=win)
            else:
                out.write(ds.read((1, 2, 3), out_shape=(3, oh, ow),
                                  resampling=Resampling.average))
        sidecar = Path(str(dst) + ".georef.json")  # round-trip contract
        sidecar.write_text(json.dumps(dict(
            transform=list(transform)[:6], crs=str(ds.crs or ORTHO_CRS),
            width=ow, height=oh, src=str(src)), indent=1))
    log(f"  {Path(dst).name}: {ow}x{oh}")
    return dict(dst=str(dst), sidecar=str(sidecar), width=ow, height=oh)


def export_orthos(workspace_dir, out_dir, max_px=None, log=print) -> list:
    """Convert every chunk orthomosaic to a BIIGLE-friendly plain image:
    8-bit RGB deflate TIFF, 256x256 tiles, geo-referencing stripped, named
    <seg>_<chunk>_ortho.tif with a .georef.json sidecar (see module doc).
    If max_px is given, the longest axis is downscaled to it and the sidecar
    transform is scaled to match, so px->UTM mapping stays exact."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    jobs = []
    for src in sorted(glob.glob(
            f"{workspace_dir}/survey/photogrammetry/seg*/chunk_*/orthomosaic.tif")):
        if Path(src).stat().st_size < MIN_ORTHO_BYTES:
            continue
        m = re.search(r"photogrammetry/(seg\d+)/(chunk_\d+)/",
                      src.replace("\\", "/"))
        jobs.append((src, out_dir / f"{m.group(1)}_{m.group(2)}_ortho.tif"))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        results = list(ex.map(
            lambda j: _export_one_ortho(j[0], j[1], max_px, log), jobs))
    return results


def detect_seed_candidates(workspace_dir, out_csv=None, stride=STRIDE,
                           min_blob_px=30, log=print) -> pd.DataFrame:
    """White-mat blob detections for BIIGLE seed annotations.

    Frames flagged by survey/frame_color_metrics.csv with white > 0.02 are
    reloaded at 1/4 resolution, colour-corrected with the pipeline's own
    fitted coefficients, thresholded at min(R,G,B) > WHITE_T, and connected
    components >= min_blob_px (reduced-res px) become one row each, with
    centre/radius scaled back to FULL-resolution frame pixels.

    The metrics file is already the stride-sampled transit set (its stride is
    in frame_color_meta.json); `stride` re-samples only if coarser than that.
    Output (survey/biigle/seed_candidates.csv) is the contract consumed by
    biigle_bridge.push_seed_annotations.
    """
    B = str(workspace_dir)
    _adopt_dive_coeffs(B)
    met = pd.read_csv(f"{B}/survey/frame_color_metrics.csv")
    meta = Path(B) / "survey" / "frame_color_meta.json"
    met_stride = json.loads(meta.read_text()).get("stride", STRIDE) \
        if meta.exists() else STRIDE
    step = max(1, stride // max(1, met_stride))
    cand = met[met.white > 0.02].iloc[::step].reset_index(drop=True)
    log(f"seeds: {len(cand)} of {len(met)} metric frames have white cover > 0.02")

    scale = [None]  # full-res px per reduced px, measured on the first frame

    def measure_scale(fn):
        full = cv2.imread(fn)
        red = cv2.imread(fn, cv2.IMREAD_REDUCED_COLOR_4)
        if full is None or red is None:
            return None
        return full.shape[1] / red.shape[1]

    def work(row):
        im = cv2.imread(row.fn, cv2.IMREAD_REDUCED_COLOR_4)
        if im is None:
            return []
        if scale[0] is None:
            scale[0] = measure_scale(row.fn) or 4.0
        s = scale[0]
        cor = correct(im, row.alt)
        mn = np.minimum(np.minimum(cor[:, :, 2], cor[:, :, 1]), cor[:, :, 0])
        mask = (mn > WHITE_T).astype(np.uint8)
        n, _, stats, cent = cv2.connectedComponentsWithStats(mask, 8)
        keep = [i for i in range(1, n)
                if stats[i, cv2.CC_STAT_AREA] >= min_blob_px]
        h, w = im.shape[:2]
        return [dict(
            frame_filename=os.path.basename(row.fn),
            cx_px=round(cent[i][0] * s, 1), cy_px=round(cent[i][1] * s, 1),
            r_px=round(0.5 * math.hypot(stats[i, cv2.CC_STAT_WIDTH],
                                        stats[i, cv2.CC_STAT_HEIGHT]) * s, 1),
            white_frac=stats[i, cv2.CC_STAT_AREA] / (h * w),
            n_blobs_in_frame=len(keep)) for i in keep]

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = [r for rs in ex.map(work, cand.itertuples(index=False))
                for r in rs]
    df = pd.DataFrame(rows, columns=["frame_filename", "cx_px", "cy_px",
                                     "r_px", "white_frac", "n_blobs_in_frame"])
    out_csv = Path(out_csv or f"{B}/survey/biigle/seed_candidates.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    log(f"seeds: {len(df)} blobs in "
        f"{df.frame_filename.nunique() if len(df) else 0} frames "
        f"(>= {min_blob_px} px at 1/4 res)")
    return df


def filter_seeds(seeds_csv, out_csv=None, min_r_px=100, top_per_frame=5) -> pd.DataFrame:
    """Reduce raw blob detections to a reviewable seed set: drop small blobs,
    keep the largest ``top_per_frame`` per frame.  Writes
    seed_candidates_filtered.csv next to the input (the file
    biigle_bridge prefers when present)."""
    s = pd.read_csv(seeds_csv)
    f = (s[s.r_px >= min_r_px].sort_values("r_px", ascending=False)
         .groupby("frame_filename").head(top_per_frame)
         .sort_values(["frame_filename", "r_px"], ascending=[True, False]))
    out_csv = Path(out_csv or Path(seeds_csv).with_name("seed_candidates_filtered.csv"))
    f.to_csv(out_csv, index=False)
    return f


def prep_all(workspace_dir, orthos_max_px=None, log=print) -> dict:
    B = Path(workspace_dir)
    out = B / "survey" / "biigle"
    df = build_frame_listing(B, log=log)
    write_metadata_csv(df, out / "metadata.csv")
    log(f"metadata.csv: {len(df)} rows")
    seeds = detect_seed_candidates(B, out / "seed_candidates.csv", log=log)
    filt = filter_seeds(out / "seed_candidates.csv")
    log(f"  filtered seeds: {len(filt)} of {len(seeds)} (r_px>=100, top 5/frame)")
    orthos = export_orthos(B, out / "orthos", max_px=orthos_max_px, log=log)
    log(f"orthos: {len(orthos)} exported"
        + (f" (max_px={orthos_max_px})" if orthos_max_px else " (native res)"))
    return dict(frames=len(df), seeds=len(seeds), orthos=len(orthos))


if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    ws = args[0] if args else "."
    mp = None
    if "--orthos-max-px" in sys.argv:
        mp = int(sys.argv[sys.argv.index("--orthos-max-px") + 1])
    def log(m): print(m, flush=True)
    prep_all(ws, orthos_max_px=mp, log=log)
    print("BIIGLE_PREP_DONE", flush=True)
