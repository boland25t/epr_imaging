#!/usr/bin/env python3
"""FathomNet megafauna detection on chunk ORTHOMOSAICS — Qt-free, tiled, georeferenced.

Companion to fathomnet_detect.py.  Same detector (FathomNet/MBARI-315k YOLOv8,
499 classes) and the *same* bucketing / exclusion vocabulary — imported from
fathomnet_detect, never re-declared — but run over the photogrammetry
orthomosaics instead of the raw down-looking frames.  The payoff is
localisation: a frame detection inherits its frame's single nav fix (+/- ~3 m,
see fathomnet_detect's docstring), whereas an ortho detection is placed by the
raster's own affine transform, i.e. to the accuracy of the photogrammetric
georeference (decimetre-ish) and with a real ground footprint in metres.  The
cost is that overlapping frames have been collapsed into one surface, so an
individual is counted once instead of once per frame it appeared in: ortho
counts are *abundance*, frame counts are *sightings*.

WHY 2 mm/px INFERENCE GSD (the --gsd default)
---------------------------------------------
These orthos are 0.86-0.94 mm/px native (not 0.5), so a native 1280 px tile
would cover only ~1.2 m — a patch of mud with one squat lobster filling a
quarter of the input, an apparent-size regime nothing in MBARI-315k resembles.
Decimation is therefore mandatory, and the question is how much.

The a-priori answer is "match the frame regime": frames are 5312 x 2988 and
fathomnet_detect runs them at imgsz=1280, a 4.15x squeeze, so at the 3-7 m
altitudes flown here (median 4.7 m, footprint ~5-6 m) one network pixel of the
training/inference distribution subtends ~5.5 m / 1280 = 4.3 mm.  That
argument is wrong, and measurably so.  Swept on J1754 seg15/chunk_03 (887 Mpx,
the densest crustacean chunk in the dive by frame-based counts), kept
detections went 4.0 mm -> 31, 3.0 mm -> 52, 2.0 mm -> 78, 1.5 mm -> 108, and
QA crops at every step were inspected at native resolution:

  4.0 mm  under-detects and the survivors include metre-scale nonsense —
          "Psychrolutes phrictus, 134 x 128 cm" on a swath edge, "Vogtia
          serrata, 157 x 184 cm" on bare sediment.
  3.0 mm  recovers Munidopsis well but keeps the same two garbage boxes.
  2.0 mm  tight boxes on organisms, no metre-scale boxes in the sample, best
          Munidopsis + Riftia yield per unit compute.  <-- default
  1.5 mm  marginal extra small detections (a 4 x 5 cm "Amphipoda" smudge) and
          it starts fragmenting a single Riftia bush into ~38 tube boxes;
          3x the runtime of 2.0 mm for that.

The frame-matching argument fails because an orthomosaic is not a frame: it is
radiometrically blended and shadowless, so it lacks the strobe-lit local
contrast that makes a 12-px animal pop in a raw frame.  The detector needs
more pixels on target to compensate, and MBARI-315k contains plenty of
close-approach imagery, so larger apparent size is in-distribution anyway.

    gsd = 0.002 m/px  ->  decimation ~2.1x  ->  1280 px tile spans 2.56 m

i.e. tiles still sit inside a single frame footprint (so tile context is no
wider than what the model was trained on) while objects get ~2x the pixels.
--gsd is exposed because it is the single most consequential knob.  Decimated
reads ALWAYS use Resampling.average: nearest-neighbour subsampling aliases
hard and turns sediment speckle into fake texture the detector reads as
animals.

OTHER GOTCHAS BAKED IN
----------------------
* Band 4 is alpha and is load-bearing.  Outside the imaged swath the RGB bands
  hold (222, 0, 0) — a saturated red field, not black — so any read that
  ignores alpha is partly garbage.  Tiles with <30% alpha coverage
  (--min-alpha) are skipped entirely; in the tiles that survive, the
  out-of-swath pixels are zeroed to black before inference.  Orthos are long
  thin survey strips inside a big axis-aligned bounding box, so this typically
  discards well over half of the nominal tile grid for free.
* No contrast stretching, anywhere — not for inference, not for the QA crops.
  The detector was trained on raw ROV imagery and the QA figure exists to show
  what the detector actually saw.
* Tiles overlap (--overlap, default 0.2) so an animal on a tile seam is whole
  in at least one tile; the duplicates that creates are merged by per-class
  greedy IoU NMS over each ortho's full box list.

Pipeline per dive:
  find_orthos()        survey/photogrammetry/seg*/chunk_*/orthomosaic.tif
                       (merged/ and *_lite.tif deliberately excluded)
  detect_ortho()       one raster -> tiled inference -> NMS -> native-px boxes
                       + UTM box centres + ground footprint in metres
  write_outputs()      CSV (everything, incl. excluded rows, flagged)
                       + GeoJSON (kept only, EPSG:32613)
  qa_figure()          6 native-res crops across buckets + whole-ortho overview

Writes to <ws>/survey/fauna/:
  ortho_detections.csv        ALL detections, 'excluded' reason where applicable
  ortho_fauna_utm.geojson     kept detections only, Point, EPSG:32613
  ortho_fauna_qa.png          visual verification figure
  ortho_fauna_summary.json    run parameters + counts + failures
  ortho_cache/                per-ortho checkpoints, keyed by parameters

The corpus is ~190 Gpx on a slow external mount, so a full pass takes hours
and is mostly I/O bound (the GPU idles between tile reads).  Each ortho's
result is therefore written to ortho_cache/ as soon as it lands, and a re-run
with the same parameters skips what is already there — an interrupted run
resumes rather than restarting, which also means the job can be stopped to
give a training run the GPU and picked up afterwards.

CLI:
  python3 fauna_ortho.py                                  # both dives, full corpus
  python3 fauna_ortho.py --dives J1754 --limit 2            # pilot, 2 orthos
  python3 fauna_ortho.py --gsd 0.003 --conf 0.3
  python3 fauna_ortho.py --batch 4                          # share a busy GPU
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import gridspec

# Single source of truth for buckets / exclusions — do not duplicate here.
from fathomnet_detect import BUCKET_ORDER, bucket_of, exclusion_of

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
WEIGHTS = Path("/home/troyboland/models/mbari_315k_yolov8.pt")
UTM_EPSG = 32613

GSD = 0.002          # m/px at inference — see module docstring for the sweep
TILE = 1280          # inference px per tile side (== fathomnet_detect IMGSZ)
OVERLAP = 0.20       # fraction of tile side shared with the next tile
MIN_ALPHA = 0.30     # skip tiles with less imaged-swath coverage than this
CONF = 0.25          # matches fathomnet_detect, so counts are comparable
IOU_NMS = 0.50       # cross-tile duplicate merge, per class
BATCH = 8            # tiles per predict() call
DIVES = ("J1754", "J1756")

ALPHA_ON = 128       # averaged alpha >= this counts as imaged

# dark palette, consistent with the other figures in this repo
BG, INK, MUT, GRID = "#0e1620", "#dbe4ec", "#9fb0bd", "#2a3846"
BUCKET_COLOR = {
    "crustacean": "#f2c94c", "worm": "#eb5757", "fish": "#56ccf2",
    "anemone": "#bb6bd9", "unknown": "#9fb0bd",
}


# ------------------------------------------------------------- discovery -----
def find_orthos(workspace_dir, limit=None) -> list[Path]:
    """Per-chunk orthomosaics for one workspace, in seg/chunk order.

    Only survey/photogrammetry/seg*/chunk_*/orthomosaic.tif: the merged
    mosaic duplicates the chunks at coarser resolution and *_lite.tif are
    decimated previews, so both would double-count.
    """
    ws = Path(workspace_dir)
    paths = sorted(ws.glob("survey/photogrammetry/seg*/chunk_*/orthomosaic.tif"),
                   key=lambda p: (p.parent.parent.name, p.parent.name))
    if limit:
        paths = paths[:int(limit)]
    return paths


def ids_of(ortho_path) -> tuple[str, str]:
    """('seg07', 'chunk_02') from .../photogrammetry/seg07/chunk_02/orthomosaic.tif."""
    p = Path(ortho_path)
    seg = p.parent.parent.name
    chunk = p.parent.name
    if not re.match(r"^seg\d+", seg):
        seg = ""
    if not re.match(r"^chunk_?\d+", chunk):
        chunk = ""
    return seg, chunk


# ------------------------------------------------------------------ tiles ----
def tile_windows(width, height, tile_native, stride_native):
    """Top-left-anchored grid of native-px windows, clipped at the edges."""
    for r0 in range(0, height, stride_native):
        h = min(tile_native, height - r0)
        if h <= 0:
            break
        for c0 in range(0, width, stride_native):
            w = min(tile_native, width - c0)
            if w <= 0:
                break
            yield Window(c0, r0, w, h)
        if r0 + tile_native >= height:
            break


def read_tile(ds, win, dec):
    """Averaged decimated RGB tile + alpha coverage, out-of-swath zeroed.

    Returns (rgb uint8 HxWx3, imaged fraction 0-1); applying the coverage floor
    is the caller's job.  RGB outside the imaged swath is (222, 0, 0) in these
    rasters, so it is replaced with black: a saturated red field is a far worse
    thing to hand a detector than a dark one, and both are obviously
    not-seafloor.
    """
    oh = max(1, int(round(win.height / dec)))
    ow = max(1, int(round(win.width / dec)))
    arr = ds.read(window=win, out_shape=(ds.count, oh, ow),
                  resampling=Resampling.average)
    alpha = arr[3] if ds.count >= 4 else np.full((oh, ow), 255, np.uint8)
    imaged = alpha >= ALPHA_ON
    cov = float(imaged.mean())
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    rgb[~imaged] = 0
    return rgb, cov


# -------------------------------------------------------------------- NMS ----
def nms_per_class(df, iou=IOU_NMS) -> pd.DataFrame:
    """Greedy per-class IoU NMS over one ortho's boxes (native px).

    Tiles overlap by design, so the same animal can be reported 2-4 times with
    slightly different boxes.  Highest confidence wins; suppression is within
    class only, so a genuine co-located worm and crab both survive.
    """
    if df.empty:
        return df
    keep_idx = []
    for _, grp in df.groupby("cls", sort=False):
        g = grp.sort_values("conf", ascending=False)
        x1 = g.x1.to_numpy(float); y1 = g.y1.to_numpy(float)
        x2 = g.x2.to_numpy(float); y2 = g.y2.to_numpy(float)
        area = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        order = np.arange(len(g))
        alive = np.ones(len(g), bool)
        idx = g.index.to_numpy()
        for i in order:
            if not alive[i]:
                continue
            keep_idx.append(idx[i])
            rest = order[(order > i) & alive[order]]
            if rest.size == 0:
                continue
            xx1 = np.maximum(x1[i], x1[rest]); yy1 = np.maximum(y1[i], y1[rest])
            xx2 = np.minimum(x2[i], x2[rest]); yy2 = np.minimum(y2[i], y2[rest])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            union = area[i] + area[rest] - inter
            with np.errstate(divide="ignore", invalid="ignore"):
                ov = np.where(union > 0, inter / union, 0.0)
            alive[rest[ov > iou]] = False
    return df.loc[sorted(keep_idx)].reset_index(drop=True)


# ----------------------------------------------------------------- detect ----
def detect_ortho(ortho_path, model, conf=CONF, gsd=GSD, tile=TILE,
                 overlap=OVERLAP, min_alpha=MIN_ALPHA, batch=BATCH,
                 device=0, iou_nms=IOU_NMS, log=print) -> pd.DataFrame:
    """Tiled detection over one orthomosaic -> georeferenced box table.

    Boxes come back in NATIVE raster pixels (so they can be cropped at full
    resolution for review) plus the UTM easting/northing of the box centre and
    the box's ground size in metres.
    """
    path = Path(ortho_path)
    seg, chunk = ids_of(path)
    rows = []
    t0 = time.time()
    n_tiles = n_run = 0

    with rasterio.open(path) as ds:
        native = float(abs(ds.transform.a))
        dec = gsd / native
        tile_n = max(1, int(round(tile * dec)))
        stride_n = max(1, int(round(tile_n * (1.0 - overlap))))
        transform = ds.transform
        wins = list(tile_windows(ds.width, ds.height, tile_n, stride_n))
        log(f"[ortho] {seg}/{chunk} {ds.width}x{ds.height} @ {native*1000:.2f} mm/px "
            f"-> dec {dec:.2f}x, {tile_n} px native tiles, {len(wins)} candidates")

        pend_meta, pend_img = [], []

        def _predict(imgs):
            """predict() with a halving retry, so a concurrent training job
            grabbing VRAM mid-run degrades throughput instead of killing us."""
            try:
                return model.predict(imgs, conf=conf, device=device,
                                     imgsz=tile, verbose=False)
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower() or len(imgs) < 2:
                    raise
                log(f"[ortho]   CUDA OOM on {len(imgs)} tiles; retrying halved")
                import torch; torch.cuda.empty_cache()
                half = len(imgs) // 2
                return _predict(imgs[:half]) + _predict(imgs[half:])

        def flush():
            nonlocal pend_meta, pend_img
            if not pend_img:
                return
            res = _predict(pend_img)
            for (win, sx, sy), r in zip(pend_meta, res):
                b = r.boxes
                if b is None or len(b) == 0:
                    continue
                xyxy = b.xyxy.cpu().numpy()
                cf = b.conf.cpu().numpy()
                cl = b.cls.cpu().numpy().astype(int)
                for (bx1, by1, bx2, by2), c, k in zip(xyxy, cf, cl):
                    # inference px -> native raster px
                    nx1 = win.col_off + bx1 * sx; nx2 = win.col_off + bx2 * sx
                    ny1 = win.row_off + by1 * sy; ny2 = win.row_off + by2 * sy
                    cx, cy = (nx1 + nx2) / 2.0, (ny1 + ny2) / 2.0
                    east, north = transform * (cx, cy)
                    rows.append((str(path), seg, chunk, model.names[int(k)],
                                 float(c), nx1, ny1, nx2, ny2,
                                 float(east), float(north),
                                 (nx2 - nx1) * native, (ny2 - ny1) * native))
            pend_meta, pend_img = [], []

        for win in wins:
            n_tiles += 1
            rgb, cov = read_tile(ds, win, dec)
            if cov < min_alpha:
                continue
            n_run += 1
            sx = win.width / rgb.shape[1]
            sy = win.height / rgb.shape[0]
            # ultralytics treats ndarray input as BGR (cv2 convention)
            pend_meta.append((win, sx, sy))
            pend_img.append(np.ascontiguousarray(rgb[:, :, ::-1]))
            if len(pend_img) >= batch:
                flush()
        flush()

    det = pd.DataFrame(rows, columns=[
        "ortho", "seg", "chunk", "cls", "conf", "x1", "y1", "x2", "y2",
        "easting", "northing", "width_m", "height_m"])
    n_raw = len(det)
    det = nms_per_class(det, iou=iou_nms)
    det["bucket"] = det.cls.map(bucket_of)
    det["excluded"] = det.cls.map(exclusion_of)
    det = det[["ortho", "seg", "chunk", "cls", "bucket", "excluded", "conf",
               "x1", "y1", "x2", "y2", "easting", "northing",
               "width_m", "height_m"]]
    det.attrs.update(n_tiles=n_tiles, n_tiles_run=n_run, n_raw=n_raw,
                     runtime_s=time.time() - t0)
    log(f"[ortho]   {n_run}/{n_tiles} tiles imaged, {n_raw} boxes -> "
        f"{len(det)} after NMS, {time.time() - t0:.0f} s")
    return det


# ------------------------------------------------------- geometry guards ----
# Two ortho-specific failure modes that fathomnet_detect's class-name
# exclusion lists cannot catch, because the offending class names are
# legitimate elsewhere.  Both are flagged in the CSV and dropped from the
# GeoJSON, exactly like the class-based exclusions.
#
# 1. "oversize".  On a 2.56 m tile the detector's domain-shift failure is a
#    single half-tile box thrown over a featureless or motion-smeared patch of
#    seafloor.  Measured over the full corpus, boxes with a ground dimension
#    over 1 m are 25% of raw keeps and their classes are almost entirely
#    pelagic gelatinous animals and NE-Pacific fish that cannot be lying on
#    this seafloor at that size (Phacellophora, Deepstaria, Erenna, Tiburonia,
#    Lobata, Bathyraja, Psychrolutes).  Critically, NOT ONE of them is in the
#    crustacean or worm bucket: the guard removes the junk family and leaves
#    every Munidopsis and Riftia detection untouched, which is why 1.0 m is
#    both safe and worth applying.  Genuine benthic megafauna here are
#    decimetre-scale.
# 2. "void".  Orthos contain interior holes where no frame covered the
#    surface; read_tile zeroes them to black, and a black blob on pale
#    sediment reads convincingly as a dark-bodied animal (the corpus QA caught
#    a 36x26 cm "Oneirodes acanthias" sitting entirely on such a hole).  A box
#    whose own footprint is mostly not-imaged is therefore rejected.
MAX_BOX_M = 1.00        # reject boxes larger than this on their long axis
MIN_BOX_ALPHA = 0.50    # reject boxes this empty of imaged pixels


def box_imaged_fraction(det, sample=64, log=print) -> np.ndarray:
    """Fraction of each box's footprint that is inside the imaged swath.

    Reads band 4 over each box window, decimated to at most `sample` px a
    side — a few hundred KB per box, so this is cheap enough to run over the
    whole corpus without re-doing inference.  NaN where the read fails.
    """
    det = det.reset_index(drop=True)          # frac is indexed POSITIONALLY
    frac = np.full(len(det), np.nan)
    if det.empty:
        return frac
    for path, grp in det.groupby("ortho", sort=False):
        try:
            ds = rasterio.open(path)
        except Exception as exc:
            log(f"[ortho] alpha check: cannot open {path}: {exc}")
            continue
        with ds:
            if ds.count < 4:
                frac[grp.index.to_numpy()] = 1.0
                continue
            for i, d in zip(grp.index.to_numpy(), grp.itertuples(index=False)):
                c0 = int(max(0, np.floor(d.x1))); r0 = int(max(0, np.floor(d.y1)))
                c1 = int(min(ds.width, np.ceil(d.x2)))
                r1 = int(min(ds.height, np.ceil(d.y2)))
                if c1 <= c0 or r1 <= r0:
                    continue
                w = Window(c0, r0, c1 - c0, r1 - r0)
                oh = max(1, min(sample, r1 - r0)); ow = max(1, min(sample, c1 - c0))
                try:
                    a = ds.read(4, window=w, out_shape=(oh, ow),
                                resampling=Resampling.average)
                except Exception as exc:      # one bad box must not lose the rest
                    log(f"[ortho] alpha check failed at box {i}: {exc}")
                    continue
                frac[i] = float((a >= ALPHA_ON).mean())
    return frac


def apply_guards(det, max_box_m=MAX_BOX_M, min_box_alpha=MIN_BOX_ALPHA,
                 log=print) -> pd.DataFrame:
    """Flag 'oversize' and 'void' detections on top of the class exclusions."""
    det = det.reset_index(drop=True).copy()
    det["excluded"] = det.excluded.fillna("")
    det["box_imaged"] = np.nan
    if det.empty:
        return det
    long_axis = np.maximum(det.width_m.to_numpy(float),
                           det.height_m.to_numpy(float))
    big = (det.excluded == "") & (long_axis > max_box_m)
    det.loc[big, "excluded"] = "oversize"
    log(f"[ortho] guard: {int(big.sum())} boxes >{max_box_m} m -> 'oversize'")

    live = det.index[det.excluded == ""].to_numpy()
    if len(live):
        # positional assignment: box_imaged_fraction returns one value per row
        # of what it was handed, in that order
        det.loc[live, "box_imaged"] = box_imaged_fraction(
            det.loc[live], log=log)
        bi = det.loc[live, "box_imaged"].to_numpy(float)
        empty = live[np.isfinite(bi) & (bi < min_box_alpha)]
        det.loc[empty, "excluded"] = "void"
        log(f"[ortho] guard: {len(empty)} boxes <{min_box_alpha:.0%} imaged "
            f"-> 'void'")
    return det


# ---------------------------------------------------------------- outputs ----
def to_geojson(det, out_path, log=print) -> int:
    """Point features at UTM box centres, kept detections only, EPSG:32613.

    Same CRS-urn convention as fathomnet_detect.to_geojson so the two layers
    drop into the same QGIS project.
    """
    keep = det[det.excluded == ""]
    feats = []
    for d in keep.itertuples(index=False):
        if not (np.isfinite(d.easting) and np.isfinite(d.northing)):
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point",
                         "coordinates": [float(d.easting), float(d.northing)]},
            "properties": {
                "cls": d.cls, "bucket": d.bucket,
                "conf": round(float(d.conf), 4),
                "ortho": d.ortho, "seg": d.seg, "chunk": d.chunk,
                "width_m": round(float(d.width_m), 3),
                "height_m": round(float(d.height_m), 3),
                "box_imaged": (None if not np.isfinite(d.box_imaged)
                               else round(float(d.box_imaged), 3)),
                "box_px": [round(float(v), 1) for v in (d.x1, d.y1, d.x2, d.y2)],
                "loc": "orthomosaic box centre (raster georeference)",
            },
        })
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name",
                  "properties": {"name": f"urn:ogc:def:crs:EPSG::{UTM_EPSG}"}},
          "features": feats}
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gj))
    log(f"[ortho] {len(feats)} points -> {out_path}")
    return len(feats)


# --------------------------------------------------------------------- QA ----
def _crop(ortho_path, box, pad_frac=0.6, min_pad=60):
    """Native-resolution RGB crop around a box, no stretch. -> (img, box_in_crop)."""
    x1, y1, x2, y2 = box
    with rasterio.open(ortho_path) as ds:
        bw, bh = x2 - x1, y2 - y1
        pad = max(min_pad, pad_frac * max(bw, bh))
        c0 = int(max(0, np.floor(x1 - pad))); r0 = int(max(0, np.floor(y1 - pad)))
        c1 = int(min(ds.width, np.ceil(x2 + pad)))
        r1 = int(min(ds.height, np.ceil(y2 + pad)))
        if c1 <= c0 or r1 <= r0:
            return None, None
        win = Window(c0, r0, c1 - c0, r1 - r0)
        arr = ds.read(window=win)
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    if arr.shape[0] >= 4:
        rgb[arr[3] < ALPHA_ON] = 0
    return rgb, (x1 - c0, y1 - r0, x2 - c0, y2 - r0)


def _overview(ortho_path, max_side=1400, pts=None, aspect=1.1, max_aspect=2.0):
    """Averaged decimated RGB of the ortho (or a detail of it) + px->display map.

    These orthos are survey strips up to 1:20 — as a figure panel that is a
    hairline in which every detection lands on the same few pixels, so it
    verifies nothing.  When the raster is more lopsided than `max_aspect` the
    long axis is cropped to a window of the panel's own aspect ratio, slid to
    wherever the detections are densest (`pts`, native px column/row pairs).
    That trades "all of it, illegibly" for "the busiest few tens of metres,
    legibly", which is what a reviewer can actually check.
    Returns (rgb, sx, sy, col_off, row_off, detail) — detail=True if cropped.
    """
    with rasterio.open(ortho_path) as ds:
        W, H = ds.width, ds.height
        c0, r0, c1, r1 = 0, 0, W, H
        detail = False
        if max(W, H) / max(1, min(W, H)) > max_aspect:
            detail = True
            if H >= W:                          # tall strip: crop rows
                vh = int(min(H, max(1, W / aspect)))
                best = 0
                if pts is not None and len(pts):
                    rows = np.asarray(pts)[:, 1]
                    # centre the window on each detection; keep the fullest
                    cands = np.clip(rows - vh / 2.0, 0, max(0, H - vh))
                    counts = [((rows >= s) & (rows <= s + vh)).sum() for s in cands]
                    best = int(cands[int(np.argmax(counts))]) if len(cands) else 0
                r0, r1 = best, min(H, best + vh)
            else:                               # wide strip: crop columns
                vw = int(min(W, max(1, H * aspect)))
                best = 0
                if pts is not None and len(pts):
                    cols = np.asarray(pts)[:, 0]
                    cands = np.clip(cols - vw / 2.0, 0, max(0, W - vw))
                    counts = [((cols >= s) & (cols <= s + vw)).sum() for s in cands]
                    best = int(cands[int(np.argmax(counts))]) if len(cands) else 0
                c0, c1 = best, min(W, best + vw)
        vw, vh = c1 - c0, r1 - r0
        f = max(1.0, max(vw, vh) / float(max_side))
        oh = max(1, int(vh / f)); ow = max(1, int(vw / f))
        arr = ds.read(window=Window(c0, r0, vw, vh),
                      out_shape=(ds.count, oh, ow), resampling=Resampling.average)
    rgb = np.ascontiguousarray(arr[:3].transpose(1, 2, 0))
    if arr.shape[0] >= 4:
        rgb[arr[3] < ALPHA_ON] = 0
    return rgb, vw / float(ow), vh / float(oh), c0, r0, detail


def qa_figure(det, out_png, dive="", gsd=GSD, conf=CONF, log=print):
    """6 native-res crops spread across buckets + one whole-ortho overview.

    The point is falsifiability: if the boxes are not on organisms this figure
    shows it at native resolution with no stretching applied.
    """
    keep = det[det.excluded == ""].copy()
    if keep.empty:
        log("[ortho] QA: nothing kept, no figure")
        return None

    # spread the 6 crops over buckets: round-robin by descending confidence
    picks, pools = [], {}
    for b in BUCKET_ORDER:
        g = keep[keep.bucket == b].sort_values("conf", ascending=False)
        if len(g):
            pools[b] = list(g.itertuples(index=False))
    while len(picks) < 6 and pools:
        for b in list(pools):
            if not pools[b]:
                pools.pop(b); continue
            picks.append(pools[b].pop(0))
            if len(picks) == 6:
                break

    # the ortho carrying the most kept detections gets the overview panel
    top_ortho = keep.ortho.value_counts().idxmax()
    ov = keep[keep.ortho == top_ortho]

    fig = plt.figure(figsize=(19, 8.4), facecolor=BG)
    gs = gridspec.GridSpec(2, 5, figure=fig, width_ratios=[1, 1, 1, 1.5, 1.5],
                           wspace=0.10, hspace=0.16,
                           left=0.012, right=0.988, top=0.90, bottom=0.02)
    for i in range(6):
        ax = fig.add_subplot(gs[i // 3, i % 3])
        ax.set_facecolor(BG); ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        if i >= len(picks):
            ax.text(0.5, 0.5, "—", ha="center", va="center", color=MUT)
            continue
        d = picks[i]
        img, bx = _crop(d.ortho, (d.x1, d.y1, d.x2, d.y2))
        if img is None:
            ax.text(0.5, 0.5, "crop failed", ha="center", va="center", color=MUT)
            continue
        ax.imshow(img)                              # no stretch, native res
        col = BUCKET_COLOR.get(d.bucket, INK)
        ax.add_patch(plt.Rectangle((bx[0], bx[1]), bx[2] - bx[0], bx[3] - bx[1],
                                   fill=False, ec=col, lw=1.8))
        ax.set_title(f"{d.cls}  [{d.bucket}]  {d.conf:.2f}\n"
                     f"{d.seg}/{d.chunk}  {d.width_m*100:.0f}x{d.height_m*100:.0f} cm",
                     color=col, fontsize=8.5, pad=4)

    axo = fig.add_subplot(gs[:, 3:])
    axo.set_facecolor(BG)
    pts = np.column_stack([(ov.x1 + ov.x2) / 2.0, (ov.y1 + ov.y2) / 2.0])
    img, sx, sy, oc, orr, detail = _overview(top_ortho, pts=pts)
    axo.imshow(img)
    shown = 0
    for b, g in ov.groupby("bucket"):
        gx = ((g.x1 + g.x2) / 2.0 - oc) / sx
        gy = ((g.y1 + g.y2) / 2.0 - orr) / sy
        m = (gx >= 0) & (gx < img.shape[1]) & (gy >= 0) & (gy < img.shape[0])
        shown += int(m.sum())
        axo.scatter(gx[m], gy[m], s=26, facecolors="none", linewidths=1.2,
                    edgecolors=BUCKET_COLOR.get(b, INK),
                    label=f"{b} ({int(m.sum())})")
    axo.set_xlim(0, img.shape[1]); axo.set_ylim(img.shape[0], 0)
    axo.set_xticks([]); axo.set_yticks([])
    for s in axo.spines.values():
        s.set_color(GRID)
    tp = Path(top_ortho)
    with rasterio.open(top_ortho) as _ds:
        res = abs(_ds.transform.a)
    wide, tall = img.shape[1] * sx * res, img.shape[0] * sy * res
    what = (f"densest {wide:.0f}x{tall:.0f} m detail, {shown} of {len(ov)} det"
            if detail else f"whole mosaic, {shown} det, {wide:.0f}x{tall:.0f} m")
    axo.set_title(f"{tp.parent.parent.name}/{tp.parent.name} — {what} "
                  f"(decimated, averaged)", color=INK, fontsize=10)
    leg = axo.legend(loc="lower right", fontsize=8, facecolor=BG,
                     edgecolor=GRID, labelcolor=INK)
    leg.get_frame().set_alpha(0.85)

    fig.suptitle(f"{dive} — orthomosaic megafauna QA  "
                 f"(inference GSD {gsd*1000:.1f} mm/px, conf {conf}, "
                 f"crops at native resolution, no contrast stretch)",
                 color=INK, fontsize=12)
    out_png = Path(out_png); out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130, facecolor=BG)
    plt.close(fig)
    log(f"[ortho] QA figure -> {out_png}")
    return out_png


# ------------------------------------------------------------------- main ----
CACHE_COLS = ["ortho", "seg", "chunk", "cls", "bucket", "excluded", "conf",
              "x1", "y1", "x2", "y2", "easting", "northing",
              "width_m", "height_m"]
# box_imaged is added by apply_guards, after the cache — the cache holds raw
# inference output so the guards can be retuned without re-running the GPU.
OUT_COLS = CACHE_COLS + ["box_imaged"]


def _cache_key(ortho_path, gsd, conf, tile, overlap, min_alpha) -> str:
    """Cache filename that encodes every parameter the result depends on, so a
    re-run with a different --gsd/--conf never silently reuses stale boxes."""
    seg, chunk = ids_of(ortho_path)
    return (f"{seg}__{chunk}__g{round(gsd*1e4)}_c{round(conf*100)}"
            f"_t{tile}_o{round(overlap*100)}_a{round(min_alpha*100)}")


def run_dive(dive, root=ROOT, weights=WEIGHTS, conf=CONF, gsd=GSD, tile=TILE,
             overlap=OVERLAP, min_alpha=MIN_ALPHA, batch=BATCH, device=0,
             iou_nms=IOU_NMS, limit=None, qa=True, cache=True, refresh=False,
             max_box_m=MAX_BOX_M, min_box_alpha=MIN_BOX_ALPHA,
             log=print) -> dict:
    """Detect over one dive's orthomosaics, resumably.

    The full corpus is ~190 Gpx on a slow external mount and takes hours, so
    every ortho's result is cached under survey/fauna/ortho_cache/ the moment
    it finishes.  A re-run with the same parameters reuses those and only
    infers the orthos that are missing: an interrupted run (machine shutdown,
    a training job needing the whole GPU) resumes instead of restarting.
    --refresh forces re-inference, --no-cache disables it entirely.
    """
    from ultralytics import YOLO

    ws = Path(root) / f"{dive}_down.eprproj"
    out = ws / "survey" / "fauna"
    out.mkdir(parents=True, exist_ok=True)
    cdir = out / "ortho_cache"
    if cache:
        cdir.mkdir(parents=True, exist_ok=True)
    orthos = find_orthos(ws, limit=limit)
    if not orthos:
        raise FileNotFoundError(f"no chunk orthomosaics under {ws}")
    log(f"[ortho] {dive}: {len(orthos)} orthomosaics, gsd {gsd*1000:.1f} mm/px, "
        f"tile {tile} px, overlap {overlap:.0%}, conf {conf}")

    model = None
    parts, failed, no_tiles = [], [], []
    t0 = time.time()
    n_tiles = n_run = n_raw = n_cached = 0
    csv_p = out / "ortho_detections.csv"

    def _flush_csv():
        """Rewrite the aggregate CSV from everything finished so far.

        Cheap (a few thousand rows) and done after every ortho, so a reboot
        mid-run leaves a CSV that is complete for the mosaics that finished
        rather than absent or truncated mid-row.
        """
        d = (pd.concat(parts, ignore_index=True) if parts
             else pd.DataFrame(columns=OUT_COLS))
        tmp = csv_p.with_suffix(".csv.tmp")
        d.to_csv(tmp, index=False)
        tmp.replace(csv_p)                    # atomic: never a half-written CSV
        return d

    for i, p in enumerate(orthos, 1):
        key = _cache_key(p, gsd, conf, tile, overlap, min_alpha)
        ccsv, cjson = cdir / f"{key}.csv", cdir / f"{key}.stats.json"
        if cache and not refresh and ccsv.exists() and cjson.exists():
            d = pd.read_csv(ccsv).fillna({"excluded": "", "bucket": "unknown"})
            d = d.reindex(columns=CACHE_COLS)
            d.attrs.update(json.loads(cjson.read_text()))
            n_cached += 1
            log(f"[ortho] [{i}/{len(orthos)}] cached ({len(d)} det) {key}")
        else:
            log(f"[ortho] [{i}/{len(orthos)}] {p}")
            if model is None:
                model = YOLO(str(weights))
            try:
                d = detect_ortho(p, model, conf=conf, gsd=gsd, tile=tile,
                                 overlap=overlap, min_alpha=min_alpha,
                                 batch=batch, device=device, iou_nms=iou_nms,
                                 log=log)
            except Exception as exc:                  # keep going, report later
                log(f"[ortho] FAILED {p}: {type(exc).__name__}: {exc}")
                failed.append({"ortho": str(p),
                               "error": f"{type(exc).__name__}: {exc}"})
                continue
            if cache:                                 # checkpoint immediately
                d.to_csv(ccsv, index=False)
                cjson.write_text(json.dumps(
                    {k: d.attrs[k] for k in
                     ("n_tiles", "n_tiles_run", "n_raw", "runtime_s")}))
        n_tiles += d.attrs["n_tiles"]; n_run += d.attrs["n_tiles_run"]
        n_raw += d.attrs["n_raw"]
        if d.attrs["n_tiles_run"] == 0:
            # every tile fell below --min-alpha: a sliver ortho whose imaged
            # swath is too thin for any tile to reach the coverage floor.
            # Not a failure, but real seafloor that went unscanned — audit it.
            no_tiles.append(str(p))
        # guards here, per ortho: each box's alpha window is read exactly once,
        # and the cache above keeps the raw boxes so thresholds stay retunable
        parts.append(apply_guards(d, max_box_m=max_box_m,
                                  min_box_alpha=min_box_alpha, log=log))
        det = _flush_csv()

    det = _flush_csv()
    log(f"[ortho] {len(det)} detections -> {csv_p}")
    n_pts = to_geojson(det, out / "ortho_fauna_utm.geojson", log=log)
    qa_p = qa_figure(det, out / "ortho_fauna_qa.png", dive=dive, gsd=gsd,
                     conf=conf, log=log) if qa else None

    kept = det[det.excluded == ""]
    summary = {
        "dive": dive, "workspace": str(ws), "weights": str(weights),
        "gsd_m_px": gsd, "tile_px": tile, "overlap": overlap,
        "min_alpha": min_alpha, "conf": conf, "iou_nms": iou_nms,
        "max_box_m": max_box_m, "min_box_alpha": min_box_alpha,
        "n_orthos": len(orthos), "n_orthos_ok": len(parts),
        "n_orthos_failed": len(failed), "failed": failed,
        "n_orthos_no_tiles": len(no_tiles), "orthos_no_tiles": no_tiles,
        "n_orthos_from_cache": int(n_cached),
        "n_tiles_candidate": int(n_tiles), "n_tiles_inferred": int(n_run),
        "n_boxes_pre_nms": int(n_raw),
        "n_detections": int(len(det)), "n_detections_kept": int(len(kept)),
        "n_excluded_midwater": int((det.excluded == "midwater").sum()),
        "n_excluded_nonfauna": int((det.excluded == "nonfauna").sum()),
        "n_excluded_oversize": int((det.excluded == "oversize").sum()),
        "n_excluded_void": int((det.excluded == "void").sum()),
        "n_geojson_points": int(n_pts),
        "by_bucket_kept": kept.bucket.value_counts().to_dict(),
        "by_class_kept": kept.cls.value_counts().to_dict(),
        "excluded_classes": det[det.excluded != ""].cls.value_counts().to_dict(),
        "runtime_s": round(time.time() - t0, 1),
        "outputs": {"detections_csv": str(csv_p),
                    "points_geojson": str(out / "ortho_fauna_utm.geojson"),
                    "qa_png": str(qa_p) if qa_p else None,
                    "summary_json": str(out / "ortho_fauna_summary.json")},
    }
    (out / "ortho_fauna_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dives", nargs="*", default=list(DIVES))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--gsd", type=float, default=GSD,
                    help="inference ground sample distance, m/px (default 0.002)")
    ap.add_argument("--tile", type=int, default=TILE)
    ap.add_argument("--overlap", type=float, default=OVERLAP)
    ap.add_argument("--min-alpha", type=float, default=MIN_ALPHA)
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--iou-nms", type=float, default=IOU_NMS)
    ap.add_argument("--max-box-m", type=float, default=MAX_BOX_M,
                    help="flag boxes longer than this (m) as 'oversize'")
    ap.add_argument("--min-box-alpha", type=float, default=MIN_BOX_ALPHA,
                    help="flag boxes less imaged than this as 'void'")
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=None,
                    help="only the first N orthomosaics per dive (pilot mode)")
    ap.add_argument("--no-qa", action="store_true")
    ap.add_argument("--no-cache", action="store_true",
                    help="do not read or write survey/fauna/ortho_cache/")
    ap.add_argument("--refresh", action="store_true",
                    help="re-infer every ortho, overwriting the cache")
    ap.add_argument("--summary", default=None, help="pooled JSON across dives")
    a = ap.parse_args(argv)

    dev = int(a.device) if str(a.device).isdigit() else a.device
    pooled = {"generated": pd.Timestamp.now("UTC").isoformat(),
              "weights": a.weights, "gsd_m_px": a.gsd, "tile_px": a.tile,
              "overlap": a.overlap, "min_alpha": a.min_alpha, "conf": a.conf,
              "localisation": "orthomosaic box centre via raster affine "
                              "transform (photogrammetric georeference)",
              "dives": {}}
    for dive in a.dives:
        pooled["dives"][dive] = run_dive(
            dive, root=a.root, weights=a.weights, conf=a.conf, gsd=a.gsd,
            tile=a.tile, overlap=a.overlap, min_alpha=a.min_alpha,
            batch=a.batch, device=dev, iou_nms=a.iou_nms, limit=a.limit,
            qa=not a.no_qa, cache=not a.no_cache, refresh=a.refresh,
            max_box_m=a.max_box_m, min_box_alpha=a.min_box_alpha)
    if a.summary:
        sp = Path(a.summary); sp.parent.mkdir(parents=True, exist_ok=True)
        sp.write_text(json.dumps(pooled, indent=2, default=str))
        print(f"[ortho] pooled summary -> {sp}")
    for dive, s in pooled["dives"].items():
        print(f"[ortho] {dive}: {s['n_detections_kept']} kept "
              f"({s['by_bucket_kept']}) from {s['n_orthos_ok']}/{s['n_orthos']} "
              f"orthos in {s['runtime_s']:.0f} s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
