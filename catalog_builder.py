#!/usr/bin/env python3
"""catalog_builder.py — "print off screenshots of all the stuff into a PDF".

A *preview catalog*: after a pipeline run this module walks everything the
workspace has produced and prints it into one PDF a PI can flip through —
tracklines, sensor rasters, anomaly tables and figures, every chunk
orthomosaic as a thumbnail, the fauna computer-vision census with a sample of
its detection crops, and an inventory of every product file on disk.

It is a *reporting* module, not a pipeline one:

  * it NEVER computes science.  Counts, tiers and buckets are read straight out
    of the CSV/JSON products that the pipeline already wrote.  The only pixels
    it makes are down-sampled renderings of products that already exist
    (thumbnails, hillshades, crops, a trackline plot from the trackline
    GeoJSON) — screenshots, in other words.
  * every product family is optional.  A family that is not on disk contributes
    a single grey "not present" line instead of disappearing, so the reader can
    tell "we have not run that yet" from "that step failed".
  * it is Qt-free and importable from anything.

    build_catalog(workspace_dir, job_id="__whole__", log=print) -> str

returns the path of the PDF, written to

    <ws>/survey/catalog/catalog_<dive>_<YYYYmmdd_HHMM>.pdf

CLI:  python3 catalog_builder.py <workspace> [--job JOB]

Rendering
---------
The document is built as a block list (headings / paragraphs / tables /
figures / thumbnail grids), then rendered twice over:

  1. an HTML page under survey/catalog/, printed to PDF by headless Chrome
     (``--no-pdf-header-footer --print-to-pdf``) — the good-looking path;
  2. if Chrome is unavailable, the same block list is laid out onto matplotlib
     PdfPages — degraded, but it still produces a PDF with the same content.

Every generated image lands in ``survey/catalog/assets/`` (re-used across runs
when the source has not changed), so a catalog can be re-printed from the HTML
without re-reading a single GeoTIFF.

Memory discipline
-----------------
This runs on a WSL VM that has been OOM-killed before, so: exactly one image is
held in memory at a time, orthomosaic/DEM reads are always decimated through
rasterio's ``out_shape`` (never a full read), full frames are opened one at a
time and closed immediately, and crop rendering is capped.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional, Sequence

# --------------------------------------------------------------------------
# Tunables (kept conservative — see "Memory discipline" above)
# --------------------------------------------------------------------------

ORTHO_THUMB_PX = 600          # long side of a chunk-orthomosaic thumbnail
DEM_THUMB_PX = 520            # long side of a DEM hillshade thumbnail
RASTER_FIG_PX = 1100          # long side of a decimated sensor-raster read
MAX_DEM_THUMBS = 24           # DEMs are huge; the grid is illustrative
MAX_CROPS = 30                # hard cap on fauna detection crops
CROP_PX = 340                 # long side of a rendered crop
CROP_CONTEXT = 0.40           # context margin, fraction of bbox long side/side
MAX_FIGURES_PER_FAMILY = 12   # don't let one figure folder run away
MAX_INVENTORY_ROWS = 420      # inventory detail table cap
SHORTLIST_ROWS = 10           # "head" of the fish shortlist
COPY_VERBATIM_BYTES = 2_000_000   # bigger source PNGs get down-scaled to JPEG
BIG_IMAGE_LONG_SIDE = 2000

CHROME_CANDIDATES = (
    "/mnt/c/Program Files/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files (x86)/Google/Chrome/Application/chrome.exe",
    "/mnt/c/Program Files/Microsoft/Edge/Application/msedge.exe",
    "google-chrome",
    "chromium-browser",
    "chromium",
)

WHOLE = "__whole__"

TIER_ORDER = ("HIGH", "MODERATE", "SCREEN")
TIER_COLOUR = {"HIGH": "#d7263d", "MODERATE": "#e8871e", "SCREEN": "#c99700"}

# Directories collapsed to ONE inventory row instead of one row per file:
# engine internals and frame dumps, which would otherwise be 35k rows.
COLLAPSE_DIRS = ("project.files", "ortho_cache", "chunk_*", "segment_*",
                 "orthos", "frames", "catalog")
# Directories never walked at all.
SKIP_DIRS = ("catalog",)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _logger(log: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    if log is None:
        return lambda _m: None
    def _log(msg: str) -> None:
        try:
            log(str(msg))
        except Exception:                                           # noqa: BLE001
            pass
    return _log


def dive_id(workspace_dir) -> str:
    """"J1754" out of ".../J1754_down.eprproj", else the bundle stem."""
    name = Path(str(workspace_dir)).name
    m = re.search(r"(J\d{3,5})", name)
    if m:
        return m.group(1)
    return re.sub(r"\.eprproj$", "", name) or "workspace"


def _fmt_bytes(n: float) -> str:
    n = float(n or 0)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:,.0f} {unit}" if unit == "B" else f"{n:,.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} TB"


def _fmt_int(n) -> str:
    try:
        return f"{int(n):,}"
    except (TypeError, ValueError):
        return "—"


def _fmt_time(ts: float) -> str:
    try:
        return datetime.fromtimestamp(float(ts)).strftime("%Y-%m-%d %H:%M")
    except (TypeError, ValueError, OSError):
        return "—"


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(text).strip().lower()).strip("_") or "item"


def _sorted_pngs(directory: Optional[Path], limit: int = MAX_FIGURES_PER_FAMILY
                 ) -> list[Path]:
    if directory is None or not directory.is_dir():
        return []
    out = sorted([p for p in directory.iterdir()
                  if p.is_file() and p.suffix.lower() in (".png", ".jpg", ".jpeg")],
                 key=lambda p: p.name)
    return out[:limit]


def _read_csv(path, **kw):
    """pandas.read_csv that returns None instead of raising."""
    try:
        import pandas as pd
        if path is None or not Path(path).is_file():
            return None
        df = pd.read_csv(path, **kw)
        return df if len(df.columns) else None
    except Exception:                                               # noqa: BLE001
        return None


def _read_json(path):
    try:
        if path is None or not Path(path).is_file():
            return None
        return json.loads(Path(path).read_text(encoding="utf-8", errors="replace"))
    except Exception:                                               # noqa: BLE001
        return None


# --------------------------------------------------------------------------
# Document model — a flat block list, rendered by either backend
# --------------------------------------------------------------------------

def h2(title: str, anchor: str = "") -> dict:
    return {"k": "h2", "t": title, "id": anchor or _slug(title)}


def h3(title: str) -> dict:
    return {"k": "h3", "t": title}


def para(text: str) -> dict:
    return {"k": "p", "t": text}


def note(text: str) -> dict:
    return {"k": "note", "t": text}


def table(cols: Sequence[str], rows: Sequence[Sequence], title: str = "",
          foot: str = "", align_right: Sequence[int] = (),
          break_cols: Sequence[int] = ()) -> dict:
    return {"k": "table", "t": title, "cols": list(cols),
            "rows": [list(r) for r in rows], "foot": foot,
            "right": set(align_right), "brk": set(break_cols)}


def image(path, cap: str = "") -> dict:
    return {"k": "img", "src": str(path), "cap": cap}


def grid(items: Sequence[dict], cols: int = 3, title: str = "",
         foot: str = "") -> dict:
    return {"k": "grid", "items": list(items), "cols": cols, "t": title,
            "foot": foot}


# --------------------------------------------------------------------------
# Workspace context: where products live, and where assets go
# --------------------------------------------------------------------------

class _Ctx:
    """Path arithmetic + the asset cache for one catalog build."""

    def __init__(self, workspace_dir, job_id: str, log) -> None:
        self.ws = Path(str(workspace_dir)).resolve()
        self.job_id = str(job_id or WHOLE)
        self.log = log
        self.survey = self.ws / "survey"
        self.is_whole = self.job_id in (WHOLE, "whole", "", "None")
        self.scope = (self.survey if self.is_whole
                      else self.survey / "jobs" / self.job_id)
        self.out_dir = self.survey / "catalog"
        self.assets = self.out_dir / "assets"
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.assets.mkdir(parents=True, exist_ok=True)
        self.dive = dive_id(self.ws)
        self._frame_roots: Optional[list[Path]] = None

    # -- product lookup ---------------------------------------------------
    def dir(self, *names: str) -> Optional[Path]:
        """First existing product directory, scope before whole-survey."""
        bases = [self.scope] if self.is_whole else [self.scope, self.survey]
        for base in bases:
            for name in names:
                cand = base / name
                if cand.is_dir():
                    return cand
        return None

    def file(self, *rels: str) -> Optional[Path]:
        bases = [self.scope] if self.is_whole else [self.scope, self.survey]
        for base in bases:
            for rel in rels:
                cand = base / rel
                if cand.is_file():
                    return cand
        return None

    # -- asset cache ------------------------------------------------------
    def asset(self, name: str) -> Path:
        return self.assets / name

    def fresh(self, dst: Path, *sources) -> bool:
        """True when ``dst`` already exists and is newer than every source."""
        try:
            if not dst.is_file() or dst.stat().st_size < 64:
                return False
            dm = dst.stat().st_mtime
            for s in sources:
                if s and Path(s).exists() and Path(s).stat().st_mtime > dm:
                    return False
            return True
        except OSError:
            return False

    def stage(self, src) -> Optional[Path]:
        """Copy an existing figure into assets/ (down-scaling the big ones).

        Keeping every referenced image inside assets/ is what makes the HTML
        re-printable on its own, and the down-scale keeps a 30-figure catalog
        from turning into a 60 MB PDF.
        """
        src = Path(str(src))
        if not src.is_file():
            return None
        try:
            size = src.stat().st_size
        except OSError:
            return None
        if size <= COPY_VERBATIM_BYTES:
            dst = self.asset(f"fig_{_slug(src.parent.name)}_{src.name}")
            if not self.fresh(dst, src):
                try:
                    shutil.copyfile(src, dst)
                except OSError:
                    return None
            return dst
        dst = self.asset(f"fig_{_slug(src.parent.name)}_{_slug(src.stem)}.jpg")
        if self.fresh(dst, src):
            return dst
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            with Image.open(src) as im:
                im = im.convert("RGB")
                im.thumbnail((BIG_IMAGE_LONG_SIDE, BIG_IMAGE_LONG_SIDE),
                             Image.LANCZOS)
                im.save(dst, "JPEG", quality=86, optimize=True)
        except Exception:                                           # noqa: BLE001
            return None
        return dst

    # -- frames -----------------------------------------------------------
    def frame_roots(self) -> list[Path]:
        """Directories that may hold extracted frames, best guess first.

        Frames live outside the workspace on this rig (fast local disk for the
        detector, BIIGLE's store for annotation), and the location has moved
        between campaigns — so this probes the known roots rather than
        hard-coding one.
        """
        if self._frame_roots is not None:
            return self._frame_roots
        d = self.dive
        cands = [
            Path(f"/home/troyboland/biigle/storage/images/{d}/frames"),
            Path(f"/home/troyboland/biigle/storage/images/{d}/frames_full"),
            Path(f"/home/troyboland/epr_census/{d}/frames_full"),
            Path(f"/home/troyboland/epr_census/{d}/frames"),
        ]
        pg = self.dir("photogrammetry")
        if pg:
            cands += sorted(pg.glob("seg*/segment_*/frames"))[:40]
        fs = self.dir("frame_sets")
        if fs:
            cands += sorted(fs.glob("*/*/segment_*/frames"))[:40]
        self._frame_roots = [c for c in cands if c.is_dir()]
        return self._frame_roots

    def find_frame(self, filename: str) -> Optional[Path]:
        for root in self.frame_roots():
            cand = root / filename
            if cand.is_file():
                return cand
        return None


# --------------------------------------------------------------------------
# Raster rendering — always decimated, always alpha-aware
# --------------------------------------------------------------------------

def _stretch(rgb, valid, lo=2, hi=98):
    """Per-channel percentile stretch over imaged pixels only.

    Deep-sea orthos are dark and blue; without this every thumbnail is a black
    rectangle.  Presentation-only, and captioned as such.
    """
    import numpy as np
    out = rgb.astype("float32").copy()
    for c in range(out.shape[2]):
        v = out[:, :, c][valid]
        if v.size < 64:
            continue
        a, b = np.percentile(v, [lo, hi])
        if b <= a:
            continue
        out[:, :, c] = np.clip((out[:, :, c] - a) / (b - a), 0, 1) * 255.0
    return out


def ortho_thumb(src: Path, dst: Path, max_px: int = ORTHO_THUMB_PX,
                background: int = 16) -> Optional[dict]:
    """Decimated, contrast-stretched thumbnail of one orthomosaic.

    Alpha is honoured explicitly: several Metashape exports bake bright red (or
    white) under alpha=0, so outside-swath pixels are forced to the dark
    background instead of being trusted.  Only the decimated array ever exists
    in memory.
    """
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image

    try:
        with rasterio.open(str(src)) as ds:
            sc = max(1.0, max(ds.width, ds.height) / float(max_px))
            oh, ow = max(1, int(ds.height / sc)), max(1, int(ds.width / sc))
            n = ds.count
            bands = [1, 2, 3] if n >= 3 else [1]
            data = ds.read(bands, out_shape=(len(bands), oh, ow),
                           resampling=Resampling.average)
            alpha = None
            if n >= 4:
                alpha = ds.read(4, out_shape=(oh, ow),
                                resampling=Resampling.nearest)
            b = ds.bounds
            px_m = float(ds.res[0]) * sc
            crs = ds.crs.to_string() if ds.crs else ""
    except Exception:                                               # noqa: BLE001
        return None

    if len(bands) == 1:
        arr = np.repeat(data, 3, axis=0)
    else:
        arr = data
    rgb = np.transpose(arr, (1, 2, 0)).astype("float32")
    del data, arr
    if alpha is not None:
        valid = alpha > 0
        del alpha
    else:
        valid = rgb.sum(2) > 0
    cover = float(valid.mean())
    if cover <= 0.0005:
        return None
    rgb = _stretch(rgb, valid)
    rgb[~valid] = float(background)
    try:
        im = Image.fromarray(np.clip(rgb, 0, 255).astype("uint8"), "RGB")
        im.save(dst, "JPEG", quality=84, optimize=True)
        im.close()
    except Exception:                                               # noqa: BLE001
        return None
    finally:
        del rgb, valid
    return {"cover": cover, "w_m": float(b.right - b.left),
            "h_m": float(b.top - b.bottom), "px_mm": px_m * 1000.0,
            "crs": crs}


def _hillshade(z, ve: float = 2.0, az: float = 315.0, alt: float = 45.0):
    import numpy as np
    dy, dx = np.gradient(z * float(ve))
    slope = np.pi / 2.0 - np.arctan(np.hypot(dx, dy))
    aspect = np.arctan2(-dx, dy)
    az_r = np.radians(360.0 - az + 90.0)
    alt_r = np.radians(alt)
    hs = (np.sin(alt_r) * np.sin(slope) +
          np.cos(alt_r) * np.cos(slope) * np.cos(az_r - aspect))
    return np.clip(hs, 0.0, 1.0)


def dem_thumb(src: Path, dst: Path, max_px: int = DEM_THUMB_PX) -> Optional[dict]:
    """Decimated hillshade of a DEM, nodata rendered dark."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from PIL import Image

    try:
        with rasterio.open(str(src)) as ds:
            sc = max(1.0, max(ds.width, ds.height) / float(max_px))
            oh, ow = max(1, int(ds.height / sc)), max(1, int(ds.width / sc))
            z = ds.read(1, out_shape=(oh, ow),
                        resampling=Resampling.average).astype("float32")
            nod = ds.nodata
            px_m = float(ds.res[0]) * sc
    except Exception:                                               # noqa: BLE001
        return None
    bad = ~np.isfinite(z)
    if nod is not None:
        bad |= np.isclose(z, float(nod))
    bad |= z < -1e5
    good = ~bad
    if good.sum() < 64:
        return None
    z[bad] = float(np.nanmedian(z[good]))
    hs = _hillshade(z)
    relief = float(np.nanmax(z[good]) - np.nanmin(z[good]))
    img = (hs * 235.0 + 10.0)
    img[bad] = 16.0
    try:
        im = Image.fromarray(np.clip(img, 0, 255).astype("uint8"), "L")
        im.save(dst, "JPEG", quality=84, optimize=True)
        im.close()
    except Exception:                                               # noqa: BLE001
        return None
    finally:
        del z, hs, img, bad, good
    return {"relief_m": relief, "px_mm": px_m * 1000.0}


def raster_figure(src: Path, dst: Path, title: str, units: str = "",
                  max_px: int = RASTER_FIG_PX) -> Optional[dict]:
    """A single-band raster (sensor grid / depth grid) as a colour-mapped PNG."""
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        with rasterio.open(str(src)) as ds:
            sc = max(1.0, max(ds.width, ds.height) / float(max_px))
            oh, ow = max(1, int(ds.height / sc)), max(1, int(ds.width / sc))
            z = ds.read(1, out_shape=(oh, ow),
                        resampling=Resampling.average).astype("float32")
            nod = ds.nodata
            b = ds.bounds
    except Exception:                                               # noqa: BLE001
        return None
    if nod is not None:
        z[np.isclose(z, float(nod))] = np.nan
    z[~np.isfinite(z)] = np.nan
    if np.isfinite(z).sum() < 16:
        return None
    vmin = float(np.nanpercentile(z, 2))
    vmax = float(np.nanpercentile(z, 98))
    if vmax <= vmin:
        vmin, vmax = float(np.nanmin(z)), float(np.nanmax(z)) or vmin + 1.0
    fw, fh = map_figsize(float(b.right - b.left), float(b.top - b.bottom))
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(fw + 1.9, fh + 0.9), dpi=140)
    ax.set_facecolor("#e9edf1")
    im = ax.imshow(z, cmap="magma", vmin=vmin, vmax=vmax,
                   extent=[b.left, b.right, b.bottom, b.top], origin="upper",
                   interpolation="nearest")
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color="#22303c")
    ax.tick_params(labelsize=7, colors="#66757f")
    ax.ticklabel_format(style="plain", useOffset=False)
    _thin_ticks(ax, fw, fh)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03)
    cb.set_label(units or title.split("·")[-1].strip(), fontsize=8,
                 color="#55636e")
    cb.ax.tick_params(labelsize=7, colors="#66757f")
    fig.tight_layout()
    fig.savefig(dst, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    del z
    return {"vmin": vmin, "vmax": vmax}


# --------------------------------------------------------------------------
# Section 1 — cover facts
# --------------------------------------------------------------------------

def _count_frames_on_disk(ctx: _Ctx) -> tuple[int, str]:
    roots = ctx.frame_roots()
    if not roots:
        return 0, "no frame directory found"
    root = roots[0]
    try:
        n = sum(1 for p in os.scandir(root)
                if p.is_file() and p.name.lower().endswith((".jpg", ".jpeg", ".png")))
    except OSError:
        return 0, str(root)
    return n, str(root)


def _segment_count(ctx: _Ctx, density) -> tuple[int, str]:
    pg = ctx.dir("photogrammetry")
    if pg:
        segs = [p for p in pg.glob("seg*") if p.is_dir()]
        if segs:
            return len(segs), "photogrammetry segments"
    if density is not None and "seg" in density.columns:
        return int(density["seg"].nunique()), "segments in the fauna census"
    interp = ctx.ws / "inputs" / "interp_full.csv"
    if interp.is_file():
        return 0, "no segmentation product on disk"
    return 0, "unknown"


def collect_facts(ctx: _Ctx) -> dict:
    """Headline numbers — every one of them READ, never recomputed."""
    import pandas as pd  # noqa: F401  (ensures a clear error if pandas is absent)

    facts: dict = {"dive": ctx.dive, "ws": str(ctx.ws),
                   "generated": datetime.now().strftime("%Y-%m-%d %H:%M"),
                   "scope": ("whole trackline" if ctx.is_whole
                             else f"job {ctx.job_id}")}

    density = _read_csv(ctx.file("fauna/fauna_density.csv"))
    facts["_density"] = density

    n_frames, froot = _count_frames_on_disk(ctx)
    facts["frames_disk"] = n_frames
    facts["frames_root"] = froot
    facts["frames_census"] = int(len(density)) if density is not None else 0

    n_seg, seg_src = _segment_count(ctx, density)
    facts["segments"] = n_seg
    facts["segments_src"] = seg_src

    facts["orthos"] = len(find_orthos(ctx))
    facts["dems"] = len(find_dems(ctx))

    win = _read_csv(ctx.file("anomaly/anomaly_windows_all.csv"))
    facts["_windows"] = win
    tiers: dict[str, int] = {}
    if win is not None and "confidence_tier" in win.columns:
        counts = win["confidence_tier"].fillna("—").value_counts()
        tiers = {str(k): int(v) for k, v in counts.items()}
    facts["tiers"] = tiers
    facts["windows"] = int(len(win)) if win is not None else 0
    sites = _read_csv(ctx.file("anomaly/anomalous_sites.csv"))
    facts["_sites"] = sites
    facts["sites"] = int(len(sites)) if sites is not None else 0

    det = _read_csv(ctx.file("fauna/fathomnet_detections.csv"))
    buckets: dict[str, int] = {}
    n_kept = n_excl = 0
    if det is not None and "bucket" in det.columns:
        if "excluded" in det.columns:
            ex = det["excluded"].fillna("").astype(str).str.strip()
            kept = det[ex == ""]
            n_excl = int(len(det) - len(kept))
        else:
            kept = det
        n_kept = int(len(kept))
        buckets = {str(k): int(v) for k, v in
                   kept["bucket"].fillna("—").value_counts().items()}
    facts["_det"] = det
    facts["buckets"] = buckets
    facts["det_kept"] = n_kept
    facts["det_excluded"] = n_excl
    return facts


def section_cover(ctx: _Ctx, facts: dict) -> list[dict]:
    tier_txt = ", ".join(f"{t} {_fmt_int(facts['tiers'][t])}"
                         for t in TIER_ORDER if t in facts["tiers"]) or "—"
    extra = [t for t in facts["tiers"] if t not in TIER_ORDER]
    if extra:
        tier_txt += ", " + ", ".join(f"{t} {_fmt_int(facts['tiers'][t])}"
                                     for t in sorted(extra))
    bucket_txt = ", ".join(f"{b} {_fmt_int(n)}" for b, n in
                           sorted(facts["buckets"].items(),
                                  key=lambda kv: -kv[1])) or "—"
    rows = [
        ("Frames on disk", _fmt_int(facts["frames_disk"]),
         facts["frames_root"]),
        ("Frames in the fauna census", _fmt_int(facts["frames_census"]),
         "rows of survey/fauna/fauna_density.csv"),
        ("Segments", _fmt_int(facts["segments"]), facts["segments_src"]),
        ("Chunk orthomosaics", _fmt_int(facts["orthos"]),
         "survey/photogrammetry/*/chunk_*/orthomosaic.tif"),
        ("Chunk DEMs", _fmt_int(facts["dems"]), "…/dem.tif"),
        ("Anomaly windows", _fmt_int(facts["windows"]), tier_txt),
        ("Anomalous sites", _fmt_int(facts["sites"]),
         "survey/anomaly/anomalous_sites.csv"),
        ("Fauna detections (kept)", _fmt_int(facts["det_kept"]), bucket_txt),
        ("Fauna detections (excluded)", _fmt_int(facts["det_excluded"]),
         "midwater / oversize / non-fauna filters"),
    ]
    return [{"k": "cover", "facts": facts},
            table(("Quantity", "Count", "Source"), rows,
                  title="Headline counts", align_right=(1,))]


# --------------------------------------------------------------------------
# Section 2 — tracklines
# --------------------------------------------------------------------------

def map_figsize(w_data: float, h_data: float, area: float = 30.0,
                min_side: float = 2.1, max_w: float = 8.4,
                max_h: float = 8.6) -> tuple[float, float]:
    """Figure size whose box matches the data aspect, at roughly constant area.

    Every map here uses an equal aspect (they are maps), and these dives are
    7 km-long ribbons: a fixed 8×6 canvas would be nine tenths white space with
    a hairline down the middle.  Sizing the canvas to the data instead keeps
    the ink density sane, and the print CSS caps the height on the page.
    """
    if not (w_data > 0 and h_data > 0):
        return (7.2, 5.4)
    ar = float(w_data) / float(h_data)
    h = math.sqrt(area / ar)
    w = ar * h
    if h > max_h:
        h, w = max_h, max_h * ar
    if w > max_w:
        w, h = max_w, max_w / ar
    return (max(min_side, w), max(min_side, h))


def _thin_ticks(ax, fig_w: float, fig_h: float) -> None:
    """Fewer ticks on a narrow/short axis, so labels never collide."""
    from matplotlib.ticker import MaxNLocator
    ax.xaxis.set_major_locator(MaxNLocator(nbins=max(2, int(fig_w * 0.9)),
                                           prune=None))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=max(2, int(fig_h * 0.9))))


def _load_track(path) -> Optional["object"]:
    """Coordinate array out of a trackline GeoJSON (LineString/MultiLineString)."""
    import numpy as np
    gj = _read_json(path)
    if not gj:
        return None
    lines = []
    for feat in gj.get("features") or []:
        geom = feat.get("geometry") or {}
        gtype, coords = geom.get("type"), geom.get("coordinates")
        if gtype == "LineString" and coords:
            lines.append(np.asarray(coords, dtype="float64")[:, :2])
        elif gtype == "MultiLineString" and coords:
            for part in coords:
                if part:
                    lines.append(np.asarray(part, dtype="float64")[:, :2])
    if not lines:
        return None
    return lines


def render_nav_trackline(ctx: _Ctx, geojson: Path, dst: Path) -> bool:
    """Plan view of the trackline GeoJSON, coloured by along-track progress."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection

    lines = _load_track(geojson)
    if not lines:
        return False
    allpts = np.vstack(lines)
    fw, fh = map_figsize(float(np.ptp(allpts[:, 0])), float(np.ptp(allpts[:, 1])))
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(fw + 1.3, fh + 0.9), dpi=140)
    for pts in lines:
        if len(pts) < 2:
            continue
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        prog = np.linspace(0, 1, len(segs))
        lc = LineCollection(segs, cmap="viridis", linewidths=1.1)
        lc.set_array(prog)
        ax.add_collection(lc)
    # annotated markers rather than a legend box: on a ribbon-shaped dive a
    # legend has nowhere to sit that is not on top of the track
    for pt, mark, colour, lbl in ((allpts[0], "o", "#2ecc71", "start"),
                                  (allpts[-1], "s", "#e74c3c", "end")):
        ax.plot(pt[0], pt[1], mark, ms=7, mfc=colour, mec="#14361f", zorder=5)
        ax.annotate(lbl, xy=(pt[0], pt[1]), xytext=(7, 6),
                    textcoords="offset points", fontsize=8, color="#22303c",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white",
                              ec="none", alpha=0.75), zorder=6)
    ax.set_aspect("equal")
    ax.margins(0.06)
    ax.set_title(f"{ctx.dive} — ROV trackline\n(colour = dive progress)",
                 fontsize=10.5, color="#22303c")
    ax.set_xlabel("easting / longitude", fontsize=8, color="#66757f")
    ax.set_ylabel("northing / latitude", fontsize=8, color="#66757f")
    ax.tick_params(labelsize=7, colors="#66757f")
    ax.ticklabel_format(style="plain", useOffset=False)
    _thin_ticks(ax, fw, fh)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.grid(alpha=0.18, lw=0.5)
    fig.tight_layout()
    fig.savefig(dst, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def render_anomaly_trackline(ctx: _Ctx, track: Optional[Path], segs: Path,
                             sites: Optional[Path], dst: Path) -> bool:
    """Trackline with the anomaly segments drawn over it, coloured by tier."""
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    gj = _read_json(segs)
    if not gj:
        return False
    by_tier: dict[str, list] = {}
    for feat in gj.get("features") or []:
        tier = str((feat.get("properties") or {}).get("confidence")
                   or (feat.get("properties") or {}).get("confidence_tier") or "SCREEN")
        geom = feat.get("geometry") or {}
        coords = geom.get("coordinates")
        if geom.get("type") == "LineString" and coords and len(coords) > 1:
            by_tier.setdefault(tier, []).append(
                np.asarray(coords, dtype="float64")[:, :2])
    if not by_tier:
        return False
    base = _load_track(track) if track else None
    extent = np.vstack([p for parts in by_tier.values() for p in parts] +
                       (list(base) if base else []))
    fw, fh = map_figsize(float(np.ptp(extent[:, 0])), float(np.ptp(extent[:, 1])))
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(fw + 2.6, fh + 0.9), dpi=140)
    if base:
        for pts in base:
            ax.plot(pts[:, 0], pts[:, 1], color="#9aa9b5", lw=0.6, zorder=1)
    handles = []
    for i, tier in enumerate(TIER_ORDER + tuple(
            t for t in sorted(by_tier) if t not in TIER_ORDER)):
        if tier not in by_tier:
            continue
        col = TIER_COLOUR.get(tier, "#6c7a89")
        ax.add_collection(LineCollection(by_tier[tier], colors=col,
                                         linewidths=2.6, zorder=3 + i,
                                         capstyle="round"))
        handles.append(Line2D([0], [0], color=col, lw=3,
                              label=f"{tier} ({len(by_tier[tier])})"))
    sgj = _read_json(sites) if sites else None
    n_sites = 0
    for feat in (sgj or {}).get("features") or []:
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Point" and geom.get("coordinates"):
            x, y = geom["coordinates"][:2]
            ax.plot(x, y, marker="o", ms=9, mfc="none", mec="#12212e", mew=1.3,
                    zorder=9)
            n_sites += 1
    if n_sites:
        handles.append(Line2D([0], [0], marker="o", mfc="none", mec="#12212e",
                              ls="", label=f"anomalous site ({n_sites})"))
    if base:
        handles.append(Line2D([0], [0], color="#9aa9b5", lw=1,
                              label="ROV trackline"))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.margins(0.06)
    ax.set_title(f"{ctx.dive} — anomaly trackline\nby confidence tier",
                 fontsize=10.5, color="#22303c")
    ax.tick_params(labelsize=7, colors="#66757f")
    ax.ticklabel_format(style="plain", useOffset=False)
    _thin_ticks(ax, fw, fh)
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(30)
        lbl.set_ha("right")
    ax.grid(alpha=0.18, lw=0.5)
    # legend outside the axes — inside, it would cover the track
    ax.legend(handles=handles, fontsize=8, framealpha=0.9,
              loc="upper left", bbox_to_anchor=(1.02, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(dst, dpi=140, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return True


def render_spectrum_trackline(ctx: _Ctx, interp: Path, dst: Path
                              ) -> Optional[list[str]]:
    """One plan-view panel per sensor channel, log-scaled, dots on the track.

    This is the "spectrum trackline" view the 3-D viewer shows: the same
    interp_full.csv the pipeline already wrote, plotted — no new science.
    """
    import numpy as np
    import pandas as pd
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        df = pd.read_csv(interp)
    except Exception:                                               # noqa: BLE001
        return None
    xcol, ycol = ("easting", "northing") if {"easting", "northing"} <= set(df.columns) \
        else ("lon", "lat")
    if xcol not in df.columns or ycol not in df.columns:
        return None
    skip = {"unix_time", "lat", "lon", "alt", "water_depth", "heading", "pitch",
            "roll", "easting", "northing", "depth", "utm_zone", "timestamp_iso"}
    chans = [c for c in df.columns
             if c not in skip and pd.api.types.is_numeric_dtype(df[c])
             and df[c].notna().sum() > 32]
    if not chans:
        return None
    df = df.dropna(subset=[xcol, ycol])
    if len(df) < 8:
        return None
    x, y = df[xcol].to_numpy(), df[ycol].to_numpy()
    # panel size follows the track's own shape, then the column count is
    # chosen so the whole sheet lands near a 1.4:1 page-friendly aspect
    pw, ph = map_figsize(float(np.ptp(x)), float(np.ptp(y)), area=9.0,
                         min_side=1.25, max_w=4.0, max_h=5.4)
    pw += 0.85                                  # colour bar + labels
    ncol = min(range(1, len(chans) + 1),
               key=lambda n: abs((n * pw) /
                                 (math.ceil(len(chans) / n) * (ph + 0.4)) - 1.4))
    nrow = int(math.ceil(len(chans) / ncol))
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(nrow, ncol,
                             figsize=(pw * ncol, (ph + 0.4) * nrow),
                             dpi=135, squeeze=False)
    for i, chan in enumerate(chans):
        ax = axes[i // ncol][i % ncol]
        v = df[chan].to_numpy(dtype="float64")
        finite = np.isfinite(v)
        pos = finite & (v > 0)
        # Log only where it buys something: gas channels span decades, but
        # log10(salinity) would just relabel a flat field.
        span = 0.0
        if pos.sum() > 32:
            lo_p, hi_p = np.nanpercentile(np.where(pos, v, np.nan), [2, 98])
            span = (hi_p / lo_p) if lo_p > 0 else 0.0
        if span > 20.0:
            c = np.log10(np.where(pos, v, np.nan))
            lab = f"log₁₀ {chan}"
        else:
            c = np.where(finite, v, np.nan)
            lab = chan
        ax.plot(x, y, color="#dde3e9", lw=0.6, zorder=1)
        lo, hi = np.nanpercentile(c, [2, 98])
        sc = ax.scatter(x, y, c=c, cmap="turbo", s=2.2, vmin=lo,
                        vmax=hi if hi > lo else lo + 1e-6, zorder=2)
        ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(chan, fontsize=9.5, color="#22303c")
        cb = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(lab, fontsize=7, color="#55636e")
        cb.ax.tick_params(labelsize=6, colors="#66757f")
    for j in range(len(chans), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"{ctx.dive} — sensor spectrum tracklines", fontsize=12,
                 color="#22303c")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(dst, dpi=135, facecolor="white", bbox_inches="tight")
    plt.close(fig)
    return chans


def section_tracklines(ctx: _Ctx) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("2 · Tracklines", "tracklines")]
    found = 0

    # -- nav trackline ---------------------------------------------------
    nav_dir = ctx.dir("nav_trackline")
    pngs = _sorted_pngs(nav_dir, 4)
    if pngs:
        for p in pngs:
            staged = ctx.stage(p)
            if staged:
                blocks.append(image(staged, f"nav trackline · {p.name}"))
                found += 1
    else:
        gj = ctx.file("nav_trackline/trackline.geojson")
        if gj is None and nav_dir:
            cand = sorted(nav_dir.glob("*trackline*.geojson"))
            gj = cand[0] if cand else None
        if gj:
            dst = ctx.asset("nav_trackline.png")
            ok = ctx.fresh(dst, gj) or render_nav_trackline(ctx, gj, dst)
            if ok:
                blocks.append(image(dst, f"ROV trackline, rendered from "
                                         f"{gj.name}"))
                found += 1
            else:
                blocks.append(note("Nav trackline: GeoJSON present but could "
                                   "not be plotted."))
        else:
            blocks.append(note("Nav trackline — not present "
                               "(survey/nav_trackline/trackline.geojson)."))

    # -- spectrum tracklines ---------------------------------------------
    spec_dir = ctx.dir("spectrum_trackline")
    spec_pngs = _sorted_pngs(spec_dir, 6)
    if spec_pngs:
        for p in spec_pngs:
            staged = ctx.stage(p)
            if staged:
                blocks.append(image(staged, f"spectrum trackline · {p.name}"))
                found += 1
    else:
        interp = ctx.ws / "inputs" / "interp_full.csv"
        if interp.is_file():
            dst = ctx.asset("spectrum_trackline.png")
            chans = None
            if ctx.fresh(dst, interp):
                chans = ["(cached)"]
            else:
                chans = render_spectrum_trackline(ctx, interp, dst)
            if chans:
                blocks.append(image(
                    dst, "Sensor spectrum tracklines, all channels — rendered "
                         "from inputs/interp_full.csv (no spectrum_trackline "
                         "product on disk)."))
                found += 1
            else:
                blocks.append(note("Spectrum tracklines — not present, and "
                                   "interp_full.csv carries no usable sensor "
                                   "channels."))
        else:
            blocks.append(note("Spectrum tracklines — not present "
                               "(survey/spectrum_trackline/*.png)."))

    # -- anomaly trackline ------------------------------------------------
    an_dir = ctx.dir("anomaly_trackline")
    an_pngs = _sorted_pngs(an_dir, 4)
    if an_pngs:
        for p in an_pngs:
            staged = ctx.stage(p)
            if staged:
                blocks.append(image(staged, f"anomaly trackline · {p.name}"))
                found += 1
    else:
        segs = ctx.file("anomaly/anomaly_segments_utm.geojson")
        if segs is None:
            adir = ctx.dir("anomaly")
            cand = sorted(adir.glob("*segments*.geojson")) if adir else []
            segs = cand[0] if cand else None
        if segs:
            dst = ctx.asset("anomaly_trackline.png")
            track = ctx.file("nav_trackline/trackline.geojson")
            sites = ctx.file("anomaly/anomalous_sites_utm.geojson")
            ok = ctx.fresh(dst, segs) or render_anomaly_trackline(
                ctx, track, segs, sites, dst)
            if ok:
                blocks.append(image(dst, "Anomaly segments over the trackline, "
                                         f"rendered from {segs.name}"))
                found += 1
            else:
                blocks.append(note("Anomaly trackline: GeoJSON present but "
                                   "could not be plotted."))
        else:
            blocks.append(note("Anomaly trackline — not present "
                               "(survey/anomaly/anomaly_segments_utm.geojson)."))
    return blocks, (f"{found} figure(s)" if found else "not present")


# --------------------------------------------------------------------------
# Section 3 — sensor / depth rasters
# --------------------------------------------------------------------------

_RUN_RE = re.compile(r"^run_\d+$")


def find_rasters(ctx: _Ctx) -> list[tuple[str, Path]]:
    """(label, path) for the newest GeoTIFF of each sensor/depth raster.

    Raster products nest as ``<family>/[<family>/]<channel>/run_NNN/x.tif``
    (the doubled family name is a migration artefact), so the label is built
    from the directory chain with run dirs and repeats of the family name
    dropped — otherwise five sensor channels collapse into one "sensor_2d".
    """
    out: dict[str, Path] = {}
    for family in ("sensor_2d", "nav_depth", "rasters", "nav_2d",
                   "depth_slice_geotiffs"):
        base = ctx.dir(family)
        if not base:
            continue
        for tif in base.rglob("*.tif"):
            try:
                if not tif.is_file() or tif.stat().st_size < 2048:
                    continue
            except OSError:
                continue
            parts = [p for p in tif.relative_to(base).parts[:-1]
                     if not _RUN_RE.match(p) and p != family]
            label = f"{family} · {'/'.join(parts)}" if parts else family
            prev = out.get(label)
            if prev is None or tif.stat().st_mtime > prev.stat().st_mtime:
                out[label] = tif
    return sorted(out.items())


def section_rasters(ctx: _Ctx) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("3 · Sensor and depth rasters", "rasters")]
    rasters = find_rasters(ctx)
    if not rasters:
        blocks.append(note("Sensor / depth rasters — not present "
                           "(survey/sensor_2d, survey/nav_depth)."))
        return blocks, "not present"
    blocks.append(para(f"{len(rasters)} gridded raster product(s); the newest "
                       f"run of each is shown."))
    made = 0
    for label, tif in rasters:
        dst = ctx.asset(f"raster_{_slug(label)}.png")
        ok = ctx.fresh(dst, tif) or bool(raster_figure(tif, dst, label))
        if ok:
            blocks.append(image(dst, f"{label} — {tif.name}"))
            made += 1
        else:
            blocks.append(note(f"{label}: {tif.name} could not be rendered."))
    return blocks, f"{made} raster(s)"


# --------------------------------------------------------------------------
# Section 4 — anomaly detection
# --------------------------------------------------------------------------

def section_anomaly(ctx: _Ctx, facts: dict) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("4 · Anomaly detection", "anomaly")]
    win = facts.get("_windows")
    if win is None:
        blocks.append(note("Anomaly windows — not present "
                           "(survey/anomaly/anomaly_windows_all.csv)."))
    else:
        cls_col = "anomaly_class" if "anomaly_class" in win.columns else None
        tier_col = "confidence_tier" if "confidence_tier" in win.columns else None
        if tier_col and cls_col:
            piv = (win.groupby([cls_col, tier_col]).size().unstack(fill_value=0))
            tiers = [t for t in TIER_ORDER if t in piv.columns]
            tiers += [c for c in piv.columns if c not in tiers]
            piv = piv[tiers]
            piv["Total"] = piv.sum(axis=1)
            piv = piv.sort_values("Total", ascending=False)
            rows = [[str(idx)] + [_fmt_int(v) for v in piv.loc[idx].tolist()]
                    for idx in piv.index]
            rows.append(["ALL CLASSES"] +
                        [_fmt_int(v) for v in piv.sum(axis=0).tolist()])
            blocks.append(table(["Anomaly class"] + [str(c) for c in piv.columns],
                                rows, title="Windows by class and confidence tier",
                                align_right=tuple(range(1, len(piv.columns) + 1))))
        elif tier_col:
            counts = win[tier_col].value_counts()
            blocks.append(table(("Confidence tier", "Windows"),
                                [[str(k), _fmt_int(v)] for k, v in counts.items()],
                                title="Windows by confidence tier",
                                align_right=(1,)))
        else:
            blocks.append(para(f"{_fmt_int(len(win))} anomaly windows "
                               f"(no confidence_tier column)."))

    combo = _read_csv(ctx.file("anomaly/anomaly_combination_summary.csv"))
    if combo is not None and len(combo):
        cols = [c for c in ("channel_combination", "indicator_signature",
                            "confidence_tier", "window_count", "distinct_sites",
                            "max_evidence_score") if c in combo.columns]
        if cols:
            sort_col = "window_count" if "window_count" in cols else cols[-1]
            top = combo.sort_values(sort_col, ascending=False).head(14)
            blocks.append(table([c.replace("_", " ") for c in cols],
                                top[cols].astype(str).values.tolist(),
                                title="Channel-combination summary (top 14 by "
                                      "window count)"))

    sites = facts.get("_sites")
    if sites is not None and len(sites):
        cols = [c for c in ("site_id", "lat", "lon", "window_count",
                            "high_windows", "best_tier", "max_evidence_score",
                            "channels") if c in sites.columns]
        sort_col = "window_count" if "window_count" in sites.columns else cols[0]
        top = sites.sort_values(sort_col, ascending=False).head(12)
        rows = []
        for _, r in top.iterrows():
            row = []
            for c in cols:
                v = r[c]
                row.append(f"{v:.5f}" if c in ("lat", "lon") and
                           isinstance(v, float) else str(v))
            rows.append(row)
        blocks.append(table([c.replace("_", " ") for c in cols], rows,
                            title="Anomalous sites (top 12 by window count)"))

    # existing anomaly figures: the detector-matrix figure set + any PNG in the
    # anomaly product folder
    figs: list[Path] = []
    adir = ctx.dir("anomaly")
    if adir:
        figs += _sorted_pngs(adir, 6)
    gm = ctx.ws / "inputs" / "grapher_matrix_figs"
    if gm.is_dir():
        variant = None
        for pref in ("masked", "nomask_both", "nomask_log", "nomask_ceiling"):
            if (gm / pref).is_dir():
                variant = gm / pref
                break
        if variant:
            wanted = sorted(
                [p for p in variant.glob("CONCURRENT_*.png")] +
                [p for p in variant.glob("*_anomaly_classification.png")] +
                [p for p in variant.glob("*_consensus_*.png")])
            figs += wanted[:MAX_FIGURES_PER_FAMILY]
            blocks.append(para(
                f"Detector-matrix figures below come from the "
                f"<code>{variant.name}</code> strategy family "
                f"(inputs/grapher_matrix_figs/{variant.name}/)."))
    if figs:
        for p in figs:
            staged = ctx.stage(p)
            if staged:
                blocks.append(image(staged, p.name))
    else:
        blocks.append(note("Anomaly figures — none on disk "
                           "(survey/anomaly/*.png, inputs/grapher_matrix_figs/)."))
    n_fig = sum(1 for b in blocks if b["k"] == "img")
    if win is None and not n_fig:
        return blocks, "not present"
    return blocks, f"{_fmt_int(facts.get('windows', 0))} windows, {n_fig} figure(s)"


# --------------------------------------------------------------------------
# Section 5 — photogrammetry
# --------------------------------------------------------------------------

def find_orthos(ctx: _Ctx) -> list[Path]:
    pg = ctx.dir("photogrammetry")
    if not pg:
        return []
    seen: dict[str, Path] = {}
    for pat in ("*/chunk_*/orthomosaic.tif", "*/chunks/chunk_*/orthomosaic.tif",
                "chunk_*/orthomosaic.tif", "*/*/chunk_*/orthomosaic.tif"):
        for p in pg.glob(pat):
            try:
                if p.is_file() and p.stat().st_size > 500_000:
                    seen[str(p)] = p
            except OSError:
                continue
    return sorted(seen.values(), key=lambda p: str(p))


def find_dems(ctx: _Ctx) -> list[Path]:
    pg = ctx.dir("photogrammetry")
    if not pg:
        return []
    seen: dict[str, Path] = {}
    for pat in ("*/chunk_*/dem.tif", "*/chunks/chunk_*/dem.tif",
                "chunk_*/dem.tif", "*/*/chunk_*/dem.tif"):
        for p in pg.glob(pat):
            try:
                if p.is_file() and p.stat().st_size > 200_000:
                    seen[str(p)] = p
            except OSError:
                continue
    return sorted(seen.values(), key=lambda p: str(p))


def _chunk_label(p: Path) -> str:
    parts = [x for x in p.parts[:-1] if x != "chunks"]
    return "/".join(parts[-2:]) if len(parts) >= 2 else p.parent.name


def section_photogrammetry(ctx: _Ctx) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("5 · Photogrammetry", "photogrammetry")]
    orthos = find_orthos(ctx)
    dems = find_dems(ctx)
    pg = ctx.dir("photogrammetry")
    if not orthos and not dems:
        blocks.append(note("Photogrammetry — not present "
                           "(no survey/photogrammetry/*/chunk_*/orthomosaic.tif)."))
        return blocks, "not present"

    merged = None
    if pg:
        for cand in ("merged/preview_ortho_merged.png", "merged/ortho_merged.tif"):
            m = pg / cand
            if m.is_file():
                merged = m
                break
    if merged is not None:
        if merged.suffix.lower() == ".png":
            staged = ctx.stage(merged)
            if staged:
                blocks.append(image(staged, "Merged survey orthomosaic "
                                            "(preview render)"))
        else:
            dst = ctx.asset("ortho_merged.jpg")
            if ctx.fresh(dst, merged) or ortho_thumb(merged, dst, 1500):
                blocks.append(image(dst, "Merged survey orthomosaic "
                                         "(decimated, contrast-stretched)"))

    gallery = ctx.dir("photogrammetry")
    gal_dir = (gallery / "ortho_gallery") if gallery else None
    contact = (gallery / "mosaics_contact_sheet.png") if gallery else None

    blocks.append(para(
        f"{_fmt_int(len(orthos))} chunk orthomosaic(s) on disk. Each thumbnail "
        f"below is a decimated (Resampling.average) read at ~{ORTHO_THUMB_PX} px "
        f"with a 2–98 % per-channel contrast stretch; the stretch is "
        f"presentation-only. Pixels outside the imaged swath are rendered dark "
        f"from the alpha band, not white."))

    items, failed = [], 0
    t0 = time.time()
    for i, p in enumerate(orthos, 1):
        label = _chunk_label(p)
        dst = ctx.asset(f"ortho_{_slug(label)}.jpg")
        meta_p = dst.with_suffix(".json")
        meta = _read_json(meta_p) if ctx.fresh(dst, p) else None
        if meta is None:
            meta = ortho_thumb(p, dst)
            if meta is not None:
                try:
                    meta_p.write_text(json.dumps(meta), encoding="utf-8")
                except OSError:
                    pass
        if meta is None:
            failed += 1
            continue
        items.append({"src": str(dst),
                      "cap": (f"{label} · {meta['w_m']:.0f}×{meta['h_m']:.0f} m · "
                              f"{meta['px_mm']:.0f} mm/px · "
                              f"{meta['cover'] * 100:.0f} % imaged")})
        if i % 20 == 0:
            ctx.log(f"    orthos {i}/{len(orthos)} ({time.time() - t0:.0f}s)")
    if items:
        blocks.append(grid(items, cols=3, title="Chunk orthomosaics",
                           foot=(f"{failed} orthomosaic(s) could not be read."
                                 if failed else "")))
    else:
        blocks.append(note("Chunk orthomosaics present on disk but none could "
                           "be read."))

    if gal_dir and gal_dir.is_dir():
        gal = _sorted_pngs(gal_dir, 4)
        if gal:
            blocks.append(h3("Close-up gallery (existing product)"))
            for p in gal:
                staged = ctx.stage(p)
                if staged:
                    blocks.append(image(staged, p.name))
    if contact and contact.is_file():
        staged = ctx.stage(contact)
        if staged:
            blocks.append(image(staged, "mosaics_contact_sheet.png (existing "
                                        "product)"))

    if dems:
        pick = dems[:MAX_DEM_THUMBS]
        blocks.append(h3("Chunk DEMs — hillshade"))
        blocks.append(para(
            f"{_fmt_int(len(dems))} DEM(s) on disk; {len(pick)} shown "
            f"(hillshade, 315° azimuth, 45° altitude, 2× vertical "
            f"exaggeration, nodata dark)."))
        ditems = []
        for p in pick:
            label = _chunk_label(p)
            dst = ctx.asset(f"dem_{_slug(label)}.jpg")
            meta_p = dst.with_suffix(".json")
            meta = _read_json(meta_p) if ctx.fresh(dst, p) else None
            if meta is None:
                meta = dem_thumb(p, dst)
                if meta is not None:
                    try:
                        meta_p.write_text(json.dumps(meta), encoding="utf-8")
                    except OSError:
                        pass
            if meta is None:
                continue
            ditems.append({"src": str(dst),
                           "cap": f"{label} · relief {meta['relief_m']:.1f} m"})
        if ditems:
            blocks.append(grid(ditems, cols=4, title=""))
        else:
            blocks.append(note("DEMs present but no hillshade could be built."))
    else:
        blocks.append(note("Chunk DEMs — not present (…/chunk_*/dem.tif)."))

    return blocks, f"{len(items)} ortho thumb(s), {len(dems)} DEM(s)"


# --------------------------------------------------------------------------
# Section 6 — fauna computer vision
# --------------------------------------------------------------------------

def _bucket_table(ctx: _Ctx, facts: dict) -> list[dict]:
    blocks: list[dict] = []
    density = facts.get("_density")
    buckets = dict(facts.get("buckets") or {})
    ortho_sum = _read_json(ctx.file("fauna/ortho_fauna_summary.json"))
    ortho_buckets = dict((ortho_sum or {}).get("by_bucket_kept") or {})

    per_frame: dict[str, int] = {}
    if density is not None:
        for col in density.columns:
            if col.startswith("n_") and col not in ("n_total", "n_raw"):
                try:
                    per_frame[col[2:]] = int(density[col].fillna(0).sum())
                except Exception:                                   # noqa: BLE001
                    continue

    names = sorted(set(buckets) | set(per_frame) | set(ortho_buckets))
    if not names:
        blocks.append(note("Fauna bucket counts — not present "
                           "(survey/fauna/fathomnet_detections.csv, "
                           "fauna_density.csv)."))
        return blocks
    rows = []
    for b in names:
        rows.append([b, _fmt_int(buckets.get(b, 0)) if b in buckets else "—",
                     _fmt_int(per_frame.get(b, 0)) if b in per_frame else "—",
                     _fmt_int(ortho_buckets.get(b, 0)) if b in ortho_buckets else "—"])
    rows.append(["TOTAL", _fmt_int(sum(buckets.values())) if buckets else "—",
                 _fmt_int(sum(per_frame.values())) if per_frame else "—",
                 _fmt_int(sum(ortho_buckets.values())) if ortho_buckets else "—"])
    blocks.append(table(("Bucket", "Frame detections (kept)",
                         "Per-frame census sum", "Orthomosaic detections"),
                        rows, title="Detections by bucket",
                        foot="Frame detections come from fathomnet_detections.csv "
                             "(excluded rows dropped); the census sum is the "
                             "n_<bucket> columns of fauna_density.csv; the "
                             "orthomosaic column is by_bucket_kept from "
                             "ortho_fauna_summary.json.",
                        align_right=(1, 2, 3)))

    if ortho_sum:
        by_class = (ortho_sum.get("by_class_kept") or {})
        if by_class:
            top = sorted(by_class.items(), key=lambda kv: -kv[1])[:15]
            blocks.append(table(("Class", "Detections"),
                                [[k, _fmt_int(v)] for k, v in top],
                                title="Top classes on the orthomosaics "
                                      "(ortho_fauna_summary.json)",
                                align_right=(1,)))
    return blocks


def _pick_crops(det, limit: int = MAX_CROPS) -> list[dict]:
    """Mechanical sample: highest-confidence detections, one per frame,
    spread evenly over the buckets.

    Deliberately NOT curated by verdict — this is the same rule as the Model
    Observation Report so the sample stays an honest picture of what the
    detector actually returns, warts included.
    """
    if det is None or not len(det) or "conf" not in det.columns:
        return []
    df = det
    if "excluded" in df.columns:
        ex = df["excluded"].fillna("").astype(str).str.strip()
        df = df[ex == ""]
    need = {"fn", "conf", "x1", "y1", "x2", "y2"}
    if not need <= set(df.columns) or not len(df):
        return []
    df = df.sort_values("conf", ascending=False, kind="mergesort")
    df = df.drop_duplicates(subset=["fn"], keep="first")   # one per frame
    if "bucket" not in df.columns:
        return df.head(limit).to_dict("records")
    buckets = list(df["bucket"].fillna("—").value_counts().index)
    per = max(1, limit // max(1, len(buckets)))
    picks: list[dict] = []
    for b in buckets:
        sub = df[df["bucket"].fillna("—") == b].head(per)
        picks.extend(sub.to_dict("records"))
    if len(picks) < limit:                       # top up from what is left
        taken = {p["fn"] for p in picks}
        for rec in df.to_dict("records"):
            if len(picks) >= limit:
                break
            if rec["fn"] not in taken:
                picks.append(rec)
                taken.add(rec["fn"])
    picks.sort(key=lambda r: (str(r.get("bucket", "")), -float(r["conf"])))
    return picks[:limit]


def render_crop(ctx: _Ctx, rec: dict, dst: Path) -> bool:
    """One detection crop with its box drawn, at most one frame in memory."""
    from PIL import Image, ImageDraw
    Image.MAX_IMAGE_PIXELS = None

    src = ctx.find_frame(str(rec["fn"]))
    if src is None:
        return False
    try:
        x1, y1, x2, y2 = (float(rec["x1"]), float(rec["y1"]),
                          float(rec["x2"]), float(rec["y2"]))
    except (TypeError, ValueError):
        return False
    im = None
    try:
        im = Image.open(src)
        im = im.convert("RGB")
        W, H = im.size
        bw, bh = max(1.0, x2 - x1), max(1.0, y2 - y1)
        pad = CROP_CONTEXT * max(bw, bh)
        cx0 = max(0, int(x1 - pad)); cy0 = max(0, int(y1 - pad))
        cx1 = min(W, int(x2 + pad)); cy1 = min(H, int(y2 + pad))
        if cx1 - cx0 < 8 or cy1 - cy0 < 8:
            return False
        crop = im.crop((cx0, cy0, cx1, cy1))
        im.close(); im = None
        draw = ImageDraw.Draw(crop)
        rect = (x1 - cx0, y1 - cy0, x2 - cx0, y2 - cy0)
        draw.rectangle(rect, outline=(255, 214, 0), width=max(2, int(
            0.006 * max(crop.size))))
        crop.thumbnail((CROP_PX, CROP_PX), Image.LANCZOS)
        crop.save(dst, "JPEG", quality=86, optimize=True)
        crop.close()
        return True
    except Exception:                                               # noqa: BLE001
        return False
    finally:
        if im is not None:
            try:
                im.close()
            except Exception:                                       # noqa: BLE001
                pass


def section_fauna(ctx: _Ctx, facts: dict) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("6 · Fauna computer vision", "fauna")]
    fauna_dir = ctx.dir("fauna")
    if fauna_dir is None:
        blocks.append(note("Fauna computer vision — not present "
                           "(no survey/fauna/)."))
        return blocks, "not present"

    blocks.extend(_bucket_table(ctx, facts))

    # -- density time series ---------------------------------------------
    ts = None
    for name in ("fauna_density_timeseries_v2.png", "fauna_density_timeseries.png",
                 "fauna_density_timeseries_v2_subsample.png"):
        cand = fauna_dir / name
        if cand.is_file():
            ts = cand
            break
    if ts is not None:
        staged = ctx.stage(ts)
        if staged:
            blocks.append(image(staged, f"Fauna density time series · {ts.name}"))
    else:
        blocks.append(note("Fauna density time-series figure — not present "
                           "(survey/fauna/fauna_density_timeseries*.png)."))

    for name, cap in (("fauna_vs_anomaly.png",
                       "Fauna density against anomaly windows"),
                      ("ortho_fauna_qa.png",
                       "Orthomosaic fauna detection QA sheet"),
                      ("multiview_qa.png", "Multi-view association QA sheet")):
        cand = fauna_dir / name
        if not cand.is_file():
            mv = ctx.dir("multiview")
            cand = (mv / name) if mv else cand
        if cand.is_file():
            staged = ctx.stage(cand)
            if staged:
                blocks.append(image(staged, f"{cap} · {cand.name}"))

    # -- fish shortlist ---------------------------------------------------
    shortlist = _read_csv(ctx.file("fauna/fish_frame_shortlist.csv"))
    if shortlist is not None and len(shortlist):
        sort_col = "max_conf" if "max_conf" in shortlist.columns else None
        top = (shortlist.sort_values(sort_col, ascending=False)
               if sort_col else shortlist).head(SHORTLIST_ROWS)
        cols = [c for c in ("fn", "n_fish", "max_conf", "classes", "seg",
                            "depth", "alt") if c in top.columns] or list(top.columns)[:6]
        rows = []
        for _, r in top.iterrows():
            row = []
            for c in cols:
                v = r[c]
                if isinstance(v, float):
                    row.append(f"{v:.3f}" if abs(v) < 1e4 else f"{v:.1f}")
                else:
                    row.append(str(v))
            rows.append(row)
        blocks.append(table([c.replace("_", " ") for c in cols], rows,
                            title=f"Fish frame shortlist — top {len(rows)} of "
                                  f"{_fmt_int(len(shortlist))} rows"))
    else:
        blocks.append(note("Fish shortlist — not present "
                           "(survey/fauna/fish_frame_shortlist.csv)."))

    # -- detection crops ---------------------------------------------------
    det = facts.get("_det")
    picks = _pick_crops(det, MAX_CROPS)
    if not picks:
        blocks.append(note("Detection crops — no detection table on disk "
                           "(survey/fauna/fathomnet_detections.csv)."))
        return blocks, "tables only"
    if not ctx.frame_roots():
        blocks.append(note(
            f"Detection crops — {len(picks)} detections selected, but no frame "
            f"directory was found on this machine, so the crops could not be "
            f"rendered."))
        return blocks, "tables only (frames offline)"

    blocks.append(h3("Detection sample"))
    blocks.append(para(
        f"A mechanical sample: the highest-confidence detection per frame, "
        f"spread evenly across buckets, capped at {MAX_CROPS}. Nothing here is "
        f"curated by verdict — these are the detector's own top calls, right or "
        f"wrong. Boxes as drawn by the model; {int(CROP_CONTEXT * 100)} % "
        f"context margin."))
    items, missing = [], 0
    for rec in picks:
        fn = str(rec["fn"])
        dst = ctx.asset(f"crop_{_slug(Path(fn).stem)}.jpg")
        if not dst.is_file() and not render_crop(ctx, rec, dst):
            missing += 1
            continue
        cls = str(rec.get("cls", "") or "")
        bucket = str(rec.get("bucket", "") or "—")
        conf = float(rec.get("conf", 0) or 0)
        items.append({"src": str(dst),
                      "cap": f"{bucket} · {cls} · conf {conf:.2f}"})
    if items:
        blocks.append(grid(items, cols=5, title="",
                           foot=(f"{missing} selected crop(s) skipped — frame "
                                 f"file not found." if missing else "")))
    else:
        blocks.append(note("Detection crops — frames could not be opened."))
    return blocks, (f"{len(items)} crop(s), "
                    f"{_fmt_int(facts.get('det_kept', 0))} detections")


# --------------------------------------------------------------------------
# Section 7 — other analysis figures
# --------------------------------------------------------------------------

def section_analysis(ctx: _Ctx) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("7 · Other analysis figures", "analysis")]
    found = 0
    for family in ("analysis_figs", "qc", "frame_stats", "biigle"):
        base = ctx.dir(family)
        for p in _sorted_pngs(base, MAX_FIGURES_PER_FAMILY):
            staged = ctx.stage(p)
            if staged:
                blocks.append(image(staged, f"{family}/{p.name}"))
                found += 1
    stats = _read_json(ctx.file("analysis_figs/analysis_stats.json"))
    if stats:
        rows = [[k, json.dumps(v)[:120]] for k, v in list(stats.items())[:20]]
        blocks.append(table(("Statistic", "Value"), rows,
                            title="analysis_stats.json"))
        found += 1
    if not found:
        blocks.append(note("Analysis figures — not present "
                           "(survey/analysis_figs/)."))
        return blocks, "not present"
    return blocks, f"{found} item(s)"


# --------------------------------------------------------------------------
# Section 8 — product inventory
# --------------------------------------------------------------------------

def _collapse_match(name: str) -> bool:
    from fnmatch import fnmatch
    return any(fnmatch(name, pat) for pat in COLLAPSE_DIRS)


def _walk_inventory(root: Path) -> tuple[list[tuple[str, int, float]], dict]:
    """(rel, bytes, mtime) rows, with engine-internal directories collapsed.

    A raw walk of survey/ is ~37 000 entries on a full dive (Metashape's
    project.files, extracted frames, the ortho cache); collapsing those to one
    row each keeps the table a table.
    """
    rows: list[tuple[str, int, float]] = []
    rollup: dict[str, list] = {}
    if not root.is_dir():
        return rows, rollup

    def _dir_size(d: Path) -> tuple[int, int, float]:
        n = total = 0
        newest = 0.0
        for dp, _dn, fns in os.walk(d):
            for fn in fns:
                try:
                    st = os.stat(os.path.join(dp, fn))
                except OSError:
                    continue
                n += 1
                total += st.st_size
                newest = max(newest, st.st_mtime)
        return n, total, newest

    def _rec(d: Path) -> None:
        try:
            entries = sorted(os.scandir(d), key=lambda e: e.name)
        except OSError:
            return
        for e in entries:
            rel = os.path.relpath(e.path, root)
            if e.is_dir(follow_symlinks=False):
                if e.name in SKIP_DIRS and d == root:
                    continue
                if _collapse_match(e.name):
                    n, total, newest = _dir_size(Path(e.path))
                    if n:
                        rows.append((rel + f"/  ({n} files)", total, newest))
                    continue
                _rec(Path(e.path))
            elif e.is_file(follow_symlinks=False):
                try:
                    st = e.stat()
                except OSError:
                    continue
                rows.append((rel, st.st_size, st.st_mtime))

    _rec(root)
    for rel, size, mt in rows:
        fam = rel.split(os.sep)[0] if os.sep in rel else "(root)"
        r = rollup.setdefault(fam, [0, 0, 0.0])
        r[0] += 1
        r[1] += size
        r[2] = max(r[2], mt)
    return rows, rollup


def section_inventory(ctx: _Ctx) -> tuple[list[dict], str]:
    blocks: list[dict] = [h2("8 · Product inventory", "inventory")]
    root = ctx.survey
    rows, rollup = _walk_inventory(root)
    if not rows:
        blocks.append(note("Product inventory — survey/ is empty."))
        return blocks, "not present"

    rl = [[fam, _fmt_int(v[0]), _fmt_bytes(v[1]), _fmt_time(v[2])]
          for fam, v in sorted(rollup.items(), key=lambda kv: -kv[1][1])]
    rl.append(["TOTAL", _fmt_int(sum(v[0] for v in rollup.values())),
               _fmt_bytes(sum(v[1] for v in rollup.values())),
               _fmt_time(max(v[2] for v in rollup.values()))])
    blocks.append(table(("Product family", "Entries", "Size", "Newest"), rl,
                        title="By product family",
                        align_right=(1, 2)))

    shown = rows[:MAX_INVENTORY_ROWS]
    detail = [[rel.replace(os.sep, "/"), _fmt_bytes(size), _fmt_time(mt)]
              for rel, size, mt in shown]
    foot = (f"{len(rows) - len(shown)} further entries omitted; engine-internal "
            f"directories ({', '.join(COLLAPSE_DIRS)}) are collapsed to one row "
            f"each." if len(rows) > len(shown) else
            f"Engine-internal directories ({', '.join(COLLAPSE_DIRS)}) are "
            f"collapsed to one row each.")
    blocks.append(table(("Path (relative to survey/)", "Size", "Modified"),
                        detail, title=f"All products under survey/ "
                                      f"({_fmt_int(len(rows))} entries)",
                        foot=foot, align_right=(1,), break_cols=(0,)))
    return blocks, f"{_fmt_int(len(rows))} entries"


# --------------------------------------------------------------------------
# HTML rendering
# --------------------------------------------------------------------------

CSS = """
:root { --ink:#1b2733; --muted:#68798a; --line:#d7dfe6; --accent:#0e7c86;
        --warn:#9a6a00; --bg:#ffffff; --soft:#f4f7f9; }
* { box-sizing:border-box; }
html,body { background:var(--bg); color:var(--ink); margin:0;
            font-family:"Segoe UI","Helvetica Neue",Arial,sans-serif;
            font-size:10.5pt; line-height:1.45; }
.page { padding:0; }
h1 { font-size:26pt; margin:0 0 4px; letter-spacing:-0.4px; }
h2 { font-size:15pt; margin:20px 0 8px; padding-bottom:5px;
     border-bottom:2px solid var(--accent); color:#10323b;
     break-after:avoid; page-break-after:avoid; }
h3 { font-size:11.5pt; margin:14px 0 6px; color:#22404a;
     break-after:avoid; page-break-after:avoid; }
p { margin:6px 0 10px; }
code { background:var(--soft); padding:1px 4px; border-radius:3px;
       font-size:9.5pt; }
.sub { color:var(--muted); font-size:10pt; }
.note { color:var(--warn); background:#fff8e8; border-left:3px solid #e0b040;
        padding:7px 10px; margin:8px 0; font-size:10pt;
        break-inside:avoid; page-break-inside:avoid; }
figure { margin:10px 0 14px; break-inside:avoid; page-break-inside:avoid; }
/* max-height as well as max-width: a merged mosaic of a 7 km transect is a
   276x2600 ribbon, and width-only scaling would make it 1.8 m tall. */
figure img { max-width:100%; max-height:225mm; width:auto; height:auto;
             display:block; margin:0 auto;
             border:1px solid var(--line); border-radius:3px;
             background:#101820; }
figcaption { font-size:8.8pt; color:var(--muted); margin-top:4px; }
.tw { overflow:visible; margin:8px 0 14px; }
table { border-collapse:collapse; width:100%; font-size:9pt;
        break-inside:auto; }
caption { caption-side:top; text-align:left; font-weight:600; font-size:10pt;
          color:#22404a; padding-bottom:5px; }
th { background:var(--soft); text-align:left; font-weight:600;
     border-bottom:1.5px solid var(--line); padding:4px 7px;
     color:#31465a; }
/* overflow-wrap (not word-break) so a narrow column cannot chop "9.90534"
   in half: it keeps each cell's min-content width at its longest word. */
td { border-bottom:1px solid #eef2f5; padding:3px 7px;
     vertical-align:top; overflow-wrap:break-word; }
tr { break-inside:avoid; page-break-inside:avoid; }
td.r, th.r { text-align:right; }
td.b, th.b { word-break:break-all; }     /* long paths may break anywhere */
tr:last-child td { border-bottom:1px solid var(--line); }
tfoot td { color:var(--muted); font-size:8.5pt; border:0; padding-top:5px; }
/* inline-block, not CSS grid: grid tracks fragment badly across printed pages
   (Chrome leaves half a page empty after every row), inline-block flows. */
.grid { margin:8px 0 14px; font-size:0; }
.cell { display:inline-block; vertical-align:top; font-size:10.5pt;
        width:var(--w,33.33%); padding:0 5px 9px 0;
        break-inside:avoid; page-break-inside:avoid; }
/* One fixed box per cell so a grid row's thumbnails share a baseline whatever
   their aspect ratio; letterboxing is dark, matching the outside-swath fill. */
.cell img { width:100%; aspect-ratio:4/3; object-fit:contain; display:block;
            border:1px solid var(--line); border-radius:3px;
            background:#101820; }
.cell .c { font-size:7.6pt; color:var(--muted); margin-top:3px;
           line-height:1.25; }
.cover { border:1px solid var(--line); border-radius:5px; padding:16px 18px;
         background:var(--soft); margin:6px 0 16px; }
.cover .kv { display:grid; grid-template-columns:150px 1fr; gap:3px 12px;
             font-size:10pt; }
.cover .kv b { color:var(--muted); font-weight:600; }
.toc { font-size:9.5pt; margin:10px 0 0; }
.toc div { padding:2px 0; border-bottom:1px dotted #e3e9ee; }
.toc .st { color:var(--muted); float:right; }
.tiers span { display:inline-block; padding:2px 8px; border-radius:10px;
              color:#fff; font-size:9pt; margin-right:5px; }
@page { size:A4; margin:11mm; }
@media print {
  figure, table, .cell, .note { break-inside:avoid; page-break-inside:avoid; }
  h2 { break-after:avoid; page-break-after:avoid; }
  .tw { overflow:visible; }
  img { max-width:100%; }
}
"""


def _esc(text) -> str:
    return html.escape(str(text), quote=False)


def _rel_src(path: str, base: Path) -> str:
    try:
        return os.path.relpath(str(path), str(base)).replace(os.sep, "/")
    except ValueError:
        return Path(str(path)).as_uri()


def render_html(blocks: list[dict], facts: dict, toc: list[tuple[str, str]],
                base: Path) -> str:
    out: list[str] = []
    for b in blocks:
        k = b["k"]
        if k == "cover":
            f = b["facts"]
            tiers = "".join(
                f'<span style="background:{TIER_COLOUR.get(t, "#6c7a89")}">'
                f'{_esc(t)} {_fmt_int(n)}</span>'
                for t, n in sorted(f["tiers"].items(),
                                   key=lambda kv: TIER_ORDER.index(kv[0])
                                   if kv[0] in TIER_ORDER else 99))
            toc_html = "".join(
                f'<div><span class="st">{_esc(status)}</span>{_esc(title)}</div>'
                for title, status in toc)
            out.append(
                f'<h1>{_esc(f["dive"])} — product catalog</h1>'
                f'<div class="sub">A printed preview of every product this '
                f'workspace currently holds.</div>'
                f'<div class="cover"><div class="kv">'
                f'<b>Dive</b><span>{_esc(f["dive"])}</span>'
                f'<b>Generated</b><span>{_esc(f["generated"])}</span>'
                f'<b>Scope</b><span>{_esc(f["scope"])}</span>'
                f'<b>Workspace</b><span><code>{_esc(f["ws"])}</code></span>'
                f'</div>'
                f'{f"<div class=tiers style=margin-top:12px>{tiers}</div>" if tiers else ""}'
                f'<div class="toc">{toc_html}</div></div>')
        elif k == "h2":
            out.append(f'<h2 id="{_esc(b["id"])}">{_esc(b["t"])}</h2>')
        elif k == "h3":
            out.append(f'<h3>{_esc(b["t"])}</h3>')
        elif k == "p":
            out.append(f'<p>{b["t"]}</p>')          # trusted, may carry <code>
        elif k == "note":
            out.append(f'<div class="note">{_esc(b["t"])}</div>')
        elif k == "img":
            cap = (f'<figcaption>{_esc(b["cap"])}</figcaption>'
                   if b["cap"] else "")
            out.append(f'<figure><img src="{_esc(_rel_src(b["src"], base))}">'
                       f'{cap}</figure>')
        elif k == "table":
            right = b.get("right") or set()
            brk = b.get("brk") or set()

            def _cls(i: int) -> str:
                c = ("r" if i in right else "") + (" b" if i in brk else "")
                return f' class="{c.strip()}"' if c.strip() else ""

            cap = f'<caption>{_esc(b["t"])}</caption>' if b["t"] else ""
            head = "".join(f"<th{_cls(i)}>{_esc(c)}</th>"
                           for i, c in enumerate(b["cols"]))
            body = []
            for r in b["rows"]:
                tds = "".join(f"<td{_cls(i)}>{_esc(v)}</td>"
                              for i, v in enumerate(r))
                body.append(f"<tr>{tds}</tr>")
            foot = (f'<tfoot><tr><td colspan="{len(b["cols"])}">'
                    f'{_esc(b["foot"])}</td></tr></tfoot>' if b.get("foot") else "")
            out.append(f'<div class="tw"><table>{cap}<thead><tr>{head}</tr>'
                       f'</thead><tbody>{"".join(body)}</tbody>{foot}</table></div>')
        elif k == "grid":
            title = f'<h3>{_esc(b["t"])}</h3>' if b["t"] else ""
            cells = "".join(
                f'<div class="cell"><img src="{_esc(_rel_src(it["src"], base))}">'
                f'<div class="c">{_esc(it["cap"])}</div></div>'
                for it in b["items"])
            foot = (f'<div class="note">{_esc(b["foot"])}</div>'
                    if b.get("foot") else "")
            width = 100.0 / max(1, int(b["cols"]))
            out.append(f'{title}<div class="grid" style="--w:{width:.4f}%">'
                       f'{cells}</div>{foot}')
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<title>{_esc(facts["dive"])} product catalog</title>'
            f"<style>{CSS}</style></head><body><div class=page>"
            f'{"".join(out)}</div></body></html>')


# --------------------------------------------------------------------------
# PDF printing — Chrome first, matplotlib fallback
# --------------------------------------------------------------------------

def _find_chrome() -> Optional[str]:
    for cand in CHROME_CANDIDATES:
        if "/" in cand:
            if Path(cand).is_file():
                return cand
        else:
            found = shutil.which(cand)
            if found:
                return found
    return None


def _winpath(p: Path) -> str:
    """Windows path for a WSL path (unchanged when not under WSL)."""
    try:
        res = subprocess.run(["wslpath", "-w", str(p)], capture_output=True,
                             text=True, timeout=20)
        if res.returncode == 0 and res.stdout.strip():
            return res.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return str(p)


def chrome_pdf(html_path: Path, pdf_path: Path, log) -> bool:
    exe = _find_chrome()
    if not exe:
        log("  chrome: not found — falling back to matplotlib")
        return False
    is_win_exe = exe.endswith(".exe")
    src = _winpath(html_path) if is_win_exe else html_path.as_uri()
    dst = _winpath(pdf_path) if is_win_exe else str(pdf_path)
    profile = html_path.parent / ".chrome_profile"
    try:
        profile.mkdir(parents=True, exist_ok=True)
    except OSError:
        pass
    cmd = [exe, "--headless", "--disable-gpu", "--no-sandbox",
           "--no-pdf-header-footer", "--run-all-compositor-stages-before-draw",
           "--virtual-time-budget=60000",
           f"--user-data-dir={_winpath(profile) if is_win_exe else profile}",
           f"--print-to-pdf={dst}", src]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    except (OSError, subprocess.SubprocessError) as exc:
        log(f"  chrome: {type(exc).__name__}: {exc}")
        return False
    if pdf_path.is_file() and pdf_path.stat().st_size > 2048:
        return True
    tail = (res.stderr or res.stdout or "").strip().splitlines()[-3:]
    log(f"  chrome: no PDF produced (rc={res.returncode}) {' | '.join(tail)}")
    return False


def _mpl_pdf(blocks: list[dict], facts: dict, toc, pdf_path: Path, log) -> bool:
    """Degraded but complete fallback: the same block list on PdfPages."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages
    import textwrap

    PW, PH = 8.27, 11.69          # A4 inches
    M = 0.43                      # 11 mm margin
    plt.rcParams["font.family"] = "DejaVu Sans"

    state = {"fig": None, "y": 0.0}

    def _new_page(pp):
        if state["fig"] is not None:
            pp.savefig(state["fig"])
            plt.close(state["fig"])
        fig = plt.figure(figsize=(PW, PH), dpi=110)
        state["fig"] = fig
        state["y"] = 1.0 - M / PH
        return fig

    def _need(pp, h):
        if state["fig"] is None or state["y"] - h < M / PH:
            _new_page(pp)

    def _text(pp, s, size=9, weight="normal", colour="#1b2733", wrap=105,
              gap=0.006):
        lines = []
        for para_ in str(s).split("\n"):
            lines += textwrap.wrap(para_, wrap) or [""]
        lh = (size + 3.4) / 72.0 / PH
        for line in lines:
            _need(pp, lh)
            state["fig"].text(M / PW, state["y"], line, size=size,
                              weight=weight, color=colour, va="top", ha="left")
            state["y"] -= lh
        state["y"] -= gap

    def _image(pp, path, cap, frac=1.0):
        try:
            img = plt.imread(str(path))
        except Exception:                                           # noqa: BLE001
            return
        ih, iw = img.shape[0], img.shape[1]
        avail_w = (PW - 2 * M) * frac
        w_in = avail_w
        h_in = w_in * ih / max(1, iw)
        max_h = PH - 2 * M - 0.4
        if h_in > max_h:
            h_in = max_h
            w_in = h_in * iw / max(1, ih)
        hf = h_in / PH
        _need(pp, hf + 0.03)
        ax = state["fig"].add_axes([M / PW, state["y"] - hf,
                                    w_in / PW, hf])
        ax.imshow(img)
        ax.axis("off")
        state["y"] -= hf + 0.004
        del img
        if cap:
            _text(pp, cap, size=6.6, colour="#68798a", wrap=150, gap=0.008)

    def _grid(pp, blk):
        """Thumbnail rows, `cols` per row — one-per-page would be 15 pages."""
        from PIL import Image
        cols = max(1, int(blk["cols"]))
        cw_in = (PW - 2 * M) / cols
        items = list(blk["items"])
        for r0 in range(0, len(items), cols):
            row = items[r0:r0 + cols]
            sizes = []
            for it in row:
                try:
                    with Image.open(it["src"]) as im:
                        sizes.append(im.size)
                except Exception:                                   # noqa: BLE001
                    sizes.append(None)
            heights = [(cw_in * 0.94) * s[1] / max(1, s[0])
                       for s in sizes if s]
            if not heights:
                continue
            rh_in = min(max(heights), (PH - 2 * M) * 0.42)
            cap_h = 3 * (6.2 + 2.6) / 72.0
            _need(pp, (rh_in + cap_h) / PH + 0.01)
            top = state["y"]
            for i, (it, size) in enumerate(zip(row, sizes)):
                if not size:
                    continue
                w_in = cw_in * 0.94
                h_in = min(rh_in, w_in * size[1] / max(1, size[0]))
                w_in = h_in * size[0] / max(1, size[1])
                try:
                    img = plt.imread(it["src"])
                except Exception:                                   # noqa: BLE001
                    continue
                ax = state["fig"].add_axes(
                    [(M + i * cw_in) / PW, top - h_in / PH,
                     w_in / PW, h_in / PH])
                ax.imshow(img)
                ax.axis("off")
                del img
                cap = textwrap.wrap(it["cap"], max(14, int(cw_in * 15)))[:3]
                cy = top - (rh_in + 0.03) / PH
                for line in cap:
                    state["fig"].text((M + i * cw_in) / PW, cy, line, size=6.2,
                                      color="#68798a", va="top", ha="left")
                    cy -= (6.2 + 2.6) / 72.0 / PH
            state["y"] = top - (rh_in + cap_h) / PH - 0.008

    def _table(pp, blk):
        if blk["t"]:
            _text(pp, blk["t"], size=10, weight="bold", colour="#22404a")
        cols = [str(c) for c in blk["cols"]]
        ncol = len(cols)
        widths = [max(len(cols[i]),
                      max((len(str(r[i])) for r in blk["rows"][:80]), default=4))
                  for i in range(ncol)]
        tot = sum(widths) or 1
        fracs = [max(0.06, w / tot) for w in widths]
        fs = 6.4 if ncol > 5 or tot > 110 else 7.4
        lh = (fs + 3.0) / 72.0 / PH

        def _row(vals, weight="normal", colour="#1b2733"):
            _need(pp, lh)
            x = M / PW
            for i, v in enumerate(vals):
                cw = fracs[i] * (PW - 2 * M) / PW
                s = str(v)
                cap = max(4, int(fracs[i] * (PW - 2 * M) * 72 / (fs * 0.56)))
                if len(s) > cap:
                    s = s[:cap - 1] + "…"
                state["fig"].text(x, state["y"], s, size=fs, weight=weight,
                                  color=colour, va="top", ha="left")
                x += cw
            state["y"] -= lh

        _row(cols, weight="bold", colour="#31465a")
        for r in blk["rows"]:
            _row(r)
        if blk.get("foot"):
            _text(pp, blk["foot"], size=6.2, colour="#68798a", wrap=170)
        state["y"] -= 0.008

    try:
        with PdfPages(str(pdf_path)) as pp:
            _new_page(pp)
            for b in blocks:
                k = b["k"]
                if k == "cover":
                    f = b["facts"]
                    _text(pp, f"{f['dive']} — product catalog", size=20,
                          weight="bold")
                    _text(pp, "A printed preview of every product this "
                              "workspace currently holds.", size=9,
                          colour="#68798a")
                    _text(pp, f"Generated {f['generated']}   ·   scope "
                              f"{f['scope']}", size=9, colour="#68798a")
                    _text(pp, f"Workspace: {f['ws']}", size=8,
                          colour="#68798a", wrap=130)
                    _text(pp, "(Rendered without Chrome — degraded layout.)",
                          size=8, colour="#9a6a00")
                    _text(pp, "Contents", size=11, weight="bold")
                    for title, status in toc:
                        _text(pp, f"  {title} — {status}", size=8.4,
                              colour="#31465a", gap=0.0)
                elif k == "h2":
                    _new_page(pp)
                    _text(pp, b["t"], size=14, weight="bold", colour="#10323b")
                elif k == "h3":
                    _text(pp, b["t"], size=10.5, weight="bold", colour="#22404a")
                elif k == "p":
                    _text(pp, re.sub(r"<[^>]+>", "", b["t"]), size=8.6)
                elif k == "note":
                    _text(pp, "! " + b["t"], size=8.6, colour="#9a6a00")
                elif k == "img":
                    _image(pp, b["src"], b["cap"])
                elif k == "table":
                    _table(pp, b)
                elif k == "grid":
                    if b["t"]:
                        _text(pp, b["t"], size=10.5, weight="bold",
                              colour="#22404a")
                    _grid(pp, b)
                    if b.get("foot"):
                        _text(pp, b["foot"], size=7, colour="#9a6a00")
            if state["fig"] is not None:
                pp.savefig(state["fig"])
                plt.close(state["fig"])
    except Exception as exc:                                        # noqa: BLE001
        log(f"  matplotlib fallback failed: {type(exc).__name__}: {exc}")
        return False
    return pdf_path.is_file() and pdf_path.stat().st_size > 1024


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def build_catalog(workspace_dir, job_id: str = WHOLE, log=print) -> str:
    """Build the preview catalog PDF for a workspace; return its path.

    ``job_id`` is the simple-UI scope sentinel: ``"__whole__"`` (the whole
    trackline / every legacy product) or a job id, whose products live under
    ``survey/jobs/<job_id>/``.  A job scope falls back to the whole-survey
    product for families the job itself has not produced.
    """
    log = _logger(log)
    ws = Path(str(workspace_dir)).resolve()
    if not ws.is_dir():
        raise FileNotFoundError(f"workspace not found: {ws}")
    t_start = time.time()
    ctx = _Ctx(ws, job_id, log)
    log(f"catalog: {ctx.dive}  ({ctx.scope})")

    log("  reading product tables …")
    facts = collect_facts(ctx)

    sections: list[tuple[str, list[dict], str]] = []

    def _run(title, fn, *args):
        try:
            blocks, status = fn(*args)
        except Exception as exc:                                    # noqa: BLE001
            log(f"  !! {title}: {type(exc).__name__}: {exc}")
            blocks = [h2(title), note(f"{title}: failed to render "
                                      f"({type(exc).__name__}: {exc}).")]
            status = "failed"
        sections.append((title, blocks, status))
        log(f"  {title}: {status}")

    log("  tracklines …")
    _run("2 · Tracklines", section_tracklines, ctx)
    log("  rasters …")
    _run("3 · Sensor and depth rasters", section_rasters, ctx)
    log("  anomaly …")
    _run("4 · Anomaly detection", section_anomaly, ctx, facts)
    log("  photogrammetry …")
    _run("5 · Photogrammetry", section_photogrammetry, ctx)
    log("  fauna …")
    _run("6 · Fauna computer vision", section_fauna, ctx, facts)
    log("  analysis figures …")
    _run("7 · Other analysis figures", section_analysis, ctx)
    log("  inventory …")
    _run("8 · Product inventory", section_inventory, ctx)

    toc = [("1 · Cover and headline counts", "")] + \
          [(title, status) for title, _b, status in sections]
    blocks = section_cover(ctx, facts)
    for _t, bl, _s in sections:
        blocks.extend(bl)

    stamp = datetime.now().strftime("%Y%m%d_%H%M")
    html_path = ctx.out_dir / f"catalog_{ctx.dive}_{stamp}.html"
    pdf_path = ctx.out_dir / f"catalog_{ctx.dive}_{stamp}.pdf"
    html_path.write_text(render_html(blocks, facts, toc, ctx.out_dir),
                         encoding="utf-8")
    log(f"  html -> {html_path}")

    if not chrome_pdf(html_path, pdf_path, log):
        log("  printing with the matplotlib fallback …")
        if not _mpl_pdf(blocks, facts, toc, pdf_path, log):
            raise RuntimeError("could not produce a PDF (Chrome and matplotlib "
                               "both failed)")
    size = pdf_path.stat().st_size if pdf_path.is_file() else 0
    log(f"catalog: {pdf_path}  ({_fmt_bytes(size)}, "
        f"{time.time() - t_start:.0f}s)")
    # drop the hidden Chrome profile so it never shows up as a "product"
    shutil.rmtree(ctx.out_dir / ".chrome_profile", ignore_errors=True)
    return str(pdf_path)


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build the product preview catalog PDF for a workspace.")
    ap.add_argument("workspace", help="path to a .eprproj workspace bundle")
    ap.add_argument("--job", default=WHOLE,
                    help='job scope ("__whole__" by default)')
    args = ap.parse_args(list(argv) if argv is not None else None)
    path = build_catalog(args.workspace, args.job)
    print(path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
