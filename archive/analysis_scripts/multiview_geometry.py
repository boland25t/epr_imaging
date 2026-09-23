#!/usr/bin/env python3
"""Frame <-> UTM geometry by registering ROV frames onto the chunk orthomosaics.

The per-frame `cameras.json` Metashape writes is positions-only and in LOCAL chunk
coordinates, so it cannot project a pixel to the seafloor.  The chunk orthomosaics
can: they are native ~0.5-1.1 mm/px GeoTIFFs in EPSG:32613 with a real transform,
and each was built from a known 350-frame slice of the dive -- so for any frame
that contributed to a mosaic a planar registration exists.  We recover it:

  crop that frame's OWN chunk mosaic around its nav fix, render both the crop and
  the frame at the same ground resolution (`GSD_M`), equalise them, SIFT + ratio
  test, fit a 4-DOF SIMILARITY with RANSAC, then compose

      full-res frame px -> working px -> crop px -> ortho px -> UTM metres

Three things make or break this, all learned the hard way on J1754 seg15:

  * SCALE.  Matching only works when frame and mosaic are rendered at the same
    ground resolution, so the frame's footprint has to be known before the match.
    Metashape sets a chunk's ortho GSD to that chunk's mean image GSD, so
    `K = FRAME_W * ortho_gsd / mean_alt` is the frame width per metre of altitude
    for that chunk -- a free, per-chunk camera calibration (K ~ 0.64-1.02 here).
    Guessing K globally wrong by 1.5x is enough to stop SIFT matching entirely.
  * THE ALPHA BAND.  Outside the imaged swath these mosaics carry red=222,
    green=blue=0 rather than zeros, so an RGB->grey crop is mostly a smooth red
    ramp and a nav-centred window can be ~85% junk.  Band 4 is the real mask:
    everything here is masked by it, and SIFT is given that mask so keypoints
    only come off real seafloor.
  * MODEL ORDER.  A full 8-DOF homography fits nonsense to a few hundred
    ambiguous matches (aspect ratios of 0.2 and 9 came back "validated" by inlier
    count).  A similarity preserves the frame's 16:9 shape by construction, which
    both regularises the fit and leaves the aspect ratio free as a check on any
    upgrade to a homography.

Acceptance is photometric, not just inlier count: the frame is warped onto the
mosaic and scored by high-pass ZNCC over the overlap (high-pass because the raw
correlation is dominated by the frame's own lighting vignette, which correlates
with nothing).  Registrations are cached to <ws>/survey/multiview/frame_H.json
(failures too, with a reason) so reruns are cheap and the success rate is
auditable.

Qt-free.  register_frames(workspace_dir, frames, frames_dir) -> {fn: Registration}
"""
from __future__ import annotations
import glob
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import from_bounds

FRAME_W = 5312            # full-resolution frame size these detections are in
FRAME_H = 2988
ASPECT = FRAME_W / FRAME_H
K_FALLBACK = 0.82         # frame width per metre of altitude, if the chunk is unknown

GSD_M = 0.005             # working ground resolution for matching (m/px)
WIN_MUL = 2.4             # ortho window / expected footprint (nav vs camera offset)
CLAHE_CLIP = 3.0
MIN_INLIERS = 25
MIN_ZNCC = 0.12           # high-pass ZNCC over the overlap; junk fits score ~0
STRONG_INLIERS = 80       # ... or this much geometric support at MIN_ZNCC_STRONG
MIN_ZNCC_STRONG = 0.08
MIN_OVERLAP = 0.25        # of the warped frame, against valid mosaic

# (gsd_m, win_mul, clahe_clip), tried in order until one verifies.  Coarse first:
# it is both the cheapest and the one that rescues the hazy, low-contrast frames
# that carry no fine texture to match but still have usable large-scale structure.
LADDER = ((0.012, 3.0, 6.0), (0.005, 2.4, 3.0), (0.008, 3.0, 6.0))
RATIO = 0.85              # SIFT Lowe ratio
RANSAC_PX = 10.0          # RANSAC threshold, working px (5 cm): terrain relief
NFEATURES = 15000         # ... makes an exact planar fit impossible anyway
MIN_VALID_FRAC = 0.04     # give up if the window is almost entirely unimaged
SCALE_LO, SCALE_HI = 0.6, 1.7      # keypoint-size ratio a true match may have
ROT_BIN_DEG = 20.0                 # rotation-vote bin width
ROT_KEEP_DEG = 30.0                # ... and how far from the winning bin to keep

_tls = threading.local()


# -- chunk index ---------------------------------------------------------------------

@dataclass
class Chunk:
    """One chunk mosaic plus the camera calibration its GSD implies."""
    name: str
    path: str
    gsd: float
    transform: object
    K: float                  # frame width (m) per metre of altitude
    n_cameras: int


class ChunkIndex:
    """The seg's chunk mosaics, and which frames each was built from.

    cameras.json is useless for projection (local coords, positions only) but it
    is exactly the frame -> chunk membership list we need: a frame is registered
    against the mosaic it actually contributed to.
    """

    def __init__(self, workspace_dir, seg="seg15", manifest=None):
        base = f"{workspace_dir}/survey/photogrammetry/{seg}"
        man = load_manifest(workspace_dir, seg) if manifest is None else manifest
        self.chunks: dict[str, Chunk] = {}
        self.of_frame: dict[str, str] = {}
        for cj in sorted(glob.glob(base + "/chunk_*/cameras.json")):
            name = Path(cj).parent.name
            ortho = Path(cj).parent / "orthomosaic.tif"
            if not ortho.is_file():
                continue
            try:
                cams = json.loads(Path(cj).read_text())
            except (OSError, ValueError):
                continue
            frames = [c + ".jpg" for c in cams]
            with rasterio.open(ortho) as ds:
                gsd, tf = ds.res[0], ds.transform
            alt = man.alt.reindex(frames).dropna() if len(man) else pd.Series(dtype=float)
            K = FRAME_W * gsd / alt.mean() if len(alt) else K_FALLBACK
            self.chunks[name] = Chunk(name, str(ortho), gsd, tf, float(K), len(frames))
            for f in frames:
                self.of_frame.setdefault(f, name)
        if not self.chunks:
            raise FileNotFoundError("no chunk orthomosaic+cameras.json under %s" % base)

    def for_frame(self, fn: str):
        return self.chunks.get(self.of_frame.get(fn, ""))

    def footprint_m(self, fn: str, alt: float) -> float:
        c = self.for_frame(fn)
        return (c.K if c else K_FALLBACK) * float(alt)


# -- nav manifest --------------------------------------------------------------------

def load_manifest(workspace_dir, seg="seg15") -> pd.DataFrame:
    """Per-frame nav from <ws>/survey/photogrammetry/<seg>/segment_*/interp.csv.

    Indexed by frame_filename; keeps easting/northing/alt plus unix_time and
    attitude when present.  Frames in several segments keep their first fix.
    """
    pat = f"{workspace_dir}/survey/photogrammetry/{seg}/segment_*/interp.csv"
    paths = sorted(glob.glob(pat))
    if not paths:
        raise FileNotFoundError("no interp.csv under %s" % pat)
    keep = ["frame_filename", "easting", "northing", "alt", "unix_time",
            "heading", "pitch", "roll"]
    parts = [pd.read_csv(p) for p in paths]
    parts = [df[[c for c in keep if c in df.columns]] for df in parts]
    man = pd.concat(parts, ignore_index=True).dropna(subset=["easting", "northing", "alt"])
    return man.drop_duplicates(subset="frame_filename", keep="first").set_index("frame_filename")


def frames_near(manifest: pd.DataFrame, e: float, n: float, radius_m=6.0) -> list[str]:
    """Frames whose nav fix is within radius_m of (e, n), nearest first.

    Such a frame only *may* image (e, n) -- its real footprint is decided by its
    homography; this is the cheap candidate filter before that test.
    """
    d = np.hypot(manifest.easting.to_numpy() - e, manifest.northing.to_numpy() - n)
    m = d <= radius_m
    return list(np.asarray(manifest.index)[m][np.argsort(d[m])])


# -- registration --------------------------------------------------------------------

@dataclass
class Registration:
    """A frame's planar mapping to the seafloor."""
    fn: str
    H: np.ndarray = None          # 3x3: full-res frame px (x, y) -> UTM (E, N)
    n_inliers: int = 0
    n_matches: int = 0
    zncc: float = 0.0
    overlap: float = 0.0
    chunk: str = ""
    footprint_m: float = 0.0      # projected frame width, metres
    fail: str = ""
    _inv: np.ndarray = field(default=None, repr=False)

    @property
    def ok(self) -> bool:
        return self.H is not None

    @property
    def H_inv(self) -> np.ndarray:
        if self._inv is None:
            self._inv = np.linalg.inv(self.H)
        return self._inv

    @property
    def gsd_m(self) -> float:
        """Ground sample distance of the full-res frame, metres per pixel."""
        return self.footprint_m / FRAME_W if self.footprint_m else 0.0

    def to_json(self) -> dict:
        d = {"n_inliers": int(self.n_inliers), "n_matches": int(self.n_matches),
             "zncc": round(float(self.zncc), 4), "overlap": round(float(self.overlap), 4),
             "chunk": self.chunk, "footprint_m": round(float(self.footprint_m), 4)}
        if self.H is not None:
            d["H"] = [float(x) for x in self.H.reshape(9)]
        if self.fail:
            d["fail"] = self.fail
        return d

    @classmethod
    def from_json(cls, fn: str, d: dict) -> "Registration":
        H = np.asarray(d["H"], dtype=float).reshape(3, 3) if "H" in d else None
        return cls(fn=fn, H=H, n_inliers=int(d.get("n_inliers", 0)),
                   n_matches=int(d.get("n_matches", 0)),
                   zncc=float(d.get("zncc", 0.0)), overlap=float(d.get("overlap", 0.0)),
                   chunk=d.get("chunk", ""), footprint_m=float(d.get("footprint_m", 0.0)),
                   fail=d.get("fail", ""))


def _kit(clip=CLAHE_CLIP):
    """Per-thread SIFT/matcher/CLAHE (none of them are thread-safe to share)."""
    if getattr(_tls, "sift", None) is None:
        _tls.sift = cv2.SIFT_create(nfeatures=NFEATURES, contrastThreshold=0.008,
                                    edgeThreshold=14)
        _tls.bf = cv2.BFMatcher()
        _tls.clahes = {}
    cl = _tls.clahes.get(clip)
    if cl is None:
        cl = _tls.clahes[clip] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8))
    return _tls.sift, _tls.bf, cl


def _prep(gray, clahe, valid=None):
    """Equalise for matching; the mosaic is colour-balanced, raw frames are not."""
    g = clahe.apply(cv2.GaussianBlur(gray, (0, 0), 0.8))
    if valid is not None:
        g = np.where(valid, g, 0).astype(np.uint8)
    return g


def _highpass(g):
    """Drop the low frequencies: a frame's ZNCC against the mosaic is otherwise
    dominated by its own lighting vignette, which correlates with nothing."""
    f = g.astype(np.float32)
    return f - cv2.GaussianBlur(f, (0, 0), 6.0)


def _zncc(a, b, mask) -> float:
    if mask.sum() < 5000:
        return 0.0
    x = a[mask].astype(np.float64)
    y = b[mask].astype(np.float64)
    x -= x.mean(); y -= y.mean()
    d = np.sqrt((x * x).sum() * (y * y).sum())
    return float((x * y).sum() / d) if d > 0 else 0.0


def read_ortho_window(ortho_path, transform, e, n, win_m, gsd_m):
    """A square `win_m` window of a mosaic at `gsd_m`, as (grey, valid, window).

    Band 4 is the alpha mask and it matters: outside the imaged swath these
    mosaics carry red=222/green=blue=0, so an unmasked grey crop is mostly a
    smooth red ramp.  Averaged (not nearest) decimation keeps the crop's
    scale-space comparable to the area-averaged frame.
    """
    side = max(64, int(round(win_m / gsd_m)))
    with rasterio.open(ortho_path) as ds:
        w = from_bounds(e - win_m / 2, n - win_m / 2, e + win_m / 2, n + win_m / 2,
                        transform)
        arr = ds.read([1, 2, 3, 4], window=w, out_shape=(4, side, side),
                      boundless=True, fill_value=0, resampling=Resampling.average)
    valid = arr[3] > 128
    grey = cv2.cvtColor(np.ascontiguousarray(np.transpose(arr[:3], (1, 2, 0))),
                        cv2.COLOR_RGB2GRAY)
    return np.where(valid, grey, 0).astype(np.uint8), valid, w


def vote_matches(matches, k1, k2, scale_lo=SCALE_LO, scale_hi=SCALE_HI,
                 bin_deg=ROT_BIN_DEG, keep_deg=ROT_KEEP_DEG):
    """Keep only matches agreeing on one rotation, at a scale we already expect.

    Both images are rendered at the same ground resolution, so a true match has
    a keypoint-size ratio near 1; and all true matches share one rotation.  A
    circular Hough vote on the SIFT keypoints' own angle difference finds that
    rotation without needing the vehicle heading or its sign convention, and
    discarding everything outside it is what lifts the inlier ratio far enough
    for RANSAC to find the real transform instead of a plausible-looking fit.
    """
    cand = []
    for m in matches:
        a, b = k1[m.queryIdx], k2[m.trainIdx]
        if a.size <= 0:
            continue
        r = b.size / a.size
        if scale_lo <= r <= scale_hi:
            cand.append((m, (b.angle - a.angle) % 360.0))
    if len(cand) < 10:
        return [m for m, _ in cand]
    ang = np.array([t for _, t in cand])
    nb = max(1, int(round(360.0 / bin_deg)))
    # vote in a circular histogram, smoothed over neighbouring bins
    hist = np.bincount((ang / 360.0 * nb).astype(int) % nb, minlength=nb).astype(float)
    hist = hist + np.roll(hist, 1) + np.roll(hist, -1)
    peak = (int(np.argmax(hist)) + 0.5) * 360.0 / nb
    d = np.abs((ang - peak + 180.0) % 360.0 - 180.0)
    return [cand[i][0] for i in np.flatnonzero(d <= keep_deg)]


def _shape_ok(H, expect_w) -> tuple[bool, float, str]:
    """Reject degenerate fits by the shape of the projected frame corners."""
    c = np.float32([[[0, 0]], [[FRAME_W, 0]], [[FRAME_W, FRAME_H]], [[0, FRAME_H]]])
    p = cv2.perspectiveTransform(c, H).reshape(-1, 2)
    if not np.isfinite(p).all():
        return False, 0.0, "non-finite corners"
    v = np.roll(p, -1, axis=0) - p
    nx = np.roll(v, -1, axis=0)
    cr = v[:, 0] * nx[:, 1] - v[:, 1] * nx[:, 0]
    if not (np.all(cr > 0) or np.all(cr < 0)):
        return False, 0.0, "non-convex footprint"
    w = 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3]))
    h = 0.5 * (np.linalg.norm(p[3] - p[0]) + np.linalg.norm(p[2] - p[1]))
    if h <= 0 or not 0.45 <= w / max(expect_w, 1e-6) <= 2.2:
        return False, w, "footprint %.2f m vs expected %.2f m" % (w, expect_w)
    if not 0.70 <= (w / h) / ASPECT <= 1.40:
        return False, w, "aspect %.2f (frame is %.2f)" % (w / h, ASPECT)
    return True, w, ""


def accepted(n_inliers, zncc, overlap) -> str:
    """"" if this fit is trustworthy, else why it is not.

    Photometric agreement is the primary test (a wrong fit scores ~0 however
    many inliers RANSAC mustered), but a fit with heavy geometric support is
    allowed a weaker correlation: the hazy frames correlate poorly even when
    correctly placed.
    """
    if n_inliers < MIN_INLIERS:
        return "only %d inliers" % n_inliers
    if overlap < MIN_OVERLAP:
        return "overlap %.2f of frame" % overlap
    if zncc >= MIN_ZNCC:
        return ""
    if zncc >= MIN_ZNCC_STRONG and n_inliers >= STRONG_INLIERS:
        return ""
    return "zncc %.2f (%d inliers)" % (zncc, n_inliers)


def register_frame(frame_path, ortho_path, nav_e, nav_n, footprint_m, transform=None,
                   gsd_m=GSD_M, win_mul=WIN_MUL, clahe_clip=CLAHE_CLIP) -> Registration:
    """Register one frame onto one mosaic.  Returns a Registration (.ok tells you).

    `footprint_m` is the frame's expected ground width -- get it from
    ChunkIndex.footprint_m(), which derives it from the chunk's own ortho GSD.
    The returned H maps FULL-RESOLUTION frame pixels straight to UTM metres.
    """
    fn = Path(frame_path).name
    sift, bf, clahe = _kit(clahe_clip)
    win = win_mul * footprint_m
    try:
        with rasterio.open(ortho_path) as ds:
            if transform is None:
                transform = ds.transform
        crop, ovalid, w = read_ortho_window(ortho_path, transform, nav_e, nav_n,
                                            win, gsd_m)
    except (OSError, rasterio.errors.RasterioError) as exc:
        return Registration(fn=fn, fail="ortho read: %s" % exc)
    if ovalid.mean() < MIN_VALID_FRAC:
        return Registration(fn=fn, fail="ortho window %.1f%% imaged" % (100 * ovalid.mean()))

    img = cv2.imread(str(frame_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return Registration(fn=fn, fail="frame unreadable")
    tw = max(64, int(round(footprint_m / gsd_m)))
    th = max(36, int(round(tw * img.shape[0] / img.shape[1])))
    small = _prep(cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA), clahe)

    # keypoints only off real seafloor: erode so the swath edge is not a feature
    kmask = cv2.erode(ovalid.astype(np.uint8), np.ones((9, 9), np.uint8)) * 255
    crop_p = _prep(crop, clahe, ovalid)
    k1, d1 = sift.detectAndCompute(small, None)
    k2, d2 = sift.detectAndCompute(crop_p, kmask)
    if d1 is None or d2 is None or len(k1) < 10 or len(k2) < 10:
        return Registration(fn=fn, fail="too few features")
    good = [m for m, nn in bf.knnMatch(d1, d2, k=2) if m.distance < RATIO * nn.distance]
    if len(good) < 10:
        return Registration(fn=fn, n_matches=len(good), fail="too few matches")
    good = vote_matches(good, k1, k2)
    if len(good) < 10:
        return Registration(fn=fn, n_matches=len(good), fail="too few consistent matches")
    src = np.float32([k1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([k2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    side = crop.shape[0]
    hp_crop = _highpass(crop_p)
    ones = np.ones_like(small, np.uint8)

    def score(M):
        """Warp the frame onto the mosaic and score the overlap photometrically."""
        Hs = np.vstack([M, [0, 0, 1]]) if M.shape[0] == 2 else M
        warp = cv2.warpPerspective(small, Hs, (side, side))
        cov = cv2.warpPerspective(ones, Hs, (side, side)) > 0
        m = cov & ovalid
        # fraction of the frame as PLACED (not as read) that lands on real mosaic
        return Hs, _zncc(_highpass(warp), hp_crop, m), m.sum() / max(float(cov.sum()), 1.0)

    # similarity first (4 DOF, keeps the frame's aspect ratio), then try to upgrade
    cands = []
    M, mask = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC,
                                          ransacReprojThreshold=RANSAC_PX,
                                          maxIters=20000, confidence=0.9995)
    if M is not None:
        cands.append(("similarity", M, int(mask.sum())))
        inl = mask.ravel().astype(bool)
        if inl.sum() >= 12:                    # upgrade on the similarity's inliers only
            Hh, hmask = cv2.findHomography(src[inl], dst[inl], cv2.RANSAC, RANSAC_PX,
                                           maxIters=20000, confidence=0.9995)
            if Hh is not None:
                cands.append(("homography", Hh, int(hmask.sum())))
    if not cands:
        return Registration(fn=fn, n_matches=len(good), fail="no transform")

    best = None
    for kind, M, ni in cands:
        Hs, z, ov = score(M)
        okq, fw, why = _shape_ok(_compose(Hs, tw, th, w, side, transform), footprint_m)
        cand = (z if okq else -1.0, kind, Hs, ni, z, ov, fw, why)
        if best is None or cand[0] > best[0]:
            best = cand
    _, kind, Hs, ni, z, ov, fw, why = best
    H = _compose(Hs, tw, th, w, side, transform)
    part = dict(fn=fn, n_inliers=ni, n_matches=len(good), zncc=z, overlap=ov,
                chunk=Path(ortho_path).parent.name, footprint_m=fw)
    bad = why or accepted(ni, z, ov)
    return Registration(fail=bad, **part) if bad else Registration(H=H, **part)


def register_frame_best(frame_path, ortho_path, nav_e, nav_n, footprint_m,
                        transform=None, ladder=LADDER, **kw) -> Registration:
    """Try the settings ladder until one verifies; else return the closest miss.

    Different frames fail for different reasons -- fine texture washed out by
    haze, or coarse structure swamped by a mosaic seam -- so no single working
    resolution registers them all, and trying a few is much cheaper than losing
    the frame.
    """
    best = None
    for gsd_m, win_mul, clip in ladder:
        r = register_frame(frame_path, ortho_path, nav_e, nav_n, footprint_m,
                           transform=transform, gsd_m=gsd_m, win_mul=win_mul,
                           clahe_clip=clip, **kw)
        if r.ok:
            return r
        if best is None or r.zncc > best.zncc:
            best = r
    return best


def _compose(Hs, tw, th, window, side, transform) -> np.ndarray:
    """full-res frame px -> working px -> crop px -> ortho px -> UTM metres."""
    A_small = np.diag([tw / FRAME_W, th / FRAME_H, 1.0])
    A_crop = np.array([[window.width / side, 0.0, window.col_off],
                       [0.0, window.height / side, window.row_off],
                       [0.0, 0.0, 1.0]])
    t = transform
    A_utm = np.array([[t.a, t.b, t.c], [t.d, t.e, t.f], [0.0, 0.0, 1.0]])
    return A_utm @ A_crop @ Hs @ A_small


# -- projection helpers --------------------------------------------------------------

def _apply(M, pts) -> np.ndarray:
    p = np.atleast_2d(np.asarray(pts, dtype=float))
    h = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ M.T
    return h[:, :2] / h[:, 2:3]


def frame_to_utm(H, px_points) -> np.ndarray:
    """Full-res frame pixels [(x, y), ...] -> UTM [(E, N), ...]."""
    return _apply(np.asarray(H, dtype=float), px_points)


def utm_to_frame(H_inv, utm_points) -> np.ndarray:
    """UTM [(E, N), ...] -> full-res frame pixels [(x, y), ...].

    Pass a Registration.H_inv (or np.linalg.inv(H)).
    """
    return _apply(np.asarray(H_inv, dtype=float), utm_points)


def frame_footprint(H) -> np.ndarray:
    """The frame's four corners on the seafloor, UTM, in pixel-corner order."""
    return frame_to_utm(H, [(0, 0), (FRAME_W, 0), (FRAME_W, FRAME_H), (0, FRAME_H)])


def sees_point(reg: Registration, e: float, n: float, margin_px=0.0) -> bool:
    """Does this frame image the UTM point (inside the frame, `margin_px` inset)?"""
    if not reg.ok:
        return False
    x, y = utm_to_frame(reg.H_inv, [(e, n)])[0]
    return (margin_px <= x <= FRAME_W - margin_px
            and margin_px <= y <= FRAME_H - margin_px)


# -- cached batch registration -------------------------------------------------------

def cache_path(workspace_dir) -> Path:
    return Path(workspace_dir) / "survey" / "multiview" / "frame_H.json"


def load_cache(workspace_dir) -> dict:
    try:
        raw = json.loads(cache_path(workspace_dir).read_text())
    except (OSError, ValueError):
        return {}
    return {fn: Registration.from_json(fn, d) for fn, d in raw.items()}


def save_cache(workspace_dir, regs: dict) -> str:
    p = cache_path(workspace_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({fn: r.to_json() for fn, r in sorted(regs.items())}, indent=1))
    return str(p)


def register_frames(workspace_dir, frames, frames_dir, seg="seg15", manifest=None,
                    index=None, workers=6, refresh=False, log=print, **kw) -> dict:
    """Register `frames` (filenames), using and updating the on-disk cache.

    Returns {fn: Registration} for every frame asked for, failures included, so
    callers can report a registration rate honestly instead of silently dropping
    the frames that did not work.
    """
    man = load_manifest(workspace_dir, seg) if manifest is None else manifest
    idx = ChunkIndex(workspace_dir, seg, manifest=man) if index is None else index
    frames = list(dict.fromkeys(frames))
    cached = load_cache(workspace_dir) if not refresh else {}
    out = {fn: cached[fn] for fn in frames if fn in cached}
    todo = [fn for fn in frames if fn not in out]
    log("register_frames: %d frames, %d cached, %d to do"
        % (len(frames), len(out), len(todo)))

    def one(fn):
        if fn not in man.index:
            return Registration(fn=fn, fail="no nav fix")
        chunk = idx.for_frame(fn)
        if chunk is None:
            return Registration(fn=fn, fail="not in any chunk reconstruction")
        row = man.loc[fn]
        try:
            return register_frame_best(f"{frames_dir}/{fn}", chunk.path,
                                       float(row.easting), float(row.northing),
                                       idx.footprint_m(fn, row.alt),
                                       transform=chunk.transform, **kw)
        except Exception as exc:                 # one bad frame must not kill the run
            return Registration(fn=fn, fail="error: %s" % exc)

    if todo:
        done = 0
        with ThreadPoolExecutor(max_workers=workers) as ex:
            for reg in ex.map(one, todo):
                out[reg.fn] = reg
                done += 1
                if done % 25 == 0 or done == len(todo):
                    log("  %d/%d attempted, %d ok"
                        % (done, len(todo), sum(1 for r in out.values() if r.ok)))
        merged = dict(cached)
        merged.update(out)
        save_cache(workspace_dir, merged)
    ok = sum(1 for r in out.values() if r.ok)
    log("register_frames: %d/%d ok (%.1f%%)"
        % (ok, len(out), 100.0 * ok / max(len(out), 1)))
    return out


def _cli(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        print("usage: multiview_geometry.py <workspace> [frames_dir] [n_sample]",
              file=sys.stderr)
        return 2
    ws = argv[0]
    frames_dir = argv[1] if len(argv) > 1 else \
        "/home/troyboland/biigle/storage/images/J1754/frames"
    n = int(argv[2]) if len(argv) > 2 else 20
    man = load_manifest(ws)
    idx = ChunkIndex(ws, manifest=man)
    print("manifest %d frames; %d chunks" % (len(man), len(idx.chunks)))
    for c in idx.chunks.values():
        print("  %s gsd=%.3f mm K=%.3f cams=%d" % (c.name, c.gsd * 1000, c.K, c.n_cameras))
    have = [f for f in man.index if idx.for_frame(f) and Path(frames_dir, f).is_file()]
    print("%d manifest frames present in %s" % (len(have), frames_dir))
    sample = have[::max(1, len(have) // n)][:n]
    regs = register_frames(ws, sample, frames_dir, manifest=man, index=idx, refresh=True)
    for fn, r in regs.items():
        print("  %s inl=%-4d z=%+.2f ov=%.2f fp=%.2fm %s"
              % (fn[-22:], r.n_inliers, r.zncc, r.overlap, r.footprint_m,
                 r.chunk if r.ok else "FAIL: " + r.fail))
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
