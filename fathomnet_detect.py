#!/usr/bin/env python3
"""FathomNet megafauna detection on survey frames — full-corpus, Qt-free.

Runs the FathomNet/MBARI-315k YOLOv8 detector (499 classes, CC-BY-4.0) over a
dive's exported down-looking frames, buckets the classes into the four
annotation groups already live on BIIGLE (crab / worm / fish / other), places
every surviving detection at its frame's nav fix (EPSG:32613) and asks whether
megafauna are denser inside the gas-anomaly windows than outside them.

Localisation is FRAME-LEVEL: a frame footprint at 3-7 m altitude spans roughly
5-7 m on the seafloor, and every detection in a frame gets that frame's single
easting/northing.  Within-frame pixel offsets are NOT converted to ground
offsets (frames carry nav but no per-pixel georef), so a point's true position
is its frame centre +/- ~3 m.  Good enough for window-scale density, useless
for individual-scale spatial statistics.

Pipeline per dive:
  detect_frames()      frame dir -> raw detection table (full-res px boxes)
  load_frame_nav()     survey/photogrammetry/seg*/segment_*/interp.csv
                       (easting/northing live here; survey/biigle/frames.csv
                        has nav but no UTM) joined on frame basename
  to_geojson()         one Point per KEPT detection + per-frame density CSV
  density_vs_anomaly() per-frame counts in vs out of survey/anomaly/
                       window_context.csv windows; medians + Mann-Whitney

Writes to <ws>/survey/fauna/:
  fathomnet_detections.csv   ALL detections incl. junk, with 'excluded' reason
  fauna_points_utm.geojson   kept detections only, EPSG:32613
  fauna_density.csv          every scanned frame, counts per bucket, nav, flags
  fauna_vs_anomaly.png       dark in/out density figure
and a pooled paper/fathomnet_full_summary.json.

CLI:
  python3 fathomnet_detect.py                     # both dives, full corpus
  python3 fathomnet_detect.py --dives J1756 --limit 50 --no-geojson
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
PAPER = ROOT / "paper"
FRAMES_ROOT = Path("/home/troyboland/biigle/storage/images")   # fast local disk
WEIGHTS = Path("/home/troyboland/models/mbari_315k_yolov8.pt")
UTM_EPSG = 32613           # EPSG:32613 — WGS84 / UTM 13N, as every other product

CONF = 0.25                # matches the seg15 pilot run pushed to BIIGLE
IMGSZ = 1280
BATCH = 32
DIVES = ("J1754", "J1756")

# ---------------------------------------------------------------- buckets ----
# Substring keywords (case-insensitive) against the 499 MBARI-315k class names.
# Morphology-level triage buckets, not taxonomy: crustacean / worm / fish /
# anemone / unknown.  The model's job is "find biology and give a coarse shape
# prior"; conclusive species ID is a downstream human step, so anything animal
# that matches no bucket lands in "unknown" rather than being guessed.  First
# bucket whose keyword matches wins, so order matters.
BUCKETS: dict[str, tuple[str, ...]] = {
    "crustacean": (
        "munidopsis", "munida", "galatheoidea", "lithodidae", "paralomis",
        "neolithodes", "chionoecetes", "chorilia", "cancridae", "moloha",
        "paguroidea", "sternostylus", "pleuroncodes", "decapoda", "caridea",
        "pandalidae", "pandalus", "plesionika", "heterocarpus", "crustacea",
        "eualus", "glyphocrangon", "acanthephyra", "pasiphaea", "crab",
        # "amphipoda" dropped 2026-09-21: 0/8 review precision (debris/shells),
        # crustacea deep-dive; those detections now route to unknown for humans
        "carapace", "isopoda", "munnopsidae", "acanthamunnopsis",
    ),
    "worm": (
        "riftia", "lamellibrachia", "siboglinidae", "polychaeta", "sabellidae",
        "serpulidae", "terebellidae", "harmothoe", "anchinothria", "osedax",
        "worm", "tube", "nemertea", "echiura", "hirudinea", "saxipendium",
        "poeobius", "swima",
    ),
    "fish": (
        "actinopterygii", "zoarcidae", "macrouridae", "coryphaenoides",
        "albatrossia", "antimora", "sebastes", "sebastolobus", "liparidae",
        "careproctus", "bathypterois", "bathysaurus", "lycodes", "lycenchelys",
        "lycodapus", "pachycara", "cataetyx", "cherublemma", "physiculus",
        "coelorinchus", "anguilliformes", "bathycongrus", "gnathophis",
        "nettastoma", "hydrolagus", "harriotta", "rajiformes", "bathyraja",
        "beringraja", "amblyraja", "tetronarce", "squalus", "apristurus",
        "pentanchidae", "parmaturus", "cephalurus", "eptatretus",
        "pleuronectiformes", "microstomus", "glyptocephalus", "embassichthys",
        "lyopsetta", "eopsetta", "symphurus", "xeneretmus", "zalembius",
        "cottidae", "psychrolutes", "chaunacops", "lophiodes", "dibranchus",
        "melanocetus", "oneirodes", "gigantactis", "anoplogaster",
        "myctophidae", "bathylagidae", "pseudobathylagus", "merluccius",
        "anoplopoma", "anarrhichthys", "liopropoma", "serranus", "scopelarchus",
        "thalassobathia", "thrissacanthias", "icichthys", "trachipterus",
        "desmodema", "alepisaurus", "anotopterus", "fish",
    ),
    "anemone": (
        "anemone", "actiniar", "actinostol", "actinernus", "hormathiid",
        "metridium", "liponema", "bolocera", "corallimorph", "cerianth",
        "zoanth", "isosicyonis",
    ),
}
BUCKET_ORDER = ("crustacean", "worm", "fish", "anemone", "unknown")

# ------------------------------------------------------------ junk filter ----
# The detector was trained on MBARI's NE-Pacific ROV corpus; on a 2500 m EPR
# down-looking frame the pelagic end of that label set can only be domain-shift
# error (usually one huge box over a turbid/backscatter field).  Excluded from
# the GeoJSON and the density analysis, KEPT in the raw CSV with excluded=
# "midwater".  One justification per line:
MIDWATER_EXCLUDE = frozenset({
    "Mola mola",                # ocean sunfish, epipelagic (<500 m) — impossible
    "Bathochordaeus",           # giant larvacean, midwater mucus house
    "Pyrosoma atlanticum",      # pelagic colonial tunicate, never on-bottom here
    "Pyrosoma detritus",        # its sinking debris — same source, same artefact
    "Salpida", "salp chain", "salp detritus", "Thetys vagina",  # pelagic salps
    "Appendicularia", "Larvacea", "Oikopleura",                 # pelagic tunicates
    "Doryteuthis opalescens",   # market squid — CA shelf/epipelagic, MBARI-local
    "Doryteuthis opalescens eggs",
    "Dosidicus gigas", "Onykia robusta", "Gonatus", "Gonatopsis",  # pelagic squid
    "Chiroteuthis calyx", "Galiteuthis", "Taonius", "Cranchiidae",
    "Helicocranchia", "Leachia danae", "Histioteuthis heteropsis",
    "Stigmatoteuthis dofleini", "Octopoteuthis deletron", "Planctoteuthis",
    "Grimalditeuthis", "Magnapinna pacifica", "Abraliopsis", "Teuthoidea",
    "Bathyteuthis berryi", "Bathyteuthis berryi eggs", "Vampyroteuthis infernalis",
    "Chaetognatha", "Eukrohnia fowleri", "Caecosagitta macrocephala",  # arrow worms
    "Pteropoda", "Thecosomata", "Gymnosomata", "Clio", "Clione",  # pelagic snails
    "Carinaria japonica", "Corolla", "Creseis virgula", "Hyalocylis striata",
    "Notobranchaea macdonaldi", "Procymbulia", "Thliptodon", "Heteropoda",
    "Euphausia", "Krill molt", "Mysida", "Boreomysis", "Copepoda",  # zooplankton
    "Euchirella bitumida", "Phronima sedentaria", "Eusergestes similis",
    "Hymenodora", "Systellaspis", "Cerataspis monstrosus",
    "Tomopteris", "Tomopterid eggcase",   # pelagic polychaete (would bucket as worm)
    "Lagenorhynchus obliquidens", "Zalophus californianus",  # dolphin, sea lion
    "Diomedea", "Uria aalge",             # albatross, murre — surface birds
    "marine snow", "mung", "ink",          # water-column particulate artefacts
    "Yoda", "LRJ complex", "marine organism", "Animalia",  # unusable placeholders
})

# Not animals at all (substrate, gear, dead material).  Also dropped from the
# spatial/density products, flagged excluded="nonfauna" in the raw CSV.
NONFAUNA_EXCLUDE = frozenset({
    "detritus", "sand", "geologic", "bacterial mat", "shell", "bone", "wood",
    "kelp", "trash", "anchor", "sinker", "equipment", "Equipment",
    "Detritus Sampler", "Suction Sampler", "inner filter", "outer filter",
    "ship container", "molt", "carcass", "medusa carcass", "eggcase",
    "Rajiformes eggcase", "Plantae detritus", "Phyllospadix-Zostera detritus",
    "Neptunea-Buccinum Complex eggcase", "Buccinidae sp. 1 eggs",
    "amphipod tube mat", "stalk", "gastrozooid",
})


def bucket_of(cls: str, buckets: dict[str, tuple[str, ...]] = BUCKETS) -> str:
    low = str(cls).lower()
    for name, keys in buckets.items():
        if any(k in low for k in keys):
            return name
    return "unknown"


def exclusion_of(cls: str, midwater=MIDWATER_EXCLUDE, nonfauna=NONFAUNA_EXCLUDE) -> str:
    """'' = keep, else the documented reason this class is dropped."""
    if cls in midwater:
        return "midwater"
    if cls in nonfauna:
        return "nonfauna"
    return ""


# ---------------------------------------------------------------- detect -----
def detect_frames(frame_dir, weights=WEIGHTS, conf=CONF, imgsz=IMGSZ,
                  batch=BATCH, device=0, limit=None, log=print) -> pd.DataFrame:
    """Run the detector over every .jpg in frame_dir.

    Returns fn, cls, conf, x1, y1, x2, y2 in ORIGINAL full-res pixels
    (ultralytics rescales boxes off the imgsz letterbox for us).  Frames with
    no detection produce no row — use the frame list for the zeros.
    """
    from ultralytics import YOLO

    paths = sorted(Path(frame_dir).glob("*.jpg"))
    if limit:
        paths = paths[:int(limit)]
    if not paths:
        raise FileNotFoundError(f"no frames in {frame_dir}")
    log(f"[fathomnet] {len(paths)} frames <- {frame_dir}")
    model = YOLO(str(weights))
    names = model.names

    rows, t0 = [], time.time()
    for i in range(0, len(paths), batch):
        chunk = paths[i:i + batch]
        try:
            res = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz,
                                device=device, verbose=False)
        except RuntimeError as exc:              # CUDA OOM -> halve and retry once
            if "out of memory" not in str(exc).lower() or len(chunk) < 2:
                raise
            log(f"[fathomnet] CUDA OOM at batch {i}; retrying in halves")
            import torch; torch.cuda.empty_cache()
            half = len(chunk) // 2
            res = []
            for sub in (chunk[:half], chunk[half:]):
                res += model.predict([str(p) for p in sub], conf=conf,
                                     imgsz=imgsz, device=device, verbose=False)
        for path, r in zip(chunk, res):
            b = r.boxes
            if b is None or len(b) == 0:
                continue
            xyxy = b.xyxy.cpu().numpy()
            cf = b.conf.cpu().numpy()
            cl = b.cls.cpu().numpy().astype(int)
            for (x1, y1, x2, y2), c, k in zip(xyxy, cf, cl):
                rows.append((path.name, names[int(k)], float(c),
                             float(x1), float(y1), float(x2), float(y2)))
        if (i // batch) % 20 == 0 or i + batch >= len(paths):
            done = min(i + batch, len(paths))
            log(f"[fathomnet]   {done}/{len(paths)} frames, {len(rows)} det, "
                f"{time.time() - t0:.0f} s")

    det = pd.DataFrame(rows, columns=["fn", "cls", "conf", "x1", "y1", "x2", "y2"])
    det["bucket"] = det.cls.map(bucket_of)
    det["excluded"] = det.cls.map(exclusion_of)
    det.attrs["n_frames"] = len(paths)
    det.attrs["frames"] = [p.name for p in paths]
    det.attrs["runtime_s"] = time.time() - t0
    log(f"[fathomnet] done: {len(det)} detections on "
        f"{det.fn.nunique() if len(det) else 0}/{len(paths)} frames in "
        f"{det.attrs['runtime_s']:.0f} s GPU")
    return det


# ------------------------------------------------------------------- nav -----
def load_frame_nav(workspace_dir) -> pd.DataFrame:
    """Frame nav incl. UTM, from the photogrammetry interp.csv manifests.

    survey/biigle/frames.csv carries lat/lon only; the per-segment manifests
    carry easting/northing (UTM 13N) for the same basenames, so join there.
    """
    ws = Path(workspace_dir)
    mans = sorted(ws.glob("survey/photogrammetry/seg*/segment_*/interp.csv"))
    if not mans:
        raise FileNotFoundError(f"no interp.csv manifests under {ws}")
    cols = ["frame_filename", "unix_time", "easting", "northing", "alt",
            "depth", "heading"]
    out = []
    for m in mans:
        d = pd.read_csv(m, usecols=lambda c: c in cols)
        d["seg"] = m.parent.parent.name
        out.append(d)
    nav = pd.concat(out, ignore_index=True)
    nav = nav.drop_duplicates("frame_filename").set_index("frame_filename")
    return nav


# --------------------------------------------------------------- products ----
def to_geojson(det_df, workspace_dir, out_path, density_path=None,
               frames=None, log=print):
    """Per-detection Point GeoJSON (EPSG:32613) + per-frame bucket density CSV.

    Every detection in a frame gets that frame's nav fix, so points stack at
    frame centres: frame-level localisation, ~+/-3 m (footprint half-width).
    """
    nav = load_frame_nav(workspace_dir)
    keep = det_df[det_df.excluded == ""].copy()
    miss = sorted(set(keep.fn) - set(nav.index))
    if miss:
        log(f"[fathomnet] WARNING {len(miss)} detected frames absent from "
            f"manifests, dropped from GeoJSON (e.g. {miss[0]})")
        keep = keep[keep.fn.isin(nav.index)]

    j = nav.reindex(keep.fn.to_numpy())
    feats = []
    for (_, d), e, n, a, dp, sg, ut in zip(
            keep.iterrows(), j.easting, j.northing, j.alt, j.depth,
            j.seg, j.unix_time):
        if not (np.isfinite(e) and np.isfinite(n)):
            continue
        feats.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [float(e), float(n)]},
            "properties": {
                "cls": d.cls, "bucket": d.bucket, "conf": round(float(d.conf), 4),
                "fn": d.fn, "alt": None if not np.isfinite(a) else round(float(a), 2),
                "depth_m": None if not np.isfinite(dp) else round(float(dp), 2),
                "seg": sg, "unix_time": float(ut),
                "box_px": [round(float(v), 1) for v in (d.x1, d.y1, d.x2, d.y2)],
                "loc": "frame-centre (frame-level localisation, ~+/-3 m)",
            },
        })
    gj = {"type": "FeatureCollection",
          "crs": {"type": "name",
                  "properties": {"name": f"urn:ogc:def:crs:EPSG::{UTM_EPSG}"}},
          "features": feats}
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(gj))
    log(f"[fathomnet] {len(feats)} points -> {out_path}")

    # per-frame density over ALL scanned frames (zeros included)
    frames = list(frames if frames is not None else det_df.attrs.get("frames", []))
    if not frames:
        frames = sorted(set(det_df.fn))
    dens = pd.DataFrame({"frame_filename": frames}).set_index("frame_filename")
    for b in BUCKET_ORDER:
        dens[f"n_{b}"] = keep[keep.bucket == b].groupby("fn").size().reindex(
            dens.index).fillna(0).astype(int)
    dens["n_total"] = dens[[f"n_{b}" for b in BUCKET_ORDER]].sum(axis=1)
    dens["n_raw"] = det_df.groupby("fn").size().reindex(dens.index).fillna(0).astype(int)
    dens = dens.join(nav[["unix_time", "easting", "northing", "alt", "depth", "seg"]])
    dens = dens.reset_index()
    if density_path:
        Path(density_path).parent.mkdir(parents=True, exist_ok=True)
        dens.to_csv(density_path, index=False)
        log(f"[fathomnet] density {len(dens)} frames -> {density_path}")
    return dens


def fish_frame_shortlist(det_df, workspace_dir, out_path, log=print) -> pd.DataFrame:
    """Frames ranked by likelihood of containing fish — a human-triage product.

    The fish bucket is recall-oriented (precision is knowingly poor); the point
    is that a biologist reviews this shortlist top-down instead of 5,000 frames.
    Midwater/nonfauna exclusions still apply.  One row per frame: n_fish, best
    confidence, classes seen, nav fix.
    """
    keep = det_df[(det_df.excluded == "") & (det_df.bucket == "fish")]
    if keep.empty:
        log("[fathomnet] fish shortlist: no candidate frames")
        return pd.DataFrame()
    g = keep.groupby("fn")
    short = pd.DataFrame({
        "n_fish": g.size(),
        "max_conf": g.conf.max().round(4),
        "classes": g.cls.agg(lambda s: "; ".join(sorted(set(s)))),
    })
    nav = load_frame_nav(workspace_dir)
    short = short.join(nav[["unix_time", "easting", "northing", "alt", "depth",
                            "seg"]], how="left")
    short = short.sort_values("max_conf", ascending=False).reset_index()
    out_path = Path(out_path); out_path.parent.mkdir(parents=True, exist_ok=True)
    short.to_csv(out_path, index=False)
    log(f"[fathomnet] fish shortlist: {len(short)} frames -> {out_path}")
    return short


# ----------------------------------------------------- density vs anomaly ----
BG, INK, MUT, GRID = "#0e1620", "#dbe4ec", "#9fb0bd", "#2a3846"
GOLD, SLATE = "#f2c94c", "#5a7d9a"        # in-window / out-of-window


def _flag_windows(dens, win, pad=0.0):
    """Boolean per frame: unix_time inside any [start-pad, end+pad] window."""
    t0 = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp) - pad
    t1 = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp) + pad
    t = dens.unix_time.to_numpy(float)
    flag = np.zeros(len(dens), bool)
    for a, b in zip(t0.to_numpy(), t1.to_numpy()):
        flag |= (t >= a) & (t <= b)
    return flag


def _boot_mean(x, n=2000, seed=20260917):
    if len(x) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    m = rng.choice(x, size=(n, len(x)), replace=True).mean(axis=1)
    return (float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)))


def density_vs_anomaly(workspace_dir, dens, out_png=None, dive="", pad=0.0,
                       log=print) -> dict:
    """Per-frame megafauna counts in vs out of gas-anomaly windows.

    Windows from survey/anomaly/window_context.csv (all tiers); the
    station/transit tag is carried so the transit-only subset — the population
    the paper's detection rate is built on — is reported alongside.
    """
    ws = Path(workspace_dir)
    win = pd.read_csv(ws / "survey" / "anomaly" / "window_context.csv")
    dens = dens.dropna(subset=["unix_time"]).copy()
    dens["in_window"] = _flag_windows(dens, win, pad)
    tr = win[win.context == "transit"] if "context" in win else win
    dens["in_transit_window"] = _flag_windows(dens, tr, pad)

    res = {"dive": dive, "pad_s": pad, "n_windows": int(len(win)),
           "n_transit_windows": int(len(tr)),
           "n_frames": int(len(dens)),
           "n_frames_in_window": int(dens.in_window.sum()),
           "n_frames_in_transit_window": int(dens.in_transit_window.sum()),
           "buckets": {}}
    for scope, col in (("all_windows", "in_window"),
                       ("transit_windows", "in_transit_window")):
        out = {}
        for b in BUCKET_ORDER + ("total",):
            c = f"n_{b}" if b != "total" else "n_total"
            a = dens.loc[dens[col], c].to_numpy(float)
            o = dens.loc[~dens[col], c].to_numpy(float)
            r = {"n_in": int(len(a)), "n_out": int(len(o)),
                 "median_in": float(np.median(a)) if len(a) else np.nan,
                 "median_out": float(np.median(o)) if len(o) else np.nan,
                 "mean_in": float(a.mean()) if len(a) else np.nan,
                 "mean_out": float(o.mean()) if len(o) else np.nan,
                 "frac_frames_any_in": float((a > 0).mean()) if len(a) else np.nan,
                 "frac_frames_any_out": float((o > 0).mean()) if len(o) else np.nan}
            r["ci95_mean_in"] = _boot_mean(a)
            r["ci95_mean_out"] = _boot_mean(o)
            r["ratio_mean"] = (float(a.mean() / o.mean())
                               if len(a) and len(o) and o.mean() > 0 else np.nan)
            if len(a) >= 3 and len(o) >= 3:
                u, p = mannwhitneyu(a, o, alternative="two-sided")
                r["mannwhitney_u"], r["p"] = float(u), float(p)
            else:
                r["mannwhitney_u"], r["p"] = np.nan, np.nan
            out[b] = r
        res["buckets"][scope] = out

    if out_png:
        _figure(res, dens, out_png, dive)
        log(f"[fathomnet] figure -> {out_png}")
    return res


def _figure(res, dens, out_png, dive):
    """Two rows: all windows (in-window often covers most of the dive) and
    transit windows only (the paper's detection population, cleaner split)."""
    cols = BUCKET_ORDER + ("total",)
    scopes = (("all_windows", f"all {res['n_windows']} windows"),
              ("transit_windows", f"transit windows only "
                                  f"({res['n_transit_windows']})"))
    fig, axes = plt.subplots(len(scopes), len(cols),
                             figsize=(2.9 * len(cols), 3.6 * len(scopes)),
                             facecolor=BG, squeeze=False)
    for row, (scope, label) in enumerate(scopes):
        stats = res["buckets"][scope]
        for col, b in enumerate(cols):
            ax, r = axes[row][col], stats[b]
            ax.set_facecolor(BG)
            means = [r["mean_in"], r["mean_out"]]
            los = [means[0] - r["ci95_mean_in"][0], means[1] - r["ci95_mean_out"][0]]
            his = [r["ci95_mean_in"][1] - means[0], r["ci95_mean_out"][1] - means[1]]
            ax.bar([0, 1], means, color=[GOLD, SLATE], width=0.62,
                   yerr=[los, his], capsize=4, error_kw=dict(ecolor=MUT, lw=1.2))
            for x, m in zip((0, 1), means):
                ax.text(x, m, f"{m:.3f}", ha="center", va="bottom",
                        color=INK, fontsize=8.5)
            ax.set_xticks([0, 1])
            ax.set_xticklabels([f"in\nn={r['n_in']}", f"out\nn={r['n_out']}"],
                               color=MUT, fontsize=8.5)
            p = r["p"]
            ptxt = "p n/a" if not np.isfinite(p) else (
                "p<0.001" if p < 1e-3 else f"p={p:.3g}")
            ratio = r["ratio_mean"]
            rtxt = f"  x{ratio:.2f}" if np.isfinite(ratio) and ratio > 0 else ""
            ax.set_title(f"{b}\n{ptxt}{rtxt}", color=INK, fontsize=10)
            ax.grid(axis="y", color=GRID, lw=0.6, alpha=0.7)
            ax.set_axisbelow(True)
            for s in ax.spines.values():
                s.set_color(GRID)
            ax.tick_params(colors=MUT, labelsize=8.5)
        axes[row][0].set_ylabel(f"{label}\ndetections / frame (mean, 95% boot CI)",
                                color=INK, fontsize=9)
    fig.suptitle(f"{dive} — FathomNet megafauna density inside gas-anomaly "
                 f"windows vs rest of dive ({res['n_frames']} frames)",
                 color=INK, fontsize=12)
    fig.text(0.5, 0.008, "frame-level localisation (frame centres, ~+/-3 m); "
             "Mann-Whitney two-sided on per-frame counts; midwater / non-fauna "
             "classes excluded", ha="center", color=MUT, fontsize=8)
    fig.tight_layout(rect=[0, 0.025, 1, 0.955])
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=150, facecolor=BG)
    plt.close(fig)


# ------------------------------------------------------------------- main ----
def run_dive(dive, frames_root=FRAMES_ROOT, root=ROOT, weights=WEIGHTS,
             conf=CONF, imgsz=IMGSZ, batch=BATCH, device=0, limit=None,
             geojson=True, log=print) -> dict:
    ws = Path(root) / f"{dive}_down.eprproj"
    out = ws / "survey" / "fauna"
    out.mkdir(parents=True, exist_ok=True)
    det = detect_frames(Path(frames_root) / dive / "frames", weights=weights,
                        conf=conf, imgsz=imgsz, batch=batch, device=device,
                        limit=limit, log=log)
    det.to_csv(out / "fathomnet_detections.csv", index=False)
    log(f"[fathomnet] raw -> {out / 'fathomnet_detections.csv'}")

    dens = to_geojson(det, ws, out / "fauna_points_utm.geojson",
                      density_path=out / "fauna_density.csv",
                      frames=det.attrs.get("frames"), log=log) if geojson else None
    stats = (density_vs_anomaly(ws, dens, out_png=out / "fauna_vs_anomaly.png",
                                dive=dive, log=log) if dens is not None else {})
    fish_frame_shortlist(det, ws, out / "fish_frame_shortlist.csv", log=log)

    kept = det[det.excluded == ""]
    summary = {
        "dive": dive,
        "workspace": str(ws),
        "weights": str(weights),
        "conf": conf, "imgsz": imgsz, "batch": batch,
        "n_frames": int(det.attrs["n_frames"]),
        "runtime_s": round(float(det.attrs["runtime_s"]), 1),
        "n_detections_raw": int(len(det)),
        "n_detections_kept": int(len(kept)),
        "n_excluded_midwater": int((det.excluded == "midwater").sum()),
        "n_excluded_nonfauna": int((det.excluded == "nonfauna").sum()),
        "frames_with_detection": int(kept.fn.nunique()),
        "by_class_raw": det.cls.value_counts().to_dict(),
        "by_class_kept": kept.cls.value_counts().to_dict(),
        "by_bucket_kept": kept.bucket.value_counts().to_dict(),
        "excluded_classes": det[det.excluded != ""].cls.value_counts().to_dict(),
        "density_vs_anomaly": stats,
        "outputs": {k: str(out / v) for k, v in (
            ("detections_csv", "fathomnet_detections.csv"),
            ("points_geojson", "fauna_points_utm.geojson"),
            ("density_csv", "fauna_density.csv"),
            ("figure", "fauna_vs_anomaly.png"))},
    }
    return summary


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dives", nargs="*", default=list(DIVES))
    ap.add_argument("--frames-root", default=str(FRAMES_ROOT))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--conf", type=float, default=CONF)
    ap.add_argument("--imgsz", type=int, default=IMGSZ)
    ap.add_argument("--batch", type=int, default=BATCH)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-geojson", action="store_true")
    ap.add_argument("--summary", default=None,
                    help="pooled JSON (default <root>/paper/fathomnet_full_summary.json)")
    a = ap.parse_args(argv)

    dev = int(a.device) if str(a.device).isdigit() else a.device
    pooled = {"generated": pd.Timestamp.now("UTC").isoformat(),
              "weights": a.weights, "conf": a.conf, "imgsz": a.imgsz,
              "buckets": {k: list(v) for k, v in BUCKETS.items()},
              "midwater_exclude": sorted(MIDWATER_EXCLUDE),
              "nonfauna_exclude": sorted(NONFAUNA_EXCLUDE),
              "localisation": "frame centre (nav fix); frame footprint ~5-7 m at "
                              "3-7 m altitude, so ~+/-3 m per point",
              "dives": {}}
    for dive in a.dives:
        pooled["dives"][dive] = run_dive(
            dive, frames_root=a.frames_root, root=a.root, weights=a.weights,
            conf=a.conf, imgsz=a.imgsz, batch=a.batch, device=dev,
            limit=a.limit, geojson=not a.no_geojson)
    sp = Path(a.summary) if a.summary else Path(a.root) / "paper" / "fathomnet_full_summary.json"
    sp.parent.mkdir(parents=True, exist_ok=True)
    sp.write_text(json.dumps(pooled, indent=2, default=str))
    print(f"[fathomnet] pooled summary -> {sp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
