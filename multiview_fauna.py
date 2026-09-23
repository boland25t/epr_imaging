#!/usr/bin/env python3
"""Multi-view consistency, deduplication and training-data propagation for the
per-frame megafauna detections.

A downward camera flying a 5 m survey sees the same patch of seafloor in several
consecutive frames, so every real animal on the bottom is imaged repeatedly while
a false positive -- a shadow, a marine-snow blob, a midwater drifter that never
lands on the mosaic -- is not.  Once every detection is placed in UTM by
`multiview_geometry`, that redundancy does three jobs at once:

  1. LOCALIZE  each detection's bbox is projected to the seafloor (metres), so a
     detection becomes a position and a ground size rather than a pixel box.
  2. FILTER + DEDUP  detections within a tolerance of each other are one organism
     (class-agnostic clustering, class assigned afterwards by vote).  n_views is
     the number of DISTINCT frames in the cluster: >= 2 is 'confirmed', 1 is
     'single-view' -- unconfirmed, and the population where the false positives
     and the midwater junk collect.  The deduplicated organism count, not the
     detection count, is what a density estimate needs.
  3. PROPAGATE  a confirmed organism's ground box is reprojected into every other
     frame that images it, excluding frames that already detected it.  Those are
     exactly the frames where the detector missed a known animal, which makes
     them the training examples worth annotating.

Step 3's boxes plus the confirmed raw detections are pushed to BIIGLE under one
new label so a human can accept or reject them in the annotation tool.

Registration fails on some frames (motion blur, unimaged mosaic zones); the
registration and localization rates are reported rather than hidden, since they
bound everything downstream.

Qt-free.  CLI:  python3 multiview_fauna.py <workspace> <detections.csv> [options]
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

import multiview_geometry as mvg

# Clustering tolerances (metres).
MIN_TOL = 0.35            # floor: registration slop + ortho/frame parallax
SIZE_FRAC = 0.5           # ... or half the organisms' mean ground size, whichever is larger
MOBILE_TOL = 0.80         # mobile animals move between views; allow them further

# Mobile classes (matched case-insensitively as substrings of the class name):
# fish, decapods/galatheids and cephalopods swim or walk between frames.
MOBILE_CLASSES = ("actinopterygii", "coryphaenoides", "antimora", "hydrolagus",
                  "mola mola", "psychrolutes", "munidopsis", "galatheoidea",
                  "amphipoda", "doryteuthis", "octopod", "octopus")

# Classes that are midwater by biology: if the multi-view logic works, these should
# land 'single-view' -- they drift past the camera and never appear on the mosaic.
MIDWATER_CLASSES = ("bathochordaeus", "pyrosoma", "mola mola", "caecosagitta",
                    "siphonophorae", "erenna", "lilyopsis")

PROP_MARGIN_PX = 40       # propagated boxes must sit this far inside the frame
MIN_BOX_PX = 24           # ... and be at least this big to be worth annotating
PROP_SLOP_M = 0.08        # registration slop a propagated box must swallow
BIIGLE_LABEL = "multiview confirmed (model)"
BIIGLE_COLOR = "00b3a4"   # teal: distinct from every existing label in tree 4617
BULK_LIMIT = 100          # server cap on POST image-annotations


def _is(cls: str, names) -> bool:
    c = str(cls).lower()
    return any(n in c for n in names)


# -- 1. localize ---------------------------------------------------------------------

def localize(dets: pd.DataFrame, regs: dict) -> pd.DataFrame:
    """Project every detection bbox onto the seafloor.

    Adds E/N (bbox centre, UTM metres), ground_w_m/ground_h_m (projected bbox
    edge lengths) and `localized`.  Detections on unregistrable frames keep
    localized=False and are carried through as the 'unlocalized' population.
    """
    out = dets.copy()
    for c in ("E", "N", "ground_w_m", "ground_h_m"):
        out[c] = np.nan
    out["localized"] = False
    out["n_inliers"] = 0
    for fn, grp in out.groupby("fn", sort=False):
        reg = regs.get(fn)
        if reg is None or not reg.ok:
            continue
        b = grp[["x1", "y1", "x2", "y2"]].to_numpy(dtype=float)
        # corners in pixel order TL, TR, BR, BL, then the centre
        px = np.concatenate([
            np.stack([b[:, 0], b[:, 1]], 1), np.stack([b[:, 2], b[:, 1]], 1),
            np.stack([b[:, 2], b[:, 3]], 1), np.stack([b[:, 0], b[:, 3]], 1),
            np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2], 1)])
        u = mvg.frame_to_utm(reg.H, px).reshape(5, len(b), 2)
        top, right = u[1] - u[0], u[3] - u[0]
        out.loc[grp.index, "E"] = u[4][:, 0]
        out.loc[grp.index, "N"] = u[4][:, 1]
        out.loc[grp.index, "ground_w_m"] = np.hypot(top[:, 0], top[:, 1])
        out.loc[grp.index, "ground_h_m"] = np.hypot(right[:, 0], right[:, 1])
        out.loc[grp.index, "localized"] = True
        out.loc[grp.index, "n_inliers"] = reg.n_inliers
    return out


# -- 2. filter + dedup ---------------------------------------------------------------

def _tolerance(sizes, classes, min_tol, size_frac, mobile_tol) -> float:
    tol = max(size_frac * float(np.mean(sizes)), min_tol)
    if any(_is(c, MOBILE_CLASSES) for c in classes):
        tol = max(tol, mobile_tol)
    return tol


def cluster_organisms(loc: pd.DataFrame, min_tol=MIN_TOL, size_frac=SIZE_FRAC,
                      mobile_tol=MOBILE_TOL) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Group localized detections into organisms (leader clustering, class-agnostic).

    Seeded from the most confident detection outward so a cluster is anchored on
    its best evidence rather than on whatever came first; the centroid is
    recomputed as members join, which keeps clusters compact instead of chaining
    across a dense Munidopsis field the way single-link would.

    Returns (detections with organism_id/verdict, organisms table).
    """
    d = loc[loc.localized].copy()
    if d.empty:
        return loc.assign(organism_id=pd.NA, verdict="unlocalized"), pd.DataFrame()
    d = d.reset_index().rename(columns={"index": "_src"})
    E = d.E.to_numpy(); N = d.N.to_numpy()
    size = 0.5 * (d.ground_w_m.to_numpy() + d.ground_h_m.to_numpy())
    cls = d.cls.to_numpy()
    order = np.argsort(-d.conf.to_numpy())
    assigned = np.full(len(d), -1)

    oid = 0
    for seed in order:
        if assigned[seed] >= 0:
            continue
        members = [seed]
        assigned[seed] = oid
        grew = True
        while grew:
            grew = False
            ce, cn = E[members].mean(), N[members].mean()
            free = np.flatnonzero(assigned < 0)
            if not len(free):
                break
            dist = np.hypot(E[free] - ce, N[free] - cn)
            for j, dj in zip(free, dist):
                tol = _tolerance(np.append(size[members], size[j]),
                                 list(cls[members]) + [cls[j]],
                                 min_tol, size_frac, mobile_tol)
                if dj <= tol:
                    members.append(j)
                    assigned[j] = oid
                    grew = True
        oid += 1

    d["organism_id"] = ["O%04d" % a for a in assigned]
    rows = []
    for o, g in d.groupby("organism_id", sort=True):
        vc = g.cls.value_counts()
        top = vc[vc == vc.max()].index
        cl = g[g.cls.isin(top)].sort_values("conf", ascending=False).cls.iloc[0]
        nv = g.fn.nunique()
        sz = 0.5 * (g.ground_w_m + g.ground_h_m)
        rows.append(dict(
            organism_id=o, cls=cl, n_views=nv, n_dets=len(g),
            conf_max=round(float(g.conf.max()), 4),
            conf_mean=round(float(g.conf.mean()), 4),
            E=round(float(g.E.mean()), 3), N=round(float(g.N.mean()), 3),
            ground_w_m=round(float(g.ground_w_m.median()), 3),
            ground_h_m=round(float(g.ground_h_m.median()), 3),
            radius_m=round(float(0.5 * sz.median()), 3),
            spread_m=round(float(np.hypot(g.E - g.E.mean(), g.N - g.N.mean()).max()), 3),
            verdict="confirmed" if nv >= 2 else "single-view",
            frames="|".join(sorted(g.fn.unique())),
            classes="|".join("%s:%d" % (k, v) for k, v in vc.items()),
        ))
    org = pd.DataFrame(rows).sort_values(["n_views", "conf_max"], ascending=False)
    verdict = dict(zip(org.organism_id, org.verdict))
    d["verdict"] = d.organism_id.map(verdict)

    full = loc.copy()
    full["organism_id"] = pd.NA
    full["verdict"] = "unlocalized"
    full.loc[d._src.to_numpy(), "organism_id"] = d.organism_id.to_numpy()
    full.loc[d._src.to_numpy(), "verdict"] = d.verdict.to_numpy()
    return full, org.reset_index(drop=True)


# -- 3. propagate --------------------------------------------------------------------

def propagate(org: pd.DataFrame, fdets: pd.DataFrame, regs: dict,
              margin_px=PROP_MARGIN_PX, min_box_px=MIN_BOX_PX) -> pd.DataFrame:
    """Reproject every confirmed organism into the other frames that image it.

    The organism's ground extent is a square centred on its UTM position; its
    corners go through each frame's inverse homography and the axis-aligned pixel
    hull of those corners is the training box.  Frames that already detected the
    organism are skipped -- those annotations exist.

    The square is the organism's radius GROWN by the positional disagreement its
    own views show (half its cluster spread, at least `PROP_SLOP_M`).  Registration
    here is good to ~0.1-0.2 m, which is ample to tell one animal from another but
    not to place a 5 cm crab pixel-perfectly, so a box sized to the animal alone
    would often miss it.  A slightly loose box that reliably contains the animal is
    what a human adjusting training annotations can actually use.
    """
    seen = fdets.dropna(subset=["organism_id"]).groupby("organism_id").fn.apply(set).to_dict()
    ok_regs = {fn: r for fn, r in regs.items() if r.ok}
    rows = []
    for o in org[org.verdict == "confirmed"].itertuples():
        r = max(float(o.radius_m), 0.05) + max(0.5 * float(o.spread_m), PROP_SLOP_M)
        sq = [(o.E - r, o.N - r), (o.E + r, o.N - r),
              (o.E + r, o.N + r), (o.E - r, o.N + r)]
        already = seen.get(o.organism_id, set())
        for fn, reg in ok_regs.items():
            if fn in already:
                continue
            p = mvg.utm_to_frame(reg.H_inv, sq)
            if not np.isfinite(p).all():
                continue
            x1, y1 = p.min(0)
            x2, y2 = p.max(0)
            if not (margin_px <= x1 and margin_px <= y1
                    and x2 <= mvg.FRAME_W - margin_px and y2 <= mvg.FRAME_H - margin_px):
                continue
            if min(x2 - x1, y2 - y1) < min_box_px:
                continue
            rows.append(dict(fn=fn, cls=o.cls, conf=o.conf_max,
                             x1=round(float(x1), 1), y1=round(float(y1), 1),
                             x2=round(float(x2), 1), y2=round(float(y2), 1),
                             organism_id=o.organism_id, source="propagated"))
    return pd.DataFrame(rows, columns=["fn", "cls", "conf", "x1", "y1", "x2", "y2",
                                       "organism_id", "source"])


# -- 4. BIIGLE -----------------------------------------------------------------------

def ensure_label(api, tree_id, name=BIIGLE_LABEL, color=BIIGLE_COLOR, log=print) -> int:
    """Idempotently ensure one label exists on an existing tree; returns its id."""
    tree = api.get("label-trees/%d" % int(tree_id))
    for l in tree.get("labels", []):
        if l.get("name") == name:
            log("label '%s' already on tree %s (id %s)" % (name, tree_id, l["id"]))
            return int(l["id"])
    made = api.post("label-trees/%d/labels" % int(tree_id),
                    json={"name": name, "color": color.lstrip("#")})
    if isinstance(made, list):
        made = made[0] if made else {}
    log("created label '%s' (#%s) on tree %s -> id %s" % (name, color, tree_id,
                                                          made.get("id")))
    return int(made["id"])


def push_boxes(api, volume_id, boxes: pd.DataFrame, label_id, log=print) -> dict:
    """Bulk-create rectangle annotations for the rows whose frame is in the volume.

    `boxes` needs fn, x1, y1, x2, y2 and conf.  Returns counts; rows whose frame
    is not a file of the volume are skipped (and counted), not an error.
    """
    from biigle_bridge import SHAPES
    names = api.get("volumes/%s/filenames" % volume_id)
    by_name = {v: int(k) for k, v in names.items()}
    anns, skipped = [], 0
    for b in boxes.itertuples():
        iid = by_name.get(b.fn)
        if iid is None:
            skipped += 1
            continue
        anns.append({"image_id": iid, "shape_id": SHAPES["rectangle"],
                     "label_id": int(label_id),
                     "confidence": float(min(1.0, max(0.01, b.conf))),
                     "points": [float(b.x1), float(b.y1), float(b.x2), float(b.y1),
                                float(b.x2), float(b.y2), float(b.x1), float(b.y2)]})
    pushed = 0
    for i in range(0, len(anns), BULK_LIMIT):
        api.post("image-annotations", json=anns[i:i + BULK_LIMIT])
        pushed += len(anns[i:i + BULK_LIMIT])
        log("  pushed %d/%d" % (pushed, len(anns)))
        time.sleep(0.3)                      # polite on top of the client throttle
    return {"pushed": pushed, "skipped_not_in_volume": skipped,
            "frames": int(boxes[boxes.fn.isin(by_name)].fn.nunique())}


# -- QA figure -----------------------------------------------------------------------

def _qa_row(axes, o, raw, pro, regs, frames_dir, max_panels):
    """Draw one organism's views along `axes`; returns the frames used."""
    import cv2
    import matplotlib.pyplot as plt

    panels = list(dict.fromkeys(list(raw.fn.unique()) + list(pro.fn.unique())))[:max_panels]
    boxes = pd.concat([raw, pro])[["x1", "y1", "x2", "y2"]]
    half = int(max(260, 0.8 * (boxes.x2 - boxes.x1).max(), 0.8 * (boxes.y2 - boxes.y1).max()))
    half = min(half, mvg.FRAME_H // 2)
    for ax, fn in zip(axes, panels):
        img = cv2.imread(str(Path(frames_dir) / fn))
        reg = regs.get(fn)
        if img is None or reg is None or not reg.ok:
            ax.axis("off")
            continue
        cx, cy = mvg.utm_to_frame(reg.H_inv, [(o.E, o.N)])[0]
        x0 = int(np.clip(cx - half, 0, mvg.FRAME_W - 2 * half))
        y0 = int(np.clip(cy - half, 0, mvg.FRAME_H - 2 * half))
        ax.imshow(cv2.cvtColor(img[y0:y0 + 2 * half, x0:x0 + 2 * half], cv2.COLOR_BGR2RGB))
        for src, colour in ((raw, "#39d353"), (pro, "#31d2f2")):
            for b in src[src.fn == fn].itertuples():
                ax.add_patch(plt.Rectangle((b.x1 - x0, b.y1 - y0), b.x2 - b.x1,
                                           b.y2 - b.y1, fill=False, ec=colour, lw=2.2))
        kind = "raw detection" if fn in set(raw.fn) else "PROPAGATED (no detection)"
        ax.plot(cx - x0, cy - y0, "+", color="#ffd166", ms=11, mew=1.6)
        ax.set_title("%s\n%s  inl=%d" % (fn[-18:], kind, reg.n_inliers), fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    for ax in axes[len(panels):]:
        ax.axis("off")
    return panels


def qa_figure(out_png, org: pd.DataFrame, fdets: pd.DataFrame, prop: pd.DataFrame,
              regs: dict, frames_dir, max_panels=4, log=print) -> str:
    """Two rows: one organism that tests the dedup, one that tests the propagation.

    Every crop is centred on the organism's reprojected UTM position, so if the
    geometry is right the same animal sits in the middle of every panel of a row.

    Row 1 is the TIGHTEST confirmed cluster -- one animal seen repeatedly, whose
    views must all land on it; that is the honest test of the geometry.  A loose
    cluster (spread >> tolerance) is an aggregation, a Riftia bush, whose "views"
    are partly different tubes, and it would test nothing.  Row 2 is the organism
    with the most propagated boxes, so the figure also shows boxes in frames where
    the detector found nothing.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cand = org[org.verdict == "confirmed"]
    if cand.empty:
        log("qa_figure: no confirmed organism to draw")
        return ""
    npro = prop.groupby("organism_id").size() if len(prop) else pd.Series(dtype=int)
    cand = cand.assign(n_prop=cand.organism_id.map(npro).fillna(0).astype(int))
    multi = cand[cand.n_views >= 3]
    rows = [(multi if len(multi) else cand).sort_values(
        ["spread_m", "n_views"], ascending=[True, False]).iloc[0]]
    withprop = cand[cand.n_prop > 0].sort_values(["n_prop", "n_views"], ascending=False)
    if len(withprop) and withprop.iloc[0].organism_id != rows[0].organism_id:
        rows.append(withprop.iloc[0])

    fig, axes = plt.subplots(len(rows), max_panels,
                             figsize=(3.9 * max_panels, 5.1 * len(rows)), squeeze=False)
    used, heads = [], []
    for r, o in enumerate(rows):
        raw = fdets[fdets.organism_id == o.organism_id]
        pro = prop[prop.organism_id == o.organism_id] if len(prop) else fdets.iloc[:0]
        used.append(_qa_row(axes[r], o, raw, pro, regs, frames_dir, max_panels))
        axes[r][0].set_ylabel(o.organism_id, fontsize=9)
        heads.append("%s — %s  %s  n_views=%d  n_prop=%d  radius=%.2f m  spread=%.3f m"
                     % ("DEDUP CHECK" if r == 0 else "PROPAGATION CHECK",
                        o.organism_id, o.cls, o.n_views, o.n_prop, o.radius_m, o.spread_m))
    fig.suptitle("multi-view QA: green = raw detection, cyan = propagated box, "
                 "+ = organism's reprojected UTM position", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94 if len(rows) > 1 else 0.90))
    if len(rows) > 1:
        fig.subplots_adjust(hspace=0.30)       # room for a header above each row
    # place each row's header clear of that row's per-panel titles, now that the
    # axes have real positions
    for r, head in enumerate(heads):
        top = max(ax.get_position().y1 for ax in axes[r])
        fig.text(0.5, min(top + 0.10 / len(rows), 0.965), head, ha="center",
                 va="bottom", fontsize=10)
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=105)
    plt.close(fig)
    log("qa_figure: %s (%s)" % (out_png, ", ".join(
        "%s:%d panels" % (o.organism_id, len(u)) for o, u in zip(rows, used))))
    return str(out_png)


# -- report --------------------------------------------------------------------------

def report(loc, org, prop, regs, log=print) -> None:
    nreg = sum(1 for r in regs.values() if r.ok)
    log("\n-- registration ------------------------------------------------")
    log("  frames attempted      %d" % len(regs))
    log("  registered            %d (%.1f%%)" % (nreg, 100.0 * nreg / max(len(regs), 1)))
    fails = pd.Series([r.fail.split(":")[0].split("(")[0].strip()
                       for r in regs.values() if not r.ok]).value_counts()
    for k, v in fails.items():
        log("    %-34s %d" % (k, v))
    log("-- localization ------------------------------------------------")
    nl = int(loc.localized.sum())
    log("  detections            %d" % len(loc))
    log("  localized             %d (%.1f%%)" % (nl, 100.0 * nl / max(len(loc), 1)))
    log("  unlocalized           %d" % (len(loc) - nl))
    log("-- organisms ---------------------------------------------------")
    log("  organisms             %d  (from %d localized detections)" % (len(org), nl))
    vc = org.verdict.value_counts()
    log("  confirmed (>=2 views) %d" % int(vc.get("confirmed", 0)))
    log("  single-view           %d" % int(vc.get("single-view", 0)))
    if len(org):
        log("  views histogram       %s" % org.n_views.value_counts().sort_index().to_dict())
        t = (org.groupby("cls").verdict.value_counts().unstack(fill_value=0)
             .reindex(columns=["confirmed", "single-view"], fill_value=0))
        t["n_dets"] = org.groupby("cls").n_dets.sum()
        log("  per class:")
        for line in t.sort_values("n_dets", ascending=False).to_string().splitlines():
            log("    " + line)
        mid = org[org.cls.map(lambda c: _is(c, MIDWATER_CLASSES))]
        if len(mid):
            sv = int((mid.verdict == "single-view").sum())
            log("  midwater-by-biology classes (%s):" % ", ".join(sorted(mid.cls.unique())))
            log("    %d/%d single-view (%.0f%%) -- theory predicts ~all"
                % (sv, len(mid), 100.0 * sv / len(mid)))
    log("-- propagation -------------------------------------------------")
    log("  propagated boxes      %d over %d frames, %d organisms"
        % (len(prop), prop.fn.nunique() if len(prop) else 0,
           prop.organism_id.nunique() if len(prop) else 0))


# -- CLI -----------------------------------------------------------------------------

def run(workspace, dets_csv, frames_dir, seg="seg15", volume_id=36137, tree_id=4617,
        push=True, refresh=False, workers=6, min_tol=MIN_TOL, mobile_tol=MOBILE_TOL,
        log=print) -> dict:
    out_dir = Path(workspace) / "survey" / "multiview"
    out_dir.mkdir(parents=True, exist_ok=True)
    dets = pd.read_csv(dets_csv)
    log("detections: %d rows, %d frames, %d classes"
        % (len(dets), dets.fn.nunique(), dets.cls.nunique()))

    man = mvg.load_manifest(workspace, seg)
    idx = mvg.ChunkIndex(workspace, seg, manifest=man)
    regs = mvg.register_frames(workspace, sorted(dets.fn.unique()), frames_dir,
                               seg=seg, manifest=man, index=idx, workers=workers,
                               refresh=refresh, log=log)

    loc = localize(dets, regs)
    fdets, org = cluster_organisms(loc, min_tol=min_tol, mobile_tol=mobile_tol)
    prop = propagate(org, fdets, regs)

    paths = {}
    for name, df in (("organisms.csv", org), ("filtered_detections.csv", fdets),
                     ("propagated_training.csv", prop)):
        p = out_dir / name
        df.to_csv(p, index=False)
        paths[name] = str(p)
        log("wrote %s (%d rows)" % (p, len(df)))
    report(loc, org, prop, regs, log=log)

    paths["multiview_qa.png"] = qa_figure(out_dir / "multiview_qa.png", org, fdets,
                                          prop, regs, frames_dir, log=log)

    stats = {}
    if push:
        from biigle_bridge import BiigleApi
        api = BiigleApi()
        label_id = ensure_label(api, tree_id, log=log)
        conf_raw = fdets[fdets.verdict == "confirmed"][
            ["fn", "cls", "conf", "x1", "y1", "x2", "y2", "organism_id"]].copy()
        log("-- BIIGLE push (volume %s, label %s) --------------------------"
            % (volume_id, label_id))
        log("  confirmed raw detections: %d" % len(conf_raw))
        stats["raw"] = push_boxes(api, volume_id, conf_raw, label_id, log=log)
        log("  propagated boxes: %d" % len(prop))
        stats["propagated"] = push_boxes(api, volume_id, prop, label_id, log=log) \
            if len(prop) else {"pushed": 0, "skipped_not_in_volume": 0, "frames": 0}
        st = api.get("volumes/%s/statistics" % volume_id)
        stats["volume"] = {"annotatedFiles": st.get("annotatedFiles"),
                           "labels": {l["name"]: l["count"]
                                      for l in st.get("annotationLabels", [])}}
        log("  pushed raw=%d propagated=%d" % (stats["raw"]["pushed"],
                                               stats["propagated"]["pushed"]))
        log("  volume %s now: %s annotated files, %s" % (
            volume_id, stats["volume"]["annotatedFiles"], stats["volume"]["labels"]))
    return {"paths": paths, "organisms": org, "detections": fdets,
            "propagated": prop, "regs": regs, "biigle": stats}


def _cli(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("workspace")
    ap.add_argument("detections_csv")
    ap.add_argument("--frames-dir", default="/home/troyboland/biigle/storage/images/J1754/frames")
    ap.add_argument("--seg", default="seg15")
    ap.add_argument("--volume-id", type=int, default=36137)
    ap.add_argument("--tree-id", type=int, default=4617)
    ap.add_argument("--no-push", action="store_true", help="skip the BIIGLE upload")
    ap.add_argument("--refresh", action="store_true", help="ignore the registration cache")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--min-tol", type=float, default=MIN_TOL)
    ap.add_argument("--mobile-tol", type=float, default=MOBILE_TOL)
    a = ap.parse_args(argv)
    run(a.workspace, a.detections_csv, a.frames_dir, seg=a.seg, volume_id=a.volume_id,
        tree_id=a.tree_id, push=not a.no_push, refresh=a.refresh, workers=a.workers,
        min_tol=a.min_tol, mobile_tol=a.mobile_tol)
    return 0


if __name__ == "__main__":
    sys.exit(_cli())
