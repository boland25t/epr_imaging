#!/usr/bin/env python3
"""Agent-in-the-loop vision verification of FathomNet megafauna detections.

Phases
------
sheets   : render contact sheets of every kept detection (bbox crop + context
           margin) so an agent with image understanding can adjudicate each one
           at *bucket* level (crab / worm / fish / other).
probe    : render a stratified sample of full frames with a labelled 8x6 grid
           overlay so the agent can record animals the detector MISSED.
stats    : fold the agent's verdicts.csv into review_stats.json (per-bucket
           confirm / relabel / reject / uncertain rates = zero-shot precision).
dataset  : build a segment-split YOLO dataset from confirmed + relabelled
           detections and the recall-probe frames (incl. true negatives).
examples : pull out example crops referenced by the written report.

Owns only new files.  Reads the per-dive detection CSVs and the frame jpgs;
writes under /mnt/f/EPR_2026_PROCESSED/paper/fauna_review/ and
/home/troyboland/models/finetune/.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

# --------------------------------------------------------------------------- #
# paths / constants
# --------------------------------------------------------------------------- #

DIVES = ("J1754", "J1756")
PROC = Path("/mnt/f/EPR_2026_PROCESSED")
FRAME_ROOT = Path("/home/troyboland/biigle/storage/images")
OUT = PROC / "paper" / "fauna_review"
SHEETS = OUT / "sheets"
PROBE = OUT / "probe"
EXAMPLES = OUT / "examples"
MODEL_OUT = Path("/home/troyboland/models/finetune")
DATASET = MODEL_OUT / "dataset"

BUCKETS = ("crab", "worm", "fish", "other")
BUCKET_ID = {b: i for i, b in enumerate(BUCKETS)}

# contact-sheet geometry: 6 x 4 cells, 256 px crop + 22 px caption bar.
# Total 1560 x 1136 px -> under the 1568 px long-edge cap, so the agent sees
# the crops at native resolution with no lossy downscale on read.
COLS, ROWS = 6, 4
CELL = 256
CAP = 22
GAP = 4
PER_SHEET = COLS * ROWS
CONTEXT = 0.40  # context margin, fraction of the bbox long side per side

# recall-probe geometry
PROBE_W = 1300
GRID_COLS, GRID_ROWS = 8, 6

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


def _font(sz: int, bold: bool = True) -> ImageFont.FreeTypeFont:
    p = FONT_B if bold else FONT_B.replace("-Bold", "")
    try:
        return ImageFont.truetype(p, sz)
    except OSError:  # pragma: no cover
        return ImageFont.load_default()


# --------------------------------------------------------------------------- #
# data loading
# --------------------------------------------------------------------------- #


def load_detections() -> pd.DataFrame:
    """All *kept* detections across dives, with a stable global det_id."""
    frames = []
    for dive in DIVES:
        csv = PROC / f"{dive}_down.eprproj" / "survey" / "fauna" / "fathomnet_detections.csv"
        df = pd.read_csv(csv)
        df["excluded"] = df["excluded"].fillna("").astype(str)
        df = df[df["excluded"] == ""].copy()
        df["dive"] = dive
        frames.append(df)
    det = pd.concat(frames, ignore_index=True)
    # stable order: dive, frame, then y then x  -> deterministic det_id
    det = det.sort_values(["dive", "fn", "y1", "x1"], kind="mergesort").reset_index(drop=True)
    det["det_id"] = np.arange(len(det))
    det["path"] = [str(FRAME_ROOT / d / "frames" / f) for d, f in zip(det["dive"], det["fn"])]
    # attach segment + altitude from the per-frame density table
    meta = load_frame_meta()
    det = det.merge(meta[["dive", "frame_filename", "seg", "alt", "depth"]],
                    left_on=["dive", "fn"], right_on=["dive", "frame_filename"], how="left")
    return det


def load_frame_meta() -> pd.DataFrame:
    """Per-frame table (every frame, incl. zero-detection frames)."""
    out = []
    for dive in DIVES:
        csv = PROC / f"{dive}_down.eprproj" / "survey" / "fauna" / "fauna_density.csv"
        df = pd.read_csv(csv)
        df["dive"] = dive
        out.append(df)
    m = pd.concat(out, ignore_index=True)
    m["path"] = [str(FRAME_ROOT / d / "frames" / f) for d, f in zip(m["dive"], m["frame_filename"])]
    return m


# --------------------------------------------------------------------------- #
# phase 1 : contact sheets
# --------------------------------------------------------------------------- #


def _crop_one(img: Image.Image, x1, y1, x2, y2) -> Image.Image:
    """Square crop around the bbox with CONTEXT margin, bbox outlined."""
    W, H = img.size
    bw, bh = x2 - x1, y2 - y1
    side = max(bw, bh) * (1 + 2 * CONTEXT)
    side = max(side, 96.0)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    half = side / 2.0
    L, T = cx - half, cy - half
    # clamp keeping the requested size where possible
    L = min(max(0.0, L), max(0.0, W - side))
    T = min(max(0.0, T), max(0.0, H - side))
    R, B = min(W, L + side), min(H, T + side)
    crop = img.crop((int(L), int(T), int(round(R)), int(round(B))))
    scale = CELL / max(crop.size)
    crop = crop.resize((max(1, int(round(crop.width * scale))),
                        max(1, int(round(crop.height * scale)))), Image.LANCZOS)
    cell = Image.new("RGB", (CELL, CELL), (18, 18, 18))
    cell.paste(crop, ((CELL - crop.width) // 2, (CELL - crop.height) // 2))
    # bbox outline in cell coords
    ox = (CELL - crop.width) // 2 - int(L) * scale
    oy = (CELL - crop.height) // 2 - int(T) * scale
    d = ImageDraw.Draw(cell)
    d.rectangle([x1 * scale + ox, y1 * scale + oy, x2 * scale + ox, y2 * scale + oy],
                outline=(60, 255, 90), width=2)
    return cell


def _render_sheet(job) -> str:
    """job = (sheet_idx, [ (det_id, path, x1,y1,x2,y2, bucket, conf, cls, dive) ... ])"""
    sheet_idx, rows = job
    sw = COLS * CELL + (COLS + 1) * GAP
    sh = ROWS * (CELL + CAP) + (ROWS + 1) * GAP
    sheet = Image.new("RGB", (sw, sh), (10, 10, 12))
    draw = ImageDraw.Draw(sheet)
    f_cap = _font(15)
    # group rows by source frame so each jpg is decoded once
    by_path: dict[str, list] = {}
    for r in rows:
        by_path.setdefault(r[1], []).append(r)
    cells: dict[int, Image.Image] = {}
    for path, rs in by_path.items():
        try:
            img = Image.open(path).convert("RGB")
        except Exception:
            continue
        for r in rs:
            try:
                cells[r[0]] = _crop_one(img, r[2], r[3], r[4], r[5])
            except Exception:
                pass
        img.close()
    for i, r in enumerate(rows):
        c, rr = i % COLS, i // COLS
        x = GAP + c * (CELL + GAP)
        y = GAP + rr * (CELL + CAP + GAP)
        cell = cells.get(r[0])
        if cell is not None:
            sheet.paste(cell, (x, y + CAP))
        else:
            draw.rectangle([x, y + CAP, x + CELL, y + CAP + CELL], fill=(40, 0, 0))
        cap = f"{r[0]:04d}  {r[6]}  {r[7]:.2f}"
        draw.rectangle([x, y, x + CELL, y + CAP], fill=(235, 235, 235))
        draw.text((x + 4, y + 3), cap, font=f_cap, fill=(0, 0, 0))
    SHEETS.mkdir(parents=True, exist_ok=True)
    op = SHEETS / f"sheet_{sheet_idx:04d}.jpg"
    sheet.save(op, quality=88)
    return str(op)


def cmd_sheets(args) -> None:
    det = load_detections()
    # deterministic shuffle so buckets are MIXED within each sheet (reduces the
    # "whole sheet is crab -> confirm everything" anchoring effect)
    order = det.sample(frac=1.0, random_state=0).reset_index(drop=True)
    order.to_csv(OUT / "sheet_order.csv", index=False)
    rows = list(zip(order.det_id, order.path, order.x1, order.y1, order.x2, order.y2,
                    order.bucket, order.conf, order.cls, order.dive))
    jobs = [(i, rows[i * PER_SHEET:(i + 1) * PER_SHEET])
            for i in range(math.ceil(len(rows) / PER_SHEET))]
    if args.limit:
        jobs = jobs[:args.limit]
    SHEETS.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, _ in enumerate(ex.map(_render_sheet, jobs, chunksize=1), 1):
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} sheets", flush=True)
    print(f"{len(jobs)} sheets -> {SHEETS}")


# --------------------------------------------------------------------------- #
# phase 2 : recall probe frames
# --------------------------------------------------------------------------- #


def _render_probe(job) -> str:
    pid, path, label = job
    img = Image.open(path).convert("RGB")
    W, H = img.size
    scale = PROBE_W / W
    img = img.resize((PROBE_W, int(round(H * scale))), Image.LANCZOS)
    d = ImageDraw.Draw(img, "RGBA")
    w, h = img.size
    cw, ch = w / GRID_COLS, h / GRID_ROWS
    f = _font(20)
    for c in range(1, GRID_COLS):
        d.line([c * cw, 0, c * cw, h], fill=(255, 255, 0, 110), width=1)
    for r in range(1, GRID_ROWS):
        d.line([0, r * ch, w, r * ch], fill=(255, 255, 0, 110), width=1)
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            tag = f"{chr(65 + r)}{c + 1}"
            d.rectangle([c * cw + 1, r * ch + 1, c * cw + 34, r * ch + 22], fill=(0, 0, 0, 150))
            d.text((c * cw + 4, r * ch + 2), tag, font=f, fill=(255, 255, 0, 255))
    d.rectangle([0, h - 24, w, h], fill=(0, 0, 0, 190))
    d.text((5, h - 22), label, font=f, fill=(255, 255, 255, 255))
    PROBE.mkdir(parents=True, exist_ok=True)
    op = PROBE / f"probe_{pid:03d}.jpg"
    img.save(op, quality=90)
    return str(op)


def cmd_probe(args) -> None:
    meta = load_frame_meta()
    meta = meta[meta["alt"].notna()].copy()
    rng = np.random.default_rng(7)
    picks = []
    for dive, g in meta.groupby("dive"):
        q = g["alt"].quantile([1 / 3, 2 / 3]).values
        g = g.assign(tercile=np.digitize(g["alt"], q))
        for terc, gg in g.groupby("tercile"):
            # half the cells from frames WITH detections, half from frames with none,
            # so the probe measures misses in both regimes
            has = gg[gg["n_total"] > 0]
            none = gg[gg["n_total"] == 0]
            n_per = args.n // (2 * 3 * 2)  # dives * terciles * (has/none)
            for src in (has, none):
                if len(src) == 0:
                    continue
                take = min(n_per, len(src))
                idx = rng.choice(len(src), size=take, replace=False)
                picks.append(src.iloc[idx])
    sel = pd.concat(picks, ignore_index=True)
    sel = sel.sample(frac=1.0, random_state=11).reset_index(drop=True)
    sel["probe_id"] = np.arange(len(sel))
    OUT.mkdir(parents=True, exist_ok=True)
    sel.to_csv(OUT / "probe_frames.csv", index=False)
    jobs = [(int(r.probe_id), r.path,
             f"probe {int(r.probe_id):03d} | {r.dive} {r.seg} alt={r.alt:.1f}m | "
             f"model found: crab {int(r.n_crab)} worm {int(r.n_worm)} fish {int(r.n_fish)} other {int(r.n_other)}")
            for r in sel.itertuples()]
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for n, _ in enumerate(ex.map(_render_probe, jobs, chunksize=1), 1):
            if n % 20 == 0:
                print(f"  {n}/{len(jobs)} probe frames", flush=True)
    print(f"{len(jobs)} probe frames -> {PROBE}")


# --------------------------------------------------------------------------- #
# phase 1b : stats
# --------------------------------------------------------------------------- #


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    hw = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - hw), min(1.0, c + hw))


def cmd_compile(args) -> None:
    """Expand the agent's compact per-sheet deviation log into verdicts.csv.

    Log format, one line per reviewed sheet::

        S<idx> <det_id>=<verdict>[|note] <det_id>=<verdict> ...

    Every detection on a reviewed sheet that is NOT listed defaults to
    ``confirm`` (the common case), which is what keeps the log small enough to
    write out by hand for ~5.7k detections.  Verdicts are ``confirm``,
    ``reject``, ``uncertain`` or ``relabel:<bucket>``.
    """
    order = pd.read_csv(OUT / "sheet_order.csv")
    sheet_of, rows_of = {}, {}
    for i, did in enumerate(order["det_id"]):
        s = i // PER_SHEET
        sheet_of[int(did)] = s
        rows_of.setdefault(s, []).append(int(did))
    log = Path(args.log).read_text().splitlines()
    out, seen_sheets, bad = [], set(), []
    for ln in log:
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        tok = ln.split()
        if not tok[0].upper().startswith("S"):
            bad.append(ln)
            continue
        s = int(tok[0][1:])
        seen_sheets.add(s)
        dev = {}
        for t in tok[1:]:
            if "=" not in t:
                bad.append(t)
                continue
            k, _, vv = t.partition("=")
            v, _, note = vv.partition("|")
            try:
                dev[int(k)] = (v.strip(), note.replace("_", " ").strip())
            except ValueError:
                bad.append(t)
        for did in rows_of.get(s, []):
            v, note = dev.pop(did, ("confirm", ""))
            out.append({"det_id": did, "sheet": s, "verdict": v, "note": note})
        for k in dev:  # listed but not on this sheet -> operator error, keep visible
            bad.append(f"S{s}:{k}-not-on-sheet")
    df = pd.DataFrame(out).drop_duplicates("det_id", keep="last").sort_values("det_id")
    det = load_detections()[["det_id", "dive", "fn", "cls", "conf", "bucket"]]
    df = df.merge(det, on="det_id", how="left")
    df.to_csv(OUT / "verdicts.csv", index=False)
    print(f"sheets reviewed {len(seen_sheets)}/{math.ceil(len(order)/PER_SHEET)}  "
          f"verdicts {len(df)}  problems {len(bad)}")
    if bad:
        print("PROBLEMS:", bad[:40])
    print(df["verdict"].str.split(":").str[0].value_counts().to_dict())


def cmd_misses(args) -> None:
    """Parse the agent's miss log into misses.csv and estimate per-bucket recall.

    Log format (whitespace separated)::

        <probe_id> <grid cell> <bucket> <size> <note>

    Recall for a bucket on the probe set is
    ``found / (found + missed)`` where ``found`` counts the model's own kept
    detections on those frames that the agent CONFIRMED or RELABELLED into that
    bucket (rejects are false positives, not finds), and ``missed`` counts the
    agent's log lines.  The count is approximate: in dense fields the agent
    logged representative individuals, not an exhaustive census, so recall in
    those buckets is an UPPER bound.
    """
    rows = []
    for ln in Path(args.log).read_text().splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        p = ln.split(None, 4)
        if len(p) < 3:
            continue
        rows.append({"probe_id": int(p[0]), "cell": p[1], "bucket": p[2],
                     "size": p[3] if len(p) > 3 else "cell",
                     "note": p[4] if len(p) > 4 else ""})
    miss = pd.DataFrame(rows)
    probe = pd.read_csv(OUT / "probe_frames.csv")
    miss = miss.merge(probe[["probe_id", "dive", "frame_filename", "seg", "alt"]],
                      on="probe_id", how="left")
    miss.to_csv(OUT / "misses.csv", index=False)

    # model detections on the probe frames, adjudicated
    det = load_detections()
    v = pd.read_csv(OUT / "verdicts.csv")
    v["kind"] = v["verdict"].astype(str).str.split(":").str[0]
    v["to"] = v["verdict"].astype(str).str.split(":").str[1].fillna("")
    d = det.merge(v[["det_id", "kind", "to"]], on="det_id", how="left")
    pf = set(zip(probe["dive"], probe["frame_filename"]))
    on_probe = d[[(a, b) in pf for a, b in zip(d["dive"], d["fn"])]].copy()
    on_probe["gt"] = np.where(on_probe["kind"] == "confirm", on_probe["bucket"], on_probe["to"])
    found = on_probe[on_probe["kind"].isin(["confirm", "relabel"])]["gt"].value_counts().to_dict()
    fp = int((on_probe["kind"] == "reject").sum())
    unc = int((on_probe["kind"] == "uncertain").sum())

    out = {"n_probe_frames": int(len(probe)),
           "n_frames_with_logged_misses": int(miss["probe_id"].nunique()),
           "n_frames_clean_no_miss": int(len(probe) - miss["probe_id"].nunique()),
           "model_dets_on_probe_frames": int(len(on_probe)),
           "model_true_positives": int(sum(found.values())),
           "model_false_positives": fp, "model_uncertain": unc,
           "logged_misses_total": int(len(miss)), "per_bucket": {}}
    for b in BUCKETS:
        f = int(found.get(b, 0))
        m = int((miss["bucket"] == b).sum())
        n = f + m
        lo, hi = _wilson(f, n)
        out["per_bucket"][b] = {"found": f, "missed": m,
                                "recall": round(f / n, 4) if n else None,
                                "recall_ci95": [round(lo, 4), round(hi, 4)]}
    f = int(sum(found.values()))
    m = int(len(miss))
    lo, hi = _wilson(f, f + m)
    out["overall"] = {"found": f, "missed": m, "recall": round(f / (f + m), 4),
                      "recall_ci95": [round(lo, 4), round(hi, 4)]}
    out["miss_size_mix"] = miss["size"].value_counts().to_dict()
    out["frames_with_dense_field_misses"] = sorted(
        miss[miss["note"].str.contains("dense|field|tuft", case=False, na=False)]["probe_id"].unique().tolist())
    (OUT / "recall_stats.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


def cmd_stats(args) -> None:
    det = load_detections().set_index("det_id")
    v = pd.read_csv(OUT / "verdicts.csv")
    v["verdict"] = v["verdict"].astype(str).str.strip()
    v["kind"] = v["verdict"].str.split(":").str[0]
    v["to"] = v["verdict"].str.split(":").str[1].fillna("")
    v = v.drop_duplicates("det_id", keep="last")
    j = v.join(det, on="det_id", rsuffix="_d")

    stats = {"n_detections_kept": int(len(det)), "n_reviewed": int(len(j)),
             "review_coverage": round(len(j) / len(det), 4), "per_bucket": {}, "overall": {},
             "per_class_top": {}, "confusion_pred_to_true": {}}

    for b in BUCKETS:
        s = j[j["bucket"] == b]
        n = len(s)
        conf = int((s["kind"] == "confirm").sum())
        rel = int((s["kind"] == "relabel").sum())
        rej = int((s["kind"] == "reject").sum())
        unc = int((s["kind"] == "uncertain").sum())
        dec = conf + rel + rej  # decided (uncertain excluded)
        lo, hi = _wilson(conf, dec)
        stats["per_bucket"][b] = {
            "n_reviewed": n, "confirm": conf, "relabel": rel, "reject": rej, "uncertain": unc,
            "precision_bucket": round(conf / dec, 4) if dec else None,
            "precision_ci95": [round(lo, 4), round(hi, 4)],
            "is_animal_rate": round((conf + rel) / dec, 4) if dec else None,
            "relabel_targets": s[s["kind"] == "relabel"]["to"].value_counts().to_dict(),
        }
    n = len(j)
    conf = int((j["kind"] == "confirm").sum())
    rel = int((j["kind"] == "relabel").sum())
    rej = int((j["kind"] == "reject").sum())
    unc = int((j["kind"] == "uncertain").sum())
    dec = conf + rel + rej
    lo, hi = _wilson(conf, dec)
    stats["overall"] = {"n_reviewed": n, "confirm": conf, "relabel": rel, "reject": rej,
                        "uncertain": unc, "n_decided": dec,
                        "precision_bucket": round(conf / dec, 4) if dec else None,
                        "precision_ci95": [round(lo, 4), round(hi, 4)],
                        "is_animal_rate": round((conf + rel) / dec, 4) if dec else None}
    # per FathomNet class (where the ML went wrong, by its own taxon label)
    for cls, s in j.groupby("cls"):
        if len(s) < 15:
            continue
        d = int(((s["kind"] != "uncertain")).sum())
        stats["per_class_top"][cls] = {
            "n": int(len(s)),
            "precision_bucket": round(int((s["kind"] == "confirm").sum()) / d, 4) if d else None,
            "reject_rate": round(int((s["kind"] == "reject").sum()) / d, 4) if d else None,
            "bucket": s["bucket"].mode().iat[0]}
    # predicted-bucket -> agent-bucket confusion (rejects folded as 'none')
    for b in BUCKETS:
        s = j[j["bucket"] == b]
        row = {}
        for _, r in s.iterrows():
            t = b if r["kind"] == "confirm" else (r["to"] if r["kind"] == "relabel"
                                                  else ("none" if r["kind"] == "reject" else "uncertain"))
            row[t] = row.get(t, 0) + 1
        stats["confusion_pred_to_true"][b] = row
    # per dive
    stats["per_dive"] = {}
    for dive, s in j.groupby("dive"):
        d = int((s["kind"] != "uncertain").sum())
        stats["per_dive"][dive] = {"n": int(len(s)),
                                   "precision_bucket": round(int((s["kind"] == "confirm").sum()) / d, 4) if d else None}
    # confidence-banded precision
    stats["precision_by_conf"] = {}
    bands = [(0.25, 0.35), (0.35, 0.45), (0.45, 0.60), (0.60, 1.01)]
    for lo_, hi_ in bands:
        s = j[(j["conf"] >= lo_) & (j["conf"] < hi_)]
        d = int((s["kind"] != "uncertain").sum())
        stats["precision_by_conf"][f"{lo_:.2f}-{hi_:.2f}"] = {
            "n": int(len(s)),
            "precision_bucket": round(int((s["kind"] == "confirm").sum()) / d, 4) if d else None,
            "is_animal_rate": round(int(s["kind"].isin(["confirm", "relabel"]).sum()) / d, 4) if d else None}

    (OUT / "review_stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: stats[k] for k in ("overall", "per_bucket")}, indent=2))


# --------------------------------------------------------------------------- #
# phase 3 : YOLO dataset
# --------------------------------------------------------------------------- #

FULL_W, FULL_H = 5312, 2988


def _norm(x1, y1, x2, y2):
    x1, x2 = max(0.0, min(x1, x2)), min(float(FULL_W), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(FULL_H), max(y1, y2))
    return ((x1 + x2) / 2 / FULL_W, (y1 + y2) / 2 / FULL_H,
            (x2 - x1) / FULL_W, (y2 - y1) / FULL_H)


def cmd_dataset(args) -> None:
    det = load_detections()
    v = pd.read_csv(OUT / "verdicts.csv")
    v["verdict"] = v["verdict"].astype(str).str.strip()
    v["kind"] = v["verdict"].str.split(":").str[0]
    v["to"] = v["verdict"].str.split(":").str[1].fillna("")
    v = v.drop_duplicates("det_id", keep="last")
    d = det.merge(v[["det_id", "kind", "to"]], on="det_id", how="left")
    d["kind"] = d["kind"].fillna("unreviewed")
    keep = d[d["kind"].isin(["confirm", "relabel"])].copy()
    keep["gt"] = np.where(keep["kind"] == "confirm", keep["bucket"], keep["to"])
    keep = keep[keep["gt"].isin(BUCKETS)]

    # frames whose detections are fully adjudicated -> usable as complete labels
    rev = d[d["kind"] != "unreviewed"]
    per_frame_tot = d.groupby(["dive", "fn"]).size()
    per_frame_rev = rev.groupby(["dive", "fn"]).size()
    full = set((per_frame_rev / per_frame_tot).dropna().pipe(lambda s: s[s >= 1.0]).index)

    labels = {}   # (dive, fn) -> list[(cls_id, cx,cy,w,h)]
    for r in keep.itertuples():
        if (r.dive, r.fn) not in full:
            continue
        labels.setdefault((r.dive, r.fn), []).append(
            (BUCKET_ID[r.gt], *_norm(r.x1, r.y1, r.x2, r.y2)))

    # recall-probe additions: agent-localised misses + verified true negatives
    mp = OUT / "misses.csv"
    probe_only: set = set()   # frames carrying coarse probe boxes -> train split only
    probe = pd.read_csv(OUT / "probe_frames.csv")
    pmap = {int(r.probe_id): (r.dive, r.frame_filename, r.seg) for r in probe.itertuples()}
    n_miss = 0
    if mp.exists():
        miss = pd.read_csv(mp)
        for r in miss.itertuples():
            if str(getattr(r, "bucket", "")) not in BUCKETS:
                continue
            if int(r.probe_id) not in pmap:
                continue
            dive, fn, _ = pmap[int(r.probe_id)]
            box = _grid_box(str(r.cell), str(getattr(r, "size", "cell")))
            if box is None:
                continue
            labels.setdefault((dive, fn), []).append((BUCKET_ID[r.bucket], *box))
            probe_only.add((dive, fn))
            n_miss += 1
    # true negatives: probe frames the agent marked clean (no dets, no misses)
    n_tn = 0
    if args.true_negatives:
        missed_pids = set()
        if mp.exists():
            missed_pids = set(pd.read_csv(mp)["probe_id"].astype(int))
        zero = probe[probe["n_total"] == 0]
        for r in zero.itertuples():
            if int(r.probe_id) in missed_pids:
                continue
            labels.setdefault((r.dive, r.frame_filename), [])
            probe_only.add((r.dive, r.frame_filename))
            n_tn += 1

    # split by SEGMENT (overlapping frames within a segment must not straddle)
    segmap = load_frame_meta().set_index(["dive", "frame_filename"])["seg"].to_dict()
    segs = sorted({f"{k[0]}/{segmap.get(k, 'segNA')}" for k in labels})
    rnd = random.Random(23)
    rnd.shuffle(segs)
    # greedy: fill val to ~20% of the boxes
    box_by_seg: dict[str, int] = {}
    for k, L in labels.items():
        s = f"{k[0]}/{segmap.get(k, 'segNA')}"
        box_by_seg[s] = box_by_seg.get(s, 0) + max(1, len(L))
    total = sum(box_by_seg.values())
    val_segs, acc = set(), 0
    # add segments while they FIT under the cap -- a single huge segment must not
    # be allowed to overshoot and swallow half the boxes into val
    for s in segs:
        if acc >= 0.18 * total:
            break
        if acc + box_by_seg[s] <= 0.24 * total:
            val_segs.add(s)
            acc += box_by_seg[s]
    print(f"segments: {len(segs)}  val: {len(val_segs)}  val box share {acc/total:.3f}")

    for split in ("train", "val"):
        (DATASET / "images" / split).mkdir(parents=True, exist_ok=True)
        (DATASET / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    boxcounts = {"train": {b: 0 for b in BUCKETS}, "val": {b: 0 for b in BUCKETS}}
    for k, L in sorted(labels.items()):
        dive, fn = k
        s = f"{dive}/{segmap.get(k, 'segNA')}"
        split = "val" if (s in val_segs and k not in probe_only) else "train"
        stem = f"{dive}__{Path(fn).stem}"
        link = DATASET / "images" / split / f"{stem}.jpg"
        src = FRAME_ROOT / dive / "frames" / fn
        if not src.exists():
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src, link)
        (DATASET / "labels" / split / f"{stem}.txt").write_text(
            "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for c, a, b, w, h in L))
        counts[split] += 1
        for c, *_ in L:
            boxcounts[split][BUCKETS[c]] += 1

    yaml = (f"path: {DATASET}\ntrain: images/train\nval: images/val\n"
            f"names:\n" + "".join(f"  {i}: {b}\n" for i, b in enumerate(BUCKETS)))
    (DATASET / "epr_fauna.yaml").write_text(yaml)
    summary = {"frames": counts, "boxes": boxcounts, "n_miss_boxes": n_miss,
               "n_probe_only_frames_forced_to_train": len(probe_only),
               "n_true_negative_frames": n_tn, "val_segments": sorted(val_segs),
               "n_segments": len(segs), "verified_boxes_total": int(len(keep)),
               "frames_fully_adjudicated": len(full)}
    (DATASET / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2)[:1200])


def _grid_box(cell: str, size: str = "cell"):
    """Grid cell tag (e.g. 'C4') -> normalised xywh box, scaled by size hint."""
    cell = cell.strip().upper()
    if len(cell) < 2 or not cell[0].isalpha() or not cell[1:].isdigit():
        return None
    r = ord(cell[0]) - 65
    c = int(cell[1:]) - 1
    if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS):
        return None
    cw, ch = 1.0 / GRID_COLS, 1.0 / GRID_ROWS
    cx, cy = (c + 0.5) * cw, (r + 0.5) * ch
    f = {"tiny": 0.18, "small": 0.30, "cell": 0.60, "medium": 0.60, "large": 0.90}.get(size.strip().lower(), 0.45)
    return (cx, cy, min(cw * f, 1.0), min(ch * f, 1.0))


# --------------------------------------------------------------------------- #
# examples for the report
# --------------------------------------------------------------------------- #


def cmd_examples(args) -> None:
    det = load_detections().set_index("det_id")
    EXAMPLES.mkdir(parents=True, exist_ok=True)
    for tok in args.ids:
        did, _, tag = tok.partition(":")
        r = det.loc[int(did)]
        img = Image.open(r["path"]).convert("RGB")
        cell = _crop_one(img, r.x1, r.y1, r.x2, r.y2)
        cell = cell.resize((512, 512), Image.LANCZOS)
        name = f"{int(did):04d}_{r.bucket}_{r.cls.replace(' ', '_').replace('/', '-')}"
        if tag:
            name += f"_{tag}"
        cell.save(EXAMPLES / f"{name}.jpg", quality=92)
        print(EXAMPLES / f"{name}.jpg")



# --------------------------------------------------------------------------- #
# taxonomy v2 : morphology-triage buckets (2026-09 user change)
# --------------------------------------------------------------------------- #
# crab -> crustacean (pure rename), worm/fish unchanged, and the old catch-all
# "other" splits into a real "anemone" bucket plus an explicit "unknown"
# fallback.  The 5,670 Phase-1 verdicts transfer 1:1 because they were recorded
# against det_id + the ORIGINAL FathomNet cls, both of which are unchanged by
# the re-bucketing (verified: 0 mismatches on dive/fn/cls/conf).

BUCKETS_V2 = ("crustacean", "worm", "fish", "anemone", "unknown")
BUCKET_ID_V2 = {b: i for i, b in enumerate(BUCKETS_V2)}
_V1_TO_V2 = {"crab": "crustacean", "worm": "worm", "fish": "fish"}


def _bucket_of_v2(cls: str) -> str:
    """Current on-disk bucket rule, imported live so this never drifts."""
    from fathomnet_detect import bucket_of
    return bucket_of(cls)


def gt_bucket_v2(kind: str, to: str, cls: str) -> str | None:
    """Agent verdict (recorded in the v1 vocabulary) -> v2 ground-truth bucket.

    ``confirm``  the model's bucket was right, so the v2 answer is just the v2
                 rule applied to the model's own taxon label.
    ``relabel``  the agent overrode the bucket.  crab/worm/fish carry straight
                 over; a relabel to the old "other" means the taxon label is
                 itself wrong, so the taxon cannot be trusted to pick
                 anemone-vs-unknown -- it only counts as anemone if the v2 rule
                 independently says so, otherwise "unknown".
    ``reject`` / ``uncertain`` -> None (background / withheld).
    """
    if kind == "confirm":
        return _bucket_of_v2(cls)
    if kind == "relabel":
        if to in _V1_TO_V2:
            return _V1_TO_V2[to]
        if to == "other":
            b = _bucket_of_v2(cls)
            return "anemone" if b == "anemone" else "unknown"
        if to in BUCKETS_V2:
            return to
    return None


def _load_verdicts_v2() -> pd.DataFrame:
    v = pd.read_csv(OUT / "verdicts.csv")
    v["verdict"] = v["verdict"].astype(str).str.strip()
    v["kind"] = v["verdict"].str.split(":").str[0]
    v["to"] = v["verdict"].str.split(":").str[1].fillna("")
    v = v.drop_duplicates("det_id", keep="last")
    v["bucket_v2"] = [_bucket_of_v2(c) for c in v["cls"]]          # model's v2 bucket
    v["gt_v2"] = [gt_bucket_v2(k, t, c) for k, t, c in
                  zip(v["kind"], v["to"], v["cls"])]               # verified v2 bucket
    return v


def cmd_stats2(args) -> None:
    """Phase-1 precision recomputed under the 5-bucket morphology taxonomy."""
    v = _load_verdicts_v2()
    stats = {"taxonomy": list(BUCKETS_V2), "n_reviewed": int(len(v)),
             "note": "same 5,670 agent verdicts, re-expressed in the v2 buckets",
             "per_bucket": {}, "confusion_pred_to_true": {}}
    for b in BUCKETS_V2:
        s = v[v["bucket_v2"] == b]
        if not len(s):
            continue
        conf = int((s["kind"] == "confirm").sum())
        rel = int((s["kind"] == "relabel").sum())
        rej = int((s["kind"] == "reject").sum())
        unc = int((s["kind"] == "uncertain").sum())
        dec = conf + rel + rej
        lo, hi = _wilson(conf, dec)
        stats["per_bucket"][b] = {
            "n_reviewed": int(len(s)), "confirm": conf, "relabel": rel,
            "reject": rej, "uncertain": unc,
            "precision_bucket": round(conf / dec, 4) if dec else None,
            "precision_ci95": [round(lo, 4), round(hi, 4)],
            "is_animal_rate": round((conf + rel) / dec, 4) if dec else None,
            "relabel_targets_v2": s[s["kind"] == "relabel"]["gt_v2"].value_counts().to_dict()}
        row = {}
        for _, r in s.iterrows():
            t = ("uncertain" if r["kind"] == "uncertain"
                 else "none" if r["kind"] == "reject" else r["gt_v2"])
            row[t] = row.get(t, 0) + 1
        stats["confusion_pred_to_true"][b] = row
    dec = int((v["kind"] != "uncertain").sum())
    lo, hi = _wilson(int((v["kind"] == "confirm").sum()), dec)
    stats["overall"] = {
        "n_decided": dec,
        "precision_bucket": round(int((v["kind"] == "confirm").sum()) / dec, 4),
        "precision_ci95": [round(lo, 4), round(hi, 4)]}
    stats["verified_boxes_by_bucket"] = v["gt_v2"].value_counts().to_dict()
    (OUT / "review_stats_v2.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


def _miss_bucket_v2(bucket: str, note: str) -> str:
    """Probe-miss bucket -> v2.  Miss boxes have no FathomNet cls, so per the
    hand-off rule they fall to "unknown" unless the agent's own note names an
    anemone (the one case: probe 109 cell A3, 'orange_anemone')."""
    if bucket in _V1_TO_V2:
        return _V1_TO_V2[bucket]
    return "anemone" if "anemone" in str(note).lower() else "unknown"


def cmd_dataset2(args) -> None:
    """5-class YOLO dataset on the SAME segment split as v1 (pinned, not
    re-derived), so v1 and v2 are measured on identical held-out frames."""
    v1sum = json.loads((Path(args.pin_split)).read_text())
    val_segs = set(v1sum["val_segments"])
    det = load_detections()
    v = _load_verdicts_v2()
    d = det.merge(v[["det_id", "kind", "to", "gt_v2"]], on="det_id", how="left")
    keep = d[d["gt_v2"].notna() & d["gt_v2"].isin(BUCKETS_V2)].copy()

    labels: dict = {}
    for r in keep.itertuples():
        labels.setdefault((r.dive, r.fn), []).append(
            (BUCKET_ID_V2[r.gt_v2], *_norm(r.x1, r.y1, r.x2, r.y2)))

    probe_only: set = set()
    probe = pd.read_csv(OUT / "probe_frames.csv")
    pmap = {int(r.probe_id): (r.dive, r.frame_filename) for r in probe.itertuples()}
    n_miss = 0
    mp = OUT / "misses.csv"
    if mp.exists():
        miss = pd.read_csv(mp)
        for r in miss.itertuples():
            if int(r.probe_id) not in pmap:
                continue
            b = _miss_bucket_v2(str(r.bucket), getattr(r, "note", ""))
            box = _grid_box(str(r.cell), str(getattr(r, "size", "cell")))
            if box is None:
                continue
            dive, fn = pmap[int(r.probe_id)]
            labels.setdefault((dive, fn), []).append((BUCKET_ID_V2[b], *box))
            probe_only.add((dive, fn))
            n_miss += 1
    n_tn = 0
    missed_pids = set(pd.read_csv(mp)["probe_id"].astype(int)) if mp.exists() else set()
    for r in probe[probe["n_total"] == 0].itertuples():
        if int(r.probe_id) in missed_pids:
            continue
        labels.setdefault((r.dive, r.frame_filename), [])
        probe_only.add((r.dive, r.frame_filename))
        n_tn += 1

    ds = Path(args.out)
    segmap = load_frame_meta().set_index(["dive", "frame_filename"])["seg"].to_dict()
    for split in ("train", "val"):
        (ds / "images" / split).mkdir(parents=True, exist_ok=True)
        (ds / "labels" / split).mkdir(parents=True, exist_ok=True)
    counts = {"train": 0, "val": 0}
    boxc = {"train": {b: 0 for b in BUCKETS_V2}, "val": {b: 0 for b in BUCKETS_V2}}
    for k, L in sorted(labels.items()):
        dive, fn = k
        seg = f"{dive}/{segmap.get(k, 'segNA')}"
        split = "val" if (seg in val_segs and k not in probe_only) else "train"
        src = FRAME_ROOT / dive / "frames" / fn
        if not src.exists():
            continue
        stem = f"{dive}__{Path(fn).stem}"
        link = ds / "images" / split / f"{stem}.jpg"
        if link.exists() or link.is_symlink():
            link.unlink()
        os.symlink(src, link)
        (ds / "labels" / split / f"{stem}.txt").write_text(
            "".join(f"{c} {a:.6f} {b:.6f} {w:.6f} {h:.6f}\n" for c, a, b, w, h in L))
        counts[split] += 1
        for c, *_ in L:
            boxc[split][BUCKETS_V2[c]] += 1

    (ds / "epr_fauna_v2.yaml").write_text(
        f"path: {ds}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {b}\n" for i, b in enumerate(BUCKETS_V2)))
    summary = {"taxonomy": list(BUCKETS_V2), "frames": counts, "boxes": boxc,
               "n_miss_boxes": n_miss, "n_true_negative_frames": n_tn,
               "n_probe_only_frames_forced_to_train": len(probe_only),
               "val_segments": sorted(val_segs),
               "split_pinned_from": str(args.pin_split),
               "verified_boxes_total": int(len(keep))}
    (ds / "dataset_summary.json").write_text(json.dumps(summary, indent=2))
    (OUT / "dataset_summary_v2.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sheets"); s.add_argument("--workers", type=int, default=12); s.add_argument("--limit", type=int, default=0); s.set_defaults(fn=cmd_sheets)
    s = sub.add_parser("probe"); s.add_argument("--workers", type=int, default=12); s.add_argument("--n", type=int, default=120); s.set_defaults(fn=cmd_probe)
    s = sub.add_parser("stats2"); s.set_defaults(fn=cmd_stats2)
    s = sub.add_parser("dataset2")
    s.add_argument("--pin-split", default="/home/troyboland/models/finetune/dataset/dataset_summary.json")
    s.add_argument("--out", default="/home/troyboland/models/finetune/dataset_v2")
    s.set_defaults(fn=cmd_dataset2)
    s = sub.add_parser("misses"); s.add_argument("--log", required=True); s.set_defaults(fn=cmd_misses)
    s = sub.add_parser("compile"); s.add_argument("--log", required=True); s.set_defaults(fn=cmd_compile)
    s = sub.add_parser("stats"); s.set_defaults(fn=cmd_stats)
    s = sub.add_parser("dataset"); s.add_argument("--true-negatives", action="store_true", default=True); s.set_defaults(fn=cmd_dataset)
    s = sub.add_parser("examples"); s.add_argument("ids", nargs="+"); s.set_defaults(fn=cmd_examples)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
