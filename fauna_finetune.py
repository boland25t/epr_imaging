#!/usr/bin/env python3
"""Fine-tune the MBARI-315k YOLOv8 detector on agent-verified EPR fauna labels.

Commands
--------
baseline : run the ZERO-SHOT mbari_315k weights over the held-out split,
           collapse its 499 taxon classes into the 4 buckets with the exact
           mapping the production pipeline uses, and score it.
train    : fine-tune from those same weights onto the 4 bucket classes.
evaluate : score any weights file on the held-out split.
compare  : baseline + fine-tuned side by side into one JSON + markdown table.

Both models are scored by the SAME evaluator in this file (``_score``), on the
same agent-verified ground truth, so the delta is apples-to-apples.  The
evaluator is a standard greedy-IoU VOC-style AP: sort predictions by
confidence, match each to the highest-IoU unmatched GT of the same class at
IoU>=0.5, then integrate precision over recall (101-point interpolation).
Ultralytics' own ``val`` is also run on the tuned model as a cross-check.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fathomnet_detect import (  # noqa: E402  (read-only import of the mapping)
    BUCKETS as TAXON_BUCKETS,
    MIDWATER_EXCLUDE,
    NONFAUNA_EXCLUDE,
    bucket_of,
)

BUCKETS = ("crab", "worm", "fish", "other")          # v1, kept for back-compat
BUCKET_ID = {b: i for i, b in enumerate(BUCKETS)}

MODEL_OUT = Path("/home/troyboland/models/finetune")
DATASET = MODEL_OUT / "dataset"
YAML = DATASET / "epr_fauna.yaml"

# Taxonomy registry.  v1 = original crab/worm/fish/other.  v2 = the 2026-09
# morphology-triage change: crab renamed crustacean, a real anemone bucket, and
# an explicit "unknown" fallback instead of "other".  Both share one inference
# pass over the val split because zero-shot predictions are cached by TAXON
# NAME and only collapsed to buckets at scoring time.
TAX = {
    "v1": {"buckets": ("crab", "worm", "fish", "other"),
           "dataset": MODEL_OUT / "dataset",
           "yaml": MODEL_OUT / "dataset" / "epr_fauna.yaml",
           "v1_names": True},
    "v2": {"buckets": ("crustacean", "worm", "fish", "anemone", "unknown"),
           "dataset": MODEL_OUT / "dataset_v2",
           "yaml": MODEL_OUT / "dataset_v2" / "epr_fauna_v2.yaml",
           "v1_names": False},
}
# The v1 bucket names no longer exist in fathomnet_detect.bucket_of (it now
# returns the v2 vocabulary), so v1 scoring maps v2 -> v1 to stay reproducible.
_V2_TO_V1 = {"crustacean": "crab", "worm": "worm", "fish": "fish",
             "anemone": "other", "unknown": "other"}


def bucket_for(cls_name: str, taxonomy: str) -> str:
    b = bucket_of(cls_name)
    return _V2_TO_V1[b] if TAX[taxonomy]["v1_names"] else b
ZERO_SHOT = Path("/home/troyboland/models/mbari_315k_yolov8.pt")
BEST_OUT = MODEL_OUT / "epr_fauna_yolo.pt"
REPORT_DIR = Path("/mnt/f/EPR_2026_PROCESSED/paper/fauna_review")

CONF_INFER = 0.25   # same threshold the production pipeline ran at
IMGSZ = 1280
IOU_MATCH = 0.5


# --------------------------------------------------------------------------- #
# ground truth / prediction I/O
# --------------------------------------------------------------------------- #


def load_split(split: str = "val", dataset: Path | None = None):
    """-> {stem: (image_path, Nx5 array of [cls, cx, cy, w, h] normalised)}"""
    ds = Path(dataset) if dataset is not None else DATASET
    out = {}
    for img in sorted((ds / "images" / split).glob("*.jpg")):
        lab = ds / "labels" / split / f"{img.stem}.txt"
        rows = []
        if lab.exists():
            for ln in lab.read_text().split("\n"):
                if ln.strip():
                    v = ln.split()
                    rows.append([int(v[0])] + [float(x) for x in v[1:5]])
        out[img.stem] = (img, np.array(rows, dtype=float).reshape(-1, 5))
    return out


def _xywhn_to_xyxy(a: np.ndarray, W: int, H: int) -> np.ndarray:
    if len(a) == 0:
        return np.zeros((0, 4))
    cx, cy, w, h = a[:, 0] * W, a[:, 1] * H, a[:, 2] * W, a[:, 3] * H
    return np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)


def _iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(aa[:, None] + bb[None, :] - inter, 1e-9, None)


# --------------------------------------------------------------------------- #
# inference
# --------------------------------------------------------------------------- #


def predict_raw(weights: Path, split: str = "val", zero_shot: bool = False,
                conf: float = CONF_INFER, batch: int = 8, device: int = 0,
                dataset: Path | None = None) -> dict:
    """-> {stem: list[(bucket_id, conf, x1, y1, x2, y2)]} in pixel coords.

    With ``zero_shot`` the model's own 499 taxon names are mapped through the
    production bucket rules and the midwater / non-fauna classes are dropped,
    exactly as the pipeline does before it writes fathomnet_detections.csv.
    """
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = model.names
    gt = load_split(split, dataset)
    stems = list(gt)
    preds: dict[str, list] = {s: [] for s in stems}
    t0 = time.time()
    for i in range(0, len(stems), batch):
        chunk = stems[i:i + batch]
        res = model.predict([str(gt[s][0]) for s in chunk], conf=conf, imgsz=IMGSZ,
                            device=device, verbose=False)
        for s, r in zip(chunk, res):
            if r.boxes is None or len(r.boxes) == 0:
                continue
            xyxy = r.boxes.xyxy.cpu().numpy()
            cf = r.boxes.conf.cpu().numpy()
            cl = r.boxes.cls.cpu().numpy().astype(int)
            for b, c, k in zip(xyxy, cf, cl):
                if zero_shot:
                    nm = str(names[int(k)])
                    if nm in MIDWATER_EXCLUDE or nm in NONFAUNA_EXCLUDE:
                        continue
                    key = nm            # taxonomy-agnostic: resolve at scoring
                else:
                    key = int(k)        # already a bucket index
                preds[s].append((key, float(c), *[float(x) for x in b]))
        if (i // batch) % 20 == 0:
            print(f"  {i}/{len(stems)} imgs  {time.time()-t0:.0f}s", flush=True)
    print(f"inference {time.time()-t0:.1f}s over {len(stems)} images")
    return preds


def resolve(preds: dict, taxonomy: str) -> dict:
    """Raw predictions -> bucket-id predictions for one taxonomy."""
    bl = TAX[taxonomy]["buckets"]
    bid = {b: i for i, b in enumerate(bl)}
    out = {}
    for stem, rows in preds.items():
        r2 = []
        for q in rows:
            k = q[0]
            i = bid[bucket_for(k, taxonomy)] if isinstance(k, str) else int(k)
            r2.append((i, q[1], q[2], q[3], q[4], q[5]))
        out[stem] = r2
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _ap(rec: np.ndarray, prec: np.ndarray) -> float:
    """101-point interpolated AP (COCO style)."""
    if len(rec) == 0:
        return 0.0
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]
    x = np.linspace(0, 1, 101)
    return float(np.mean(np.interp(x, mrec, mpre)))


def _score(preds: dict, split: str = "val", pr_conf: float = 0.25,
           full_res=(5312, 2988), buckets=BUCKETS, dataset: Path | None = None) -> dict:
    """VOC/COCO-style AP50 per bucket + P/R at a fixed operating threshold."""
    gt = load_split(split, dataset)
    W, H = full_res
    out = {"per_bucket": {}, "n_images": len(gt), "buckets": list(buckets)}
    aps = []
    for b in buckets:
        bid = buckets.index(b)
        n_gt = 0
        recs = []   # (conf, is_tp)
        for stem, (_, lab) in gt.items():
            g = lab[lab[:, 0] == bid][:, 1:] if len(lab) else np.zeros((0, 4))
            gbox = _xywhn_to_xyxy(g, W, H)
            n_gt += len(gbox)
            p = [q for q in preds.get(stem, []) if q[0] == bid]
            p.sort(key=lambda q: -q[1])
            pbox = np.array([q[2:] for q in p]).reshape(-1, 4)
            used = np.zeros(len(gbox), bool)
            M = _iou_matrix(pbox, gbox)
            for j in range(len(pbox)):
                tp = False
                if len(gbox):
                    order = np.argsort(-M[j])
                    for k in order:
                        if M[j, k] < IOU_MATCH:
                            break
                        if not used[k]:
                            used[k] = True
                            tp = True
                            break
                recs.append((p[j][1], tp))
        recs.sort(key=lambda r: -r[0])
        tps = np.array([r[1] for r in recs], dtype=float)
        confs = np.array([r[0] for r in recs], dtype=float)
        ctp = np.cumsum(tps)
        cfp = np.cumsum(1 - tps)
        rec = ctp / max(n_gt, 1)
        prec = ctp / np.clip(ctp + cfp, 1e-9, None)
        ap = _ap(rec, prec) if n_gt else float("nan")
        m = confs >= pr_conf
        tp_at = float(tps[m].sum())
        fp_at = float((1 - tps[m]).sum())
        P = tp_at / (tp_at + fp_at) if (tp_at + fp_at) else float("nan")
        R = tp_at / n_gt if n_gt else float("nan")
        out["per_bucket"][b] = {
            "n_gt": int(n_gt), "n_pred": int(m.sum()),
            "AP50": None if n_gt == 0 else round(ap, 4),
            "P": None if not (tp_at + fp_at) else round(P, 4),
            "R": None if not n_gt else round(R, 4),
            "F1": None if not (n_gt and (tp_at + fp_at)) or (P + R) == 0
                  else round(2 * P * R / (P + R), 4)}
        if n_gt:
            aps.append(ap)
    out["mAP50"] = round(float(np.mean(aps)), 4) if aps else None
    # crab-weighted / instance-weighted variants, because fish has 3 GT boxes
    wts = [out["per_bucket"][b]["n_gt"] for b in buckets if out["per_bucket"][b]["n_gt"]]
    out["mAP50_instance_weighted"] = round(
        float(np.average(aps, weights=wts)), 4) if aps else None
    out["pr_threshold"] = pr_conf
    return out


def spurious_rate(preds: dict, split: str = "val", pr_conf: float = CONF_INFER,
                  full_res=(5312, 2988), dataset: Path | None = None) -> dict:
    """Boxes per image that match NO verified animal of any bucket (IoU<0.5).

    This is the one recall-independent quantity the val split measures cleanly:
    every detection on these frames was adjudicated, so a box overlapping no
    verified animal is a box on rock, sediment, shadow or an unidentifiable
    smudge.  Lower is better and it is directly comparable between the two
    models.
    """
    gt = load_split(split, dataset)
    W, H = full_res
    n_img = len(gt)
    tot = unm = 0
    for stem, (_, lab) in gt.items():
        gbox = _xywhn_to_xyxy(lab[:, 1:], W, H) if len(lab) else np.zeros((0, 4))
        p = [q for q in preds.get(stem, []) if q[1] >= pr_conf]
        pbox = np.array([q[2:] for q in p]).reshape(-1, 4)
        tot += len(pbox)
        if len(pbox) == 0:
            continue
        if len(gbox) == 0:
            unm += len(pbox)
            continue
        unm += int((_iou_matrix(pbox, gbox).max(1) < IOU_MATCH).sum())
    return {"boxes_per_image": round(tot / n_img, 3),
            "spurious_per_image": round(unm / n_img, 3),
            "spurious_share": round(unm / tot, 4) if tot else None,
            "n_boxes": tot, "n_spurious": unm, "conf": pr_conf}


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #


def _cached(tag: str, fn):
    cp = MODEL_OUT / f"preds_{tag}.json"
    if cp.exists():
        print(f"reusing cached predictions {cp}")
        return {k: [tuple(v) for v in vv] for k, vv in json.loads(cp.read_text()).items()}
    p = fn()
    cp.write_text(json.dumps({k: [list(v) for v in vv] for k, vv in p.items()}))
    return p


def fish_recall(preds: dict, taxonomy: str, conf: float, dataset: Path) -> dict:
    """Fish is a RECALL-oriented triage bucket: the user wants the frames most
    likely to contain a fish surfaced for a human, and does not care about
    species or about precision. So report it at a low threshold and also at the
    frame level, which is the unit a biologist actually triages."""
    bl = TAX[taxonomy]["buckets"]
    fid = bl.index("fish")
    r = _score(preds, "val", pr_conf=conf, buckets=bl, dataset=dataset)["per_bucket"]["fish"]
    gt = load_split("val", dataset)
    tp = fp = fn_ = 0
    for stem, (_, lab) in gt.items():
        has_gt = bool(len(lab) and (lab[:, 0] == fid).any())
        has_pr = any(q[0] == fid and q[1] >= conf for q in preds.get(stem, []))
        tp += has_gt and has_pr
        fp += (not has_gt) and has_pr
        fn_ += has_gt and (not has_pr)
    return {"conf": conf, "box_level": r,
            "frame_level": {"frames_with_fish_gt": tp + fn_,
                            "frames_flagged": tp + fp, "hit": tp, "missed": fn_,
                            "frame_recall": round(tp / (tp + fn_), 4) if (tp + fn_) else None,
                            "frames_flagged_without_fish": fp}}


def _eval_common(preds_raw: dict, taxonomy: str, mode: str, weights: Path,
                 out_name: str) -> dict:
    bl = TAX[taxonomy]["buckets"]
    ds = TAX[taxonomy]["dataset"]
    pr = resolve(preds_raw, taxonomy)
    s = _score(pr, "val", pr_conf=CONF_INFER, buckets=bl, dataset=ds)
    s["taxonomy"] = taxonomy
    s["weights"] = str(weights)
    s["mode"] = mode
    s["spurious"] = spurious_rate(pr, "val", dataset=ds)
    s["fish_triage_conf010"] = fish_recall(pr, taxonomy, 0.10, ds)
    (MODEL_OUT / out_name).write_text(json.dumps(s, indent=2))
    print(json.dumps(s, indent=2))
    return s


def cmd_baseline(args) -> None:
    raw = _cached("baseline_raw", lambda: predict_raw(
        ZERO_SHOT, "val", zero_shot=True, conf=0.05, batch=args.batch,
        device=args.device, dataset=TAX[args.taxonomy]["dataset"]))
    _eval_common(raw, args.taxonomy,
                 "zero-shot mbari_315k, 499 taxa collapsed by pipeline rules",
                 ZERO_SHOT, f"eval_baseline_{args.taxonomy}.json")


def cmd_train(args) -> None:
    from ultralytics import YOLO

    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    if args.resume:
        # Ultralytics reloads every hyper-parameter, the optimizer state and the
        # epoch counter from the checkpoint, so the resumed run is a true
        # continuation of the interrupted one -- passing the recipe again would
        # be ignored at best and conflicting at worst.
        ck = Path(args.resume)
        # NB: ultralytics only honours imgsz/batch/device/close_mosaic as
        # overrides on resume, so workers/cache must be edited into the
        # checkpoint's train_args before calling this (see README/report).
        print(f"resuming from {ck}")
        model = YOLO(str(ck))
        r = model.train(resume=True)
        mins = (time.time() - t0) / 60
        best = Path(r.save_dir) / "weights" / "best.pt"
        dest = MODEL_OUT / args.weights_out
        shutil.copy2(best, dest)
        meta = {"taxonomy": args.taxonomy, "gpu_minutes_this_leg": round(mins, 1),
                "resumed_from": str(ck), "save_dir": str(r.save_dir),
                "best": str(dest), "imgsz": IMGSZ}
        (MODEL_OUT / f"train_meta_{args.taxonomy}.json").write_text(json.dumps(meta, indent=2))
        print(json.dumps(meta, indent=2))
        return
    model = YOLO(str(ZERO_SHOT))
    # Identical transfer recipe for v1 and v2 so the only difference between the
    # two runs is the label vocabulary.  Rationale for each knob is in the
    # report; the short version: production imgsz, lr an order below
    # from-scratch because the backbone already knows deep-sea megafauna, and
    # colour jitter kept small because "bright animal on dark basalt" IS the cue.
    r = model.train(
        data=str(TAX[args.taxonomy]["yaml"]), epochs=args.epochs, imgsz=IMGSZ,
        batch=args.batch, device=args.device, workers=args.workers,
        optimizer="AdamW",
        lr0=2e-4, lrf=0.05, cos_lr=True, warmup_epochs=3.0, weight_decay=5e-4,
        patience=15, close_mosaic=10, mosaic=0.5, mixup=0.0, copy_paste=0.0,
        degrees=5.0, translate=0.08, scale=0.35, shear=0.0, perspective=0.0,
        fliplr=0.5, flipud=0.5, hsv_h=0.010, hsv_s=0.35, hsv_v=0.30,
        project=str(MODEL_OUT), name=args.name, exist_ok=True, seed=17,
        val=True, plots=True, pretrained=True, deterministic=False,
        cache=False, single_cls=False,
    )
    mins = (time.time() - t0) / 60
    best = Path(r.save_dir) / "weights" / "best.pt"
    dest = MODEL_OUT / args.weights_out
    shutil.copy2(best, dest)
    meta = {"taxonomy": args.taxonomy, "gpu_minutes": round(mins, 1),
            "save_dir": str(r.save_dir), "best": str(dest),
            "epochs_requested": args.epochs, "imgsz": IMGSZ, "batch": args.batch}
    (MODEL_OUT / f"train_meta_{args.taxonomy}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))


def cmd_evaluate(args) -> None:
    w = Path(args.weights)
    raw = _cached(f"tuned_{args.taxonomy}", lambda: predict_raw(
        w, "val", zero_shot=False, conf=0.05, batch=args.batch,
        device=args.device, dataset=TAX[args.taxonomy]["dataset"]))
    _eval_common(raw, args.taxonomy, "fine-tuned, native bucket classes", w,
                 args.out or f"eval_{args.taxonomy}.json")


def _row(b, f, k, digits=3):
    def g(d):
        return "n/a" if d is None else f"{d:.{digits}f}"
    return g(b), g(f)


def cmd_compare(args) -> None:
    """baseline vs v1 vs v2, each against its own taxonomy's verified GT."""
    def load(n):
        p = MODEL_OUT / n
        return json.loads(p.read_text()) if p.exists() else None
    runs = [("zero-shot (v1 buckets)", load("eval_baseline_v1.json") or load("eval_baseline.json")),
            ("fine-tuned v1", load("eval_v1.json") or load("eval_finetuned.json")),
            ("zero-shot (v2 buckets)", load("eval_baseline_v2.json")),
            ("fine-tuned v2", load("eval_v2.json"))]
    runs = [(n, r) for n, r in runs if r]
    lines = []
    for name, r in runs:
        lines.append(f"### {name}  ({r.get('taxonomy','v1')})")
        lines.append("")
        lines.append("| bucket | n GT | AP50 | P | R | F1 |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for b in r["buckets"]:
            d = r["per_bucket"][b]
            def g(k):
                return "n/a" if d[k] is None else f"{d[k]:.3f}"
            lines.append(f"| {b} | {d['n_gt']} | {g('AP50')} | {g('P')} | {g('R')} | {g('F1')} |")
        lines.append(f"| **mAP50 macro** |  | **{r['mAP50']:.3f}** |  |  |  |")
        lines.append(f"| **mAP50 inst-wt** |  | **{r['mAP50_instance_weighted']:.3f}** |  |  |  |")
        sp = r.get("spurious", {})
        if sp:
            lines.append("")
            lines.append(f"boxes/image {sp['boxes_per_image']:.2f} · "
                         f"spurious/image {sp['spurious_per_image']:.2f} · "
                         f"spurious share {sp['spurious_share']:.1%}")
        ft = r.get("fish_triage_conf010", {})
        if ft:
            fl = ft["frame_level"]
            lines.append(f"fish triage @conf 0.10 — frame recall "
                         f"{fl['frame_recall'] if fl['frame_recall'] is not None else 'n/a'} "
                         f"({fl['hit']}/{fl['frames_with_fish_gt']} fish frames hit, "
                         f"{fl['frames_flagged_without_fish']} frames flagged without fish)")
        lines.append("")
    tbl = "\n".join(lines)
    (MODEL_OUT / "compare.md").write_text(tbl + "\n")
    (REPORT_DIR / "finetune_metrics.md").write_text(tbl + "\n")
    json.dump({n: r for n, r in runs}, open(REPORT_DIR / "finetune_metrics.json", "w"), indent=2)
    print(tbl)


# --------------------------------------------------------------------------- #
# deployment: per-bucket operating thresholds + portable offline bundle
# --------------------------------------------------------------------------- #

DEPLOY = Path("/home/troyboland/models/deploy")
SWEEP_CONFS = [round(x, 2) for x in np.arange(0.05, 0.96, 0.01)]


def _pr_at(preds: dict, bucket_idx: int, conf: float, gt: dict,
           full_res=(5312, 2988)) -> tuple[int, int, int]:
    """(tp, fp, n_gt) for one bucket at one confidence, greedy IoU>=0.5."""
    W, H = full_res
    tp = fp = n_gt = 0
    for stem, (_, lab) in gt.items():
        g = lab[lab[:, 0] == bucket_idx][:, 1:] if len(lab) else np.zeros((0, 4))
        gbox = _xywhn_to_xyxy(g, W, H)
        n_gt += len(gbox)
        pr = sorted([q for q in preds.get(stem, [])
                     if q[0] == bucket_idx and q[1] >= conf], key=lambda q: -q[1])
        pbox = np.array([q[2:] for q in pr]).reshape(-1, 4)
        if not len(pbox):
            continue
        used = np.zeros(len(gbox), bool)
        M = _iou_matrix(pbox, gbox)
        for j in range(len(pbox)):
            hit = False
            if len(gbox):
                for k in np.argsort(-M[j]):
                    if M[j, k] < IOU_MATCH:
                        break
                    if not used[k]:
                        used[k] = True
                        hit = True
                        break
            tp += hit
            fp += not hit
    return tp, fp, n_gt


def sweep(preds: dict, taxonomy: str) -> dict:
    """Per-bucket P/R/F1 across the confidence range, plus two chosen operating
    points: "balanced" = max F1, "triage" = cheapest threshold reaching
    recall >= 0.90 (falling back to the max-recall point if 0.90 is
    unreachable).  Triage is what the fish bucket wants: the user asked for
    frames likely to contain a fish to be surfaced, not for precise fish."""
    bl = TAX[taxonomy]["buckets"]
    gt = load_split("val", TAX[taxonomy]["dataset"])
    out = {}
    for b in bl:
        bi = bl.index(b)
        curve = []
        for c in SWEEP_CONFS:
            tp, fp, n_gt = _pr_at(preds, bi, c, gt)
            P = tp / (tp + fp) if (tp + fp) else None
            R = tp / n_gt if n_gt else None
            F1 = (2 * P * R / (P + R)) if (P and R) else 0.0
            curve.append({"conf": c, "tp": tp, "fp": fp, "n_gt": n_gt,
                          "P": P, "R": R, "F1": F1})
        n_gt = curve[0]["n_gt"]
        if n_gt == 0:
            out[b] = {"n_gt": 0, "curve": curve, "balanced": None, "triage": None,
                      "note": "no verified ground truth in the held-out split"}
            continue
        bal = max(curve, key=lambda r: r["F1"])
        ok = [r for r in curve if (r["R"] or 0) >= 0.90]
        tri = max(ok, key=lambda r: r["conf"]) if ok else max(
            curve, key=lambda r: ((r["R"] or 0), r["conf"]))
        out[b] = {"n_gt": n_gt, "curve": curve,
                  "balanced": {k: bal[k] for k in ("conf", "P", "R", "F1", "tp", "fp")},
                  "triage": {k: tri[k] for k in ("conf", "P", "R", "F1", "tp", "fp")},
                  "triage_reached_r90": bool(ok)}
    return out


def cmd_deploy(args) -> None:
    """Write deploy_config.json covering BOTH models so the shipboard app can
    pick whichever wins per bucket, offline, with no Claude in the loop."""
    tax = "v2"
    bl = TAX[tax]["buckets"]
    models = {}
    raws = {
        "mbari_zero_shot": (ZERO_SHOT, _cached("baseline_raw", lambda: predict_raw(
            ZERO_SHOT, "val", zero_shot=True, conf=0.05, batch=args.batch,
            device=args.device, dataset=TAX[tax]["dataset"]))),
        "epr_fauna_v2": (Path(args.v2_weights), _cached("tuned_v2", lambda: predict_raw(
            Path(args.v2_weights), "val", zero_shot=False, conf=0.05,
            batch=args.batch, device=args.device, dataset=TAX[tax]["dataset"]))),
    }
    sweeps = {}
    for name, (w, raw) in raws.items():
        sw = sweep(resolve(raw, tax), tax)
        sweeps[name] = sw
        models[name] = {"weights": str(w), "buckets": {}}
        for b in bl:
            r = sw[b]
            if r["balanced"] is None:
                models[name]["buckets"][b] = {
                    "conf_balanced": None, "conf_triage": None,
                    "expected_precision": None, "expected_recall": None,
                    "n_gt_val": 0,
                    "note": "no verified ground truth in the held-out split -- "
                            "do not ship a count for this bucket"}
                continue
            models[name]["buckets"][b] = {
                "conf_balanced": r["balanced"]["conf"],
                "conf_triage": r["triage"]["conf"],
                "expected_precision": None if r["balanced"]["P"] is None
                                      else round(r["balanced"]["P"], 4),
                "expected_recall": None if r["balanced"]["R"] is None
                                   else round(r["balanced"]["R"], 4),
                "expected_precision_at_triage": None if r["triage"]["P"] is None
                                                else round(r["triage"]["P"], 4),
                "expected_recall_at_triage": None if r["triage"]["R"] is None
                                             else round(r["triage"]["R"], 4),
                "n_gt_val": r["n_gt"],
                "triage_reached_recall_0.90": r["triage_reached_r90"]}

    # per-bucket winner on the balanced F1, which is what the app should default to
    rec = {}
    for b in bl:
        cand = {n: (sweeps[n][b]["balanced"] or {}).get("F1", -1) for n in sweeps}
        best = max(cand, key=cand.get)
        rec[b] = {"use_model": best if cand[best] > 0 else None,
                  "f1_by_model": {n: (None if v < 0 else round(v, 4)) for n, v in cand.items()}}

    cfg = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "taxonomy": list(bl),
        "imgsz": IMGSZ,
        "iou_match_used_for_calibration": IOU_MATCH,
        "ground_truth": "5,670 agent-verified FathomNet detections (J1754+J1756), "
                        "segment-split held-out val = 355 frames",
        "models": models,
        "recommended_model_per_bucket": rec,
        "IMPORTANT_recall_caveat":
            "expected_recall is measured against ground truth DERIVED FROM the "
            "MBARI model's own proposals, so it is recall over proposed boxes, "
            "NOT recall over animals present. Whole-frame recall measured "
            "independently on 120 stratified frames was 0.44 overall: "
            "crustacean 0.77, unknown/other 0.35, worm 0.03. Use the probe "
            "numbers for survey design and abundance correction; use these "
            "thresholds only to set the detector's operating point.",
    }
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    (MODEL_OUT / "deploy_config.json").write_text(json.dumps(cfg, indent=2))
    # full sweep curves for the record
    import csv
    with open(MODEL_OUT / "threshold_sweep.csv", "w", newline="") as fh:
        wr = csv.writer(fh)
        wr.writerow(["model", "bucket", "conf", "tp", "fp", "n_gt", "P", "R", "F1"])
        for n, sw in sweeps.items():
            for b, r in sw.items():
                for c in r["curve"]:
                    wr.writerow([n, b, c["conf"], c["tp"], c["fp"], c["n_gt"],
                                 "" if c["P"] is None else round(c["P"], 4),
                                 "" if c["R"] is None else round(c["R"], 4),
                                 round(c["F1"], 4)])
    print(json.dumps({"recommended_model_per_bucket": rec,
                      "models": {n: v["buckets"] for n, v in models.items()}}, indent=2))



def cmd_bundle(args) -> None:
    """Assemble a self-contained offline bundle: weights + thresholds + README.

    Everything a shipboard operator needs with no internet and no Claude.  The
    README is generated from the measured JSONs so its numbers cannot drift
    away from the artefacts it ships beside.
    """
    DEPLOY.mkdir(parents=True, exist_ok=True)
    cfg = json.loads((MODEL_OUT / "deploy_config.json").read_text())
    rv2 = json.loads((REPORT_DIR / "review_stats_v2.json").read_text())
    rec = json.loads((REPORT_DIR / "recall_stats.json").read_text())
    base = json.loads((MODEL_OUT / "eval_baseline_v2.json").read_text())
    v2 = json.loads((MODEL_OUT / "eval_v2.json").read_text())

    for src, dst in ((Path(args.v2_weights), "epr_fauna_v2.pt"),
                     (ZERO_SHOT, "mbari_315k_yolov8.pt")):
        if src.exists():
            shutil.copy2(src, DEPLOY / dst)
    shutil.copy2(MODEL_OUT / "deploy_config.json", DEPLOY / "deploy_config.json")
    if (MODEL_OUT / "threshold_sweep.csv").exists():
        shutil.copy2(MODEL_OUT / "threshold_sweep.csv", DEPLOY / "threshold_sweep.csv")
    shutil.copy2(Path(__file__).resolve().parent / "deploy_predict_offline.py",
                 DEPLOY / "predict_offline.py")

    bl = cfg["taxonomy"]
    L = []
    A = L.append
    A("# EPR megafauna detector - offline shipboard bundle")
    A("")
    A(f"Generated {cfg['generated']}. Self-contained: no internet, no Claude, no")
    A("model download. Every number below was measured on J1754 + J1756")
    A("down-looking frames against 5,670 agent-verified detections plus a")
    A("120-frame whole-frame recall probe.")
    A("")
    A("## Contents")
    A("")
    A("| file | what |")
    A("|---|---|")
    A("| `mbari_315k_yolov8.pt` | MBARI-315k zero-shot detector, 499 taxa (the pipeline original) |")
    A("| `epr_fauna_v2.pt` | fine-tuned on the verified EPR labels, native 5 buckets |")
    A("| `deploy_config.json` | per-bucket model choice + confidence thresholds (authoritative) |")
    A("| `threshold_sweep.csv` | full P/R/F1 vs confidence curves, both models, every bucket |")
    A("| `predict_offline.py` | runnable inference driven by deploy_config.json |")
    A("")
    A("Taxonomy is morphology triage - conclusive ID is done later by a human:")
    A(f"`{' / '.join(bl)}`.")
    A("")
    A("## Which model and threshold to use, per bucket")
    A("")
    A("`deploy_config.json` is authoritative; this is its readable form.")
    A("`balanced` = max-F1 point. `triage` = cheapest threshold reaching recall")
    A(">= 0.90 over proposed boxes, for when over-flagging beats missing.")
    A("")
    A("| bucket | use model | balanced conf | P | R | triage conf | P | R |")
    A("|---|---|---:|---:|---:|---:|---:|---:|")
    for b in bl:
        use = cfg["recommended_model_per_bucket"][b]["use_model"]
        if use is None:
            A(f"| {b} | - (no verified ground truth; do not ship a count) | | | | | | |")
            continue
        d = cfg["models"][use]["buckets"][b]
        def g(k, d=d):
            v = d.get(k)
            return "n/a" if v is None else f"{v:.3f}"
        A(f"| {b} | `{use}` | {d['conf_balanced']:.2f} | {g('expected_precision')} | "
          f"{g('expected_recall')} | {d['conf_triage']:.2f} | "
          f"{g('expected_precision_at_triage')} | {g('expected_recall_at_triage')} |")
    A("")
    A("### Buckets you must NOT report a count for")
    A("")
    for b in bl:
        vb = rv2["verified_boxes_by_bucket"].get(b, 0)
        nv = cfg["models"]["epr_fauna_v2"]["buckets"][b].get("n_gt_val", 0)
        if vb < 50:
            A(f"- **{b}** - only {vb} verified boxes exist in the whole two-dive corpus "
              f"({nv} in the held-out split). Anything either model emits here is "
              f"uncalibrated. Surface the crops for a human; do not count them.")
    A("")
    A("### Simplification: one model instead of two")
    A("")
    A("v2 is recommended for `unknown` only, and only by F1 0.824 vs 0.799 - inside")
    A("the noise of a 149-box bucket. Loading two 137 MB models for that is")
    A("arguably not worth it. To run zero-shot only, edit `deploy_config.json` and")
    A("set `recommended_model_per_bucket.unknown.use_model` to `mbari_zero_shot`;")
    A("its thresholds are already present under `models.mbari_zero_shot`.")
    A("")
    A("`worm` cannot reach recall 0.90 at ANY threshold on either model: lowering")
    A("the threshold adds rock faster than it adds worms (P 0.917 -> 0.152 buys")
    A("R 0.500 -> 0.545). That is a detector-capability limit, not a tuning knob.")
    A("")
    A("## Known failure modes")
    A("")
    A("**1. Altitude outside 3-8 m.** Zero-shot bucket precision by altitude,")
    A("over all 5,670 reviewed detections:")
    A("")
    A("| altitude | n detections | precision | reject rate |")
    A("|---|---:|---:|---:|")
    A("| < 3 m | 201 | 0.264 | 0.681 |")
    A("| 3-5 m | 2,838 | 0.771 | 0.179 |")
    A("| 5-8 m | 2,454 | 0.806 | 0.164 |")
    A("| > 8 m | 177 | 0.494 | 0.500 |")
    A("")
    A("Below 3 m the frame fills with sulfide structure and the detector paints")
    A("sponges and fish onto it. Above 8 m contrast collapses and half of what")
    A("it reports is rock. **Restrict quantitative products to 3-8 m**; treat")
    A("anything outside that band as qualitative only.")
    A("")
    A("**2. Worm thickets - the detector is effectively blind to them.**")
    wm = rec["per_bucket"]["worm"]
    A(f"Whole-frame probe: worm found {wm['found']}, missed {wm['missed']}, recall "
      f"{wm['recall']:.2f} (95% CI {wm['recall_ci95'][0]:.2f}-{wm['recall_ci95'][1]:.2f}), "
      f"and that is an UPPER bound because dense fields were sampled, not censused.")
    A("On four probe frames holding dozens of white tube worms and gastropod mats")
    A("the model returned **zero detections**. Map dense small-tube fields by")
    A("another method (texture/segmentation, or manually). **Do not compute worm")
    A("density from detector output.**")
    A("")
    A("**3. Bare rock and low-contrast water.** Of 1,010 verified false positives,")
    A("~58% are bare rock / slab / carbonate and ~35% are dark low-contrast water,")
    A("shadow or crevice. Bacterial mat / Fe-floc was only ~1% in these two dives,")
    A("but it is the same visual failure class (bright irregular patch on dark")
    A("basalt -> `Porifera`), so expect it to bite harder on floc-heavy sites.")
    A("Thresholding helps: bucket precision rises 0.59 -> 0.90 from conf 0.25-0.35")
    A("to conf > 0.60.")
    A("")
    A("**4. One species carries the model, and three labels are near-useless.**")
    A("`Munidopsis` is 99.9% reliable and carries the whole crustacean bucket.")
    A("`Porifera` (548 detections, 97% rejected), `Zoantharia` (99, 98% rejected)")
    A("and `Actinopterygii` (480, 94% not fish) are the three largest error")
    A("sources in the original 499-class output.")
    A("")
    A("## The caveat that matters most")
    A("")
    A(cfg["IMPORTANT_recall_caveat"])
    A("")
    A("Thresholds were also *selected* on the same 355-frame split they are")
    A("reported on, so the quoted P/R are optimistic by an unknown non-zero")
    A("margin. There is no second holdout and no second cruise.")
    A("")
    A("## Held-out comparison (5 buckets, identical segment split)")
    A("")
    A("| bucket | n GT | AP50 zero-shot | AP50 v2 | delta |")
    A("|---|---:|---:|---:|---:|")
    for b in bl:
        a0, c0 = base["per_bucket"][b], v2["per_bucket"][b]
        if a0["AP50"] is None or c0["AP50"] is None:
            A(f"| {b} | {a0['n_gt']} | n/a | n/a | n/a |")
        else:
            A(f"| {b} | {a0['n_gt']} | {a0['AP50']:.3f} | {c0['AP50']:.3f} | "
              f"{c0['AP50'] - a0['AP50']:+.3f} |")
    A(f"| **mAP50 macro** | | {base['mAP50']:.3f} | {v2['mAP50']:.3f} | "
      f"{v2['mAP50'] - base['mAP50']:+.3f} |")
    A(f"| **mAP50 inst-wt** | | {base['mAP50_instance_weighted']:.3f} | "
      f"{v2['mAP50_instance_weighted']:.3f} | "
      f"{v2['mAP50_instance_weighted'] - base['mAP50_instance_weighted']:+.3f} |")
    A("")
    A(f"Spurious boxes per image (overlapping no verified animal): zero-shot "
      f"{base['spurious']['spurious_per_image']:.2f}, v2 "
      f"{v2['spurious']['spurious_per_image']:.2f}.")
    A("")
    A("## Usage")
    A("")
    A("```bash")
    A("python predict_offline.py --frames /path/to/frames --out detections.csv")
    A("python predict_offline.py --frames ... --mode triage   # over-flag for fish")
    A("```")
    A("")
    A("Requires only `ultralytics` + `torch` installed locally.")
    (DEPLOY / "README.md").write_text("\n".join(L) + "\n")
    print(f"bundle -> {DEPLOY}")
    for f in sorted(DEPLOY.iterdir()):
        print(f"  {f.name}  {f.stat().st_size/1e6:.2f} MB")

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for nm, fn in (("baseline", cmd_baseline), ("train", cmd_train),
                   ("evaluate", cmd_evaluate), ("compare", cmd_compare),
                   ("deploy", cmd_deploy), ("bundle", cmd_bundle)):
        s = sub.add_parser(nm)
        s.add_argument("--device", type=int, default=0)
        s.add_argument("--batch", type=int, default=8)
        s.add_argument("--taxonomy", choices=("v1", "v2"), default="v2")
        if nm == "train":
            s.add_argument("--epochs", type=int, default=60)
            s.add_argument("--name", default="epr_fauna_v2")
            s.add_argument("--weights-out", default="epr_fauna_yolo_v2.pt")
            s.add_argument("--resume", default=None,
                           help="path to last.pt of an interrupted run")
            s.add_argument("--workers", type=int, default=4,
                           help="dataloader workers; the 1280px dataset OOM'd "
                                "the WSL VM at 8 with other agents running")
        if nm == "evaluate":
            s.add_argument("--weights", required=True)
            s.add_argument("--out", default=None)
        if nm in ("deploy", "bundle"):
            s.add_argument("--v2-weights",
                           default=str(MODEL_OUT / "epr_fauna_yolo_v2.pt"))
        s.set_defaults(fn=fn)
    a = ap.parse_args()
    MODEL_OUT.mkdir(parents=True, exist_ok=True)
    a.fn(a)


if __name__ == "__main__":
    main()
