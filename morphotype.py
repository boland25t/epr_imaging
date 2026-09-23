#!/usr/bin/env python3
"""Sub-bucket crustacean MORPHOTYPE classification for ROV seafloor frames.

WHY THIS EXISTS
---------------
The FathomNet/MBARI box detector cannot resolve crustacean taxonomy below the
bucket level.  It labels 3,135 of the 3,341 reviewer-confirmed crustacean boxes
on J1754+J1756 as ``Munidopsis`` (93.8%), which reads as a "97.5% Munidopsis"
community -- but that number is the detector's label prior, not an observation.
Within-bucket species labels from the detector are NOT evidence: the same
animal is scattered across ``Munidopsis`` / ``Galatheoidea`` / ``Pandalus
amplus`` / ``Sternostylus`` / ``Decapoda``, and 157 reviewer-confirmed
crustaceans carry labels like ``Asteroidea`` and ``Actinopterygii``.

This module replaces the detector's sub-bucket taxonomy with a morphotype call
made from the pixels, distilled from hand labels on the confirmed crops:

    squat_lobster    galatheid habitus: broad flattened carapace, long forward
                     chelipeds, 3-4 pairs of splayed pereiopods
    not_crustacean   no arthropod habitus in the box (ophiuroid arms, fish,
                     holothurian, shell, or no animal at all)
    indeterminate    an animal is present but the crop cannot be morphotyped
                     (motion blur, too dark, too small, cropped) -- a QC flag,
                     not a morphotype

``caridean_shrimp`` is a documented morphotype but is **not** a model class.
Only 2 carideans exist in the whole confirmed set (n=2 is untrainable); see
CARIDEAN_DET_IDS and ``atypicality`` below.

Runtime is fully offline: pure torch + torchvision + Pillow, architectures built
with ``weights=None`` and local state_dicts loaded from disk.  No network, no
ultralytics, no Qt, no Claude.  CPU is supported.

USAGE
-----
    # one crop image
    python morphotype.py predict CROP.jpg

    # boxes on a full frame (x1,y1,x2,y2 in frame pixels)
    python morphotype.py boxes FRAME.jpg --box 2863,595,3230,1036 --box ...

    # a detection CSV (needs det_id/path/x1/y1/x2/y2) -> morphotype per row
    python morphotype.py batch --dets detections.csv --out morphotype.csv

    # model card + operating points
    python morphotype.py info

    # library
    from morphotype import predict_crop, predict_boxes
    predict_crop("crop.jpg")["morphotype"]
    predict_boxes("frame.jpg", [(x1, y1, x2, y2), ...])

Weights resolve in this order: ``weights=`` / ``--weights``,
``$MORPHOTYPE_WEIGHTS``, then the first hit in DEFAULT_WEIGHT_PATHS.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

__all__ = [
    "CLASSES", "SQUAT_LOBSTER", "NOT_CRUSTACEAN", "INDETERMINATE",
    "CARIDEAN_SHRIMP", "CARIDEAN_DET_IDS", "IMG_SIZE", "PAD_FACTOR",
    "BALANCED_THRESHOLD", "HIGH_PRECISION_THRESHOLD", "ATYPICALITY_P98",
    "MorphotypeModel", "predict_crop", "predict_boxes", "crop_box",
    "load_model", "resolve_weights",
]

SQUAT_LOBSTER = "squat_lobster"
NOT_CRUSTACEAN = "not_crustacean"
INDETERMINATE = "indeterminate"
CARIDEAN_SHRIMP = "caridean_shrimp"          # documented, never predicted
CLASSES = [SQUAT_LOBSTER, NOT_CRUSTACEAN, INDETERMINATE]

IMG_SIZE = 224
PAD_FACTOR = 1.6            # crop side = 1.6 * max(box_w, box_h); must match training
MIN_CROP_PX = 96
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Operating points, measured on the held-out segments (see model card in info).
BALANCED_THRESHOLD = 0.50           # P(squat_lobster); P=0.931 R=0.946
HIGH_PRECISION_THRESHOLD = 0.80     # P=0.953 R=0.871
ATYPICALITY_P98 = 0.2522            # 98th pct; flags 2% of crops for human review

# The only two carideans found in 3,341 confirmed crustacean boxes (J1754).
CARIDEAN_DET_IDS = (807, 2418)

DEFAULT_WEIGHT_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "weights", "morphotype_resnet18.pt"),
    os.path.expanduser("~/models/deploy/morphotype/morphotype_resnet18.pt"),
    "/mnt/f/EPR_2026_PROCESSED/paper/morphotype/morphotype_resnet18.pt",
)
DEFAULT_ATYP_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "weights", "morphotype_atypicality.pt"),
    os.path.expanduser("~/models/deploy/morphotype/morphotype_atypicality.pt"),
    "/mnt/f/EPR_2026_PROCESSED/paper/morphotype/morphotype_atypicality.pt",
)

_MODEL_CACHE: dict = {}


# --------------------------------------------------------------------------
# weights / model
# --------------------------------------------------------------------------
def resolve_weights(weights: str | None = None,
                    env: str = "MORPHOTYPE_WEIGHTS",
                    defaults: tuple = DEFAULT_WEIGHT_PATHS) -> str:
    """Return a readable weights path or raise FileNotFoundError."""
    cands = []
    if weights:
        cands.append(weights)
    val = os.environ.get(env)
    if val:
        cands.append(val)
    cands.extend(defaults)
    for p in cands:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "morphotype weights not found. Tried:\n  "
        + "\n  ".join(str(c) for c in cands)
        + f"\nPass --weights / weights= or set ${env}."
    )


def crop_box(image, x1: float, y1: float, x2: float, y2: float,
             pad: float = PAD_FACTOR):
    """Cut the training-equivalent square context crop around a box.

    ``image`` is a PIL image; one image is held in memory at a time.
    """
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    side = max(max(x2 - x1, y2 - y1) * pad, MIN_CROP_PX)
    l, t = int(cx - side / 2), int(cy - side / 2)
    r, b = int(cx + side / 2), int(cy + side / 2)
    l = max(0, min(l, image.width - 8))
    t = max(0, min(t, image.height - 8))
    r = min(image.width, max(r, l + 16))
    b = min(image.height, max(b, t + 16))
    return image.crop((l, t, r, b))


class MorphotypeModel:
    """Crop classifier + atypicality scorer. Processes one image at a time."""

    def __init__(self, weights: str | None = None,
                 atypicality_weights: str | None = None,
                 device: str | None = None):
        import torch
        from torchvision import transforms as T
        from torchvision.models import resnet18

        self._torch = torch
        self.weights_path = resolve_weights(weights)
        ck = torch.load(self.weights_path, map_location="cpu", weights_only=False)

        self.classes = list(ck.get("classes", CLASSES))
        self.img_size = int(ck.get("img_size", IMG_SIZE))
        self.mean = tuple(ck.get("mean", IMAGENET_MEAN))
        self.std = tuple(ck.get("std", IMAGENET_STD))
        try:
            self.squat_index = self.classes.index(SQUAT_LOBSTER)
        except ValueError:
            self.squat_index = 0

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        net = resnet18(weights=None)
        net.fc = torch.nn.Linear(512, len(self.classes))
        net.load_state_dict(ck["state_dict"])
        self.net = net.eval().to(self.device)

        self.tf = T.Compose([
            T.Resize(self.img_size), T.CenterCrop(self.img_size),
            T.ToTensor(), T.Normalize(self.mean, self.std)])

        # optional atypicality head (frozen ImageNet features + centroid)
        self.atyp = None
        self.atyp_threshold = ATYPICALITY_P98
        try:
            ap = resolve_weights(atypicality_weights, "MORPHOTYPE_ATYPICALITY",
                                 DEFAULT_ATYP_PATHS)
            ak = torch.load(ap, map_location="cpu", weights_only=False)
            b = resnet18(weights=None)
            b.fc = torch.nn.Identity()
            b.load_state_dict(ak["state_dict"])
            self.atyp = b.eval().to(self.device)
            self.centroid = ak["centroid"].to(self.device)
            self.atyp_threshold = float(ak.get("threshold_p98", ATYPICALITY_P98))
            self.atypicality_path = ap
        except (FileNotFoundError, KeyError, RuntimeError):
            self.atypicality_path = None

    # ----------------------------------------------------------------
    def _forward(self, pil_images: list) -> list:
        torch = self._torch
        batch = torch.stack([self.tf(im.convert("RGB")) for im in pil_images])
        batch = batch.to(self.device)
        out = []
        with torch.no_grad():
            probs = torch.softmax(self.net(batch), 1).cpu().numpy()
            atyp = None
            if self.atyp is not None:
                f = self.atyp(batch).flatten(1)
                f = f / f.norm(dim=1, keepdim=True)
                c = self.centroid / self.centroid.norm()
                atyp = (1.0 - (f @ c)).cpu().numpy()
        for i in range(len(pil_images)):
            p = probs[i]
            rec = {
                "morphotype": self.classes[int(p.argmax())],
                "confidence": float(p.max()),
                "probs": {c: float(p[j]) for j, c in enumerate(self.classes)},
                "p_squat_lobster": float(p[self.squat_index]),
            }
            if atyp is not None:
                rec["atypicality"] = float(atyp[i])
                rec["review_for_caridean"] = bool(atyp[i] >= self.atyp_threshold)
            out.append(rec)
        return out

    def predict_crop(self, image) -> dict:
        """Classify one already-cut crop (path or PIL image)."""
        from PIL import Image
        if isinstance(image, (str, os.PathLike)):
            with Image.open(image) as im:
                return self._forward([im])[0]
        return self._forward([image])[0]

    def predict_boxes(self, frame, boxes, batch_size: int = 32) -> list:
        """Classify boxes ``[(x1,y1,x2,y2), ...]`` on one frame."""
        from PIL import Image
        close = False
        if isinstance(frame, (str, os.PathLike)):
            frame = Image.open(frame)
            close = True
        try:
            frame = frame.convert("RGB")
            crops = [crop_box(frame, *b[:4]) for b in boxes]
        finally:
            if close:
                frame.close()
        results = []
        for i in range(0, len(crops), batch_size):
            results.extend(self._forward(crops[i:i + batch_size]))
        for r, b in zip(results, boxes):
            r["box"] = tuple(float(v) for v in b[:4])
        return results


def load_model(weights: str | None = None, device: str | None = None,
               atypicality_weights: str | None = None) -> MorphotypeModel:
    key = (weights, device, atypicality_weights)
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = MorphotypeModel(weights, atypicality_weights, device)
    return _MODEL_CACHE[key]


def predict_crop(image, weights: str | None = None,
                 device: str | None = None) -> dict:
    """Morphotype one crop. See MorphotypeModel.predict_crop."""
    return load_model(weights, device).predict_crop(image)


def predict_boxes(frame, boxes, weights: str | None = None,
                  device: str | None = None) -> list:
    """Morphotype boxes on one frame. See MorphotypeModel.predict_boxes."""
    return load_model(weights, device).predict_boxes(frame, boxes)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
MODEL_CARD = """\
morphotype -- sub-bucket crustacean morphotype classifier (EPR J1754/J1756)

  classes          squat_lobster | not_crustacean | indeterminate
  not a class      caridean_shrimp -- only n=2 in 3,341 confirmed boxes
  input            224 px crop, side = 1.6 x max(box_w, box_h), min 96 px
  trained on       1,483 hand-labelled confirmed crustacean crops
  split            by SEGMENT (dive+seg); no frame straddles a split

held-out test (441 crops, 12 segments never seen in training)
  squat_lobster    P 0.929 [0.898-0.952]   R 0.946 [0.918-0.967]   n=371
  indeterminate    P 0.707 [0.573-0.819]   R 0.621 [0.493-0.738]   n=66
  not_crustacean   P 0.400 [0.053-0.853]   R 0.500 [0.068-0.932]   n=4   <- n too small
  binary squat_lobster vs rest, t=0.50:  P 0.931 R 0.946 F1 0.939

operating points on P(squat_lobster)
  balanced        0.50   P 0.931  R 0.946
  high precision  0.80   P 0.953  R 0.871

caridean triage
  atypicality = 1 - cos(frozen-ImageNet-feature, squat_lobster centroid).
  Both known carideans rank 8 and 34 of 3,341. Threshold %.4f (98th pct)
  flags 2%% of crops; review those by eye -- the model itself cannot call them.
""" % ATYPICALITY_P98


def _cmd_predict(a):
    m = load_model(a.weights, a.device)
    for p in a.images:
        r = m.predict_crop(p)
        extra = ""
        if "atypicality" in r:
            extra = f"  atypicality={r['atypicality']:.3f}" + (
                "  REVIEW_FOR_CARIDEAN" if r["review_for_caridean"] else "")
        print(f"{p}\t{r['morphotype']}\tconf={r['confidence']:.3f}"
              f"\tP(squat)={r['p_squat_lobster']:.3f}{extra}")


def _cmd_boxes(a):
    boxes = [tuple(float(v) for v in b.split(",")) for b in a.box]
    for r in predict_boxes(a.frame, boxes, a.weights, a.device):
        print(f"{r['box']}\t{r['morphotype']}\tconf={r['confidence']:.3f}"
              f"\tP(squat)={r['p_squat_lobster']:.3f}")


def _cmd_batch(a):
    from PIL import Image
    m = load_model(a.weights, a.device)
    with open(a.dets, newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        print("no rows", file=sys.stderr)
        return 1
    rows.sort(key=lambda r: r.get("path", ""))
    out, counts, cur, im = [], {}, None, None
    try:
        for r in rows:
            p = r["path"]
            if p != cur:
                if im is not None:
                    im.close()
                im = Image.open(p).convert("RGB")       # one frame in memory
                cur = p
            c = crop_box(im, float(r["x1"]), float(r["y1"]),
                         float(r["x2"]), float(r["y2"]))
            res = m.predict_crop(c)
            counts[res["morphotype"]] = counts.get(res["morphotype"], 0) + 1
            out.append({
                "det_id": r.get("det_id", ""), "path": p,
                "morphotype": res["morphotype"],
                "confidence": round(res["confidence"], 4),
                "p_squat_lobster": round(res["p_squat_lobster"], 4),
                "atypicality": round(res.get("atypicality", float("nan")), 4),
                "review_for_caridean": int(res.get("review_for_caridean", 0)),
            })
    finally:
        if im is not None:
            im.close()
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0].keys()))
        w.writeheader()
        w.writerows(out)
    print(f"wrote {a.out}  n={len(out)}")
    for k, v in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:16s} {v:6d}  {v / len(out):6.1%}")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--weights")
    ap.add_argument("--device", choices=["cpu", "cuda"])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("predict", help="classify crop image(s)")
    p.add_argument("images", nargs="+")
    p.set_defaults(fn=_cmd_predict)

    p = sub.add_parser("boxes", help="classify boxes on one frame")
    p.add_argument("frame")
    p.add_argument("--box", action="append", required=True, metavar="x1,y1,x2,y2")
    p.set_defaults(fn=_cmd_boxes)

    p = sub.add_parser("batch", help="classify a detection CSV")
    p.add_argument("--dets", required=True)
    p.add_argument("--out", default="morphotype.csv")
    p.set_defaults(fn=_cmd_batch)

    p = sub.add_parser("info", help="model card and operating points")
    p.set_defaults(fn=lambda a: print(MODEL_CARD))

    a = ap.parse_args(argv)
    return a.fn(a) or 0


if __name__ == "__main__":
    sys.exit(main())
