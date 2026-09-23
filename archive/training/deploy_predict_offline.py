#!/usr/bin/env python3
"""Offline megafauna detection using the bundled weights + deploy_config.json.

Shipped verbatim into /home/troyboland/models/deploy/ as predict_offline.py.
No internet, no Claude, nothing from the epr_imaging repo: the bucket rules are
inlined below so the bundle is genuinely self-contained.

Reads deploy_config.json, and for each bucket runs whichever model that config
recommends at that bucket's threshold, then writes one CSV of kept detections.

    python predict_offline.py --frames /path/to/frames --out detections.csv
    python predict_offline.py --frames ... --mode triage   # over-flag, don't miss
"""

import argparse
import csv
import json
from pathlib import Path

from ultralytics import YOLO

HERE = Path(__file__).resolve().parent


# --- bucket rules, inlined so the bundle needs nothing from the repo ---------
_KEYS = {
    "crustacean": ("munidopsis", "munida", "galatheoidea", "lithodidae", "paralomis",
                   "neolithodes", "chionoecetes", "chorilia", "cancridae", "moloha",
                   "paguroidea", "sternostylus", "pleuroncodes", "decapoda", "caridea",
                   "pandalidae", "pandalus", "plesionika", "heterocarpus", "crustacea",
                   "eualus", "glyphocrangon", "acanthephyra", "pasiphaea", "crab",
                   "carapace", "amphipoda", "isopoda", "munnopsidae", "acanthamunnopsis"),
    "worm": ("riftia", "lamellibrachia", "siboglinidae", "polychaeta", "sabellidae",
             "serpulidae", "terebellidae", "harmothoe", "anchinothria", "osedax",
             "worm", "tube", "nemertea", "echiura", "hirudinea", "saxipendium",
             "poeobius", "swima"),
    "fish": ("actinopterygii", "zoarcidae", "macrouridae", "coryphaenoides",
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
             "desmodema", "alepisaurus", "anotopterus", "fish"),
    "anemone": ("anemone", "actiniar", "actinostol", "actinernus", "hormathiid",
                "metridium", "liponema", "bolocera", "corallimorph", "cerianth",
                "zoanth", "isosicyonis"),
}

# Midwater / non-fauna taxa the production pipeline drops before analysis: on a
# 2500 m down-looking frame these can only be domain-shift error.
_DROP = frozenset({
    "Mola mola", "Bathochordaeus", "Pyrosoma atlanticum", "Pyrosoma detritus",
    "Salpida", "salp chain", "salp detritus", "Thetys vagina", "Appendicularia",
    "Larvacea", "Oikopleura", "Chaetognatha", "Pteropoda", "Thecosomata",
    "Gymnosomata", "Euphausia", "Krill molt", "Mysida", "Copepoda",
    "Tomopteris", "Tomopterid eggcase", "marine snow", "mung", "ink",
    "detritus", "sand", "geologic", "bacterial mat", "shell", "bone", "wood",
    "molt", "carcass", "eggcase", "stalk", "marine organism", "Animalia",
})


def bucket_of(cls):
    low = str(cls).lower()
    for name, keys in _KEYS.items():
        if any(k in low for k in keys):
            return name
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", required=True, help="directory of frame images")
    ap.add_argument("--out", default="detections.csv")
    ap.add_argument("--mode", choices=("balanced", "triage"), default="balanced")
    ap.add_argument("--device", default=0, help="GPU index, or 'cpu'")
    ap.add_argument("--batch", type=int, default=8)
    a = ap.parse_args()

    cfg = json.loads((HERE / "deploy_config.json").read_text())
    imgsz = cfg["imgsz"]
    buckets = cfg["taxonomy"]

    # which model each bucket wants, and at what threshold
    want = {}
    for b in buckets:
        use = cfg["recommended_model_per_bucket"][b]["use_model"]
        if use is None:
            continue
        thr = cfg["models"][use]["buckets"][b].get(f"conf_{a.mode}")
        if thr is None:
            continue
        want.setdefault(use, {})[b] = float(thr)
    if not want:
        raise SystemExit("deploy_config.json recommends no usable bucket")

    exts = (".jpg", ".jpeg", ".png")
    frames = sorted(p for p in Path(a.frames).iterdir() if p.suffix.lower() in exts)
    if not frames:
        raise SystemExit(f"no images under {a.frames}")
    print(f"{len(frames)} frames | mode={a.mode} | models={list(want)}")
    for m, bt in want.items():
        print(f"  {m}: " + ", ".join(f"{b}@{t:.2f}" for b, t in sorted(bt.items())))

    rows = []
    for model_name, bthr in want.items():
        wname = Path(cfg["models"][model_name]["weights"]).name
        wpath = HERE / wname
        if not wpath.exists():
            wpath = HERE / ("epr_fauna_v2.pt" if "v2" in model_name
                            else "mbari_315k_yolov8.pt")
        if not wpath.exists():
            raise SystemExit(f"missing weights for {model_name} ({wname})")
        model = YOLO(str(wpath))
        names = model.names
        zero_shot = len(names) > len(buckets)      # the 499-class taxon model
        floor = min(bthr.values())
        for i in range(0, len(frames), a.batch):
            chunk = frames[i:i + a.batch]
            res = model.predict([str(p) for p in chunk], conf=floor, imgsz=imgsz,
                                device=a.device, verbose=False)
            for fp, r in zip(chunk, res):
                if r.boxes is None or len(r.boxes) == 0:
                    continue
                for box, cf, cl in zip(r.boxes.xyxy.cpu().numpy(),
                                       r.boxes.conf.cpu().numpy(),
                                       r.boxes.cls.cpu().numpy().astype(int)):
                    raw = str(names[int(cl)])
                    if zero_shot and raw in _DROP:
                        continue
                    bucket = bucket_of(raw) if zero_shot else raw
                    if bucket not in bthr or float(cf) < bthr[bucket]:
                        continue
                    rows.append({"fn": fp.name, "bucket": bucket,
                                 "cls": raw, "conf": round(float(cf), 4),
                                 "x1": round(float(box[0]), 1),
                                 "y1": round(float(box[1]), 1),
                                 "x2": round(float(box[2]), 1),
                                 "y2": round(float(box[3]), 1),
                                 "model": model_name, "mode": a.mode})
            if (i // a.batch) % 25 == 0:
                print(f"  {model_name}: {i}/{len(frames)}", flush=True)

    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["fn", "bucket", "cls", "conf", "x1", "y1",
                                           "x2", "y2", "model", "mode"])
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} detections -> {a.out}")
    print("REMINDER: worm counts from this detector are NOT usable (whole-frame "
          "recall ~0.03). Restrict quantitative products to 3-8 m altitude. "
          "Do not report counts for buckets flagged uncalibrated in README.md.")


if __name__ == "__main__":
    main()
