#!/usr/bin/env python3
"""Tile-level worm-cover estimation for ROV seafloor frames.

WHY THIS EXISTS
---------------
The box detector cannot count tube worms in dense tube fields: audited recall is
0.03, and it collapses exactly where worms are abundant, so its worm counts carry
the *wrong sign* against dissolved CH4.  This module replaces counting individuals
with a coverage measure: the frame is cut into a 9 x 16 grid of ~332 px tiles, each
tile is classified into

    worm_tubes                 discrete, non-branching white/pale polychaete tubes
    bacterial_mat_or_filament  diffuse mat, anastomosing filament webs, Fe floc
    other_seafloor             basalt, sediment, shells, crabs, shadow, shimmer

and the deliverable per frame is ``cover_fraction`` = share of tiles that are
worm-positive.  The third class exists because fine worm tubes and bacterial
filament webs are genuinely confusable; keeping mat as its own label stops mat
from being scored as worms.

Runtime is fully offline: pure torch + torchvision + Pillow, architecture built
with ``weights=None`` and a local state_dict loaded from disk.  No network, no
ultralytics, no API calls.  CPU is supported (about 1-3 s per frame).

USAGE
-----
    # one frame
    python worm_cover.py predict /path/frame.jpg

    # a directory of frames -> worm_cover.csv
    python worm_cover.py batch --frames /path/frames --out worm_cover.csv

    # write into a workspace at survey/fauna/worm_cover.csv
    python worm_cover.py batch --frames /path/frames --workspace /mnt/f/...eprproj

    # library
    from worm_cover import predict_frame
    r = predict_frame("frame.jpg")
    r["cover_fraction"]   # 0.0 - 1.0
    r["tile_grid"]        # 9 rows x 16 cols of class names

Weights are resolved in this order: ``weights=`` argument, ``--weights``,
``$WORM_COVER_WEIGHTS``, then the first hit in DEFAULT_WEIGHT_PATHS.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys

__all__ = [
    "CLASSES", "WORM", "MAT", "OTHER", "GRID_ROWS", "GRID_COLS", "TILE_PX",
    "WormCoverModel", "predict_frame", "load_model", "resolve_weights",
]

CLASSES = ["worm_tubes", "bacterial_mat_or_filament", "other_seafloor"]
WORM, MAT, OTHER = CLASSES
GRID_ROWS, GRID_COLS = 9, 16
TILE_PX = 332                      # native tile size the model was trained at
ROW_LETTERS = "ABCDEFGHI"
IMG_SIZE = 224
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
FRAME_EXTS = (".jpg", ".jpeg", ".png", ".tif", ".tiff")

DEFAULT_WEIGHT_PATHS = (
    os.path.join(os.path.dirname(os.path.abspath(__file__)),
                 "weights", "worm_cover_resnet18.pt"),
    os.path.expanduser("~/models/deploy/worm_cover/worm_cover_resnet18.pt"),
    "/mnt/f/EPR_2026_PROCESSED/paper/worm_cover/worm_cover_resnet18_v2.pt",
)

_MODEL_CACHE: dict = {}


# --------------------------------------------------------------------------
# weights / model
# --------------------------------------------------------------------------
def resolve_weights(weights: str | None = None) -> str:
    """Return a readable weights path or raise FileNotFoundError."""
    cands = []
    if weights:
        cands.append(weights)
    env = os.environ.get("WORM_COVER_WEIGHTS")
    if env:
        cands.append(env)
    cands.extend(DEFAULT_WEIGHT_PATHS)
    for p in cands:
        if p and os.path.isfile(p):
            return p
    raise FileNotFoundError(
        "worm-cover weights not found. Tried:\n  " + "\n  ".join(str(c) for c in cands)
        + "\nPass --weights / weights= or set $WORM_COVER_WEIGHTS."
    )


class WormCoverModel:
    """Tile classifier wrapper. Holds one torch model; processes one frame at a time."""

    def __init__(self, weights: str | None = None, device: str | None = None):
        import torch
        from torchvision.models import resnet18

        self._torch = torch
        self.weights_path = resolve_weights(weights)
        ck = torch.load(self.weights_path, map_location="cpu", weights_only=False)

        self.classes = list(ck.get("classes", CLASSES))
        self.img_size = int(ck.get("img_size", IMG_SIZE))
        self.tile_px = int(ck.get("tile_px", TILE_PX))
        grid = ck.get("grid", [GRID_ROWS, GRID_COLS])
        self.grid_rows, self.grid_cols = int(grid[0]), int(grid[1])
        self.mean = tuple(ck.get("mean", IMAGENET_MEAN))
        self.std = tuple(ck.get("std", IMAGENET_STD))
        try:
            self.worm_index = self.classes.index(WORM)
        except ValueError:
            self.worm_index = 0
        # Calling a tile worm on argmax over-fires: at the measured held-out
        # operating point argmax gives P 0.58 / R 0.68, while P(worm) >= 0.70
        # gives P 0.74 / R 0.55 and the best frame-level cover accuracy
        # (MAE 0.071, r2 0.87). The checkpoint carries that threshold.
        self.recommended_worm_threshold = ck.get("recommended_worm_threshold")
        self.operating_points = ck.get("operating_points", {})

        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        arch = ck.get("arch", "resnet18")
        if arch != "resnet18":
            raise ValueError("unsupported arch in checkpoint: %r" % arch)
        model = resnet18(weights=None)                 # offline: no download
        model.fc = torch.nn.Linear(512, len(self.classes))
        state = ck["state_dict"] if "state_dict" in ck else ck
        model.load_state_dict(state)
        model.eval().to(self.device)
        self.model = model
        self._mean_t = torch.tensor(self.mean).view(1, 3, 1, 1).to(self.device)
        self._std_t = torch.tensor(self.std).view(1, 3, 1, 1).to(self.device)

    # ---- internals -------------------------------------------------------
    def _tiles(self, im):
        """Yield (row, col, PIL tile) for the grid. Tiles derive from the grid so
        frames that are not exactly 5312x2988 still work (scale is reported)."""
        w, h = im.size
        tw, th = w / self.grid_cols, h / self.grid_rows
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                box = (int(round(c * tw)), int(round(r * th)),
                       int(round((c + 1) * tw)), int(round((r + 1) * th)))
                yield r, c, im.crop(box)

    def _frame_tensor(self, im):
        """Resize the whole frame once to (cols*S, rows*S) and return a uint8
        array, so tiles are plain 224x224 slices. Equivalent to cropping each
        tile and resizing it, but ~4x faster (one resample instead of 144)."""
        import numpy as np
        from PIL import Image
        S = self.img_size
        target = (self.grid_cols * S, self.grid_rows * S)
        if im.size != target:
            im = im.resize(target, Image.BILINEAR, reducing_gap=2.0)
        return np.asarray(im, dtype=np.uint8)

    def _batched_logits_fast(self, arr, coords, batch_size):
        import numpy as np, torch
        S = self.img_size
        out = []
        for i in range(0, len(coords), batch_size):
            chunk = coords[i:i + batch_size]
            stack = np.empty((len(chunk), S, S, 3), dtype=np.uint8)
            for j, (r, c) in enumerate(chunk):
                stack[j] = arr[r * S:(r + 1) * S, c * S:(c + 1) * S, :]
            t = torch.from_numpy(stack).to(self.device)
            t = t.permute(0, 3, 1, 2).float().div_(255.0)
            t = (t - self._mean_t) / self._std_t
            with torch.no_grad():
                out.append(self.model(t).float().cpu())
        return torch.cat(out) if out else torch.empty(0, len(self.classes))

    def _batched_logits(self, tiles, batch_size):
        """Reference path: crop each tile at native size then resize it."""
        import torch
        from PIL import Image
        out = []
        for i in range(0, len(tiles), batch_size):
            chunk = tiles[i:i + batch_size]
            arr = torch.empty(len(chunk), 3, self.img_size, self.img_size)
            for j, t in enumerate(chunk):
                t = t.resize((self.img_size, self.img_size), Image.BILINEAR)
                a = torch.frombuffer(bytearray(t.tobytes()), dtype=torch.uint8)
                arr[j] = a.view(self.img_size, self.img_size, 3).permute(2, 0, 1).float() / 255.0
            arr = arr.to(self.device)
            arr = (arr - self._mean_t) / self._std_t
            with torch.no_grad():
                out.append(self.model(arr).float().cpu())
        return torch.cat(out) if out else torch.empty(0, len(self.classes))

    # ---- public ----------------------------------------------------------
    def predict_frame(self, jpg_path: str, batch_size: int = 48,
                      worm_threshold: float | None = None,
                      fast: bool = True) -> dict:
        """Classify every tile of one frame.

        Returns a dict with cover_fraction, tile_grid, prob_grid, class counts and
        the tile scale actually used. Only this one frame is held in memory.
        ``fast=False`` uses the per-tile reference resize path.
        """
        import torch
        from PIL import Image

        with Image.open(jpg_path) as src:
            im = src.convert("RGB")
        width, height = im.size
        coords = [(r, c) for r in range(self.grid_rows) for c in range(self.grid_cols)]
        if fast:
            arr = self._frame_tensor(im)
            im.close()
            logits = self._batched_logits_fast(arr, coords, batch_size)
            del arr
        else:
            tiles = [t for _, _, t in self._tiles(im)]
            logits = self._batched_logits(tiles, batch_size)
            del tiles
            im.close()

        probs = torch.softmax(logits, dim=1).numpy()
        thr = self.recommended_worm_threshold if worm_threshold is None else worm_threshold
        if thr is not None and thr < 0:          # explicit opt-out -> plain argmax
            thr = None
        idx = probs.argmax(axis=1)
        if thr is not None:
            # worm only when it clears the threshold; otherwise fall back to the
            # best non-worm class, so a rejected worm cannot stay "worm".
            nonworm = [k for k in range(len(self.classes)) if k != self.worm_index]
            idx = [self.worm_index if probs[i, self.worm_index] >= thr
                   else max(nonworm, key=lambda k: probs[i, k])
                   for i in range(len(idx))]

        grid = [[None] * self.grid_cols for _ in range(self.grid_rows)]
        pgrid = [[0.0] * self.grid_cols for _ in range(self.grid_rows)]
        counts = {c: 0 for c in self.classes}
        for k, (r, c) in enumerate(coords):
            lab = self.classes[int(idx[k])]
            grid[r][c] = lab
            pgrid[r][c] = float(probs[k, self.worm_index])
            counts[lab] += 1

        n = len(coords)
        tile_w = width / self.grid_cols
        return {
            "frame": os.path.basename(jpg_path),
            "path": os.path.abspath(jpg_path),
            "cover_fraction": counts.get(WORM, 0) / n if n else 0.0,
            "mat_fraction": counts.get(MAT, 0) / n if n else 0.0,
            "mean_worm_prob": float(probs[:, self.worm_index].mean()) if n else 0.0,
            "n_tiles": n,
            "n_worm_tiles": counts.get(WORM, 0),
            "n_mat_tiles": counts.get(MAT, 0),
            "n_other_tiles": counts.get(OTHER, 0),
            "tile_grid": grid,
            "prob_grid": pgrid,
            "frame_width": width,
            "frame_height": height,
            "tile_px_used": tile_w,
            "scale_ok": abs(tile_w - self.tile_px) / self.tile_px < 0.25,
        }


def load_model(weights: str | None = None, device: str | None = None) -> WormCoverModel:
    """Cached model factory so batch runs load the weights once."""
    key = (weights or "", device or "")
    if key not in _MODEL_CACHE:
        _MODEL_CACHE[key] = WormCoverModel(weights=weights, device=device)
    return _MODEL_CACHE[key]


def predict_frame(jpg_path: str, weights: str | None = None, device: str | None = None,
                  batch_size: int = 48, worm_threshold: float | None = None) -> dict:
    """Module-level convenience: {cover_fraction, tile_grid, ...} for one frame."""
    return load_model(weights, device).predict_frame(
        jpg_path, batch_size=batch_size, worm_threshold=worm_threshold)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
CSV_COLS = ["frame", "cover_fraction", "mat_fraction", "mean_worm_prob",
            "n_worm_tiles", "n_mat_tiles", "n_other_tiles", "n_tiles",
            "frame_width", "frame_height", "scale_ok", "path"]


def _list_frames(frames_dir: str, limit: int = 0) -> list:
    out = [os.path.join(frames_dir, f) for f in sorted(os.listdir(frames_dir))
           if f.lower().endswith(FRAME_EXTS)]
    return out[:limit] if limit else out


def _grid_string(grid) -> str:
    sym = {WORM: "W", MAT: "M", OTHER: "."}
    return "|".join("".join(sym.get(c, "?") for c in row) for row in grid)


def cmd_predict(a) -> int:
    m = load_model(a.weights, a.device)
    for p in a.frames:
        r = m.predict_frame(p, batch_size=a.batch_size, worm_threshold=a.worm_threshold)
        print("%s  cover=%.3f  mat=%.3f  worm_tiles=%d/%d%s"
              % (r["frame"], r["cover_fraction"], r["mat_fraction"],
                 r["n_worm_tiles"], r["n_tiles"],
                 "" if r["scale_ok"] else "  [WARN tile scale %.0f px != %d px]"
                 % (r["tile_px_used"], m.tile_px)))
        if a.show_grid:
            sym = {WORM: "W", MAT: "M", OTHER: "."}
            for i, row in enumerate(r["tile_grid"]):
                print("   %s %s" % (ROW_LETTERS[i], "".join(sym.get(c, "?") for c in row)))
    return 0


def cmd_batch(a) -> int:
    paths = list(a.frames or [])
    if a.frames_dir:
        paths.extend(_list_frames(a.frames_dir, a.limit))
    if not paths:
        print("no frames found", file=sys.stderr)
        return 2

    out = a.out
    if not out:
        if a.workspace:
            out = os.path.join(a.workspace, "survey", "fauna", "worm_cover.csv")
        else:
            out = "worm_cover.csv"
    os.makedirs(os.path.dirname(os.path.abspath(out)), exist_ok=True)

    m = load_model(a.weights, a.device)
    tile_out = None
    if a.tiles_out:
        os.makedirs(os.path.dirname(os.path.abspath(a.tiles_out)), exist_ok=True)
        tile_out = open(a.tiles_out, "w", newline="")
        tw = csv.writer(tile_out)
        tw.writerow(["frame", "row", "col", "label", "worm_prob"])

    done = 0
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(CSV_COLS + (["grid"] if a.with_grid else []))
        for p in paths:                                  # one frame at a time
            try:
                r = m.predict_frame(p, batch_size=a.batch_size,
                                    worm_threshold=a.worm_threshold)
            except Exception as exc:                     # keep the batch alive
                print("SKIP %s: %s" % (os.path.basename(p), exc), file=sys.stderr)
                continue
            row = [r["frame"], "%.5f" % r["cover_fraction"], "%.5f" % r["mat_fraction"],
                   "%.5f" % r["mean_worm_prob"], r["n_worm_tiles"], r["n_mat_tiles"],
                   r["n_other_tiles"], r["n_tiles"], r["frame_width"],
                   r["frame_height"], int(r["scale_ok"]), r["path"]]
            if a.with_grid:
                row.append(_grid_string(r["tile_grid"]))
            w.writerow(row)
            if tile_out:
                for i, gr in enumerate(r["tile_grid"]):
                    for j, lab in enumerate(gr):
                        tw.writerow([r["frame"], ROW_LETTERS[i], j + 1, lab,
                                     "%.4f" % r["prob_grid"][i][j]])
            done += 1
            if done % 50 == 0:
                fh.flush()
                print("  %d/%d frames" % (done, len(paths)), flush=True)
    if tile_out:
        tile_out.close()
    print("wrote %d frames -> %s" % (done, out))
    return 0


def cmd_info(a) -> int:
    m = load_model(a.weights, a.device)
    print("weights : %s" % m.weights_path)
    print("arch    : resnet18  img_size=%d  device=%s" % (m.img_size, m.device))
    print("grid    : %d x %d tiles of %d px (native)" % (m.grid_rows, m.grid_cols, m.tile_px))
    print("classes : %s" % ", ".join(m.classes))
    print("worm thr: %s (default)" % m.recommended_worm_threshold)
    for k, v in (m.operating_points or {}).items():
        print("  %-24s %s" % (k, v))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="worm_cover",
        description="Tile-level worm-cover fraction for ROV seafloor frames (offline).")
    p.add_argument("--weights", default=None, help="path to worm_cover_resnet18.pt")
    p.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    p.add_argument("--batch-size", type=int, default=48, dest="batch_size")
    p.add_argument("--worm-threshold", type=float, default=None, dest="worm_threshold",
                   help="call a tile worm if P(worm) >= this. Default: the checkpoint's recommended threshold (0.70). Pass -1 for plain argmax.")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("predict", help="one or more frames to stdout")
    sp.add_argument("frames", nargs="+")
    sp.add_argument("--show-grid", action="store_true", dest="show_grid")
    sp.set_defaults(func=cmd_predict)

    sb = sub.add_parser("batch", help="a frame directory -> worm_cover.csv")
    sb.add_argument("--frames", dest="frames_dir", default=None, help="directory of frames")
    sb.add_argument("frames", nargs="*", default=None, help="explicit frame paths")
    sb.add_argument("--workspace", default=None,
                    help="write to <workspace>/survey/fauna/worm_cover.csv")
    sb.add_argument("--out", default=None)
    sb.add_argument("--tiles-out", default=None, dest="tiles_out",
                    help="also write a per-tile CSV here")
    sb.add_argument("--with-grid", action="store_true", dest="with_grid",
                    help="add a compact per-frame grid string column")
    sb.add_argument("--limit", type=int, default=0)
    sb.set_defaults(func=cmd_batch)

    si = sub.add_parser("info", help="show the loaded model")
    si.set_defaults(func=cmd_info)
    return p


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return a.func(a)


if __name__ == "__main__":
    sys.exit(main())
