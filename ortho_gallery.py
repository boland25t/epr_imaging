#!/usr/bin/env python3
"""Zoomed orthomosaic close-up gallery for the survey report.

Selects the best-covered chunk orthomosaics (spread across segments), applies a
per-tile percentile contrast stretch (deep-sea orthos are dark and blue; the
stretch is presentation-only and clearly captioned as such), and renders them
as multi-panel figures into survey/photogrammetry/ortho_gallery/.

Qt-free.  render_gallery(workspace_dir, ...) -> list of figure paths.
"""
from __future__ import annotations
import glob
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def stretch(rgb: np.ndarray, valid: np.ndarray, lo=2, hi=98) -> np.ndarray:
    """Percentile contrast stretch per channel over valid (imaged) pixels."""
    out = rgb.astype(np.float32).copy()
    for c in range(3):
        v = out[:, :, c][valid]
        if v.size < 100:
            continue
        a, b = np.percentile(v, [lo, hi])
        if b <= a:
            continue
        out[:, :, c] = np.clip((out[:, :, c] - a) / (b - a), 0, 1) * 255
    return out


def _read_tile(path: str, max_px: int):
    with rasterio.open(path) as ds:
        sc = max(1.0, max(ds.width, ds.height) / max_px)
        img = ds.read([1, 2, 3], out_shape=(3, int(ds.height / sc), int(ds.width / sc)),
                      resampling=Resampling.average)
        b = ds.bounds
    rgb = np.transpose(img, (1, 2, 0)).astype(np.float32)
    valid = rgb.sum(2) > 0
    # crop to the largest connected imaged patch (plus margin) so panels are
    # not dominated by empty extent between scattered fragments
    from scipy import ndimage
    lab, nlab = ndimage.label(valid)
    if nlab > 1:
        sizes = ndimage.sum(valid, lab, range(1, nlab + 1))
        valid_main = lab == (int(np.argmax(sizes)) + 1)
    else:
        valid_main = valid
    ys, xs = np.where(valid_main)
    if len(ys):
        m = max(8, int(0.02 * max(rgb.shape[:2])))
        y0, y1 = max(0, ys.min() - m), min(rgb.shape[0], ys.max() + m)
        x0, x1 = max(0, xs.min() - m), min(rgb.shape[1], xs.max() + m)
        px_x = (b.right - b.left) / rgb.shape[1]
        px_y = (b.top - b.bottom) / rgb.shape[0]
        from rasterio.coords import BoundingBox
        b = BoundingBox(left=b.left + x0 * px_x, right=b.left + x1 * px_x,
                        top=b.top - y0 * px_y, bottom=b.top - y1 * px_y)
        rgb, valid = rgb[y0:y1, x0:x1], valid[y0:y1, x0:x1]
    return rgb, valid, b


def pick_chunks(workspace_dir, n=21, per_seg_cap=2, exclude=()) -> list[str]:
    """Rank chunk orthos by imaged coverage, keep segment diversity."""
    cands = []
    for p in sorted(glob.glob(
            f"{workspace_dir}/survey/photogrammetry/seg*/chunk_*/orthomosaic.tif")):
        norm = p.replace("\\", "/")
        if any(f"/{x}/" in norm + "/" for x in exclude):
            continue
        try:
            with rasterio.open(p) as ds:
                sc = max(1.0, max(ds.width, ds.height) / 400)
                a = ds.read(1, out_shape=(int(ds.height / sc), int(ds.width / sc)),
                            resampling=Resampling.nearest)
                px_m = ds.res[0] * sc
            frac = float((a > 0).mean())
            cov_m2 = frac * a.size * px_m * px_m
            # density-weighted score: prefer well-filled patches over long
            # sparse ribbons whose extent is mostly empty
            cands.append((cov_m2 * frac ** 0.7, p))
        except Exception:
            continue
    cands.sort(reverse=True)
    picked, seg_ct = [], {}
    for cov, p in cands:
        seg = Path(p).parts[-3]
        if seg_ct.get(seg, 0) >= per_seg_cap:
            continue
        picked.append(p)
        seg_ct[seg] = seg_ct.get(seg, 0) + 1
        if len(picked) >= n:
            break
    return picked


def render_gallery(workspace_dir, n=21, per_fig=3, max_px=1700,
                   log_fn=print) -> list[str]:
    B = str(workspace_dir)
    out_dir = Path(B) / "survey" / "photogrammetry" / "ortho_gallery"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    picks = pick_chunks(B, n=n)
    log_fn(f"gallery: {len(picks)} chunk close-ups selected")
    figs = []
    plt.rcParams["font.family"] = "DejaVu Sans"
    for gi in range(0, len(picks), per_fig):
        group = picks[gi:gi + per_fig]
        fig, axes = plt.subplots(1, len(group), figsize=(5.0 * len(group), 8.6), dpi=150)
        axes = np.atleast_1d(axes)
        fig.patch.set_facecolor("#0e1620")
        for ax, p in zip(axes, group):
            rgb, valid, b = _read_tile(p, max_px)
            srgb = stretch(rgb, valid) / 255.0
            alpha = np.where(valid, 1.0, 0.0)
            ax.set_facecolor("#0e1620")
            ax.imshow(np.dstack([srgb, alpha]),
                      extent=[b.left, b.right, b.bottom, b.top], origin="upper",
                      interpolation="bilinear")
            ax.set_aspect("equal")
            ax.set_xticks([]); ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color("#2a3846")
            xl, yl = ax.get_xlim(), ax.get_ylim()
            x0 = xl[0] + (xl[1] - xl[0]) * 0.06
            y0 = yl[0] + (yl[1] - yl[0]) * 0.045
            ax.plot([x0, x0 + 10], [y0, y0], color="w", lw=3)
            ax.text(x0 + 5, y0 + (yl[1] - yl[0]) * 0.014, "10 m", color="w",
                    ha="center", fontsize=9)
            seg, chunk = Path(p).parts[-3], Path(p).parts[-2]
            ax.set_title(f"{seg} / {chunk}  ·  {b.right-b.left:.0f}×{b.top-b.bottom:.0f} m",
                         fontsize=9.5, color="#c8d3dc", pad=5)
        fig.tight_layout()
        fp = out_dir / f"gallery_{gi // per_fig + 1:02d}.png"
        fig.savefig(fp, dpi=150, facecolor="#0e1620", bbox_inches="tight")
        plt.close(fig)
        figs.append(str(fp))
        log_fn(f"  wrote {fp.name}")
    return figs


if __name__ == "__main__":
    render_gallery(sys.argv[1] if len(sys.argv) > 1 else ".")
