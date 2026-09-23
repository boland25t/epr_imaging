#!/usr/bin/env python3
"""Native-resolution mosaic review deck (personal QA, not for presentation).

Every chunk orthomosaic of the given dives, cut into tiles capped at
MAX_DOWNSAMPLE x native resolution, contrast-stretched, written as JPEGs with
a local HTML deck (one tile per slide; arrow keys / index navigation; click
the star to mark picks — the pick list box collects slide ids for reporting
back which belong in the presentation deck).

Output: <ROOT>/slides/review/index.html + tiles/*.jpg
Qt-free.  build_review_deck(dives) -> index path.
"""
from __future__ import annotations
import glob
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from PIL import Image

from ortho_gallery import stretch

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
OUT = ROOT / "slides" / "review"
TILE_OUT_PX = 4096          # long side of an output tile JPEG
MAX_DOWNSAMPLE = 2          # never coarser than 2x native unless tile cap forces it
MAX_TILES = 8               # per mosaic; exceeded -> coarsen and note it
JPEG_Q = 85


def _valid_bbox(ds):
    """Row/col bounding box of imaged pixels, from a decimated alpha/rgb read."""
    sc = max(1, int(max(ds.width, ds.height) / 1024))
    a = ds.read(1, out_shape=(ds.height // sc, ds.width // sc),
                resampling=Resampling.nearest)
    ys, xs = np.where(a > 0)
    if not len(ys):
        return None
    return (int(xs.min() * sc), int(ys.min() * sc),
            min(ds.width, int((xs.max() + 1) * sc)),
            min(ds.height, int((ys.max() + 1) * sc)))


def _tile_plan(w, h):
    """Split a (w, h) native-pixel region into tiles honouring the caps."""
    src_tile = TILE_OUT_PX * MAX_DOWNSAMPLE
    nx, ny = math.ceil(w / src_tile), math.ceil(h / src_tile)
    if nx * ny > MAX_TILES:
        # coarsen: grow tile source size until the count fits
        scale = math.sqrt(nx * ny / MAX_TILES)
        src_tile = int(src_tile * scale) + 1
        nx, ny = math.ceil(w / src_tile), math.ceil(h / src_tile)
    return nx, ny, src_tile


def _render_mosaic(args):
    dive, path = args
    label = f"{dive} {'/'.join(Path(path).parts[-3:-1])}"
    slug = f"{dive}_{Path(path).parts[-3]}_{Path(path).parts[-2]}"
    rows = []
    with rasterio.open(path) as ds:
        bb = _valid_bbox(ds)
        if bb is None:
            return rows
        x0, y0, x1, y1 = bb
        w, h = x1 - x0, y1 - y0
        nx, ny, src_tile = _tile_plan(w, h)
        px_m = ds.res[0]
        for j in range(ny):
            for i in range(nx):
                cx0 = x0 + i * src_tile
                cy0 = y0 + j * src_tile
                cw = min(src_tile, x1 - cx0)
                ch = min(src_tile, y1 - cy0)
                out_w = min(TILE_OUT_PX, cw)
                out_h = max(1, int(ch * out_w / cw))
                data = ds.read([1, 2, 3],
                               window=Window(cx0, cy0, cw, ch),
                               out_shape=(3, out_h, out_w),
                               resampling=Resampling.average)
                rgb = np.transpose(data, (1, 2, 0)).astype(np.float32)
                valid = rgb.sum(2) > 0
                if valid.mean() < 0.02:
                    continue
                rgb = stretch(rgb, valid)
                rgb[~valid] = 14  # near-black ground for unimaged pixels
                eff_mm = cw / out_w * px_m * 1000
                tid = f"{slug}_t{j}{i}"
                fp = OUT / "tiles" / f"{tid}.jpg"
                Image.fromarray(rgb.astype(np.uint8)).save(fp, quality=JPEG_Q)
                rows.append(dict(
                    id=tid, file=f"tiles/{fp.name}", label=label,
                    tile=f"tile {j * nx + i + 1}/{nx * ny}" if nx * ny > 1 else "full",
                    extent=f"{cw * px_m:.0f}x{ch * px_m:.0f} m",
                    res=f"{eff_mm:.1f} mm/px (native {px_m * 1000:.1f})"))
    return rows


def build_review_deck(dives, log=print) -> str:
    (OUT / "tiles").mkdir(parents=True, exist_ok=True)
    jobs = []
    for dive in dives:
        for p in sorted(glob.glob(
                str(ROOT / f"{dive}_down.eprproj" /
                    "survey/photogrammetry/seg*/chunk_*/orthomosaic.tif"))):
            if Path(p).stat().st_size > 1e6:
                jobs.append((dive, p))
    log(f"{len(jobs)} mosaics to tile")
    all_rows = []
    with ThreadPoolExecutor(max_workers=6) as ex:
        for k, rows in enumerate(ex.map(_render_mosaic, jobs), 1):
            all_rows.extend(rows)
            if k % 10 == 0:
                log(f"  {k}/{len(jobs)} mosaics done ({len(all_rows)} tiles)")
    log(f"tiles: {len(all_rows)}")

    slides = "\n".join(
        f'<section class="slide" id="{r["id"]}">'
        f'<img loading="lazy" src="{r["file"]}">'
        f'<div class="cap"><input type="checkbox" class="pick" data-id="{r["id"]}">'
        f'<b>{r["label"]}</b> &middot; {r["tile"]} &middot; {r["extent"]} &middot; '
        f'{r["res"]} &middot; <span class="mono">{r["id"]}</span></div></section>'
        for r in all_rows)
    toc = "\n".join(f'<a href="#{r["id"]}">{r["label"]} {r["tile"]}</a>'
                    for r in all_rows)
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Mosaic Review Deck</title><style>
* {{ margin:0; box-sizing:border-box }}
body {{ background:#0b1119; color:#dbe4ec; font-family:'Segoe UI',sans-serif; display:flex }}
nav {{ width:250px; height:100vh; overflow-y:auto; position:sticky; top:0;
      background:#0e1620; padding:10px; font-size:11px; flex-shrink:0 }}
nav a {{ display:block; color:#9fb0bd; text-decoration:none; padding:2px 4px }}
nav a:hover {{ color:#fff; background:#16222e }}
main {{ flex:1 }}
.slide {{ min-height:100vh; display:flex; flex-direction:column;
         align-items:center; justify-content:center; padding:12px; border-bottom:1px solid #1c2833 }}
.slide img {{ max-width:100%; max-height:92vh }}
.cap {{ padding:8px; font-size:14px; color:#c8d3dc }}
.mono {{ font-family:monospace; color:#6ec6e6 }}
#picks {{ position:fixed; right:12px; top:12px; background:#131f2b; border:1px solid #2a3846;
         padding:10px; font-size:12px; max-width:260px; z-index:9 }}
#picks textarea {{ width:100%; height:90px; background:#0b1119; color:#6ec6e6;
                  border:1px solid #2a3846; font-family:monospace; font-size:11px }}
input.pick {{ transform:scale(1.4); margin-right:8px }}
</style></head><body>
<nav>{toc}</nav><main>{slides}</main>
<div id="picks"><b>Picks</b> (check boxes; copy this list back to Claude)<br>
<textarea id="picklist" readonly></textarea></div>
<script>
const key='mosaic_picks';
const saved=new Set(JSON.parse(localStorage.getItem(key)||'[]'));
document.querySelectorAll('input.pick').forEach(cb=>{{
  cb.checked=saved.has(cb.dataset.id);
  cb.addEventListener('change',()=>{{
    cb.checked?saved.add(cb.dataset.id):saved.delete(cb.dataset.id);
    localStorage.setItem(key,JSON.stringify([...saved]));render();}});
}});
function render(){{document.getElementById('picklist').value=[...saved].join('\\n');}}
render();
</script></body></html>"""
    idx = OUT / "index.html"
    idx.write_text(html, encoding="utf-8")
    log(f"wrote {idx}")
    return str(idx)


if __name__ == "__main__":
    build_review_deck(sys.argv[1:] or ["J1756", "J1754"])
    print("REVIEW_DECK_DONE")
