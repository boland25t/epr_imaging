#!/usr/bin/env python3
"""Frame atlas — ~100 representative colour-corrected frames spanning the full
range of every gas channel (CH4 rising, CO2 rising, O2 falling), drawn from the
transit-only analysed set.  For each gas the pool is split into quantile bins
and picks are spread across the dive, so the grid reads as a visual ramp:
what the seafloor looks like as that gas changes.

Qt-free.  build_atlas(workspace_dir, out_path=None) -> str (HTML path).
"""
from __future__ import annotations
import re
import base64, sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np, pandas as pd, cv2

from frame_color_analysis import correct

PER_GAS = 33          # frames per gas section  (3 x 33 = 99 ≈ 100)
BINS = 11             # quantile bins per gas; 3 picks per bin
THUMB = (480, 270)

GASES = [
    ("CH4", "CH₄", "ascending", "#7a5cc9",
     "Methane — note the pale, high-brightness ground appearing in the upper "
     "rows as CH₄ climbs toward the vent-field core."),
    ("CO2", "CO₂", "ascending", "#0e7c86",
     "Carbon dioxide — co-delivered with vent fluid; elevation is modest and "
     "broad rather than patchy."),
    ("O2", "O₂", "descending", "#c1272d",
     "Oxygen — ordered toward depletion; the range is narrow, and drawdown is "
     "subtle even over the brightest ground."),
]


def _pick(pool: pd.DataFrame, gas: str, order: str) -> pd.DataFrame:
    d = pool.dropna(subset=[gas]).copy()
    d["bin"] = pd.qcut(d[gas].rank(method="first"), BINS, labels=False)
    picks = []
    per = max(1, PER_GAS // BINS)
    for b, grp in d.groupby("bin"):
        g = grp.sort_values("t")
        idx = np.linspace(0, len(g) - 1, per).round().astype(int)
        picks.append(g.iloc[np.unique(idx)])
    out = pd.concat(picks).drop_duplicates(subset="fn")
    return out.sort_values(gas, ascending=(order == "ascending"))


def _thumb_b64(fn, alt) -> str | None:
    im = cv2.imread(fn, cv2.IMREAD_REDUCED_COLOR_4)
    if im is None: return None
    im = cv2.resize(im, THUMB)
    cor = correct(im, alt)
    ok, buf = cv2.imencode(".jpg", cor, [cv2.IMWRITE_JPEG_QUALITY, 82])
    if not ok: return None
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


def _dive_name(B) -> str:
    m = re.search(r"(J\d{4})", Path(B).name)
    return m.group(1) if m else Path(B).name


def build_atlas(workspace_dir, out_path=None) -> str:
    B = str(workspace_dir)
    global _DIVE
    _DIVE = _dive_name(B)
    d = pd.read_csv(f"{B}/survey/frame_color_metrics.csv")
    base = {g: d[d.white <= 0.02][g].median() for g, *_ in GASES}
    sections = []
    n_total = 0
    for gas, pretty, order, color, blurb in GASES:
        sel = _pick(d, gas, order)
        rows = list(sel.itertuples(index=False))
        with ThreadPoolExecutor(max_workers=10) as ex:
            thumbs = list(ex.map(lambda r: _thumb_b64(r.fn, r.alt), rows))
        cards = []
        for r, th in zip(rows, thumbs):
            if th is None: continue
            n_total += 1
            tt = datetime.fromtimestamp(r.t, timezone.utc).strftime("%H:%M")
            fold = r.__getattribute__(gas) / base[gas] if base[gas] else 0
            gasvals = " &middot; ".join(
                (f"<b style='color:{color}'>{p} {getattr(r, g):,.0f}</b>"
                 if g == gas else f"{p} {getattr(r, g):,.0f}")
                for g, p, *_ in GASES)
            mat = (f"bright px {r.white*100:.0f}%" if r.white > 0.005 else "no bright cover")
            cards.append(
                f"<div class='card'><img alt='frame at {tt} UTC' src='{th}'>"
                f"<div class='meta'><div class='gv'>{gasvals}</div>"
                f"<div class='sub'>{mat} &middot; {fold:.1f}&times; baseline &middot; "
                f"alt {r.alt:.1f} m &middot; {r.seg} &middot; {tt} UTC</div></div></div>")
        arrow = "low &rarr; high" if order == "ascending" else "high &rarr; low (toward depletion)"
        sections.append(
            f"<section><p class='sec-label'>{pretty} &middot; {len(cards)} frames &middot; {arrow}</p>"
            f"<h2>Seafloor across the {pretty} range</h2><p>{blurb}</p>"
            f"<div class='grid'>{''.join(cards)}</div></section>")
    html = _render("".join(sections), n_total, d)
    out = out_path or f"{B}/FRAME_ATLAS.html"
    Path(out).write_text(html, encoding="utf-8")
    return out


def _render(sections_html, n_total, d) -> str:
    dive = _DIVE
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{dive} Frame Atlas</title>
<link rel="preconnect" href="https://fonts.googleapis.com"><link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Spectral:ital,wght@0,500;0,600;1,400&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap">
<style>
:root{{--paper:#fff;--ground:#eef1f4;--ink:#16222b;--muted:#5a6b78;--dim:#8493a0;
  --line:#dde4ea;--line2:#eaeff3;--accent:#0e6d78;--accent-soft:#e3eef0;
  --shadow:0 1px 2px rgba(20,40,55,.05),0 8px 30px rgba(20,40,55,.06);}}
@media(prefers-color-scheme:dark){{:root:not([data-theme="light"]){{--paper:#131c24;--ground:#0d151b;
  --ink:#e8eef3;--muted:#9fb0bd;--dim:#71828f;--line:#26333d;--line2:#1c272f;--accent:#3fb6c2;
  --accent-soft:#152a2e;--shadow:0 1px 2px rgba(0,0,0,.3),0 10px 34px rgba(0,0,0,.35);}}}}
:root[data-theme="dark"]{{--paper:#131c24;--ground:#0d151b;--ink:#e8eef3;--muted:#9fb0bd;--dim:#71828f;
  --line:#26333d;--line2:#1c272f;--accent:#3fb6c2;--accent-soft:#152a2e;
  --shadow:0 1px 2px rgba(0,0,0,.3),0 10px 34px rgba(0,0,0,.35);}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--ground);color:var(--ink);
  font-family:"IBM Plex Sans",system-ui,sans-serif;font-size:15px;line-height:1.6;
  -webkit-font-smoothing:antialiased}}
.wrap{{max-width:1160px;margin:0 auto;padding:0 24px 60px}}
header.top{{padding:52px 8px 10px}}
.eyebrow{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--accent);margin:0 0 12px}}
h1{{font-family:"Spectral",Georgia,serif;font-weight:600;font-size:40px;line-height:1.08;
  margin:0 0 12px;letter-spacing:-.01em;text-wrap:balance}}
.lede{{font-family:"Spectral",serif;font-size:18px;font-style:italic;color:var(--muted);
  margin:0 0 8px;max-width:62ch}}
section{{padding:34px 8px 6px}}
.sec-label{{font-family:"IBM Plex Mono",monospace;font-size:12px;letter-spacing:.14em;
  text-transform:uppercase;color:var(--accent);margin:0 0 6px;display:flex;align-items:center;gap:10px}}
.sec-label::before{{content:"";width:20px;height:1px;background:var(--accent)}}
h2{{font-family:"Spectral",serif;font-weight:600;font-size:27px;margin:0 0 10px;letter-spacing:-.01em}}
section>p{{color:var(--muted);max-width:70ch;margin:0 0 20px}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:16px}}
.card{{background:var(--paper);border:1px solid var(--line);border-radius:7px;overflow:hidden;
  box-shadow:var(--shadow);break-inside:avoid;page-break-inside:avoid}}
.card img{{display:block;width:100%;height:auto}}
.meta{{padding:9px 13px 11px}}
.gv{{font-family:"IBM Plex Mono",monospace;font-size:12.5px;font-variant-numeric:tabular-nums;
  letter-spacing:-.01em}}
.sub{{font-size:11.5px;color:var(--dim);margin-top:3px}}
.note{{border-left:3px solid var(--accent);background:var(--accent-soft);padding:13px 18px;
  border-radius:0 6px 6px 0;margin:16px 8px 8px;color:var(--muted);font-size:13.5px;max-width:80ch}}
.note b{{color:var(--ink)}}
footer{{padding:36px 8px 0;color:var(--dim);font-size:13px}}
footer .mono{{font-family:"IBM Plex Mono",monospace;color:var(--muted)}}
@media print{{
  *{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
  body{{background:#fff}}
  .wrap{{max-width:none;padding:0 20px}}
  .card{{box-shadow:none}}
  .grid{{grid-template-columns:repeat(3,1fr);gap:10px}}
  h2{{break-after:avoid}}
}}
</style></head><body><div class="wrap">
<header class="top">
  <p class="eyebrow">Jason Dive {dive} &middot; East Pacific Rise 9&deg;N</p>
  <h1>Frame Atlas</h1>
  <p class="lede">{n_total} colour-corrected seafloor frames sampled across the full range of
  every dissolved-gas channel &mdash; a visual ramp from background to the vent-field core.</p>
</header>
<div class="note"><b>How frames were chosen.</b> All frames come from the transit-only analysed
set ({len(d):,} frames; station-keeping at known sites excluded). For each gas the set is split
into {BINS} equal-count bins spanning its full range, with picks spread across the dive so no
single pass dominates. Every frame is altitude-normalised and white-balanced; concentrations are
in native sensor units, and &ldquo;&times; baseline&rdquo; is relative to the bare-seafloor median
of the section&rsquo;s gas. A frame may appear in more than one section.</div>
{sections_html}
<footer>
  <div class="mono">Jason Dive {dive} &middot; UTM 13N (EPSG:32613) &middot;
  generated {datetime.now():%Y-%m-%d}</div>
  Companion to the {dive} Survey Report &mdash; imagery &times; chemistry section.
</footer>
</div></body></html>"""


if __name__ == "__main__":
    print(build_atlas(sys.argv[1] if len(sys.argv) > 1 else "."))
