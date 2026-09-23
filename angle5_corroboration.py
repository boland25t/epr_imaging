#!/usr/bin/env python3
"""Angle 5 — optical corroboration of TRANSIT gas-anomaly detections.

For the methods paper: do the detector's transit windows (happenstance
encounters while surveying — the detection population) coincide with
bright-pixel seafloor cover more often than chance?  Imagery exists only
where photogrammetry segments ran (~1 frame / 5 s), so many windows hold
no frames at all — those are 'unimaged', not negative, and are excluded
from corroboration rates (their count is reported).

Per dive (J1754, J1756), transit windows only:
  window record — frames whose t falls in [start, end] (±PAD_VARIANT s
                  sensitivity variant): n_frames, n_cover (white>COVER_T),
                  any_cover, median bright/white.
  control       — per imaged window, N_CONTROLS pseudo-windows of the same
                  duration centred on random transit frames of the same dive
                  that do not overlap ANY real window (any tier, station
                  included); each is imaged by construction (>=1 frame).
  tests         — Fisher exact on pooled any-cover counts (windows vs pooled
                  controls; controls are 20-per-window pseudo-replicates, so
                  the control margin is anti-conservative) and Mann-Whitney
                  (two-sided) on per-span median brightness.

Writes to /mnt/f/EPR_2026_PROCESSED/paper/:
  angle5_window_optics.csv   per-window records, both dives, pad-0 and pad-10
  angle5_summary.json        rates / ratios / p-values incl. the ±10 s variant
  angle5_fig.png             dark two-panel: window-vs-control rate bars per
                             dive + pooled; corroboration rate by channel family

Qt-free.
"""
from __future__ import annotations
import json, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, mannwhitneyu
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
PAPER = ROOT / "paper"
COVER_T = 0.02            # bright-pixel cover threshold, as analysis_figures
N_CONTROLS = 20           # pseudo-windows per imaged real window
PAD_VARIANT = 10.0        # s — sensitivity pad on window edges
PADS = (0.0, PAD_VARIANT)
SEED = 20260916
FAMILIES = ("CO2", "CH4", "O2", "Temperature")

BG, INKD, MUT, GRID = "#0e1620", "#dbe4ec", "#9fb0bd", "#2a3846"
GOLD, SLATE = "#f2c94c", "#5a7d9a"      # window rate / matched-control rate


def _load(dive, root=ROOT):
    ws = Path(root) / f"{dive}_down.eprproj"
    win = pd.read_csv(ws / "survey" / "anomaly" / "window_context.csv")
    win["t0"] = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp)
    win["t1"] = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp)
    fr = pd.read_csv(ws / "survey" / "frame_color_metrics.csv",
                     usecols=["t", "white", "bright"]).dropna().sort_values("t")
    return win, fr


def _span(ft, fw, fb, a, b):
    i, j = int(np.searchsorted(ft, a)), int(np.searchsorted(ft, b, side="right"))
    w = fw[i:j]
    return dict(n_frames=j - i, n_cover=int((w > COVER_T).sum()),
                any_cover=bool((w > COVER_T).any()),
                median_bright=float(np.median(fb[i:j])) if j > i else np.nan,
                median_white=float(np.median(w)) if j > i else np.nan)


def window_optics(dive, win, fr, pad=0.0) -> pd.DataFrame:
    """Per-transit-window optical record; n_frames == 0 means 'unimaged'."""
    ft, fw, fb = (fr[c].to_numpy() for c in ("t", "white", "bright"))
    rows = [dict(dive=dive, window_id=r.window_id, start_time=r.start_time,
                 end_time=r.end_time, confidence_tier=r.confidence_tier,
                 channels=r.channels, duration_s=r.duration_s,
                 median_speed=r.median_speed,
                 **_span(ft, fw, fb, r.t0 - pad, r.t1 + pad))
            for r in win[win.context == "transit"].itertuples()]
    return pd.DataFrame(rows)


def control_spans(dive, win, fr, opt, pad=0.0, n_controls=N_CONTROLS) -> pd.DataFrame:
    """Matched pseudo-windows: per imaged real window, n_controls spans of the
    same (padded) duration centred on transit frames of the same dive, none
    overlapping any real window (any tier, station included).  Centred on a
    frame, so each span is imaged by construction."""
    ft, fw, fb = (fr[c].to_numpy() for c in ("t", "white", "bright"))
    excl = win[["t0", "t1"]].to_numpy()          # every real window, any context
    rng = np.random.default_rng([SEED, int(pad), *dive.encode()])
    rows = []
    for r in opt[opt.n_frames > 0].itertuples():
        half = r.duration_s / 2 + pad
        ok = np.ones(len(ft), bool)
        for a, b in excl:                        # span vs padded real window
            ok &= (ft < a - pad - half) | (ft > b + pad + half)
        idx = np.flatnonzero(ok)
        if not len(idx):
            continue
        for c in ft[rng.choice(idx, size=min(n_controls, len(idx)), replace=False)]:
            rows.append(dict(dive=dive, window_id=r.window_id, t_centre=float(c),
                             duration_s=float(2 * half),
                             **_span(ft, fw, fb, c - half, c + half)))
    return pd.DataFrame(rows)


def corroboration(opt, ctl) -> dict:
    """Rates, ratio, Fisher exact and brightness Mann-Whitney for one
    window-record set against its matched controls."""
    imaged = opt[opt.n_frames > 0]
    kw, nw = int(imaged.any_cover.sum()), len(imaged)
    kc, nc = int(ctl.any_cover.sum()), len(ctl)
    blk = dict(n_transit_windows=int(len(opt)), n_imaged=nw,
               n_unimaged=int(len(opt) - nw),
               unimaged_frac=round(1 - nw / len(opt), 4) if len(opt) else None,
               window_cover=kw, window_rate=round(kw / nw, 4) if nw else None,
               control_spans=nc, control_cover=kc,
               control_rate=round(kc / nc, 4) if nc else None)
    if nw and nc:
        orr, p = fisher_exact([[kw, nw - kw], [kc, nc - kc]])
        blk.update(ratio=round((kw / nw) / (kc / nc), 3) if kc else None,
                   fisher_odds_ratio=round(float(orr), 3), fisher_p=float(p))
        bw, bc = imaged.median_bright.dropna(), ctl.median_bright.dropna()
        if len(bw) and len(bc):
            blk.update(bright_window_median=round(float(bw.median()), 3),
                       bright_control_median=round(float(bc.median()), 3),
                       brightness_p=float(mannwhitneyu(bw, bc,
                                                       alternative="two-sided")[1]))
    return blk


def breakdown(opt, ctl, by="family") -> dict:
    """Any-cover rate among imaged windows split by channel family or tier,
    each against its own matched control spans."""
    imaged = opt[opt.n_frames > 0]
    if by == "family":
        groups = [(f, imaged.channels.fillna("").str.contains(f)) for f in FAMILIES]
        groups.append(("any", pd.Series(True, index=imaged.index)))
    else:
        groups = [(t, imaged.confidence_tier == t)
                  for t in ("HIGH", "MODERATE", "SCREEN")]
    out = {}
    for name, m in groups:
        sub = imaged[m]
        keys = set(zip(sub.dive, sub.window_id))
        c = ctl[[k in keys for k in zip(ctl.dive, ctl.window_id)]]
        rate = round(float(sub.any_cover.mean()), 4) if len(sub) else None
        crate = round(float(c.any_cover.mean()), 4) if len(c) else None
        out[name] = dict(n=int(len(sub)), cover=int(sub.any_cover.sum()), rate=rate,
                         control_rate=crate,
                         ratio=round(rate / crate, 3) if rate is not None and crate
                         else None)
    return out


def _style(ax, title=None, ylabel=None):
    ax.set_facecolor(BG)
    if title: ax.set_title(title, fontsize=10.5, color=INKD)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9, color=MUT)
    ax.tick_params(colors=MUT, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, lw=0.5, alpha=0.6)
    ax.set_axisbelow(True)


def make_figure(blocks, fam, out_png):
    """Left: window-vs-control any-cover rate per dive + pooled, count labels.
    Right: pooled corroboration rate by channel family, control reference."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.9), dpi=150)
    fig.patch.set_facecolor(BG)
    ax, names = axes[0], list(blocks)
    x, w = np.arange(len(names)), 0.36
    for j, (col, lab, rk, kk, nk) in enumerate(
            ((GOLD, "anomaly windows", "window_rate", "window_cover", "n_imaged"),
             (SLATE, "matched controls", "control_rate", "control_cover",
              "control_spans"))):
        vals = [100 * (blocks[n][rk] or 0) for n in names]
        ax.bar(x + (j - 0.5) * w, vals, w * 0.94, color=col, label=lab)
        for xi, n, v in zip(x + (j - 0.5) * w, names, vals):
            ax.text(xi, v + 1.2, f"{blocks[n][kk]}/{blocks[n][nk]}", ha="center",
                    fontsize=7.5, color=INKD)
    ax.set_xticks(x); ax.set_xticklabels(names, fontsize=9, color=INKD)
    ax.legend(fontsize=8, frameon=False, labelcolor=INKD, loc="upper center", ncol=2)
    _style(ax, title="Bright-cover rate: windows vs matched controls",
           ylabel="% of spans with bright-pixel cover >2%")
    top = max(100 * (blocks[n]["window_rate"] or 0) for n in names)
    ax.set_ylim(0, max(1.35 * top, 30))

    ax = axes[1]
    labels = {"CO2": "CO$_2$", "CH4": "CH$_4$", "O2": "O$_2$",
              "Temperature": "Temp", "any": "Any"}
    keys = [k for k in (*FAMILIES, "any") if fam.get(k, {}).get("n")]
    vals = [100 * fam[k]["rate"] for k in keys]
    ax.bar(np.arange(len(keys)), vals, 0.6, color=GOLD)
    for xi, (k, v) in enumerate(zip(keys, vals)):
        ax.text(xi, v + 1.2, f"{fam[k]['cover']}/{fam[k]['n']}", ha="center",
                fontsize=7.5, color=INKD)
    cr = 100 * (fam.get("any", {}).get("control_rate") or 0)
    ax.axhline(cr, color=SLATE, lw=1.2, ls="--")
    ax.text(len(keys) - 0.45, cr + 1.0, "matched-control rate", ha="right",
            fontsize=7.5, color=SLATE)
    ax.set_xticks(range(len(keys)))
    ax.set_xticklabels([labels[k] for k in keys], fontsize=9, color=INKD)
    _style(ax, title="Corroboration rate by implicated channel (pooled)",
           ylabel="% of imaged windows with cover")
    ax.set_ylim(0, max(ax.get_ylim()[1], 30))
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close(fig); print(f"  wrote {Path(out_png).name}", flush=True)


def run(dives=("J1754", "J1756"), root=ROOT, out_dir=PAPER) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    opts = {p: {} for p in PADS}; ctls = {p: {} for p in PADS}
    for dive in dives:
        win, fr = _load(dive, root)
        for pad in PADS:
            o = window_optics(dive, win, fr, pad)
            opts[pad][dive] = o
            ctls[pad][dive] = control_spans(dive, win, fr, o, pad)
        o = opts[0.0][dive]
        print(f"{dive}: {len(o)} transit windows, {int((o.n_frames > 0).sum())} "
              f"imaged, {int((o.n_frames == 0).sum())} unimaged; "
              f"{len(fr)} transit frames", flush=True)

    summary = dict(generated=datetime.now(timezone.utc).isoformat(),
                   cover_threshold=COVER_T, n_controls_per_window=N_CONTROLS,
                   pad_variant_s=PAD_VARIANT, dives={}, pooled={},
                   note="controls are 20 matched pseudo-windows per imaged real "
                        "window (not independent draws); Fisher p treats them as "
                        "independent, so the control margin is anti-conservative")
    for pad in PADS:
        key = f"pad{int(pad)}"
        for dive in dives:
            summary["dives"].setdefault(dive, {})[key] = corroboration(
                opts[pad][dive], ctls[pad][dive])
        summary["pooled"][key] = corroboration(
            pd.concat(opts[pad].values(), ignore_index=True),
            pd.concat(ctls[pad].values(), ignore_index=True))
    opt0 = pd.concat(opts[0.0].values(), ignore_index=True)
    ctl0 = pd.concat(ctls[0.0].values(), ignore_index=True)
    summary["family_breakdown"] = breakdown(opt0, ctl0, "family")
    summary["tier_breakdown"] = breakdown(opt0, ctl0, "tier")

    csv = opt0.copy()
    opt10 = pd.concat(opts[PAD_VARIANT].values(), ignore_index=True)
    for c in ("n_frames", "n_cover", "any_cover"):
        csv[f"{c}_pad10"] = opt10[c].to_numpy()
    csv.to_csv(out_dir / "angle5_window_optics.csv", index=False)

    blocks = {d: summary["dives"][d]["pad0"] for d in dives}
    blocks["pooled"] = summary["pooled"]["pad0"]
    make_figure(blocks, summary["family_breakdown"], out_dir / "angle5_fig.png")

    p = summary["pooled"]["pad0"]
    headline = (f"imaged transit windows show bright cover at "
                f"{100 * p['window_rate']:.0f}% ({p['window_cover']}/{p['n_imaged']})"
                f" vs {100 * p['control_rate']:.1f}% in matched controls — "
                f"{p['ratio']:.1f}x, p={p['fisher_p']:.2g}")
    if p["n_imaged"] < 30:
        headline += (f"; CAUTION: only {p['n_imaged']} imaged transit windows "
                     "pooled — too few for a strong claim")
    summary["headline"] = headline
    (out_dir / "angle5_summary.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out_dir / 'angle5_window_optics.csv'}")
    print(f"wrote {out_dir / 'angle5_summary.json'}")
    print(headline)
    return summary


if __name__ == "__main__":
    run(tuple(sys.argv[1:]) or ("J1754", "J1756"))
