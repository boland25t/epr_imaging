#!/usr/bin/env python3
"""Render the imagery-x-chemistry analysis figures into the bundle
(<workspace>/survey/analysis_figs/) plus analysis_stats.json for the survey
report.

Framing: the relationship between altitude-corrected seafloor BRIGHTNESS and
methane anomalies, on transit-only frames.  Two complementary tests:
  boolean    — is a frame inside a detector anomaly window?  (Concentration at
               the sensor depends on altitude — the same seep reads stronger
               when flown lower — so the detector's baselined windows are the
               altitude-fair target.)
  continuous — in-situ concentration at the frame, cross-checked inside a
               fixed altitude band.
No biological interpretation is asserted anywhere.
"""
from __future__ import annotations
import glob, json, sys
from pathlib import Path

import numpy as np, pandas as pd, cv2
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from frame_color_analysis import correct, ALT_REF, load_frame_manifest

INK, MUTED, DIM, LINE = "#16222b", "#5a6b78", "#8a97a3", "#cdd6df"
TEAL, PURPLE, RED = "#0e7c86", "#7a5cc9", "#c1272d"
COVER_T = 0.02          # "bright-pixel cover" threshold (fraction of frame)


def _style(ax, title=None, xlabel=None, ylabel=None):
    if title: ax.set_title(title, fontsize=11, color=INK)
    if xlabel: ax.set_xlabel(xlabel, fontsize=9.5, color=MUTED)
    if ylabel: ax.set_ylabel(ylabel, fontsize=9.5, color=MUTED)
    ax.tick_params(colors=DIM, labelsize=8)
    for s in ("top", "right"): ax.spines[s].set_visible(False)
    for s in ("left", "bottom"): ax.spines[s].set_color(LINE)


def _save(fig, path, fc="white"):
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor=fc)
    plt.close(fig); print(f"  wrote {Path(path).name}", flush=True)


def _window_membership(d, win, mask):
    w = win[mask]
    ws = pd.to_datetime(w.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    we = pd.to_datetime(w.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    return np.array([bool(np.any((t >= ws) & (t <= we))) for t in d.t])


WIN_FAMILIES = [("co2", "CO2", "CO$_2$"), ("ch4", "CH4", "CH$_4$"),
                ("o2", "O2", "O$_2$"), ("temp", "Temperature", "Temp"),
                ("any", None, "Any channel")]


def fig_bool(d, out):
    """Left: window rate (CH4 vs any-channel) by scene-brightness quintile.
    Right: in-window rates for bright-cover vs bare frames, per window family."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 3.7), dpi=150,
                             gridspec_kw={"width_ratios": [1, 1.35]})
    q = pd.qcut(d.bright, 5, labels=False)
    ax = axes[0]
    w = 0.38
    for j, (col, lab, iw) in enumerate(((TEAL, "CH$_4$ windows", d.in_ch4),
                                        ("#9aa7b3", "any channel", d.in_any))):
        rates = [iw[q == i].mean() * 100 for i in range(5)]
        ax.bar(np.arange(5) + (j - 0.5) * w, rates, w, color=col, label=lab)
    ax.set_xticks(range(5))
    ax.set_xticklabels(["dimmest", "", "middle", "", "brightest"], fontsize=8.5, color=MUTED)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    _style(ax, title="Window rate vs scene brightness",
           xlabel="scene-brightness quintile", ylabel="% of frames inside a window")
    hp = d.white > COVER_T
    ax = axes[1]
    x = np.arange(len(WIN_FAMILIES))
    for j, (lab, col, sel) in enumerate((("bare seafloor", DIM, ~hp),
                                         ("bright-pixel cover >2%", PURPLE, hp))):
        vals = [d[f"in_{k}"][sel].mean() * 100 for k, *_ in WIN_FAMILIES]
        ax.bar(x + (j - 0.5) * 0.34, vals, 0.34, color=col, label=lab)
        for xi, v in zip(x + (j - 0.5) * 0.34, vals):
            ax.text(xi, v + 0.9, f"{v:.0f}", ha="center", fontsize=7.5, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels([p for *_, p in WIN_FAMILIES], fontsize=8.5, color=INK)
    ax.legend(fontsize=8, frameon=False, loc="upper left")
    _style(ax, title="In-window rate by window family",
           ylabel="% of frames inside a window")
    fig.tight_layout(); _save(fig, out)
    return [float(d.in_any[q == i].mean() * 100) for i in range(5)]


def fig_scatters(d, out, band_rhos):
    """Brightness vs each gas concentration, one panel per channel."""
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 3, figsize=(10.6, 3.5), dpi=150)
    for ax, (g, lab, log, col) in zip(axes, (("CH4", "CH$_4$", True, PURPLE),
                                             ("CO2", "CO$_2$", False, TEAL),
                                             ("O2", "O$_2$", False, RED))):
        ax.scatter(d.bright, d[g].clip(1) if log else d[g], s=7, color=col,
                   alpha=0.3, lw=0)
        if log: ax.set_yscale("log")
        ax.text(0.97, 0.05, f"band ρ = {band_rhos.get(g, 0):+.2f}",
                transform=ax.transAxes, ha="right", fontsize=8.5, color=MUTED)
        _style(ax, title=f"{lab} vs brightness",
               xlabel="corrected scene brightness",
               ylabel=f"{lab} (sensor units{', log' if log else ''})")
    fig.tight_layout(); _save(fig, out)


def fig_map(d, B, out):
    trk = np.array(json.load(open(f"{B}/survey/nav_trackline/trackline.geojson"))
                   ["features"][0]["geometry"]["coordinates"])
    sites_gj = Path(B) / "survey" / "anomaly" / "anomalous_sites_utm.geojson"
    sites = json.load(open(sites_gj)) if sites_gj.is_file() else {"features": []}
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(8.6, 8.2), dpi=150)
    fig.patch.set_facecolor("#0e1620"); ax.set_facecolor("#0e1620")
    ax.plot(trk[:, 0], trk[:, 1], color="#3a4b5c", lw=0.6, zorder=1)
    sc = ax.scatter(d.E, d.N, s=10, c=d.bright, cmap="cividis", alpha=0.9,
                    lw=0, zorder=3)
    hp = d[d.white > COVER_T]
    ax.scatter(hp.E, hp.N, s=52, facecolor="none", edgecolor="#f2c94c", lw=0.9, zorder=4)
    for f in sites["features"]:
        x, y = f["geometry"]["coordinates"]
        ax.plot(x, y, marker="o", ms=9, mfc="none", mec="#e8eef3", mew=1.1, zorder=5)
    cb = fig.colorbar(sc, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("corrected scene brightness", fontsize=9, color="#9fb0bd")
    cb.ax.tick_params(colors="#71828f", labelsize=7)
    x0, y0 = np.nanmin(d.E) + 8, np.nanmin(d.N) + 10
    ax.plot([x0, x0 + 50], [y0, y0], color="w", lw=3)
    ax.text(x0 + 25, y0 + 7, "50 m", color="w", ha="center", fontsize=9)
    leg = [Line2D([0], [0], marker="o", ls="", mfc="#8f9c48", mec="none", ms=5,
                  label="transit frame (colour = brightness)"),
           Line2D([0], [0], marker="o", ls="", mfc="none", mec="#f2c94c", ms=8,
                  label="bright-pixel cover >2%"),
           Line2D([0], [0], marker="o", ls="", mfc="none", mec="#e8eef3", ms=8,
                  label="ranked anomaly site")]
    ax.legend(handles=leg, loc="upper right", fontsize=8, framealpha=0.9,
              facecolor="#16222e", edgecolor="#2a3846", labelcolor="#dbe4ec")
    ax.set_aspect("equal"); ax.tick_params(colors="#5c6b7a", labelsize=7)
    for s in ax.spines.values(): s.set_color("#2a3846")
    _save(fig, out, fc="#0e1620")


def fig_attenuation(B, out):
    mf = load_frame_manifest(B).iloc[::30]
    from concurrent.futures import ThreadPoolExecutor

    def work(row):
        im = cv2.imread(row.fn, cv2.IMREAD_REDUCED_COLOR_8)
        if im is None: return None
        return (row.alt, im[:, :, 2].mean(), im[:, :, 1].mean(), im[:, :, 0].mean())
    with ThreadPoolExecutor(max_workers=10) as ex:
        rows = [r for r in ex.map(work, mf.itertuples(index=False)) if r is not None]
    a = pd.DataFrame(rows, columns=["alt", "R", "G", "B"]).dropna()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, ax = plt.subplots(figsize=(7.2, 4.2), dpi=150)
    slopes = {}
    for ch, c in (("R", RED), ("G", "#2e8b57"), ("B", "#2a6fdb")):
        y = np.log(a[ch].clip(1))
        ax.scatter(a.alt, y, s=10, color=c, alpha=0.4, lw=0)
        m, b = np.polyfit(a.alt, y, 1); slopes[ch] = float(m)
        xs = np.linspace(a.alt.min(), a.alt.max(), 10)
        ax.plot(xs, m * xs + b, color=c, lw=2, label=f"{ch}:  {m:+.2f} / m")
    ax.legend(fontsize=9.5, frameon=False, title="attenuation slope", title_fontsize=9)
    _style(ax, title="Colour-channel attenuation with altitude",
           xlabel="vehicle altitude above seafloor (m)",
           ylabel="ln(mean channel value)")
    _save(fig, out)
    return slopes, len(a)


def fig_ccdemo(d, out):
    cand = d[((d.alt - ALT_REF).abs() < 0.4) & (d.white < 0.01)].copy()
    row = cand.iloc[(cand.bright - cand.bright.median()).abs().argsort().iloc[0]]
    im = cv2.resize(cv2.imread(row.fn, cv2.IMREAD_REDUCED_COLOR_4), (720, 405))
    cor = correct(im, row.alt)
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.1), dpi=150)
    for ax, img, t in zip(axes, (im, cor),
                          ("Raw frame — water-column blue shift",
                           f"Corrected — altitude-normalised ({row.alt:.1f} m) + white balance")):
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(t, fontsize=9.5, color=INK)
        for s in ax.spines.values(): s.set_color(LINE)
    fig.tight_layout(); _save(fig, out)


def fig_examples(d, out):
    top = d.iloc[d.white.argmax()]
    pool = d[(d.white < 0.001) & ((d.alt - ALT_REF).abs() < 1)]
    typ = pool.iloc[(pool.CH4 - pool.CH4.median()).abs().argsort().iloc[0]]
    base_ch4 = d[d.white <= COVER_T].CH4.median()
    plt.rcParams["font.family"] = "DejaVu Sans"
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 3.35), dpi=150)
    for ax, row, tt in zip(
            axes, (typ, top),
            ("Typical transit frame — no bright cover, CH$_4$ ≈ background",
             f"Highest bright-cover frame — {top.white*100:.0f}% cover, "
             f"CH$_4$ {top.CH4/base_ch4:.1f}× background")):
        im = cv2.resize(cv2.imread(row.fn, cv2.IMREAD_REDUCED_COLOR_4), (720, 405))
        ax.imshow(cv2.cvtColor(correct(im, row.alt), cv2.COLOR_BGR2RGB))
        ax.set_xticks([]); ax.set_yticks([]); ax.set_title(tt, fontsize=9.5, color=INK)
        for s in ax.spines.values(): s.set_color(LINE)
    fig.tight_layout(); _save(fig, out)
    return {"top_cover": float(top.white), "top_ch4_fold": float(top.CH4 / base_ch4)}


def main(workspace):
    B = str(workspace)
    out = Path(B) / "survey" / "analysis_figs"; out.mkdir(parents=True, exist_ok=True)
    d = pd.read_csv(f"{B}/survey/frame_color_metrics.csv")
    win_csv = Path(B) / "survey" / "anomaly" / "anomaly_windows_all.csv"
    has_anom = win_csv.is_file()
    if has_anom:
        win = pd.read_csv(win_csv)
        for key, chan, _ in WIN_FAMILIES:
            mask = (win.index >= 0) if chan is None else win.channels.fillna("").str.contains(chan)
            d[f"in_{key}"] = _window_membership(d, win, mask)
        print(f"{len(d)} transit-frame records; any-channel windows cover "
              f"{d.in_any.mean()*100:.0f}% of frames", flush=True)
    else:
        print(f"{len(d)} transit-frame records; NO anomaly products yet — "
              "window-membership figures/stats skipped", flush=True)
    from scipy.stats import spearmanr, mannwhitneyu, fisher_exact
    hp = d.white > COVER_T
    band = d[(d.alt >= 4) & (d.alt <= 6.5)]
    hpb = band.white > COVER_T

    def bool_block(iw):
        tab = [[int((hp & iw).sum()), int((hp & ~iw).sum())],
               [int((~hp & iw).sum()), int((~hp & ~iw).sum())]]
        orr, p = fisher_exact(tab, alternative="greater")
        u, pm = mannwhitneyu(d[iw].bright, d[~iw].bright, alternative="two-sided")
        return dict(rate_cover=float(iw[hp].mean()), rate_bare=float(iw[~hp].mean()),
                    odds_ratio=float(orr), p=float(p),
                    delta_bright=float(2 * u / (iw.sum() * (~iw).sum()) - 1),
                    n_in=int(iw.sum()))
    band_rhos = {g: float(spearmanr(band.bright, band[g])[0]) for g in ("CH4", "CO2", "O2")}
    stats = {
        "n_frames": int(len(d)), "n_cover": int(hp.sum()),
        "cover_frac": float(hp.mean()),
        "spearman": {m: {g: dict(zip(("rho", "p"), map(float, spearmanr(d[m], d[g]))))
                     for g in ("CO2", "CH4", "O2")}
                     for m in ("bright", "white", "matidx")},
        "rho_bright_alt": float(spearmanr(d.bright, d.alt)[0]),
        "rho_white_alt": float(spearmanr(d.white, d.alt)[0]),
        "bool": ({k: bool_block(d[f"in_{k}"].to_numpy()) for k, *_ in WIN_FAMILIES}
                 if has_anom else {}),
        "conc": {"cover_ch4_fold": float(d[hp].CH4.median() / d[~hp].CH4.median()),
                 "p": float(mannwhitneyu(d[hp].CH4, d[~hp].CH4, alternative="greater")[1]),
                 "band": dict(n=int(len(band)),
                              rho_bright_ch4=band_rhos["CH4"],
                              rho_bright_co2=band_rhos["CO2"],
                              rho_bright_o2=band_rhos["O2"],
                              cover_fold=float(band[hpb].CH4.median() / band[~hpb].CH4.median()),
                              cover_fold_co2=float(band[hpb].CO2.median() / band[~hpb].CO2.median()),
                              p=float(mannwhitneyu(band[hpb].CH4, band[~hpb].CH4,
                                                   alternative="greater")[1]))},
    }
    sites_gj = Path(B) / "survey" / "anomaly" / "anomalous_sites_utm.geojson"
    if has_anom and sites_gj.is_file():
        spts = np.array([f["geometry"]["coordinates"]
                         for f in json.load(open(sites_gj))["features"]])
        if len(spts):
            dmin = np.min(np.hypot(spts[:, 0, None] - d.E.to_numpy()[None, :],
                                   spts[:, 1, None] - d.N.to_numpy()[None, :]), axis=1)
            stats["sites_total"] = int(len(spts))
            stats["sites_outside"] = int((dmin > 12).sum())
    meta = Path(B) / "survey" / "frame_color_meta.json"
    if meta.exists():
        stats["sampling"] = json.loads(meta.read_text())
        # adopt the dive's fitted correction so demo/example frames match the metrics
        from frame_color_analysis import C as _C
        _C.update(stats["sampling"].get("coeffs", {}))
    if has_anom:
        stats["q_rates_any"] = fig_bool(d, out / "bool_brightness.png")
    fig_scatters(d, out / "brightness_gases.png", band_rhos)
    fig_map(d, B, out / "brightness_map.png")
    if has_anom:
        import region_portrait
        region_portrait.build_portrait(B, out_path=str(out / "region_portrait_core.png"))
        print("  wrote region_portrait_core.png", flush=True)
    slopes, n_att = fig_attenuation(B, out / "attenuation.png")
    stats["attenuation"] = {"slopes": slopes, "n": n_att}
    fig_ccdemo(d, out / "colorcorrection_demo.png")
    stats.update(fig_examples(d, out / "example_frames.png"))
    if has_anom:
        import geomorph_correlation
        g = geomorph_correlation.run(B, fig_path=str(out / "geomorph.png"))
        g.pop("fig_b64", None)
        stats["geomorph"] = g
    # remove stale mat-era figures so the report can't embed them
    stale_figs = ["mat_gas_foldchange.png", "whiteness_ch4.png", "mat_map.png",
                  "brightness_ch4.png"]
    if not has_anom:
        stale_figs += ["bool_brightness.png", "region_portrait_core.png", "geomorph.png"]
    for stale in stale_figs:
        try: (out / stale).unlink()
        except OSError: pass
    (out / "analysis_stats.json").write_text(json.dumps(stats, indent=1))
    print("FIGS_DONE", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else ".")
