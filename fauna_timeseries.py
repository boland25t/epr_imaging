#!/usr/bin/env python3
"""Per-frame megafauna areal density and its time series.

Owns the footprint model and the density computation (previously ad-hoc —
critique_fauna.md MOD-6): appends area/density columns to
survey/fauna/fauna_density.csv and regenerates the time-series figure from
this committed code.

Footprint model: a 5312x2988 frame at ~1 mm/px near 5 m altitude spans
roughly 5.3 x 3.0 m, i.e. width ~ K_W * alt with K_W ~= 1.06 and aspect
2988/5312, giving area ~= AREA_K * alt^2.  MOD-1 of the critique measured
K_W ~= 0.86-0.88 from the chunk orthomosaics themselves (K_W_MEASURED); the
nominal 1.06 stays the DEFAULT (it matches every product delivered so far) and
the choice is stated on the figure and in the sidecar.  With 0.87 the
footprint shrinks by (0.87/1.06)^2 ~= 0.67, i.e. densities rise ~1.5x.

Altitude band (review 02 P0-2): densities are computed ONLY for frames whose
altitude lies inside ALT_BAND (3-8 m, the photogrammetry gate).  Below it the
footprint is tiny and the altimeter's 0.6 m dropout/floor value (ALT_SENTINEL)
inflates per-frame density; above it small fauna fall below detection.  Out
of band, every dens_* column is NaN and alt_valid is False; area_m2 is still
reported for every frame.  The column notes live in the sidecar
fauna_density.meta.json.

Qt-free.  build(workspace_dir) -> figure path.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

K_W = 1.06           # frame ground width / altitude (nominal, the default)
K_W_MEASURED = 0.87  # ortho-measured alternative (critique MOD-1), NOT the default
ASPECT = 2988 / 5312
AREA_K = K_W * K_W * ASPECT     # ~0.63 * alt^2 m^2 per frame
ALT_BAND = (3.0, 8.0)  # valid altitude band for density (m, inclusive)
ALT_SENTINEL = 0.6     # altimeter dropout / floor value: never a real altitude
GAP_S = 120          # inter-frame gap that breaks an imaged span
ROLL_N = 12          # rolling-median window (frames, ~1 min at 5 s cadence)
YCAP_Q = 0.99        # display cap quantile (disclosed on the figure)
BUCKETS = ("crustacean", "worm", "fish", "anemone", "unknown", "total")


def area_k(k_w: float = K_W) -> float:
    return k_w * k_w * ASPECT


def column_notes(k_w: float = K_W, alt_band=ALT_BAND) -> dict:
    lo, hi = alt_band
    return {
        "n_<bucket>": "kept frame DETECTIONS per frame (not individuals; "
                      "overlapping frames re-detect the same animal)",
        "alt": "vehicle altitude above seafloor, m (DPA altimeter)",
        "depth": "vehicle depth, m, NEGATIVE down (-2555 = 2555 m below surface)",
        "area_m2": f"frame footprint = {area_k(k_w):.3f} * alt^2 m^2 "
                   f"(K_W={k_w}, aspect {ASPECT:.4f}); reported for every frame",
        "alt_valid": f"True when {lo:g} <= alt <= {hi:g} m and alt != "
                     f"{ALT_SENTINEL:g} (altimeter sentinel)",
        "dens_<bucket>": f"n_<bucket> / area_m2 in detections per m^2, ONLY "
                         f"for alt_valid frames; NaN outside the {lo:g}-{hi:g} m "
                         f"band",
    }


def add_density_columns(ws: str, k_w: float = K_W, alt_band=ALT_BAND,
                        log=None) -> pd.DataFrame:
    """Append area_m2, alt_valid and per-bucket dens_* columns to
    fauna_density.csv (altitude-banded; see module docstring), and write the
    fauna_density.meta.json sidecar (column notes + provenance)."""
    path = Path(ws) / "survey" / "fauna" / "fauna_density.csv"
    d = pd.read_csv(path)
    lo, hi = alt_band
    alt = pd.to_numeric(d.alt, errors="coerce")
    d["area_m2"] = (area_k(k_w) * alt ** 2).round(3)
    valid = alt.between(lo, hi) & ~np.isclose(alt.fillna(-1), ALT_SENTINEL)
    d["alt_valid"] = valid
    for b in BUCKETS:
        d[f"dens_{b}"] = (d[f"n_{b}"] / d.area_m2).where(valid).round(4)
    d.to_csv(path, index=False)
    _write_sidecar(ws, d, k_w, alt_band)
    if log:
        log(f"[density] {int(valid.sum())}/{len(d)} frames in the "
            f"{lo:g}-{hi:g} m band get a density; the rest are NaN")
    return d


def _write_sidecar(ws, d, k_w, alt_band) -> None:
    from reporting_common import provenance, write_json
    fauna = Path(ws) / "survey" / "fauna"
    prov_src = fauna / "fauna_provenance.json"
    model = None
    try:
        import json
        model = json.loads(prov_src.read_text()).get("model")
    except (OSError, ValueError):
        pass
    alt = pd.to_numeric(d.alt, errors="coerce")
    write_json(fauna / "fauna_density.meta.json", {
        "product": "fauna_density",
        "columns": column_notes(k_w, alt_band),
        "footprint": {"K_W": k_w, "K_W_nominal": K_W,
                      "K_W_measured_alternative": K_W_MEASURED,
                      "aspect": ASPECT, "area_k": round(area_k(k_w), 4),
                      "note": "K_W=0.87 (ortho-measured) would raise densities "
                              "~1.5x; the nominal 1.06 is the stated default"},
        "alt_band_m": list(alt_band), "alt_sentinel_m": ALT_SENTINEL,
        "n_frames": int(len(d)), "n_alt_valid": int(d.alt_valid.sum()),
        "n_alt_sentinel": int(np.isclose(alt.fillna(-1), ALT_SENTINEL).sum()),
        "provenance": provenance(
            [fauna / "fathomnet_detections.csv", fauna / "fauna_density.csv"],
            model=model),
    })
def build(ws: str, log=print, k_w: float = K_W, alt_band=ALT_BAND) -> str:
    d = add_density_columns(ws, k_w, alt_band, log=log).dropna(subset=["unix_time", "alt"])
    d = d.sort_values("unix_time").reset_index(drop=True)
    t0 = d.unix_time.min()
    d["hrs"] = (d.unix_time - t0) / 3600
    gap = d.unix_time.diff().fillna(0) > GAP_S
    spans = d.groupby(gap.cumsum()).unix_time.agg(["min", "max"])

    def broken(series):
        r = series.rolling(ROLL_N, center=True, min_periods=3).median().copy()
        r[gap] = np.nan
        return r

    lo_b, hi_b = alt_band
    n_valid = int(d.alt_valid.sum()) if "alt_valid" in d else 0
    q = d.dens_total.quantile(YCAP_Q)
    ycap = max(q if np.isfinite(q) else 0.0, 0.5)
    win_p = Path(ws) / "survey" / "anomaly" / "window_context.csv"
    win = (pd.read_csv(win_p) if win_p.is_file()
           else pd.DataFrame(columns=["confidence_tier", "context",
                                      "start_time", "end_time"]))
    # the full window inventory is deliberately high-recall and blankets the
    # dive; shade only HIGH-tier transit windows so shading stays legible
    win = win[(win.confidence_tier == "HIGH") & (win.context == "transit")]
    wt0 = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    wt1 = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()

    fig, ax = plt.subplots(figsize=(13, 4.6), dpi=150)
    fig.patch.set_facecolor("#0e1620")
    ax.set_facecolor("#0e1620")
    for a, b in zip(wt0, wt1):                # shade only over imaged spans
        for _, s in spans.iterrows():
            lo, hi = max(a, s["min"]), min(b, s["max"])
            if hi > lo:
                ax.axvspan((lo - t0) / 3600, (hi - t0) / 3600,
                           color="#ff5d6e", alpha=0.22, lw=0)
    prev = None
    for _, s in spans.iterrows():             # grey no-imagery blocks
        if prev is not None and s["min"] - prev > GAP_S:
            ax.axvspan((prev - t0) / 3600, (s["min"] - t0) / 3600,
                       color="#1c2833", alpha=0.9, lw=0, zorder=0.5)
        prev = s["max"]
    ax.scatter(d.hrs, d.dens_total.clip(upper=ycap), s=3, color="#3d5666",
               alpha=0.55, lw=0)
    ax.plot(d.hrs, broken(d.dens_total), color="#6ec6e6", lw=1.7,
            label="all fauna (rolling median)")
    # bucket name only: the worm bucket mixes tubeworms, serpulids and
    # zero-shot calls such as Swima (review 02 P0-3), so no taxon is named
    ax.plot(d.hrs, broken(d.dens_worm), color="#e05c62", lw=1.5,
            label="worm bucket (rolling median; model buckets, taxa unaudited)")
    imaged_h = (spans["max"] - spans["min"]).sum() / 3600
    ax.set_xlim(d.hrs.min() - 0.1, d.hrs.max() + 0.1)
    ax.set_ylim(0, ycap * 1.04)
    t0_txt = pd.to_datetime(t0, unit="s", utc=True).strftime("%Y-%m-%d %H:%M:%SZ")
    ax.set_xlabel(f"hours since {t0_txt} (UTC)  ·  grey = no imagery; lines "
                  f"break across gaps and outside the altitude band",
                  color="#93a3af", fontsize=10)
    ax.set_ylabel("frame detections / m²", color="#93a3af", fontsize=10)
    dive = Path(ws).name.split("_")[0]
    ax.set_title(f"{dive} — per-frame megafauna detection density "
                 f"({imaged_h:.1f} h imaged; red = HIGH-tier transit windows)\n"
                 f"density only for {lo_b:g}–{hi_b:g} m altitude "
                 f"({n_valid:,}/{len(d):,} frames); footprint "
                 f"{area_k(k_w):.2f}·alt² (K_W={k_w:g}, nominal; ortho-measured "
                 f"{K_W_MEASURED:g} would give ~{(K_W / K_W_MEASURED) ** 2:.1f}× "
                 f"higher); y cap p99 = {ycap:.2f}/m²",
                 color="#dce4eb", fontsize=10)
    ax.tick_params(colors="#6b7b87", labelsize=8)
    for sp in ax.spines.values():
        sp.set_color("#263442")
    ax.legend(fontsize=8.5, framealpha=0.9, facecolor="#16222e",
              edgecolor="#2a3846", labelcolor="#dbe4ec")
    fig.tight_layout()
    out = Path(ws) / "survey" / "fauna" / "fauna_density_timeseries_v2.png"
    fig.savefig(out, dpi=150, facecolor="#0e1620", bbox_inches="tight")
    plt.close(fig)
    log(f"{dive}: {imaged_h:.1f} h imaged -> {out}")
    return str(out)


if __name__ == "__main__":
    for dv in (sys.argv[1:] or ["J1754", "J1756"]):
        build(f"/mnt/f/EPR_2026_PROCESSED/{dv}_down.eprproj")
