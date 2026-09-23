#!/usr/bin/env python3
"""Megafauna occurrence table: every detection joined to nav, sensors, anomalies.

One row per kept detection (excluded classes dropped): what was seen, when,
where the vehicle was, what every gas/CTD channel read at that instant, and
whether the detectors called a gas anomaly there (window id, tier, class,
channels, station/transit context).

Sources, all already on disk per workspace:
- survey/fauna/fathomnet_detections.csv       (frame detections, re-bucketed)
- survey/photogrammetry manifests              (frame -> unix_time/UTM, via
                                                fathomnet_detect.load_frame_nav)
- inputs/interp_full.csv                       (1 Hz nav + sensor channels)
- survey/anomaly/window_context.csv            (anomaly windows + context)

Sensor values are merge_asof'd to the frame time (nearest, <= TOL_S apart).
Sensor time delays are honoured upstream when interp_full is built; they are
currently all zero, so no extra shift is applied here.

Qt-free.  build(workspace_dir) -> path to survey/fauna/occurrences.csv.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from fathomnet_detect import load_frame_nav

TOL_S = 5.0                      # max sensor<->frame time mismatch
SENSORS = ("CH4 Concentration", "CO2 Concentration", "O2 Concentration",
           "Temperature", "Salinity")
WINDOW_COLS = ("window_id", "confidence_tier", "anomaly_class", "channels",
               "context", "median_speed", "site_id")


def _load_windows(ws: Path) -> pd.DataFrame:
    win = pd.read_csv(ws / "survey" / "anomaly" / "window_context.csv")
    win["t0"] = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp)
    win["t1"] = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp)
    return win


def _window_at(win: pd.DataFrame, t: np.ndarray) -> pd.DataFrame:
    """For each time, the containing anomaly window (highest evidence_score on
    overlap), or blanks.  Vectorised per window: windows are few, times many."""
    cols = {"in_anomaly_window": np.zeros(len(t), bool)}
    cols |= {c: np.full(len(t), None, dtype=object) for c in WINDOW_COLS}
    score = np.full(len(t), -np.inf)
    ev = win.evidence_score if "evidence_score" in win else pd.Series(0, index=win.index)
    for i, w in win.iterrows():
        hit = (t >= w.t0) & (t <= w.t1) & (float(ev[i]) > score)
        if not hit.any():
            continue
        score[hit] = float(ev[i])
        cols["in_anomaly_window"][hit] = True
        for c in WINDOW_COLS:
            cols[c][hit] = w[c]
    return pd.DataFrame(cols)


def build(workspace_dir, det_csv=None, out_csv=None, log=print) -> str:
    ws = Path(workspace_dir)
    det_csv = Path(det_csv or ws / "survey" / "fauna" / "fathomnet_detections.csv")
    det = pd.read_csv(det_csv)
    det["excluded"] = det.excluded.fillna("")
    det = det[det.excluded == ""].drop(columns=["excluded"])

    nav = load_frame_nav(ws)
    det = det.join(nav, on="fn", how="inner").reset_index(drop=True)
    det = det.sort_values("unix_time", kind="stable").reset_index(drop=True)

    ip = pd.read_csv(ws / "inputs" / "interp_full.csv",
                     usecols=lambda c: c in ("unix_time", "lat", "lon",
                                             "heading", "water_depth") + SENSORS
                     ).sort_values("unix_time")
    det = pd.merge_asof(det, ip, on="unix_time", direction="nearest",
                        tolerance=TOL_S)

    det = pd.concat([det, _window_at(_load_windows(ws),
                                     det.unix_time.to_numpy(float))], axis=1)

    det.insert(0, "dive", ws.name.split("_")[0])
    det.insert(2, "timestamp_iso",
               pd.to_datetime(det.unix_time, unit="s", utc=True)
                 .dt.strftime("%Y-%m-%dT%H:%M:%SZ"))
    for c in ("conf", "easting", "northing"):
        det[c] = det[c].round(4)
    for c in SENSORS:
        if c in det:
            det[c] = det[c].round(5)

    out_csv = Path(out_csv or ws / "survey" / "fauna" / "occurrences.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    det.to_csv(out_csv, index=False)
    n_win = int(det.in_anomaly_window.sum())
    log(f"[occurrences] {ws.name}: {len(det)} occurrences "
        f"({n_win} inside anomaly windows, {det.fn.nunique()} frames) -> {out_csv}")
    return str(out_csv)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--dives", nargs="*", default=["J1754", "J1756"])
    ap.add_argument("--root", default="/mnt/f/EPR_2026_PROCESSED")
    a = ap.parse_args(argv)
    for dv in a.dives:
        build(Path(a.root) / f"{dv}_down.eprproj")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
