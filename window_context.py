#!/usr/bin/env python3
"""Classify anomaly windows by vehicle behaviour: deliberate on-station
measurement vs happenstance transit encounter.

'Spikes' in the gas record mix two populations: manual measurements (the ROV
parked with the sensor in a vent — dive-plan targeted, not a detection) and
anomalies encountered while the vehicle was genuinely surveying.  Vehicle
speed separates them: below V_STATION m/s the vehicle is station-keeping.

For each dive's survey/anomaly/anomaly_windows_all.csv this computes the
median smoothed vehicle speed across every window's time span and tags it
  station  — median speed < V_STATION (deliberate measurement)
  transit  — otherwise (happenstance encounter; the detection population)
plus the moving fraction of the window and its channel list, writing
survey/anomaly/window_context.csv and returning the DataFrame.

summarize_dives(dives) aggregates the per-dive station/transit split by tier
and channel family for the cross-dive methods comparison.

Qt-free.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
V_STATION = 0.08          # m/s — same threshold as nav segmentation
SMOOTH_N = 15             # rolling-median window (samples @1 Hz), as nav_segments


def _speed_series(workspace_dir):
    ip = pd.read_csv(Path(workspace_dir) / "inputs" / "interp_full.csv",
                     usecols=["unix_time", "easting", "northing"]).dropna()
    ip = ip.sort_values("unix_time")
    t = ip.unix_time.to_numpy()
    dt = np.diff(t)
    spd = np.zeros(len(t))
    spd[1:] = np.divide(np.hypot(np.diff(ip.easting.to_numpy()),
                                 np.diff(ip.northing.to_numpy())), dt,
                        out=np.zeros(len(dt)), where=dt > 0)
    sm = pd.Series(spd).rolling(SMOOTH_N, center=True, min_periods=1).median()
    return t, sm.to_numpy()


def classify_windows(workspace_dir, v_station=V_STATION) -> pd.DataFrame:
    ws = Path(workspace_dir)
    win = pd.read_csv(ws / "survey" / "anomaly" / "anomaly_windows_all.csv")
    t, spd = _speed_series(ws)
    t0 = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    t1 = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    med = np.empty(len(win))
    moving_frac = np.empty(len(win))
    for i, (a, b) in enumerate(zip(t0, t1)):
        m = (t >= a) & (t <= b)
        s = spd[m] if m.any() else np.interp([(a + b) / 2], t, spd)
        med[i] = float(np.median(s))
        moving_frac[i] = float((s >= v_station).mean())
    win["median_speed"] = med.round(4)
    win["moving_frac"] = moving_frac.round(3)
    win["context"] = np.where(med < v_station, "station", "transit")
    win["duration_s"] = (t1 - t0).round(1)
    out = ws / "survey" / "anomaly" / "window_context.csv"
    win.to_csv(out, index=False)
    return win


def summarize_dives(dives, root=ROOT) -> pd.DataFrame:
    rows = []
    for dive in dives:
        ws = root / f"{dive}_down.eprproj"
        try:
            w = classify_windows(ws)
        except FileNotFoundError as e:
            print(f"{dive}: skipped ({e})")
            continue
        for ctx in ("station", "transit"):
            sub = w[w.context == ctx]
            tier = sub["confidence_tier"] if "confidence_tier" in sub.columns else sub.get("confidence")
            row = dict(dive=dive, context=ctx, windows=len(sub),
                       high=int((tier == "HIGH").sum()),
                       moderate=int((tier == "MODERATE").sum()),
                       screen=int((tier == "SCREEN").sum()),
                       median_duration_s=float(sub.duration_s.median()) if len(sub) else 0.0)
            for chan in ("CO2", "CH4", "O2", "Temperature"):
                row[chan.lower()] = int(sub.channels.fillna("").str.contains(chan).sum())
            rows.append(row)
    df = pd.DataFrame(rows)
    out = root / "cross_dive_window_context.csv"
    df.to_csv(out, index=False)
    print(f"wrote {out}")
    return df


if __name__ == "__main__":
    dives = sys.argv[1:] or ["J1754", "J1755", "J1756", "J1758", "J1759", "J1760", "J1761"]
    print(summarize_dives(dives).to_string(index=False))
