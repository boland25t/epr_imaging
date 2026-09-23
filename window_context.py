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
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/mnt/f/EPR_2026_PROCESSED")
V_STATION = 0.08          # m/s — same threshold as nav segmentation
SMOOTH_N = 15             # rolling-median window (samples @1 Hz), as nav_segments


#: A leading/trailing run of identical positions at least this long is the old
#: interp build's edge-hold padding, not navigation (see product_catalog).
HELD_EDGE_MIN_ROWS = 60


def _iso_to_unix(value):
    if not value:
        return None
    try:
        ts = pd.Timestamp(str(value).strip())
    except (ValueError, TypeError):
        return None
    if ts.tzinfo is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return (ts - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)


def nav_span(workspace_dir):
    """(t0, t1) unix span in which the dive has REAL positions, or None.

    From workspace.json's latitude/longitude source spans (the positional
    renav — not navigation_file.start/end, which include the altimeter).
    """
    import json
    try:
        data = json.loads((Path(workspace_dir) / "workspace.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    nav = data.get("navigation_file") or {}
    starts, ends = [], []
    for key in ("latitude_source", "longitude_source"):
        src = nav.get(key) or {}
        a, b = _iso_to_unix(src.get("start_time")), _iso_to_unix(src.get("end_time"))
        if a is None or b is None:
            return None
        starts.append(a)
        ends.append(b)
    if max(starts) >= min(ends):
        return None
    return float(max(starts)), float(min(ends))


def _trim_held_edges(ip: pd.DataFrame) -> pd.DataFrame:
    """Drop leading/trailing constant-position runs (edge-hold padding)."""
    if len(ip) < 2:
        return ip
    e, n = ip.easting.to_numpy(), ip.northing.to_numpy()
    same = (e[1:] == e[:-1]) & (n[1:] == n[:-1])
    i0 = 0
    while i0 < len(ip) - 1 and same[i0]:
        i0 += 1
    i1 = len(ip) - 1
    while i1 > 0 and same[i1 - 1]:
        i1 -= 1
    start = i0 if i0 >= HELD_EDGE_MIN_ROWS else 0
    stop = i1 + 1 if (len(ip) - 1 - i1) >= HELD_EDGE_MIN_ROWS else len(ip)
    return ip.iloc[start:max(start, stop)]


def _speed_series(workspace_dir):
    """(t, smoothed speed) over the rows that carry a REAL position.

    Rows outside the stored nav span, and edge-held rows of an interp table
    built before the nav-span fix, are excluded: a held position has speed 0
    and would classify any window there as 'station' (review 06 P1-2 #4).
    """
    ip = pd.read_csv(Path(workspace_dir) / "inputs" / "interp_full.csv",
                     usecols=["unix_time", "easting", "northing"]).dropna()
    ip = ip.sort_values("unix_time", kind="stable")
    span = nav_span(workspace_dir)
    if span is not None:
        inside = (ip.unix_time >= span[0] - 1.0) & (ip.unix_time <= span[1] + 1.0)
        if inside.any():
            ip = ip[inside]
    ip = _trim_held_edges(ip)
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
    med = np.full(len(win), np.nan)
    moving_frac = np.full(len(win), np.nan)
    # A window not fully inside the real-navigation span cannot be classified
    # by vehicle speed: it is "no_nav", never "station".
    if len(t):
        no_nav = (t0 < t[0] - 1.0) | (t1 > t[-1] + 1.0)
    else:
        no_nav = np.ones(len(win), dtype=bool)
    for i, (a, b) in enumerate(zip(t0, t1)):
        if no_nav[i]:
            continue
        m = (t >= a) & (t <= b)
        s = spd[m] if m.any() else np.interp([(a + b) / 2], t, spd)
        med[i] = float(np.median(s))
        moving_frac[i] = float((s >= v_station).mean())
    win["median_speed"] = np.round(med, 4)
    win["moving_frac"] = np.round(moving_frac, 3)
    win["context"] = np.where(no_nav, "no_nav",
                              np.where(med < v_station, "station", "transit"))
    win["duration_s"] = (t1 - t0).round(1)
    out = ws / "survey" / "anomaly" / "window_context.csv"
    tmp = out.with_name(f".{out.name}.{os.getpid()}.tmp")
    win.to_csv(tmp, index=False)
    os.replace(tmp, out)                  # readers never see a half-written table
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
