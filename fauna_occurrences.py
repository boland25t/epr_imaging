#!/usr/bin/env python3
"""Megafauna frame-detection table: every detection joined to nav, sensors, anomalies.

One row per kept FRAME DETECTION (excluded classes dropped) -- not per
individual: frames overlap heavily (~0.25 m spacing), so the same animal is
detected in several consecutive frames.  The file keeps its historical name
(occurrences.csv) but its first line is a '#' comment saying so; read it with
``pandas.read_csv(path, comment="#")``.  Columns lead with the bucket (the only
level that was checked); the model's class name is ``model_label_unaudited``.
Sensor columns carry their units from workspace.json (e.g. "pCH4 (uatm)" --
CO2/CH4 are partial pressures).  ``depth`` is vehicle depth, m, NEGATIVE down;
``water_depth`` is positive down.  Column notes + provenance are written to
occurrences.meta.json.

What was seen, when,
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
import json
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


HEADER_NOTE = ("# FRAME DETECTIONS, NOT INDIVIDUALS: overlapping frames re-detect "
               "the same animal. model_label_unaudited is zero-shot FathomNet/MBARI "
               "output (trust bucket only). depth = vehicle depth m NEGATIVE down; "
               "water_depth positive down. Units in column names. See "
               "occurrences.meta.json. Read with pandas.read_csv(path, comment='#').")


def _sensor_columns(ws: Path) -> dict:
    """{interp_full column: unit-bearing output name} for every sensor channel
    workspace.json declares (falls back to the historical five)."""
    from reporting_common import channel_label, sensor_units
    units = sensor_units(ws)
    names = list(units) or list(SENSORS)
    return {n: channel_label(n, units.get(n, "")) for n in names}


def build(workspace_dir, det_csv=None, out_csv=None, log=print) -> str:
    ws = Path(workspace_dir)
    det_csv = Path(det_csv or ws / "survey" / "fauna" / "fathomnet_detections.csv")
    det = pd.read_csv(det_csv)
    det["excluded"] = det.excluded.fillna("")
    det = det[det.excluded == ""].drop(columns=["excluded"])

    nav = load_frame_nav(ws)
    det = det.join(nav, on="fn", how="inner").reset_index(drop=True)
    det = det.sort_values("unix_time", kind="stable").reset_index(drop=True)

    sensors = _sensor_columns(ws)
    # heading comes from the frame manifest (load_frame_nav); pulling it from
    # interp_full as well produced heading_x/heading_y merge artefacts.
    wanted = ("unix_time", "lat", "lon", "water_depth") + tuple(sensors)
    ip = pd.read_csv(ws / "inputs" / "interp_full.csv",
                     usecols=lambda c: c in wanted).sort_values("unix_time")
    ip = ip.drop(columns=[c for c in ip.columns
                          if c != "unix_time" and c in det.columns])
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
    for c in sensors:
        if c in det:
            det[c] = det[c].round(5)
    det = det.rename(columns={**sensors, "cls": "model_label_unaudited"})
    # bucket first, the unaudited model label after it (review 02 P0-3)
    lead = [c for c in ("dive", "fn", "timestamp_iso", "bucket",
                        "model_label_unaudited", "conf") if c in det.columns]
    det = det[lead + [c for c in det.columns if c not in lead]]

    out_csv = Path(out_csv or ws / "survey" / "fauna" / "occurrences.csv")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as fh:
        fh.write(HEADER_NOTE + "\n")
        det.to_csv(fh, index=False)
    _write_meta(ws, out_csv, det, sensors, det_csv)
    n_win = int(det.in_anomaly_window.sum())
    log(f"[occurrences] {ws.name}: {len(det)} frame detections (not individuals; "
        f"{n_win} inside anomaly windows, {det.fn.nunique()} frames) -> {out_csv}")
    return str(out_csv)


def _write_meta(ws: Path, out_csv: Path, det: pd.DataFrame, sensors: dict,
                det_csv: Path) -> None:
    from reporting_common import provenance, write_json
    model = None
    try:
        model = json.loads((out_csv.parent / "fauna_provenance.json").read_text()
                           ).get("model")
    except (OSError, ValueError):
        pass
    write_json(out_csv.with_name("occurrences.meta.json"), {
        "product": "fauna_frame_detections (occurrences.csv)",
        "rows": int(len(det)),
        "row_unit": "one kept frame DETECTION; not an individual organism "
                    "(overlapping frames re-detect the same animal)",
        "header_comment_line": HEADER_NOTE,
        "columns": {
            "bucket": "coarse morphology bucket (the only level checked)",
            "model_label_unaudited": "zero-shot FathomNet/MBARI class name; "
                                     "NOT an identification",
            "conf": "detector confidence 0-1",
            "timestamp_iso": "frame time, UTC",
            "depth": "vehicle depth from the frame manifest, m, NEGATIVE down",
            "water_depth": "vehicle depth from interp_full, m, POSITIVE down",
            "alt": "altitude above seafloor, m",
            "easting/northing": "frame-centre fix, EPSG:32613 (UTM 13N), m, "
                                "~+/-3 m frame-level localisation",
            **{v: f"sensor channel '{k}' at the frame time (nearest 1 Hz "
                  f"sample within {TOL_S:g} s)" for k, v in sensors.items()},
        },
        "provenance": provenance(
            [det_csv, ws / "inputs" / "interp_full.csv",
             ws / "survey" / "anomaly" / "window_context.csv"], model=model),
    })


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
