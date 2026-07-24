#!/usr/bin/env python3
"""Fuse per-channel MATLAB anomaly events into sites and video-review windows.

Inputs are the four GrapherMatrix run configurations, CTD1 T/S corroboration
flags, the J1754 interpolated sensor table, and the corrected raw navigation
file. Outputs are CSV/GeoJSON datasets and an annotated PDF report.
"""

from __future__ import annotations

import json
import math
import zipfile
from pathlib import Path

import matplotlib
# Force the non-interactive Agg backend BEFORE pyplot is imported.  This module
# only ever writes PDFs, but the GUI imports it and runs it on a worker thread;
# with PySide6 loaded matplotlib would otherwise auto-select the Qt backend and
# construct QObjects off the main thread, which aborts the process.  (qc_report
# and point_cloud_pipeline avoid pyplot entirely for the same reason.)
matplotlib.use("Agg", force=True)

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import find_peaks, peak_widths


REPO = Path(__file__).resolve().parent
EVENT_ROOT = REPO / "grapher_matrix_figs"
INTERP = REPO / "BioIntervals/BioIntervalsOnly/interp_full.csv"
RAW_NAV = REPO / "BioIntervals/BioIntervalsOnly/inputs/nav/J1754_renav_shift_25m_041deg_1hz.csv"
TS_RESULTS = REPO / "ts_analysis_results.mat"
OUT = REPO / "anomaly_site_catalog"


def configure(
    repo: Path | str | None = None,
    event_root: Path | str | None = None,
    interp: Path | str | None = None,
    raw_nav: Path | str | None = None,
    ts_results: Path | str | None = None,
    out: Path | str | None = None,
) -> dict[str, Path]:
    """Point the builder at a specific workspace.

    The pipeline functions read these as module globals, so the app rebinds them
    here rather than threading a config object through every function.  Any
    argument left as None keeps its current value; passing only ``repo`` re-derives
    the defaults relative to that repo.

    Returns the resolved paths, so a caller can log exactly what was used.
    """
    global REPO, EVENT_ROOT, INTERP, RAW_NAV, TS_RESULTS, OUT

    if repo is not None:
        REPO = Path(repo).resolve()
        # Re-derive every repo-relative default; explicit args below still win.
        EVENT_ROOT = REPO / "grapher_matrix_figs"
        TS_RESULTS = REPO / "ts_analysis_results.mat"
        OUT = REPO / "anomaly_site_catalog"
        INTERP = REPO / "BioIntervals/BioIntervalsOnly/interp_full.csv"
        RAW_NAV = (
            REPO / "BioIntervals/BioIntervalsOnly/inputs/nav"
                   "/J1754_renav_shift_25m_041deg_1hz.csv"
        )

    if event_root is not None:
        EVENT_ROOT = Path(event_root).resolve()
    if interp is not None:
        INTERP = Path(interp).resolve()
    if raw_nav is not None:
        RAW_NAV = Path(raw_nav).resolve()
    if ts_results is not None:
        TS_RESULTS = Path(ts_results).resolve()
    if out is not None:
        OUT = Path(out).resolve()

    return current_paths()


def current_paths() -> dict[str, Path]:
    """The paths the next run will use (for logging / preflight checks)."""
    return {
        "repo": REPO,
        "event_root": EVENT_ROOT,
        "interp": INTERP,
        "raw_nav": RAW_NAV,
        "ts_results": TS_RESULTS,
        "out": OUT,
    }

CONFIGS = ("masked", "nomask_log", "nomask_ceiling", "nomask_both")
MATRIX_CHANNELS = ("CO2", "CH4", "O2", "temp")
CHANNELS = ("CO2", "CH4", "O2", "temp", "salinity")
METHODS = ("D1_height", "D2_mass", "D3_peak", "D3_direct_peak", "D4_local", "TS_curve")
CHANNEL_PRETTY = {
    "CO2": "CO2", "CH4": "CH4", "O2": "O2",
    "temp": "Temperature", "salinity": "Salinity",
}
MERGE_GAP_SECONDS = 30
VIDEO_CONTEXT_SECONDS = 120
MAX_VIDEO_CLIP_SECONDS = 600
CLIP_CONTEXT_SECONDS = 60
SITE_RADIUS_M = 25.0
MIN_CONFIG_SUPPORT = 2


def direct_ch4_peak_events(interp: pd.DataFrame) -> pd.DataFrame:
    raw = interp["CH4"].to_numpy(dtype=float)
    log_signal = np.log1p(np.maximum(raw, 0))
    smooth = pd.Series(log_signal).rolling(5, center=True, min_periods=1).median().to_numpy()
    local_level = (
        pd.Series(raw).rolling(601, center=True, min_periods=60).median()
        .bfill().ffill().to_numpy()
    )
    peaks, properties = find_peaks(smooth, distance=30, prominence=0.25)
    keep = (raw[peaks] >= 1.08 * np.maximum(local_level[peaks], 0.1)) | (
        raw[peaks] - local_level[peaks] >= 0.25
    )
    peaks = peaks[keep]
    prominences = properties["prominences"][keep]
    widths, _, left_ips, right_ips = peak_widths(smooth, peaks, rel_height=0.75)
    rows = []
    for peak, prominence, width, left, right in zip(
        peaks, prominences, widths, left_ips, right_ips
    ):
        i0 = max(0, int(math.floor(left)))
        i1 = min(len(interp) - 1, int(math.ceil(right)))
        # Prominence bases can span an entire elevated plume plateau. Preserve
        # the distinct peak for review without repainting the whole plateau.
        i0 = max(i0, int(peak) - 150)
        i1 = min(i1, int(peak) + 150)
        rows.append(
            {
                "start_time": interp.time.iloc[i0],
                "end_time": interp.time.iloc[i1],
                "duration_s": i1 - i0 + 1,
                "method": "D3_direct_peak",
                "consensus": 1,
                "max_consensus_fraction": 1.0,
                "method_threshold": 1,
                "class": "DIRECT-PEAK",
                "peak_z": np.nan,
                "mass": np.nan,
                "config": "direct_signal",
                "channel": "CH4",
                "peak_value": float(raw[peak]),
                "direct_prominence": float(prominence),
            }
        )
    # A five-second median intentionally suppresses one- or two-sample spikes,
    # but some of those are among the largest raw CH4 peaks. Add a narrow raw
    # prominence path and only keep peaks not already represented nearby.
    raw_peaks, raw_properties = find_peaks(log_signal, distance=30, prominence=0.35)
    raw_keep = (
        (raw[raw_peaks] >= 1.08 * np.maximum(local_level[raw_peaks], 0.1))
        | (raw[raw_peaks] - local_level[raw_peaks] >= 0.25)
    )
    raw_peaks = raw_peaks[raw_keep]
    raw_prominences = raw_properties["prominences"][raw_keep]
    for peak, prominence in zip(raw_peaks, raw_prominences):
        if len(peaks) and np.min(np.abs(peaks - peak)) <= 10:
            continue
        i0 = max(0, int(peak) - 15)
        i1 = min(len(interp) - 1, int(peak) + 15)
        rows.append(
            {
                "start_time": interp.time.iloc[i0],
                "end_time": interp.time.iloc[i1],
                "duration_s": i1 - i0 + 1,
                "method": "D3_direct_peak",
                "consensus": 1,
                "max_consensus_fraction": 1.0,
                "method_threshold": 1,
                "class": "DIRECT-RAW-PEAK",
                "peak_z": np.nan,
                "mass": np.nan,
                "config": "direct_signal",
                "channel": "CH4",
                "peak_value": float(raw[peak]),
                "direct_prominence": float(prominence),
            }
        )
    return pd.DataFrame(rows)


def load_events(interp: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for config in CONFIGS:
        for channel in MATRIX_CHANNELS:
            path = EVENT_ROOT / config / f"fine_method_events_{channel}.csv"
            frame = pd.read_csv(path)
            frame = frame.rename(
                columns={
                    "method_consensus": "consensus",
                    "method_consensus_fraction": "max_consensus_fraction",
                }
            )
            frame["class"] = "DETECTOR-FAMILY"
            frame["peak_z"] = np.nan
            frame["mass"] = np.nan
            frame["config"] = config
            frame["channel"] = channel
            rows.append(frame)
    rows.append(direct_ch4_peak_events(interp))
    mat = loadmat(TS_RESULTS)
    detector_masks = np.asarray(mat["detectorMasksS"]).astype(bool)
    salinity_methods = ("D1_height", "D2_mass", "D3_peak", "D4_local")
    for method, mask in zip(salinity_methods, detector_masks.T):
        event_rows = []
        for start, end in contiguous_intervals(mask, interp["time"]):
            event_rows.append({
                "start_time": start, "end_time": end,
                "duration_s": int((end - start).total_seconds()) + 1,
                "method": method, "consensus": 1,
                "max_consensus_fraction": 1.0, "method_threshold": 1,
                "class": "CTD1-SALINITY-DETECTOR", "peak_z": np.nan,
                "mass": np.nan, "config": "ctd1_reference", "channel": "salinity",
            })
        if event_rows:
            rows.append(pd.DataFrame(event_rows))
    # T-S curve departures are independent physical-property evidence. Bridge
    # only short gaps and require 10 seconds so isolated residuals do not
    # repaint the track as a broad salinity anomaly.
    curve = pd.Series(np.asarray(mat["curveAnom"]).ravel().astype(bool))
    curve = curve.rolling(6, center=True, min_periods=1).max().astype(bool).to_numpy()
    curve_rows = []
    for start, end in contiguous_intervals(curve, interp["time"]):
        duration = int((end - start).total_seconds()) + 1
        if duration >= 10:
            curve_rows.append({
                "start_time": start, "end_time": end, "duration_s": duration,
                "method": "TS_curve", "consensus": 1,
                "max_consensus_fraction": 1.0, "method_threshold": 1,
                "class": "CTD1-TS-CURVE", "peak_z": np.nan, "mass": np.nan,
                "config": "ctd1_reference", "channel": "salinity",
            })
    if curve_rows:
        rows.append(pd.DataFrame(curve_rows))
    events = pd.concat(rows, ignore_index=True)
    events["start_time"] = pd.to_datetime(events["start_time"], utc=True)
    events["end_time"] = pd.to_datetime(events["end_time"], utc=True)
    for col in ("consensus", "max_consensus_fraction", "peak_z", "mass"):
        events[col] = pd.to_numeric(events[col], errors="coerce")
    events = events[events["end_time"] >= events["start_time"]].copy()
    return events.sort_values(["start_time", "end_time"])


def load_navigation() -> tuple[pd.DataFrame, pd.DataFrame]:
    interp = pd.read_csv(
        INTERP,
        usecols=["timestamp_iso", "alt", "water_depth", "CO2", "CH4", "O2", "Temperature", "Salinity"],
    )
    interp["time"] = pd.to_datetime(interp.pop("timestamp_iso"), utc=True)
    interp = interp.sort_values("time").reset_index(drop=True)

    nav = pd.read_csv(
        RAW_NAV,
        header=None,
        names=["date", "clock", "lat", "lon", "water_depth", "heading", "pitch", "roll", "flag"],
        dtype={"date": str, "clock": str, "flag": str},
        low_memory=False,
    )
    nav["time"] = pd.to_datetime(
        nav["date"] + " " + nav["clock"], format="%m/%d/%y %H:%M:%S", utc=True
    )
    nav = nav.sort_values("time").reset_index(drop=True)
    return interp, nav


def contiguous_intervals(mask: np.ndarray, times: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    mask = np.asarray(mask, dtype=bool).ravel()
    changes = np.diff(np.r_[False, mask, False].astype(np.int8))
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1) - 1
    return [(times.iloc[a], times.iloc[b]) for a, b in zip(starts, ends)]


def load_ctd1_flags(interp: pd.DataFrame) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp]]]:
    mat = loadmat(TS_RESULTS)
    flags = {}
    for source, label in (
        ("spikeAnom", "ctd1_ts_spike"),
        ("baseAnom", "ctd1_background_difference"),
        ("curveAnom", "ctd1_ts_curve"),
    ):
        values = np.asarray(mat[source]).ravel().astype(bool)
        if len(values) != len(interp):
            raise ValueError(f"{source} length {len(values)} does not match interp rows {len(interp)}")
        flags[label] = contiguous_intervals(values, interp["time"])
    return flags


def fuse_events(events: pd.DataFrame) -> list[dict]:
    fused: list[dict] = []
    current: list[pd.Series] = []
    right_edge: pd.Timestamp | None = None
    gap = pd.Timedelta(seconds=MERGE_GAP_SECONDS)
    for _, row in events.iterrows():
        if right_edge is None or row.start_time <= right_edge + gap:
            current.append(row)
            right_edge = row.end_time if right_edge is None else max(right_edge, row.end_time)
        else:
            fused.append(summarize_component(current))
            current = [row]
            right_edge = row.end_time
    if current:
        fused.append(summarize_component(current))
    return fused


def build_exact_signature_windows(events: pd.DataFrame, interp: pd.DataFrame) -> list[dict]:
    """Build exact channel windows from detector-family evidence.

    D3 peak support qualifies directly. Non-D3 samples require simultaneous
    support from at least two detector families. Every family must reproduce in
    at least MIN_CONFIG_SUPPORT processing configurations.
    """
    n = len(interp)
    time_ns = interp.time.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    support = np.zeros((n, len(CHANNELS), len(METHODS)), dtype=np.int8)
    consensus = np.zeros((n, len(CHANNELS), len(METHODS)), dtype=np.int16)
    special_mask = np.zeros((n, len(CHANNELS), len(METHODS)), dtype=bool)
    config_masks: dict[tuple[str, str, str], np.ndarray] = {}
    for config in CONFIGS:
        for channel in CHANNELS:
            for method in METHODS:
                config_masks[(config, channel, method)] = np.zeros(n, dtype=bool)
    for _, row in events.iterrows():
        ci = CHANNELS.index(row.channel)
        mi = METHODS.index(row.method)
        left = np.searchsorted(time_ns, row.start_time.value, side="left")
        right = np.searchsorted(time_ns, row.end_time.value, side="right")
        if row.config in CONFIGS:
            config_masks[(row.config, row.channel, row.method)][left:right] = True
        else:
            special_mask[left:right, ci, mi] = True
        consensus[left:right, ci, mi] = np.maximum(
            consensus[left:right, ci, mi], int(row.consensus)
        )
    for ci, channel in enumerate(CHANNELS):
        for mi, method in enumerate(METHODS):
            for config in CONFIGS:
                support[:, ci, mi] += config_masks[(config, channel, method)]
    support = np.maximum(support, special_mask.astype(np.int8) * MIN_CONFIG_SUPPORT)

    method_active = support >= MIN_CONFIG_SUPPORT
    peak_active = (
        method_active[:, :, METHODS.index("D3_peak")]
        | method_active[:, :, METHODS.index("D3_direct_peak")]
    )
    reference_active = method_active[:, :, METHODS.index("TS_curve")]
    multi_method_active = method_active.sum(axis=2) >= 2
    channel_active = peak_active | reference_active | multi_method_active
    bit_values = (2 ** np.arange(len(CHANNELS))).astype(np.int16)
    signature_code = (channel_active * bit_values).sum(axis=1)
    changes = np.flatnonzero(np.r_[True, signature_code[1:] != signature_code[:-1], True])
    output = []
    for left, right_exclusive in zip(changes[:-1], changes[1:]):
        code = int(signature_code[left])
        if code == 0 or right_exclusive - left < 10:
            continue
        right = right_exclusive - 1
        active = [CHANNELS[i] for i in range(len(CHANNELS)) if code & bit_values[i]]
        per_channel_support = {
            ch: int(support[left:right_exclusive, CHANNELS.index(ch), :].max()) for ch in active
        }
        used_configs = {
            cfg for cfg in CONFIGS for ch in active for method in METHODS
            if config_masks[(cfg, ch, method)][left:right_exclusive].any()
        }
        channel_methods = {}
        peak_channels = []
        all_methods = set()
        for ch in active:
            ci = CHANNELS.index(ch)
            methods_here = [
                method for mi, method in enumerate(METHODS)
                if method_active[left:right_exclusive, ci, mi].any()
            ]
            channel_methods[ch] = methods_here
            all_methods.update(methods_here)
            if "D3_peak" in methods_here or "D3_direct_peak" in methods_here:
                peak_channels.append(CHANNEL_PRETTY[ch])
        source_overlap = events[
            (events.start_time <= interp.time.iloc[right])
            & (events.end_time >= interp.time.iloc[left])
            & events.channel.isin(active)
        ]
        output.append(
            {
                "start_time": interp.time.iloc[left],
                "end_time": interp.time.iloc[right],
                "channels": ",".join(CHANNEL_PRETTY[ch] for ch in active),
                "channel_count": len(active),
                "configs": ",".join(sorted(used_configs, key=CONFIGS.index)),
                "config_count": len(used_configs),
                "mean_channel_config_support": float(np.mean(list(per_channel_support.values()))),
                "min_channel_config_support": min(per_channel_support.values()),
                "channel_config_support": ";".join(
                    f"{CHANNEL_PRETTY[ch]}:{per_channel_support[ch]}" for ch in active
                ),
                "event_classes": "DETECTOR-AWARE-FINE",
                "detector_methods": ",".join(m for m in METHODS if m in all_methods),
                "detector_count": len(all_methods),
                "channel_detector_methods": ";".join(
                    f"{CHANNEL_PRETTY[ch]}:{'+'.join(channel_methods[ch])}" for ch in active
                ),
                "peak_finder_present": bool(peak_channels),
                "peak_finder_channels": ",".join(peak_channels) if peak_channels else "none",
                "max_consensus": int(consensus[left:right_exclusive, :, :].max()),
                "max_consensus_fraction": float(
                    max(
                        consensus[left:right_exclusive, :, :].max() / 20.0,
                        1.0 if special_mask[left:right_exclusive, :, :].any() else 0.0,
                    )
                ),
                "total_source_events": len(source_overlap),
            }
        )
    return output


def summarize_component(rows: list[pd.Series]) -> dict:
    channels = sorted({str(r.channel) for r in rows}, key=CHANNELS.index)
    configs = sorted({str(r.config) for r in rows}, key=CONFIGS.index)
    channel_support = {
        ch: len({str(r.config) for r in rows if r.channel == ch}) for ch in channels
    }
    return {
        "start_time": min(r.start_time for r in rows),
        "end_time": max(r.end_time for r in rows),
        "channels": ",".join(CHANNEL_PRETTY[ch] for ch in channels),
        "channel_count": len(channels),
        "configs": ",".join(configs),
        "config_count": len(configs),
        "channel_config_support": ";".join(
            f"{CHANNEL_PRETTY[ch]}:{channel_support[ch]}" for ch in channels
        ),
        "event_classes": ",".join(sorted({str(r["class"]) for r in rows})),
        "max_consensus": int(max(r.consensus for r in rows)),
        "max_peak_z": float(max(r.peak_z for r in rows)),
        "total_source_events": len(rows),
    }


def classify_indicators(channel_text: str) -> tuple[str, str, str, int]:
    channels = set(channel_text.split(","))
    signature_parts = []
    if "CO2" in channels:
        signature_parts.append("CO2↑")
    if "CH4" in channels:
        signature_parts.append("CH4↑")
    if "O2" in channels:
        signature_parts.append("O2↓")
    if "Temperature" in channels:
        signature_parts.append("Temperature↑")
    if "Salinity" in channels:
        signature_parts.append("Salinity±")
    signature = " + ".join(signature_parts)

    order = len(channels)
    order_name = {
        1: "SINGLE", 2: "PAIR", 3: "TRIPLET",
        4: "FOUR-CHANNEL", 5: "FIVE-CHANNEL",
    }[order]
    combination = "/".join(
        name for name in ("CO2", "CH4", "O2", "Temperature", "Salinity") if name in channels
    )
    label = f"{order_name}: {signature}"
    return label, signature, combination, order


def overlaps(intervals: list[tuple[pd.Timestamp, pd.Timestamp]], start: pd.Timestamp, end: pd.Timestamp) -> bool:
    return any(a <= end and b >= start for a, b in intervals)


def nearest_row(table: pd.DataFrame, when: pd.Timestamp) -> pd.Series:
    idx = table["time"].searchsorted(when)
    candidates = [max(0, idx - 1), min(len(table) - 1, idx)]
    best = min(candidates, key=lambda i: abs(table.iloc[i]["time"] - when))
    return table.iloc[best]


def add_context(windows: pd.DataFrame, interp: pd.DataFrame, nav: pd.DataFrame, ctd_flags: dict) -> pd.DataFrame:
    sensor_start = max(interp["time"].min(), nav["time"].min())
    sensor_end = min(interp["time"].max(), nav["time"].max())
    windows = windows[(windows.start_time >= sensor_start) & (windows.end_time <= sensor_end)].copy()

    records = []
    for _, row in windows.iterrows():
        mid = row.start_time + (row.end_time - row.start_time) / 2
        navrow = nearest_row(nav, mid)
        introw = nearest_row(interp, mid)
        evidence = [
            name for name, intervals in ctd_flags.items()
            if overlaps(intervals, row.start_time, row.end_time)
        ]
        breadth = {1: 8.0, 2: 18.0, 3: 26.0, 4: 32.0, 5: 35.0}[int(row.channel_count)]
        robustness = float(row.mean_channel_config_support) / 4.0 * 25.0
        detector_diversity = min(float(row.detector_count), 4.0) / 4.0 * 20.0
        consensus = min(float(row.max_consensus_fraction), 1.0) * 10.0
        corroboration = 10.0 if evidence else 0.0
        score = round(
            min(
                100.0,
                breadth + robustness + detector_diversity + consensus + corroboration,
            ),
            1,
        )
        tier = "HIGH" if score >= 65 else ("MODERATE" if score >= 40 else "SCREEN")
        rec = row.to_dict()
        anomaly_class, signature, combination, combination_order = classify_indicators(
            str(row.channels)
        )
        rec.update(
            {
                "mid_time": mid,
                "duration_s": int((row.end_time - row.start_time).total_seconds()) + 1,
                "review_start": max(sensor_start, row.start_time - pd.Timedelta(seconds=VIDEO_CONTEXT_SECONDS)),
                "review_end": min(sensor_end, row.end_time + pd.Timedelta(seconds=VIDEO_CONTEXT_SECONDS)),
                "review_duration_s": int(
                    (
                        min(sensor_end, row.end_time + pd.Timedelta(seconds=VIDEO_CONTEXT_SECONDS))
                        - max(sensor_start, row.start_time - pd.Timedelta(seconds=VIDEO_CONTEXT_SECONDS))
                    ).total_seconds()
                ),
                "lat": float(navrow.lat),
                "lon": float(navrow.lon),
                "water_depth_m": float(navrow.water_depth),
                "altitude_m": float(introw.alt),
                "CO2_mid": float(introw.CO2),
                "CH4_mid": float(introw.CH4),
                "O2_mid": float(introw.O2),
                "Temperature_mid": float(introw.Temperature),
                "Salinity_mid": float(introw.Salinity),
                "ctd1_corroboration": ",".join(evidence) if evidence else "none",
                "anomaly_class": anomaly_class,
                "indicator_signature": signature,
                "channel_combination": combination,
                "combination_order": combination_order,
                "evidence_score": score,
                "confidence_tier": tier,
            }
        )
        records.append(rec)
    result = pd.DataFrame(records)
    chronological = result.sort_values("start_time").index
    result.loc[chronological, "window_id"] = [
        f"WINDOW-{i:03d}" for i in range(1, len(result) + 1)
    ]
    tier_order = {"HIGH": 0, "MODERATE": 1, "SCREEN": 2}
    result["_tier_order"] = result.confidence_tier.map(tier_order)
    return result.sort_values(["_tier_order", "evidence_score", "start_time"], ascending=[True, False, True])


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6_371_000.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * radius * math.asin(math.sqrt(a))


def assign_sites(windows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    chronological = windows.sort_values("start_time").copy()
    sites: list[dict] = []
    site_ids: dict[int, str] = {}
    for idx, row in chronological.iterrows():
        matches = [
            (haversine_m(row.lat, row.lon, site["lat"], site["lon"]), i)
            for i, site in enumerate(sites)
        ]
        distance, match = min(matches, default=(float("inf"), -1))
        if distance > SITE_RADIUS_M:
            sites.append({"lat": row.lat, "lon": row.lon, "members": [idx]})
            match = len(sites) - 1
        else:
            sites[match]["members"].append(idx)
            members = chronological.loc[sites[match]["members"]]
            sites[match]["lat"] = float(members.lat.mean())
            sites[match]["lon"] = float(members.lon.mean())
        site_ids[idx] = f"SITE-{match + 1:03d}"
    windows = windows.copy()
    windows["site_id"] = pd.Series(site_ids)

    order = {"HIGH": 3, "MODERATE": 2, "SCREEN": 1}
    summaries = []
    for i, site in enumerate(sites, start=1):
        group = windows.loc[site["members"]]
        channels = sorted({c for text in group.channels for c in text.split(",")})
        best_tier = max(group.confidence_tier, key=order.get)
        summaries.append(
            {
                "site_id": f"SITE-{i:03d}",
                "lat": float(group.lat.mean()),
                "lon": float(group.lon.mean()),
                "window_count": len(group),
                "high_windows": int((group.confidence_tier == "HIGH").sum()),
                "moderate_windows": int((group.confidence_tier == "MODERATE").sum()),
                "best_tier": best_tier,
                "max_evidence_score": float(group.evidence_score.max()),
                "channels": ",".join(channels),
                "first_time": group.start_time.min(),
                "last_time": group.end_time.max(),
            }
        )
    return windows, pd.DataFrame(summaries).sort_values(
        ["max_evidence_score", "window_count"], ascending=[False, False]
    )


def iso(value) -> str:
    return pd.Timestamp(value).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_video_clips(windows: pd.DataFrame) -> pd.DataFrame:
    clips = []
    for _, row in windows.sort_values("start_time").iterrows():
        clip_start = row.start_time
        clip_number = 1
        while clip_start <= row.end_time:
            anomaly_end = min(
                row.end_time,
                clip_start + pd.Timedelta(seconds=MAX_VIDEO_CLIP_SECONDS - 1),
            )
            clips.append(
                {
                    "clip_id": f"{row.window_id}-CLIP-{clip_number:02d}",
                    "window_id": row.window_id,
                    "site_id": row.site_id,
                    "confidence_tier": row.confidence_tier,
                    "evidence_score": row.evidence_score,
                    "review_start": clip_start - pd.Timedelta(seconds=CLIP_CONTEXT_SECONDS),
                    "review_end": anomaly_end + pd.Timedelta(seconds=CLIP_CONTEXT_SECONDS),
                    "anomaly_start": clip_start,
                    "anomaly_end": anomaly_end,
                    "channels": row.channels,
                    "indicator_signature": row.indicator_signature,
                    "anomaly_class": row.anomaly_class,
                    "channel_combination": row.channel_combination,
                    "combination_order": row.combination_order,
                    "detector_methods": row.detector_methods,
                    "detector_count": row.detector_count,
                    "channel_detector_methods": row.channel_detector_methods,
                    "peak_finder_present": row.peak_finder_present,
                    "peak_finder_channels": row.peak_finder_channels,
                    "ctd1_corroboration": row.ctd1_corroboration,
                    "lat": row.lat,
                    "lon": row.lon,
                    "altitude_m": row.altitude_m,
                }
            )
            clip_start = anomaly_end + pd.Timedelta(seconds=1)
            clip_number += 1
    return pd.DataFrame(clips)


def write_qgis_layer(windows: pd.DataFrame, nav: pd.DataFrame) -> None:
    qgis_dir = OUT / "qgis"
    qgis_dir.mkdir(exist_ok=True)
    features = []
    for _, row in windows.sort_values("start_time").iterrows():
        segment = nav[
            (nav.time >= row.start_time.floor("s"))
            & (nav.time <= row.end_time.ceil("s"))
        ]
        if len(segment) < 2:
            continue
        coords = [
            [round(float(lon), 8), round(float(lat), 8)]
            for lon, lat in zip(segment.lon, segment.lat)
            if np.isfinite(lon) and np.isfinite(lat)
        ]
        if len(coords) < 2:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "LineString", "coordinates": coords},
                "properties": {
                    "window_id": row.window_id,
                    "site_id": row.site_id,
                    "confidence": row.confidence_tier,
                    "score": float(row.evidence_score),
                    "n_channels": int(row.combination_order),
                    "combination": row.channel_combination,
                    "indicators": row.indicator_signature,
                    "class": row.anomaly_class,
                    "start_utc": iso(row.start_time),
                    "end_utc": iso(row.end_time),
                    "duration_s": int(row.duration_s),
                    "config_support": row.channel_config_support,
                    "detectors": row.detector_methods,
                    "detector_count": int(row.detector_count),
                    "channel_detectors": row.channel_detector_methods,
                    "d3_peak": bool(row.peak_finder_present),
                    "d3_channels": row.peak_finder_channels,
                    "max_consensus": int(row.max_consensus),
                    "ctd1_evidence": row.ctd1_corroboration,
                    "altitude_m": round(float(row.altitude_m), 2),
                    "water_depth_m": round(float(row.water_depth_m), 2),
                },
            }
        )
    layer = {
        "type": "FeatureCollection",
        "name": "J1754_anomaly_segments",
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
        },
        "metadata": {
            "crs": "EPSG:4326",
            "geometry": "LineString",
            "source_navigation": str(RAW_NAV.relative_to(REPO)),
            "event_boundary": "fine detector families; D3 direct or >=2 methods; >=2 configs",
            "feature_count": len(features),
        },
        "features": features,
    }
    geojson_path = qgis_dir / "J1754_anomaly_segments.geojson"
    geojson_path.write_text(json.dumps(layer, separators=(",", ":")))

    context = {
        "type": "FeatureCollection",
        "name": "J1754_trackline_context",
        "crs": layer["crs"],
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [round(float(lon), 8), round(float(lat), 8)]
                        for lon, lat in zip(nav.lon, nav.lat)
                        if np.isfinite(lon) and np.isfinite(lat)
                    ],
                },
                "properties": {"dive": "J1754", "role": "trackline context"},
            }
        ],
    }
    context_path = qgis_dir / "J1754_trackline_context.geojson"
    context_path.write_text(json.dumps(context, separators=(",", ":")))

    qml_path = qgis_dir / "J1754_anomaly_segments.qml"
    qml_path.write_text(
        """<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.34.15-Prizren" styleCategories="Symbology|Labeling">
  <renderer-v2 type="categorizedSymbol" attr="confidence" enableorderby="0">
    <categories>
      <category value="HIGH" label="HIGH" symbol="0" render="true"/>
      <category value="MODERATE" label="MODERATE" symbol="1" render="true"/>
      <category value="SCREEN" label="SCREEN" symbol="2" render="true"/>
    </categories>
    <symbols>
      <symbol name="0" type="line" alpha="1" clip_to_extent="1">
        <layer class="SimpleLine" enabled="1">
          <Option type="Map"><Option name="line_color" value="198,40,40,255" type="QString"/><Option name="line_width" value="1.4" type="QString"/><Option name="line_width_unit" value="MM" type="QString"/><Option name="capstyle" value="round" type="QString"/><Option name="joinstyle" value="round" type="QString"/></Option>
        </layer>
      </symbol>
      <symbol name="1" type="line" alpha="1" clip_to_extent="1">
        <layer class="SimpleLine" enabled="1">
          <Option type="Map"><Option name="line_color" value="239,125,0,255" type="QString"/><Option name="line_width" value="1.0" type="QString"/><Option name="line_width_unit" value="MM" type="QString"/><Option name="capstyle" value="round" type="QString"/><Option name="joinstyle" value="round" type="QString"/></Option>
        </layer>
      </symbol>
      <symbol name="2" type="line" alpha="1" clip_to_extent="1">
        <layer class="SimpleLine" enabled="1">
          <Option type="Map"><Option name="line_color" value="61,110,168,255" type="QString"/><Option name="line_width" value="0.7" type="QString"/><Option name="line_width_unit" value="MM" type="QString"/><Option name="capstyle" value="round" type="QString"/><Option name="joinstyle" value="round" type="QString"/></Option>
        </layer>
      </symbol>
    </symbols>
  </renderer-v2>
  <labeling type="simple">
    <settings>
      <text-style fieldName="window_id" isExpression="0" fontSize="8" fontSizeUnit="Point"/>
      <placement placement="2" repeatDistance="0"/>
      <rendering scaleVisibility="1" minimumScale="0" maximumScale="25000"/>
    </settings>
  </labeling>
</qgis>
"""
    )
    readme_path = qgis_dir / "README.txt"
    readme_path.write_text(
        "J1754 anomaly layers for AT50-45-QGIS-feb-2026.qgs\n\n"
        "CRS: EPSG:4326 (WGS 84), matching the QGIS project.\n"
        "Primary layer: J1754_anomaly_segments.geojson\n"
        "Context layer: J1754_trackline_context.geojson\n"
        "Style: J1754_anomaly_segments.qml\n\n"
        "QGIS: Layer > Add Layer > Add Vector Layer, select both GeoJSON files.\n"
        "Then open anomaly layer Properties > Symbology > Style > Load Style and\n"
        "select the QML file. The style categorizes line segments by confidence.\n"
        "Use 'combination' or 'n_channels' for pair/triplet/n-wise categorization.\n"
    )
    zip_path = qgis_dir / "J1754_anomaly_QGIS_upload.zip"
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in (geojson_path, context_path, qml_path, readme_path):
            archive.write(path, arcname=path.name)


def write_outputs(
    windows: pd.DataFrame,
    sites: pd.DataFrame,
    clips: pd.DataFrame,
    nav: pd.DataFrame,
) -> None:
    OUT.mkdir(exist_ok=True)
    clean = windows.drop(columns=["_tier_order"]).copy()
    for col in ("start_time", "end_time", "mid_time", "review_start", "review_end"):
        clean[col] = clean[col].map(iso)
    clean.to_csv(OUT / "anomaly_windows_all.csv", index=False)

    combo = (
        windows.groupby(
            ["combination_order", "channel_combination", "indicator_signature",
             "confidence_tier"],
            as_index=False,
        )
        .agg(
            window_count=("window_id", "count"),
            total_duration_s=("duration_s", "sum"),
            median_duration_s=("duration_s", "median"),
            distinct_sites=("site_id", "nunique"),
            max_evidence_score=("evidence_score", "max"),
        )
        .sort_values(["combination_order", "channel_combination", "confidence_tier"])
    )
    combo.to_csv(OUT / "anomaly_combination_summary.csv", index=False)

    queue_cols = [
        "window_id", "site_id", "confidence_tier", "evidence_score", "review_start", "review_end",
        "start_time", "end_time", "channels", "channel_count", "config_count",
        "indicator_signature", "anomaly_class", "channel_combination",
        "combination_order", "detector_methods", "detector_count",
        "channel_detector_methods", "peak_finder_present", "peak_finder_channels",
        "max_consensus", "max_consensus_fraction", "event_classes",
        "ctd1_corroboration", "lat", "lon",
        "water_depth_m", "altitude_m",
    ]
    clean.sort_values(
        ["confidence_tier", "evidence_score"], ascending=[True, False]
    )[queue_cols].to_csv(OUT / "video_review_queue.csv", index=False)

    clip_out = clips.copy()
    for col in ("review_start", "review_end", "anomaly_start", "anomaly_end"):
        clip_out[col] = clip_out[col].map(iso)
    clip_out.to_csv(OUT / "video_review_clips.csv", index=False)

    site_out = sites.copy()
    for col in ("first_time", "last_time"):
        site_out[col] = site_out[col].map(iso)
    site_out.to_csv(OUT / "anomalous_sites.csv", index=False)

    features = []
    for _, row in sites.iterrows():
        props = row.to_dict()
        props.pop("lat"); props.pop("lon")
        props["first_time"] = iso(props["first_time"])
        props["last_time"] = iso(props["last_time"])
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [row.lon, row.lat]},
                "properties": props,
            }
        )
    (OUT / "anomalous_sites.geojson").write_text(
        json.dumps({"type": "FeatureCollection", "features": features}, indent=2)
    )
    write_qgis_layer(windows, nav)


# --------------------------------------------------------------------------
# Layout helpers — measure text so nothing is ever clipped
# --------------------------------------------------------------------------
# Rendered widths are measured with TextPath rather than estimated from
# character counts: the report mixes proportional text with wide glyphs (↑ ↓ –)
# and long unbreakable identifiers, so a characters-per-column guess silently
# overflows.  Everything below sizes from the real glyph extents.

_CHAR_WIDTH_CACHE: dict[str, dict[str, float]] = {}
_FALLBACK_EM = 0.62          # width of an unmeasured glyph, in em


def _char_widths(weight: str) -> dict[str, float]:
    """Per-character advance widths at 1 pt, measured once per font weight.

    Measuring whole strings with TextPath is accurate but far too slow here —
    the tables re-wrap thousands of cells at each candidate font size.  Glyph
    widths are additive for this font, so measuring the character set once and
    summing is both fast and accurate enough to guarantee a fit (a small safety
    margin is applied by the callers' padding).
    """
    cached = _CHAR_WIDTH_CACHE.get(weight)
    if cached is not None:
        return cached
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties

    charset = (
        " !\"#$%&'()*+,-./0123456789:;<=>?@"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ[\\]^_`"
        "abcdefghijklmnopqrstuvwxyz{|}~"
        "±°↑↓–—‘’“”…×·≥≤"
    )
    prop = FontProperties(size=100.0, weight=weight)
    widths: dict[str, float] = {}
    for ch in charset:
        try:
            # Measure in a stable context ("nn" brackets) to capture advance
            # width rather than the inked bounding box, which drops side bearings.
            pair = float(TextPath((0, 0), f"n{ch}n", prop=prop).get_extents().width)
            base = float(TextPath((0, 0), "nn", prop=prop).get_extents().width)
            widths[ch] = max(0.0, (pair - base)) / 100.0
        except Exception:                                # noqa: BLE001
            widths[ch] = _FALLBACK_EM
    widths.setdefault(" ", 0.318)
    _CHAR_WIDTH_CACHE[weight] = widths
    return widths


def text_width_pt(text: str, fontsize: float, weight: str = "normal") -> float:
    """Rendered width of `text` in points at `fontsize`."""
    if not text:
        return 0.0
    table = _char_widths(weight)
    em = sum(table.get(ch, _FALLBACK_EM) for ch in text)
    return em * fontsize


def soften(text: str) -> str:
    """Insert break opportunities into long unbreakable identifiers.

    Values like ``ctd1_ts_spike,ctd1_background_difference,ctd1_ts_curve`` are a
    single whitespace-free token, so textwrap cannot split them at all and the
    cell overflows the table.  Adding a space after each comma (and allowing a
    break after underscores) gives the wrapper something to work with, without
    changing the value that a reader sees.
    """
    out = str(text).replace(",", ", ").replace("_", "_​")
    return " ".join(out.split())


def wrap_to_width(text: str, max_width_pt: float, fontsize: float) -> list[str]:
    """Wrap `text` into lines that each fit within `max_width_pt`.

    Greedy word wrap on measured widths.  A single word wider than the limit is
    hard-split so it still fits rather than bleeding past the margin.
    """
    raw_words = soften(text).split()
    if not raw_words:
        return [""]

    # Pass 1 — hard-split any single word that cannot fit on a line by itself,
    # so an over-long token can never bleed past the margin regardless of where
    # it happens to fall in the sentence.
    words: list[str] = []
    for word in raw_words:
        if text_width_pt(word, fontsize) <= max_width_pt:
            words.append(word)
            continue
        piece = ""
        for ch in word:
            if piece and text_width_pt(piece + ch, fontsize) > max_width_pt:
                words.append(piece)
                piece = ch
            else:
                piece += ch
        if piece:
            words.append(piece)

    # Pass 2 — ordinary greedy wrap on measured widths.
    lines: list[str] = []
    line = ""
    for word in words:
        trial = word if not line else f"{line} {word}"
        if not line or text_width_pt(trial, fontsize) <= max_width_pt:
            line = trial
        else:
            lines.append(line)
            line = word
    if line:
        lines.append(line)
    return [ln.replace("​", "") for ln in lines] or [""]


def caption(fig, text: str, fontsize: float = 9.0, pad_in: float = 0.10) -> None:
    """Place a wrapped caption along the bottom of `fig` without overlap.

    The caption is wrapped to the figure width and the figure's bottom margin is
    then expanded to reserve exactly the space the wrapped block needs, so the
    caption can never sit on top of an x-axis label or run off the page edge.
    """
    fig_w_in, fig_h_in = fig.get_size_inches()
    left_in = 0.14
    usable_pt = (fig_w_in - 2 * left_in) * 72.0
    lines = wrap_to_width(text, usable_pt, fontsize)

    line_h_in = fontsize * 1.35 / 72.0
    block_in = len(lines) * line_h_in + pad_in

    # Reserve room beneath the axes for the caption block.  Constrained-layout
    # figures also get a small horizontal inset so a long y-tick label (e.g.
    # "T–S reference-curve departure") is not pushed flush against the page edge.
    inset = 0.06 / fig_w_in
    try:
        if fig.get_constrained_layout():
            y0 = block_in / fig_h_in
            fig.get_layout_engine().set(
                rect=(inset, y0, 1 - 2 * inset, 1 - y0 - inset)
            )
        else:
            bottom = fig.subplotpars.bottom
            fig.subplots_adjust(bottom=max(bottom, block_in / fig_h_in + 0.02))
    except Exception:                                    # noqa: BLE001
        fig.subplots_adjust(bottom=block_in / fig_h_in + 0.02)

    fig.text(
        left_in / fig_w_in, (pad_in * 0.45) / fig_h_in,
        "\n".join(lines), fontsize=fontsize, va="bottom", ha="left", linespacing=1.35,
    )


def text_page(pdf: PdfPages, title: str, paragraphs: list[str]) -> None:
    """Render prose onto as many portrait pages as the content needs.

    Lines are wrapped to the measured text column and the page is broken when
    the next paragraph would cross the bottom margin, so text can never run off
    the end of a page.  ``bbox_inches='tight'`` is deliberately NOT used: it
    changes the page size per page and crops anything near an edge.
    """
    page_w, page_h = 8.5, 11.0
    left, right, top, bottom = 0.75, 0.75, 1.05, 0.75      # inches
    fontsize, title_size = 9.5, 18.0
    line_h = fontsize * 1.45 / 72.0                        # inches per line
    para_gap = line_h * 0.7
    col_pt = (page_w - left - right) * 72.0
    max_y = page_h - top
    min_y = bottom

    # Pre-wrap every paragraph once, then greedily fill pages.
    blocks = [wrap_to_width(p, col_pt, fontsize) for p in paragraphs]

    fig = None
    y = 0.0
    first = True
    for lines in blocks:
        need = len(lines) * line_h + para_gap
        if fig is None or (y - need) < min_y:
            if fig is not None:
                pdf.savefig(fig)
                plt.close(fig)
            fig = plt.figure(figsize=(page_w, page_h), facecolor="white")
            heading = title if first else f"{title} (cont.)"
            fig.text(left / page_w, 1 - (top - 0.45) / page_h, heading,
                     fontsize=title_size, weight="bold", va="top")
            first = False
            y = max_y - 0.30
        fig.text(left / page_w, y / page_h, "\n".join(lines),
                 fontsize=fontsize, va="top", linespacing=1.45)
        y -= need
    if fig is not None:
        pdf.savefig(fig)
        plt.close(fig)


def table_pages(pdf: PdfPages, frame: pd.DataFrame, title: str, rows_per_page: int = 24) -> None:
    """Render landscape tables with every cell fully inside its column.

    Column widths are derived from MEASURED glyph extents rather than assumed
    characters-per-column, and long unbreakable identifiers (comma-joined CTD1
    evidence, for example) are given break opportunities first.  The font is
    then reduced until the widest wrapped line in every column fits the width
    that column was actually allocated, so nothing overflows a cell or the page.
    """
    page_w, page_h = 11.0, 8.5
    table_left, table_right = 0.02, 0.98
    avail_pt = (table_right - table_left) * page_w * 72.0
    cell_pad_pt = 7.0                                   # padding inside each cell
    max_lines_per_page = 34

    columns = [str(c) for c in frame.columns]
    raw = frame.copy().astype(str)

    # Per-column demand, measured at a 1 pt reference and scaled by font size:
    #   floor_em  — the widest UNBREAKABLE token (a site id, a latitude, a header
    #               word).  A column narrower than this splits a value mid-token
    #               ("SITE-08/6"), so it is a hard minimum.
    #   want_em   — a comfortable width for typical content, capped so one very
    #               wide column cannot starve the others.
    floor_em, want_em = [], []
    for column in columns:
        tokens = [w for value in raw[column] for w in soften(value).split()] or [column]
        widest_token = max(text_width_pt(t, 1.0) for t in tokens)
        header_word = max(
            (text_width_pt(w, 1.0, weight="bold") for w in column.split()), default=0.0
        )
        floor_em.append(max(widest_token, header_word))
        typical = float(np.percentile(
            [text_width_pt(soften(v), 1.0) for v in raw[column]] or [1.0], 82
        ))
        want_em.append(max(typical, floor_em[-1]))

    def layout(size: float) -> tuple[list[float], bool]:
        """Column widths (pt) at `size`; False if unbreakable tokens cannot fit."""
        floors = [e * size + cell_pad_pt for e in floor_em]
        if sum(floors) > avail_pt:
            return floors, False
        # Give every column its floor, then share the slack in proportion to how
        # much more its typical content would like.
        extra = [max(0.0, w * size - f) for w, f in zip(want_em, floors)]
        slack = avail_pt - sum(floors)
        total_extra = sum(extra)
        if total_extra <= 0:
            share = [slack / len(floors)] * len(floors)
        else:
            share = [slack * e / total_extra for e in extra]
        return [f + s for f, s in zip(floors, share)], True

    fontsize = 7.0
    col_pt, fits = layout(fontsize)
    while not fits and fontsize > 4.6:
        fontsize = round(fontsize - 0.2, 1)
        col_pt, fits = layout(fontsize)

    inner = [max(12.0, w - cell_pad_pt) for w in col_pt]
    wrapped_cols = {}
    for column, width_pt in zip(columns, inner):
        wrapped_cols[column] = (
            [wrap_to_width(v, width_pt, fontsize) for v in raw[column]],
            wrap_to_width(column, width_pt, fontsize),
        )

    wrapped = pd.DataFrame(
        {c: ["\n".join(lines) or " " for lines in wrapped_cols[c][0]] for c in columns}
    )
    headers = ["\n".join(wrapped_cols[c][1]) for c in columns]
    header_lines = max(len(wrapped_cols[c][1]) for c in columns)
    col_widths = [w / sum(col_pt) for w in col_pt]

    # Paginate by rendered line count, not record count: a row needing three
    # lines takes three times the vertical room of a one-line row.
    pages: list[tuple[int, pd.DataFrame, list[int]]] = []
    start = 0
    while start < len(wrapped):
        rows, line_counts, used = [], [], 0
        while start + len(rows) < len(wrapped) and len(rows) < rows_per_page:
            row = wrapped.iloc[start + len(rows)]
            lines = max(str(value).count("\n") + 1 for value in row)
            if rows and used + lines > max_lines_per_page:
                break
            rows.append(row)
            line_counts.append(lines)
            used += lines
        page = pd.DataFrame(rows, columns=wrapped.columns)
        pages.append((start, page, line_counts))
        start += len(page)

    for start, page, line_counts in pages:
        fig = plt.figure(figsize=(page_w, page_h))
        # Axes spans the WHOLE figure so the table bbox (in axes coords) maps
        # 1:1 onto page fractions.  With plt.subplots' default margins the axes
        # is only ~78% of the page and every measured width would be too
        # generous, pushing cell text past its column.
        ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        ax.axis("off")
        fig.suptitle(f"{title} — rows {start + 1}–{start + len(page)}",
                     fontsize=14, y=0.975)
        top, bot = 0.935, 0.03                           # leave room for the title
        span = top - bot
        table = ax.table(
            cellText=page.values,
            colLabels=headers,
            loc="center", cellLoc="left", colLoc="left",
            colWidths=col_widths,
            bbox=[table_left, bot, table_right - table_left, span],
        )
        table.auto_set_font_size(False)
        table.set_fontsize(fontsize)
        units = header_lines + sum(line_counts)
        for (r, _), cell in table.get_celld().items():
            cell.set_linewidth(0.4)
            cell.PAD = 0.04
            if r == 0:
                cell.set_facecolor("#d9e8f5")
                cell.set_text_props(weight="bold")
                cell.set_height(span * header_lines / units)
            else:
                cell.set_height(span * line_counts[r - 1] / units)
                cell.get_text().set_va("center")
        pdf.savefig(fig)
        plt.close(fig)


def channel_windows(windows: pd.DataFrame, channel: str) -> pd.DataFrame:
    return windows[windows.channels.str.split(",").map(lambda values: channel in values)]


def anomaly_mask(times: pd.Series, intervals: pd.DataFrame) -> np.ndarray:
    mask = np.zeros(len(times), dtype=bool)
    values = times.astype("datetime64[ns, UTC]").astype("int64").to_numpy()
    for _, row in intervals.iterrows():
        left = np.searchsorted(values, row.start_time.value, side="left")
        right = np.searchsorted(values, row.end_time.value, side="right")
        mask[left:right] = True
    return mask


def add_event_spans(ax, intervals: pd.DataFrame, alpha: float = 0.12) -> None:
    colors = {"HIGH": "#d73027", "MODERATE": "#f28e2b", "SCREEN": "#4e79a7"}
    for _, row in intervals.sort_values("start_time").iterrows():
        ax.axvspan(row.start_time, row.end_time, color=colors[row.confidence_tier],
                   alpha=alpha, lw=0)


def stacked_sensor_page(pdf: PdfPages, windows: pd.DataFrame, interp: pd.DataFrame) -> None:
    specs = [
        ("CO2", "CO₂ (µatm; symlog)", "#3268a8"),
        ("CH4", "CH₄ (µatm; log scale)", "#6a3d9a"),
        ("O2", "O₂ (µM; depletion is anomalous)", "#2a9d8f"),
        ("Temperature", "Temperature (°C)", "#d95f02"),
        ("Salinity", "Salinity (PSU)", "#007c91"),
    ]
    fig, axes = plt.subplots(5, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    fig.suptitle("J1754 sensor record with independently flagged anomaly regions",
                 fontsize=18, weight="bold")
    for ax, (column, ylabel, color) in zip(axes, specs):
        intervals = channel_windows(windows, column)
        mask = anomaly_mask(interp.time, intervals)
        values = interp[column].to_numpy(dtype=float)
        add_event_spans(ax, intervals, 0.10)
        ax.plot(interp.time, values, color="#8a9299", lw=0.55, alpha=0.75)
        ax.plot(interp.time, np.where(mask, values, np.nan), color=color, lw=1.1,
                label=f"{column} flagged")
        if column == "CH4":
            ax.set_yscale("symlog", linthresh=1)
        elif column == "CO2":
            ax.set_yscale("symlog", linthresh=1000)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.18)
        ax.legend(loc="upper right", fontsize=8)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    axes[-1].set_xlabel("UTC time")
    caption(
        fig,
        "Colored signal segments are samples inside that channel’s fused anomaly windows. "
        "Background shading: red=HIGH, orange=MODERATE, blue=SCREEN confidence.",
    )
    pdf.savefig(fig)
    plt.close(fig)


def individual_sensor_pages(pdf: PdfPages, windows: pd.DataFrame, interp: pd.DataFrame) -> None:
    specs = [
        ("CO2", "CO₂ (µatm; symlog)", "#3268a8", "High excursions indicate CO₂ enrichment."),
        ("CH4", "CH₄ (µatm; symlog)", "#6a3d9a", "High excursions indicate CH₄ enrichment."),
        ("O2", "O₂ (µM)", "#2a9d8f", "Low excursions are oriented as oxygen-depletion anomalies."),
        ("Temperature", "Temperature (°C)", "#d95f02", "High excursions indicate thermal anomalies."),
        ("Salinity", "Salinity (PSU)", "#007c91",
         "Flags combine sharp salinity transients with sustained departures from the CTD1 T–S reference curve."),
    ]
    for column, ylabel, color, interpretation in specs:
        intervals = channel_windows(windows, column)
        mask = anomaly_mask(interp.time, intervals)
        values = interp[column].to_numpy(dtype=float)
        fig, (ax, strip) = plt.subplots(
            2, 1, figsize=(14, 7.5), sharex=True,
            gridspec_kw={"height_ratios": [5, 0.65]}, constrained_layout=True
        )
        fig.suptitle(f"{column}: individually flagged anomalous regions",
                     fontsize=18, weight="bold")
        add_event_spans(ax, intervals, 0.10)
        ax.plot(interp.time, values, color="#a6adb3", lw=0.6, label="Sensor reading")
        ax.plot(interp.time, np.where(mask, values, np.nan), color=color, lw=1.25,
                label="Flagged reading")
        if column == "CH4":
            ax.set_yscale("symlog", linthresh=1)
        elif column == "CO2":
            ax.set_yscale("symlog", linthresh=1000)
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        tier_y = {"SCREEN": 1, "MODERATE": 2, "HIGH": 3}
        tier_color = {"SCREEN": "#4e79a7", "MODERATE": "#f28e2b", "HIGH": "#d73027"}
        for _, row in intervals.iterrows():
            strip.fill_between(
                [row.start_time, row.end_time], 0, tier_y[row.confidence_tier],
                color=tier_color[row.confidence_tier], alpha=0.9
            )
        strip.set_yticks([1, 2, 3], ["Screen", "Moderate", "High"])
        strip.set_ylim(0, 3.2)
        strip.grid(axis="x", alpha=0.15)
        strip.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        strip.set_xlabel("UTC time")
        caption(
            fig,
            f"{interpretation} {len(intervals)} fused windows contain this indicator. "
            "The lower strip shows the confidence assigned after cross-channel/config fusion.",
        )
        pdf.savefig(fig)
        plt.close(fig)


def detector_method_pages(pdf: PdfPages, events: pd.DataFrame, interp: pd.DataFrame) -> None:
    colors = {
        "D1_height": "#2f6db0",
        "D2_mass": "#e28e2c",
        "D3_peak": "#b51f8a",
        "D3_direct_peak": "#7b1fa2",
        "D4_local": "#2a9d62",
        "TS_curve": "#00838f",
    }
    labels = {
        "D1_height": "D1 height",
        "D2_mass": "D2 mass",
        "D3_peak": "D3 peak / prominence",
        "D3_direct_peak": "D3 direct-signal peak",
        "D4_local": "D4 local distribution",
        "TS_curve": "T–S reference-curve departure",
    }
    column_map = {
        "CO2": "CO2", "CH4": "CH4", "O2": "O2",
        "temp": "Temperature", "salinity": "Salinity",
    }
    for channel in CHANNELS:
        column = column_map[channel]
        values = interp[column].to_numpy(dtype=float)
        method_masks = {}
        for method in METHODS:
            if method in ("D3_direct_peak", "TS_curve") or channel == "salinity":
                subset = events[
                    (events.channel == channel) & (events.method == method)
                ]
                method_masks[method] = anomaly_mask(interp.time, subset)
            else:
                cfg_masks = []
                for config in CONFIGS:
                    subset = events[
                        (events.channel == channel)
                        & (events.method == method)
                        & (events.config == config)
                    ]
                    cfg_masks.append(anomaly_mask(interp.time, subset))
                method_masks[method] = np.sum(cfg_masks, axis=0) >= MIN_CONFIG_SUPPORT
        qualified = (
            method_masks["D3_peak"] | method_masks["D3_direct_peak"]
            | method_masks["TS_curve"] | (
            np.sum([method_masks[m] for m in METHODS], axis=0) >= 2
            )
        )

        fig, (ax, raster) = plt.subplots(
            2, 1, figsize=(14, 7.7), sharex=True,
            gridspec_kw={"height_ratios": [4.7, 1.25]}, constrained_layout=True
        )
        fig.suptitle(
            f"{CHANNEL_PRETTY[channel]}: detector-family evidence and retained peaks",
            fontsize=17, weight="bold"
        )
        ax.plot(interp.time, values, color="#a4abb0", lw=0.55, label="Sensor reading")
        ax.plot(interp.time, np.where(qualified, values, np.nan), color="#c62828",
                lw=1.2, label="Retained detector-aware anomaly")
        if channel == "CH4":
            ax.set_yscale("symlog", linthresh=1)
        elif channel == "CO2":
            ax.set_yscale("symlog", linthresh=1000)
        ax.set_ylabel(column)
        ax.grid(alpha=0.2)
        ax.legend(loc="upper right")
        for y, method in enumerate(METHODS, start=1):
            mask = method_masks[method]
            raster.fill_between(
                interp.time, y - 0.32, y + 0.32, where=mask,
                color=colors[method], step="mid", alpha=0.95
            )
        raster.set_yticks(range(1, len(METHODS) + 1), [labels[m] for m in METHODS])
        raster.set_ylim(0.4, len(METHODS) + 0.6)
        raster.grid(axis="x", alpha=0.15)
        raster.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        raster.set_xlabel("UTC time")
        caption(
            fig,
            "Each method row requires reproduction in ≥2 configurations. Red signal is retained "
            "where either D3 peak path is active OR at least two other detector families agree. "
            "For salinity, the CTD1 detector masks and independent T–S curve test replace "
            "cross-configuration reproduction.",
        )
        pdf.savefig(fig)
        plt.close(fig)


def ts_analysis_pages(pdf: PdfPages) -> None:
    pages = [
        (
            "TS_CTD1.png",
            "CTD1 reference temperature–salinity structure",
            "CTD cast 1 is the sole reference cast used here. Its fitted T–S relation provides "
            "the expected water-mass curve against which the dive is evaluated; other CTD casts "
            "are deliberately excluded.",
        ),
        (
            "TS_anom_salinity.png",
            "Salinity transient evidence",
            "Salinity peak evidence is sparse: D1 finds one event and D2–D4 find two each "
            "(143 union samples). These are treated as localized salinity anomalies, rather "
            "than expanding them across neighboring elevated regions.",
        ),
        (
            "TS_anom_curve.png",
            "Dive departures from the CTD1 T–S curve",
            "After removing the bulk salinity offset of −0.031 PSU, 2,041 of 85,920 samples "
            "(2.38%) depart from the CTD1 reference by more than 0.05 PSU. These flags indicate "
            "unusual water-mass properties; they are not automatically interpreted as sharp "
            "salinity sensor peaks.",
        ),
        (
            "TS_anom_both.png",
            "Transient versus sustained T–S behavior",
            "The revised interpretation separates short detector-scale excursions from sustained "
            "background differences. Temperature has 2,991 background-difference samples; "
            "salinity has none under the same baseline test. This argues against calling the "
            "entire dive broadly salinity-anomalous.",
        ),
        (
            "TS_combined.png",
            "CTD1 and dive T–S comparison",
            "The combined view is most useful for distinguishing motion along the expected "
            "water-mass mixing curve from movement across it. Cross-curve displacement receives "
            "independent salinity/T–S evidence in the catalog; along-curve temperature change "
            "remains primarily temperature evidence.",
        ),
    ]
    root = EVENT_ROOT / "ctd_ts"
    # NOTE: the loop variable is `blurb`, not `caption` — `caption` is the
    # module-level layout helper and must not be shadowed here.
    for filename, title, blurb in pages:
        image = plt.imread(root / filename)
        fig = plt.figure(figsize=(11, 8.5))
        fig.suptitle(title, fontsize=17, weight="bold", y=0.96)

        # Wrap first, then size the image band around the space the text needs,
        # so a long blurb pushes the image up instead of running off the page.
        size = 10.5
        lines = wrap_to_width(blurb, (11.0 - 2 * 0.75) * 72.0, size)
        text_h = (len(lines) * size * 1.35 / 72.0 + 0.28) / 8.5

        ax = fig.add_axes([0.06, text_h + 0.03, 0.88, 0.90 - text_h - 0.03])
        ax.imshow(image)
        ax.axis("off")
        fig.text(0.75 / 11.0, 0.12 / 8.5, "\n".join(lines),
                 fontsize=size, va="bottom", ha="left", linespacing=1.35)
        pdf.savefig(fig)
        plt.close(fig)


def trackline_page(pdf: PdfPages, windows: pd.DataFrame, nav: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 9), facecolor="#f4f1e8")
    fig.subplots_adjust(left=0.05, right=0.98, top=0.89, bottom=0.20, wspace=0.20)
    fig.suptitle("J1754 trackline with anomaly segments and priority-site callouts",
                 fontsize=18, weight="bold")
    colors = {"HIGH": "#c62828", "MODERATE": "#ef7d00", "SCREEN": "#3d6ea8"}
    widths = {"HIGH": 4.0, "MODERATE": 3.0, "SCREEN": 2.2}
    top_sites = (
        windows[windows.confidence_tier == "HIGH"]
        .sort_values(["evidence_score", "start_time"], ascending=[False, True])
        .drop_duplicates("site_id")
        .head(14)
    )
    split_lat = float(nav.lat.median())
    views = [
        ("Complete track", float(nav.lat.min()), float(nav.lat.max())),
        ("Southern detail", float(nav.lat.min()), split_lat),
        ("Northern detail", split_lat, float(nav.lat.max())),
    ]
    lon_mid = float((nav.lon.min() + nav.lon.max()) / 2)
    lon_half = float(nav.lon.max() - nav.lon.min()) * 1.65
    for ax, (title, low_lat, high_lat) in zip(axes, views):
        ax.set_facecolor("#e7eff1")
        ax.plot(nav.lon, nav.lat, color="#607d8b", lw=1.15, alpha=0.78,
                solid_capstyle="round", zorder=1)
        for tier in ("SCREEN", "MODERATE", "HIGH"):
            for _, row in windows[windows.confidence_tier == tier].sort_values("start_time").iterrows():
                segment = nav[(nav.time >= row.start_time) & (nav.time <= row.end_time)]
                if len(segment) >= 2:
                    ax.plot(segment.lon, segment.lat, color=colors[tier], lw=widths[tier],
                            alpha=0.92, solid_capstyle="round", zorder=2)
        ax.set_xlim(lon_mid - lon_half, lon_mid + lon_half)
        pad = max((high_lat - low_lat) * 0.025, 0.0002)
        ax.set_ylim(low_lat - pad, high_lat + pad)
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xlabel("Longitude")
        if ax is axes[0]:
            ax.set_ylabel("Latitude (degrees north)")
        ax.ticklabel_format(useOffset=False)
        ax.set_aspect(1 / math.cos(math.radians(float(nav.lat.mean()))))
        ax.grid(color="white", lw=1.1)
        ax.annotate("N", xy=(0.93, 0.91), xytext=(0.93, 0.82), xycoords="axes fraction",
                    ha="center", fontsize=10, weight="bold",
                    arrowprops=dict(arrowstyle="-|>", lw=1.2, color="#263238"))

        if title != "Complete track":
            panel_sites = top_sites[(top_sites.lat >= low_lat) & (top_sites.lat <= high_lat)]
            left = True
            for _, row in panel_sites.iterrows():
                text_x = lon_mid - lon_half * 0.92 if left else lon_mid + lon_half * 0.92
                align = "left" if left else "right"
                ax.annotate(
                    f"{row.site_id}  {row.evidence_score:.0f}%",
                    xy=(row.lon, row.lat), xytext=(text_x, row.lat),
                    textcoords="data", ha=align, va="center", fontsize=7.2, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.18", fc="white", ec=colors["HIGH"], alpha=0.9),
                    arrowprops=dict(arrowstyle="-", color="#70777b", lw=0.65),
                    zorder=4,
                )
                left = not left
    axes[0].scatter(nav.lon.iloc[0], nav.lat.iloc[0], marker=">", s=65, color="#1b5e20",
                    edgecolor="white", zorder=5)
    axes[0].scatter(nav.lon.iloc[-1], nav.lat.iloc[-1], marker="s", s=45, color="#212121",
                    edgecolor="white", zorder=5)

    from matplotlib.lines import Line2D
    legend = [
        Line2D([0], [0], color="#607d8b", lw=1.2, label="Trackline"),
        Line2D([0], [0], color=colors["HIGH"], lw=4, label="HIGH anomaly"),
        Line2D([0], [0], color=colors["MODERATE"], lw=3, label="MODERATE anomaly"),
        Line2D([0], [0], color=colors["SCREEN"], lw=2.2, label="SCREEN anomaly"),
    ]
    axes[0].legend(handles=legend, loc="lower left", fontsize=8, framealpha=0.95)
    key_lines = [
        f"{r.site_id}: {r.indicator_signature} ({r.anomaly_class})"
        for _, r in top_sites.iterrows()
    ]
    # Wrap each key entry to its own column so long indicator signatures cannot
    # run into the neighbouring column or off the right edge.
    key_size = 7.2
    col_pt = 0.41 * 15.0 * 72.0
    wrapped_keys: list[str] = []
    for line in key_lines:
        parts = wrap_to_width(line, col_pt, key_size)
        wrapped_keys.append(parts[0])
        wrapped_keys.extend("    " + p for p in parts[1:])   # hanging indent
    midpoint = math.ceil(len(wrapped_keys) / 2)
    fig.text(0.07, 0.135, "\n".join(wrapped_keys[:midpoint]), fontsize=key_size, va="top")
    fig.text(0.52, 0.135, "\n".join(wrapped_keys[midpoint:]), fontsize=key_size, va="top")

    cap_size = 8.5
    cap_lines = wrap_to_width(
        "Highlighted geometry is the actual navigated line during each anomaly window—not "
        "a midpoint symbol. Detail panels and the callout key identify the strongest unique sites.",
        (15.0 - 2 * 1.05) * 72.0, cap_size,
    )
    fig.text(0.07, 0.02, "\n".join(cap_lines), fontsize=cap_size,
             va="bottom", linespacing=1.35)
    pdf.savefig(fig)
    plt.close(fig)


def build_report(
    windows: pd.DataFrame,
    sites: pd.DataFrame,
    clips: pd.DataFrame,
    interp: pd.DataFrame,
    nav: pd.DataFrame,
    events: pd.DataFrame,
) -> None:
    report = OUT / "Anomaly_Site_and_Video_Review_Report.pdf"
    counts = windows.confidence_tier.value_counts()
    high = int(counts.get("HIGH", 0))
    moderate = int(counts.get("MODERATE", 0))
    screen = int(counts.get("SCREEN", 0))
    with PdfPages(report) as pdf:
        text_page(
            pdf,
            "J1754 Anomaly Site Catalog and Video Review Queue",
            [
                f"Outcome. The fused catalog contains {len(windows)} anomaly windows at "
                f"{len(sites)} spatial sites: {high} HIGH, {moderate} MODERATE, and "
                f"{screen} SCREEN candidates. Every row is tied to corrected raw "
                "navigation and a review interval with two minutes of context on both sides.",
                "Decision rule. Events use the original FINE regime (5-sigma entry, "
                "2-sigma exit, 20-second minimum, 4000-second baseline). Each detector "
                "family is evaluated separately across its 20 smoother/baseline strategies "
                "at 25% within-family agreement. D3 peak/prominence support qualifies a "
                "peak directly. CH4 also has a direct-signal D3 path on a five-second "
                "median-smoothed log signal (prominence >=0.25, local increase >=8% or "
                "absolute increase >=0.25), plus a narrow raw-log prominence path for "
                "one- or two-sample peaks; non-D3 regions require at least two detector families. "
                "Every gas/temperature family must reproduce in at least two processing "
                "configurations. Salinity is evaluated separately with CTD1-calibrated D1–D4 "
                "detectors plus an independent T–S reference-curve departure test. "
                "Windows end whenever the exact active channel combination changes; no "
                "transitive cross-channel union is used. Evidence score combines channel "
                "breadth, configuration robustness, detector-family diversity, within-family "
                "consensus, and CTD1 corroboration."
                "HIGH is >=65; MODERATE is >=40; SCREEN retains every remaining positive event.",
                "Interpretation. HIGH and MODERATE windows are the primary video-review queue. "
                "SCREEN rows ensure no one-channel event is silently discarded. These are "
                "conclusive screening labels, not confirmed geological or biological "
                "abnormalities; video and photogrammetric inspection supplies that confirmation.",
                "Why this differs from the prior catalog. The earlier catalog consumed the "
                "fine+coarse 25%-consensus event CSVs and transitively merged all overlapping "
                "channels/configurations. The first correction then overcompensated by requiring "
                "40/80 agreement across unlike detectors, which suppressed D3-only peaks. This "
                "edition preserves detector-family evidence, explicitly includes D3 peak finder, "
                "and retains exact SINGLE/PAIR/TRIPLET/FOUR-/FIVE-CHANNEL classes.",
                "Salinity and T–S interpretation. Sharp salinity transients and departures "
                "from the CTD1 T–S reference curve are retained as separate evidence paths. "
                "The T–S curve path is gap-bridged only over five seconds and must persist for "
                "at least ten seconds. A curve departure means unusual water-mass properties, "
                "not necessarily a salinity sensor spike or hydrothermal source.",
                "Navigation correction. interp_full.csv contains longitude=0 and invalid UTM "
                "coordinates. Site coordinates in this product come from the correct raw "
                "renavigation longitude column, joined by timestamp. Do not use the current "
                "interp_full longitude/easting/northing for site mapping.",
                f"Actionable clips. Long parent windows are divided into at most "
                f"{MAX_VIDEO_CLIP_SECONDS // 60}-minute anomaly clips, each with one minute "
                "of visual context on both sides. video_review_clips.csv is the operational "
                "review list; anomaly_windows_all.csv preserves the unsplit scientific events.",
                "Video timing. No video files or video manifest are configured in this workspace. "
                "The review queue therefore reports absolute UTC intervals. If the video clock is "
                "UTC-synchronized, inspect those intervals directly; otherwise a video-to-UTC "
                "offset must be supplied before frame/timecode fields can be generated.",
            ],
        )

        stacked_sensor_page(pdf, windows, interp)
        individual_sensor_pages(pdf, windows, interp)
        detector_method_pages(pdf, events, interp)
        ts_analysis_pages(pdf)
        trackline_page(pdf, windows, nav)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        class_counts = (
            windows.groupby(["anomaly_class", "confidence_tier"]).size()
            .unstack(fill_value=0)
            .reindex(columns=["HIGH", "MODERATE", "SCREEN"], fill_value=0)
            .sort_values(["HIGH", "MODERATE"], ascending=False)
        )
        class_counts.plot(
            kind="barh", stacked=True, ax=ax,
            color=["#c62828", "#ef7d00", "#3d6ea8"]
        )
        ax.invert_yaxis()
        ax.set_xlabel("Fused anomaly windows")
        ax.set_ylabel("")
        # Category names are long ("FIVE-CHANNEL: CO2↑ + CH4↑ + …"); reserve a
        # left margin wide enough for the widest one so no label is clipped.
        tick_size = 8.0
        ax.tick_params(axis="y", labelsize=tick_size)
        widest_label = max(
            (text_width_pt(str(name), tick_size) for name in class_counts.index),
            default=0.0,
        )
        left_frac = min(0.62, (widest_label + 16.0) / (11.0 * 72.0))
        fig.subplots_adjust(left=left_frac, right=0.98, top=0.90)
        # Title on the FIGURE, not the axes: the axes are pushed far right by the
        # label margin, so an axes-centred title would overflow the page.
        fig.suptitle("Anomaly classification: indicator combinations and confidence",
                     fontsize=15, weight="bold", y=0.965)
        ax.grid(axis="x", alpha=0.2)
        ax.legend(title="Confidence")
        caption(
            fig,
            "Classes are exhaustive exact combinations. A pair ends when a third channel "
            "becomes active; the resulting triplet is a separate row. Indicator combinations "
            "describe observations, not a proven geological or biological cause.",
        )
        pdf.savefig(fig)
        plt.close(fig)

        combo_table = (
            windows.groupby(
                ["combination_order", "channel_combination", "indicator_signature",
                 "confidence_tier"],
                as_index=False,
            )
            .agg(
                Windows=("window_id", "count"),
                Seconds=("duration_s", "sum"),
                Sites=("site_id", "nunique"),
            )
            .sort_values(["combination_order", "channel_combination", "confidence_tier"])
        )
        combo_table.columns = [
            "N", "Combination", "Indicator signature", "Tier", "Windows", "Seconds", "Sites"
        ]
        table_pages(pdf, combo_table, "Exact single/pair/triplet/four-channel classifications", 24)

        fig, ax = plt.subplots(figsize=(11, 8.5))
        colors = {"HIGH": "#c62828", "MODERATE": "#ef6c00", "SCREEN": "#547aa5"}
        for tier in ("SCREEN", "MODERATE", "HIGH"):
            group = windows[windows.confidence_tier == tier]
            ax.scatter(group.mid_time, group.evidence_score, s=18 + 8 * group.channel_count,
                       c=colors[tier], label=f"{tier} ({len(group)})", alpha=0.75)
        ax.set_title("Fused anomaly windows over time")
        ax.set_ylabel("Evidence score")
        ax.set_xlabel("UTC time")
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d\n%H:%M"))
        ax.grid(alpha=0.25)
        ax.legend()
        caption(fig, "Marker size increases with the number of anomalous sensor channels. "
                     "HIGH/MODERATE windows form the primary video-review queue.")
        pdf.savefig(fig)
        plt.close(fig)

        # Select the 30 strongest windows by evidence, then present them in
        # CHRONOLOGICAL order so the table reads as a sequence along the dive.
        # window_id is already assigned by time, so this is also ID order.
        top = (
            windows.sort_values(["evidence_score", "start_time"], ascending=[False, True])
            .head(30)
            .sort_values("start_time")
            .copy()
        )
        top["UTC anomaly window"] = top.apply(
            lambda r: f"{r.start_time:%m-%d %H:%M:%S}–{r.end_time:%H:%M:%S}", axis=1
        )
        top_table = top[
            ["window_id", "site_id", "confidence_tier", "evidence_score", "UTC anomaly window",
             "indicator_signature", "anomaly_class", "config_count", "max_consensus",
             "ctd1_corroboration"]
        ].rename(columns={
            "window_id": "Window",
            "site_id": "Site", "confidence_tier": "Tier", "evidence_score": "Score",
            "indicator_signature": "Indicators", "anomaly_class": "Class",
            "config_count": "Configs",
            "max_consensus": "Consensus", "ctd1_corroboration": "CTD1 evidence",
        })
        table_pages(pdf, top_table,
                    "Highest-priority anomaly windows (top 30 by evidence, in time order)", 15)

        # Chronological: a reviewer scrubs the video forward once, in order,
        # rather than jumping around the dive by tier.  Tier and score stay as
        # columns so priority is still visible on every row.
        queue = clips[clips.confidence_tier.isin(["HIGH", "MODERATE"])].sort_values(
            "review_start"
        ).copy()
        queue["Review UTC"] = queue.apply(
            lambda r: f"{r.review_start:%m-%d %H:%M:%S}–{r.review_end:%H:%M:%S}", axis=1
        )
        queue_table = queue[
            ["clip_id", "site_id", "confidence_tier", "evidence_score", "Review UTC",
             "indicator_signature", "anomaly_class", "lat", "lon", "altitude_m"]
        ].copy()
        queue_table["lat"] = queue_table["lat"].map(lambda x: f"{x:.6f}")
        queue_table["lon"] = queue_table["lon"].map(lambda x: f"{x:.6f}")
        queue_table["altitude_m"] = queue_table["altitude_m"].map(lambda x: f"{x:.1f}")
        queue_table.columns = ["Clip", "Site", "Tier", "Score", "Video review (UTC)",
                               "Indicators", "Class", "Latitude", "Longitude", "Alt m"]
        table_pages(pdf, queue_table,
                    "Primary video-review clips (HIGH + MODERATE, in time order)", 20)

        # Keep the 60 strongest sites, then list them by first encounter so the
        # catalog follows the trackline rather than the score ranking.
        site_table = sites.head(60).sort_values("first_time").copy()
        site_table["lat"] = site_table.lat.map(lambda x: f"{x:.6f}")
        site_table["lon"] = site_table.lon.map(lambda x: f"{x:.6f}")
        site_table["First seen (UTC)"] = site_table.first_time.map(
            lambda t: f"{t:%m-%d %H:%M:%S}"
        )
        site_table = site_table[
            ["site_id", "First seen (UTC)", "best_tier", "max_evidence_score",
             "window_count", "high_windows", "moderate_windows", "channels", "lat", "lon"]
        ]
        site_table.columns = ["Site", "First seen (UTC)", "Best tier", "Max score",
                              "Windows", "High", "Moderate", "Channels",
                              "Latitude", "Longitude"]
        table_pages(pdf, site_table, "Spatial site catalog (in order of first encounter)", 22)

        text_page(
            pdf,
            "Required video-review annotation",
            [
                "For each HIGH and MODERATE queue row, record: video file, first and last "
                "frame/timecode inspected, visibility/lighting quality, seafloor visible "
                "(yes/no), biological feature, geological feature, fluid/turbidity evidence, "
                "vehicle manipulation or sensor-handling evidence, and reviewer confidence.",
                "Use four outcome labels: ABNORMALITY PRESENT, NO VISIBLE ABNORMALITY, "
                "UNINTERPRETABLE, or VEHICLE/INSTRUMENT ARTIFACT. Preserve the SCREEN queue "
                "for a second-pass sensitivity review or blinded negative/control sampling.",
                "After review, the scientifically strongest dataset will join these visual "
                "labels back to anomaly_windows_all.csv. That labeled table becomes suitable "
                "for estimating precision by tier and for training or evaluating image-based "
                "models without conflating statistical anomalies with visually confirmed sites.",
            ],
        )


def preflight() -> list[str]:
    """Return a list of human-readable problems with the configured inputs.

    Empty list means the run can proceed.  Callers (the app) use this to fail
    fast with a clear message instead of a mid-pipeline traceback.
    """
    problems: list[str] = []
    if not INTERP.is_file():
        problems.append(f"Interpolated sensor table not found: {INTERP}")
    else:
        # Fail with an actionable message rather than a pandas usecols traceback.
        required = ["timestamp_iso", "alt", "water_depth",
                    "CO2", "CH4", "O2", "Temperature", "Salinity"]
        try:
            header = pd.read_csv(INTERP, nrows=0).columns.tolist()
        except Exception as exc:                      # noqa: BLE001
            problems.append(f"Could not read {INTERP}: {exc}")
        else:
            missing = [c for c in required if c not in header]
            if missing:
                problems.append(
                    f"{INTERP.name} is missing required column(s): "
                    f"{', '.join(missing)} (at {INTERP}). "
                    "The catalog needs an interp table that includes Salinity; "
                    "rebuild interp_full.csv with the salinity channel configured."
                )
    if not RAW_NAV.is_file():
        problems.append(f"Raw navigation file not found: {RAW_NAV}")
    if not EVENT_ROOT.is_dir():
        problems.append(f"Event root not found: {EVENT_ROOT}")
    else:
        found = sorted(EVENT_ROOT.glob("*/fine_core_events_*.csv"))
        if not found:
            problems.append(
                f"No detector event CSVs under {EVENT_ROOT} "
                "(run the MATLAB detector + exporters first)"
            )
    # TS_RESULTS is optional: it only adds CTD1 corroboration.
    return problems


def run(log=print) -> dict:
    """Run the full catalog pipeline against the configured paths.

    Returns a summary dict (counts + output paths) for the caller to report.
    Raises FileNotFoundError if preflight fails.
    """
    problems = preflight()
    if problems:
        raise FileNotFoundError("; ".join(problems))

    if not TS_RESULTS.is_file():
        log(f"  note: {TS_RESULTS.name} absent - CTD1 T/S corroboration disabled")

    log("  loading navigation and interpolated sensors ...")
    interp, nav = load_navigation()
    log("  loading detector events ...")
    events = load_events(interp)
    ctd_flags = load_ctd1_flags(interp)
    log("  fusing events into signature windows ...")
    windows = pd.DataFrame(build_exact_signature_windows(events, interp))
    windows = add_context(windows, interp, nav, ctd_flags)
    log("  assigning spatial sites ...")
    windows, sites = assign_sites(windows)
    clips = make_video_clips(windows)
    log("  writing CSV/GeoJSON/QGIS outputs ...")
    write_outputs(windows, sites, clips, nav)
    log("  building PDF report ...")
    build_report(windows, sites, clips, interp, nav, events)

    counts = windows.confidence_tier.value_counts()
    summary = {
        "windows": int(len(windows)),
        "clips": int(len(clips)),
        "sites": int(len(sites)),
        "high": int(counts.get("HIGH", 0)),
        "moderate": int(counts.get("MODERATE", 0)),
        "screen": int(counts.get("SCREEN", 0)),
        "out_dir": OUT,
        "report_pdf": OUT / "Anomaly_Site_and_Video_Review_Report.pdf",
        "windows_csv": OUT / "anomaly_windows_all.csv",
        "sites_csv": OUT / "anomalous_sites.csv",
        "sites_geojson": OUT / "anomalous_sites.geojson",
        "clips_csv": OUT / "video_review_clips.csv",
        "qgis_zip": OUT / "qgis" / "J1754_anomaly_QGIS_upload.zip",
    }
    log(
        f"  wrote {summary['windows']} windows, {summary['clips']} review clips, "
        f"{summary['sites']} sites ({summary['high']} HIGH, "
        f"{summary['moderate']} MODERATE, {summary['screen']} SCREEN) to {OUT}"
    )
    return summary


def main(argv: list[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, help="repository/workspace root")
    parser.add_argument("--event-root", type=Path, help="dir holding <config>/fine_*_events_*.csv")
    parser.add_argument("--interp", type=Path, help="interpolated sensor CSV")
    parser.add_argument("--raw-nav", type=Path, help="raw 1 Hz navigation CSV")
    parser.add_argument("--ts-results", type=Path, help="ts_analysis_results.mat (optional)")
    parser.add_argument("--out", type=Path, help="output directory")
    args = parser.parse_args(argv)

    configure(
        repo=args.repo, event_root=args.event_root, interp=args.interp,
        raw_nav=args.raw_nav, ts_results=args.ts_results, out=args.out,
    )
    run()


if __name__ == "__main__":
    main()
