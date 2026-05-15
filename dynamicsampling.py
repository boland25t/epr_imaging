import math
import numpy as np
import pandas as pd


EARTH_RADIUS_M = 6371008.8


def _series_to_epoch_seconds(s: pd.Series) -> np.ndarray:
    """
    Accepts either numeric unix seconds or datetime-like timestamps.
    Returns float epoch seconds.
    """
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float).to_numpy()

    dt = pd.to_datetime(s, utc=True, errors="raise")
    return (dt.astype("int64") / 1e9).to_numpy(dtype=float)


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """
    Vectorized great-circle distance in meters.
    Lat/lon are decimal degrees.
    """
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def create_dynamic_sample_schedule(
    nav_df: pd.DataFrame,
    *,
    time_col: str = "unix_time",
    lat_col: str = "lat",
    lon_col: str = "lon",
    alt_col: str | None = "alt",

    # Use this if you already know the desired distance between sampled frames.
    target_spacing_m: float | None = None,

    # Use these if you want spacing computed from camera footprint.
    along_track_fov_deg: float | None = None,
    overlap_fraction: float | None = None,
    target_overlap_m: float | None = None,

    # Lower bound on sampling frequency.
    min_frequency_hz: float = 0.1,

    # Safety controls.
    min_spacing_m: float = 0.25,
    pause_when_stationary: bool = False,
    stationary_speed_mps: float = 0.03,

    include_first: bool = True,
    include_last: bool = False,
) -> pd.DataFrame:
    """
    Creates a list of target sample timestamps based on vehicle speed and desired
    real-world frame spacing / overlap.

    Returns a DataFrame with:
        - sample_time_s
        - relative_time_s
        - track_distance_m
        - altitude_m
        - speed_mps
        - target_spacing_m
        - effective_frequency_hz
        - reason

    Modes:

    1. Constant real-world spacing:
        target_spacing_m=2.0

    2. Constant percent overlap from altitude and FOV:
        along_track_fov_deg=90
        overlap_fraction=0.75

    3. Approximately constant physical overlap length:
        along_track_fov_deg=90
        target_overlap_m=4.0

    Notes:
        min_frequency_hz is a true lower bound unless pause_when_stationary=True.
    """

    if nav_df.empty:
        return pd.DataFrame()

    required = [time_col, lat_col, lon_col]
    if alt_col is not None:
        required.append(alt_col)

    missing = [c for c in required if c not in nav_df.columns]
    if missing:
        raise ValueError(f"Missing required navigation columns: {missing}")

    df = nav_df[required].copy()
    df = df.dropna(subset=required)

    if df.empty:
        return pd.DataFrame()

    df["_t"] = _series_to_epoch_seconds(df[time_col])
    df = df.sort_values("_t").drop_duplicates("_t").reset_index(drop=True)

    if len(df) < 2:
        return pd.DataFrame()

    t = df["_t"].to_numpy(dtype=float)
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)

    if alt_col is not None:
        alt = df[alt_col].to_numpy(dtype=float)
    else:
        alt = np.zeros(len(df), dtype=float)

    dt = np.diff(t)
    valid_dt = dt > 0

    if not np.all(valid_dt):
        raise ValueError("Navigation timestamps must be strictly increasing.")

    horizontal_m = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])
    vertical_m = np.diff(alt)
    segment_m = np.sqrt(horizontal_m**2 + vertical_m**2)

    cumulative_m = np.concatenate([[0.0], np.cumsum(segment_m)])
    segment_speed_mps = np.divide(
        segment_m,
        dt,
        out=np.zeros_like(segment_m),
        where=dt > 0,
    )

    max_interval_s = 1.0 / min_frequency_hz if min_frequency_hz > 0 else math.inf

    def spacing_for_altitude(altitude_m: float) -> float:
        if target_spacing_m is not None:
            return max(float(target_spacing_m), min_spacing_m)

        if along_track_fov_deg is None:
            raise ValueError(
                "Provide either target_spacing_m, or along_track_fov_deg plus "
                "overlap_fraction or target_overlap_m."
            )

        footprint_m = 2.0 * max(float(altitude_m), 0.0) * math.tan(
            math.radians(along_track_fov_deg) / 2.0
        )

        if target_overlap_m is not None:
            spacing_m = footprint_m - float(target_overlap_m)
        elif overlap_fraction is not None:
            if not 0.0 <= overlap_fraction < 1.0:
                raise ValueError("overlap_fraction must be in [0, 1).")
            spacing_m = footprint_m * (1.0 - float(overlap_fraction))
        else:
            raise ValueError(
                "When using along_track_fov_deg, provide either "
                "overlap_fraction or target_overlap_m."
            )

        return max(spacing_m, min_spacing_m)

    samples = []

    def add_sample(sample_t, sample_dist, sample_alt, speed_mps, spacing_m, reason):
        if samples:
            previous_t = samples[-1]["sample_time_s"]
            effective_frequency_hz = 1.0 / max(sample_t - previous_t, 1e-9)
        else:
            effective_frequency_hz = np.nan

        samples.append(
            {
                "sample_time_s": float(sample_t),
                "relative_time_s": float(sample_t - t[0]),
                "track_distance_m": float(sample_dist),
                "altitude_m": float(sample_alt),
                "speed_mps": float(speed_mps),
                "target_spacing_m": float(spacing_m),
                "effective_frequency_hz": float(effective_frequency_hz)
                if not np.isnan(effective_frequency_hz)
                else np.nan,
                "reason": reason,
            }
        )

    if include_first:
        first_spacing = spacing_for_altitude(alt[0])
        add_sample(
            sample_t=t[0],
            sample_dist=0.0,
            sample_alt=alt[0],
            speed_mps=segment_speed_mps[0],
            spacing_m=first_spacing,
            reason="first",
        )
        last_sample_t = t[0]
        last_sample_dist = 0.0
        next_target_dist = first_spacing
    else:
        last_sample_t = t[0]
        last_sample_dist = 0.0
        next_target_dist = spacing_for_altitude(alt[0])

    for i in range(1, len(t)):
        t0 = t[i - 1]
        t1 = t[i]
        dt_i = t1 - t0

        dist0 = cumulative_m[i - 1]
        dist1 = cumulative_m[i]
        seg_dist = dist1 - dist0

        alt0 = alt[i - 1]
        alt1 = alt[i]

        speed_i = segment_speed_mps[i - 1]

        while True:
            t_by_distance = math.inf

            if seg_dist > 0 and next_target_dist <= dist1:
                frac = (next_target_dist - dist0) / seg_dist
                if 0.0 <= frac <= 1.0:
                    t_by_distance = t0 + frac * dt_i

            t_by_frequency = last_sample_t + max_interval_s

            if pause_when_stationary and speed_i < stationary_speed_mps:
                t_by_frequency = math.inf

            if t_by_frequency < t0:
                t_by_frequency = t0

            if t_by_frequency > t1:
                t_by_frequency = math.inf

            next_event_t = min(t_by_distance, t_by_frequency)

            if not math.isfinite(next_event_t):
                break

            if next_event_t <= last_sample_t + 1e-9:
                break

            frac_t = (next_event_t - t0) / dt_i
            frac_t = min(max(frac_t, 0.0), 1.0)

            event_dist = dist0 + frac_t * seg_dist
            event_alt = alt0 + frac_t * (alt1 - alt0)

            spacing_m = spacing_for_altitude(event_alt)

            if t_by_distance <= t_by_frequency:
                reason = "distance"
            else:
                reason = "min_frequency"

            add_sample(
                sample_t=next_event_t,
                sample_dist=event_dist,
                sample_alt=event_alt,
                speed_mps=speed_i,
                spacing_m=spacing_m,
                reason=reason,
            )

            last_sample_t = next_event_t
            last_sample_dist = event_dist
            next_target_dist = last_sample_dist + spacing_m

    if include_last:
        last_spacing = spacing_for_altitude(alt[-1])
        if not samples or t[-1] > samples[-1]["sample_time_s"] + 1e-9:
            add_sample(
                sample_t=t[-1],
                sample_dist=cumulative_m[-1],
                sample_alt=alt[-1],
                speed_mps=segment_speed_mps[-1],
                spacing_m=last_spacing,
                reason="last",
            )

    return pd.DataFrame(samples)
