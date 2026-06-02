# dynamicsampling.py — Distance-adaptive frame sample schedule generator
#
# In "dynamic sampling" mode, instead of extracting frames at a fixed clock
# interval (e.g. every 1 second), the pipeline extracts frames at the moments
# when the vehicle has travelled a specified real-world distance (e.g. every
# 2 metres along-track).  This keeps frame overlap approximately constant
# regardless of vehicle speed, which is important for photogrammetric mosaics
# and benthic habitat surveys.
#
# create_dynamic_sample_schedule() is the public entry point.  It takes a
# navigation DataFrame and returns a list of target sample timestamps (and
# diagnostic metadata) that the pipeline uses to seek to specific video frames.
#
# Spacing can be specified three ways:
#   1. Constant along-track distance (target_spacing_m).
#   2. Constant percentage overlap derived from altitude and camera FOV.
#   3. Constant physical overlap length derived from altitude and camera FOV.
#
# A min_frequency_hz floor ensures the schedule never goes longer than
# 1/min_frequency_hz seconds between samples, even when the vehicle is moving
# very slowly.

import math
import numpy as np
import pandas as pd


# Mean radius of the Earth in metres, used for the haversine formula.
# This value is the IUGG mean radius, accurate to within 0.1% globally.
EARTH_RADIUS_M = 6371008.8


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _series_to_epoch_seconds(s: pd.Series) -> np.ndarray:
    """Convert a pandas Series to a float64 array of Unix epoch seconds.

    Accepts two forms:
      - Numeric (int or float) columns already in Unix seconds.
      - Datetime-like strings or pandas Timestamp objects, which are converted
        via pd.to_datetime with UTC interpretation.

    Args:
        s: A pandas Series containing timestamps in either numeric or
           datetime-like format.

    Returns:
        float64 numpy array of seconds since 1970-01-01T00:00:00 UTC.
    """
    if pd.api.types.is_numeric_dtype(s):
        # Already numeric — treat as Unix seconds and return as-is.
        return s.astype(float).to_numpy()

    # Parse as timezone-aware (UTC) datetime so the epoch conversion is
    # unambiguous regardless of the local system timezone.
    dt = pd.to_datetime(s, utc=True, errors="raise")
    return (dt.astype("int64") / 1e9).to_numpy(dtype=float)


def _haversine_m(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Vectorised great-circle distance in metres between coordinate pairs.

    Uses the haversine formula, which is numerically stable for both very
    short (sub-metre) and very long distances.

    Args:
        lat1, lon1: Starting coordinates in decimal degrees (scalar or array).
        lat2, lon2: Ending coordinates in decimal degrees (scalar or array).

    Returns:
        Distance in metres (same shape as the inputs).
    """
    # Convert decimal degrees to radians for the trig functions.
    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    # Haversine formula: a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    )

    # d = 2R · arcsin(√a)
    return 2.0 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_dynamic_sample_schedule(
    nav_df: pd.DataFrame,
    *,
    # Column names in nav_df to use for time, position, and altitude.
    time_col: str = "unix_time",
    lat_col:  str = "lat",
    lon_col:  str = "lon",
    alt_col:  str | None = "alt",

    # --- Spacing specification (choose one mode) ---

    # Mode 1: Direct distance — sample every N metres along-track.
    target_spacing_m: float | None = None,

    # Mode 2 & 3: Camera-footprint-derived spacing.  Requires altitude data.
    along_track_fov_deg: float | None = None,   # Half-angle of camera FOV along track
    overlap_fraction:    float | None = None,   # Desired overlap as fraction [0, 1)
    target_overlap_m:    float | None = None,   # Desired overlap in metres

    # Minimum sampling rate — guarantees at least one sample per 1/N seconds
    # even when the vehicle is nearly stationary.
    min_frequency_hz: float = 0.1,

    # Safety floor: never produce a spacing smaller than this, regardless of
    # altitude or overlap settings, to avoid sampling at unreasonably high rates.
    min_spacing_m: float = 0.25,

    # When True, frequency-based sampling is suppressed when the vehicle speed
    # is below stationary_speed_mps (avoids bursts of frames while hovering).
    pause_when_stationary: bool = False,
    stationary_speed_mps:  float = 0.03,

    # Whether to force a sample at the very first and last nav points.
    include_first: bool = True,
    include_last:  bool = False,
) -> pd.DataFrame:
    """Generate a list of target sample timestamps for distance-adaptive frame extraction.

    The algorithm walks the vehicle's navigation track and places a sample
    whenever the vehicle has covered target_spacing_m metres since the last
    sample, or when 1/min_frequency_hz seconds have elapsed since the last
    sample (whichever comes first).

    The spacing at each point can be constant (Mode 1) or altitude-dependent
    (Modes 2/3), allowing the frame footprint overlap to remain constant when
    the vehicle changes altitude (e.g. during descent/ascent).

    Args:
        nav_df:               DataFrame containing navigation data.
        time_col:             Name of the Unix-seconds timestamp column.
        lat_col, lon_col:     Names of the latitude and longitude columns.
        alt_col:              Name of the altitude/depth column, or None for 2-D.
        target_spacing_m:     Desired along-track spacing in metres (Mode 1).
        along_track_fov_deg:  Camera FOV half-angle along track in degrees (Modes 2/3).
        overlap_fraction:     Desired fractional overlap [0, 1) (Mode 2).
        target_overlap_m:     Desired overlap in metres (Mode 3).
        min_frequency_hz:     Lower bound on sampling rate (samples per second).
        min_spacing_m:        Hard floor on computed spacing (metres).
        pause_when_stationary: Suppress frequency-floor sampling when hovering.
        stationary_speed_mps: Speed threshold below which the vehicle is "hovering".
        include_first:        Always add a sample at the first nav point.
        include_last:         Always add a sample at the last nav point.

    Returns:
        A DataFrame with one row per planned sample and columns:
          - sample_time_s       : Unix epoch seconds at which to extract the frame
          - relative_time_s     : Seconds since the first nav point
          - track_distance_m    : Cumulative along-track distance in metres
          - altitude_m          : Interpolated altitude at this sample
          - speed_mps           : Vehicle speed in the preceding segment (m/s)
          - target_spacing_m    : The spacing that triggered this sample
          - effective_frequency_hz : Actual sampling rate since the previous sample
          - reason              : "first", "last", "distance", or "min_frequency"

        Returns an empty DataFrame if nav_df is empty or has fewer than 2 rows.

    Raises:
        ValueError: If required columns are missing, timestamps are non-monotone,
                    or an invalid combination of spacing parameters is given.
    """

    # --- Input validation ---

    if nav_df.empty:
        return pd.DataFrame()

    # Build the list of required columns based on whether altitude is provided.
    required = [time_col, lat_col, lon_col]
    if alt_col is not None:
        required.append(alt_col)

    missing = [c for c in required if c not in nav_df.columns]
    if missing:
        raise ValueError(f"Missing required navigation columns: {missing}")

    # Work on a copy to avoid mutating the caller's DataFrame.
    df = nav_df[required].copy()

    # Drop rows where any required column is NaN; we can't compute distances
    # or speeds without complete position data.
    df = df.dropna(subset=required)

    if df.empty:
        return pd.DataFrame()

    # Convert timestamps to float64 Unix seconds and sort/deduplicate.
    df["_t"] = _series_to_epoch_seconds(df[time_col])
    df = df.sort_values("_t").drop_duplicates("_t").reset_index(drop=True)

    # Need at least two points to define a track segment.
    if len(df) < 2:
        return pd.DataFrame()

    # --- Extract numpy arrays for fast vectorised operations ---

    t   = df["_t"].to_numpy(dtype=float)
    lat = df[lat_col].to_numpy(dtype=float)
    lon = df[lon_col].to_numpy(dtype=float)

    # If no altitude column is configured, treat the vehicle as flying at sea
    # level (altitude=0).  This is fine for Mode 1 (constant spacing) and
    # produces a constant footprint in Mode 2/3.
    if alt_col is not None:
        alt = df[alt_col].to_numpy(dtype=float)
    else:
        alt = np.zeros(len(df), dtype=float)

    # --- Compute per-segment geometry ---

    dt = np.diff(t)  # Duration of each segment in seconds.

    # Validate monotone timestamps after deduplication.
    valid_dt = dt > 0
    if not np.all(valid_dt):
        raise ValueError("Navigation timestamps must be strictly increasing.")

    # Horizontal distance between consecutive nav points (ignoring altitude).
    horizontal_m = _haversine_m(lat[:-1], lon[:-1], lat[1:], lon[1:])

    # Vertical distance between consecutive nav points.
    vertical_m = np.diff(alt)

    # True 3-D along-track distance per segment (Pythagoras in h + v space).
    segment_m = np.sqrt(horizontal_m**2 + vertical_m**2)

    # Cumulative distance from the start of the track to each nav point.
    # cumulative_m[0] = 0.0 (start), cumulative_m[-1] = total track length.
    cumulative_m = np.concatenate([[0.0], np.cumsum(segment_m)])

    # Speed in m/s for each segment; 0 where dt=0 (degenerate segments).
    segment_speed_mps = np.divide(
        segment_m, dt,
        out=np.zeros_like(segment_m),
        where=dt > 0,
    )

    # Maximum time gap between samples before the frequency floor fires.
    max_interval_s = 1.0 / min_frequency_hz if min_frequency_hz > 0 else math.inf

    # --- Spacing function (dispatches to the requested mode) ---

    def spacing_for_altitude(altitude_m: float) -> float:
        """Return the desired along-track spacing in metres at the given altitude.

        Selects the appropriate mode based on which parameters were supplied.
        Enforces min_spacing_m as a hard floor.
        """
        # Mode 1: constant spacing, independent of altitude.
        if target_spacing_m is not None:
            return max(float(target_spacing_m), min_spacing_m)

        # Modes 2 & 3 require the camera FOV parameter.
        if along_track_fov_deg is None:
            raise ValueError(
                "Provide either target_spacing_m, or along_track_fov_deg plus "
                "overlap_fraction or target_overlap_m."
            )

        # Compute the along-track footprint of the camera at this altitude.
        # footprint = 2 * altitude * tan(FOV/2)
        footprint_m = 2.0 * max(float(altitude_m), 0.0) * math.tan(
            math.radians(along_track_fov_deg) / 2.0
        )

        # Mode 3: constant overlap length.
        if target_overlap_m is not None:
            spacing_m = footprint_m - float(target_overlap_m)

        # Mode 2: constant fractional overlap.
        elif overlap_fraction is not None:
            if not 0.0 <= overlap_fraction < 1.0:
                raise ValueError("overlap_fraction must be in [0, 1).")
            # spacing = footprint * (1 - overlap); overlap fraction of the
            # footprint is shared between consecutive frames.
            spacing_m = footprint_m * (1.0 - float(overlap_fraction))

        else:
            raise ValueError(
                "When using along_track_fov_deg, provide either "
                "overlap_fraction or target_overlap_m."
            )

        return max(spacing_m, min_spacing_m)

    # --- Sample accumulator ---

    samples = []  # Each entry is a dict that becomes one row of the output DataFrame.

    def add_sample(sample_t, sample_dist, sample_alt, speed_mps, spacing_m, reason):
        """Append one sample to the accumulator.

        Computes the effective sampling frequency as 1 / elapsed_since_previous.
        """
        if samples:
            # Time since the previous sample; avoid division by zero.
            previous_t = samples[-1]["sample_time_s"]
            effective_frequency_hz = 1.0 / max(sample_t - previous_t, 1e-9)
        else:
            effective_frequency_hz = np.nan  # No previous sample to compare against.

        samples.append({
            "sample_time_s":          float(sample_t),
            "relative_time_s":        float(sample_t - t[0]),
            "track_distance_m":       float(sample_dist),
            "altitude_m":             float(sample_alt),
            "speed_mps":              float(speed_mps),
            "target_spacing_m":       float(spacing_m),
            "effective_frequency_hz": float(effective_frequency_hz)
                                      if not np.isnan(effective_frequency_hz) else np.nan,
            "reason": reason,
        })

    # --- Optionally include the first nav point ---

    if include_first:
        first_spacing = spacing_for_altitude(alt[0])
        add_sample(
            sample_t=t[0], sample_dist=0.0, sample_alt=alt[0],
            speed_mps=segment_speed_mps[0], spacing_m=first_spacing, reason="first",
        )
        last_sample_t    = t[0]
        last_sample_dist = 0.0
        next_target_dist = first_spacing   # First distance threshold after the initial sample.
    else:
        last_sample_t    = t[0]
        last_sample_dist = 0.0
        next_target_dist = spacing_for_altitude(alt[0])

    # --- Walk each segment, firing samples as distance or time thresholds are crossed ---

    for i in range(1, len(t)):
        t0 = t[i - 1]   # Segment start time
        t1 = t[i]       # Segment end time
        dt_i = t1 - t0  # Segment duration in seconds

        dist0 = cumulative_m[i - 1]  # Cumulative distance at segment start
        dist1 = cumulative_m[i]      # Cumulative distance at segment end
        seg_dist = dist1 - dist0     # Distance covered in this segment

        alt0 = alt[i - 1]  # Altitude at segment start
        alt1 = alt[i]      # Altitude at segment end

        speed_i = segment_speed_mps[i - 1]  # Speed during this segment

        # Multiple samples may fire within a single segment (e.g. when the
        # vehicle moves very slowly over a long time span, and min_frequency_hz
        # fires several times before the vehicle covers target_spacing_m).
        while True:
            # --- Distance-triggered sample time ---
            t_by_distance = math.inf  # Default: no distance crossing in this segment.

            if seg_dist > 0 and next_target_dist <= dist1:
                # The distance threshold is reached somewhere inside this segment.
                # Linearly interpolate time within the segment.
                frac = (next_target_dist - dist0) / seg_dist
                if 0.0 <= frac <= 1.0:
                    t_by_distance = t0 + frac * dt_i

            # --- Frequency-floor-triggered sample time ---
            t_by_frequency = last_sample_t + max_interval_s

            # Suppress frequency-triggered sampling when vehicle is hovering.
            if pause_when_stationary and speed_i < stationary_speed_mps:
                t_by_frequency = math.inf

            # Clamp frequency-trigger to within the current segment.
            if t_by_frequency < t0:
                t_by_frequency = t0   # Already overdue; fire at segment start.
            if t_by_frequency > t1:
                t_by_frequency = math.inf  # Would fire after segment ends; defer.

            # --- Choose whichever trigger fires first ---
            next_event_t = min(t_by_distance, t_by_frequency)

            # No trigger in this segment; move to the next one.
            if not math.isfinite(next_event_t):
                break

            # Guard against numerical precision causing an event at or before
            # the last sample (1 ns tolerance).
            if next_event_t <= last_sample_t + 1e-9:
                break

            # Interpolate position and altitude at the event time.
            frac_t   = min(max((next_event_t - t0) / dt_i, 0.0), 1.0)
            event_dist = dist0 + frac_t * seg_dist
            event_alt  = alt0  + frac_t * (alt1 - alt0)

            spacing_m = spacing_for_altitude(event_alt)

            # Tag the reason so callers can analyse the schedule.
            reason = "distance" if t_by_distance <= t_by_frequency else "min_frequency"

            add_sample(
                sample_t=next_event_t, sample_dist=event_dist,
                sample_alt=event_alt, speed_mps=speed_i,
                spacing_m=spacing_m, reason=reason,
            )

            # Advance state for the next iteration of the while loop.
            last_sample_t    = next_event_t
            last_sample_dist = event_dist
            next_target_dist = last_sample_dist + spacing_m  # Next distance target.

    # --- Optionally include the last nav point ---

    if include_last:
        last_spacing = spacing_for_altitude(alt[-1])
        # Only add if the last nav point is meaningfully after the most recent sample.
        if not samples or t[-1] > samples[-1]["sample_time_s"] + 1e-9:
            add_sample(
                sample_t=t[-1], sample_dist=cumulative_m[-1], sample_alt=alt[-1],
                speed_mps=segment_speed_mps[-1], spacing_m=last_spacing, reason="last",
            )

    return pd.DataFrame(samples)
