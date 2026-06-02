# sensor_service.py — CSV loading, timestamp normalisation, and interpolation
#
# This module handles all sensor and navigation CSV data.  Its main jobs are:
#
#   1. Reading CSV files (and the special whitespace-delimited .ppi navigation
#      format used by some underwater vehicle systems).
#   2. Converting whatever timestamp format the instrument used into a uniform
#      representation: float64 seconds since the Unix epoch (1970-01-01 UTC).
#   3. Interpolating sensor values onto an arbitrary target time grid so the
#      pipeline can align sensor readings with video frame timestamps.
#   4. Building config objects (SensorFileConfig, TimeValueSourceConfig,
#      NavigationConfig) that describe what was found and how to reload it.
#
# Timestamp normalisation is the trickiest part.  Field instruments write times
# in many formats: raw Unix seconds (or ms/µs/ns variants), ISO strings,
# mixed-format strings, and sometimes date and time in separate columns with
# decimal-minute notation (e.g. "12:19.5" = 12 hours 19.5 minutes).

from __future__ import annotations

import re            # Used only for the decimal-minute pre-processing regex.
from pathlib import Path

import numpy as np
import pandas as pd

from models import NavigationConfig, SensorChannel, SensorFileConfig, TimeValueSourceConfig

# Regex that detects "HH:MM.frac" (decimal-minute time with exactly one colon).
# Capturing groups: group(1) = "HH:MM", group(2) = fractional part after the dot.
# Example match: "12:19.5" → group(1)="12:19", group(2)="5"
_DECIMAL_MIN_RE = re.compile(r"(\d{1,2}:\d{2})\.(\d+)")


class SensorService:
    """Static-method collection for loading and processing sensor/nav CSV data.

    All methods are @staticmethod because there is no instance state; the class
    acts as a namespace.  Callers import SensorService and call methods directly
    without instantiating it.
    """

    # Common column names that instruments use for their timestamp column.
    # Used by the import dialog to auto-select the timestamp column when
    # presenting a preview of a new file.
    TIMESTAMP_CANDIDATES = [
        "timestamp",
        "time",
        "datetime",
        "date_time",
        "unix_timestamp",
        "unix_time",
        "epoch",
    ]

    # Column names expected in raw .ppi files (whitespace-separated, no header).
    # These are the raw names before the timestamp column is assembled.
    _PPI_RAW_COLS = ["_date", "_time", "lat", "lon", "alt", "heading", "pitch", "roll", "flag"]

    # Column names after combining _date and _time into a single "timestamp" column.
    _PPI_COLS = ["timestamp", "lat", "lon", "alt", "heading", "pitch", "roll", "flag"]

    # ---------------------------------------------------------------------------
    # File I/O helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _is_ppi(path: str | Path) -> bool:
        """Return True when the file extension indicates a .ppi navigation file."""
        return Path(path).suffix.lower() == ".ppi"

    @staticmethod
    def _read_ppi(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
        """Read a whitespace-delimited .ppi file into a DataFrame.

        .ppi files have no header row; columns are in a fixed order defined by
        _PPI_RAW_COLS.  The separate _date ("1/18/26") and _time ("15:52:25")
        columns are merged into a single "timestamp" column ("1/18/26 15:52:25")
        so downstream code can treat .ppi files the same as CSV files.

        Args:
            path:  Path to the .ppi file.
            nrows: If set, read only this many rows (used for preview).
        """
        kwargs: dict = {"sep": r"\s+", "header": None, "names": SensorService._PPI_RAW_COLS}
        if nrows is not None:
            kwargs["nrows"] = nrows
        df = pd.read_csv(path, **kwargs)

        # Concatenate date and time strings with a space separator so pandas
        # can parse them as "1/18/26 15:52:25" using its mixed-format parser.
        df["timestamp"] = df["_date"].astype(str) + " " + df["_time"].astype(str)

        # Drop the raw split columns; callers only need the merged set.
        return df[SensorService._PPI_COLS]

    @staticmethod
    def _read_file(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
        """Dispatch to the appropriate file reader based on file extension.

        Args:
            path:  Path to a CSV or .ppi file.
            nrows: Limit rows read (useful for preview/column-sniffing).
        """
        if SensorService._is_ppi(path):
            return SensorService._read_ppi(path, nrows=nrows)
        kwargs: dict = {}
        if nrows is not None:
            kwargs["nrows"] = nrows
        # Standard CSV; pandas will infer delimiter, quoting, etc.
        return pd.read_csv(path, **kwargs)

    @staticmethod
    def read_preview(csv_path: str | Path, nrows: int = 20) -> pd.DataFrame:
        """Read the first nrows rows of a file for display in the import dialog."""
        return SensorService._read_file(csv_path, nrows=nrows)

    @staticmethod
    def read_columns(csv_path: str | Path) -> list[str]:
        """Return the list of column names from a file without reading any data rows."""
        df = SensorService._read_file(csv_path, nrows=0)
        return [str(column) for column in df.columns]

    # ---------------------------------------------------------------------------
    # Timestamp normalisation
    # ---------------------------------------------------------------------------

    @staticmethod
    def _fix_decimal_minutes(val: object) -> str:
        """Convert a decimal-minute time string to HH:MM:SS format.

        Some navigation instruments record time as HH:MM.fraction rather than
        HH:MM:SS (i.e. fractional minutes rather than whole seconds).  pandas
        cannot parse this natively, so we pre-process before calling to_datetime.

        Only strings with exactly one colon are modified (strings with two
        colons already have a seconds field and need no change).

        Example:
            "1/18/26 12:19.5" → "1/18/26 12:19:30"
            "15:52:25.1"      → unchanged (two colons)

        Args:
            val: A raw value from the timestamp column (may be non-string).

        Returns:
            The (possibly modified) string.
        """
        s = val if isinstance(val, str) else str(val)

        # Only transform strings that have exactly one colon (HH:MM.frac).
        # Strings with two colons already contain a seconds component.
        if s.count(":") != 1:
            return s

        m = _DECIMAL_MIN_RE.search(s)
        if not m:
            return s  # One colon but no decimal fraction — leave it alone.

        hm, frac = m.group(1), m.group(2)

        # Convert fractional minutes to whole seconds by multiplying by 60.
        # Round to the nearest second; sub-second precision is not preserved.
        secs = int(round(float(f"0.{frac}") * 60))

        # Replace the "HH:MM.frac" portion with "HH:MM:SS", preserving any
        # surrounding text (e.g. a leading date).
        return s[: m.start()] + f"{hm}:{secs:02d}" + s[m.end() :]

    @staticmethod
    def normalize_timestamps(series: pd.Series) -> pd.Series:
        """Convert a raw timestamp column to float64 Unix seconds.

        Handles all timestamp representations encountered in the field:

          • Numeric columns:
              - Unix seconds    (~1.74e9 in 2026)
              - Unix milliseconds (~1.74e12)
              - Unix microseconds (~1.74e15)
              - Unix nanoseconds  (~1.74e18)
            The median value is used to classify the scale and divide
            accordingly.

          • String/mixed columns:
              - ISO strings, common date formats (via pandas mixed parser)
              - Decimal-minute notation ("HH:MM.frac") after pre-processing

        Returns a Series of float64 values representing seconds since the Unix
        epoch (1970-01-01T00:00:00 UTC).  Rows that could not be parsed are
        NaN.
        """
        if series.empty:
            return pd.Series(dtype="float64")

        # --- Attempt numeric interpretation first ---
        # If the column is already numbers (or coercible to numbers), treat it
        # as a Unix timestamp and apply the appropriate scale factor.
        numeric = pd.to_numeric(series, errors="coerce")
        if numeric.notna().sum() >= max(3, int(0.7 * len(series.dropna()))):
            # Majority of non-null values are numeric.
            cleaned = numeric.astype("float64")
            finite = cleaned[np.isfinite(cleaned)]
            if finite.empty:
                return cleaned

            # Use the median to choose scale, avoiding sensitivity to outliers
            # (e.g. a single garbled row with an extreme value).
            median = float(np.nanmedian(finite))
            if median > 1e16:       # nanoseconds  (~1.74e18 for 2026)
                return cleaned / 1e9
            if median > 1e13:       # microseconds (~1.74e15 for 2026)
                return cleaned / 1e6
            if median > 1e10:       # milliseconds (~1.74e12 for 2026)
                return cleaned / 1e3
            return cleaned          # seconds      (~1.74e9  for 2026), no scaling needed

        # --- String/mixed timestamp parsing ---
        str_series = series.astype(str).str.strip()

        # Try pandas' flexible mixed-format parser.  dayfirst=False means
        # ambiguous dates like "01/02/26" are interpreted as Jan 2 (MM/DD),
        # matching the US convention used by most survey instruments.
        dt = pd.to_datetime(str_series, errors="coerce", format="mixed", dayfirst=False)

        # If fewer than half of non-null rows parsed successfully, try the
        # decimal-minute pre-processor and retry.
        total = int(series.notna().sum())
        if total > 0 and int(dt.notna().sum()) < max(1, total // 2):
            preprocessed = str_series.map(SensorService._fix_decimal_minutes)
            dt2 = pd.to_datetime(preprocessed, errors="coerce", format="mixed", dayfirst=False)
            if dt2.notna().sum() > dt.notna().sum():
                dt = dt2  # Pre-processed version parsed more rows; use it.

        # pandas datetime64 stores nanoseconds from the epoch.  The actual
        # dtype unit can vary (ns/us/ms/s) depending on the pandas version and
        # range of values; we inspect it to apply the correct divisor.
        unit = np.datetime_data(dt.dtype)[0]  # e.g. "ns", "us", "ms", "s"
        scale = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}.get(unit, 1e9)

        # Convert to int64 (nanoseconds from epoch in the "ns" case) and
        # divide by scale to get float64 seconds.  NaT rows become NaN via
        # the .where() mask.
        ints = dt.astype("int64")
        ints = ints.where(dt.notna(), np.nan)
        return ints / scale

    @staticmethod
    def _parse_separate_date_time(date_series: pd.Series, time_series: pd.Series) -> pd.Series:
        """Combine separate date and time columns into a Unix-seconds Series.

        Used when an instrument logs date and time in different columns, which
        is common in older navigation systems and some CTD loggers.

        Also handles:
          - Decimal-minute times (HH:MM.frac) in the time column.
          - Hours >= 24 in the time column (e.g. "25:03:00" meaning 1 hour 3
            minutes into the following day), which some instruments use when
            the date does not reset at midnight.

        Args:
            date_series: Column containing date strings (e.g. "1/18/26").
            time_series: Column containing time strings (e.g. "15:52:25").

        Returns:
            float64 Series of Unix seconds.
        """
        # Parse the date column to get midnight of each date as a Unix epoch value.
        date_dt = pd.to_datetime(
            date_series.astype(str).str.strip(),
            errors="coerce",
            format="mixed",
            dayfirst=False,
        )

        # Determine the dtype unit and convert to seconds since epoch.
        # This gives us the Unix timestamp for midnight of each date row.
        unit = np.datetime_data(date_dt.dtype)[0]
        scale = {"ns": 1e9, "us": 1e6, "ms": 1e3, "s": 1.0}.get(unit, 1e9)
        date_epoch = date_dt.astype("int64").astype("float64") / scale
        date_epoch[date_dt.isna()] = np.nan  # Preserve NaN rows from failed parses.

        def _time_to_secs(val) -> float:
            """Convert a time string (HH:MM:SS or HH:MM.frac) to seconds-of-day.

            Supports hours >= 24 for instruments that don't reset at midnight.
            Returns NaN for values that can't be parsed.
            """
            if pd.isna(val):
                return np.nan
            # Apply decimal-minute fix before splitting on ":".
            s = SensorService._fix_decimal_minutes(str(val).strip())
            parts = s.split(":")
            try:
                return (
                    int(parts[0]) * 3600.0
                    + int(parts[1]) * 60.0
                    + (float(parts[2]) if len(parts) > 2 else 0.0)
                )
            except (ValueError, IndexError):
                return np.nan

        # Convert each time string to seconds since midnight.
        time_secs = time_series.map(_time_to_secs)

        # Add midnight-epoch to seconds-of-day to get the full Unix timestamp.
        return date_epoch + time_secs

    @staticmethod
    def _get_timestamp_series(
        df: pd.DataFrame,
        timestamp_column: str,
        date_column: str | None = None,
    ) -> pd.Series:
        """Return a Unix-seconds Series from a loaded DataFrame.

        Dispatches to _parse_separate_date_time when a separate date column is
        configured; otherwise falls back to normalize_timestamps on the single
        combined timestamp column.

        Args:
            df:               The full DataFrame loaded from the file.
            timestamp_column: Name of the primary timestamp (or time) column.
            date_column:      Optional name of a separate date column.
        """
        if date_column is not None and date_column in df.columns:
            return SensorService._parse_separate_date_time(df[date_column], df[timestamp_column])
        return SensorService.normalize_timestamps(df[timestamp_column])

    # ---------------------------------------------------------------------------
    # Config builders (used by import dialogs)
    # ---------------------------------------------------------------------------

    @staticmethod
    def build_config(
        csv_path: str | Path,
        timestamp_column: str,
        channels: list[SensorChannel],
        date_column: str | None = None,
    ) -> SensorFileConfig:
        """Read a sensor CSV and return a fully-populated SensorFileConfig.

        Computes start_time and end_time from the timestamp range so the UI can
        display coverage on the timeline without re-reading the file later.

        Raises:
            ValueError: If no valid timestamps are found in the specified column.
        """
        df = SensorService._read_file(csv_path)

        # Parse timestamps and drop NaN rows before computing the range.
        timestamps = SensorService._get_timestamp_series(df, timestamp_column, date_column).dropna()
        if timestamps.empty:
            raise ValueError(f"No valid timestamps found in column '{timestamp_column}' for {csv_path}")

        # Convert Unix-second extremes back to naive datetimes for storage in
        # the config object (unit="s" tells pandas the input is seconds).
        start_dt = pd.to_datetime(timestamps.min(), unit="s").to_pydatetime()
        end_dt   = pd.to_datetime(timestamps.max(), unit="s").to_pydatetime()

        return SensorFileConfig(
            csv_path=Path(csv_path),
            timestamp_column=timestamp_column,
            date_column=date_column,
            channels=channels,
            start_time=start_dt,
            end_time=end_dt,
        )

    @staticmethod
    def build_time_value_source_config(
        csv_path: str | Path,
        timestamp_column: str,
        value_column: str,
        date_column: str | None = None,
    ) -> TimeValueSourceConfig:
        """Read a navigation CSV column and return a TimeValueSourceConfig.

        Validates that both the timestamp column and the value column have
        usable data before returning the config.

        Raises:
            ValueError: If timestamps or values are entirely non-numeric/missing.
        """
        df = SensorService._read_file(csv_path)

        timestamps = SensorService._get_timestamp_series(df, timestamp_column, date_column).dropna()
        if timestamps.empty:
            raise ValueError(f"No valid timestamps found in column '{timestamp_column}' for {csv_path}")

        # Coerce the value column to numeric; complain if nothing parsed.
        values = pd.to_numeric(df[value_column], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"No valid numeric values found in column '{value_column}' for {csv_path}")

        start_dt = pd.to_datetime(timestamps.min(), unit="s").to_pydatetime()
        end_dt   = pd.to_datetime(timestamps.max(), unit="s").to_pydatetime()

        return TimeValueSourceConfig(
            csv_path=Path(csv_path),
            timestamp_column=timestamp_column,
            value_column=value_column,
            date_column=date_column,
            start_time=start_dt,
            end_time=end_dt,
        )

    @staticmethod
    def build_navigation_config(
        latitude_source: TimeValueSourceConfig,
        longitude_source: TimeValueSourceConfig,
        altitude_source: TimeValueSourceConfig | None = None,
        depth_source: TimeValueSourceConfig | None = None,
        pitch_source: TimeValueSourceConfig | None = None,
        roll_source: TimeValueSourceConfig | None = None,
    ) -> NavigationConfig:
        """Bundle navigation TimeValueSourceConfig objects into a NavigationConfig.

        This is a thin factory method; no validation or file I/O is done here.
        """
        return NavigationConfig(
            latitude_source=latitude_source,
            longitude_source=longitude_source,
            altitude_source=altitude_source,
            depth_source=depth_source,
            pitch_source=pitch_source,
            roll_source=roll_source,
        )

    # ---------------------------------------------------------------------------
    # DataFrame loaders (used by the pipeline and timeline)
    # ---------------------------------------------------------------------------

    @staticmethod
    def load_sensor_dataframe(config: SensorFileConfig) -> pd.DataFrame:
        """Load a sensor CSV and return a cleaned, time-sorted DataFrame.

        The returned DataFrame contains:
          - The original timestamp column
          - The original date column (if configured)
          - One column per configured SensorChannel
          - A new "unix_time" column (float64 seconds since epoch)

        Rows with missing timestamps are dropped.  Duplicate timestamps are
        also dropped (keeping the first occurrence) because downstream
        interpolation requires a monotone time axis.

        Raises:
            ValueError: If any configured column is missing from the file.
        """
        df = SensorService._read_file(config.csv_path)

        # Build the list of columns we need from the file.
        cols = [config.timestamp_column] + [channel.source_column for channel in config.channels]
        if config.date_column is not None and config.date_column not in cols:
            cols = [config.date_column] + cols

        # Fail loudly if a configured column doesn't exist; a silent empty
        # column would corrupt downstream interpolation without an obvious error.
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {config.csv_path.name}: {missing}")

        result = df[cols].copy()

        # Add unix_time as a new column alongside the original timestamp column
        # so callers can use either the raw timestamps or the normalised seconds.
        result["unix_time"] = SensorService._get_timestamp_series(df, config.timestamp_column, config.date_column)

        # Drop rows with no parseable timestamp (they can't be placed on the
        # time axis) and sort so interpolation works correctly.
        result = result.dropna(subset=["unix_time"]).sort_values("unix_time")

        # Remove exact duplicate timestamps; keeping duplicates would make
        # numpy.interp behave unpredictably on non-unique x values.
        result = result.drop_duplicates(subset=["unix_time"])
        return result

    @staticmethod
    def load_time_value_dataframe(config: TimeValueSourceConfig) -> pd.DataFrame:
        """Load a navigation CSV column and return a two-column (unix_time, value) DataFrame.

        Used to load latitude, longitude, and altitude series for the pipeline
        and for the map/timeline widgets.  Returns only the two columns needed
        for interpolation; all other columns are discarded.

        Raises:
            ValueError: If the required columns are missing from the file.
        """
        df = SensorService._read_file(config.csv_path)

        cols = [config.timestamp_column, config.value_column]
        if config.date_column is not None and config.date_column not in cols:
            cols = [config.date_column] + cols

        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {config.csv_path.name}: {missing}")

        result = df[cols].copy()

        # Compute unix_time from the configured timestamp (and optional date) column.
        result["unix_time"] = SensorService._get_timestamp_series(df, config.timestamp_column, config.date_column)

        # Coerce the value column to numeric; non-numeric entries become NaN.
        result["value"] = pd.to_numeric(result[config.value_column], errors="coerce")

        # Drop rows where either the timestamp or the value is missing, then
        # sort and deduplicate on the time axis.
        result = result.dropna(subset=["unix_time", "value"]).sort_values("unix_time")
        result = result.drop_duplicates(subset=["unix_time"])

        # Return only the two columns the pipeline needs; drop the original
        # raw timestamp and value columns to keep the result clean.
        return result[["unix_time", "value"]]

    # ---------------------------------------------------------------------------
    # Interpolation
    # ---------------------------------------------------------------------------

    @staticmethod
    def build_sensor_raster_dataframe(
        nav_config: "NavigationConfig",
        sensor_configs: "list[SensorFileConfig]",
    ) -> "pd.DataFrame":
        """Build a sensor raster DataFrame using sensor-native timestamps.

        Each row corresponds to one sensor reading at its own native timestamp.
        The lat, lon, and alt columns are interpolated from the nav sources to
        geolocate each reading — sensor values themselves are NOT resampled.

        This is distinct from interp.csv generation (which resamples everything
        to video-frame timestamps).  No pipeline run is required; only a
        NavigationConfig and at least one SensorFileConfig are needed.

        Args:
            nav_config:     NavigationConfig with lat/lon (and optionally alt) sources.
            sensor_configs: List of SensorFileConfig objects; only configs that have
                            at least one channel are used.

        Returns:
            DataFrame with columns: unix_time, lat, lon, alt,
            plus one column per sensor channel (keyed by display_name).
            Sorted by unix_time with reset integer index.

        Raises:
            ValueError: If nav data cannot be loaded or no sensor data is found.
        """
        # Load nav position series (full resolution, sorted).
        lat_df = SensorService.load_time_value_dataframe(nav_config.latitude_source)
        lon_df = SensorService.load_time_value_dataframe(nav_config.longitude_source)
        lat_df = lat_df.sort_values("unix_time").reset_index(drop=True)
        lon_df = lon_df.sort_values("unix_time").reset_index(drop=True)

        alt_df: "pd.DataFrame | None" = None
        if nav_config.altitude_source is not None:
            try:
                alt_df = SensorService.load_time_value_dataframe(
                    nav_config.altitude_source
                ).sort_values("unix_time").reset_index(drop=True)
            except Exception:
                alt_df = None

        active_configs = [sc for sc in sensor_configs if sc.channels]
        if not active_configs:
            raise ValueError("No sensor channels configured.")

        frames: list = []
        for sensor_config in active_configs:
            sensor_df = SensorService.load_sensor_dataframe(sensor_config)
            sensor_times = sensor_df["unix_time"].to_numpy(dtype=float)

            # Geolocate each sensor reading by interpolating nav to sensor timestamps.
            lats = SensorService.interpolate_series(
                pd.Series(sensor_times),
                lat_df["unix_time"],
                lat_df["value"],
            )
            lons = SensorService.interpolate_series(
                pd.Series(sensor_times),
                lon_df["unix_time"],
                lon_df["value"],
            )

            row_df = pd.DataFrame({
                "unix_time": sensor_times,
                "lat":       lats,
                "lon":       lons,
            })

            if alt_df is not None:
                row_df["alt"] = SensorService.interpolate_series(
                    pd.Series(sensor_times),
                    alt_df["unix_time"],
                    alt_df["value"],
                )
            else:
                row_df["alt"] = np.nan

            for channel in sensor_config.channels:
                col_name = channel.display_name or channel.source_column
                row_df[col_name] = (
                    pd.to_numeric(sensor_df[channel.source_column], errors="coerce")
                    .to_numpy(dtype=float)
                )

            frames.append(row_df)

        if not frames:
            raise ValueError("Could not load any sensor data.")

        result = (
            pd.concat(frames, ignore_index=True)
            .sort_values("unix_time")
            .reset_index(drop=True)
        )
        # Drop rows where GPS could not be resolved.
        result = result.dropna(subset=["lat", "lon"]).reset_index(drop=True)
        return result

    @staticmethod
    def interpolate_series(
        target_unix_time: pd.Series,
        source_unix_time: pd.Series,
        source_values: pd.Series,
    ) -> np.ndarray:
        """Linearly interpolate source_values onto target_unix_time.

        Uses numpy.interp, which:
          - Returns the first/last value for target times outside the source range
            (clamped extrapolation rather than NaN or extrapolation).
          - Requires source_unix_time to be sorted in ascending order.
          - Handles NaN target times by producing NaN outputs.

        Infinite or NaN values in the source are masked out before interpolation
        so they don't corrupt the result.

        Args:
            target_unix_time: Times at which to evaluate the interpolation
                              (e.g. the unix_time column of master.csv).
            source_unix_time: Times of the source measurements.
            source_values:    Measurement values at source_unix_time.

        Returns:
            A float64 numpy array, same length as target_unix_time, with
            interpolated values.
        """
        # Convert all inputs to plain numpy float arrays to avoid pandas
        # index-alignment overhead and ensure numpy.interp sees clean arrays.
        sx = pd.to_numeric(source_unix_time, errors="coerce").to_numpy(dtype=float)
        sy = pd.to_numeric(source_values,    errors="coerce").to_numpy(dtype=float)
        tx = pd.to_numeric(target_unix_time, errors="coerce").to_numpy(dtype=float)

        # Remove rows where either the time or value is non-finite (NaN, inf).
        # Non-finite x values would break numpy.interp's sorted-array assumption.
        mask = np.isfinite(sx) & np.isfinite(sy)
        sx = sx[mask]
        sy = sy[mask]

        # Edge cases: if we have no usable source data, return a constant array.
        if len(sx) == 0:
            return np.full_like(tx, np.nan, dtype=float)
        if len(sx) == 1:
            # Only one source point; everything gets that value (flat extrapolation).
            return np.full_like(tx, sy[0], dtype=float)

        # Sort by time in case the source data wasn't already sorted.
        order = np.argsort(sx)
        sx = sx[order]
        sy = sy[order]

        # numpy.interp clamps target values outside [sx[0], sx[-1]] to the
        # boundary values (left=sy[0], right=sy[-1]), which is the desired
        # behaviour for GPS/sensor data where we don't want to extrapolate.
        return np.interp(tx, sx, sy, left=sy[0], right=sy[-1])
