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

import re            # Decimal-minute pre-processing, time-only and zone detection.
import time
from pathlib import Path

import numpy as np
import pandas as pd

from models import NavigationConfig, SensorChannel, SensorFileConfig, TimeValueSourceConfig

# Regex that detects "HH:MM.frac" (decimal-minute time with exactly one colon).
# Capturing groups: group(1) = "HH:MM", group(2) = fractional part after the dot.
# Example match: "12:19.5" → group(1)="12:19", group(2)="5"
_DECIMAL_MIN_RE = re.compile(r"(\d{1,2}:\d{2})\.(\d+)")

# A clock time with NO date part ("05:32:09", "5:32:09.25", "12:19.5").  pandas
# silently stamps such values with TODAY's date, so a column of them makes the
# result depend on the day the pipeline runs — refuse it instead.
_TIME_ONLY_RE = re.compile(r"^\s*\d{1,2}:\d{2}(:\d{2})?(\.\d+)?\s*$")

# A trailing time-zone designator: "Z", "UTC", "GMT", "+00:00", "-0600".
_ZONE_RE = re.compile(r"(Z|UTC|GMT|[+-]\d{2}:?\d{2})\s*$", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Ingest guards (review 10 P0-3 / P1-2).  Every threshold is a module constant
# so a site with genuinely different ranges can adjust it in one place.
# ---------------------------------------------------------------------------

#: Absolute plausibility window for any timestamp (unix seconds).  Anything
#: before 1995 is a unit/column mistake (e.g. CH4 values read as epoch seconds).
PLAUSIBLE_T_MIN = 788918400.0            # 1995-01-01T00:00:00Z
#: Allowed slack past "now" (clock skew between logger and this machine).
PLAUSIBLE_FUTURE_S = 2 * 86400.0
#: A dive-scale file never spans more than this around its median time; rows
#: further out are single corrupt dates ("1/1/00") that would otherwise build a
#: decades-long interpolation grid.
PLAUSIBLE_HALF_WINDOW_S = 3 * 86400.0

#: Physically possible value ranges per navigation role (inclusive).  Values
#: outside become NaN, never pass through.  Altitude above the seafloor must be
#: strictly positive (<= 0 is a dropout/sentinel).
NAV_VALID_RANGES: dict[str, tuple[float, float]] = {
    "lat":     (-90.0, 90.0),
    "lon":     (-180.0, 180.0),
    "alt":     (1e-9, 500.0),
    "depth":   (-11000.0, 11000.0),
    "heading": (-360.0, 360.0),
    "pitch":   (-180.0, 180.0),
    "roll":    (-180.0, 180.0),
}

#: Classic logger fill values.  Exact matches become NaN in every numeric
#: channel (sensor and nav).  -999 etc. are never real readings for the
#: instruments on this vehicle (CH4/CO2 µatm, O2 µM, T °C, S PSU).
SENTINEL_VALUES = (-9999.0, -999.0, 9999.0, -99999.0, 99999.0, -9999.9, -999.9)


def _naive_dt(unix_seconds):
    """Naive-UTC datetime from unix seconds (microsecond precision, no warning)."""
    from datetime import datetime, timedelta
    return datetime(1970, 1, 1) + timedelta(microseconds=round(float(unix_seconds) * 1e6))


def _file_label(path) -> str:
    """Short, human file label for error messages."""
    try:
        return str(Path(path))
    except Exception:                                               # noqa: BLE001
        return str(path)


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
    def _read_file(
        path: str | Path,
        nrows: int | None = None,
        no_header: bool = False,
    ) -> pd.DataFrame:
        """Dispatch to the appropriate file reader based on file extension.

        Args:
            path:      Path to a CSV or .ppi file.
            nrows:     Limit rows read (useful for preview/column-sniffing).
            no_header: When True, load with header=None and rename integer
                       columns to their string equivalents ("0", "1", …).
        """
        label = _file_label(path)
        try:
            if SensorService._is_ppi(path):
                return SensorService._read_ppi(path, nrows=nrows)
            kwargs: dict = {}
            if nrows is not None:
                kwargs["nrows"] = nrows
            if no_header:
                kwargs["header"] = None
            else:
                # Jason CSVs end data rows with a dangling comma.  Without
                # index_col=False pandas turns column 0 into the index and
                # shifts every column one left (a headered DPA file then reads
                # ALTITUDE from the "BOTTOM LOCK" column; a sensor file's
                # datetime column holds CH4 values).  Review 10 P0-3 / P1-1.
                kwargs["index_col"] = False
            try:
                df = pd.read_csv(path, **kwargs)
            except pd.errors.ParserError:
                if not no_header:
                    raise
                # Ragged headerless file: only SOME rows carry the dangling
                # comma (or a ",,").  Size the frame from the widest row.
                width = SensorService._max_fields(path, nrows)
                df = pd.read_csv(path, header=None, names=list(range(width)),
                                 **({"nrows": nrows} if nrows is not None else {}))
                empty_tail = [c for c in reversed(df.columns) if df[c].isna().all()]
                # keep at least as many columns as the common row width
                for c in empty_tail[: max(0, width - SensorService._common_fields(path))]:
                    df = df.drop(columns=[c])
        except pd.errors.EmptyDataError as exc:
            raise ValueError(f"{label}: file is empty (no columns to parse)") from exc
        except pd.errors.ParserError as exc:
            raise ValueError(f"{label}: could not parse CSV — {exc}") from exc
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"{label}: file not found (moved or drive not mounted? re-import it)") from exc
        if no_header:
            df.columns = [str(c) for c in df.columns]
        return df

    @staticmethod
    def _field_counts(path, nrows=None, limit=200000) -> list[int]:
        counts: list[int] = []
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            for i, line in enumerate(handle):
                if (nrows is not None and i >= nrows) or i >= limit:
                    break
                if line.strip():
                    counts.append(line.rstrip("\r\n").count(",") + 1)
        return counts

    @staticmethod
    def _max_fields(path, nrows=None) -> int:
        counts = SensorService._field_counts(path, nrows)
        return max(counts) if counts else 1

    @staticmethod
    def _common_fields(path) -> int:
        counts = SensorService._field_counts(path, None, 5000)
        if not counts:
            return 1
        values, freq = np.unique(counts, return_counts=True)
        return int(values[int(np.argmax(freq))])

    @staticmethod
    def read_preview(
        csv_path: str | Path,
        nrows: int = 20,
        no_header: bool = False,
    ) -> pd.DataFrame:
        """Read the first nrows rows of a file for display in the import dialog."""
        return SensorService._read_file(csv_path, nrows=nrows, no_header=no_header)

    @staticmethod
    def read_columns(
        csv_path: str | Path,
        no_header: bool = False,
    ) -> list[str]:
        """Return the list of column names from a file without reading any data rows."""
        if no_header:
            # header=None with nrows=0 returns an empty frame; read 1 row instead.
            df = SensorService._read_file(csv_path, nrows=1, no_header=True)
        else:
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
        non_null = str_series[series.notna() & (str_series != "") &
                              (str_series.str.lower() != "nan")]
        column = f"'{series.name}'" if series.name is not None else "timestamp column"

        # A clock time with no date ("05:32:09") parses to TODAY's date: the
        # result would change with the day the pipeline runs (review 10 T).
        if len(non_null):
            sample = non_null.iloc[:: max(1, len(non_null) // 2000)]
            if sample.map(lambda s: bool(_TIME_ONLY_RE.match(s))).mean() >= 0.5:
                raise ValueError(
                    f"timestamp column {column} holds clock times with no date "
                    f"(e.g. {sample.iloc[0]!r}); choose the column that carries the "
                    "full date+time, or configure a separate date column")

        # Time-zone-aware strings ("…Z", "… UTC", "…-06:00") are converted to
        # naive UTC here.  This is NOT a timezone guess: the value states its
        # own zone.  Naive values are never shifted (all project data is one
        # timezone).  A column mixing zoned and naive values is ambiguous and
        # refused.
        zoned = non_null.map(lambda s: bool(_ZONE_RE.search(s))) if len(non_null) else non_null
        n_zoned = int(zoned.sum()) if len(non_null) else 0
        if n_zoned and n_zoned < len(non_null):
            raise ValueError(
                f"timestamp column {column} mixes time-zone-qualified values "
                f"({n_zoned}) with naive ones ({len(non_null) - n_zoned}); "
                "the naive rows are ambiguous — make the column consistent")
        utc = bool(n_zoned)

        def _parse(values: pd.Series) -> pd.Series:
            try:
                out = pd.to_datetime(values, errors="coerce", format="mixed",
                                     dayfirst=False, utc=utc)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"timestamp column {column}: {exc}") from exc
            if isinstance(out.dtype, pd.DatetimeTZDtype):
                out = out.dt.tz_convert("UTC").dt.tz_localize(None)
            elif out.dtype == object:        # pandas left mixed zones unconverted
                out = pd.to_datetime(values, errors="coerce", format="mixed",
                                     dayfirst=False, utc=True)
                out = out.dt.tz_convert("UTC").dt.tz_localize(None)
            return out

        # Try pandas' flexible mixed-format parser.  dayfirst=False means
        # ambiguous dates like "01/02/26" are interpreted as Jan 2 (MM/DD),
        # matching the US convention used by most survey instruments.
        dt = _parse(str_series)

        # If fewer than half of non-null rows parsed successfully, try the
        # decimal-minute pre-processor and retry.
        total = int(series.notna().sum())
        if total > 0 and int(dt.notna().sum()) < max(1, total // 2):
            preprocessed = str_series.map(SensorService._fix_decimal_minutes)
            dt2 = _parse(preprocessed)
            if dt2.notna().sum() > dt.notna().sum():
                dt = dt2  # Pre-processed version parsed more rows; use it.

        # Unit-agnostic conversion to float seconds (NaT -> NaN).  Works for any
        # datetime64 resolution (ns/us/ms/s) without inspecting the dtype.
        secs = (dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1)
        return pd.Series(np.asarray(secs, dtype="float64"), index=series.index)

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

        if isinstance(date_dt.dtype, pd.DatetimeTZDtype):
            date_dt = date_dt.dt.tz_convert("UTC").dt.tz_localize(None)
        # Unix timestamp for midnight of each date row (unit-agnostic; NaT ->
        # NaN so failed parses stay missing).
        date_epoch = pd.Series(
            np.asarray((date_dt - pd.Timestamp("1970-01-01")) / pd.Timedelta(seconds=1),
                       dtype="float64"), index=date_series.index)

        def _time_to_secs(val) -> float:
            """Convert a time string (HH:MM:SS or HH:MM.frac) to seconds-of-day.

            Supports hours >= 24 for instruments that don't reset at midnight.
            Returns NaN for values that can't be parsed or have out-of-range
            minutes/seconds (which indicate malformed data, not intentional overflow).
            """
            if pd.isna(val):
                return np.nan
            # Apply decimal-minute fix before splitting on ":".
            s = SensorService._fix_decimal_minutes(str(val).strip())
            parts = s.split(":")
            try:
                h   = int(parts[0])
                m   = int(parts[1])
                sec = float(parts[2]) if len(parts) > 2 else 0.0
                if not (0 <= m < 60 and 0.0 <= sec < 60.0):
                    return np.nan  # malformed — minutes/seconds out of range
                return h * 3600.0 + m * 60.0 + sec
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

    @staticmethod
    def _checked_timestamps(
        df: pd.DataFrame,
        timestamp_column: str,
        date_column: str | None,
        csv_path,
    ) -> tuple[pd.Series, list[str]]:
        """Unix-seconds Series with implausible values set to NaN, plus notes.

        Every error names the file.  Implausible = before 1995, more than two
        days in the future, or more than PLAUSIBLE_HALF_WINDOW_S from the
        file's median time (one corrupt "1/1/00" row).  Review 10 P0-3.
        """
        label = _file_label(csv_path)
        try:
            unix = SensorService._get_timestamp_series(df, timestamp_column, date_column)
        except ValueError as exc:
            raise ValueError(f"{label}: {exc}") from exc
        unix = pd.to_numeric(unix, errors="coerce").astype("float64")
        notes: list[str] = []
        n_raw = int(len(unix))
        n_unparsed = int(unix.isna().sum())
        if n_unparsed:
            notes.append(f"{n_unparsed:,} of {n_raw:,} rows have an unparseable timestamp "
                         f"in '{timestamp_column}' — dropped")
        now = time.time()
        absolute_ok = (unix >= PLAUSIBLE_T_MIN) & (unix <= now + PLAUSIBLE_FUTURE_S)
        finite = unix.notna()
        if finite.any() and not (absolute_ok & finite).any():
            lo = pd.to_datetime(float(unix[finite].min()), unit="s")
            hi = pd.to_datetime(float(unix[finite].max()), unit="s")
            raise ValueError(
                f"{label}: no plausible timestamps in column '{timestamp_column}' "
                f"(values parse to {lo} .. {hi}); wrong timestamp column, or the "
                "columns are shifted (dangling comma / wrong header setting)")
        bad_abs = finite & ~absolute_ok
        if bad_abs.any():
            notes.append(f"{int(bad_abs.sum()):,} rows with implausible timestamps "
                         "(before 1995 or in the future) dropped")
        unix = unix.where(absolute_ok)
        if unix.notna().any():
            median = float(unix.median())
            far = unix.notna() & ((unix - median).abs() > PLAUSIBLE_HALF_WINDOW_S)
            if far.any():
                examples = ", ".join(str(pd.to_datetime(float(v), unit="s"))
                                     for v in unix[far].iloc[:3])
                notes.append(f"{int(far.sum()):,} rows more than "
                             f"{PLAUSIBLE_HALF_WINDOW_S / 86400:g} days from the file's "
                             f"median time dropped (e.g. {examples})")
                unix = unix.where(~far)
        if not unix.notna().any():
            raise ValueError(f"{label}: no valid timestamps in column '{timestamp_column}'")
        return unix, notes

    @staticmethod
    def _sort_dedup(result: pd.DataFrame, notes: list[str]) -> pd.DataFrame:
        """Stable time sort + first-in-file-order dedup, with counts noted."""
        times = result["unix_time"].to_numpy(dtype=float)
        backwards = int((np.diff(times) < 0).sum()) if len(times) > 1 else 0
        if backwards:
            notes.append(f"{backwards:,} out-of-order timestamp(s) — rows re-sorted")
        # Stable sort: among equal timestamps the FIRST row in file order
        # survives the dedup below (quicksort made that arbitrary; review 10 P1-4).
        result = result.sort_values("unix_time", kind="stable")
        dups = int(result["unix_time"].duplicated().sum())
        if dups:
            notes.append(f"{dups:,} duplicate timestamp(s) — first occurrence in the file kept")
        return result.drop_duplicates(subset=["unix_time"], keep="first")

    @staticmethod
    def _clean_values(values: pd.Series, kind: str | None, name: str,
                      notes: list[str]) -> pd.Series:
        """Numeric coercion + sentinel / physical-range filtering -> NaN."""
        values = pd.to_numeric(values, errors="coerce").astype("float64")
        sentinel = values.isin(SENTINEL_VALUES)
        if sentinel.any():
            notes.append(f"{int(sentinel.sum()):,} sentinel value(s) "
                         f"({', '.join(f'{v:g}' for v in sorted(set(values[sentinel])))}) "
                         f"in '{name}' set to NaN")
            values = values.where(~sentinel)
        bounds = NAV_VALID_RANGES.get(kind or "")
        if bounds is not None:
            lo, hi = bounds
            out = values.notna() & ((values < lo) | (values > hi))
            if out.any():
                notes.append(f"{int(out.sum()):,} {kind} value(s) outside the physical "
                             f"range [{lo:g}, {hi:g}] in '{name}' set to NaN "
                             f"(e.g. {values[out].iloc[0]:g})")
                values = values.where(~out)
        return values

    @staticmethod
    def time_bounds(csv_path, timestamp_column: str, date_column: str | None = None,
                    no_header: bool = False):
        """(start, end) naive-UTC datetimes of a file's PLAUSIBLE timestamps.

        Parsed over the whole column (not first/last line), with the same
        guards the pipeline applies, so an import dialog can show — and compare
        against nav — exactly the span the build will use.  Raises ValueError
        naming the file when nothing usable is found.
        """
        df = SensorService._read_file(csv_path, no_header=no_header)
        if timestamp_column not in df.columns:
            raise ValueError(f"{_file_label(csv_path)}: no column '{timestamp_column}'")
        unix, _notes = SensorService._checked_timestamps(df, timestamp_column,
                                                         date_column, csv_path)
        unix = unix.dropna()
        return (_naive_dt(unix.min()),
                _naive_dt(unix.max()))

    @staticmethod
    def span_overlap(a_start: float, a_end: float, b_start: float, b_end: float) -> float:
        """Fraction of span A covered by span B (0..1); unix seconds."""
        length = float(a_end) - float(a_start)
        if length <= 0:
            return 1.0 if b_start <= a_start <= b_end else 0.0
        inter = min(float(a_end), float(b_end)) - max(float(a_start), float(b_start))
        return max(0.0, inter) / length

    # ---------------------------------------------------------------------------
    # Config builders (used by import dialogs)
    # ---------------------------------------------------------------------------

    @staticmethod
    def build_config(
        csv_path: str | Path,
        timestamp_column: str,
        channels: list[SensorChannel],
        date_column: str | None = None,
        no_header: bool = False,
    ) -> SensorFileConfig:
        """Read a sensor CSV and return a fully-populated SensorFileConfig.

        Computes start_time and end_time from the timestamp range so the UI can
        display coverage on the timeline without re-reading the file later.

        Raises:
            ValueError: If no valid timestamps are found in the specified column.
        """
        df = SensorService._read_file(csv_path, no_header=no_header)
        if timestamp_column not in df.columns:
            raise ValueError(f"{_file_label(csv_path)}: no column '{timestamp_column}'")
        missing = [c.source_column for c in channels if c.source_column not in df.columns]
        if missing:
            raise ValueError(f"{_file_label(csv_path)}: missing channel column(s) {missing}")

        # Parse timestamps and drop NaN / implausible rows before computing the range.
        timestamps, _notes = SensorService._checked_timestamps(
            df, timestamp_column, date_column, csv_path)
        timestamps = timestamps.dropna()
        if timestamps.empty:
            raise ValueError(f"No valid timestamps found in column '{timestamp_column}' for {csv_path}")

        # Convert Unix-second extremes back to naive datetimes for storage in
        # the config object (unit="s" tells pandas the input is seconds).
        start_dt = _naive_dt(timestamps.min())
        end_dt   = _naive_dt(timestamps.max())

        return SensorFileConfig(
            csv_path=Path(csv_path),
            timestamp_column=timestamp_column,
            date_column=date_column,
            channels=channels,
            start_time=start_dt,
            end_time=end_dt,
            no_header=no_header,
        )

    @staticmethod
    def build_time_value_source_config(
        csv_path: str | Path,
        timestamp_column: str,
        value_column: str,
        date_column: str | None = None,
        no_header: bool = False,
    ) -> TimeValueSourceConfig:
        """Read a navigation CSV column and return a TimeValueSourceConfig.

        Validates that both the timestamp column and the value column have
        usable data before returning the config.

        Raises:
            ValueError: If timestamps or values are entirely non-numeric/missing.
        """
        df = SensorService._read_file(csv_path, no_header=no_header)
        for col in (timestamp_column, value_column):
            if col not in df.columns:
                raise ValueError(f"{_file_label(csv_path)}: no column '{col}'")

        timestamps, _notes = SensorService._checked_timestamps(
            df, timestamp_column, date_column, csv_path)
        timestamps = timestamps.dropna()
        if timestamps.empty:
            raise ValueError(f"No valid timestamps found in column '{timestamp_column}' for {csv_path}")

        # Coerce the value column to numeric; complain if nothing parsed.
        values = pd.to_numeric(df[value_column], errors="coerce")
        if values.notna().sum() == 0:
            raise ValueError(f"No valid numeric values found in column '{value_column}' for {csv_path}")

        start_dt = _naive_dt(timestamps.min())
        end_dt   = _naive_dt(timestamps.max())

        return TimeValueSourceConfig(
            csv_path=Path(csv_path),
            timestamp_column=timestamp_column,
            value_column=value_column,
            date_column=date_column,
            start_time=start_dt,
            end_time=end_dt,
            no_header=no_header,
        )

    @staticmethod
    def build_navigation_config(
        latitude_source: TimeValueSourceConfig,
        longitude_source: TimeValueSourceConfig,
        altitude_source: TimeValueSourceConfig | None = None,
        depth_source: TimeValueSourceConfig | None = None,
        heading_source: TimeValueSourceConfig | None = None,
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
            heading_source=heading_source,
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
        df = SensorService._read_file(config.csv_path, no_header=getattr(config, "no_header", False))

        # Build the list of columns we need from the file.
        # Two channels may legitimately point at the SAME source column (e.g. the
        # same series under two display names, or the same channel imported
        # twice).  df[cols] with a repeated name yields duplicate columns, and
        # every later ``sensor_df[source_column]`` then returns a DataFrame
        # instead of a Series — which blows up in pd.to_numeric.  Select each
        # source column exactly once, in first-seen order.
        cols = list(dict.fromkeys(
            [config.timestamp_column]
            + [channel.source_column for channel in config.channels]))
        if config.date_column is not None and config.date_column not in cols:
            cols = [config.date_column] + cols

        # Fail loudly if a configured column doesn't exist; a silent empty
        # column would corrupt downstream interpolation without an obvious error.
        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {config.csv_path.name}: {missing} "
                             f"({_file_label(config.csv_path)})")

        result = df[cols].copy()

        # Add unix_time as a new column alongside the original timestamp column
        # so callers can use either the raw timestamps or the normalised seconds.
        result["unix_time"], notes = SensorService._checked_timestamps(
            df, config.timestamp_column, config.date_column, config.csv_path)

        # Sentinel fill values (-999, -9999, …) become NaN in every channel.
        for channel in config.channels:
            col = channel.source_column
            if col in result.columns and col != config.timestamp_column:
                result[col] = SensorService._clean_values(result[col], None, col, notes)
                if not result[col].notna().any():
                    notes.append(f"channel '{col}' has no numeric values at all")

        # Drop rows with no parseable timestamp (they can't be placed on the
        # time axis), then a STABLE sort + first-in-file dedup so interpolation
        # sees a strictly increasing axis.
        result = result.dropna(subset=["unix_time"])
        result = SensorService._sort_dedup(result, notes)
        result.attrs["ingest_notes"] = [f"{_file_label(config.csv_path)}: {n}" for n in notes]
        return result

    @staticmethod
    def load_time_value_dataframe(
        config: TimeValueSourceConfig,
        negate: bool = False,
        kind: str | None = None,
    ) -> pd.DataFrame:
        """Load a navigation CSV column and return a two-column (unix_time, value) DataFrame.

        Used to load latitude, longitude, and altitude series for the pipeline
        and for the map/timeline widgets.  Returns only the two columns needed
        for interpolation; all other columns are discarded.

        Args:
            config: The source configuration describing file, timestamp, and value columns.
            negate: When True, multiply all values by -1 after loading.  Used to
                    convert positive depth readings to negative Z coordinates.
            kind:   Navigation role ("lat", "lon", "alt", "depth", "heading",
                    "pitch", "roll").  When given, physically impossible values
                    (NAV_VALID_RANGES) become NaN and are dropped.  Sentinel fill
                    values are dropped for every kind.

        The returned frame carries ``attrs["ingest_notes"]``: one line per
        dropped/cleaned category (counts), each prefixed with the file path.

        Raises:
            ValueError: If the required columns are missing from the file, or no
                        plausible timestamp exists (the message names the file).
        """
        df = SensorService._read_file(config.csv_path, no_header=getattr(config, "no_header", False))

        cols = [config.timestamp_column, config.value_column]
        if config.date_column is not None and config.date_column not in cols:
            cols = [config.date_column] + cols

        missing = [col for col in cols if col not in df.columns]
        if missing:
            raise ValueError(f"Missing columns in {config.csv_path.name}: {missing} "
                             f"({_file_label(config.csv_path)})")

        result = df[cols].copy()

        # Compute unix_time from the configured timestamp (and optional date) column.
        result["unix_time"], notes = SensorService._checked_timestamps(
            df, config.timestamp_column, config.date_column, config.csv_path)

        # Coerce the value column to numeric; sentinel and physically
        # impossible values become NaN (then dropped below).
        result["value"] = SensorService._clean_values(
            result[config.value_column], kind, str(config.value_column), notes)

        # Drop rows where either the timestamp or the value is missing, then
        # stable-sort and deduplicate on the time axis.
        result = result.dropna(subset=["unix_time", "value"])
        if result.empty:
            raise ValueError(f"{_file_label(config.csv_path)}: no usable rows in "
                             f"column '{config.value_column}' after cleaning")
        result = SensorService._sort_dedup(result, notes)

        # Apply optional sign inversion (e.g. depth positive → negative).
        if negate:
            result = result.copy()
            result["value"] = -result["value"]

        # Return only the two columns the pipeline needs; drop the original
        # raw timestamp and value columns to keep the result clean.
        out = result[["unix_time", "value"]].copy()
        out.attrs["ingest_notes"] = [f"{_file_label(config.csv_path)} "
                                     f"[{kind or config.value_column}]: {n}" for n in notes]
        return out

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
        *,
        edge_tolerance_s: float = 2.0,
        max_gap_s: float | None = None,
        clamp: bool = False,
    ) -> np.ndarray:
        """Linearly interpolate source_values onto target_unix_time.

        Target times OUTSIDE the source's own time coverage get NaN — never the
        first/last value held constant (the old edge-hold fabricated hours of
        frozen positions and gas readings; review 02 P0-1 / 06 P1-2 / 10 P0-2).
        ``edge_tolerance_s`` lets a target a hair outside the range (sub-second
        clock rounding) take the edge value.  ``max_gap_s`` additionally blanks
        targets that fall inside a source gap longer than that (a straight line
        across a 20-minute dropout is not data; review 10 P1-3).  ``clamp=True``
        restores the legacy edge-hold for display-only callers that ask for it.

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
            # One source point: it only speaks for times right beside it.
            if clamp:
                return np.full_like(tx, sy[0], dtype=float)
            out = np.full_like(tx, np.nan, dtype=float)
            out[np.abs(tx - sx[0]) <= edge_tolerance_s] = sy[0]
            return out

        # Stable sort by time in case the source data wasn't already sorted.
        order = np.argsort(sx, kind="stable")
        sx = sx[order]
        sy = sy[order]

        out = np.interp(tx, sx, sy, left=sy[0], right=sy[-1])
        if clamp:
            return out
        tol = max(0.0, float(edge_tolerance_s))
        outside = (tx < sx[0] - tol) | (tx > sx[-1] + tol)
        out[outside] = np.nan
        if max_gap_s is not None and max_gap_s > 0:
            gaps = np.diff(sx)
            if (gaps > max_gap_s).any():
                idx = np.searchsorted(sx, tx, side="right")      # sx[idx-1] <= t < sx[idx]
                inside = (idx > 0) & (idx < len(sx))
                span = np.full(tx.shape, 0.0)
                span[inside] = sx[idx[inside]] - sx[idx[inside] - 1]
                # a target sitting exactly on a sample is data, not a gap
                on_sample = np.zeros(tx.shape, dtype=bool)
                # (absolute tolerance: np.isclose's relative rtol is ~5 h at 1.7e9 s)
                on_sample[inside] = np.abs(tx[inside] - sx[idx[inside] - 1]) <= 1e-6
                out[inside & (span > max_gap_s) & ~on_sample] = np.nan
        return out
