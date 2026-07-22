"""
interval_io.py — Import interval boundaries from user-supplied CSV files.

The primary format is the app's own "Export Intervals (CSV)" output
(job_id, job_name, interval, start_time, end_time, duration_s, …), which makes
export → edit → import a lossless round trip.  Parsing is deliberately
lenient beyond that: any CSV whose header contains a recognisable start and
end column is accepted, with ISO-8601 or unix-seconds timestamp values.

Timestamps follow the app-wide convention: naive datetimes, all sources in
the same timezone (no offsets applied).
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
from pathlib import Path

from models import SelectedTimeRange

# Case-insensitive header candidates, in priority order.
_START_COLS = ("start_time", "start", "begin", "begin_time", "t0", "from")
_END_COLS   = ("end_time", "end", "stop", "stop_time", "t1", "to")
_SOURCE_COLS = ("source",)
_DESC_COLS   = ("threshold_desc", "description", "desc", "note", "label")


def _parse_timestamp(raw: str) -> datetime | None:
    """Parse one timestamp cell: ISO-8601 (with or without date) or unix seconds.

    Returns a naive datetime, or None when the cell can't be parsed.
    """
    text = (raw or "").strip()
    if not text:
        return None
    # Unix seconds (int or float).
    try:
        unix = float(text)
        # Guard against "20260115" style compact dates being read as epoch.
        if 1e8 <= unix <= 4e9:
            return datetime.fromtimestamp(unix, tz=timezone.utc).replace(tzinfo=None)
    except ValueError:
        pass
    # ISO-8601 and close variants.
    candidates = [text, text.replace("Z", ""), text.replace(" ", "T")]
    for cand in candidates:
        try:
            dt = datetime.fromisoformat(cand)
            return dt.replace(tzinfo=None) if dt.tzinfo else dt
        except ValueError:
            continue
    # Compact formats some tools export.
    for fmt in ("%Y%m%dT%H%M%S", "%Y%m%d_%H%M%S", "%Y-%m-%d %H:%M:%S",
                "%m/%d/%Y %H:%M:%S", "%m/%d/%y %H:%M:%S"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _find_column(header: list[str], candidates: tuple[str, ...]) -> str | None:
    lowered = {h.strip().lower(): h for h in header}
    for cand in candidates:
        if cand in lowered:
            return lowered[cand]
    return None


def parse_intervals_csv(path: str | Path) -> tuple[list[SelectedTimeRange], list[str]]:
    """Parse a CSV of interval boundaries into SelectedTimeRange objects.

    Returns (intervals, warnings).  Row-level problems (unparseable timestamp,
    start ≥ end) become warnings and the row is dropped; only a missing/
    unrecognisable header raises.

    Raises:
        ValueError: file unreadable, empty, or no start/end columns found.
    """
    path = Path(path)
    try:
        with open(path, "r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            header = reader.fieldnames or []
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"Could not read {path.name}: {exc}") from exc

    if not header:
        raise ValueError(f"{path.name}: file has no header row.")

    start_col = _find_column(header, _START_COLS)
    end_col   = _find_column(header, _END_COLS)
    if start_col is None or end_col is None:
        raise ValueError(
            f"{path.name}: no recognisable interval columns. Need a start column "
            f"({'/'.join(_START_COLS)}) and an end column ({'/'.join(_END_COLS)}). "
            f"Found: {', '.join(header)}"
        )

    source_col = _find_column(header, _SOURCE_COLS)
    desc_col   = _find_column(header, _DESC_COLS)

    intervals: list[SelectedTimeRange] = []
    warnings:  list[str] = []
    for i, row in enumerate(rows, start=2):   # start=2: header is line 1
        start_dt = _parse_timestamp(row.get(start_col, ""))
        end_dt   = _parse_timestamp(row.get(end_col, ""))
        if start_dt is None or end_dt is None:
            bad = start_col if start_dt is None else end_col
            warnings.append(
                f"line {i}: unparseable {bad} value '{row.get(bad, '')}' — row skipped"
            )
            continue
        if start_dt >= end_dt:
            warnings.append(
                f"line {i}: start ({start_dt.isoformat()}) is not before end "
                f"({end_dt.isoformat()}) — row skipped"
            )
            continue
        source = (row.get(source_col, "") or "").strip().lower() if source_col else ""
        if source not in ("manual", "threshold"):
            source = "manual"
        desc = (row.get(desc_col, "") or "").strip() if desc_col else ""
        intervals.append(SelectedTimeRange(
            start_time=start_dt,
            end_time=end_dt,
            source=source,
            threshold_desc=desc or f"imported: {path.name}",
        ))

    if not intervals and not warnings:
        warnings.append(f"{path.name}: no data rows found.")
    return intervals, warnings


def coverage_warnings(
    intervals: list[SelectedTimeRange],
    nav_start: datetime | None,
    nav_end: datetime | None,
    video_ranges: list[tuple[datetime, datetime]] | None = None,
) -> list[str]:
    """Flag imported intervals that fall outside nav or video coverage.

    Purely advisory — an interval with no video still yields nav/sensor
    products (job_interp), so nothing is rejected here.
    """
    out: list[str] = []
    if nav_start is not None and nav_end is not None:
        n_outside = sum(
            1 for iv in intervals
            if iv.end_time < nav_start or iv.start_time > nav_end
        )
        if n_outside:
            out.append(
                f"{n_outside} interval(s) fall entirely outside navigation coverage "
                f"({nav_start:%Y-%m-%d %H:%M:%S} → {nav_end:%Y-%m-%d %H:%M:%S})."
            )
    if video_ranges:
        n_no_video = sum(
            1 for iv in intervals
            if not any(vs < iv.end_time and ve > iv.start_time
                       for vs, ve in video_ranges)
        )
        if n_no_video:
            out.append(
                f"{n_no_video} interval(s) have no overlapping video — frame "
                "extraction will produce nothing for them (nav/sensor products still work)."
            )
    return out
