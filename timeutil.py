"""
timeutil.py — UTC time helpers with this project's naive-datetime convention.

WHY THIS EXISTS
---------------
`datetime.utcnow()` and `datetime.utcfromtimestamp()` are deprecated and are
scheduled for removal from Python.  The replacements the deprecation warning
suggests are NOT drop-in for this codebase:

    datetime.utcnow()               -> datetime.now(timezone.utc)
    datetime.utcfromtimestamp(ts)   -> datetime.fromtimestamp(ts, timezone.utc)

Both of those return TIMEZONE-AWARE datetimes.  Every datetime in this
application is NAIVE and understood to be UTC — survey timestamps, job
intervals, video ranges and sensor tables are all one timezone by project
convention (see interval_io._parse_timestamp, which strips a trailing "Z").
Mixing aware and naive datetimes raises TypeError on comparison and subtraction,
so a naive swap to the suggested API would break interval maths throughout.

These helpers do the conversion the deprecated functions did — compute in UTC,
then drop the tzinfo — so behaviour is bit-for-bit identical while using only
non-deprecated APIs.

    utc_now()               ==  datetime.utcnow()
    utc_from_timestamp(ts)  ==  datetime.utcfromtimestamp(ts)

If the project ever moves to timezone-aware datetimes, this is the single place
to change.
"""

from __future__ import annotations

from datetime import datetime, timezone

__all__ = ["utc_now", "utc_from_timestamp"]


def utc_now() -> datetime:
    """Current UTC time as a NAIVE datetime (drop-in for datetime.utcnow())."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


def utc_from_timestamp(timestamp: float) -> datetime:
    """Unix seconds → NAIVE UTC datetime.

    Drop-in for datetime.utcfromtimestamp(). The result carries no tzinfo, which
    is what the rest of the app expects.
    """
    return datetime.fromtimestamp(float(timestamp), timezone.utc).replace(tzinfo=None)
