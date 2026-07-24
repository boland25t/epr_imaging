"""timeutil must be a bit-for-bit drop-in for the deprecated datetime APIs.

These tests pin the exact property that makes the migration safe: the results
are NAIVE and numerically identical to datetime.utcfromtimestamp / utcnow.
"""

from __future__ import annotations

import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from timeutil import utc_from_timestamp, utc_now


@pytest.mark.parametrize("ts", [
    0.0,                 # epoch
    1_768_514_900.0,     # a real survey timestamp from this project
    1_768_514_900.5,     # sub-second
    2_000_000_000.0,     # future
    -86_400.0,           # before epoch
])
def test_matches_deprecated_utcfromtimestamp(ts):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = datetime.utcfromtimestamp(ts)
    assert utc_from_timestamp(ts) == expected


def test_result_is_naive():
    """Aware results would raise TypeError against the app's naive datetimes."""
    assert utc_from_timestamp(1_768_514_900.0).tzinfo is None
    assert utc_now().tzinfo is None


def test_utc_now_matches_deprecated_within_a_second():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        expected = datetime.utcnow()
    assert abs((utc_now() - expected).total_seconds()) < 1.0


def test_accepts_int_and_str_numeric():
    assert utc_from_timestamp(1_768_514_900) == utc_from_timestamp(1_768_514_900.0)
    assert utc_from_timestamp("1768514900") == utc_from_timestamp(1_768_514_900.0)


def test_naive_arithmetic_still_works():
    """The whole point: results must interoperate with the app's naive datetimes."""
    a = utc_from_timestamp(1_768_514_900.0)
    b = datetime(2026, 1, 15, 22, 8, 20)          # naive, as produced by interval_io
    assert (a - b).total_seconds() == 0.0          # no TypeError
    assert a == b


def test_uses_no_deprecated_api():
    """Calling the helpers must not emit a DeprecationWarning."""
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        utc_now()
        utc_from_timestamp(1_768_514_900.0)


def test_roundtrip_against_utc_aware():
    ts = 1_768_514_900.0
    aware = datetime.fromtimestamp(ts, timezone.utc)
    assert utc_from_timestamp(ts) == aware.replace(tzinfo=None)
