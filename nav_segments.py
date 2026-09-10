"""Nav-velocity segmentation — split a dive into moving (traverse) segments.

Qt-free service shared by the headless batch and the GUI.  A "segment" is a
time interval over which the vehicle sustains motion above a speed threshold;
these are the natural photogrammetry scopes (traverse legs) as opposed to
station-keeping / hovering time, which reconstructs poorly.  The app surfaces
each segment as a highlighted interval on the trackline selector; the same
function drives headless chunking, so both are reproducible from interp.

The algorithm, deliberately simple:
  1. horizontal speed from (easting, northing) between consecutive samples,
     smoothed with a rolling median (default 15 s) to reject nav jitter;
  2. samples with smoothed speed >= v_min are "moving";
  3. consecutive moving samples merge into a segment, tolerating stationary
     gaps up to gap_s seconds (so a brief pause doesn't split a traverse);
  4. segments whose horizontal path is shorter than min_path_m are dropped
     (too little baseline to reconstruct).
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass, asdict
from datetime import datetime, timezone


@dataclass
class Segment:
    t_start: float          # unix seconds
    t_end: float
    path_m: float           # horizontal path length over the segment
    dur_s: float
    n_samples: int

    @property
    def iso_start(self) -> str:
        return datetime.fromtimestamp(self.t_start, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    @property
    def iso_end(self) -> str:
        return datetime.fromtimestamp(self.t_end, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")

    def as_dict(self) -> dict:
        d = asdict(self)
        d["iso_start"], d["iso_end"] = self.iso_start, self.iso_end
        return d


def _rolling_median(x, w):
    if w <= 1:
        return list(x)
    n = len(x)
    half = w // 2
    out = [0.0] * n
    for i in range(n):
        a = max(0, i - half); b = min(n, i + half + 1)
        s = sorted(x[a:b]); m = len(s)
        out[i] = s[m // 2] if m % 2 else 0.5 * (s[m // 2 - 1] + s[m // 2])
    return out


def _load(interp_csv):
    t = []; e = []; n = []
    with open(interp_csv, newline="") as f:
        rd = csv.DictReader(f)
        cols = {c.lower(): c for c in (rd.fieldnames or [])}
        tc = cols.get("unix_time"); ec = cols.get("easting"); nc = cols.get("northing")
        if not (tc and ec and nc):
            raise ValueError("interp csv needs unix_time, easting, northing columns")
        for r in rd:
            try:
                t.append(float(r[tc])); e.append(float(r[ec])); n.append(float(r[nc]))
            except (ValueError, TypeError, KeyError):
                continue
    # sort by time
    order = sorted(range(len(t)), key=lambda i: t[i])
    return [t[i] for i in order], [e[i] for i in order], [n[i] for i in order]


def detect_moving_segments(interp_csv, v_min=0.08, gap_s=90.0,
                           min_path_m=8.0, smooth_s=15):
    """Return a list of Segment covering the moving (traverse) parts of a dive.

    v_min      : m/s sustained-speed threshold separating moving from hovering.
    gap_s      : a stationary gap up to this many seconds does not split a segment.
    min_path_m : discard segments with less horizontal path than this.
    smooth_s   : rolling-median window (samples ~1 Hz, so ~seconds).
    """
    t, e, n = _load(interp_csv)
    N = len(t)
    if N < 3:
        return []
    spd = [0.0] * N
    for i in range(1, N):
        dt = t[i] - t[i - 1]
        spd[i] = math.hypot(e[i] - e[i - 1], n[i] - n[i - 1]) / dt if dt > 0 else 0.0
    sm = _rolling_median(spd, max(1, int(smooth_s)))
    moving = [s >= v_min for s in sm]

    segs = []
    i = 0
    while i < N:
        if not moving[i]:
            i += 1
            continue
        j = i          # last confirmed moving index
        gap = 0.0
        k = i + 1
        while k < N:
            if moving[k]:
                j = k; gap = 0.0
            else:
                gap += t[k] - t[k - 1]
                if gap > gap_s:
                    break
            k += 1
        path = sum(math.hypot(e[a] - e[a - 1], n[a] - n[a - 1]) for a in range(i + 1, j + 1))
        if path >= min_path_m:
            segs.append(Segment(t_start=t[i], t_end=t[j], path_m=path,
                                 dur_s=t[j] - t[i], n_samples=j - i + 1))
        i = k + 1 if k > j else j + 1
    return segs


if __name__ == "__main__":
    import sys, json
    csvp = sys.argv[1] if len(sys.argv) > 1 else "inputs/interp_full.csv"
    vmin = float(sys.argv[2]) if len(sys.argv) > 2 else 0.08
    segs = detect_moving_segments(csvp, v_min=vmin)
    tot = sum(s.path_m for s in segs)
    print(f"{len(segs)} segments  moving_path={tot:.0f} m  moving_time={sum(s.dur_s for s in segs)/3600:.1f} h")
    for i, s in enumerate(segs, 1):
        print(f"  seg{i:02d} {s.iso_start}->{s.iso_end}  {s.path_m:6.1f} m  {s.dur_s/60:5.1f} min")
    if "--json" in sys.argv:
        print(json.dumps([s.as_dict() for s in segs], indent=2))
