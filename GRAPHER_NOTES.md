# Grapher.m baseline work — handoff summary

File: `Grapher.m`
Data: `interp_full.csv` (nav + CO2/CH4/O2/Temperature/alt, ~1 sample/sec, ~40hr record)

MATLAB here is **base only** — no Signal Processing, Statistics, or Econometrics
Toolbox. No `lowpass`, no `corr`, no `movquant`. `movmedian`, `movmad` (confirmed
median-based via probe), `smoothdata`, `fillmissing`, `gradient` all exist.

## Current state of Grapher.m

4 figures, all working, linked x-axes:
1. Raw masked channels (speed, valid mask, alt/CO2/CH4/temp/O2)
2. Same, smoothed for display only (30s gaussian, per-segment so NaN gaps aren't bridged)
3. **Baseline A**: drift-tracking median filter + local (movmad-based) sigma,
   one-sided iterative clipping, hysteresis anomaly detection (enterK=5, exitK=2,
   minDurSec=20). `baselineWindowSec = 4000` (raised from 1000 after a window
   sweep showed 1000s absorbed ~32-44% of CO2/CH4 anomaly mass into the baseline;
   4000s is where the anomaly-mass integral plateaus).
4. **Baseline B**: derivative-based excision (smooth → threshold the slope on a
   robust z → walk forward to a level-based settle point → cut segment out →
   re-baseline on the hole-filled signal). Has an amplitude gate (`B_minAmpK=4`)
   added after discovering the slope-only test was shredding quiet O2 data
   (22.8% excised while only 0.1% was ever flagged as anomalous).

Both approaches converge closely on CO2 (14 events / 7.3% both) and temp
(86 vs 85 events) — good cross-validation. O2 disagrees (4 vs 1 event), flagged
as the one place to eyeball.

Baseline computed per-segment (`baselineWithinSegments`), where "segment" =
contiguous run of the existing speed-based validity mask (see `validMask`,
`segmentStarts`/`segmentEnds` earlier in the script).

## THE UNRESOLVED BUG (user caught this, was mid-diagnosis when thread got compressed)

`baselineWithinSegments` → `clampWindow` **silently shrinks the baseline window
to fit inside the segment** when the segment is shorter than `baselineWindowSec`.
Diagnostic run confirmed:

- 20 total segments; **14 are "DEGRADED"** (shorter than 4000s, so they get a
  materially shrunk window — some down to 126-1410 samples/seconds).
- Of those, **segments 8 and 9 are actively contaminated**: their head/mid CO2
  ratio is 2.37x and 12.9x respectively — i.e. they *start* inside the tail of
  a masked-out plume, so the segment's own median is dragged upward by the very
  anomaly the baseline is supposed to reject. Their own local baseline is biased
  high because there isn't enough clean record within the segment to outvote
  the contaminated edge.
- This directly confirms the user's objection: "some segments obviously contain
  the tails of large anomalies that were masked out" — short segments can't
  self-characterize a baseline, especially when they abut the mask boundary.

**Not yet fixed.** Candidate approaches (undiscussed with user, pick one together):
1. Require a minimum segment duration before trusting a local baseline; below
   it, borrow/extrapolate baseline from the nearest adequately-long segment.
2. Extend the invalidBuffer/exclusion so segments don't start/end immediately
   adjacent to a masked plume (currently `invalidBuffer = 30`s — may be too
   short relative to plume tail decay).
3. Pool baseline estimation across nearby segments (weighted by distance in
   time) rather than treating each segment as fully independent.
4. Flag degraded segments explicitly in the plot (e.g. grey hatch) rather than
   silently trusting a clamped-window baseline.

## Also requested, not yet done

1. **Linear baseline fit per segment** — user wants a straight-line trend fit
   (not just the nonparametric median baseline) to characterize drift. First
   attempt used `corr()` which isn't available (no Statistics Toolbox) — use
   manual Pearson (`sum(bc.*sc)/sqrt(sum(bc.^2)*sum(sc.^2))` pattern, already
   used successfully in a later diagnostic) or `polyfit`/`polyval` (both base
   MATLAB, confirmed available). **Must fit per-segment, using only quiet
   (non-anomalous) samples within that segment** — a single record-wide line
   is not meaningful given segment gaps. Should probably wait until the
   degraded-segment bug above is addressed, since a linear fit on a
   contaminated segment will be just as biased.

2. **Second set of log-scaled figures** — user asked for a log-scale plot pair
   (presumably CO2/CH4/O2 which span decades; alt/temp less clearly need it).
   Not started. Watch for non-positive values before log-transforming (a
   partial dynamic-range check was run but not completed/applied).

## Noise model check (completed, useful, not yet acted on)

Manual Pearson correlation between local baseline level and local sigma, plus
CV (sigma/baseline) at low vs high baseline quartiles:

| Chan | corr(sigma,baseline) | CV_low | CV_high | reading |
|---|---|---|---|---|
| alt  | 0.793  | 0.136  | 0.361  | leans multiplicative |
| CO2  | 0.807  | 0.008  | 0.021  | CV tiny either way — essentially additive in practice |
| CH4  | 0.940  | 0.074  | 0.371  | multiplicative — CV grows ~5x with level |
| temp | 0.774  | 0.0007 | 0.0068 | additive, noise negligible vs level |
| O2   | -0.602 | 0.0013 | 0.0004 | negative corr is noise/small-sample artifact, not signal |

Practical implication: CH4 (and maybe alt) baseline/anomaly work might belong in
log space; CO2/temp are fine in linear space. This bears on the log-scale figure
request too — for CH4 a log axis isn't just about dynamic range display, it may
better match the actual noise structure.

## Process note for next session

The last several turns spawned a fresh whole-script MATLAB diagnostic per
question instead of building incrementally on prior output — slow, and made it
hard to give quick "yes I'm working" acknowledgements to the user's short
check-in messages. Prefer: keep one running diagnostic scratch script, append
to it, avoid re-deriving things already computed (segment lists, baseline
structs) each time.

## Verified environment facts (don't re-derive)

- `movmad` exists, confirmed median-based (1.4826*movmad recovered true sigma=1
  from synthetic noisy+spike data; a 50-sigma 5-sample spike inflated local MAD
  only 22%).
- `movmad(..., "omitnan")` supported; default (no omitnan) propagates NaN.
- No Statistics/Econometrics Toolbox → no `corr`, no `movquant`.
- `gradient`, `smoothdata`, `fillmissing`, `polyfit`, `polyval` all confirmed
  present (base MATLAB).
- Running headless: `matlab -batch "addpath('/home/troyboland/epr_imaging/epr_imaging'); set(0,'DefaultFigureVisible','off'); run('script.m')"`
  — must addpath explicitly and use an absolute path for `filename` since batch
  mode's cwd isn't the script's directory.
- Local functions defined in Grapher.m are NOT visible to separate scratch
  scripts run via `run()` — must prepend the script body (everything before
  the first `function` line) rather than calling Grapher.m's own functions
  from an external probe script.
