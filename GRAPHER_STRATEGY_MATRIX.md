# Grapher.m — 4×4 Anomaly-Detection Strategy Matrix

**Scientific goal, points 1 & 2:** for a sensor channel on a Jason dive/trackline,
(1) establish a model of the *local baseline* reading, and (2) identify *anomalies*
as deviations from that baseline. This document fixes **4 baseline techniques** and
**4 detector techniques** that combine into **16 distinct full strategies**, so we can
see *which combinations flag which parts of the data*. There is no ground truth today;
the comparison is visual + overlap-based, not scored against a rubric.

---

## Design contract (what makes the 16 clean and comparable)

Every strategy is `baseline(x) → b, σ` followed by `detector(x, b, σ) → anomaly mask`,
run **within each contiguous valid segment** (reusing the existing
`baselineWithinSegments` scaffold and speed/mask segmentation, so no estimator bridges
the NaN gaps the validity mask creates). Two functions with fixed signatures:

```
[b, sigma] = BASELINE(x, dt, opts)      % b = local baseline, sigma = local noise scale
mask       = DETECTOR(x, b, sigma, dt, opts)   % logical anomaly mask
```

Decoupling baseline from detector is the whole point: today Approach A and Approach B
each hard-wire *one* baseline to *one* detector. Splitting them lets, e.g., the
derivative-excision baseline be paired with a peak detector, or the running-median
baseline be paired with a spectral detector — combinations that don't exist yet.

Held constant across all 16 (so differences are attributable to the method, not the
knobs): per-segment operation, one-sided detection (plumes are positive excursions for
CO₂/CH₄; sign is per-channel and configurable), robust `σ` measured on the
anomaly-removed residual, and the same channel set (CO₂, CH₄, O₂, Temperature, alt).

---

## The 4 baseline techniques

Each answers "what is the *background* level here?" with a different bias/failure mode.

### B1 — Running-median filter (drift-tracking, iterative clip)
**What:** `movmedian` over a wide window (~4000 s), then 1–3 passes that clip
one-sided excursions above `b + k·σ` and re-median, then a Gaussian polish.
**This is the current Approach A.** Rank-based, so immune to spike *magnitude*.
**Fails when:** an excursion occupies >~50 % of the window (long plumes) — the 50 % law
pulls the baseline up into the event.
**Env:** `movmedian`, `movmad`, `smoothdata` — all base MATLAB. ✔

### B2 — Derivative-excision baseline (edge-find, cut, bridge, re-median)
**What:** smooth → threshold the robust-z of the derivative to find event *onsets* →
walk each onset forward to a level-based settle point → excise the whole event (with an
amplitude gate) → linear-fill the holes → median the cleaned signal.
**This is the current Approach B.** Duration-agnostic, so long plumes don't defeat it.
**Fails when:** a genuinely fast *baseline* change (thermocline crossing) looks like a
plume onset and gets excised; depends on sharp onsets.
**Env:** `gradient`, `movmad`, `fillmissing` — base MATLAB. ✔

### B3 — Robust Gaussian low-pass baseline (spike-replace, then smooth)
**What:** resample to a uniform grid → flag |x − local median| > k·σ outliers →
replace them with linear interpolation → Gaussian low-pass at a chosen cutoff period →
map back to original timestamps.
**This function already exists in Grapher.m** (`robustLowpassBaseline`) but is computed
and then *never used* — it's an orphan. Promoting it to a first-class baseline makes it
earn its keep. Frequency-domain framing (cutoff period) instead of rank-domain, so it
characterizes drift as "everything slower than T_cutoff."
**Fails when:** outlier pre-replacement misses a broad plume (only sharp spikes get
replaced), letting the low-pass ride up into it — the frequency-domain analogue of B1's
50 % law.
**Env:** `movmedian`, `smoothdata` (gaussian), `interp1`, `fillmissing` — base MATLAB. ✔

### B4 — Robust per-segment polynomial / linear detrend (parametric)
**What:** fit a low-order polynomial (degree 0/1/2, configurable) to the **quiet
(non-anomalous) samples only** within each segment via `polyfit`, iteratively
re-weighting: fit → drop samples > k·σ above the fit → refit until stable. The baseline
is the evaluated polynomial (`polyval`).
**This is the "linear baseline fit" the notes explicitly requested but never built.**
Parametric and global-within-segment, so it can't chase a plume the way a moving window
can — the strongest structural contrast to B1/B2/B3.
**Fails when:** real background curvature within a segment exceeds the polynomial order
(under-fit) — but that's a visible, honest failure, not a silent contamination.
**Env:** `polyfit`, `polyval` — base MATLAB (confirmed present). ✔
*(Robust `robustfit` is NOT available — hence the manual iterative-reweight loop.)*

**Why these four:** they span the two axes that actually matter here —
*rank vs. frequency vs. parametric* domain, and *local-moving vs. global-within-segment*
support. B1 (rank/local), B3 (frequency/local), B2 (rank/local + event surgery),
B4 (parametric/global). Any anomaly that survives all four is robust to the baseline
model choice; anomalies that appear under only one tell us the baseline assumption is
doing the work.

---

## The 4 detector techniques

Each takes `(x, b, σ)` and answers "which samples are anomalous?" differently.

### D1 — Hysteresis robust-z threshold (enter/exit + min-duration)
**What:** `z = (x − b)/σ`; enter an event when `z > enterK` (≈5), stay in until
`z < exitK` (≈2), reject events shorter than `minDurSec`.
**This is the current detector** shared by Approaches A and B. Hysteresis stops noise
chopping one plume into several; min-duration rejects sub-transit blips.
**Character:** amplitude-based, level-triggered. The reference detector.

### D2 — Anomaly-mass / area threshold (integrated excursion)
**What:** within each above-`exitK` run, require the *integrated* positive residual
`∫(x − b) dt` over the run to exceed a threshold (calibrated from the record's own mass
distribution). A tall-but-brief spike and a low-but-sustained plume are scored on the
same currency.
**Why it's distinct from D1:** D1 triggers on instantaneous height; D2 triggers on total
excess. The notes' *own* window-sweep diagnostic already computes anomaly mass — this
promotes that quantity from a tuning diagnostic to a detector. Catches long, low plumes
D1's height threshold misses; ignores single-sample height spikes with no mass.
**Env:** cumulative-sum / trapz over residual — base MATLAB. ✔

### D3 — Peak / prominence detector (`findpeaks` on the residual)
**What:** run `findpeaks` on the residual `r = x − b` with `MinPeakProminence = k·σ`,
`MinPeakDistance`, `MinPeakWidth`; expand each accepted peak to its half-prominence (or
`exitK·σ`) shoulders to form the event span.
**Why it's distinct:** D1/D2 are level/area threshold-crossers; D3 is a *shape*
detector — prominence measures how much a peak stands out from its surrounding
baseline-relative context, so it separates two merged plumes D1 would report as one long
event, and rejects a slow shoulder that clears a level threshold but isn't a peak.
**Env:** `findpeaks` — **now present** (Signal Toolbox is installed in R2026a; verified).
This is new capability the old notes predate. ✔

### D4 — `isoutlier` moving-window detector (distributional)
**What:** `isoutlier(r, "movmedian", W)` (or `"grubbs"`/`"gesd"` window modes) on the
residual — flags samples that are outliers relative to a local moving distribution,
independent of the global `enterK`. Then apply the shared hysteresis/min-duration
grouping so its events are comparable to D1–D3.
**Why it's distinct:** D1–D3 all reference the single fitted `σ`; D4 re-derives locality
from its *own* moving-window MAD, so it responds where the baseline's `σ` estimate and
the true local spread disagree — a useful independent check on the σ model itself
(cf. the noise-model finding that CH₄ is multiplicative and CO₂ additive).
**Env:** `isoutlier` — base MATLAB (confirmed present). ✔

**Why these four:** they detect on four different features of the same residual —
**height** (D1), **area** (D2), **shape/prominence** (D3), **local distribution** (D4).
An event flagged by all four is unambiguous; the *pattern* of which detectors fire is
itself diagnostic of event morphology (sharp vs. sustained vs. merged vs. subtle).

---

## The 16 strategies

|          | **D1** Hysteresis-z | **D2** Anomaly-mass | **D3** Peak/prominence | **D4** `isoutlier`-moving |
|----------|:---:|:---:|:---:|:---:|
| **B1** Running-median | ✅ *(= today's Approach A)* | S2 | S3 | S4 |
| **B2** Derivative-excision | ✅ *(= today's Approach B)* | S6 | S7 | S8 |
| **B3** Robust Gaussian low-pass | S9 | S10 | S11 | S12 |
| **B4** Robust polynomial detrend | S13 | S14 | S15 | S16 |

The two diagonal cells marked ✅ are the strategies Grapher.m already runs — so the
harness *contains* the current script as two of its 16 cells and we can confirm it
reproduces today's numbers (CO₂: 14 events/7.3 %; temp: 86 vs 85) before trusting the
other 14.

---

## How the 16 get compared (visual + overlap, no ground truth)

Per the decision: **no scoring rubric, no truth labels** — the deliverable is *seeing
which combinations flag which parts of the data*. Concretely, per channel:

1. **Overlay strip figure** — the channel with all 16 flagged-event masks stacked as
   rows beneath it (or a 16-row "raster" of flags on a shared time axis), so a glance
   shows where methods agree and where they diverge.
2. **Agreement heatmap** — a 16×16 Jaccard-overlap matrix of the flagged sample sets
   (which strategies flag the same samples), to see clusters of methods that behave
   alike vs. outlier methods. This is description, not scoring — it says *how similar*,
   not *which is right*.
3. **Consensus track** — for each sample, the count (0–16) of strategies flagging it,
   plotted as a heat track; high-consensus regions are candidate events for the
   photogrammetry cross-check (question 3), low-consensus regions are where the method
   choice matters.
4. **Descriptive table** — per strategy: event count, % time flagged, total anomaly
   mass, median event duration. Reported side by side, explicitly *not* ranked.

---

## Environment reconciliation (differs from GRAPHER_NOTES.md — reverified in R2026a)

| Function | Old note said | Verified now | Enables |
|---|---|---|---|
| `findpeaks` | (assumed absent w/ Signal TB) | **present** (Signal) | **D3** |
| `isoutlier` | not mentioned | **present** (base) | **D4** |
| `filloutliers`, `prctile`, `quantile`, `ischange`, `detrend` | not mentioned | **present** (base) | robust helpers |
| `polyfit` / `polyval` | present | present | **B4** |
| `corr`, `movquant`, `robustfit`, `mad`, `wavedec`/`cwt` | absent | **still absent** | ⇒ manual Pearson, manual reweight, no wavelet detector |

The still-absent set is why the matrix has **no wavelet/CWT baseline** and **no
`robustfit` parametric baseline** — B4 uses an explicit iterative-reweight loop instead.

---

## Open design choices to confirm before building the harness

1. **Per-channel sign & log-space.** CO₂/CH₄ are one-sided positive and span decades
   (CH₄ noise is multiplicative per the noise-model table) → likely detect in log space;
   O₂ anomalies may be *negative* (depletion) → two-sided or sign-flipped; temp/alt are
   additive. Should the harness carry a per-channel `sign` + `logSpace` flag?
2. **The degraded-segment bug still stands.** `clampWindow` silently shrinks the baseline
   window inside short segments, and segments 8–9 are contaminated by masked-plume tails.
   All four baselines inherit this. Recommend fixing it *once* in the shared segment
   scaffold before running 16 strategies on top of it, so we're not comparing 16 flavors
   of the same contamination. (Candidate fixes are listed in GRAPHER_NOTES.md.)
3. **Shared vs. per-strategy knobs.** `enterK/exitK/minDurSec` shared across detectors
   keeps the comparison fair; but D2 (mass) and D3 (prominence) need their own natural
   thresholds. Propose: shared `enterK/exitK/minDur` where they apply, plus one
   documented native threshold each for D2/D3, calibrated from the record's own
   distribution (self-referential, not hand-tuned).
