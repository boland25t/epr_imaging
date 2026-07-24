# Grapher Anomaly-Detection Strategies — Baseline & Detector Reference

This documents every baseline and detector technique implemented in
`GrapherMatrix.m`, how each works, its failure mode, and what the comparison has
revealed on the J1754 record (`interp_full.csv`: nav + CO₂/CH₄/O₂/Temperature/alt,
~1 Hz, ~24 h, 85 920 rows). It supports scientific-goal questions 1 (model the
local baseline) and 2 (identify anomalies as deviations).

**Everything runs *within contiguous valid segments*** (speed-mask → segments),
so no estimator bridges the masked gaps. **Every channel is oriented first** so
that "anomaly" always means "large *positive* residual": a per-channel `sign`
(−1 for O₂, whose anomalies are depletions) flips the data, and a `logSpace`
flag (CH₄ only, whose noise is multiplicative) moves it to log space before any
baseline or detector sees it.

---

## Part 1 — Baseline techniques (question 1)

A baseline is a **per-timestamp series `b(t)`** — the model of the drifting
background. It need not have a closed form; a value at every sample is the model.
All four are non-constant and track drift; they differ in *how* they decide what
is background versus excursion.

### B1 — Running-median filter (drift-tracking, iterative one-sided clip)
- **How:** a wide moving median (`movmedian`, ~4000 s window) tracks the slow
  background by construction (a running median is already local). Three passes
  then clip one-sided excursions above `b + clipK·σ` and re-median, correcting
  the duty-cycle bias a plain median still carries. A final Gaussian pass polishes
  the median's staircase.
- **Domain:** rank-based, local-moving. Immune to spike *magnitude*.
- **Failure mode:** the **50 % law** — an excursion occupying more than ~half the
  window pulls the baseline up into itself. Long plumes defeat it. (This is why
  the window was raised to 4000 s: at 1000 s it absorbed 32–44 % of the CO₂/CH₄
  anomaly mass.)

### B2 — Derivative-excision baseline (find edges, cut whole events, re-median)
- **How:** denoise → threshold the robust-z of the *derivative* to find event
  *onsets* → walk each onset forward to a level-based settle point → excise the
  whole event (guarded by an amplitude gate so quiet noise isn't shredded) →
  linear-fill the holes → median the cleaned signal.
- **Domain:** rank-based + explicit event surgery. **Duration-agnostic**, so long
  plumes don't defeat it — the whole event is cut regardless of length.
- **Failure mode:** assumes plumes have *sharp onsets*; a genuinely fast baseline
  change (a thermocline crossing) looks like a plume onset and gets excised too.

### B3 — Robust Gaussian low-pass (strip plumes, then smooth at a cutoff period)
- **How:** an **iterative, one-sided** outlier pre-pass removes positive
  excursions (a single wide-median pass is itself biased upward by a big plume,
  so it re-flags against the cleaned signal ~3× to converge on a plume-free
  support), then a Gaussian low-pass at a chosen cutoff period (~4000 s) keeps
  "everything slower than the background."
- **Domain:** frequency-based, local. Frames the baseline as a low-pass problem.
- **Failure mode / history:** originally used a **30 s** outlier window that could
  not span multi-minute plumes, so plumes survived the pre-pass and the low-pass
  was dragged into every excursion — B3 tracked the raw data far too closely.
  **Fixed** by widening the outlier window to ~1000 s and iterating. It is the
  frequency-domain analogue of B1's 50 % law: without event-aware pre-removal, a
  low-pass has no defense against sustained excursions.
- **Note:** correctly follows genuine *downward* troughs (real low background),
  since the one-sided pre-pass only strips plumes — a trough is not an anomaly for
  a positive-sign channel.

### B4 — Robust per-segment polynomial detrend (parametric)
- **How:** fit a low-order polynomial (degree 2 here) to the **quiet
  (non-anomalous) samples only** within each segment via `polyfit`, iteratively
  reweighted: fit → drop samples above `clipK·σ` → refit, ~3×. (`robustfit` isn't
  available in this MATLAB, so the reweighting loop is explicit.) The baseline is
  the evaluated polynomial — the **only technique that is a closed-form model**
  with coefficients you can write down, extrapolate, and compare across dives.
- **Domain:** parametric, global-within-segment.
- **Failure mode:** real background curvature exceeding the polynomial order is
  under-fit — but that's a *visible, honest* failure, not silent contamination.
- **Observed strength:** because a quadratic is *structurally incapable* of
  chasing a fast excursion, it neither rides up into plumes (B1's flaw) nor dives
  into sharp dips — it treats them as the anomalies they are. On the CO₂ record it
  is the most robust of the four (see Findings).

**Degraded segments (shared by all four).** 21 of 23 valid segments are shorter
than the baseline window and cannot resolve their own drift. Such a segment gets
a **flat baseline anchored to its own robust quiet level** (plumes stripped),
falling back to a record-wide pooled quiet level only when it is too short
(<30 samples) to estimate even that. These are marked `degraded` and shaded grey
in the figures. Anchoring to the record-wide pool instead was wrong — it flagged a
whole locally-elevated segment (segment 9 went 1222/1222 → 402/1222 once fixed).

---

## Part 2 — Detector techniques (question 2)

A detector takes `(x, b, σ)` and returns a logical anomaly mask. All four detect
on the residual `r = x − b`, but on **four different features** of it, so the
*pattern* of which detectors fire is itself diagnostic of event morphology.

### D1 — Hysteresis robust-z threshold (HEIGHT)
- **How:** `z = (x−b)/σ`; enter an event when `z > enterK` (5), stay in until
  `z < exitK` (2), reject events shorter than `minDurSec` (20 s).
- **Feature:** instantaneous height. Hysteresis stops noise chopping one plume
  into several; min-duration rejects sub-transit blips. The reference detector
  (Grapher.m's shared detector; the B1×D1 and B2×D1 cells reproduce Grapher's
  Approach A / B exactly, which validates the harness).

### D2 — Anomaly-mass / area threshold (AREA)
- **How:** form candidate runs at the *loose* exitK level (so low sustained plumes
  that never spike above enterK are still considered), compute each run's
  integrated positive residual in σ-seconds, and keep runs whose mass exceeds a
  self-referential robust fence (`median + enterK·MAD` of candidate masses, floored
  at a minDur-at-exit burst).
- **Feature:** total excess, not height. A tall brief spike and a low sustained
  plume are judged on the same currency — so D2 can *accept* a long low plume D1
  misses and *reject* a thin spike D1 keeps.

### D3 — Peak prominence (SHAPE)
- **How:** `findpeaks` on the residual-z with `MinPeakProminence = enterK`, then
  expand each accepted peak out to its exitK shoulders to form the event span.
- **Feature:** prominence — how much a peak stands out from its local
  baseline-relative context. Separates two merged plumes D1 reports as one long
  event, and rejects a slow shoulder that clears a level threshold but isn't a
  peak. Tends to yield fewer, cleaner, longer-attributed events.

### D4 — `isoutlier` moving-window (LOCAL DISTRIBUTION)
- **How:** `isoutlier(r, "movmedian", W)` flags samples that are outliers vs their
  *own* local moving distribution, independent of the fitted σ; positive-side only;
  each flagged sample is grown to its surrounding above-exit run.
- **Feature:** local distributional outlier. Because it re-derives locality from
  its own moving MAD rather than the fitted σ, it fires where the σ model and the
  true local spread disagree — an independent check on the noise model (relevant
  since CH₄ is multiplicative and CO₂ additive).

---

## Part 3 — Findings so far (visual + overlap; no ground truth, no ranking)

1. **Baseline family dominates detector choice.** In the 16×16 Jaccard heatmaps,
   the 12 moving-window strategies (B1/B2/B3 × any detector) form one
   high-agreement block; the 4 B4-polynomial strategies stand apart. *Which
   baseline you pick changes what counts as background far more than which
   detector you pick.*
2. **The polynomial (B4) is the most robust baseline on CO₂.** Around Jan 16
   04:00–06:00 the raw signal has a sharp dip; B1/B2/B3 follow it down (a moving
   estimator tracks fast changes into the baseline), while B4 sweeps smoothly
   through and reports the dip as an excursion. Up to ~700 units of
   baseline-to-baseline disagreement there — the clearest illustration of the
   parametric-vs-nonparametric tradeoff.
3. **B3 was broken and is now fixed.** Its short outlier window let plumes survive
   the pre-pass; widening + iterating the pre-pass makes it a fair fourth baseline
   that sits on the background rather than chasing it.
4. **Detectors genuinely diverge** once D2/D4 were corrected off D1: on CO₂/B1,
   D1≈17–22 events (height), D2 more (catches low sustained runs), D3≈12 (merges
   to fewer, longer), D4 its own (local-distribution trigger).

---

## Part 4 — The smoothing axis (implemented: 5 × 4 × 4 = 80 strategies)

A third axis, **input pre-smoothing**, is now applied *per segment* to the
transformed channel before the baseline is fit. Crucially, **detection still
compares the ORIGINAL (unsmoothed) signal to that baseline** — smoothing changes
the baseline-fitting *input*, not the data being judged — so this is the honest
"does pre-denoising the fit input change the anomaly picture?" test.

The five smoothers (`smoothSpanSec ≈ 11 s`):
- **S0none** — control (no pre-smoothing).
- **S1gauss** — short Gaussian window. Mild denoise, blurs edges a little.
- **S2median** — short `movmedian`. Robust to spikes, sharper edges than Gaussian.
- **S3sgolay** — Savitzky-Golay (`sgolay`). Polynomial-fit smoothing that
  preserves peak shape/height — matters for the prominence detector D3.
- **S4hampel** — Hampel filter: replace points >3 local MADs from the local
  median with that median, leave the rest untouched. The only "smoother" that
  edits *only* outliers (sensor glitches) without smearing clean data.

**Caveat (documented):** B2 already smooths internally before differentiating, so
"S0none" is not "no smoothing anywhere" — it is "no *additional* pre-smoothing."

**Preliminary finding (CO₂, B4-poly + D1):** all five smoothers give ~15 events
at ~17 %, Jaccard ≈ 0.95–1.0 vs the None control — **pre-smoothing is nearly
irrelevant for the polynomial baseline**, because a robust quadratic fit already
ignores sample-level noise. Whether pre-smoothing matters more for the
moving-window baselines (B1/B2/B3) and noisier channels (O₂'s subtle depletions)
is what the full run's smoothing-comparison figures show.

---

## Part 5 — Two regimes + scale classification (large/long anomalies)

The fine-tuned models **miss large and long anomalies by construction**: a plume
longer than ~half the 4000 s baseline window is absorbed *into* the baseline (the
50 % law / B3's frequency analogue), so the residual over it collapses and no
detector fires; and `maxEventSec` caps B2 at 1200 s. A single giant plume also
inflates the local σ near it, raising the threshold right where the event is.

The fix is a **second regime of the same 5×4×4 engine**, retuned for scale — not
just looser thresholds:

| Setting | Fine (short transits) | Coarse (large/long) | Why |
|---|---|---|---|
| baselineWindowSec | 4 000 | **20 000** (~5.5 h) | long plumes stay in the residual, not the baseline |
| sigmaWindowSec | 4 800 | 24 000 | pool σ so one giant plume can't hide itself |
| enterK / exitK | 5 / 2 | 3.5 / 1.5 | broad swells have lower peak-z but far more mass |
| minDurSec | 20 | 120 | a "long-event" floor |
| maxEventSec (B2) | 1 200 | 14 400 | allow events up to 4 h |
| polyOrder (B4) | 2 | 1 | over 5 h a line is the honest drift model |

Both regimes run in full (2 × 80 = **160 strategies/channel**). Their flags are
**merged into a consensus** (votes per sample), cut into events wherever ≥ 25 %
of the 160 strategies agree, and each event is **classified by scale**:

- **LARGE** — anomaly mass in the top 20 % of events (any duration). Wins over
  the duration split: a big event is "large" even if also long.
- **LONG-SUSTAINED** — duration ≥ 600 s (beyond the fine regime's reach).
- **SHORT-TRANSIT** — everything else (brief, any height).

Mass/peak are measured against the **coarse B4-poly baseline** (a wide-window fit
that does not ride up into long plumes), so long-event mass isn't collapsed.

**Per-regime transform (fixes a hidden-plume bug).** CH₄ is detected in log
space for its multiplicative noise — good for small/subtle anomalies, but log
*compresses a large multiplicative plume into a tiny deviation*, and a
plume-heavy segment inflates the log-σ, so the fine (log) regime missed CH₄'s
biggest plumes entirely (the 7443-peak plume at 20:45 was flagged by **0 of 80**
strategies, while CO₂'s comparable plume in linear space was caught by 76/80).
Fix: the transform is now **per regime** — the FINE regime keeps log (subtle
events); the COARSE regime always uses **linear** space, where a 10× spike reads
as ~3 σ. That plume is now flagged by **80/80** coarse strategies and classified
LARGE (13-min event). Only log-flagged channels (CH₄) are affected; the linear
channels are unchanged.

**Result (consensus events per channel):**

| Channel | LARGE | LONG-SUSTAINED | SHORT-TRANSIT |
|---|---|---|---|
| CO₂ | 4 | 1 | 14 |
| CH₄ | 3 | 5 | 6 |
| O₂ | 8 | 0 | 34 |
| temp | 20 | 0 | 100 |
| alt | 8 | 0 | 33 |

(CH₄ counts are after the per-regime-transform fix below; the previously-hidden
large plumes now appear.)

The gas channels now expose the long events the fine regime missed — e.g. CH₄
has an 11-minute LONG-SUSTAINED event (20:04) and a **39-minute LARGE** event
(23:53, peak-z 4.6, mass 3854) that the fine-only models were absorbing into the
baseline. Each channel's classified events are written to
`anomaly_events_<channel>.csv` (ISO `start_time`/`end_time` + `class`,
`consensus`, `peak_z`, `mass`) — import-ready for the photogrammetry cross-check.

---

## Part 6 — Masking vs spike-suppression (4 run configs)

The speed-validity mask discards low-speed samples — but a vehicle *hovering in a
plume* is masked precisely when the strongest readings occur, and masking is a
blunt way to stop colossal spikes from dragging a moving baseline up. So the
matrix now runs once per **run config**:

| tag | masking | spike suppression | record retained |
|---|---|---|---|
| `masked` | speed/gap mask | none (original) | 71 % |
| `nomask_log` | **off** | log-scale ALL channels (both regimes) | 100 % |
| `nomask_ceiling` | **off** | per-channel 99th-pctile clip (fit input only) | 100 % |
| `nomask_both` | **off** | ceiling + log | 100 % |

The key mechanic: the **baseline-fit input** and the **detection signal** are now
distinct. The ceiling clips (or log compresses) only the *fit* input, so a
colossal spike can't drag the baseline up — but detection still sees the true
height, so the spike is still flagged.

**Proof it works (CH₄, the 9395-peak spike at 22:07, true background ~7):**

| run | coarse baseline near the spike | spike flagged |
|---|---|---|
| masked | dragged up to **878** | 0 % (missed) |
| nomask_log | **36** | 71 % ✓ |
| nomask_ceiling | **34** | 75 % ✓ |
| nomask_both | **36** | 71 % ✓ |

Masking inflated the baseline to 878 and missed the spike; all three suppression
runs hold the baseline at the true background and catch it. **No-masking recovers
more events on every channel; `nomask_ceiling` is the most sensitive** (e.g. CH₄
8 LONG-SUSTAINED vs 5 masked; temp finds 3 LONG-SUSTAINED no other run does).
The polynomial baseline (B4) stays flat through the spikes in both regimes.

Outputs per run: `grapher_matrix_figs/<tag>/` (30 figures),
`grapher_matrix_results_<tag>.mat`, and `anomaly_events_<channel>.csv` in that
folder.

---

## Figures produced (per channel, 6 each = 30 total)

0. **anomaly classification** — both regimes merged; channel with events coloured
   by scale class (LARGE / LONG-SUSTAINED / SHORT-TRANSIT) + a 0-to-160 consensus
   track. The payoff figure for the large/long question.

1. **baseline comparison** — all 4 baseline models overlaid (y-scaled to the
   baseline band so they're visible), plus a baseline-disagreement track.
2. **16 strategies** — channel + 16-row flag raster (one smoother slice).
3. **strategy agreement (Jaccard)** — 16×16 flagged-sample overlap heatmap.
4. **consensus** — channel + 0-to-16 count of strategies flagging each sample.
5. **smoothing comparison** — for a fixed baseline+detector, flag raster across
   the 5 smoothers + Jaccard of each vs the None control.

Masks + metrics for all 80 strategies × 5 channels are saved to
`grapher_matrix_results.mat` for the photogrammetry cross-check (questions 3–5).
