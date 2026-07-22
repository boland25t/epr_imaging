%% Settings

filename = "interp_full.csv";

speedThreshold    = 0.05;
invalidBuffer     = 30;      % seconds before and after each invalid region
shortGapThreshold = 60;      % seconds; shorter buffered gaps become valid

navWindow = 300;
navSmooth = "gaussian";

% Plot toggles
showSpeed       = true;
showValidMask   = true;
showAltitude    = true;
showCO2         = true;
showCH4         = true;
showTemperature = true;
showO2          = true;

showRegionShading = true;

% Region shading settings
shadeValidRegions   = true;
shadeInvalidRegions = true;

validShadeColor   = [0.85, 1.00, 0.85];  % light green
invalidShadeColor = [1.00, 0.85, 0.85];  % light red
regionShadeAlpha  = 0.20;

showSegmentBoundaries = true;
linkTimeAxes          = true;


%% Load CSV

data = readtable(filename);


%% Extract channels

time_channel = datetime( ...
    data.timestamp_iso, ...
    "InputFormat", "yyyy-MM-dd'T'HH:mm:ss");

time_channel = time_channel(:);

lat_channel  = data.lat(:);
lon_channel  = data.lon(:);
alt_channel  = data.alt(:);

CO2_channel  = data.CO2(:);
CH4_channel  = data.CH4(:);
O2_channel   = data.O2(:);
temp_channel = data.Temperature(:);


%% Calculate navigation speed

v = navSpeed( ...
    time_channel, ...
    lat_channel, ...
    lon_channel, ...
    alt_channel, ...
    "Window", navWindow, ...
    "Smooth", navSmooth);

v = v(:);


%% Convert timestamps to elapsed seconds

tSec = seconds(time_channel - time_channel(1));
tSec = tSec(:);

% Typical sample interval
dt = median(diff(tSec), "omitnan");

if ~isfinite(dt) || dt <= 0
    error("Could not determine a valid sampling interval.");
end


%% Create initial validity mask

initialValidMask = isfinite(v) & (v > speedThreshold);
initialValidMask = initialValidMask(:);


%% Expand invalid regions by +/- invalidBuffer

initialInvalidMask = ~initialValidMask;

invalidTransitions = diff([false; initialInvalidMask; false]);

invalidStarts = find(invalidTransitions == 1);
invalidEnds   = find(invalidTransitions == -1) - 1;

bufferedInvalidMask = initialInvalidMask;

if invalidBuffer > 0
    for k = 1:numel(invalidStarts)
        iStart = invalidStarts(k);
        iEnd   = invalidEnds(k);

        bufferStartTime = tSec(iStart) - invalidBuffer;
        bufferEndTime   = tSec(iEnd)   + invalidBuffer;

        inBufferedRegion = ...
            tSec >= bufferStartTime & ...
            tSec <= bufferEndTime;

        bufferedInvalidMask(inBufferedRegion) = true;
    end
end

% Convert back to a validity mask after buffering
validMask = ~bufferedInvalidMask;


%% Convert short buffered invalid intervals back to valid

invalidMask = ~validMask;

invalidTransitions = diff([false; invalidMask; false]);

invalidStarts = find(invalidTransitions == 1);
invalidEnds   = find(invalidTransitions == -1) - 1;

for k = 1:numel(invalidStarts)
    iStart = invalidStarts(k);
    iEnd   = invalidEnds(k);

    gapDuration = tSec(iEnd) - tSec(iStart) + dt;

    if gapDuration < shortGapThreshold
        validMask(iStart:iEnd) = true;
    end
end


%% Replace invalid samples with NaN

alt_channel(~validMask)  = NaN;
CO2_channel(~validMask)  = NaN;
CH4_channel(~validMask)  = NaN;
O2_channel(~validMask)   = NaN;
temp_channel(~validMask) = NaN;


%% Find contiguous valid segments

validTransitions = diff([false; validMask; false]);

segmentStarts = find(validTransitions == 1);
segmentEnds   = find(validTransitions == -1) - 1;

numSegments = numel(segmentStarts);


%% Store valid segments in a structure

segments = struct( ...
    "indices", {}, ...
    "time", {}, ...
    "alt", {}, ...
    "CO2", {}, ...
    "CH4", {}, ...
    "O2", {}, ...
    "temp", {});

for k = 1:numSegments
    idx = segmentStarts(k):segmentEnds(k);

    segments(k).indices = idx;
    segments(k).time    = time_channel(idx);
    segments(k).alt     = alt_channel(idx);
    segments(k).CO2     = CO2_channel(idx);
    segments(k).CH4     = CH4_channel(idx);
    segments(k).O2      = O2_channel(idx);
    segments(k).temp    = temp_channel(idx);
end

fprintf("Found %d valid segments.\n", numSegments);

%% Robust low-pass baseline settings

% Approximate shortest variation that should be considered part of the
% baseline. Increase this value for a smoother/slower baseline.
baselineCutoffPeriodSec = 1000;     % 5 minutes

% Window used to identify sharp outliers before calculating the baseline.
outlierWindowSec = 30;

% Larger values remove only more extreme outliers.
outlierThreshold = 4.0;


%% Calculate baseline trendlines

CO2_baseline = robustLowpassBaseline( ...
    time_channel, CO2_channel, ...
    baselineCutoffPeriodSec, ...
    outlierWindowSec, ...
    outlierThreshold);

CH4_baseline = robustLowpassBaseline( ...
    time_channel, CH4_channel, ...
    baselineCutoffPeriodSec, ...
    outlierWindowSec, ...
    outlierThreshold);

O2_baseline = robustLowpassBaseline( ...
    time_channel, O2_channel, ...
    baselineCutoffPeriodSec, ...
    outlierWindowSec, ...
    outlierThreshold);

temp_baseline = robustLowpassBaseline( ...
    time_channel, temp_channel, ...
    baselineCutoffPeriodSec, ...
    outlierWindowSec, ...
    outlierThreshold);

alt_baseline = robustLowpassBaseline( ...
    time_channel, alt_channel, ...
    baselineCutoffPeriodSec, ...
    outlierWindowSec, ...
    outlierThreshold);


%% Calculate baseline-relative anomalies

CO2_anomaly  = CO2_channel  - CO2_baseline;
CH4_anomaly  = CH4_channel  - CH4_baseline;
O2_anomaly   = O2_channel   - O2_baseline;
temp_anomaly = temp_channel - temp_baseline;
alt_anomaly  = alt_channel  - alt_baseline;

%% Baseline comparison settings (Figures 3 and 4)

% Two competing ways to keep spikes out of a drifting baseline.
%
%   Approach A (median filter): reject spikes by RANK.  Immune to spike
%       magnitude, but governed by the 50% law - an excursion is only rejected
%       if it occupies less than half the window.  Long plumes defeat it.
%
%   Approach B (derivative excision): find spikes by their EDGES and cut the
%       whole event out.  Duration-agnostic, so it handles long plumes that
%       Approach A absorbs - but it depends on plumes having sharp onsets.
%
% Both estimate a LOCAL (time-varying) sigma, so a drifting noise level does
% not make the anomaly threshold meaningless in quiet or noisy stretches.

% Shared. baselineWindowSec must satisfy: plume duration << window << drift
% timescale.  If those two timescales overlap, the decomposition is ill-posed
% and no setting here rescues it.
%
% 4000s is not a guess - it is where a window sweep stops changing the answer.
% Total anomaly mass (integral of the positive residual) vs window on this
% record grows and then plateaus:
%
%   window:     250s     500s    1000s    2000s    4000s    8000s   16000s
%   CO2:     2.65e+06 6.47e+06 1.75e+07 2.55e+07 2.62e+07 2.56e+07 2.56e+07
%   CH4:     2.11e+05 9.35e+05 2.90e+06 4.86e+06 5.12e+06 5.19e+06 5.20e+06
%
% At 1000s the baseline was riding up into the plumes and absorbing ~32% of the
% CO2 and ~44% of the CH4 anomaly mass.  Everything has plateaued by 4000s.
% This also agrees with the 50% law from the measured event durations (CO2 max
% 381s, CH4 max 529s): a 4000s window puts the longest event at ~4x margin
% inside the half-window limit.
%
% NOTE: the median baseline LEVEL is a useless diagnostic here - it moves <0.2%
% across a 64x window sweep, because it is set by the quiet 95% of the record.
% The bias is local to the plumes, so only the anomaly-mass integral sees it.
baselineWindowSec = 4000;
sigmaWindowSec    = 4800;   % wide: sigma needs many quiet samples to pool

% Shared anomaly detection.  One-sided (plumes are positive excursions),
% with hysteresis so noise cannot chop one plume into several detections.
anomalyEnterK   = 5;    % enter an anomaly above this robust z
anomalyExitK    = 2;    % stay in it until z drops below this
anomalyMinDurSec = 20;  % reject events shorter than a plausible transit

% Approach A only
A_clipK = 2.0;   % clip positive excursions above b + k*sigma before re-median
A_nIter = 3;

% Approach B only
B_smoothSec   = 5;     % denoise BEFORE differentiating (d/dt amplifies noise)
B_slopeK      = 5;     % robust z on the derivative that opens an event
B_settleK     = 2;     % close the event once level returns within k*sigma
B_maxEventSec = 1200;  % safety cap (longest measured event is CH4 at 529s)
B_padSec      = 5;     % excise a little either side of each event

% Amplitude gate. A slope test alone has no sense of scale: in a very quiet
% stretch the local derivative sigma collapses, so any faint wiggle clears
% slopeK and gets excised. Without this gate, O2 had 22.8% of the record cut
% out while only 0.1% was ever flagged as an anomaly - the detector was
% shredding flat, quiet data. An event must also be a real LEVEL excursion.
B_minAmpK = 4;         % peak must exceed pre-event level by k*sigma

% Degraded-segment handling (shared by both approaches, see
% baselineWithinSegments).  A segment shorter than minResolvableFrac * window
% cannot resolve its own drift, so its baseline is anchored to a pooled,
% record-wide QUIET level instead of its own (possibly plume-contaminated)
% median.  poolOutlierK sets how aggressively plumes are stripped before that
% pooled level is measured.
minResolvableFrac = 1.0;   % require a full window of clean record to self-baseline
poolOutlierK      = 3.0;   % MADs from the global median that count as "plume"

optsA = struct( ...
    "baselineWindowSec", baselineWindowSec, ...
    "sigmaWindowSec",    sigmaWindowSec, ...
    "enterK",            anomalyEnterK, ...
    "exitK",             anomalyExitK, ...
    "minDurSec",         anomalyMinDurSec, ...
    "clipK",             A_clipK, ...
    "nIter",             A_nIter, ...
    "minResolvableFrac", minResolvableFrac, ...
    "poolOutlierK",      poolOutlierK);

optsB = struct( ...
    "baselineWindowSec", baselineWindowSec, ...
    "sigmaWindowSec",    sigmaWindowSec, ...
    "enterK",            anomalyEnterK, ...
    "exitK",             anomalyExitK, ...
    "minDurSec",         anomalyMinDurSec, ...
    "smoothSec",         B_smoothSec, ...
    "slopeK",            B_slopeK, ...
    "settleK",           B_settleK, ...
    "maxEventSec",       B_maxEventSec, ...
    "padSec",            B_padSec, ...
    "minAmpK",           B_minAmpK, ...
    "minResolvableFrac", minResolvableFrac, ...
    "poolOutlierK",      poolOutlierK);


%% Calculate both baselines

% Channels are baselined WITHIN each contiguous valid segment.  Running the
% baseline across the whole record would let the median window bridge the NaN
% gaps the validity mask just created, interpolating a baseline across parked
% stretches where there is no data to support one.
baselineChannels = { ...
    "alt",  alt_channel
    "CO2",  CO2_channel
    "CH4",  CH4_channel
    "temp", temp_channel
    "O2",   O2_channel };

baseA = struct();
baseB = struct();

for k = 1:size(baselineChannels, 1)
    name = baselineChannels{k, 1};
    x    = baselineChannels{k, 2};

    [bA, sA, anomA, ~, degA] = baselineWithinSegments( ...
        x, segmentStarts, segmentEnds, dt, "median", optsA);

    [bB, sB, anomB, cutB, degB] = baselineWithinSegments( ...
        x, segmentStarts, segmentEnds, dt, "derivative", optsB);

    baseA.(name) = struct("b", bA, "sigma", sA, "anom", anomA, ...
                          "raw", x, "excised", false(size(x)), "degraded", degA);
    baseB.(name) = struct("b", bB, "sigma", sB, "anom", anomB, ...
                          "raw", x, "excised", cutB, "degraded", degB);
end

% Duty cycle decides whether Approach A's clipping iteration is even needed:
% below ~10% elevated the plain median filter is already unbiased.  The
% "degraded" column reports the fraction of valid samples on a pooled flat
% anchor because their segment was too short to resolve its own drift.
fprintf("\n%-6s | %-22s | %-22s | %-10s | %s\n", ...
    "Chan", "A: median filter", "B: deriv excision", "duty cycle", "degraded");
fprintf("%s\n", repmat("-", 1, 92));

for k = 1:size(baselineChannels, 1)
    name = baselineChannels{k, 1};
    nValid = nnz(isfinite(baselineChannels{k, 2}));

    nA = countEvents(baseA.(name).anom);
    nB = countEvents(baseB.(name).anom);
    duty = 100 * nnz(baseB.(name).excised) / max(1, nValid);
    degr = 100 * nnz(baseA.(name).degraded) / max(1, nValid);

    fprintf("%-6s | %3d events, %5.1f%% flagged | %3d events, %5.1f%% flagged | %8.1f%% | %6.1f%%\n", ...
        name, ...
        nA, 100*nnz(baseA.(name).anom)/max(1,nValid), ...
        nB, 100*nnz(baseB.(name).anom)/max(1,nValid), ...
        duty, degr);
end
fprintf("\n");


%% Display smoothing (Figure 2)

% Window for the smoothed copies shown in Figure 2.  DISPLAY ONLY: the raw
% channels, the validity mask, and the baselines/anomalies above are untouched.
displaySmoothWindowSec = 30;
displaySmoothMethod    = "gaussian";

displayWindowSamples = max(3, round(displaySmoothWindowSec / dt));

% Smooth WITHIN each contiguous valid segment.  Smoothing the whole record at
% once would let smoothdata bridge the NaN gaps, silently filling in regions
% the validity mask just removed and smearing data across hovers.
alt_smooth  = smoothWithinSegments(alt_channel,  segmentStarts, segmentEnds, displayWindowSamples, displaySmoothMethod);
CO2_smooth  = smoothWithinSegments(CO2_channel,  segmentStarts, segmentEnds, displayWindowSamples, displaySmoothMethod);
CH4_smooth  = smoothWithinSegments(CH4_channel,  segmentStarts, segmentEnds, displayWindowSamples, displaySmoothMethod);
O2_smooth   = smoothWithinSegments(O2_channel,   segmentStarts, segmentEnds, displayWindowSamples, displaySmoothMethod);
temp_smooth = smoothWithinSegments(temp_channel, segmentStarts, segmentEnds, displayWindowSamples, displaySmoothMethod);

% Speed has no NaN gaps, so it can be smoothed as a whole.
v_smooth = smoothdata(v, displaySmoothMethod, displayWindowSamples);


%% Plot enabled channels

plotEnabled = [
    showSpeed
    showValidMask
    showAltitude
    showCO2
    showCH4
    showTemperature
    showO2
];

numPlots = nnz(plotEnabled);

if numPlots == 0
    warning("No plots are enabled.");
    return;
end

% Gathered once so the two figures cannot drift apart.
plotOpts = struct( ...
    "plotEnabled",           plotEnabled, ...
    "numPlots",              numPlots, ...
    "speedThreshold",        speedThreshold, ...
    "showRegionShading",     showRegionShading, ...
    "shadeValidRegions",     shadeValidRegions, ...
    "shadeInvalidRegions",   shadeInvalidRegions, ...
    "validShadeColor",       validShadeColor, ...
    "invalidShadeColor",     invalidShadeColor, ...
    "regionShadeAlpha",      regionShadeAlpha, ...
    "showSegmentBoundaries", showSegmentBoundaries, ...
    "linkTimeAxes",          linkTimeAxes, ...
    "segmentStarts",         segmentStarts, ...
    "numSegments",           numSegments, ...
    "anomalyEnterK",         anomalyEnterK);

% --- Figure 1: raw (masked) channels ---
rawChannels = struct( ...
    "v", v, "validMask", validMask, "alt", alt_channel, ...
    "CO2", CO2_channel, "CH4", CH4_channel, ...
    "temp", temp_channel, "O2", O2_channel);

ax1 = plotChannelFigure( ...
    "Masked Time-Series Channels", ...
    time_channel, rawChannels, plotOpts, "");

% --- Figure 2: identical layout, smoothed channels ---
smoothChannels = struct( ...
    "v", v_smooth, "validMask", validMask, "alt", alt_smooth, ...
    "CO2", CO2_smooth, "CH4", CH4_smooth, ...
    "temp", temp_smooth, "O2", O2_smooth);

ax2 = plotChannelFigure( ...
    "Masked Time-Series Channels - Smoothed", ...
    time_channel, smoothChannels, plotOpts, ...
    sprintf(" (smoothed %gs)", displaySmoothWindowSec));

% --- Figure 3: Approach A - median-filter baseline ---
ax3 = plotBaselineFigure( ...
    "Baseline A - Median Filter", ...
    time_channel, baselineChannels, baseA, plotOpts, ...
    sprintf("median filter, %gs window", baselineWindowSec), false);

% --- Figure 4: Approach B - derivative excision ---
ax4 = plotBaselineFigure( ...
    "Baseline B - Derivative Excision", ...
    time_channel, baselineChannels, baseB, plotOpts, ...
    sprintf("derivative excision, %gs window", baselineWindowSec), true);

% Common time axis across ALL figures so panning one pans the rest.
if linkTimeAxes
    allAx = [ax1(:); ax2(:); ax3(:); ax4(:)];
    allAx = allAx(isgraphics(allAx));

    if numel(allAx) > 1
        linkaxes(allAx, "x");
    end
end


function [b, sigma, isAnom, isExcised, isDegraded] = baselineWithinSegments( ...
    x, segStarts, segEnds, dt, method, opts)
%BASELINEWITHINSEGMENTS Run a baseline estimator on each valid segment.
%
% Keeps the estimator from bridging the NaN gaps the validity mask created.
%
% DEGRADED-SEGMENT HANDLING (fixes the short-segment self-contamination bug):
% a segment shorter than the baseline window cannot self-characterize a drifting
% baseline - there aren't enough clean samples within it to outvote a
% contaminated edge.  On this record 21 of 23 segments are shorter than the
% 4000 s window, and segment 9 starts inside the tail of a masked-out plume, so
% its own median baseline is dragged ~46% above the true quiet level - which
% would suppress the very anomalies the baseline exists to expose.
%
% Fix: split segments into two tiers.
%   RESOLVABLE  (length >= minResolvableFrac * window)  -> self-characterize as
%               before; there is enough clean record to resolve local drift.
%   DEGRADED    (too short)                             -> anchor a FLAT baseline
%               to a pooled, record-wide QUIET level (robust median of all valid
%               samples with plumes removed), not to the segment's own possibly
%               contaminated median.  These samples are returned in isDegraded so
%               the plots can mark them (grey hatch) instead of silently trusting
%               a clamped-window fit.
%
% The pooled quiet anchor is computed ONCE from the whole channel, so a short
% segment abutting a plume no longer inherits that plume's level.

    x = x(:);
    n = numel(x);

    b          = nan(n, 1);
    sigma      = nan(n, 1);
    isAnom     = false(n, 1);
    isExcised  = false(n, 1);
    isDegraded = false(n, 1);

    % --- Pooled, record-wide quiet level + spread (the degraded-segment anchor).
    % Robust: drop samples more than poolOutlierK MADs from the global median
    % before taking the anchor, so masked-plume tails cannot lift it.
    poolOutlierK = getfielddef(opts, "poolOutlierK", 3.0);
    xv = x;   % x already carries NaN outside valid samples (channels are masked)
    gMed = median(xv, "omitnan");
    gMad = 1.4826 * median(abs(xv - gMed), "omitnan");
    gMad = max(gMad, eps);
    quiet = xv;
    quiet(abs(xv - gMed) > poolOutlierK * gMad) = NaN;
    pooledLevel = median(quiet, "omitnan");
    pooledSigma = max(1.4826 * median(abs(quiet - pooledLevel), "omitnan"), eps);
    if ~isfinite(pooledLevel)
        pooledLevel = gMed;   % fall back if the channel is all-outlier (unlikely)
    end

    % A segment must be at least this fraction of the baseline window to be
    % trusted to resolve its own drift; shorter ones get the pooled anchor.
    minResolvableFrac = getfielddef(opts, "minResolvableFrac", 1.0);
    wReqSamples       = opts.baselineWindowSec / dt;
    minResolvable     = max(5, round(minResolvableFrac * wReqSamples));

    for k = 1:numel(segStarts)
        idx = segStarts(k):segEnds(k);

        if numel(idx) < 5
            % Too short even to smooth: flat pooled anchor, marked degraded.
            b(idx)          = pooledLevel;
            sigma(idx)      = pooledSigma;
            isDegraded(idx) = true;
            continue;
        end

        xs = x(idx);

        % Interior NaNs would poison the moving statistics; the mask gaps are
        % already excluded by segmenting, so anything left is a sensor dropout.
        if any(isnan(xs))
            xs = fillmissing(xs, "linear", "EndValues", "nearest");
        end

        if numel(idx) < minResolvable
            % DEGRADED: too short to resolve DRIFT, but a short segment can still
            % estimate a flat LEVEL.  Anchor to the SEGMENT'S OWN robust quiet
            % median (plumes stripped), NOT the record-wide pool - anchoring a
            % locally-elevated segment to the record mean flagged its entire
            % extent (segment 9 went 1222/1222).  Fall back to the pool only when
            % the segment is too short (<30 samples) to strip its own plumes.
            % Detection still runs against the flat anchor so real excursions in
            % a short segment are not lost.
            level = segmentQuietLevel(xs, getfielddef(opts, "poolOutlierK", 3.0), 30);
            if ~isfinite(level), level = pooledLevel; end
            bk = repmat(level, numel(xs), 1);
            Ws = clampWindow(opts.sigmaWindowSec / dt, numel(xs));
            [sk, ak] = localSigmaAndAnomalies(xs, bk, dt, Ws, opts);
            sk = max(sk, eps);
            ck = false(size(xs));
            isDegraded(idx) = true;
        else
            switch method
                case "median"
                    [bk, sk, ak] = baselineMedian(xs, dt, opts);
                    ck = false(size(xs));
                case "derivative"
                    [bk, sk, ak, ck] = baselineDerivative(xs, dt, opts);
                otherwise
                    error("Unknown baseline method '%s'.", method);
            end
        end

        b(idx)         = bk;
        sigma(idx)     = sk;
        isAnom(idx)    = ak;
        isExcised(idx) = ck;
    end

    % Restore the mask: never report a baseline where there is no data.
    b(~isfinite(x))          = NaN;
    sigma(~isfinite(x))      = NaN;
    isDegraded(~isfinite(x)) = false;
end


function val = getfielddef(s, name, default)
%GETFIELDDEF Return s.(name) if present, else default.  Keeps new options
% backward-compatible with existing optsA/optsB structs.

    if isfield(s, name) && ~isempty(s.(name))
        val = s.(name);
    else
        val = default;
    end
end


function level = segmentQuietLevel(xs, outlierK, minSamples)
%SEGMENTQUIETLEVEL Robust quiet median of one segment, plumes stripped.
%
% A short segment cannot resolve DRIFT but can still estimate a flat LEVEL, as
% long as it has enough samples to strip its own excursions first.  Returns NaN
% when the segment is too short for that, so the caller falls back to the
% record-wide pooled anchor.

    xs = xs(isfinite(xs));
    if numel(xs) < minSamples
        level = NaN;
        return;
    end
    m = median(xs, "omitnan");
    s = max(1.4826 * median(abs(xs - m), "omitnan"), eps);
    quiet = xs;
    quiet(abs(xs - m) > outlierK * s) = NaN;
    level = median(quiet, "omitnan");
    if ~isfinite(level)
        level = m;
    end
end


function [b, sigma, isAnom] = baselineMedian(x, dt, opts)
%BASELINEMEDIAN Approach A - drift-tracking median filter with local sigma.
%
% A running median is ALREADY local, so it tracks drift by construction; the
% window length is the knob for how much drift it follows.  The iteration only
% corrects duty-cycle bias: the median is not perfectly spike-immune below the
% 50% law, it degrades gracefully (30% of a window elevated pulls the baseline
% up by ~0.57 sigma).  Clipping is one-sided because plumes are one-sided - a
% symmetric trim would still let them lift the baseline.

    n  = numel(x);
    Wb = clampWindow(opts.baselineWindowSec / dt, n);
    Ws = clampWindow(opts.sigmaWindowSec    / dt, n);

    b = movmedian(x, Wb, "omitnan");

    for it = 1:opts.nIter
        r     = x - b;
        sigma = max(1.4826 * movmad(r, Ws, "omitnan"), eps);
        xClip = min(x, b + opts.clipK .* sigma);
        b     = movmedian(xClip, Wb, "omitnan");
    end

    % Median output is piecewise-constant between rank changes; polish it so
    % the baseline does not look like a staircase over the data.
    b = smoothdata(b, "gaussian", Wb);

    [sigma, isAnom] = localSigmaAndAnomalies(x, b, dt, Ws, opts);
end


function [b, sigma, isAnom, isExcised] = baselineDerivative(x, dt, opts)
%BASELINEDERIVATIVE Approach B - find spike EDGES, cut the whole event out.
%
% Duration-agnostic: an event is excised however long it lasts, so this does
% not suffer the median filter's 50% law.  The cost is that it assumes plumes
% have sharp onsets, and that a genuinely fast baseline change (a thermocline
% crossing) looks identical to a plume onset and gets excised too.

    n  = numel(x);
    Wb = clampWindow(opts.baselineWindowSec / dt, n);
    Ws = clampWindow(opts.sigmaWindowSec    / dt, n);
    Wd = clampWindow(opts.smoothSec         / dt, n);

    % 1. Denoise BEFORE differentiating.  Differentiation amplifies white noise
    %    by ~sqrt(2)/dt, so the derivative of raw data is nearly all noise and
    %    every real edge is buried in it.
    xs = smoothdata(x, "gaussian", Wd);

    % 2. Robust threshold on the derivative itself.
    d      = gradient(xs, dt);
    dSigma = max(1.4826 * movmad(d, Ws, "omitnan"), eps);
    rising = (d ./ dSigma) > opts.slopeK;

    % 3. Level scale, used to decide when an event has settled.
    lvlSigma = max(1.4826 * movmad( ...
        x - movmedian(x, Wb, "omitnan"), Ws, "omitnan"), eps);

    % 4. Walk each rising edge forward until the signal settles back to its
    %    pre-event level.  The derivative can only OPEN an event, never close
    %    one: the flat top of a long plume has zero derivative, so a pure
    %    derivative test would cut the edges and leave the plume body behind.
    isExcised = false(n, 1);

    maxLen = max(1, round(opts.maxEventSec / dt));
    pad    = max(0, round(opts.padSec / dt));
    edges  = find(diff([false; rising]) == 1);

    for k = 1:numel(edges)
        i0 = edges(k);

        if isExcised(i0)
            continue;   % already inside an event we cut
        end

        preStart = max(1, i0 - Wd);
        preLevel = median(x(preStart:i0), "omitnan");
        tol      = opts.settleK * lvlSigma(i0);

        iEnd = min(n, i0 + maxLen);

        for i = (i0 + 1):min(n, i0 + maxLen)
            if abs(xs(i) - preLevel) <= tol
                iEnd = i;
                break;
            end
        end

        % Amplitude gate: the slope test has no sense of scale, so require the
        % event to actually go somewhere before cutting it out.  Otherwise a
        % quiet channel (tiny local derivative sigma) gets shredded by noise.
        peak = max(xs(i0:iEnd), [], "omitnan");

        if (peak - preLevel) < opts.minAmpK * lvlSigma(i0)
            continue;
        end

        isExcised(max(1, i0 - pad):min(n, iEnd + pad)) = true;
    end

    % 5. Cut the events out entirely and bridge the holes, so the baseline is
    %    supported only by samples the detector believes are background.
    xCut = x;
    xCut(isExcised) = NaN;
    xCut = fillmissing(xCut, "linear", "EndValues", "nearest");

    b = movmedian(xCut, Wb, "omitnan");
    b = smoothdata(b, "gaussian", Wb);

    [sigma, isAnom] = localSigmaAndAnomalies(x, b, dt, Ws, opts);
end


function [sigma, isAnom] = localSigmaAndAnomalies(x, b, dt, Ws, opts)
%LOCALSIGMAANDANOMALIES Time-varying noise scale, then one-sided detection.
%
% sigma is measured on the residual with anomalies REMOVED, so it pools quiet
% samples from across the whole record instead of assuming any one stretch is
% representative of a drifting dataset.

    r  = x - b;
    s0 = max(1.4826 * movmad(r, Ws, "omitnan"), eps);

    quiet = r;
    quiet(r ./ s0 > opts.enterK) = NaN;   % drop plumes before measuring noise

    sigma = 1.4826 * movmad(quiet, Ws, "omitnan");
    sigma = max(fillmissing(sigma, "linear", "EndValues", "nearest"), eps);

    z      = r ./ sigma;
    isAnom = hysteresisMask(z, opts.enterK, opts.exitK);
    isAnom = rejectShort(isAnom, max(1, round(opts.minDurSec / dt)));
end


function m = hysteresisMask(z, enterK, exitK)
%HYSTERESISMASK Enter above enterK, stay in until below exitK.
%
% Stops a single noisy sample dipping below the threshold from chopping one
% plume into several separate detections.

    z = z(:);
    m = false(size(z));
    inEvent = false;

    for i = 1:numel(z)
        if ~isfinite(z(i))
            inEvent = false;
        elseif ~inEvent && z(i) > enterK
            inEvent = true;
        elseif inEvent && z(i) < exitK
            inEvent = false;
        end

        m(i) = inEvent;
    end
end


function m = rejectShort(m, minSamples)
%REJECTSHORT Drop events shorter than a physically plausible transit time.

    d = diff([false; m(:); false]);
    s = find(d == 1);
    e = find(d == -1) - 1;

    for k = 1:numel(s)
        if (e(k) - s(k) + 1) < minSamples
            m(s(k):e(k)) = false;
        end
    end
end


function w = clampWindow(windowSamples, n)
%CLAMPWINDOW Force a moving window to be odd, >= 3, and no longer than n.

    w = max(3, round(windowSamples));

    if mod(w, 2) == 0
        w = w + 1;
    end

    w = min(w, n);

    if mod(w, 2) == 0
        w = max(3, w - 1);
    end
end


function n = countEvents(mask)
%COUNTEVENTS Number of contiguous true runs in a logical mask.

    n = nnz(diff([false; mask(:); false]) == 1);
end


function y = smoothWithinSegments(x, segStarts, segEnds, windowSamples, method)
%SMOOTHWITHINSEGMENTS Smooth each contiguous valid segment independently.
%
% Values outside the segments stay NaN, so the validity mask is preserved and
% the smoother never bridges a gap (which would smear data across a hover).

    x = x(:);
    y = nan(size(x));

    for k = 1:numel(segStarts)
        idx = segStarts(k):segEnds(k);

        if numel(idx) >= 3
            w = min(windowSamples, numel(idx));
            y(idx) = smoothdata(x(idx), method, w);
        else
            % Too short to smooth meaningfully - pass through unchanged.
            y(idx) = x(idx);
        end
    end
end


function ax = plotChannelFigure(figName, time_channel, ch, opts, titleSuffix)
%PLOTCHANNELFIGURE Build one stacked time-series figure.
%
% Shared by the raw and smoothed figures so their layout, shading, segment
% boundaries and axis linking are identical by construction.
%
%   ch   - struct with fields v, validMask, alt, CO2, CH4, temp, O2
%   opts - struct of toggles/settings (see plotOpts above)

    fig = figure("Name", figName);

    tl = tiledlayout( ...
        fig, ...
        opts.numPlots, ...
        1, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    ax = gobjects(opts.numPlots, 1);
    plotNumber = 0;

    showSpeed       = opts.plotEnabled(1);
    showValidMask   = opts.plotEnabled(2);
    showAltitude    = opts.plotEnabled(3);
    showCO2         = opts.plotEnabled(4);
    showCH4         = opts.plotEnabled(5);
    showTemperature = opts.plotEnabled(6);
    showO2          = opts.plotEnabled(7);

    % --- Speed ---
    if showSpeed
        plotNumber = plotNumber + 1;
        ax(plotNumber) = nexttile(tl);

        plot(ax(plotNumber), time_channel, ch.v, "LineWidth", 1);
        hold(ax(plotNumber), "on");

        yline( ...
            ax(plotNumber), ...
            opts.speedThreshold, ...
            "--", ...
            "Speed threshold", ...
            "HandleVisibility", "off");

        hold(ax(plotNumber), "off");

        ylabel(ax(plotNumber), "Speed");
        title(ax(plotNumber), "Navigation speed" + titleSuffix);
        grid(ax(plotNumber), "on");
    end

    % --- Validity mask (never smoothed - it is boolean) ---
    if showValidMask
        plotNumber = plotNumber + 1;
        ax(plotNumber) = nexttile(tl);

        stairs( ...
            ax(plotNumber), ...
            time_channel, ...
            double(ch.validMask), ...
            "LineWidth", 1);

        ylim(ax(plotNumber), [-0.1, 1.1]);
        yticks(ax(plotNumber), [0, 1]);
        yticklabels(ax(plotNumber), ["Invalid", "Valid"]);

        ylabel(ax(plotNumber), "Mask");
        title(ax(plotNumber), "Final validity mask");
        grid(ax(plotNumber), "on");
    end

    % --- Data channels ---
    channelSpecs = {
        showAltitude,    ch.alt,  "Altitude",    "Altitude"
        showCO2,         ch.CO2,  "CO_2",        "CO_2"
        showCH4,         ch.CH4,  "CH_4",        "CH_4"
        showTemperature, ch.temp, "Temperature", "Temperature"
        showO2,          ch.O2,   "O_2",         "O_2"
    };

    for k = 1:size(channelSpecs, 1)
        if ~channelSpecs{k, 1}
            continue;
        end

        plotNumber = plotNumber + 1;
        ax(plotNumber) = nexttile(tl);

        plot(ax(plotNumber), time_channel, channelSpecs{k, 2}, "LineWidth", 1);

        ylabel(ax(plotNumber), channelSpecs{k, 3});
        title(ax(plotNumber), channelSpecs{k, 4} + titleSuffix);
        grid(ax(plotNumber), "on");
    end

    % --- Shade valid and invalid regions ---
    if opts.showRegionShading
        for iAx = 1:opts.numPlots
            shadeTimeRegions( ...
                ax(iAx), ...
                time_channel, ...
                ch.validMask, ...
                opts.shadeValidRegions, ...
                opts.shadeInvalidRegions, ...
                opts.validShadeColor, ...
                opts.invalidShadeColor, ...
                opts.regionShadeAlpha);
        end
    end

    % --- Valid-segment boundaries ---
    if opts.showSegmentBoundaries && opts.numSegments > 1
        boundaryTimes = time_channel(opts.segmentStarts(2:end));

        for iAx = 1:opts.numPlots
            hold(ax(iAx), "on");

            for k = 1:numel(boundaryTimes)
                xline( ...
                    ax(iAx), ...
                    boundaryTimes(k), ...
                    "--", ...
                    sprintf("Segment %d", k + 1), ...
                    "LabelVerticalAlignment", "bottom", ...
                    "HandleVisibility", "off");
            end

            hold(ax(iAx), "off");
        end
    end

    % --- Link time axes within this figure ---
    if opts.linkTimeAxes && opts.numPlots > 1
        linkaxes(ax, "x");
    end

    xlabel(ax(end), "Time");
end


function ax = plotBaselineFigure( ...
    figName, time_channel, baselineChannels, res, opts, methodLabel, showExcised)
%PLOTBASELINEFIGURE Overlay a baseline on its channel and mark detections.
%
% Shared by Figures 3 and 4 so the two approaches are compared on identical
% axes.  Only the data channels appear - speed and the validity mask have no
% baseline to show.
%
%   res         - struct keyed by channel name, each with fields
%                 b, sigma, anom, raw, excised
%   showExcised - also shade the regions the estimator cut out (Approach B)

    labels = struct( ...
        "alt",  "Altitude", ...
        "CO2",  "CO_2", ...
        "CH4",  "CH_4", ...
        "temp", "Temperature", ...
        "O2",   "O_2");

    nCh = size(baselineChannels, 1);

    fig = figure("Name", figName);

    tl = tiledlayout(fig, nCh, 1, ...
        "TileSpacing", "compact", ...
        "Padding", "compact");

    title(tl, figName + " (" + methodLabel + ")");

    ax = gobjects(nCh, 1);

    for k = 1:nCh
        name = baselineChannels{k, 1};
        r    = res.(name);

        ax(k) = nexttile(tl);
        hold(ax(k), "on");

        % Raw first and pale, so the baseline reads clearly on top of it.
        plot(ax(k), time_channel, r.raw, ...
            "Color", [0.70, 0.70, 0.70], ...
            "LineWidth", 0.5, ...
            "DisplayName", "Raw");

        plot(ax(k), time_channel, r.b, ...
            "Color", [0.85, 0.10, 0.10], ...
            "LineWidth", 1.5, ...
            "DisplayName", "Baseline");

        % Detection band: anything above this is called an anomaly.
        plot(ax(k), time_channel, r.b + opts.anomalyEnterK .* r.sigma, ...
            "--", ...
            "Color", [0.10, 0.35, 0.85], ...
            "LineWidth", 0.75, ...
            "DisplayName", sprintf("Baseline + %g sigma", opts.anomalyEnterK));

        hold(ax(k), "off");

        ylabel(ax(k), labels.(name));
        title(ax(k), labels.(name));
        grid(ax(k), "on");

        if k == 1
            legend(ax(k), "Location", "northeast", "Box", "off");
        end
    end

    % Shade AFTER plotting so the patches inherit settled y-limits.
    for k = 1:nCh
        name = baselineChannels{k, 1};
        r    = res.(name);

        % Grey: segment too short to resolve its own drift, so the baseline
        % here is a pooled record-wide anchor, not a local fit.  Drawn first
        % (widest, most-behind) so anomaly/excision shading reads on top.
        if isfield(r, "degraded")
            shadeMask(ax(k), time_channel, r.degraded, ...
                [0.60, 0.60, 0.60], 0.18);   % grey: degraded (pooled anchor)
        end

        if showExcised
            shadeMask(ax(k), time_channel, r.excised, ...
                [1.00, 0.85, 0.55], 0.35);   % amber: cut from the baseline fit
        end

        shadeMask(ax(k), time_channel, r.anom, ...
            [1.00, 0.45, 0.45], 0.35);       % red: reported anomaly
    end

    if opts.showSegmentBoundaries && opts.numSegments > 1
        boundaryTimes = time_channel(opts.segmentStarts(2:end));

        for k = 1:nCh
            hold(ax(k), "on");

            for j = 1:numel(boundaryTimes)
                xline(ax(k), boundaryTimes(j), "--", ...
                    "HandleVisibility", "off");
            end

            hold(ax(k), "off");
        end
    end

    if opts.linkTimeAxes && nCh > 1
        linkaxes(ax, "x");
    end

    xlabel(ax(end), "Time");
end


function shadeMask(ax, timeValues, mask, faceColor, faceAlpha)
%SHADEMASK Shade every contiguous true run of a mask, behind the data.

    if ~any(mask)
        return;
    end

    originalHoldState = ishold(ax);
    hold(ax, "on");

    yLimits = ylim(ax);

    addMaskPatches(ax, timeValues(:), logical(mask(:)), ...
        yLimits, faceColor, faceAlpha);

    ylim(ax, yLimits);

    if ~originalHoldState
        hold(ax, "off");
    end
end


function shadeTimeRegions( ...
    ax, ...
    timeValues, ...
    validMask, ...
    shadeValid, ...
    shadeInvalid, ...
    validColor, ...
    invalidColor, ...
    faceAlpha)

timeValues = timeValues(:);
validMask  = logical(validMask(:));

originalHoldState = ishold(ax);
hold(ax, "on");

% Keep the existing y-limits fixed while creating patches
yLimits = ylim(ax);

if shadeValid
    addMaskPatches( ...
        ax, ...
        timeValues, ...
        validMask, ...
        yLimits, ...
        validColor, ...
        faceAlpha);
end

if shadeInvalid
    addMaskPatches( ...
        ax, ...
        timeValues, ...
        ~validMask, ...
        yLimits, ...
        invalidColor, ...
        faceAlpha);
end

% Restore the original y-limits
ylim(ax, yLimits);

if ~originalHoldState
    hold(ax, "off");
end
end


function addMaskPatches( ...
    ax, ...
    timeValues, ...
    mask, ...
    yLimits, ...
    faceColor, ...
    faceAlpha)

transitions = diff([false; mask; false]);

regionStarts = find(transitions == 1);
regionEnds   = find(transitions == -1) - 1;

for k = 1:numel(regionStarts)
    iStart = regionStarts(k);
    iEnd   = regionEnds(k);

    xStart = timeValues(iStart);
    xEnd   = timeValues(iEnd);

    % Extend the right edge to the next sample when possible
    if iEnd < numel(timeValues)
        xEnd = timeValues(iEnd + 1);
    end

    hPatch = patch( ...
        ax, ...
        [xStart, xEnd, xEnd, xStart], ...
        [yLimits(1), yLimits(1), yLimits(2), yLimits(2)], ...
        faceColor, ...
        "FaceAlpha", faceAlpha, ...
        "EdgeColor", "none", ...
        "HandleVisibility", "off");

    % Put shading behind plotted data
    uistack(hPatch, "bottom");
end
end

function baseline = robustLowpassBaseline( ...
    time, signal, cutoffPeriodSec, outlierWindowSec, outlierThreshold)
%ROBUSTLOWPASSBASELINE Calculate a spike-resistant low-pass baseline.
%
% The function:
%   1. Removes invalid values.
%   2. Resamples onto a uniformly spaced time grid.
%   3. Identifies extreme deviations from a moving median.
%   4. Replaces those deviations before filtering.
%   5. Applies a low-pass filter.
%   6. Interpolates the baseline back to the original timestamps.
%
% Inputs:
%   time                datetime vector
%   signal              numeric data vector
%   cutoffPeriodSec     shortest baseline period, in seconds
%   outlierWindowSec    moving robust-statistics window, in seconds
%   outlierThreshold    number of robust standard deviations
%
% Output:
%   baseline            baseline at the original timestamps

    time   = time(:);
    signal = signal(:);

    baseline = nan(size(signal));

    valid = ~isnat(time) & isfinite(signal);

    if nnz(valid) < 3
        warning("Not enough valid points to calculate baseline.");
        return;
    end

    validTime   = time(valid);
    validSignal = signal(valid);

    % Sort chronologically.
    [validTime, sortIndex] = sort(validTime);
    validSignal = validSignal(sortIndex);

    % Remove duplicate timestamps.
    [validTime, uniqueIndex] = unique(validTime, "stable");
    validSignal = validSignal(uniqueIndex);

    if numel(validTime) < 3
        warning("Not enough unique timestamps to calculate baseline.");
        return;
    end

    % Estimate the original sampling interval.
    dtSec = median(seconds(diff(validTime)), "omitnan");

    if ~isfinite(dtSec) || dtSec <= 0
        error("Could not determine a valid sampling interval.");
    end

    sampleRateHz = 1 / dtSec;
    cutoffHz     = 1 / cutoffPeriodSec;

    if cutoffHz >= sampleRateHz / 2
        error( ...
            "Baseline cutoff frequency %.6f Hz must be below " + ...
            "the Nyquist frequency %.6f Hz.", ...
            cutoffHz, sampleRateHz / 2);
    end

    % Create a uniform time grid because the smoothing window is defined in
    % samples, so it only maps to a fixed time span on regularly sampled data.
    uniformTime = ( ...
        validTime(1):seconds(dtSec):validTime(end) ...
        ).';

    originalTimeSec = seconds(validTime - validTime(1));
    uniformTimeSec  = seconds(uniformTime - validTime(1));

    uniformSignal = interp1( ...
        originalTimeSec, ...
        validSignal, ...
        uniformTimeSec, ...
        "linear");

    % Force the robust-statistics window to be an odd number of samples.
    windowSamples = max(3, round(outlierWindowSec * sampleRateHz));

    if mod(windowSamples, 2) == 0
        windowSamples = windowSamples + 1;
    end

    windowSamples = min(windowSamples, numel(uniformSignal));

    if mod(windowSamples, 2) == 0
        windowSamples = max(3, windowSamples - 1);
    end

    % Calculate local robust center and spread.
    localMedian = movmedian( ...
        uniformSignal, ...
        windowSamples, ...
        "omitnan");

    absoluteDeviation = abs(uniformSignal - localMedian);

    localMAD = movmedian( ...
        absoluteDeviation, ...
        windowSamples, ...
        "omitnan");

    % Convert median absolute deviation to a robust estimate of sigma.
    robustSigma = 1.4826 * localMAD;

    % Prevent zero-spread regions from causing every small difference
    % to be classified as an outlier.
    minimumSigma = max( ...
        eps, ...
        1e-9 * max(abs(uniformSignal), [], "omitnan"));

    robustSigma = max(robustSigma, minimumSigma);

    outlierMask = absoluteDeviation > ...
        outlierThreshold .* robustSigma;

    % Replace detected anomalies only in the temporary signal used to
    % calculate the baseline. The original measurements remain unchanged.
    cleanedSignal = uniformSignal;
    cleanedSignal(outlierMask) = NaN;

    cleanedSignal = fillmissing( ...
        cleanedSignal, ...
        "linear", ...
        "EndValues", "nearest");

    % Apply the low-pass by Gaussian smoothing.
    %
    % smoothdata is base MATLAB, so this needs no Signal Processing Toolbox
    % (this install is base MATLAB only, so lowpass() does not exist here).
    % It is also better suited to a baseline than lowpass(): at these cutoff
    % ratios (e.g. 1/1000 Hz against a ~1 Hz sample rate) a minimum-order IIR
    % design rings and overshoots at the record edges, which would corrupt the
    % baseline exactly where it is least constrained. A Gaussian window has no
    % ringing and no phase distortion.
    baselineWindowSamples = max(3, round(cutoffPeriodSec * sampleRateHz));
    baselineWindowSamples = min(baselineWindowSamples, numel(cleanedSignal));

    uniformBaseline = smoothdata( ...
        cleanedSignal, ...
        "gaussian", ...
        baselineWindowSamples);

    % Interpolate the baseline back to the original timestamps.  Done directly
    % at every original valid timestamp, which also covers the duplicates that
    % were dropped from the uniform grid.
    baseline(valid) = interp1( ...
        uniformTimeSec, ...
        uniformBaseline, ...
        seconds(time(valid) - validTime(1)), ...
        "linear", ...
        "extrap");
end