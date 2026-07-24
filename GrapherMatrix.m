%% GrapherMatrix.m — 4 baselines x 4 detectors = 16 anomaly-detection strategies
%
% Successor to the original Grapher prototype. That prototype ran exactly TWO
% of these 16 strategies (B1xD1 and B2xD1); this script generalises the
% baseline and the detector into swappable pieces and runs the full matrix so we
% can see WHICH COMBINATIONS FLAG WHICH PARTS OF THE DATA.  There is no ground
% truth here: the comparison is visual + overlap-based, not scored.
%
% Design contract (see docs/anomaly/GRAPHER_STRATEGY_MATRIX.md):
%   baseline(x, dt, opts)          -> b, sigma, degraded   (per contiguous segment)
%   detector(x, b, sigma, dt, opts)-> logical anomaly mask
% Both run WITHIN each valid segment via runWithinSegments(), which also carries
% the degraded-segment fix (short segments anchored to a pooled quiet level).
%
% Per-channel handling: each channel declares a SIGN (+1 plumes up, -1 for
% depletions like O2) and a LOGSPACE flag (CH4 noise is multiplicative -> detect
% in log space).  transformChannel() applies these before any baseline/detector
% sees the data, so every strategy operates on a channel already oriented so
% that "anomaly" means "large positive residual".
%
% Headless run:
%   matlab -batch "addpath('/home/troyboland/epr_imaging/epr_imaging'); ...
%     set(0,'DefaultFigureVisible','off'); run('GrapherMatrix.m')"

%% Settings
filename = "interp_full.csv";

% Preprocessing (identical to Grapher.m so segments match).
speedThreshold    = 0.05;
invalidBuffer     = 30;
shortGapThreshold = 60;
navWindow         = 300;
navSmooth         = "gaussian";

% ----------------------------------------------------------------------------
% Two REGIMES of the same 5x4x4 engine.
%
% FINE  = the original tuning (short transits): a 4000 s baseline window, tight
%         event limits.  It MISSES large/long anomalies by construction - a plume
%         longer than ~half the baseline window is absorbed INTO the baseline (the
%         50% law / its frequency analogue), so the residual over it collapses and
%         the detector sees nothing.  maxEventSec caps B2 at 1200 s outright.
%
% COARSE = same models, retuned for LARGE and LONG events.  The single change that
%         matters most is a MUCH WIDER baseline window (20000 s ~= 5.5 h): a long
%         plume then stays in the residual instead of becoming the baseline.  Event
%         length limits and min-duration are raised to match, sigma is pooled over a
%         wider window (so one giant plume doesn't inflate the local noise floor and
%         hide itself), and the enter threshold is relaxed slightly because a broad
%         swell has lower peak-z than a sharp spike but far more mass.
%
% Both regimes are run in full; their flags are merged and classified by SCALE
% afterwards (SHORT-TRANSIT / LARGE / LONG-SUSTAINED).
% ----------------------------------------------------------------------------

% Shared baseline / detector knobs (identical to Grapher.m's tuned values).
fineOpts = struct();
fineOpts.baselineWindowSec = 4000;
fineOpts.sigmaWindowSec    = 4800;
fineOpts.enterK            = 5;      % D1/D4 hysteresis enter
fineOpts.exitK             = 2;      % D1/D4 hysteresis exit
fineOpts.minDurSec         = 20;     % reject sub-transit events
% B1 (median filter)
fineOpts.clipK  = 2.0;
fineOpts.nIter  = 3;
% B2 (derivative excision)
fineOpts.smoothSec   = 5;
fineOpts.slopeK      = 5;
fineOpts.settleK     = 2;
fineOpts.maxEventSec = 1200;
fineOpts.padSec      = 5;
fineOpts.minAmpK     = 4;
% B3 (robust gaussian low-pass)
fineOpts.cutoffPeriodSec = 4000;    % match the median window's drift timescale
% The outlier pre-pass must strip WHOLE plumes before low-passing, or the
% smoother is dragged into every excursion (a 30 s window can't see a
% multi-minute plume, so the plume looks "normal" to it and survives).
fineOpts.b3OutlierWinSec = 1000;    % was 30 - too short to span plumes (B3 tracked them)
fineOpts.b3OutlierK      = 3.0;
% B4 (robust polynomial detrend)
fineOpts.polyOrder = 2;             % 0=level, 1=line, 2=quadratic drift
fineOpts.polyIter  = 3;             % reweighting passes dropping >clipK residuals
% Degraded-segment fix (shared scaffold)
fineOpts.minResolvableFrac = 1.0;
fineOpts.poolOutlierK      = 3.0;

% COARSE regime: start from the fine tuning, then widen everything scale-related.
coarseOpts = fineOpts;
coarseOpts.baselineWindowSec = 20000;   % ~5.5 h: long plumes stay in the residual
coarseOpts.sigmaWindowSec    = 24000;   % pool noise over an even wider span
coarseOpts.cutoffPeriodSec   = 20000;   % B3 low-pass cutoff tracks the wide window
coarseOpts.b3OutlierWinSec   = 6000;    % strip much longer plumes before low-passing
coarseOpts.enterK            = 3.5;     % broad swells have lower peak-z, more mass
coarseOpts.exitK             = 1.5;
coarseOpts.minDurSec         = 120;     % a "long" event floor: reject < 2 min
coarseOpts.maxEventSec       = 14400;   % allow B2 events up to 4 h
coarseOpts.minAmpK           = 3;       % relax B2's amplitude gate a little
coarseOpts.polyOrder         = 1;       % over a 5 h span a line is the honest drift model

regimes = struct("fine", fineOpts, "coarse", coarseOpts);
regimeNames = fieldnames(regimes);

% Per-channel orientation.  {name, column, sign, logSpace}
%   sign     +1: anomalies are positive excursions (plumes)
%            -1: anomalies are depletions (flip so residual is positive)
%   logSpace true: baseline/detect in log space (multiplicative noise)
% Altitude is NOT a sensor channel for our purposes and is removed here.
channelSpec = {
%   name     column          sign  log
    "CO2",   "CO2",           +1,  false
    "CH4",   "CH4",           +1,  true
    "temp",  "Temperature",   +1,  false
    "O2",    "O2",            -1,  false   % O2 anomalies are depletions
};

% Which channels to run the full matrix on.  Altitude dropped (not a sensor).
runChannels = ["CO2", "CH4", "O2", "temp"];

% Known SENSOR-ERROR windows to blank per channel (set to NaN, excluded from all
% calculations).  {channel, startISO, endISO}.  The CO2 sensor sat at a flat
% floor value (0.671) for ~36 min while the CO2/CH4 sensors were toggled off/on
% (SAGE_WAND note at 05:42) - that stretch is not real data.
sensorErrorWindows = {
    "CO2", "2026-01-16T05:07:00", "2026-01-16T05:44:00"
};

makeFigures = true;   % overlay + agreement + consensus figures per channel

% Optional user event list: a CSV of single timestamps to mark on every
% time-series figure with a vertical dashed line.  Set to "" to disable.  The
% loader is lenient: it uses the first column that parses as datetimes/unix
% seconds (any header, ISO or "yyyy-MM-dd'T'HH:mm:ss" or unix), so an exported
% list drops straight in.
eventMarkerFile = "user_events.csv";

% ----------------------------------------------------------------------------
% RUN CONFIGS - the whole matrix is run once per config below.  Each config sets
% whether speed masking is applied and HOW colossal spikes are kept from dragging
% the baseline up (masking used to do that; the no-mask runs suppress spikes for
% the BASELINE FIT instead, while detection still sees the true raw height):
%   mask     : true  = original speed/gap validity mask
%              false = no masking, whole record is one continuous valid span
%   suppress : "none"    - no spike suppression (only meaningful WITH masking)
%              "log"     - log-scale ALL channels so multiplicative spikes compress
%              "ceiling" - clip the baseline-fit input at a per-channel high
%                          percentile (detection still sees true height)
%              "both"    - ceiling AND log
% ceilingPct is the per-channel percentile the "ceiling"/"both" clip uses.
% Figures + event CSVs for each config land in grapher_matrix_figs/<tag>/.
% ----------------------------------------------------------------------------
ceilingPct = 99;   % robust per-channel ceiling for the clip (baseline-fit only)
runConfigs = struct( ...
    "tag",      {"masked",       "nomask_log",    "nomask_ceiling",  "nomask_both"}, ...
    "mask",     {true,           false,           false,             false}, ...
    "suppress", {"none",         "log",           "ceiling",         "both"});

%% Load once — masking/segments are rebuilt per run config
data = readtable(filename);
time_channel = datetime(data.timestamp_iso, "InputFormat", "yyyy-MM-dd'T'HH:mm:ss");
time_channel = time_channel(:);

% Blank known sensor-error windows to NaN BEFORE anything reads a channel, so
% they are excluded from every baseline, detector, metric and figure.  (alt is
% only used by navSpeed below - a navigation use, not a sensor channel.)
for k = 1:size(sensorErrorWindows, 1)
    colName = char(sensorErrorWindows{k,1});
    t0 = datetime(sensorErrorWindows{k,2}, "InputFormat", "yyyy-MM-dd'T'HH:mm:ss");
    t1 = datetime(sensorErrorWindows{k,3}, "InputFormat", "yyyy-MM-dd'T'HH:mm:ss");
    win = time_channel >= t0 & time_channel <= t1;
    if any(win) && ismember(colName, data.Properties.VariableNames)
        data.(colName)(win) = NaN;
        fprintf("Sensor-error blank: %s set to NaN for %d samples (%s to %s).\n", ...
            colName, nnz(win), string(t0), string(t1));
    end
end

lat = data.lat(:); lon = data.lon(:); alt = data.alt(:);
v = navSpeed(time_channel, lat, lon, alt, "Window", navWindow, "Smooth", navSmooth);
v = v(:);

tSec = seconds(time_channel - time_channel(1));
dt   = median(diff(tSec), "omitnan");
if ~isfinite(dt) || dt <= 0, error("Bad sampling interval."); end

% Load the optional user event timestamps once (shared across all runs).
eventMarkers = loadEventMarkers(eventMarkerFile, fileparts(char(filename)));
if ~isempty(eventMarkers)
    fprintf("Event markers: %d timestamps loaded from %s.\n", numel(eventMarkers), eventMarkerFile);
end

%% The smoothers, baselines, and detectors as name->handle maps
% Smoothing axis: each smoother is applied to the transformed input (per segment,
% so it never bridges a masked gap) BEFORE the baseline sees it.  Detection still
% compares the ORIGINAL (unsmoothed) transformed signal to that baseline, so
% smoothing changes the baseline-fitting input, not the data being judged - the
% honest "does pre-denoising change the anomaly picture?" test.  S0none is the
% control.  smoothSpanSec is each smoother's window in seconds.
smoothSpanSec = 11;   % short display-scale smoothing (~11 samples at 1 Hz)
smoothers = struct( ...
    "S0none",    @(x, dt) x, ...
    "S1gauss",   @(x, dt) smoothWin(x, "gaussian", smoothSpanSec, dt), ...
    "S2median",  @(x, dt) smoothWin(x, "movmedian", smoothSpanSec, dt), ...
    "S3sgolay",  @(x, dt) smoothWin(x, "sgolay",    smoothSpanSec, dt), ...
    "S4hampel",  @(x, dt) hampelSmooth(x, smoothSpanSec, dt, 3.0));

baselines = struct( ...
    "B1_median",     @baselineMedian, ...
    "B2_derivative", @baselineDerivative, ...
    "B3_lowpass",    @baselineLowpass, ...
    "B4_poly",       @baselinePoly);

detectors = struct( ...
    "D1_hysteresis", @detectHysteresis, ...
    "D2_mass",       @detectMass, ...
    "D3_prominence", @detectProminence, ...
    "D4_isoutlier",  @detectIsoutlier);

sNames_ = fieldnames(smoothers);
bNames  = fieldnames(baselines);
dNames  = fieldnames(detectors);
nS = numel(sNames_); nB = numel(bNames); nD = numel(dNames);
fprintf("Matrix: %d smoothers x %d baselines x %d detectors = %d strategies/channel.\n", ...
    nS, nB, nD, nS * nB * nD);

%% Run the whole matrix once PER RUN CONFIG (masked baseline + 3 no-mask runs)
nR = numel(regimeNames);

for runIdx = 1:numel(runConfigs)
  cfg = runConfigs(runIdx);
  fprintf("\n########## RUN '%s'  (mask=%d, suppress=%s) ##########\n", ...
      cfg.tag, cfg.mask, cfg.suppress);

  % --- Validity mask for this run: original speed/gap mask, or all-valid. ---
  if cfg.mask
      validMask = buildValidMask(v, tSec, dt, speedThreshold, invalidBuffer, shortGapThreshold);
  else
      validMask = true(numel(tSec), 1);   % no masking: whole record is valid
  end
  % Sensor dropouts (NaN in the channel) are still excluded per channel below;
  % here segments come only from the validity mask.
  [segStarts, segEnds] = maskToSegments(validMask);
  fprintf("  %d valid segments (%.0f%% of record retained).\n", ...
      numel(segStarts), 100*mean(validMask));

  useLogAll  = ismember(cfg.suppress, ["log", "both"]);   % log-scale ALL channels
  useCeiling = ismember(cfg.suppress, ["ceiling", "both"]);

  % results.(chan).(regime).(strategyName) = struct(mask,b,sigma,degraded,metrics)
  results = struct();

for ci = 1:numel(runChannels)
    chName = runChannels(ci);
    spec = specFor(channelSpec, chName);
    raw  = data.(spec.column)(:);
    xMasked = raw; xMasked(~validMask) = NaN;

  for ri = 1:nR
    regName = regimeNames{ri};
    opts = regimes.(regName);

    % Transform space, PER REGIME and PER RUN CONFIG:
    %  - Base rule (masked run): log for CH4 in the FINE regime only; a large
    %    multiplicative plume is compressed by log and inflates the log-sigma,
    %    so the COARSE regime (large/long events) uses LINEAR where a 10x spike
    %    reads as ~3 sigma.
    %  - A "log"/"both" run forces log for ALL channels in BOTH regimes: this
    %    replaces speed masking as the way to stop colossal spikes dragging the
    %    baseline up (log shrinks their magnitude at the source).
    useLog = useLogAll || (spec.logSpace && strcmp(regName, "fine"));
    [xT, backT] = transformChannel(xMasked, spec.sign, useLog); %#ok<ASGLU>

    % Baseline-fit input vs detection signal are now DISTINCT:
    %  xDetect = xT              (true height - detection must still see the spike)
    %  xBase   = ceiling-clipped xT when useCeiling  (so the fit isn't dragged up)
    % Both live in the SAME space (the baseline b comes out in that space, so
    % x-b stays valid).  The ceiling clips only the FIT input, never detection.
    xDetect = xT;
    if useCeiling
        xBaseFit = clipCeiling(xT, ceilingPct);
    else
        xBaseFit = xT;
    end

    fprintf("\n=== %s [%s regime] (sign=%+d, log=%d, ceiling=%d) ===\n", ...
        chName, regName, spec.sign, useLog, useCeiling);
    fprintf("%-34s | %6s %8s %9s %8s\n", "strategy", "events", "%flag", "mass", "medDur");
    fprintf("%s\n", repmat('-', 1, 74));

    for si = 1:nS
        % Pre-smooth the BASELINE-FIT input, per segment (never bridge a gap).
        xS = applySmootherWithinSegments( ...
            xBaseFit, segStarts, segEnds, dt, smoothers.(sNames_{si}));

        for bi = 1:nB
            % Baseline is fit on the SMOOTHED (and ceiling-clipped) input.
            [b, sigma, degraded] = runWithinSegments( ...
                xS, segStarts, segEnds, dt, baselines.(bNames{bi}), opts);

            for di = 1:nD
                % Detection compares the true-height signal (xDetect) to the
                % baseline: smoothing/ceiling changed the FIT input, not the data
                % judged, so a colossal spike still produces a large residual.
                mask = runDetectorWithinSegments( ...
                    xDetect, b, sigma, segStarts, segEnds, dt, detectors.(dNames{di}), opts);
                mask(~isfinite(xDetect)) = false;

                sName = sNames_{si} + "_" + bNames{bi} + "_" + dNames{di};
                nEvents = countRuns(mask);
                pctFlag = 100 * nnz(mask) / max(1, nnz(isfinite(xDetect)));
                resid   = xDetect - b; resid(~mask) = 0; resid(~isfinite(resid)) = 0;
                anomMass = sum(max(resid, 0)) * dt;
                medDur   = medianEventDurationSec(mask, dt);

                results.(chName).(regName).(sName) = struct( ...
                    "mask", mask, "b", b, "sigma", sigma, "degraded", degraded, ...
                    "nEvents", nEvents, "pctFlag", pctFlag, "anomMass", anomMass, ...
                    "medDurSec", medDur, "smoother", string(sNames_{si}), ...
                    "baseline", string(bNames{bi}), ...
                    "detector", string(dNames{di}));

                fprintf("%-34s | %6d %7.1f%% %9.3g %7.0fs\n", ...
                    sName, nEvents, pctFlag, anomMass, medDur);
            end   % di
        end   % bi
    end   % si
  end   % ri (regime)
end   % ci

%% Merge both regimes' flags and classify anomalies by SCALE
% Consensus = how many of ALL strategies across BOTH regimes flag each sample
% (2 regimes x 80 = 160 votes).  A merged event is any contiguous run flagged by
% at least `consensusMinFrac` of those strategies; each is then classified:
%   SHORT-TRANSIT  - duration < longThreshSec (the coarse regime's minDur)
%   LARGE          - anomaly mass in the top massTopFrac of events (any duration)
%   LONG-SUSTAINED - duration >= longThreshSec (beyond the fine regime's reach)
% LARGE takes precedence over the duration split (a big event is "large" even if
% also long).  Classification uses the COARSE-regime consensus baseline for the
% residual so long-event mass isn't collapsed by a fine baseline riding into it.
consensusMinFrac = 0.25;   % >=25% of the 160 strategies must agree to call a sample
longThreshSec    = 600;    % events this long+ are "long-sustained" (fine regime tops out here)
massTopFrac      = 0.20;   % top 20% of event masses are "large"

events = struct();   % events.(chan) = table-like struct array of classified events
for ci = 1:numel(runChannels)
    chName = runChannels(ci);
    spec = specFor(channelSpec, chName);
    raw  = data.(spec.column)(:); rawMasked = raw; rawMasked(~validMask) = NaN;
    % Classification measures mass/peak against the COARSE B4-poly baseline.  That
    % baseline is LINEAR in the base runs, but a "log"/"both" run forces log for
    % ALL channels in BOTH regimes - so match the coarse space here, or the
    % residual would mix log and linear.
    coarseUsesLog = useLogAll;   % coarse is linear unless the run logs everything
    [xT, ~] = transformChannel(rawMasked, spec.sign, coarseUsesLog);

    events.(chName) = classifyAnomalies( ...
        results.(chName), regimeNames, sNames_, bNames, dNames, ...
        xT, dt, time_channel, consensusMinFrac, longThreshSec, massTopFrac);
end

%% Comparison outputs per channel
% The 16-strategy figures are shown for ONE smoother slice + ONE regime (the FINE
% S0none control) so they stay legible; the smoothing-comparison figure holds
% baseline+detector fixed; the NEW classification figure merges both regimes.
figSmoother    = "S0none";
figRegime      = "fine";
smoothCompCell = "B4_poly_D1_hysteresis";

% Per-run output folder for figures + event CSVs.
runFigDir = fullfile(fileparts(char(filename)), "grapher_matrix_figs", cfg.tag);
if ~exist(runFigDir, "dir"), mkdir(runFigDir); end

if makeFigures
    for ci = 1:numel(runChannels)
        chName = runChannels(ci);
        spec = specFor(channelSpec, chName);
        raw = data.(spec.column)(:); raw(~validMask) = NaN;
        makeBaselineComparisonFigure(chName, time_channel, raw, results.(chName).(figRegime), ...
            bNames, spec, figSmoother, eventMarkers);
        makeComparisonFigures(chName, time_channel, raw, results.(chName).(figRegime), ...
            bNames, dNames, spec, figSmoother, eventMarkers);
        makeSmoothingComparisonFigure(chName, time_channel, raw, results.(chName).(figRegime), ...
            sNames_, spec, smoothCompCell, eventMarkers);
        makeClassificationFigure(chName, time_channel, raw, events.(chName), ...
            results.(chName), regimeNames, sNames_, bNames, dNames, spec, eventMarkers);
    end

    % ---- Combined cross-channel analysis (end-of-report material) ----
    % Pull each channel's masked raw series + its merged-consensus anomaly track.
    chNames = runChannels;
    chRaw = struct(); chConsensus = struct();
    for ci = 1:numel(chNames)
        ch = chNames(ci);
        sp = specFor(channelSpec, ch);
        rr = data.(sp.column)(:); rr(~validMask) = NaN;
        chRaw.(ch) = rr;
        chConsensus.(ch) = consensusTrack(results.(ch), regimeNames, sNames_, bNames, dNames);
    end
    makePairwiseOverlays(chNames, time_channel, chRaw, chConsensus, eventMarkers);
    makeConcurrentAnomalyFigure(chNames, time_channel, chRaw, chConsensus, ...
        events, dt, eventMarkers);

    % Export every open figure for this run into its own folder, then close them
    % so the next run starts clean (interactive display re-runs a single config).
    figs = findobj("Type", "figure");
    for i = 1:numel(figs)
        nm = regexprep(get(figs(i), "Name"), "[^\w]+", "_");
        % 200 DPI (was 110) so the figures stay crisp when placed in the report PDF.
        exportgraphics(figs(i), fullfile(runFigDir, nm + ".png"), "Resolution", 200);
    end
    if numel(runConfigs) > 1, close(figs); end   % keep windows only for a single-config run
    fprintf("  [%s] exported %d figures -> %s\n", cfg.tag, numel(figs), runFigDir);
end

% Save the flagged masks + metrics + classified events for this run (tagged), for
% downstream use (e.g. the app's job-interval importer for the photogrammetry
% cross-check).  Event CSVs also land in the per-run figure folder, tagged.
outMat = fullfile(fileparts(char(filename)), "grapher_matrix_results_" + cfg.tag + ".mat");
save(outMat, "results", "events", "time_channel", "validMask", "segStarts", ...
    "segEnds", "runChannels", "regimeNames", "sNames_", "bNames", "dNames", "-v7");
fprintf("  [%s] saved results -> %s\n", cfg.tag, outMat);

for ci = 1:numel(runChannels)
    chName = runChannels(ci);
    writeEventsCsv(events.(chName), chName, runFigDir);
end

end   % runIdx (run config)


%% ======================================================================
%% Preprocessing helpers (mask + segments) — mirror Grapher.m exactly
%% ======================================================================

function validMask = buildValidMask(v, tSec, dt, speedThreshold, invalidBuffer, shortGapThreshold)
    initialValid = isfinite(v) & (v > speedThreshold);
    initialInvalid = ~initialValid;
    tr = diff([false; initialInvalid; false]);
    iS = find(tr == 1); iE = find(tr == -1) - 1;
    buffered = initialInvalid;
    if invalidBuffer > 0
        for k = 1:numel(iS)
            b0 = tSec(iS(k)) - invalidBuffer;
            b1 = tSec(iE(k)) + invalidBuffer;
            buffered(tSec >= b0 & tSec <= b1) = true;
        end
    end
    validMask = ~buffered;
    inv = ~validMask;
    tr = diff([false; inv; false]);
    iS = find(tr == 1); iE = find(tr == -1) - 1;
    for k = 1:numel(iS)
        if (tSec(iE(k)) - tSec(iS(k)) + dt) < shortGapThreshold
            validMask(iS(k):iE(k)) = true;
        end
    end
    validMask = validMask(:);
end

function [segStarts, segEnds] = maskToSegments(validMask)
    tr = diff([false; validMask(:); false]);
    segStarts = find(tr == 1);
    segEnds   = find(tr == -1) - 1;
end

function spec = specFor(channelSpec, name)
    for k = 1:size(channelSpec, 1)
        if channelSpec{k, 1} == name
            spec = struct("name", channelSpec{k,1}, "column", channelSpec{k,2}, ...
                          "sign", channelSpec{k,3}, "logSpace", channelSpec{k,4});
            return;
        end
    end
    error("No channel spec for '%s'.", name);
end


%% ======================================================================
%% Merge both regimes' flags, extract events, classify by scale
%% ======================================================================

function ev = classifyAnomalies(chRes, regimeNames, sNames_, bNames, dNames, ...
                                xT, dt, tvec, consensusMinFrac, longThreshSec, massTopFrac)
%CLASSIFYANOMALIES Combine every strategy of BOTH regimes into a consensus, cut
% it into events, and label each SHORT-TRANSIT / LARGE / LONG-SUSTAINED.
%
% Returns a struct array (one entry per event) with fields:
%   iStart iEnd tStart tEnd durSec consensus peakZ mass class

    n = numel(xT);
    nStrat = 0;
    consensus = zeros(n, 1);
    % Stack every strategy mask across both regimes into a consensus vote count.
    for ri = 1:numel(regimeNames)
        R = chRes.(regimeNames{ri});
        for si = 1:numel(sNames_)
            for bi = 1:numel(bNames)
                for di = 1:numel(dNames)
                    key = sNames_{si} + "_" + bNames{bi} + "_" + dNames{di};
                    consensus = consensus + double(R.(key).mask);
                    nStrat = nStrat + 1;
                end
            end
        end
    end

    % Consensus baseline for measuring event mass/peak: use the median baseline
    % of the COARSE regime's B4_poly slice (a wide-window fit that does NOT ride
    % up into long plumes), so long-event residuals aren't collapsed.
    coarseKey = "S0none_B4_poly_D1_hysteresis";
    if isfield(chRes, "coarse")
        bRef = chRes.coarse.(coarseKey).b;
    else
        bRef = chRes.(regimeNames{1}).(coarseKey).b;
    end
    resid = xT - bRef;

    % Merged mask: samples agreed on by at least consensusMinFrac of strategies.
    voteThresh = max(1, round(consensusMinFrac * nStrat));
    merged = consensus >= voteThresh;

    [s, e] = runsOf(merged);
    ev = struct("iStart", {}, "iEnd", {}, "tStart", {}, "tEnd", {}, ...
                "durSec", {}, "consensus", {}, "peakZ", {}, "mass", {}, "class", {});
    if isempty(s)
        fprintf("  [classify] no consensus events (>= %d of %d strategies).\n", voteThresh, nStrat);
        return;
    end

    % Robust z of the reference residual, for a peak-z per event.
    Ws = max(3, round(4800 / dt));
    sig = max(1.4826 * movmad(resid, Ws, "omitnan"), eps);
    z = resid ./ sig;

    masses = zeros(numel(s), 1);
    for k = 1:numel(s)
        seg = s(k):e(k);
        masses(k) = sum(max(resid(seg), 0)) * dt;   % positive-residual mass
    end
    largeMassThresh = quantile(masses, 1 - massTopFrac);

    for k = 1:numel(s)
        seg = s(k):e(k);
        durSec = (e(k) - s(k) + 1) * dt;
        peakZ  = max(z(seg), [], "omitnan");
        m      = masses(k);
        % LARGE wins over the duration split; then long vs short by duration.
        if m >= largeMassThresh
            cls = "LARGE";
        elseif durSec >= longThreshSec
            cls = "LONG-SUSTAINED";
        else
            cls = "SHORT-TRANSIT";
        end
        ev(end+1) = struct( ...
            "iStart", s(k), "iEnd", e(k), ...
            "tStart", tvec(s(k)), "tEnd", tvec(e(k)), ...
            "durSec", durSec, ...
            "consensus", max(consensus(seg)), ...
            "peakZ", peakZ, "mass", m, "class", cls); %#ok<AGROW>
    end

    nL = sum(arrayfun(@(x) x.class == "LARGE", ev));
    nG = sum(arrayfun(@(x) x.class == "LONG-SUSTAINED", ev));
    nS_ = sum(arrayfun(@(x) x.class == "SHORT-TRANSIT", ev));
    fprintf("  [classify] %d events (>= %d/%d strategies): %d LARGE, %d LONG-SUSTAINED, %d SHORT-TRANSIT.\n", ...
        numel(ev), voteThresh, nStrat, nL, nG, nS_);
end

function writeEventsCsv(ev, chName, outdir)
%WRITEEVENTSCSV Flat CSV of classified events, import-ready (start/end + class).
    p = fullfile(outdir, "anomaly_events_" + chName + ".csv");
    if isempty(ev)
        writematrix(["start_time","end_time","duration_s","class","consensus","peak_z","mass"], p);
        return;
    end
    T = table( ...
        string({ev.tStart}.'), string({ev.tEnd}.'), ...
        [ev.durSec].', string({ev.class}.'), [ev.consensus].', ...
        [ev.peakZ].', [ev.mass].', ...
        'VariableNames', {'start_time','end_time','duration_s','class','consensus','peak_z','mass'});
    % Use ISO strings for the times so the app's interval importer can read them.
    for i = 1:numel(ev)
        T.start_time(i) = string(datetime(ev(i).tStart, "Format","yyyy-MM-dd'T'HH:mm:ss"));
        T.end_time(i)   = string(datetime(ev(i).tEnd,   "Format","yyyy-MM-dd'T'HH:mm:ss"));
    end
    writetable(T, p);
end

function [xt, back] = transformChannel(x, sgn, logSpace)
%TRANSFORMCHANNEL Orient (and optionally log) a channel so anomalies read as
% large POSITIVE residuals.  `back` maps a transformed value back to raw units
% (for reporting), though the matrix works entirely in transformed space.
    x = x(:);
    if logSpace
        % Shift so the minimum positive value maps safely into log space; keep
        % NaNs as NaN.  Depletion channels (sign -1) are flipped BEFORE log.
        if sgn < 0
            xs = -x;
        else
            xs = x;
        end
        shift = 0;
        mn = min(xs, [], "omitnan");
        if ~(mn > 0)
            shift = 1 - mn;   % make strictly positive
        end
        xt = log(xs + shift);
        back = @(t) sgn * (exp(t) - shift);
    else
        xt = sgn * x;
        back = @(t) sgn * t;
    end
end


%% ======================================================================
%% Shared within-segment scaffold (carries the degraded-segment fix)
%% ======================================================================

function [b, sigma, degraded] = runWithinSegments(x, segStarts, segEnds, dt, baselineFn, opts)
%RUNWITHINSEGMENTS Run a baseline estimator per segment, with the two-tier
% degraded-segment fix: short segments get a flat pooled-quiet anchor rather
% than a self-contaminated local fit.  Generalises Grapher.m's
% baselineWithinSegments to an arbitrary baseline handle.
    x = x(:); n = numel(x);
    b = nan(n,1); sigma = nan(n,1); degraded = false(n,1);

    poolOutlierK = getdef(opts, "poolOutlierK", 3.0);
    gMed = median(x, "omitnan");
    gMad = max(1.4826 * median(abs(x - gMed), "omitnan"), eps);
    quiet = x; quiet(abs(x - gMed) > poolOutlierK * gMad) = NaN;
    pooledLevel = median(quiet, "omitnan");
    pooledSigma = max(1.4826 * median(abs(quiet - pooledLevel), "omitnan"), eps);
    if ~isfinite(pooledLevel), pooledLevel = gMed; end

    minResolvableFrac = getdef(opts, "minResolvableFrac", 1.0);
    minResolvable = max(5, round(minResolvableFrac * opts.baselineWindowSec / dt));

    for k = 1:numel(segStarts)
        idx = segStarts(k):segEnds(k);
        if numel(idx) < 5
            b(idx) = pooledLevel; sigma(idx) = pooledSigma; degraded(idx) = true;
            continue;
        end
        xs = x(idx);
        if any(isnan(xs))
            xs = fillmissing(xs, "linear", "EndValues", "nearest");
        end
        if numel(idx) < minResolvable
            % DEGRADED: too short to resolve DRIFT, but a short segment can still
            % estimate a flat LEVEL.  Anchor to the SEGMENT'S OWN robust quiet
            % median (plumes stripped) - not the record-wide pool, which would be
            % wrong wherever the local background differs from the record mean
            % (that mismatch flags or misses a whole segment).  Fall back to the
            % pool only when the segment is too short to estimate even a level.
            level = segmentQuietLevel(xs, opts.poolOutlierK, minLevelSamples());
            if ~isfinite(level), level = pooledLevel; end
            bk = repmat(level, numel(xs), 1);
            Ws = clampWin(opts.sigmaWindowSec / dt, numel(xs));
            sk = max(localSigma(xs, bk, Ws, opts), eps);
            b(idx) = bk;
            sigma(idx) = sk;
            degraded(idx) = true;
        else
            [bk, sk] = baselineFn(xs, dt, opts);
            b(idx) = bk;
            sigma(idx) = max(sk, eps);
        end
    end
    b(~isfinite(x)) = NaN; sigma(~isfinite(x)) = NaN; degraded(~isfinite(x)) = false;
end

function mask = runDetectorWithinSegments(x, b, sigma, segStarts, segEnds, dt, detectorFn, opts)
%RUNDETECTORWITHINSEGMENTS Apply a detector to (x,b,sigma) per segment so an
% event can never span a masked gap.
    x = x(:); mask = false(numel(x), 1);
    for k = 1:numel(segStarts)
        idx = segStarts(k):segEnds(k);
        if numel(idx) < 3, continue; end
        xs = x(idx);
        if any(isnan(xs))
            xs = fillmissing(xs, "linear", "EndValues", "nearest");
        end
        mask(idx) = detectorFn(xs, b(idx), sigma(idx), dt, opts);
    end
end


%% ======================================================================
%% The 4 baselines:  [b, sigma] = baseline(x, dt, opts)
%% ======================================================================

function [b, sigma] = baselineMedian(x, dt, opts)
%B1 Running-median filter with one-sided iterative clip (Grapher Approach A).
    n = numel(x);
    Wb = clampWin(opts.baselineWindowSec / dt, n);
    Ws = clampWin(opts.sigmaWindowSec    / dt, n);
    b = movmedian(x, Wb, "omitnan");
    for it = 1:opts.nIter
        r = x - b;
        sigma = max(1.4826 * movmad(r, Ws, "omitnan"), eps);
        xClip = min(x, b + opts.clipK .* sigma);
        b = movmedian(xClip, Wb, "omitnan");
    end
    b = smoothdata(b, "gaussian", Wb);
    sigma = localSigma(x, b, Ws, opts);
end

function [b, sigma] = baselineDerivative(x, dt, opts)
%B2 Derivative-excision: find event onsets, cut whole events, re-median the
%   hole-filled signal (Grapher Approach B; excision reused internally).
    n = numel(x);
    Wb = clampWin(opts.baselineWindowSec / dt, n);
    Ws = clampWin(opts.sigmaWindowSec    / dt, n);
    Wd = clampWin(opts.smoothSec         / dt, n);
    xs = smoothdata(x, "gaussian", Wd);
    d  = gradient(xs, dt);
    dSigma = max(1.4826 * movmad(d, Ws, "omitnan"), eps);
    rising = (d ./ dSigma) > opts.slopeK;
    lvlSigma = max(1.4826 * movmad(x - movmedian(x, Wb, "omitnan"), Ws, "omitnan"), eps);
    excised = false(n, 1);
    maxLen = max(1, round(opts.maxEventSec / dt));
    pad    = max(0, round(opts.padSec / dt));
    edges  = find(diff([false; rising]) == 1);
    for k = 1:numel(edges)
        i0 = edges(k);
        if excised(i0), continue; end
        preStart = max(1, i0 - Wd);
        preLevel = median(x(preStart:i0), "omitnan");
        tol = opts.settleK * lvlSigma(i0);
        iEnd = min(n, i0 + maxLen);
        for i = (i0+1):min(n, i0 + maxLen)
            if abs(xs(i) - preLevel) <= tol, iEnd = i; break; end
        end
        peak = max(xs(i0:iEnd), [], "omitnan");
        if (peak - preLevel) < opts.minAmpK * lvlSigma(i0), continue; end
        excised(max(1,i0-pad):min(n,iEnd+pad)) = true;
    end
    xCut = x; xCut(excised) = NaN;
    xCut = fillmissing(xCut, "linear", "EndValues", "nearest");
    b = movmedian(xCut, Wb, "omitnan");
    b = smoothdata(b, "gaussian", Wb);
    sigma = localSigma(x, b, Ws, opts);
end

function [b, sigma] = baselineLowpass(x, dt, opts)
%B3 Robust Gaussian low-pass: strip whole plumes, then smooth at a chosen cutoff
%   period (frequency-domain framing of the baseline).  Promotes Grapher.m's
%   orphaned robustLowpassBaseline to a first-class method.
%
%   The outlier pre-pass is ITERATIVE and ONE-SIDED: a single pass with a wide
%   median is itself biased upward by a big plume (the plume is in the window
%   that computes the "normal" level), so it under-removes.  Re-flagging against
%   the cleaned signal a few times converges on a plume-free support.  One-sided
%   because plumes are positive after transform - clipping symmetrically would
%   also eat the quiet troughs the baseline should honour.
    n = numel(x);
    Wo = clampWin(opts.b3OutlierWinSec / dt, n);
    cleaned = x;
    for it = 1:3
        localMed = movmedian(cleaned, Wo, "omitnan");
        absdev   = abs(cleaned - localMed);
        localMad = movmedian(absdev, Wo, "omitnan");
        rsig = max(1.4826 * localMad, max(eps, 1e-9 * max(abs(x), [], "omitnan")));
        % One-sided: only positive excursions above the local level are plumes.
        outlier = (x - localMed) > opts.b3OutlierK .* rsig;
        cleaned = x; cleaned(outlier) = NaN;
        cleaned = fillmissing(cleaned, "linear", "EndValues", "nearest");
    end
    Wc = clampWin(opts.cutoffPeriodSec / dt, n);
    b = smoothdata(cleaned, "gaussian", Wc);
    Ws = clampWin(opts.sigmaWindowSec / dt, n);
    sigma = localSigma(x, b, Ws, opts);
end

function [b, sigma] = baselinePoly(x, dt, opts)
%B4 Robust per-segment polynomial detrend: iteratively reweighted polyfit on the
%   quiet (non-anomalous) samples only.  Parametric, global-within-segment -
%   cannot chase a plume the way a moving window can.  ('robustfit' is not
%   available here, so the reweighting loop is explicit.)
    n = numel(x);
    t = (0:n-1).' * dt;               % local time in seconds
    good = isfinite(x);
    ord = min(opts.polyOrder, max(0, nnz(good) - 1));
    Ws = clampWin(opts.sigmaWindowSec / dt, n);
    keep = good;
    p = [];
    for it = 1:max(1, opts.polyIter)
        if nnz(keep) <= ord, keep = good; end        % never fewer points than dof
        p = polyfit(t(keep), x(keep), ord);
        fit = polyval(p, t);
        r = x - fit;
        s = max(1.4826 * median(abs(r(good) - median(r(good),"omitnan")), "omitnan"), eps);
        % One-sided: drop only positive excursions (plumes) from the fit set.
        keep = good & (r < opts.clipK * s);
    end
    b = polyval(p, t);
    sigma = localSigma(x, b, Ws, opts);
end


%% ======================================================================
%% The 4 detectors:  mask = detector(x, b, sigma, dt, opts)
%% ======================================================================

function mask = detectHysteresis(x, b, sigma, dt, opts)
%D1 Robust-z hysteresis: enter above enterK, hold until below exitK, reject
%   sub-minDur events (Grapher.m's shared detector).
    z = (x - b) ./ max(sigma, eps);
    mask = hysteresis(z, opts.enterK, opts.exitK);
    mask = rejectShort(mask, max(1, round(opts.minDurSec / dt)));
end

function mask = detectMass(x, b, sigma, dt, opts)
%D2 Anomaly-mass: form candidate runs at the LOOSE exitK level (so low sustained
%   plumes that never spike above enterK are still considered), then keep runs
%   whose INTEGRATED positive residual exceeds a self-referential threshold.
%   Judges a low-sustained plume and a tall-brief spike on the same currency, so
%   it genuinely diverges from D1's height trigger (it can accept an event D1
%   rejects, and reject a thin spike D1 accepts).
    z = (x - b) ./ max(sigma, eps);
    % Candidate runs: everything continuously above exitK (looser than enterK).
    cand = z > opts.exitK;
    r = max(x - b, 0);
    [s, e] = runsOf(cand);
    if isempty(s), mask = false(numel(x), 1); return; end

    % Mass per candidate run, in sigma-seconds (channel-comparable).
    massVals = zeros(numel(s), 1);
    for k = 1:numel(s)
        seg = s(k):e(k);
        massVals(k) = sum(r(seg) ./ max(sigma(seg), eps)) * dt;
    end
    % Self-referential threshold: keep runs whose mass stands out from the bulk
    % of candidate-run masses (robust upper fence), never below a floor equal to
    % a minDur burst at the enter level.  This lets D2 disagree with D1 both ways.
    medM = median(massVals, "omitnan");
    madM = max(1.4826 * median(abs(massVals - medM), "omitnan"), eps);
    floorMass = opts.exitK * opts.minDurSec;             % minDur run at exit height
    massThresh = max(floorMass, medM + opts.enterK * madM);

    mask = false(numel(x), 1);
    keep = massVals >= massThresh;
    for k = find(keep(:)).'
        mask(s(k):e(k)) = true;
    end
    mask = rejectShort(mask, max(1, round(opts.minDurSec / dt)));
end

function mask = detectProminence(x, b, sigma, dt, opts)
%D3 Peak prominence: findpeaks on the residual with MinPeakProminence = enterK
%   sigma, then expand each peak to its exitK-sigma shoulders.  A SHAPE detector
%   - separates merged plumes and rejects slow shoulders that clear a level test.
    r = (x - b) ./ max(sigma, eps);
    r(~isfinite(r)) = 0;
    mask = false(numel(x), 1);
    if numel(r) < 4, return; end
    % findpeaks requires MinPeakDistance strictly < signal length - 1; segments
    % can be shorter than the (coarse-regime) minDur, so clamp it.
    minDist = max(1, min(round(opts.minDurSec / dt), numel(r) - 2));
    [~, locs] = findpeaks(r, ...
        "MinPeakProminence", opts.enterK, ...
        "MinPeakDistance",   minDist);
    for k = 1:numel(locs)
        p = locs(k);
        % Walk out to where the residual drops below exitK on each side.
        i = p; while i > 1 && r(i) > opts.exitK, i = i - 1; end
        j = p; while j < numel(r) && r(j) > opts.exitK, j = j + 1; end
        mask(i:j) = true;
    end
    mask = rejectShort(mask, minDist);
end

function mask = detectIsoutlier(x, b, sigma, dt, opts)
%D4 isoutlier moving-window on the residual: flags samples that are outliers vs
%   their OWN local moving distribution (movmedian MAD), independent of the
%   fitted sigma - an independent check on the sigma model.  Stands on its own
%   flags: an outlier run is grown to its contiguous exitK-level shoulders so
%   events are comparable to D1-D3, but the TRIGGER is isoutlier, not the fitted
%   z, so D4 fires where the local spread and the pooled sigma disagree.
    r = x - b;
    W = clampWin(opts.sigmaWindowSec / dt, numel(r));
    out = isoutlier(r, "movmedian", W) & (r > 0);   % positive-side outliers only
    if ~any(out), mask = false(numel(x), 1); return; end

    % Grow each outlier to the surrounding run where the residual stays above
    % exitK*sigma (the settle level), so a flagged spike carries its shoulders.
    aboveExit = (r ./ max(sigma, eps)) > opts.exitK;
    mask = false(numel(x), 1);
    [s, e] = runsOf(aboveExit);
    for k = 1:numel(s)
        if any(out(s(k):e(k)))          % this above-exit run contains an outlier
            mask(s(k):e(k)) = true;
        end
    end
    mask = rejectShort(mask, max(1, round(opts.minDurSec / dt)));
end


%% ======================================================================
%% Shared detector/baseline primitives
%% ======================================================================

function sigma = localSigma(x, b, Ws, opts)
%LOCALSIGMA Time-varying noise scale from the anomaly-removed residual.
    r = x - b;
    s0 = max(1.4826 * movmad(r, Ws, "omitnan"), eps);
    quiet = r; quiet(r ./ s0 > opts.enterK) = NaN;
    sigma = 1.4826 * movmad(quiet, Ws, "omitnan");
    sigma = max(fillmissing(sigma, "linear", "EndValues", "nearest"), eps);
end

function m = hysteresis(z, enterK, exitK)
    z = z(:); m = false(size(z)); inEvent = false;
    for i = 1:numel(z)
        if ~isfinite(z(i)), inEvent = false;
        elseif ~inEvent && z(i) > enterK, inEvent = true;
        elseif inEvent && z(i) < exitK, inEvent = false;
        end
        m(i) = inEvent;
    end
end

function m = rejectShort(m, minSamples)
    [s, e] = runsOf(m);
    for k = 1:numel(s)
        if (e(k) - s(k) + 1) < minSamples, m(s(k):e(k)) = false; end
    end
end

function [s, e] = runsOf(mask)
    d = diff([false; mask(:); false]);
    s = find(d == 1); e = find(d == -1) - 1;
end

function n = countRuns(mask)
    n = nnz(diff([false; mask(:); false]) == 1);
end

function durSec = medianEventDurationSec(mask, dt)
    [s, e] = runsOf(mask);
    if isempty(s), durSec = 0; return; end
    durSec = median((e - s + 1)) * dt;
end

function w = clampWin(wSamples, n)
    w = max(3, round(wSamples));
    if mod(w, 2) == 0, w = w + 1; end
    w = min(w, n);
    if mod(w, 2) == 0, w = max(3, w - 1); end
end

function y = clipCeiling(x, pct)
%CLIPCEILING Cap x at its own high percentile so colossal spikes cannot drag a
% baseline fit upward.  Applied to the BASELINE-FIT input only (detection sees
% the true height).  One-sided: only the top is clipped (anomalies are positive
% after transform); the lower tail is left alone so real troughs are honoured.
    y = x;
    hi = prctile(x(isfinite(x)), pct);
    if isfinite(hi)
        y(x > hi) = hi;
    end
end


%% ======================================================================
%% User event markers (vertical dashed lines on every time-series figure)
%% ======================================================================

function evTimes = loadEventMarkers(evFile, baseDir)
%LOADEVENTMARKERS Read single event timestamps from a CSV.  Lenient: uses the
% first column whose values parse as datetimes (ISO / common formats) or unix
% seconds.  Returns a datetime column vector (empty if the file is absent/empty
% or nothing parses).
    evTimes = datetime.empty(0, 1);
    if isempty(char(evFile)), return; end
    p = string(evFile);
    if ~isfile(p), p = fullfile(baseDir, char(evFile)); end
    if ~isfile(p)
        fprintf("Event markers: file not found (%s) - no markers drawn.\n", evFile);
        return;
    end
    try
        T = readtable(p, "TextType", "string");
    catch
        % Fall back to a headerless read (one column of bare timestamps).
        try
            T = readtable(p, "ReadVariableNames", false, "TextType", "string");
        catch
            fprintf("Event markers: could not read %s.\n", p);
            return;
        end
    end
    if isempty(T), return; end

    fmts = ["yyyy-MM-dd'T'HH:mm:ss.SSS", "yyyy-MM-dd'T'HH:mm:ss", ...
            "yyyy-MM-dd HH:mm:ss", "MM/dd/uuuu HH:mm:ss", "MM/dd/uu HH:mm:ss"];
    for c = 1:width(T)
        col = T{:, c};
        dtv = parseDatetimeColumn(col, fmts);
        if nnz(~isnat(dtv)) >= max(1, 0.5 * numel(dtv))
            evTimes = dtv(~isnat(dtv));
            evTimes = evTimes(:);
            return;
        end
    end
    fprintf("Event markers: no column in %s parsed as timestamps.\n", p);
end

function dtv = parseDatetimeColumn(col, fmts)
%PARSEDATETIMECOLUMN Try to turn one table column into datetimes.
    n = numel(col);
    dtv = NaT(n, 1);
    if isdatetime(col)
        dtv = col(:); return;
    end
    s = string(col);
    % Strip a trailing UTC "Z" marker and treat the time as NAIVE - this project
    % treats every source as the same timezone with no offset (see the interp
    % CSVs, which have no Z), so we must NOT let datetime apply a UTC->local shift.
    s = regexprep(s, "[Zz]$", "");
    % Unix seconds?
    num = str2double(s);
    isNum = ~isnan(num);
    if any(isNum) && all(num(isNum) > 1e8 & num(isNum) < 4e9)
        dtv(isNum) = datetime(num(isNum), "ConvertFrom", "posixtime");
        return;
    end
    % Datetime strings: try each format, keep the first that parses most rows.
    best = NaT(n, 1); bestHits = 0;
    for f = 1:numel(fmts)
        try
            cand = datetime(s, "InputFormat", fmts(f));
        catch
            continue;
        end
        hits = nnz(~isnat(cand));
        if hits > bestHits, best = cand; bestHits = hits; end
    end
    if bestHits == 0
        try, best = datetime(s); catch, end   % last resort: auto-detect
    end
    dtv = best;
end

function overlayEventMarkers(axList, evTimes)
%OVERLAYEVENTMARKERS Draw a thin vertical dashed line at each event time on
% every axis in axList.  No-op when there are no events.  Markers are excluded
% from legends and pushed behind the data.
    if isempty(evTimes), return; end
    for a = 1:numel(axList)
        ax = axList(a);
        if ~isgraphics(ax), continue; end
        for k = 1:numel(evTimes)
            xl = xline(ax, evTimes(k), "--", ...
                "Color", [0.20 0.20 0.20], "Alpha", 0.55, "LineWidth", 0.75, ...
                "HandleVisibility", "off");
            try, uistack(xl, "bottom"); catch, end
        end
    end
end


%% ======================================================================
%% Input smoothers (the smoothing axis).  Each maps x -> smoothed x.
%% Applied per segment via applySmootherWithinSegments so no gap is bridged.
%% ======================================================================

function y = applySmootherWithinSegments(x, segStarts, segEnds, dt, smootherFn)
%APPLYSMOOTHERWITHINSEGMENTS Smooth each valid segment independently.
% Values outside segments stay NaN (mask preserved); a segment too short to
% smooth passes through unchanged.
    x = x(:); y = x;
    for k = 1:numel(segStarts)
        idx = segStarts(k):segEnds(k);
        if numel(idx) < 3, continue; end
        xs = x(idx);
        if any(isnan(xs))
            xs = fillmissing(xs, "linear", "EndValues", "nearest");
        end
        y(idx) = smootherFn(xs, dt);
    end
end

function y = smoothWin(x, method, spanSec, dt)
%SMOOTHWIN smoothdata wrapper with a seconds->samples window.
    w = clampWin(spanSec / dt, numel(x));
    y = smoothdata(x, method, w);
end

function y = hampelSmooth(x, spanSec, dt, k)
%HAMPELSMOOTH Robust spike removal: replace points > k local MADs from the local
% median with that median, leaving the rest untouched.  Unlike Gaussian/median
% smoothing it does NOT smear clean data - it only edits outliers - so it is the
% honest "remove sensor glitches without blurring plume edges" pre-pass.
    x = x(:); n = numel(x);
    w = clampWin(spanSec / dt, n);
    med = movmedian(x, w, "omitnan");
    localMad = movmedian(abs(x - med), w, "omitnan");
    sigma = 1.4826 * localMad;
    out = abs(x - med) > k * max(sigma, eps);
    y = x; y(out) = med(out);
end

function v = getdef(s, name, default)
    if isfield(s, name) && ~isempty(s.(name)), v = s.(name); else, v = default; end
end

function n = minLevelSamples()
%MINLEVELSAMPLES Fewest samples needed to trust a segment's own quiet level.
% Below this a short segment can't reliably strip its own plumes, so the
% record-wide pooled anchor is the honest fallback.
    n = 30;
end

function level = segmentQuietLevel(xs, outlierK, minSamples)
%SEGMENTQUIETLEVEL Robust quiet median of one segment (plumes stripped).
% Returns NaN when the segment is too short to estimate a level, so the caller
% falls back to the record-wide pool.
    xs = xs(isfinite(xs));
    if numel(xs) < minSamples
        level = NaN; return;
    end
    m = median(xs, "omitnan");
    s = max(1.4826 * median(abs(xs - m), "omitnan"), eps);
    quiet = xs; quiet(abs(xs - m) > outlierK * s) = NaN;
    level = median(quiet, "omitnan");
    if ~isfinite(level), level = m; end
end


%% ======================================================================
%% Comparison figures: overlay strip, 16x16 Jaccard, consensus track
%% ======================================================================

function makeClassificationFigure(chName, t, raw, ev, chRes, regimeNames, ...
                                  sNames_, bNames, dNames, spec, evTimes)
%MAKECLASSIFICATIONFIGURE The payoff figure: both regimes merged, events coloured
% by scale class, with a consensus track and per-class counts.  Answers "where
% are the large and long anomalies, and how are they classified?" directly.

    % Consensus across BOTH regimes (same count classifyAnomalies used).
    n = numel(t); consensus = zeros(n, 1); nStrat = 0;
    for ri = 1:numel(regimeNames)
        R = chRes.(regimeNames{ri});
        for si = 1:numel(sNames_)
            for bi = 1:numel(bNames)
                for di = 1:numel(dNames)
                    key = sNames_{si} + "_" + bNames{bi} + "_" + dNames{di};
                    consensus = consensus + double(R.(key).mask);
                    nStrat = nStrat + 1;
                end
            end
        end
    end

    classColors = struct( ...
        "SHORT_TRANSIT",   [0.35 0.55 0.85], ...   % blue
        "LARGE",           [0.85 0.20 0.20], ...   % red
        "LONG_SUSTAINED",  [0.95 0.60 0.15]);      % orange

    fig = figure("Name", sprintf("%s - anomaly classification", chName));
    tl = tiledlayout(fig, 3, 1, "TileSpacing", "compact", "Padding", "compact");
    nL = sum(arrayfun(@(x) x.class=="LARGE", ev));
    nG = sum(arrayfun(@(x) x.class=="LONG-SUSTAINED", ev));
    nSh = sum(arrayfun(@(x) x.class=="SHORT-TRANSIT", ev));
    title(tl, sprintf("%s - anomalies classified by scale (both regimes merged): %d LARGE, %d LONG-SUSTAINED, %d SHORT-TRANSIT", ...
        chName, nL, nG, nSh), "Interpreter", "none");

    % --- Top: channel with class-coloured event spans ---
    axTop = nexttile(tl, [2 1]);
    plot(axTop, t, raw, "Color", [0.25 0.25 0.25], "LineWidth", 0.5);
    hold(axTop, "on");
    yl = ylim(axTop);
    seen = struct("SHORT_TRANSIT", false, "LARGE", false, "LONG_SUSTAINED", false);
    for k = 1:numel(ev)
        key = strrep(ev(k).class, "-", "_");
        col = classColors.(key);
        i0 = ev(k).iStart; i1 = min(ev(k).iEnd + 1, numel(t));
        showLegend = ~seen.(key);
        p = patch(axTop, [t(i0) t(i1) t(i1) t(i0)], [yl(1) yl(1) yl(2) yl(2)], ...
            col, "FaceAlpha", 0.30, "EdgeColor", "none");
        if showLegend
            set(p, "DisplayName", ev(k).class); seen.(key) = true;
        else
            set(p, "HandleVisibility", "off");
        end
        uistack(p, "bottom");
    end
    ylim(axTop, yl);
    hold(axTop, "off");
    ylabel(axTop, chName); grid(axTop, "on");
    title(axTop, "Channel with classified events (large/long are what the coarse regime added)");
    if numel(ev) > 0, legend(axTop, "Location", "northeast", "Box", "off", "Interpreter","none"); end

    % --- Bottom: consensus track across both regimes ---
    axC = nexttile(tl);
    area(axC, t, consensus, "FaceColor", [0.55 0.55 0.55], "EdgeColor", "none");
    ylim(axC, [0, nStrat]); ylabel(axC, "votes");
    title(axC, sprintf("Consensus: strategies (of %d, both regimes) flagging each sample", nStrat));
    grid(axC, "on"); xlabel(axC, "Time");
    overlayEventMarkers([axTop, axC], evTimes);
    linkaxes([axTop, axC], "x");
end


function makeComparisonFigures(chName, t, raw, chResults, bNames, dNames, spec, smPrefix, evTimes)
    % 16-strategy figures for ONE smoother slice (smPrefix), keyed
    % <smoother>_<baseline>_<detector>.
    sNames = strings(0); labels = strings(0);
    for bi = 1:numel(bNames)
        for di = 1:numel(dNames)
            sNames(end+1) = smPrefix + "_" + bNames{bi} + "_" + dNames{di}; %#ok<AGROW>
            labels(end+1) = bNames{bi} + "_" + dNames{di};                  %#ok<AGROW>
        end
    end
    nS = numel(sNames);
    M = false(numel(t), nS);
    for k = 1:nS
        M(:, k) = chResults.(sNames(k)).mask;
    end

    % --- Figure: channel on top, 16-row flag raster below (shared x) ---
    fig = figure("Name", sprintf("%s - 16 strategies (%s)", chName, smPrefix));
    tl = tiledlayout(fig, 5, 1, "TileSpacing", "compact", "Padding", "compact");
    title(tl, sprintf("%s (sign %+d, log %d) - which strategies flag what  [smoother: %s]", ...
        chName, spec.sign, spec.logSpace, smPrefix), "Interpreter", "none");

    axTop = nexttile(tl, [2 1]);
    plot(axTop, t, raw, "Color", [0.2 0.2 0.2], "LineWidth", 0.5);
    ylabel(axTop, chName); grid(axTop, "on");
    title(axTop, "Channel (raw, masked)");

    axRas = nexttile(tl, [3 1]);
    % Raster: strategy index on Y, time on X, filled where flagged.
    hold(axRas, "on");
    for k = 1:nS
        yy = nS - k + 1;
        [s, e] = localRuns(M(:, k));
        for r = 1:numel(s)
            patch(axRas, [t(s(r)) t(e(r)) t(e(r)) t(s(r))], ...
                  [yy-0.45 yy-0.45 yy+0.45 yy+0.45], [0.85 0.30 0.30], ...
                  "EdgeColor", "none");
        end
    end
    hold(axRas, "off");
    ylim(axRas, [0.5, nS+0.5]);
    yticks(axRas, 1:nS);
    yticklabels(axRas, flipud(labels(:)));
    set(axRas, "TickLabelInterpreter", "none");
    xlabel(axRas, "Time"); title(axRas, "Flagged events by strategy");
    grid(axRas, "on");
    overlayEventMarkers([axTop, axRas], evTimes);
    linkaxes([axTop, axRas], "x");

    % --- Figure: 16x16 Jaccard agreement heatmap ---
    J = ones(nS, nS);
    for a = 1:nS
        for b = a+1:nS
            inter = nnz(M(:,a) & M(:,b));
            uni   = nnz(M(:,a) | M(:,b));
            jac = inter / max(1, uni);
            J(a,b) = jac; J(b,a) = jac;
        end
    end
    figJ = figure("Name", sprintf("%s - strategy agreement Jaccard (%s)", chName, smPrefix));
    axJ = axes(figJ);
    imagesc(axJ, J); axis(axJ, "square"); colorbar(axJ); clim(axJ, [0 1]);
    colormap(axJ, parula);
    xticks(axJ, 1:nS); yticks(axJ, 1:nS);
    xticklabels(axJ, labels); yticklabels(axJ, labels);
    set(axJ, "TickLabelInterpreter", "none");
    xtickangle(axJ, 90);
    title(axJ, sprintf("%s - flagged-sample Jaccard overlap (1 = identical)", chName), ...
        "Interpreter", "none");

    % --- Figure: consensus track (0..16 strategies flagging each sample) ---
    consensus = sum(M, 2);
    figC = figure("Name", sprintf("%s - consensus (%s)", chName, smPrefix));
    tlc = tiledlayout(figC, 2, 1, "TileSpacing", "compact", "Padding", "compact");
    axc1 = nexttile(tlc);
    plot(axc1, t, raw, "Color", [0.2 0.2 0.2], "LineWidth", 0.5);
    ylabel(axc1, chName); grid(axc1, "on"); title(axc1, "Channel");
    axc2 = nexttile(tlc);
    area(axc2, t, consensus, "FaceColor", [0.85 0.35 0.35], "EdgeColor", "none");
    ylim(axc2, [0, nS]); ylabel(axc2, "strategies");
    title(axc2, sprintf("Consensus: how many of %d strategies flag each sample", nS));
    grid(axc2, "on"); xlabel(axc2, "Time");
    overlayEventMarkers([axc1, axc2], evTimes);
    linkaxes([axc1, axc2], "x");
end

function [s, e] = localRuns(mask)
    d = diff([false; mask(:); false]);
    s = find(d == 1); e = find(d == -1) - 1;
end


%% ======================================================================
%% Combined cross-channel analysis (end-of-report figures)
%% ======================================================================

function cons = consensusTrack(chRes, regimeNames, sNames_, bNames, dNames)
%CONSENSUSTRACK Votes per sample across ALL strategies of BOTH regimes (0..160).
    n = numel(chRes.(regimeNames{1}).(sNames_{1} + "_" + bNames{1} + "_" + dNames{1}).mask);
    cons = zeros(n, 1);
    for ri = 1:numel(regimeNames)
        R = chRes.(regimeNames{ri});
        for si = 1:numel(sNames_)
            for bi = 1:numel(bNames)
                for di = 1:numel(dNames)
                    cons = cons + double(R.(sNames_{si} + "_" + bNames{bi} + "_" + dNames{di}).mask);
                end
            end
        end
    end
end

function z = znorm(x)
%ZNORM Robust 0..1-ish normalisation for overlaying channels of different units:
% center on the median, scale by the MAD, then squash outliers for display.
    x = x(:);
    m = median(x, "omitnan");
    s = max(1.4826 * median(abs(x - m), "omitnan"), eps);
    z = (x - m) ./ s;
end

function makePairwiseOverlays(chNames, t, chRaw, chConsensus, evTimes) %#ok<INUSD>
%MAKEPAIRWISEOVERLAYS One figure per channel PAIR: the two channels overlaid on
% a shared robust-normalised axis (so different units are comparable), with each
% channel's consensus-anomaly track shaded, so co-located anomalies are obvious.
    pairs = nchoosek(1:numel(chNames), 2);
    cols = [0.85 0.20 0.20; 0.20 0.40 0.85];   % channel A red, channel B blue
    for p = 1:size(pairs,1)
        a = chNames(pairs(p,1)); b = chNames(pairs(p,2));
        za = znorm(chRaw.(a)); zb = znorm(chRaw.(b));

        fig = figure("Name", sprintf("PAIR %s vs %s", a, b));
        tl = tiledlayout(fig, 3, 1, "TileSpacing","compact","Padding","compact");
        title(tl, sprintf("%s vs %s - overlay (robust-normalised) + anomaly co-occurrence", a, b), ...
            "Interpreter","none");

        axO = nexttile(tl, [2 1]); hold(axO,"on");
        plot(axO, t, za, "Color", cols(1,:), "LineWidth", 0.6, "DisplayName", a);
        plot(axO, t, zb, "Color", cols(2,:), "LineWidth", 0.6, "DisplayName", b);
        hold(axO,"off"); grid(axO,"on");
        ylabel(axO, "robust z (median/MAD)");
        legend(axO, "Location","northeast", "Box","off", "Interpreter","none");
        title(axO, "Both channels overlaid");
        % clip y for readability (plumes run off-scale)
        ylim(axO, [-4 10]);
        overlayEventMarkers(axO, evTimes);

        % Bottom: normalised consensus of each channel (0..1), overlaid, so where
        % BOTH are high the anomalies co-occur.
        axC = nexttile(tl);
        ca = chConsensus.(a) / max(1, max(chConsensus.(a)));
        cb = chConsensus.(b) / max(1, max(chConsensus.(b)));
        hold(axC,"on");
        area(axC, t, ca, "FaceColor", cols(1,:), "FaceAlpha", 0.45, "EdgeColor","none", "DisplayName", a+" anomalies");
        area(axC, t, cb, "FaceColor", cols(2,:), "FaceAlpha", 0.45, "EdgeColor","none", "DisplayName", b+" anomalies");
        hold(axC,"off"); grid(axC,"on"); ylim(axC,[0 1]);
        ylabel(axC, "consensus (norm)"); xlabel(axC, "Time");
        legend(axC, "Location","northeast", "Box","off", "Interpreter","none");
        title(axC, "Anomaly consensus - overlap = co-occurring anomalies");
        overlayEventMarkers(axC, evTimes);
        linkaxes([axO, axC], "x");
    end
end

function makeConcurrentAnomalyFigure(chNames, t, chRaw, chConsensus, events, dt, evTimes)
%MAKECONCURRENTANOMALYFIGURE Find and highlight the times where the MOST channels
% are anomalous at once (cross-channel co-occurrence).  A channel is "anomalous"
% at a sample when >=25% of its 160 strategies flag it; the concurrency count is
% how many channels are anomalous simultaneously.  Peaks in that count are the
% multi-channel events most worth inspecting for questions 3-5.
    nCh = numel(chNames);
    n = numel(t);
    perChanFlag = false(n, nCh);
    for ci = 1:nCh
        c = chConsensus.(chNames(ci));
        perChanFlag(:, ci) = c >= 0.25 * max(1, max(c));   % channel anomalous here
    end
    concurrency = sum(perChanFlag, 2);   % 0..nCh channels anomalous per sample

    % --- Figure 1: the concurrency track + each channel's flag raster ---
    fig = figure("Name", "CONCURRENT anomalies - cross-channel");
    tl = tiledlayout(fig, nCh + 2, 1, "TileSpacing","compact","Padding","compact");
    title(tl, "Cross-channel concurrent anomalies (where the most channels flag together)", ...
        "Interpreter","none");
    % top: concurrency count
    axTop = nexttile(tl, [2 1]);
    area(axTop, t, concurrency, "FaceColor",[0.55 0.20 0.55], "EdgeColor","none");
    ylim(axTop, [0 nCh]); ylabel(axTop, "channels"); grid(axTop,"on");
    title(axTop, sprintf("How many of the %d channels are anomalous at once", nCh));
    overlayEventMarkers(axTop, evTimes);
    % per-channel flag rows
    axList = axTop;
    for ci = 1:nCh
        ax = nexttile(tl);
        [s,e] = localRuns(perChanFlag(:,ci));
        hold(ax,"on");
        for r = 1:numel(s)
            patch(ax, [t(s(r)) t(e(r)) t(e(r)) t(s(r))], [0 0 1 1], [0.80 0.25 0.25], "EdgeColor","none");
        end
        hold(ax,"off"); ylim(ax,[0 1]); set(ax,"YTick",[]);
        ylabel(ax, chNames(ci), "Rotation",0, "HorizontalAlignment","right", "Interpreter","none");
        grid(ax,"on");
        if ci < nCh, set(ax,"XTickLabel",[]); end
        overlayEventMarkers(ax, evTimes);
        axList(end+1) = ax; %#ok<AGROW>
    end
    xlabel(axList(end), "Time");
    linkaxes(axList, "x");

    % --- Figure 2: top concurrent windows listed + zoomed context ---
    % Group contiguous samples where >=2 channels concur, rank by peak concurrency.
    concur2 = concurrency >= 2;
    [s,e] = localRuns(concur2);
    if ~isempty(s)
        pk = zeros(numel(s),1);
        for k = 1:numel(s), pk(k) = max(concurrency(s(k):e(k))); end
        dur = (e - s + 1) * dt;
        score = pk + dur/max(dur);          % rank by peak, tie-break on duration
        [~, order] = sort(score, "descend");
        topN = min(6, numel(order));

        fig2 = figure("Name", "CONCURRENT anomalies - top windows");
        tl2 = tiledlayout(fig2, topN, 1, "TileSpacing","compact","Padding","compact");
        title(tl2, sprintf("Top %d multi-channel concurrent-anomaly windows (all channels overlaid, robust-normalised)", topN), ...
            "Interpreter","none");
        pad = round(120/dt);   % +/-2 min context
        cols = lines(numel(chNames));
        for j = 1:topN
            k = order(j);
            i0 = max(1, s(k)-pad); i1 = min(numel(t), e(k)+pad);
            ax = nexttile(tl2); hold(ax,"on");
            for ci = 1:numel(chNames)
                z = znorm(chRaw.(chNames(ci)));
                plot(ax, t(i0:i1), z(i0:i1), "Color", cols(ci,:), "LineWidth", 0.8, ...
                     "DisplayName", chNames(ci));
            end
            % shade the concurrent window
            yl = [-4 10]; ylim(ax, yl);
            patch(ax, [t(s(k)) t(e(k)) t(e(k)) t(s(k))], [yl(1) yl(1) yl(2) yl(2)], ...
                  [0.55 0.20 0.55], "FaceAlpha",0.12, "EdgeColor","none", "HandleVisibility","off");
            hold(ax,"off"); grid(ax,"on");
            title(ax, sprintf("%s  -  peak %d/%d channels, %.0f s", ...
                string(t(s(k))), pk(k), numel(chNames), dur(k)), "Interpreter","none");
            if j == 1, legend(ax, "Location","northeastoutside", "Box","off", "Interpreter","none"); end
            overlayEventMarkers(ax, evTimes);
        end
    end
end


function makeBaselineComparisonFigure(chName, t, raw, chResults, bNames, spec, smPrefix, evTimes)
%MAKEBASELINECOMPARISONFIGURE Overlay all 4 baseline MODELS on one channel.
%
% The baseline is a per-timestamp series (the "model" of the drifting background)
% - so this figure answers "how do the four baseline techniques differ as models
% of the background?" directly, on shared axes.  The stored baselines live in
% TRANSFORMED space (sign-flipped / log for some channels); each is inverted
% back to raw units here so it overlays the raw channel meaningfully.  Baselines
% are taken from the smPrefix smoother slice.

    % One representative baseline per family (all 4 detectors share the baseline).
    nB = numel(bNames);
    B = nan(numel(t), nB);
    degraded = false(numel(t), 1);
    for bi = 1:nB
        rep = chResults.(smPrefix + "_" + bNames{bi} + "_D1_hysteresis");
        B(:, bi) = invTransform(rep.b, raw, spec.sign, spec.logSpace);
        degraded = degraded | rep.degraded;   % same across detectors
    end

    colors = [
        0.85 0.10 0.10    % B1 median   - red
        0.10 0.45 0.85    % B2 deriv    - blue
        0.10 0.65 0.30    % B3 lowpass  - green
        0.60 0.20 0.75    % B4 poly     - purple
    ];
    prettyB = ["B1 median filter", "B2 derivative excision", ...
               "B3 Gaussian low-pass", "B4 polynomial detrend"];

    fig = figure("Name", sprintf("%s - baseline comparison", chName));
    tl = tiledlayout(fig, 3, 1, "TileSpacing", "compact", "Padding", "compact");
    title(tl, sprintf("%s - four baseline models compared (sign %+d, log %d)", ...
        chName, spec.sign, spec.logSpace), "Interpreter", "none");

    % --- Top (2 tiles): raw channel + all four baselines overlaid ---
    axTop = nexttile(tl, [2 1]);
    hold(axTop, "on");
    plot(axTop, t, raw, "Color", [0.65 0.65 0.65], "LineWidth", 0.5, ...
        "DisplayName", "Raw");
    for bi = 1:nB
        plot(axTop, t, B(:, bi), "Color", colors(bi, :), "LineWidth", 1.4, ...
            "DisplayName", prettyB(bi));
    end
    ylabel(axTop, chName);
    legend(axTop, "Location", "northeast", "Box", "off", "Interpreter", "none");

    % Auto-scale the y-axis to the BASELINE range (plus a little headroom), not
    % the raw range: on channels like CO2/CH4 the plumes are ~50x the baseline,
    % so a raw-scaled axis flattens all four baselines into one line.  This plot
    % is about the baselines, so let the plumes run off-scale.  Set the limit
    % BEFORE shading so the grey degraded patches span the visible band.
    bLo = min(B(:), [], "omitnan");
    bHi = max(B(:), [], "omitnan");
    if isfinite(bLo) && isfinite(bHi) && bHi > bLo
        pad = 0.15 * (bHi - bLo);
        ylim(axTop, [bLo - pad, bHi + pad]);
        scaleNote = "  [y scaled to baselines; plumes run off-scale]";
    else
        scaleNote = "";
    end
    % Grey-shade the degraded stretches (flat pooled/own-quiet anchor, not a fit).
    shadeMaskLocal(axTop, t, degraded, [0.6 0.6 0.6], 0.12);
    hold(axTop, "off");
    title(axTop, "Channel + baselines (grey = degraded: flat anchor, not a drift fit)" + scaleNote);
    grid(axTop, "on");

    % --- Bottom: spread BETWEEN the baselines (max - min across the four) ---
    axSpread = nexttile(tl);
    spread = max(B, [], 2) - min(B, [], 2);
    area(axSpread, t, spread, "FaceColor", [0.85 0.55 0.20], "EdgeColor", "none");
    ylabel(axSpread, "max-min");
    title(axSpread, "Disagreement between the four baselines (where the model choice matters)");
    grid(axSpread, "on");
    xlabel(axSpread, "Time");

    overlayEventMarkers([axTop, axSpread], evTimes);
    linkaxes([axTop, axSpread], "x");
end


function makeSmoothingComparisonFigure(chName, t, raw, chResults, sNames_, spec, cell, evTimes)
%MAKESMOOTHINGCOMPARISONFIGURE How pre-smoothing changes the anomaly picture,
% holding baseline+detector fixed (`cell`, e.g. "B4_poly_D1_hysteresis").  Shows
% the channel, a flag raster with one row per smoother, and the pairwise Jaccard
% of each smoother's flags against the S0none control - so "does denoising the
% input change the flags?" is answered directly.
    nSm = numel(sNames_);
    M = false(numel(t), nSm);
    for k = 1:nSm
        M(:, k) = chResults.(sNames_{k} + "_" + cell).mask;
    end

    fig = figure("Name", sprintf("%s - smoothing comparison (%s)", chName, cell));
    tl = tiledlayout(fig, 5, 1, "TileSpacing", "compact", "Padding", "compact");
    title(tl, sprintf("%s - effect of input pre-smoothing on flags  [fixed: %s]", ...
        chName, cell), "Interpreter", "none");

    axTop = nexttile(tl, [2 1]);
    plot(axTop, t, raw, "Color", [0.2 0.2 0.2], "LineWidth", 0.5);
    ylabel(axTop, chName); grid(axTop, "on"); title(axTop, "Channel (raw, masked)");

    axRas = nexttile(tl, [2 1]);
    hold(axRas, "on");
    for k = 1:nSm
        yy = nSm - k + 1;
        [s, e] = localRuns(M(:, k));
        for r = 1:numel(s)
            patch(axRas, [t(s(r)) t(e(r)) t(e(r)) t(s(r))], ...
                  [yy-0.4 yy-0.4 yy+0.4 yy+0.4], [0.30 0.45 0.80], "EdgeColor", "none");
        end
    end
    hold(axRas, "off");
    ylim(axRas, [0.5, nSm+0.5]); yticks(axRas, 1:nSm);
    yticklabels(axRas, flipud(string(sNames_(:))));
    set(axRas, "TickLabelInterpreter", "none");
    title(axRas, "Flagged events by smoother"); grid(axRas, "on");
    overlayEventMarkers([axTop, axRas], evTimes);
    linkaxes([axTop, axRas], "x");

    % Bottom: Jaccard of each smoother vs the None control (agreement with no-smooth).
    axJ = nexttile(tl);
    ctrl = M(:, 1);
    jac = zeros(nSm, 1);
    for k = 1:nSm
        inter = nnz(M(:,k) & ctrl); uni = nnz(M(:,k) | ctrl);
        jac(k) = inter / max(1, uni);
    end
    bar(axJ, jac, "FaceColor", [0.30 0.45 0.80]);
    ylim(axJ, [0 1]); xticks(axJ, 1:nSm);
    xticklabels(axJ, string(sNames_)); set(axJ, "TickLabelInterpreter", "none");
    ylabel(axJ, "Jaccard vs None");
    title(axJ, "Flag agreement with the no-smoothing control (1 = smoothing changed nothing)");
    grid(axJ, "on");
end

function y = invTransform(bt, raw, sgn, logSpace)
%INVTRANSFORM Map a transformed baseline series back to raw channel units,
% mirroring transformChannel's forward math (shift derived from the raw data).
    bt = bt(:);
    if logSpace
        if sgn < 0, xs = -raw; else, xs = raw; end
        mn = min(xs, [], "omitnan");
        shift = 0; if ~(mn > 0), shift = 1 - mn; end
        y = sgn * (exp(bt) - shift);
    else
        y = sgn * bt;
    end
end

function shadeMaskLocal(ax, t, mask, faceColor, faceAlpha)
%SHADEMASKLOCAL Shade contiguous true runs of mask behind the data.
    if ~any(mask), return; end
    held = ishold(ax); hold(ax, "on");
    yl = ylim(ax);
    d = diff([false; mask(:); false]);
    s = find(d == 1); e = find(d == -1) - 1;
    for k = 1:numel(s)
        i0 = s(k); i1 = min(e(k) + 1, numel(t));
        p = patch(ax, [t(i0) t(i1) t(i1) t(i0)], [yl(1) yl(1) yl(2) yl(2)], ...
            faceColor, "FaceAlpha", faceAlpha, "EdgeColor", "none", ...
            "HandleVisibility", "off");
        uistack(p, "bottom");
    end
    ylim(ax, yl);
    if ~held, hold(ax, "off"); end
end
