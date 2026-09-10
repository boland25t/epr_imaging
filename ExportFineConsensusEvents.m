%% ExportFineConsensusEvents.m — strict, peak-preserving event boundaries
%
% Exports events from the FINE regime only. A sample is a core anomaly when at
% least 50% of the 80 fine-regime strategies agree. This deliberately excludes
% the broad coarse-regime shoulders used for large/long-event context.

% `repo` may be injected by the caller (the app runs this via matlab -batch and
% pre-assigns it); fall back to the development path for a bare interactive run.
if ~exist("repo","var") || strlength(string(repo)) == 0
    repo = "/home/troyboland/epr_imaging/epr_imaging";
end
repo = string(repo);
configs = ["masked","nomask_log","nomask_ceiling","nomask_both"];
matNames = [
    "grapher_matrix_results_masked.mat"
    "grapher_matrix_results_nomask_log.mat"
    "grapher_matrix_results_nomask_ceiling.mat"
    "grapher_matrix_results_nomask_both.mat"
];
minFrac = 0.50;

for ci = 1:numel(configs)
    A = load(fullfile(pwd,matNames(ci)),"results","runChannels","time_channel");
    outdir = fullfile(pwd,"grapher_matrix_figs",configs(ci));
    for ch = reshape(A.runChannels,1,[])
        R = A.results.(ch).fine;
        keys = fieldnames(R);
        consensus = zeros(numel(A.time_channel),1);
        for ki = 1:numel(keys)
            consensus = consensus + double(R.(keys{ki}).mask);
        end
        threshold = ceil(minFrac*numel(keys));
        core = consensus >= threshold;
        d = diff([false;core;false]);
        starts = find(d==1);
        ends = find(d==-1)-1;
        n = numel(starts);
        start_time = strings(n,1);
        end_time = strings(n,1);
        duration_s = zeros(n,1);
        max_consensus = zeros(n,1);
        max_consensus_fraction = zeros(n,1);
        for i=1:n
            start_time(i) = string(datetime(A.time_channel(starts(i)), ...
                "Format","yyyy-MM-dd'T'HH:mm:ss"));
            end_time(i) = string(datetime(A.time_channel(ends(i)), ...
                "Format","yyyy-MM-dd'T'HH:mm:ss"));
            duration_s(i) = ends(i)-starts(i)+1;
            max_consensus(i) = max(consensus(starts(i):ends(i)));
            max_consensus_fraction(i) = max_consensus(i)/numel(keys);
        end
        T = table(start_time,end_time,duration_s,max_consensus, ...
            max_consensus_fraction);
        writetable(T,fullfile(outdir,"fine_core_events_"+ch+".csv"));
        fprintf("%-14s %-12s %3d core events (>= %d/%d fine strategies)\n", ...
            configs(ci),ch,n,threshold,numel(keys));
    end
end
