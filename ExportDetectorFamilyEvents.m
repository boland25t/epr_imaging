%% ExportDetectorFamilyEvents.m — per-detector fine-regime evidence
%
% Each detector family has 20 strategies (5 smoothers x 4 baselines). Export a
% family event where >=25% (5/20) agree. Downstream fusion retains D3 peak
% events directly and requires at least two families for non-D3 regions.

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
methodNames = ["D1_height","D2_mass","D3_peak","D4_local"];
methodTokens = ["_D1_","_D2_","_D3_","_D4_"];
minFrac = 0.25;

for ci = 1:numel(configs)
    A = load(fullfile(repo,matNames(ci)),"results","runChannels","time_channel");
    outdir = fullfile(repo,"grapher_matrix_figs",configs(ci));
    for ch = reshape(A.runChannels,1,[])
        R = A.results.(ch).fine;
        keys = string(fieldnames(R));
        rows = cell(0,7);
        for mi=1:numel(methodNames)
            selected = keys(contains(keys,methodTokens(mi)));
            consensus = zeros(numel(A.time_channel),1);
            for key=reshape(selected,1,[])
                consensus = consensus + double(R.(key).mask);
            end
            threshold = ceil(minFrac*numel(selected));
            active = consensus >= threshold;
            d = diff([false;active;false]);
            starts = find(d==1);
            ends = find(d==-1)-1;
            for i=1:numel(starts)
                maxVote = max(consensus(starts(i):ends(i)));
                rows(end+1,:) = { ...
                    string(datetime(A.time_channel(starts(i)), ...
                        "Format","yyyy-MM-dd'T'HH:mm:ss")), ...
                    string(datetime(A.time_channel(ends(i)), ...
                        "Format","yyyy-MM-dd'T'HH:mm:ss")), ...
                    ends(i)-starts(i)+1, methodNames(mi), maxVote, ...
                    maxVote/numel(selected), threshold}; %#ok<AGROW>
            end
            fprintf("%-14s %-12s %-10s %3d events (>= %d/%d)\n", ...
                configs(ci),ch,methodNames(mi),numel(starts),threshold,numel(selected));
        end
        T = cell2table(rows,'VariableNames', ...
            {'start_time','end_time','duration_s','method','method_consensus', ...
             'method_consensus_fraction','method_threshold'});
        writetable(T,fullfile(outdir,"fine_method_events_"+ch+".csv"));
    end
end
