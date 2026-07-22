function v = navSpeed(t, lat, lon, alt, opts)
%NAVSPEED  Vehicle speed (m/s) at each nav/channel row.
%
%   v = navSpeed(t, lat, lon, alt)
%
%   Nav and channel share a time axis, so this returns one speed per row,
%   aligned 1:1 with the input (same length, same order — nothing dropped).
%
%   INPUTS
%     t        - timestamps (Unix seconds or datetime), N x 1
%     lat, lon - decimal degrees, N x 1
%     alt      - vertical position, metres, N x 1.  [] for horizontal-only.
%                Use DEPTH for vehicle motion; ALTITUDE tracks terrain and
%                adds vertical speed the vehicle didn't actually have.
%
%   OPTIONS
%     Smooth - 'movmedian' (default) | 'gaussian' | 'none'
%     Window - smoothing window in samples (default 5)
%
%   OUTPUT
%     v - N x 1 speed (m/s).  NaN where speed is undefined (bad position,
%         non-increasing time).
%
%   EXAMPLE
%     v = navSpeed(T.unix_time, T.lat, T.lon, T.depth);
%     T.speed_mps = v;                 % drops straight into the table
%     hovering    = v < 0.03;

    arguments
        t
        lat
        lon
        alt = []
        opts.Smooth (1,:) char {mustBeMember(opts.Smooth, ...
                    {'movmedian','gaussian','none'})} = 'movmedian'
        opts.Window (1,1) double {mustBePositive} = 5
    end

    R = 6371000;                        % Earth radius, metres

    if isdatetime(t), t = posixtime(t); end
    t   = double(t(:));
    lat = double(lat(:));
    lon = double(lon(:));
    if isempty(alt), alt = zeros(size(t)); else, alt = double(alt(:)); end

    N = numel(t);
    v = nan(N,1);
    if N < 2, return; end

    % --- Per-segment 3-D distance (haversine horizontal + vertical) ---
    la1 = deg2rad(lat(1:end-1)); la2 = deg2rad(lat(2:end));
    lo1 = deg2rad(lon(1:end-1)); lo2 = deg2rad(lon(2:end));
    a = sin((la2-la1)/2).^2 + cos(la1).*cos(la2).*sin((lo2-lo1)/2).^2;
    horiz = 2 * R * asin(sqrt(a));
    seg   = sqrt(horiz.^2 + diff(alt).^2);      % (N-1) x 1

    % --- Speed AT each point ---
    % Interior: path length across BOTH adjacent segments over their total
    % duration (a duration-weighted centred difference).  Uses path length,
    % not the i-1 -> i+1 chord, so curvature isn't under-measured.
    dt_c = t(3:end) - t(1:end-2);
    num  = seg(1:end-1) + seg(2:end);
    good = dt_c > 0;
    v(2:end-1) = nan;
    v([false; good; false]) = num(good) ./ dt_c(good);

    % Endpoints: one-sided.
    if t(2)   > t(1),     v(1) = seg(1)   / (t(2)-t(1));     end
    if t(end) > t(end-1), v(N) = seg(end) / (t(end)-t(end-1)); end

    % --- Smooth (median first: robust to nav-jitter spikes) ---
    switch opts.Smooth
        case 'movmedian', v = movmedian(v, opts.Window, 'omitnan');
        case 'gaussian',  v = smoothdata(v, 'gaussian', opts.Window, 'omitnan');
    end
end