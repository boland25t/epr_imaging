#!/usr/bin/env python3
"""Frame-level colour/brightness vs dissolved-gas analysis (between-station transits).

Hypothesis: after altitude-normalised colour correction, whiter frames (white
bacterial matting) coincide with elevated dissolved gas — tested on the survey's
TRANSIT data only.  Stationary periods correspond to deliberate measurements at
already-known high-CH4 sites (dive-plan bias), so frames below V_MIN vehicle
speed are excluded; what remains is the un-targeted record between stations.

Per-frame ground truth (timestamp, altitude, position, CO2/CH4/O2) comes from the
sampling pipeline's own per-segment manifests (segment_*/interp.csv) — frame
filenames are never parsed for time.

The two reconstructions omitted from the survey report (seg16/chunk_02,
seg17/chunk_01) are omitted as *mosaic products* only; their source frames are
ordinary photographs and ARE included here (user decision).
"""
from __future__ import annotations
import glob, json, re, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np, pandas as pd, cv2

# Altitude-attenuation model, fitted on per-frame manifest altitudes
# (ln(channel) vs altitude; see analysis_figs/attenuation.png) + tuned white
# balance.  The altitude term is clamped to +/-ALT_CLAMP m of the reference so
# range extremes are not over-amplified into saturation.
C = {"R": 0.39, "G": 0.25, "B": 0.12}
ALT_REF = 5.0
ALT_CLAMP = 2.0
WB = {"R": 1.35, "G": 1.0, "B": 0.83}
WHITE_T = 130    # per-channel floor for a "white/mat" pixel (0-255, corrected)

STRIDE = 5       # analyse every 5th extracted frame (~1.25 m spacing)
WORKERS = 10     # parallel reads (I/O-bound on the /mnt/f mount)
V_MIN = 0.08     # m/s — below this the vehicle is on-station; frames dropped


def correct(im_bgr, alt):
    da = float(np.clip(alt - ALT_REF, -ALT_CLAMP, ALT_CLAMP))
    f = im_bgr.astype(np.float32)
    f[:, :, 2] *= np.exp(C["R"] * da) * WB["R"]
    f[:, :, 1] *= np.exp(C["G"] * da) * WB["G"]
    f[:, :, 0] *= np.exp(C["B"] * da) * WB["B"]
    return np.clip(f, 0, 255).astype(np.uint8)


def measure(cor_bgr):
    b, g, r = cor_bgr[:, :, 0], cor_bgr[:, :, 1], cor_bgr[:, :, 2]
    mn = np.minimum(np.minimum(r, g), b)
    white = float((mn > WHITE_T).mean())      # mat fraction: bright AND neutral
    bright = float(cor_bgr.mean())
    matidx = float((mn / 255.0).mean())
    return white, bright, matidx


def load_frame_manifest(workspace_dir) -> pd.DataFrame:
    """Authoritative per-frame table from segment_*/interp.csv manifests."""
    rows = []
    for mc in sorted(glob.glob(
            f"{workspace_dir}/survey/photogrammetry/seg*/segment_*/interp.csv")):
        seg = re.search(r"photogrammetry/(seg\d+)/", mc.replace("\\", "/")).group(1)
        m = pd.read_csv(mc)
        ren = {}
        for c in m.columns:
            cl = c.lower().replace(" ", "")
            if cl.startswith("co2"): ren[c] = "CO2"
            elif cl.startswith("ch4"): ren[c] = "CH4"
            elif cl.startswith("o2"): ren[c] = "O2"
        m = m.rename(columns=ren)
        base = Path(mc).parent / "frames"
        m["fn"] = [str(base / f) for f in m.frame_filename]
        m["seg"] = seg
        rows.append(m[["fn", "seg", "unix_time", "alt", "easting", "northing",
                       "CO2", "CH4", "O2"]])
    return (pd.concat(rows, ignore_index=True)
            .dropna(subset=["unix_time", "alt"])
            .sort_values("unix_time").reset_index(drop=True))


def vehicle_speed_at(workspace_dir, times) -> np.ndarray:
    """Smoothed horizontal speed (m/s) at `times` — same rolling-median
    smoothing as nav_segments, from the global interp."""
    ip = pd.read_csv(f"{workspace_dir}/inputs/interp_full.csv",
                     usecols=["unix_time", "easting", "northing"]).dropna()
    ip = ip.sort_values("unix_time")
    t = ip.unix_time.to_numpy(); e = ip.easting.to_numpy(); n = ip.northing.to_numpy()
    dt = np.diff(t); spd = np.zeros(len(t))
    spd[1:] = np.divide(np.hypot(np.diff(e), np.diff(n)), dt,
                        out=np.zeros(len(dt)), where=dt > 0)
    sm = pd.Series(spd).rolling(15, center=True, min_periods=1).median().to_numpy()
    return np.interp(np.asarray(times, float), t, sm)


def fit_attenuation(workspace_dir, sample_stride=30, log=print) -> dict:
    """Refit the per-channel attenuation slopes on THIS dive's frames (raw
    channel means vs manifest altitude) and update the module coefficients in
    place, so correct() and every importer use the dive's own model."""
    mf = load_frame_manifest(str(workspace_dir)).iloc[::sample_stride]
    from concurrent.futures import ThreadPoolExecutor

    def work(row):
        im = cv2.imread(row.fn, cv2.IMREAD_REDUCED_COLOR_8)
        if im is None: return None
        return (row.alt, im[:, :, 2].mean(), im[:, :, 1].mean(), im[:, :, 0].mean())
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        rows = [r for r in ex.map(work, mf.itertuples(index=False)) if r is not None]
    a = pd.DataFrame(rows, columns=["alt", "R", "G", "B"]).dropna()
    fit = {ch: float(-np.polyfit(a.alt, np.log(a[ch].clip(1)), 1)[0])
           for ch in ("R", "G", "B")}
    C.update(fit)
    log(f"attenuation refit on {len(a)} frames: " +
        ", ".join(f"{k} -{v:.2f}/m" for k, v in fit.items()))
    return fit


def build_table(workspace_dir, log=print, refit=True) -> pd.DataFrame:
    B = str(workspace_dir)
    if refit:
        fit_attenuation(B, log=log)
    mf = load_frame_manifest(B)
    n_manifest = len(mf)
    mf["speed"] = vehicle_speed_at(B, mf.unix_time)
    mf = mf[mf.speed >= V_MIN].reset_index(drop=True)
    n_moving = len(mf)
    mf = mf.iloc[::STRIDE].reset_index(drop=True)
    log(f"{n_manifest} manifest frames -> {n_moving} in transit (speed >= {V_MIN} m/s) "
        f"-> {len(mf)} analysed (every {STRIDE}th)")
    win_csv = Path(B) / "survey" / "anomaly" / "anomaly_windows_all.csv"
    if win_csv.is_file():
        win = pd.read_csv(win_csv)
        ws = pd.to_datetime(win.start_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
        we = pd.to_datetime(win.end_time, utc=True).map(pd.Timestamp.timestamp).to_numpy()
    else:
        log("  no anomaly windows yet — 'anom' column will be all-False")
        ws = we = np.array([])
    done = [0]

    def work(row):
        im = cv2.imread(row.fn, cv2.IMREAD_REDUCED_COLOR_8)
        if im is None: return None
        im = cv2.resize(im, (240, 135))
        white, bright, matidx = measure(correct(im, row.alt))
        done[0] += 1
        if done[0] % 250 == 0: log(f"  {done[0]}/{len(mf)}")
        t = row.unix_time
        return dict(fn=row.fn, seg=row.seg, t=t, alt=row.alt, speed=row.speed,
                    CO2=row.CO2, CH4=row.CH4, O2=row.O2,
                    E=row.easting, N=row.northing,
                    white=white, bright=bright, matidx=matidx,
                    anom=bool(np.any((t >= ws) & (t <= we))))
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        out = [r for r in ex.map(work, mf.itertuples(index=False)) if r is not None]
    df = pd.DataFrame(out)
    df.to_csv(f"{B}/survey/frame_color_metrics.csv", index=False)
    Path(f"{B}/survey/frame_color_meta.json").write_text(json.dumps(dict(
        n_manifest=n_manifest, n_moving=n_moving, n_analysed=len(df),
        stride=STRIDE, v_min=V_MIN, coeffs=dict(C), wb=dict(WB)), indent=1))
    return df


def report(df, workspace_dir):
    from scipy.stats import spearmanr, mannwhitneyu
    out = [f"frames analysed (transit only): {len(df)}",
           f"whiteness: median {df.white.median():.3f}  mean {df.white.mean():.3f}  "
           f"max {df.white.max():.3f}",
           f"altitude: {df.alt.min():.1f}-{df.alt.max():.1f} m", ""]
    out.append(f"{'metric':10}{'vs CO2':>16}{'vs CH4':>16}{'vs O2':>16}")
    for m in ("white", "matidx", "bright"):
        cells = []
        for g in ("CO2", "CH4", "O2"):
            d = df[[m, g]].replace([np.inf, -np.inf], np.nan).dropna()
            rho, p = spearmanr(d[m], d[g])
            cells.append(f"rho={rho:+.2f} p={'<.001' if p < 1e-3 else f'{p:.3f}'}")
        out.append(f"{m:10}" + "".join(f"{c:>16}" for c in cells))
    mat, bare = df[df.white > 0.02], df[df.white <= 0.02]
    if len(mat) > 5:
        out.append("")
        for g in ("CH4", "CO2", "O2"):
            u, p = mannwhitneyu(mat[g].dropna(), bare[g].dropna(),
                                alternative=("less" if g == "O2" else "greater"))
            out.append(f"{g}: mat median {mat[g].median():.1f} vs bare "
                       f"{bare[g].median():.1f}  p={'<.001' if p < 1e-3 else f'{p:.3f}'}")
    txt = "\n".join(out)
    Path(f"{workspace_dir}/survey/frame_color_report.txt").write_text(txt)
    return txt


if __name__ == "__main__":
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    def log(m): print(m, flush=True)
    df = build_table(ws, log)
    print(report(df, ws), flush=True)
    print("ANALYSIS_DONE", flush=True)
