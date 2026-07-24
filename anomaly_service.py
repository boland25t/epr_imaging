"""Anomaly detection + site-catalog service (Qt-free).

Wraps the two halves of the anomaly pipeline so the GUI can drive them:

  1. DETECTOR (MATLAB)  GrapherMatrix.m -> grapher_matrix_results_<config>.mat
                        ExportFineConsensusEvents.m  -> fine_core_events_<ch>.csv
                        ExportDetectorFamilyEvents.m -> fine_method_events_<ch>.csv
  2. CATALOG (Python)   build_anomaly_site_catalog.py -> windows/sites/clips CSV,
                        GeoJSON, QGIS bundle, and the PDF review report.

Only step 1 needs MATLAB.  When MATLAB is unavailable the catalog step still
runs against whatever event CSVs already exist, so the app degrades gracefully
instead of failing outright.

Like the other services in this app this module is Qt-free: every long-running
call takes a ``log_fn`` (GUI + file, tee'd by the caller) and an optional
``file_log_fn`` (complete, uncapped task log), plus a ``cancel_cb`` polled
between stages.  main_window.py / stack_runner.py own the threading.
"""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Callable, Iterable, Optional

# Detector artefacts, in run order.
_MATLAB_STAGES = (
    ("GrapherMatrix.m",              "anomaly matrix (4 configs x 160 strategies/channel)"),
    ("ExportFineConsensusEvents.m",  "fine-consensus core events"),
    ("ExportDetectorFamilyEvents.m", "per-detector-family events"),
)

CONFIGS = ("masked", "nomask_log", "nomask_ceiling", "nomask_both")

# Confidence tiers, most to least severe.  These strings must match
# build_anomaly_site_catalog.py and the QGIS .qml symbology.
TIERS = ("HIGH", "MODERATE", "SCREEN")

# Shared tier colours so the map, the tab and the report agree.
TIER_COLORS = {
    "HIGH":     "#c62828",
    "MODERATE": "#ef7d00",
    "SCREEN":   "#3d6ea8",
}

# Cap GUI log lines per MATLAB stage; the rest goes to the file log only.
_MAX_GUI_LOG_LINES = 200


# --------------------------------------------------------------------------
# MATLAB availability
# --------------------------------------------------------------------------
def matlab_binary() -> Optional[str]:
    """Path to the MATLAB executable, or None if it isn't on PATH."""
    return shutil.which("matlab")


def matlab_unavailable_reason() -> Optional[str]:
    """Human-readable reason MATLAB can't be used, or None if it can.

    Mirrors photogrammetry_service.metashape_unavailable_reason() so the GUI can
    show one consistent style of explanation.
    """
    if not matlab_binary():
        return (
            "MATLAB was not found on PATH. The anomaly DETECTOR stage needs MATLAB "
            "(GrapherMatrix.m). Install MATLAB or add it to PATH; the catalog stage "
            "can still run against previously exported event CSVs."
        )
    return None


def matlab_version(timeout: int = 60) -> Optional[str]:
    """MATLAB version string, or None if it could not be queried."""
    exe = matlab_binary()
    if not exe:
        return None
    try:
        out = subprocess.run(
            [exe, "-batch", "disp(version)"],
            capture_output=True, text=True, timeout=timeout,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    for line in (out.stdout or "").splitlines():
        line = line.strip()
        # Skip MATLAB's startup warnings; the version line starts with a digit.
        if line and line[0].isdigit():
            return line
    return None


# --------------------------------------------------------------------------
# Detector stage (MATLAB)
# --------------------------------------------------------------------------
def detector_inputs_ready(interp_csv: Path) -> Optional[str]:
    """Return a problem string if the detector can't run, else None."""
    interp_csv = Path(interp_csv)
    if not interp_csv.is_file():
        return f"Interpolated sensor table not found: {interp_csv}"
    return None


def run_detector(
    repo: Path,
    interp_csv: Path,
    log_fn: Callable[[str], None],
    file_log_fn: Optional[Callable[[str], None]] = None,
    cancel_cb: Optional[Callable[[], bool]] = None,
    timeout_s: int = 24 * 3600,
) -> dict:
    """Run the MATLAB detector chain against ``interp_csv``.

    GrapherMatrix.m reads a RELATIVE "interp_full.csv", so MATLAB is started with
    its working directory set to the folder containing ``interp_csv``; the two
    exporters get ``repo`` injected. Returns a summary dict.

    Raises RuntimeError if MATLAB is unavailable or a stage fails.
    """
    repo = Path(repo).resolve()
    interp_csv = Path(interp_csv).resolve()

    reason = matlab_unavailable_reason()
    if reason:
        raise RuntimeError(reason)
    problem = detector_inputs_ready(interp_csv)
    if problem:
        raise RuntimeError(problem)

    if interp_csv.name != "interp_full.csv":
        log_fn(
            f"  note: GrapherMatrix.m opens 'interp_full.csv' by name; "
            f"using working directory {interp_csv.parent} where the file is "
            f"actually named '{interp_csv.name}'"
        )

    workdir = interp_csv.parent
    exe = matlab_binary()
    log_fn(f"  MATLAB: {exe}")
    ver = matlab_version()
    if ver:
        log_fn(f"  MATLAB version: {ver}")
    log_fn(f"  repo:        {repo}")
    log_fn(f"  working dir: {workdir}")

    results: dict[str, float] = {}
    t_all = time.time()
    for script, description in _MATLAB_STAGES:
        if cancel_cb and cancel_cb():
            raise RuntimeError("Cancelled before " + script)
        if not (repo / script).is_file():
            raise RuntimeError(f"Missing MATLAB script: {repo / script}")

        log_fn(f"  [{script}] {description} …")
        t0 = time.time()
        # addpath(repo)  -> find the .m files
        # cd(workdir)    -> resolve GrapherMatrix's relative interp_full.csv
        # repo=...       -> injected for the exporters (they honour it if set)
        # figures off    -> headless
        stmt = (
            f"addpath('{_m_escape(repo)}');"
            f"cd('{_m_escape(workdir)}');"
            f"repo=\"{_m_escape(repo)}\";"
            "set(0,'DefaultFigureVisible','off');"
            f"run('{_m_escape(repo / script)}');"
        )
        _matlab_run(exe, stmt, script, log_fn, file_log_fn, cancel_cb, timeout_s)
        dt = time.time() - t0
        results[script] = dt
        log_fn(f"  [{script}] done in {_fmt_duration(dt)}")

    total = time.time() - t_all
    log_fn(f"  detector chain complete in {_fmt_duration(total)}")
    return {
        "stages": results,
        "total_s": total,
        "workdir": workdir,
        "mat_files": sorted(str(p) for p in workdir.glob("grapher_matrix_results_*.mat")),
    }


def _matlab_run(
    exe: str,
    statement: str,
    label: str,
    log_fn: Callable[[str], None],
    file_log_fn: Optional[Callable[[str], None]],
    cancel_cb: Optional[Callable[[], bool]],
    timeout_s: int,
) -> None:
    """Run one `matlab -batch <statement>`, streaming output.

    GUI output is capped at _MAX_GUI_LOG_LINES lines (MATLAB is chatty and the
    matrix run emits thousands); the complete stream always reaches file_log_fn.
    """
    cmd = [exe, "-batch", statement]
    log_fn(f"  $ {exe} -batch \"{_ellipsize(statement, 160)}\"")
    if file_log_fn:
        file_log_fn(f"  full statement: {statement}")

    env = dict(os.environ)
    env.setdefault("QT_QPA_PLATFORM", "offscreen")   # headless figure rendering

    try:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to launch MATLAB for {label}: {exc}") from exc

    emitted = 0
    capped = False
    tail: list[str] = []          # keep the last lines for error reporting
    deadline = time.time() + timeout_s
    try:
        for raw in proc.stdout:
            line = raw.rstrip("\n").rstrip()
            if not line.strip():
                continue
            tail.append(line)
            if len(tail) > 40:
                tail.pop(0)
            if file_log_fn:
                file_log_fn(f"    {line}")
            if emitted < _MAX_GUI_LOG_LINES:
                log_fn(f"    {line}")
                emitted += 1
            elif not capped:
                log_fn(
                    f"    … (GUI output capped at {_MAX_GUI_LOG_LINES} lines; "
                    "full MATLAB output continues in the task log file)"
                )
                capped = True

            if cancel_cb and cancel_cb():
                proc.kill()
                raise RuntimeError(f"Cancelled during {label}")
            if time.time() > deadline:
                proc.kill()
                raise RuntimeError(f"{label} exceeded {timeout_s}s timeout")
    finally:
        try:
            proc.stdout.close()
        except Exception:
            pass

    code = proc.wait()
    if code != 0:
        detail = "\n".join(f"      {t}" for t in tail[-15:])
        raise RuntimeError(
            f"{label} failed (MATLAB exit code {code}).\n"
            f"    last output:\n{detail}"
        )


# --------------------------------------------------------------------------
# Catalog stage (Python)
# --------------------------------------------------------------------------
def run_catalog(
    repo: Path,
    interp_csv: Optional[Path] = None,
    raw_nav_csv: Optional[Path] = None,
    event_root: Optional[Path] = None,
    ts_results: Optional[Path] = None,
    out_dir: Optional[Path] = None,
    log_fn: Callable[[str], None] = print,
) -> dict:
    """Build the anomaly site catalog (windows, sites, clips, QGIS, PDF).

    Thin wrapper over build_anomaly_site_catalog so the GUI never imports the
    script's globals directly.  Returns the builder's summary dict.
    """
    import build_anomaly_site_catalog as builder

    paths = builder.configure(
        repo=repo,
        event_root=event_root,
        interp=interp_csv,
        raw_nav=raw_nav_csv,
        ts_results=ts_results,
        out=out_dir,
    )
    for key, value in paths.items():
        log_fn(f"  {key:<11}{value}")

    problems = builder.preflight()
    if problems:
        raise RuntimeError("; ".join(problems))

    return builder.run(log=log_fn)


# --------------------------------------------------------------------------
# Reading the catalog back (for the GUI table / map / interval selector)
# --------------------------------------------------------------------------
def catalog_paths(out_dir: Path) -> dict[str, Path]:
    """Canonical artefact paths inside a catalog output directory."""
    out_dir = Path(out_dir)
    return {
        "windows_csv":   out_dir / "anomaly_windows_all.csv",
        "sites_csv":     out_dir / "anomalous_sites.csv",
        "sites_geojson": out_dir / "anomalous_sites.geojson",
        "clips_csv":     out_dir / "video_review_clips.csv",
        "combos_csv":    out_dir / "anomaly_combination_summary.csv",
        "report_pdf":    out_dir / "Anomaly_Site_and_Video_Review_Report.pdf",
        "qgis_dir":      out_dir / "qgis",
        "qgis_zip":      out_dir / "qgis" / "J1754_anomaly_QGIS_upload.zip",
    }


def load_windows(out_dir: Path) -> list[dict]:
    """Load anomaly_windows_all.csv as a list of plain dicts.

    Returns [] when the catalog has not been built yet, so the GUI can show an
    empty table rather than raising.
    """
    csv_path = catalog_paths(out_dir)["windows_csv"]
    if not csv_path.is_file():
        return []
    with csv_path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        for key in ("evidence_score", "lat", "lon", "water_depth_m", "altitude_m"):
            row[key] = _to_float(row.get(key))
        for key in ("duration_s", "review_duration_s", "channel_count", "detector_count"):
            row[key] = _to_int(row.get(key))
    return rows


def tier_counts(windows: Iterable[dict]) -> dict[str, int]:
    """Count windows per confidence tier (always includes every tier key)."""
    counts = {tier: 0 for tier in TIERS}
    for row in windows:
        tier = str(row.get("confidence_tier", "")).upper()
        if tier in counts:
            counts[tier] += 1
    return counts


def windows_to_intervals(
    windows: Iterable[dict],
    use_review_window: bool = True,
) -> list[dict]:
    """Convert catalog windows into interval dicts for the Jobs tab.

    Args:
        windows: rows from load_windows() (already filtered/selected by the GUI).
        use_review_window: True  -> review_start/review_end (anomaly padded by
                           the catalog's VIDEO_CONTEXT_SECONDS, what the video
                           review queue uses);
                           False -> tight start_time/end_time.

    Returns dicts carrying BOTH the raw ISO strings and parsed naive datetimes
    (``start_dt`` / ``end_dt``).  Parsing reuses interval_io's single timestamp
    parser, so the catalog's trailing "Z" is stripped exactly the way imported
    interval CSVs are — this project treats all data as one timezone.
    Rows whose timestamps cannot be parsed, or where start >= end, are skipped.
    """
    from interval_io import _parse_timestamp

    start_key, end_key = (
        ("review_start", "review_end") if use_review_window else ("start_time", "end_time")
    )
    out: list[dict] = []
    for row in windows:
        start, end = row.get(start_key), row.get(end_key)
        if not start or not end:
            continue
        start_dt, end_dt = _parse_timestamp(start), _parse_timestamp(end)
        if start_dt is None or end_dt is None or start_dt >= end_dt:
            continue
        out.append({
            "start_time": start,
            "end_time": end,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "window_id": row.get("window_id", ""),
            "site_id": row.get("site_id", ""),
            "confidence_tier": row.get("confidence_tier", ""),
            "evidence_score": row.get("evidence_score"),
            "channels": row.get("channels", ""),
            "anomaly_class": row.get("anomaly_class", ""),
            "source": "anomaly",
            "threshold_desc": (
                f"anomaly {row.get('window_id','')} "
                f"[{row.get('confidence_tier','')}] {row.get('channels','')}"
            ).strip(),
        })
    return out


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _m_escape(value) -> str:
    """Escape a path for embedding in a single-quoted MATLAB string."""
    return str(value).replace("'", "''")


def _ellipsize(text: str, limit: int) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _fmt_duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m {seconds % 60}s"
    return f"{seconds // 3600}h {(seconds % 3600) // 60}m"


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_int(value):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None
