"""product_catalog.py — the entire backend for the simplified EPR Imaging UI.

One module, Qt-free, that answers three questions for every product the app can
make:

    * what products exist?              ProductType.discover(ws, job)
    * how do I make another one?        ProductType.generate(ws, job, settings, log)
    * what work is "everything"?        default_run_all(ws, log_fn, job)

Nothing here reimplements a pipeline.  Every product is produced by the
battle-tested services (``pipeline_service``, ``photogrammetry_service``,
``output_service``, ``anomaly_service``, ``survey_report``, ``qgis_project``);
this module only decides *scope* (whole trackline vs a job's intervals),
*paths*, and *reuse*.

Scope model
-----------
A **Job** is a named set of [t0, t1] unix intervals.  ``WHOLE_TRACKLINE`` is the
implicit job that means "no restriction".  Jobs live in workspace.json under
``"simple_jobs"``; creating a job from a base job never mutates the base — it
mints a NEW job carrying the base intervals plus the new ones.

Products are stored so that scope is legible from the path:

    <ws>/survey/…                       whole-trackline products (also: every
                                        legacy/batch product predating this UI)
    <ws>/survey/jobs/<job_id>/…         one job's products
    <ws>/survey/frame_sets/<scope>/…    frame sets (scanned across all scopes
                                        for recycling)

Every product this module writes drops a ``simple_meta.json`` beside itself
carrying ``job_id`` + a human "characteristic" string; discovery prefers that
file and falls back to path-based scoping, so pre-existing products made by the
old UI/batch runs show up under WHOLE_TRACKLINE instead of vanishing.

Frame recycling (the core principle)
------------------------------------
Frame extraction is the expensive step, so the sampling grid is made
*deterministic*: sample times are the crossings of a cumulative along-track
distance grid anchored at the dive's first nav fix (plus a minimum-frequency
floor).  The grid therefore depends only on (interp_full.csv, spacing) — never
on the requested interval — so a whole-trackline run and an interval run ask
for exactly the same timestamps where they overlap.  Before extracting,
``frame_set.generate`` scans every frame set already on disk (and, optionally,
the legacy ``survey/photogrammetry/*/segment_*`` dirs) and links the frames
whose timestamps are already covered; only genuinely uncovered sub-spans reach
the video decoder.

Explicitly dropped: nav 3-D PLY and sensor "interpolation cloud" 3-D PLY.  They
are never generated and never surfaced.
"""
from __future__ import annotations

import calendar
import json
import math
import os
import shutil
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------
# Constants / defaults
# --------------------------------------------------------------------------

WHOLE_TRACKLINE = "__whole__"          # job id sentinel: "no interval restriction"

_REPO = str(Path(__file__).resolve().parent)
_META = "simple_meta.json"             # per-product provenance written by generate()
_SET_META = "set_meta.json"            # per-frame-set manifest

# Fallback channel list (matches batch_service.SENSOR_CHANNELS) used only when a
# workspace cannot be read.
_FALLBACK_CHANNELS = ["CO2 Concentration", "CH4 Concentration", "O2 Concentration",
                      "Salinity", "Temperature"]

try:      # the adopted "recipe1" block — see batch_service.DEFAULT_PHOTO_SETTINGS
    from batch_service import DEFAULT_PHOTO_SETTINGS as _PHOTO_SETTINGS
except Exception:                                                   # noqa: BLE001
    # Fallback MUST mirror batch_service.DEFAULT_PHOTO_SETTINGS, whose settings
    # come from /home/troyboland/epr_claude_paper/docs/RECIPE_REFERENCE.md §2.
    _PHOTO_SETTINGS = dict(
        quality_threshold=0.0, align_accuracy="High", use_nav_reference=True,
        key_point_limit=40000, tie_point_limit=10000, adaptive_fitting=True,
        generic_preselect=True,
        nav_accuracy_h=0.1, nav_accuracy_v=0.05,
        rotation_mode="mount_corrected",
        mount_yaw_offset_deg=180.0, mount_pitch_offset_deg=-22.0,
        nav_rotation_accuracy_deg=30.0,
        fixed_calibration=dict(f=3836.98, cx=0.0, cy=0.0, k1=-0.2607,
                               k2=0.4638, k3=-0.2959, width=5312, height=2988),
        build_dense=True, dense_quality="Low", depth_filter="Moderate",
        export_dense_ply=True, build_mesh=True, mesh_source="Dense cloud",
        mesh_surface="Height Field", mesh_faces="Medium",
        mesh_interpolation="Enabled", export_mesh_obj=True, build_dem=True,
        export_dem=True, build_orthomosaic=True, ortho_surface="DEM",
        make_report=True, save_project=True,
    )

#: Fauna computer vision.  The FINE-TUNE FAILED (see deploy_config.json's
#: recommended_model_per_bucket / f1_by_model): the MBARI 315k zero-shot model
#: wins on every bucket that matters, so it — not epr_fauna_v2 — is production.
#:
#: Locations (review 08 P1-1).  Set these environment variables on any machine
#: other than the author's:
#:   EPR_FAUNA_WEIGHTS        path to mbari_315k_yolov8.pt (MBARI FathomNet 315k)
#:   EPR_FAUNA_DEPLOY_CONFIG  path to deploy/deploy_config.json (optional; only
#:                            used when per-bucket thresholds are switched on)
#: Documented fallbacks (the author's workstation) apply when a variable is unset.
FAUNA_WEIGHTS_ENV = "EPR_FAUNA_WEIGHTS"
FAUNA_DEPLOY_CONFIG_ENV = "EPR_FAUNA_DEPLOY_CONFIG"
_FAUNA_WEIGHTS_FALLBACK = "/home/troyboland/models/mbari_315k_yolov8.pt"
_FAUNA_DEPLOY_CONFIG_FALLBACK = "/home/troyboland/models/deploy/deploy_config.json"
FAUNA_WEIGHTS = Path(os.environ.get(FAUNA_WEIGHTS_ENV) or _FAUNA_WEIGHTS_FALLBACK)
FAUNA_DEPLOY_CONFIG = Path(os.environ.get(FAUNA_DEPLOY_CONFIG_ENV)
                           or _FAUNA_DEPLOY_CONFIG_FALLBACK)
FAUNA_MODEL_KEY = "mbari_zero_shot"

DEFAULTS: dict = {
    # sampling
    "sampling_mode": "dynamic",
    "spacing_m": 0.25,
    "min_frequency_hz": 0.1,
    "reuse_tolerance_s": 0.5,
    "reuse_legacy": True,
    "min_run_samples": 15,         # shorter covered/uncovered runs get coalesced
    # photogrammetry
    "chunk_size": 350,             # HARD policy: cap on frames per Metashape chunk
    "chunk_target_min": 250,       # HARD policy: chunks aim for 250-350 frames
    "alt_max_m": 8.0,              # HARD policy: altitude gate for photogrammetry
    "min_chunk_frames": 15,        # HARD policy: drop sub-spans smaller than this
    "photo_settings": dict(_PHOTO_SETTINGS),
    # rasters
    "cell_size_m": 5.0,
    "crs_mode": "utm",
    "fill_method": "idw",
    # fauna computer vision (zero-shot MBARI weights)
    "fauna_weights": str(FAUNA_WEIGHTS),
    "fauna_imgsz": 1280,
    "fauna_batch": 8,              # 5312x2988 JPEGs: 8 is ~380 MB of decode RAM
    "fauna_device": "auto",        # auto | cuda | cpu | "0"
    "fauna_conf": 0.25,            # floor used only when deploy_config is absent
    # Default OFF (flat conf 0.25) so fresh runs match the published census and
    # the pristine J1754/J1758 products (rehearsal 2026-09-22 found the per-bucket
    # post-filter drops 420/445 fish vs those products — a 15x on-stage mismatch).
    # The calibrated per-bucket operating points remain available as a setting.
    "fauna_bucket_thresholds": False,
}

TIER_COLOUR = {"HIGH": "#d7263d", "MODERATE": "#e8871e", "SCREEN": "#e0a800"}

# Log-line conventions every log_fn sink may rely on (the UI colours/collapses
# on these prefixes, so they are part of the contract, not decoration).
LOG_FAIL = "!! "        # a failure: one compact line, rendered red
LOG_WARN = "note: "    # a warning that did not stop the step
LOG_DETAIL = "  · "     # supporting detail (traceback): collapsible


def one_line(exc: BaseException, limit: int = 240) -> str:
    """An exception as a single, compact, human-readable line."""
    text = " ".join(str(exc).split()) or exc.__class__.__name__
    if str(exc):
        text = f"{type(exc).__name__}: {text}"
    return text if len(text) <= limit else text[:limit - 1] + "…"


# --------------------------------------------------------------------------
# Dataclasses (the contract's vocabulary)
# --------------------------------------------------------------------------

@dataclass
class Interval:
    """A half-open-ish time window in UTC unix seconds."""
    t0: float
    t1: float

    def as_list(self) -> list[float]:
        return [float(self.t0), float(self.t1)]

    def contains(self, t: float) -> bool:
        return self.t0 <= t <= self.t1


@dataclass
class Job:
    job_id: str                          # "job_0001" or WHOLE_TRACKLINE
    name: str                            # display, e.g. "Job 3 (2 intervals)"
    intervals: list[Interval] = field(default_factory=list)   # empty for WHOLE

    @property
    def is_whole(self) -> bool:
        return self.job_id == WHOLE_TRACKLINE

    def to_dict(self) -> dict:
        return {"job_id": self.job_id, "name": self.name,
                "intervals": [iv.as_list() for iv in self.intervals]}


@dataclass
class ProductInstance:
    type_key: str
    label: str                           # "2026-09-18 14:02 · dynamic 0.25 m"
    path: str                            # primary artifact path
    created_at: float
    job_id: str
    view_paths: list[str] = field(default_factory=list)


@dataclass
class ProductType:
    """One row of the Products tree.

    ``settings_schema`` entries are ``(key, label, kind, default, extra)`` with
    kind in {'float','int','str','choice','bool'}; ``extra`` carries the choice
    list for 'choice' (or None).  ``schema(ws)`` returns the same list with
    workspace-dependent choices (e.g. sensor channels) filled in — prefer it
    when a workspace is available.
    """
    key: str
    label: str
    settings_schema: list[tuple] = field(default_factory=list)
    _discover: Optional[Callable] = None
    _generate: Optional[Callable] = None
    _schema_fn: Optional[Callable] = None
    # Set by discover() when it swallows an exception, so a caller (the UI) can
    # tell "this scope has no products" apart from "discovery broke".
    last_error: Optional[str] = None

    # -- introspection ----------------------------------------------------
    def schema(self, ws: Optional[str] = None) -> list[tuple]:
        if self._schema_fn is not None and ws:
            try:
                return self._schema_fn(ws)
            except Exception:                                       # noqa: BLE001
                pass
        return list(self.settings_schema)

    def defaults(self, ws: Optional[str] = None) -> dict:
        return {row[0]: row[3] for row in self.schema(ws)}

    # -- behaviour --------------------------------------------------------
    def discover(self, ws: str, job: Job) -> list[ProductInstance]:
        if self._discover is None:
            return []
        try:
            out = self._discover(ws, job) or []
        except Exception as exc:                                    # noqa: BLE001
            self.last_error = f"{type(exc).__name__}: {exc}"
            return []
        self.last_error = None
        return sorted(out, key=lambda p: p.created_at, reverse=True)

    def generate(self, ws: str, job: Job, settings: Optional[dict] = None,
                 log_fn: Optional[Callable[[str], None]] = None) -> ProductInstance:
        if self._generate is None:
            raise RuntimeError(f"{self.key}: not generatable")
        log = _logger(log_fn)
        merged = self.defaults(ws)
        merged.update(settings or {})
        log(f"=== {self.label} — {job.name} ===")
        with run_lock(ws, f"{self.key} ({job.name})"):
            return self._generate(ws, job, merged, log)


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------

def _logger(log_fn: Optional[Callable[[str], None]]) -> Callable[[str], None]:
    sink = log_fn or print

    def _log(message: str) -> None:
        try:
            sink(str(message))
        except Exception:                                           # noqa: BLE001
            pass
    return _log


def naive_utc(ts: float) -> datetime:
    """Unix seconds -> naive-UTC datetime (the app's datetime convention)."""
    return datetime.fromtimestamp(float(ts), tz=timezone.utc).replace(tzinfo=None)


def unix(dt: datetime) -> float:
    """Naive-UTC datetime -> unix seconds (matches PipelineService's timegm)."""
    return float(calendar.timegm(dt.timetuple()))


def _stamp(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts if ts is not None else time.time()).strftime(
        "%Y%m%dT%H%M%S")


def _label(ts: float, characteristic: str) -> str:
    return f"{datetime.fromtimestamp(float(ts)):%Y-%m-%d %H:%M} · {characteristic}"


def _mtime(path) -> float:
    try:
        return float(Path(path).stat().st_mtime)
    except OSError:
        return 0.0


def _resolver(ws: str):
    from workspace_paths import PathResolver
    return PathResolver(str(ws))


def _ws_json(ws: str) -> Path:
    return Path(ws) / "workspace.json"


def _ws_data(ws: str) -> dict:
    """Typed workspace configs via ConfigService (nav/sensor objects resolved)."""
    from config_service import ConfigService
    return ConfigService.load_workspace(str(_ws_json(ws)))


def _raw_ws(ws: str) -> dict:
    """Raw workspace.json.  {} only when the file is ABSENT; a present but
    unreadable/corrupt file raises ``config_service.WorkspaceFileError`` so no
    read-modify-write caller can turn it into a jobs-only file (review 04 P1-1,
    03 P1-2)."""
    from config_service import read_workspace_json
    return read_workspace_json(_ws_json(ws))


def _write_raw_ws(ws: str, data: dict) -> None:
    """Rewrite workspace.json preserving every other key.

    Atomic: unique temp name in the same directory + fsync + os.replace (the
    old shared ``workspace.json.tmp`` let two writers crash each other).
    Callers doing read-modify-write hold ``_ws_lock(ws)`` around both halves.
    """
    from config_service import atomic_write_text
    atomic_write_text(_ws_json(ws), json.dumps(data, indent=2))


def _ws_lock(ws: str):
    """Cross-process lock around a workspace.json read-modify-write."""
    from config_service import workspace_json_lock
    return workspace_json_lock(_ws_json(ws))


# --------------------------------------------------------------------------
# Run lock: one generating process per workspace (review 04 P1-2)
# --------------------------------------------------------------------------

_RUN_LOCK_NAME = ".epr_run.lock"
_RUN_LOCK_DEPTH: dict[str, int] = {}


def run_lock_path(ws: str) -> Path:
    return Path(ws) / _RUN_LOCK_NAME


def run_lock_holder(ws: str) -> Optional[dict]:
    """Who is generating in this workspace right now (None = nobody).

    A lock left by a dead process on this host counts as free (stale).  The
    UI can call this on open to warn "a run is already in progress".
    """
    from config_service import lock_holder
    return lock_holder(run_lock_path(ws))


class run_lock(object):
    """Hold the workspace run lock for the duration of a generate/default run.

    Re-entrant within one process (default_run_all -> generate -> generate),
    refuses with a clear RuntimeError while ANOTHER live process holds it, and
    silently takes over a stale lock (dead pid on this host).
    """

    def __init__(self, ws: str, task: str = "") -> None:
        self.ws, self.task = str(ws), task
        self.key = str(run_lock_path(ws))

    def __enter__(self):
        from config_service import try_acquire_lock
        depth = _RUN_LOCK_DEPTH.get(self.key, 0)
        if depth:
            _RUN_LOCK_DEPTH[self.key] = depth + 1
            return self
        if not Path(self.ws).is_dir():
            raise FileNotFoundError(f"workspace directory not found: {self.ws}")
        if not try_acquire_lock(self.key, task=self.task):
            holder = run_lock_holder(self.ws) or {}
            started = holder.get("started_at")
            when = (datetime.fromtimestamp(float(started), timezone.utc)
                    .strftime("%Y-%m-%d %H:%M:%SZ") if started else "?")
            raise RuntimeError(
                f"another run is already working in this workspace (pid "
                f"{holder.get('pid')} on {holder.get('host')}, task "
                f"'{holder.get('task')}', since {when}); wait for it to finish. "
                f"If that process is gone, delete {self.key}")
        _RUN_LOCK_DEPTH[self.key] = 1
        return self

    def __exit__(self, *_exc):
        from config_service import release_lock
        depth = _RUN_LOCK_DEPTH.get(self.key, 0) - 1
        if depth > 0:
            _RUN_LOCK_DEPTH[self.key] = depth
            return False
        _RUN_LOCK_DEPTH.pop(self.key, None)
        release_lock(self.key)
        return False


def job_slug(job_id: str) -> str:
    return "whole" if job_id == WHOLE_TRACKLINE else str(job_id)


def _scope_root(ws: str, job: Job, create: bool = False) -> Path:
    """Product base for a scope: survey/ for whole, survey/jobs/<id>/ for a job."""
    survey = Path(_resolver(ws).survey_products())
    base = survey if job.is_whole else survey / "jobs" / job_slug(job.job_id)
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _write_meta(directory: Path, type_key: str, job: Job, characteristic: str,
                primary: str, views: Sequence[str] = (), extra: Optional[dict] = None) -> dict:
    meta = {
        "type_key": type_key,
        "job_id": job.job_id,
        "job_name": job.name,
        "intervals": [iv.as_list() for iv in job.intervals],
        "characteristic": characteristic,
        "created_at": time.time(),
        "primary": str(primary),
        "views": [str(v) for v in views],
    }
    if extra:
        meta.update(extra)
    try:
        Path(directory).mkdir(parents=True, exist_ok=True)
        (Path(directory) / _META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except OSError:
        pass
    return meta


def _read_meta(directory) -> Optional[dict]:
    try:
        return json.loads((Path(directory) / _META).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _meta_job(directory) -> str:
    """Owning job of a product directory: its meta's job_id, else legacy=whole."""
    meta = _read_meta(directory)
    return str(meta.get("job_id") or WHOLE_TRACKLINE) if meta else WHOLE_TRACKLINE


def _instance(type_key: str, job_id: str, primary, characteristic: str,
              views: Sequence[str] = (), created_at: Optional[float] = None) -> ProductInstance:
    ts = created_at if created_at is not None else _mtime(primary)
    vlist = [str(v) for v in views if v and Path(str(v)).exists()]
    if str(primary) not in vlist and Path(str(primary)).exists():
        vlist.insert(0, str(primary))
    return ProductInstance(type_key=type_key, label=_label(ts, characteristic),
                           path=str(primary), created_at=ts, job_id=job_id,
                           view_paths=vlist)


def _fmt_int(n) -> str:
    return f"{int(n):,}"


def _unique_dir(parent: Path, name: str) -> Path:
    """Create and return ``parent/name``, suffixing if it already exists.

    Run directories are named from a whole-second timestamp, so two runs inside
    the same second would otherwise land in — and overwrite — one directory.
    """
    Path(parent).mkdir(parents=True, exist_ok=True)
    candidate = Path(parent) / name
    n = 1
    while candidate.exists():
        n += 1
        candidate = Path(parent) / f"{name}_{n}"
    candidate.mkdir(parents=True)
    return candidate


SUPERSEDED_DIR = "_superseded"


def _utc_stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _supersede(ws: str, paths: Sequence, reason: str, log: Optional[Callable] = None,
               stamp_dir: Optional[Path] = None) -> Optional[Path]:
    """Move existing ``paths`` to ``<ws>/_superseded/<UTC-stamp>/<relpath>``.

    Canonical products are never overwritten in place (review 06 P0-1): the
    old files keep their workspace-relative layout under one stamped folder,
    with ``superseded.json`` recording why.  Paths outside the workspace keep
    their absolute layout under ``_abs/``.  Returns the stamped folder, or None
    when nothing existed.  Directories are moved whole.
    """
    log = _logger(log)
    ws_path = Path(ws).resolve()
    existing = [Path(p) for p in paths if p and Path(p).exists()]
    if not existing:
        return None
    if stamp_dir is None:
        root = ws_path / SUPERSEDED_DIR
        stamp_dir = root / _utc_stamp()
        n = 1
        while stamp_dir.exists():
            n += 1
            stamp_dir = root / f"{_utc_stamp()}_{n}"
    moved: list[dict] = []
    for src in existing:
        src_abs = src.resolve()
        try:
            rel = src_abs.relative_to(ws_path)
        except ValueError:
            rel = Path("_abs") / str(src_abs).lstrip("/")
        dest = stamp_dir / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src_abs), str(dest))
        moved.append({"from": str(src_abs), "to": str(dest)})
    record_path = stamp_dir / "superseded.json"
    record = {"reason": reason, "superseded_at_utc": _utc_stamp(), "moved": []}
    try:
        record = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    record.setdefault("moved", []).extend(moved)
    record_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    log(f"  {LOG_WARN}superseded {len(moved)} existing file(s) -> {stamp_dir} ({reason})")
    return stamp_dir


def _restore_superseded(stamp_dir: Optional[Path], log: Optional[Callable] = None,
                        new_paths: Sequence = ()) -> None:
    """Roll a failed replacement back: move the partial NEW outputs aside into
    ``<stamp>/_failed_partial/`` and put the superseded originals back."""
    log = _logger(log)
    if stamp_dir is None:
        return
    try:
        record = json.loads((Path(stamp_dir) / "superseded.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    failed = Path(stamp_dir) / "_failed_partial"
    for p in new_paths:
        p = Path(p)
        if p.exists():
            dest = failed / p.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(p), str(dest))
    for item in record.get("moved", []):
        src, dest = Path(item["to"]), Path(item["from"])
        if not src.exists():
            continue
        if dest.exists():                       # partial new output in the way
            aside = failed / dest.name
            aside.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dest), str(aside))
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dest))
    record["rolled_back_utc"] = _utc_stamp()
    (Path(stamp_dir) / "superseded.json").write_text(json.dumps(record, indent=2),
                                                     encoding="utf-8")
    log(f"  {LOG_WARN}replacement failed — previous files restored "
        f"(partial new output kept in {failed})")


def _anomaly_outputs(directory: Path) -> list[Path]:
    """Everything a catalog run rewrites in an anomaly directory (files + qgis/)."""
    directory = Path(directory)
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir()
                  if p.is_file() or p.name == "qgis")


def _merged_outputs(ws: str) -> list[Path]:
    merged = Path(_resolver(ws).survey_products()) / "photogrammetry" / "merged"
    return sorted(p for p in merged.iterdir()) if merged.is_dir() else []


def _census_outputs(ws: str) -> list[Path]:
    census = Path(_resolver(ws).survey_products()) / "fauna"
    names = list(FAUNA_PRODUCT_FILES) + [_META, "fauna_run_summary.json"]
    return [census / n for n in names if (census / n).is_file()]


def replacement_preview(ws: str, job: Optional[Job] = None) -> list[str]:
    """Canonical files a Default run on ``job`` would REPLACE (moved to
    ``<ws>/_superseded/<stamp>/`` first).  Empty list = nothing is replaced.

    For the UI's confirmation dialog (review 06 P0-1): whole scope lists the
    fauna census, the anomaly catalog, the merged ortho/DEM and
    SURVEY_REPORT.html; a job lists its own anomaly directory.
    """
    scope = job or whole_job()
    out: list[Path] = []
    if scope.is_whole:
        out += _census_outputs(ws)
        out += _anomaly_outputs(_anomaly_dir(ws, scope))
        out += _merged_outputs(ws)
        report = Path(ws) / "SURVEY_REPORT.html"
        if report.is_file():
            out.append(report)
    else:
        out += _anomaly_outputs(_anomaly_dir(ws, scope))
    return [str(p) for p in out]


def _unique_file(parent: Path, stem: str, suffix: str) -> Path:
    """``parent/stem+suffix``, numbered if taken (same whole-second runs)."""
    Path(parent).mkdir(parents=True, exist_ok=True)
    candidate = Path(parent) / f"{stem}{suffix}"
    n = 1
    while candidate.exists():
        n += 1
        candidate = Path(parent) / f"{stem}_{n}{suffix}"
    return candidate


# --------------------------------------------------------------------------
# Sampling identity: which products a sampling regime multiplies, and which it
# does not
# --------------------------------------------------------------------------
#
# A job is a set of intervals.  Inside one job the PI may run several *sampling
# regimes* ("dynamic 0.25 m", "fixed 1.0 s", …), and each regime yields its own
# interp/frame set — so anything that CONSUMES FRAMES is multiplied by it:
#
#     frame_set          the frames themselves
#     photogrammetry     one mesh/ortho set per frame set
#     fauna_detection    the detector runs over a frame set
#
# Everything else is computed from the 1 Hz nav/sensor table alone.  Re-running
# it under a second sampling regime would produce a byte-identical product, so
# those types are SAMPLING-INDEPENDENT: identity is the job alone, and the tree
# lists exactly one entry per generated product no matter how many sampling
# runs the job carries.
#
#     nav_trackline, depth_raster, sensor_raster, anomaly_detection,
#     spectrum_trackline, anomaly_trackline, survey_report
#
# Concretely: sampling-dependent types put the sampling settings in their
# instance directory name, in their meta, and at the FRONT of their label;
# sampling-independent types never mention sampling anywhere.

SAMPLING_DEPENDENT: frozenset = frozenset({"frame_set", "photogrammetry",
                                           "fauna_detection"})

#: The only sampling settings that change which frames come out.
SAMPLING_KEYS = ("sampling_mode", "spacing_m", "min_frequency_hz")


def is_sampling_dependent(type_key: str) -> bool:
    """True when a second sampling regime yields a genuinely different product."""
    return str(type_key) in SAMPLING_DEPENDENT


def sampling_settings(settings: Optional[dict] = None) -> dict:
    """The sampling triple, defaulted — the identity of a sampling regime."""
    src = dict(settings or {})
    return {key: type(DEFAULTS[key])(src.get(key, DEFAULTS[key]))
            for key in SAMPLING_KEYS}


def sampling_label(settings: Optional[dict] = None) -> str:
    """Human sampling technique: "dynamic 0.25 m" / "fixed 1 s"."""
    s = sampling_settings(settings)
    mode = str(s["sampling_mode"])
    if mode == "fixed":
        hz = float(s["min_frequency_hz"]) or 1.0
        return f"fixed {1.0 / hz:g} s"
    return f"{mode} {float(s['spacing_m']):g} m"


def sampling_slug(settings: Optional[dict] = None) -> str:
    """Filesystem-safe form of ``sampling_label`` ("dynamic_0p25m")."""
    text = sampling_label(settings).replace(".", "p").replace(" ", "_")
    return "".join(c if (c.isalnum() or c == "_") else "" for c in text)


def sampling_matches(a: Optional[dict], b: Optional[dict]) -> bool:
    """Do two settings dicts name the same sampling regime?"""
    left, right = sampling_settings(a), sampling_settings(b)
    if str(left["sampling_mode"]) != str(right["sampling_mode"]):
        return False
    return all(abs(float(left[k]) - float(right[k])) <= 1e-9
               for k in ("spacing_m", "min_frequency_hz"))


def frame_set_sampling(set_dir) -> dict:
    """The sampling regime a frame set on disk was built with."""
    try:
        meta = json.loads((Path(set_dir) / _SET_META).read_text(encoding="utf-8"))
        return sampling_settings(meta.get("settings") or {})
    except (OSError, ValueError):
        return sampling_settings()


# --------------------------------------------------------------------------
# Job store
# --------------------------------------------------------------------------

def _coerce_intervals(intervals) -> list[Interval]:
    """Accept Interval / (t0,t1) / [t0,t1] / {"t0":..,"t1":..} uniformly."""
    out: list[Interval] = []
    for item in intervals or []:
        if isinstance(item, Interval):
            t0, t1 = item.t0, item.t1
        elif isinstance(item, dict):
            t0, t1 = item.get("t0"), item.get("t1")
        else:
            t0, t1 = item[0], item[1]
        t0, t1 = float(t0), float(t1)
        if t1 < t0:
            t0, t1 = t1, t0
        if t1 > t0:
            out.append(Interval(t0, t1))
    out.sort(key=lambda iv: iv.t0)
    # drop exact duplicates (a derived job re-adding one of its base intervals)
    deduped: list[Interval] = []
    for iv in out:
        if not deduped or (iv.t0, iv.t1) != (deduped[-1].t0, deduped[-1].t1):
            deduped.append(iv)
    return deduped


def whole_job() -> Job:
    return Job(job_id=WHOLE_TRACKLINE, name="Whole trackline", intervals=[])


def load_jobs(ws: str) -> list[Job]:
    """Every job in the workspace, WHOLE_TRACKLINE first."""
    jobs = [whole_job()]
    try:
        records = _raw_ws(ws).get("simple_jobs") or []
    except ValueError:
        # Read-only listing: an unreadable workspace.json shows no jobs.  The
        # writers below still refuse (they call _raw_ws directly).
        records = []
    for record in records:
        try:
            jobs.append(Job(
                job_id=str(record["job_id"]),
                name=str(record.get("name") or record["job_id"]),
                intervals=_coerce_intervals(record.get("intervals")),
            ))
        except Exception:                                           # noqa: BLE001
            continue
    return jobs


def get_job(ws: str, job_id: Optional[str]) -> Job:
    if not job_id or job_id == WHOLE_TRACKLINE:
        return whole_job()
    for job in load_jobs(ws):
        if job.job_id == job_id:
            return job
    return whole_job()


def create_job(ws: str, intervals, base_job: Optional[Job] = None) -> Job:
    """Persist and return a NEW job.

    With ``base_job`` the new job carries the base's intervals PLUS the new
    ones; the base job is never modified (spec: never mutate).
    """
    new = _coerce_intervals(intervals)
    if base_job is not None and not base_job.is_whole:
        new = _coerce_intervals(list(base_job.intervals) + new)
    if not new:
        raise ValueError("create_job: no usable intervals")

    with _ws_lock(ws):
        return _create_job_locked(ws, new)


def _create_job_locked(ws: str, new: list) -> Job:
    data = _raw_ws(ws)                    # raises on a corrupt file — never {}
    records = list(data.get("simple_jobs") or [])
    used = set()
    for record in records:
        try:
            used.add(int(str(record.get("job_id", "")).split("_")[-1]))
        except (ValueError, IndexError):
            pass
    # Monotonic ids: a deleted job's number is never handed out again, so a new
    # job can never adopt product directories the old one left behind.
    used.add(int(data.get("simple_jobs_high_water") or 0))
    number = max(used) + 1 if used else 1
    data["simple_jobs_high_water"] = number
    job = Job(job_id=f"job_{number:04d}",
              name=f"Job {number} ({len(new)} interval{'s' if len(new) != 1 else ''})",
              intervals=new)
    records.append(job.to_dict())
    data["simple_jobs"] = records
    _write_raw_ws(ws, data)
    return job


def rename_job(ws: str, job_id: str, name: str) -> Job:
    """Give a job a new display name.  Always allowed (nothing depends on it:
    products are attributed by ``job_id``)."""
    if not job_id or job_id == WHOLE_TRACKLINE:
        raise ValueError("the whole trackline cannot be renamed")
    label = " ".join(str(name).split())
    if not label:
        raise ValueError("a job needs a non-empty name")
    with _ws_lock(ws):
        data = _raw_ws(ws)                # raises on a corrupt file — never {}
        records = list(data.get("simple_jobs") or [])
        for record in records:
            if str(record.get("job_id")) == str(job_id):
                record["name"] = label
                data["simple_jobs"] = records
                _write_raw_ws(ws, data)
                break
        else:
            raise ValueError(f"no such job: {job_id}")
    return get_job(ws, job_id)


def job_product_counts(ws: str, job: Job) -> dict[str, int]:
    """{type_key: n} for the types that actually have products in this scope."""
    return {key: len(items) for key, items in discover_all(ws, job).items() if items}


def delete_job(ws: str, job_id: str, force: bool = False) -> dict[str, int]:
    """Forget a job.  Refuses while it still owns products (they would become
    unreachable: discovery attributes every product to a job_id).

    Returns the product counts that were found (empty on a plain delete).  Pass
    ``force`` only when the caller has already dealt with the products.
    """
    if not job_id or job_id == WHOLE_TRACKLINE:
        raise ValueError("the whole trackline cannot be deleted")
    _raw_ws(ws)          # a corrupt workspace.json refuses here, by name
    job = get_job(ws, job_id)
    if job.is_whole:
        raise ValueError(f"no such job: {job_id}")
    counts = job_product_counts(ws, job)
    if counts and not force:
        raise RuntimeError(
            f"{job.name} still has products ("
            + ", ".join(f"{k}×{v}" for k, v in sorted(counts.items()))
            + ") — delete or move them first, or rename the job instead")
    with _ws_lock(ws):
        data = _raw_ws(ws)                # raises on a corrupt file — never {}
        records = [r for r in (data.get("simple_jobs") or [])
                   if str(r.get("job_id")) != str(job_id)]
        data["simple_jobs"] = records
        _write_raw_ws(ws, data)
    return counts


# --------------------------------------------------------------------------
# interp_full.csv + the trackline
# --------------------------------------------------------------------------

def interp_path(ws: str) -> Path:
    return Path(_resolver(ws).interp_full())


def _nonempty(path) -> bool:
    try:
        return Path(path).is_file() and Path(path).stat().st_size > 0
    except OSError:
        return False


#: Bump when the interp build's semantics change.  Part of the fingerprint of a
#: NEWLY built table; a legacy table (no meta) is adopted, not rebuilt.
INTERP_SCHEMA = "interp-v2 (nav-span grid, NaN outside source coverage)"
_INTERP_INPUT_KEYS = ("navigation_file", "sensor_files", "depth_source", "speed_source")
#: Keys that are DERIVED from the files (their parsed span), not configuration.
_VOLATILE_KEYS = ("start_time", "end_time")


def interp_meta_path(ws: str) -> Path:
    """``interp_full.meta.json`` beside interp_full.csv (fingerprint + shape)."""
    return interp_path(ws).with_name("interp_full.meta.json")


def _strip_volatile(value):
    if isinstance(value, dict):
        return {k: _strip_volatile(v) for k, v in value.items() if k not in _VOLATILE_KEYS}
    if isinstance(value, list):
        return [_strip_volatile(v) for v in value]
    return value


def _csv_paths(value, out: list) -> list:
    if isinstance(value, dict):
        for k, v in value.items():
            if k == "csv_path" and v:
                out.append(str(v))
            else:
                _csv_paths(v, out)
    elif isinstance(value, list):
        for v in value:
            _csv_paths(v, out)
    return out


def interp_fingerprint(ws: str) -> dict:
    """Identity of interp_full.csv's INPUTS.

    The nav/sensor/depth/speed configuration as saved in workspace.json (minus
    the derived start/end spans; channel names, columns and ``time_delay_s``
    included), plus every referenced CSV's size and mtime, plus the build
    schema.  Any change means the table on disk no longer describes the inputs
    (review 10 P0-1 / 06 P1-1).
    """
    import hashlib
    data = _raw_ws(ws)
    config = {k: _strip_volatile(data.get(k)) for k in _INTERP_INPUT_KEYS}
    base = _ws_json(ws).parent
    files: dict[str, dict] = {}
    for stored in sorted(set(_csv_paths(config, []))):
        path = Path(stored) if Path(stored).is_absolute() else (base / stored)
        try:
            st = path.stat()
            files[stored] = {"size": int(st.st_size), "mtime_ns": int(st.st_mtime_ns)}
        except OSError:
            files[stored] = {"missing": True}
    body = {"schema": INTERP_SCHEMA, "config": config, "files": files}
    digest = hashlib.sha256(json.dumps(body, sort_keys=True, default=str)
                            .encode("utf-8")).hexdigest()
    return {"fingerprint": digest, "schema": INTERP_SCHEMA, "config": config,
            "files": files}


def _read_interp_meta(ws: str) -> Optional[dict]:
    try:
        return json.loads(interp_meta_path(ws).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _write_interp_meta(ws: str, fp: dict, adopted: bool = False) -> None:
    target = interp_path(ws)
    meta = dict(fp)
    try:
        st = target.stat()
        meta["interp_size"] = int(st.st_size)
        meta["interp_mtime_ns"] = int(st.st_mtime_ns)
    except OSError:
        pass
    meta["written_at"] = time.time()
    meta["adopted_legacy"] = bool(adopted)
    try:
        from config_service import atomic_write_text
        atomic_write_text(interp_meta_path(ws), json.dumps(meta, indent=2, default=str))
    except OSError:
        pass


def interp_staleness(ws: str) -> tuple[bool, str]:
    """(stale?, why) for the interp_full.csv on disk.

    Stale when its recorded input fingerprint differs from the current inputs,
    or its size differs from what was written (truncated/edited).  A legacy
    table with no fingerprint is stale only if an input file is NEWER than it;
    otherwise it is adopted as-is.
    """
    target = interp_path(ws)
    if not _nonempty(target):
        return True, "interp_full.csv is missing"
    current = interp_fingerprint(ws)
    meta = _read_interp_meta(ws)
    if meta is None:
        try:
            built = target.stat().st_mtime_ns
        except OSError:
            return True, "interp_full.csv unreadable"
        newer = [name for name, st in current["files"].items()
                 if not st.get("missing") and int(st.get("mtime_ns") or 0) > built]
        if newer:
            return True, ("an input changed after it was built (no fingerprint on file): "
                          + ", ".join(Path(n).name for n in newer))
        return False, "legacy (no fingerprint) — adopted"
    try:
        if int(meta.get("interp_size", -1)) != int(target.stat().st_size):
            return True, (f"interp_full.csv is {target.stat().st_size:,} bytes but was "
                          f"written as {int(meta.get('interp_size', -1)):,} (truncated or edited)")
    except OSError:
        return True, "interp_full.csv unreadable"
    if meta.get("fingerprint") == current["fingerprint"]:
        return False, "fingerprint matches"
    reasons: list[str] = []
    if meta.get("schema") != current["schema"]:
        reasons.append(f"build schema {meta.get('schema')!r} -> {current['schema']!r}")
    old_cfg, new_cfg = meta.get("config") or {}, current["config"]
    for key in _INTERP_INPUT_KEYS:
        if json.dumps(old_cfg.get(key), sort_keys=True, default=str) != \
                json.dumps(new_cfg.get(key), sort_keys=True, default=str):
            reasons.append(f"{key} configuration changed")
    old_files, new_files = meta.get("files") or {}, current["files"]
    for name in sorted(set(old_files) | set(new_files)):
        if old_files.get(name) != new_files.get(name):
            what = ("added" if name not in old_files else
                    "no longer used" if name not in new_files else "modified (size/mtime)")
            reasons.append(f"{Path(name).name} {what}")
    return True, "; ".join(reasons) or "input fingerprint changed"


def ensure_interp(ws: str, log_fn: Optional[Callable[[str], None]] = None) -> Path:
    """Path to interp_full.csv, (re)building it through the real pipeline when
    it is absent OR stale (inputs changed since it was built — the backstop to
    the UI invalidating on import)."""
    log = _logger(log_fn)
    target = interp_path(ws)
    if _nonempty(target):
        stale, reason = interp_staleness(ws)
        if not stale:
            if _read_interp_meta(ws) is None:
                _write_interp_meta(ws, interp_fingerprint(ws), adopted=True)
                log(f"  {LOG_WARN}interp_full.csv predates input fingerprinting — "
                    "adopted as-is (rows outside the navigation span are ignored on "
                    "read; delete it to rebuild a clean table)")
            return target
        log(f"{LOG_WARN}interp_full.csv is stale — {reason}. Rebuilding.")
        moved = None
        try:
            moved = _supersede(ws, [target, interp_meta_path(ws)],
                               f"interp_full.csv rebuilt: {reason}", log)
            if moved:
                log(f"  previous table kept in {moved}")
        except Exception as exc:                                    # noqa: BLE001
            log(f"  {LOG_WARN}could not keep a copy of the old table: {one_line(exc)}")
        try:
            return _build_interp(ws, target, log)
        except BaseException:
            # A failed rebuild must not leave the dive with NO table: put the
            # stale one back (it stays stale, so the next run retries).
            if moved is not None and not _nonempty(target):
                _restore_superseded(moved, log)
            raise
    log(f"interp_full.csv missing — building via pipeline ({target})")
    return _build_interp(ws, target, log)


def _build_interp(ws: str, target: Path, log: Callable) -> Path:
    from pipeline_service import PipelineService, PipelineConfig
    fingerprint = interp_fingerprint(ws)          # BEFORE reading: inputs as used
    data = _ws_data(ws)
    inputs = Path(_resolver(ws).inputs_dir(create=True))
    cfg = PipelineConfig(
        video_directory=Path(data.get("video_directory") or ws),
        output_directory=inputs,
        job_id=0,
        video_filename_time_format=data.get("filename_datetime_format", ""),
        videos=[],
        selected_intervals=[],
        navigation_file=data.get("navigation_file"),
        sensor_files=data.get("sensor_files") or [],
        depth_source=data.get("depth_source"),
        speed_source=data.get("speed_source"),
        selected_steps=["build_full_interp"],
        workspace_directory=str(inputs),
        log_callback=lambda m: log("  " + str(m)),
    )
    PipelineService(log_fn=lambda m: log("  " + str(m))).run(cfg)
    if not _nonempty(target):
        raise RuntimeError(f"interp_full.csv was not produced at {target}")
    _write_interp_meta(ws, fingerprint)
    log(f"  interp_full.csv built: {target}")
    return target


# ---- read-side guards for interp tables built before the nav-span fix -------
#
# Tables built by the old union-of-sources grid carry hours of rows whose
# position is the first/last renav fix held constant (06 P1-2).  They are NOT
# rewritten; every reader here clips them to the positional source's real span.

#: A leading/trailing run of IDENTICAL positions at least this long (rows) is
#: treated as edge-hold padding when no stored nav span is available.
HELD_EDGE_MIN_ROWS = 60


def _stored_time(value) -> Optional[float]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is not None:
        return float(calendar.timegm(dt.utctimetuple())) + dt.microsecond / 1e6
    return unix(dt) + dt.microsecond / 1e6


def nav_span(ws: str) -> Optional[tuple[float, float]]:
    """Positional (lat ∩ lon) source span as recorded in workspace.json.

    ``navigation_file.start_time/end_time`` is the UNION with the altimeter and
    is deliberately not used.  None when unknown.
    """
    try:
        nav = (_raw_ws(ws).get("navigation_file") or {})
    except ValueError:
        return None
    starts, ends = [], []
    for key in ("latitude_source", "longitude_source"):
        src = nav.get(key) or {}
        a, b = _stored_time(src.get("start_time")), _stored_time(src.get("end_time"))
        if a is None or b is None:
            return None
        starts.append(a)
        ends.append(b)
    if not starts or max(starts) >= min(ends):
        return None
    return max(starts), min(ends)


def _held_edge_bounds(df: pd.DataFrame) -> tuple[int, int]:
    """[i0, i1) row range after dropping leading/trailing constant-position
    runs of >= HELD_EDGE_MIN_ROWS rows (time-sorted frame)."""
    cols = [c for c in (("lat", "lon"), ("easting", "northing"))
            if c[0] in df.columns and c[1] in df.columns]
    n = len(df)
    if not cols or n < 2:
        return 0, n
    a = df[cols[0][0]].to_numpy(dtype=float)
    b = df[cols[0][1]].to_numpy(dtype=float)
    same = (a[1:] == a[:-1]) & (b[1:] == b[:-1])      # row k+1 repeats row k
    i0 = 0
    while i0 < n - 1 and same[i0]:
        i0 += 1
    i1 = n - 1
    while i1 > 0 and same[i1 - 1]:
        i1 -= 1
    lead = i0            # rows 0..i0-1 repeat row i0's position
    trail = n - 1 - i1   # rows i1+1..n-1 repeat row i1's position
    start = i0 if lead >= HELD_EDGE_MIN_ROWS else 0
    stop = i1 + 1 if trail >= HELD_EDGE_MIN_ROWS else n
    return start, max(start, stop)


def clip_to_nav(ws: str, df: pd.DataFrame, log: Optional[Callable] = None) -> pd.DataFrame:
    """Rows of a time-sorted interp frame that carry a REAL position.

    Clips to the stored positional span (±1 s), then drops any remaining
    leading/trailing edge-hold run.  A table built after the fix passes
    through unchanged.
    """
    if df.empty or "unix_time" not in df.columns:
        return df
    n0 = len(df)
    span = nav_span(ws)
    if span is not None:
        t = df["unix_time"].to_numpy(dtype=float)
        keep = (t >= span[0] - 1.0) & (t <= span[1] + 1.0)
        if keep.any():
            df = df[keep]
    i0, i1 = _held_edge_bounds(df)
    if (i0, i1) != (0, len(df)):
        df = df.iloc[i0:i1]
    if log is not None and len(df) != n0:
        log(f"  {LOG_WARN}ignored {n0 - len(df):,} interp row(s) outside the "
            "navigation span (edge-held positions from an older interp build)")
    return df


def track_polyline(ws: str) -> np.ndarray:
    """Nx3 array of [unix_time, easting, northing], time-ordered.

    Builds interp_full.csv through the pipeline when it is missing, so the UI
    can draw a trackline for a freshly imported dive.  Rows without a real
    position (NaN, or edge-held padding in an older table) are excluded.
    """
    path = ensure_interp(ws)
    wanted = ("unix_time", "easting", "northing", "lat", "lon")
    df = pd.read_csv(path, usecols=lambda c: c in wanted)
    for column in ("unix_time", "easting", "northing"):
        if column not in df.columns:
            raise ValueError(f"{path} lacks a '{column}' column")
    df = df.dropna(subset=["unix_time", "easting", "northing"]).sort_values(
        "unix_time", kind="stable").reset_index(drop=True)
    df = clip_to_nav(ws, df)
    return df[["unix_time", "easting", "northing"]].to_numpy(dtype=float)


def _interp_df(ws: str, columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """interp_full.csv as a time-sorted frame, clipped to the navigation span."""
    path = ensure_interp(ws)
    if columns:
        wanted = set(columns) | {"unix_time", "lat", "lon", "easting", "northing"}
        df = pd.read_csv(path, usecols=lambda c: c in wanted)
    else:
        df = pd.read_csv(path)
    df = df.sort_values("unix_time", kind="stable").reset_index(drop=True)
    df = clip_to_nav(ws, df).reset_index(drop=True)
    if columns:
        df = df[[c for c in df.columns if c in set(columns)]]
    return df


def job_intervals(ws: str, job: Job) -> list[Interval]:
    """The job's intervals, or one interval spanning the whole dive."""
    if job.intervals:
        return list(job.intervals)
    track = track_polyline(ws)
    if not len(track):
        raise RuntimeError("empty trackline — cannot determine dive extent")
    return [Interval(float(track[0, 0]), float(track[-1, 0]))]


def _mask_intervals(times: np.ndarray, intervals: Sequence[Interval]) -> np.ndarray:
    times = np.asarray(times, dtype=float)
    if not intervals:
        return np.ones(times.shape, dtype=bool)
    mask = np.zeros(times.shape, dtype=bool)
    for iv in intervals:
        mask |= (times >= iv.t0) & (times <= iv.t1)
    return mask


def job_interp_csv(ws: str, job: Job, log_fn: Optional[Callable] = None) -> Path:
    """interp CSV restricted to a job's intervals (interp_full itself for whole).

    Written as ``<scope>/interp_job.csv``; services that take an interp path
    (rasters, reports) then see only the job's rows.
    """
    if job.is_whole:
        return ensure_interp(ws)
    log = _logger(log_fn)
    df = _interp_df(ws)
    keep = df[_mask_intervals(df["unix_time"].to_numpy(dtype=float), job.intervals)]
    out = _scope_root(ws, job, create=True) / "interp_job.csv"
    keep.to_csv(out, index=False)
    log(f"  job interp: {len(keep):,}/{len(df):,} rows -> {out}")
    if keep.empty:
        raise RuntimeError(f"{job.name}: no interp rows inside the job's intervals")
    return out


def sensor_channels(ws: str) -> list[str]:
    """Sensor channel display names for this workspace (config first, CSV second)."""
    names: list[str] = []
    try:
        for sensor_file in _ws_data(ws).get("sensor_files") or []:
            for channel in getattr(sensor_file, "channels", []) or []:
                name = getattr(channel, "display_name", "") or getattr(channel, "source_column", "")
                if name and name not in names:
                    names.append(name)
    except Exception:                                               # noqa: BLE001
        pass
    if not names:
        try:
            from output_service import sensor_channels_from_csv
            names = [c for c in sensor_channels_from_csv(str(interp_path(ws)))]
        except Exception:                                           # noqa: BLE001
            names = []
    return names or list(_FALLBACK_CHANNELS)


# --------------------------------------------------------------------------
# Deterministic sampling grid + recycling maths
# --------------------------------------------------------------------------

def sampling_grid(ws: str, spacing_m: float = None, min_frequency_hz: float = None) -> np.ndarray:
    """Deterministic frame sample times for the WHOLE dive.

    Sample k lands where the cumulative along-track distance (from the dive's
    first nav fix) first reaches ``k * spacing_m``; a minimum-frequency floor
    inserts clock-spaced samples inside long stationary gaps.  Because the grid
    is anchored at the dive start and never at the request, any interval's
    samples are a strict subset of the whole-dive grid — which is what makes
    frame recycling exact rather than approximate.
    """
    spacing = float(DEFAULTS["spacing_m"] if spacing_m is None else spacing_m)
    floor_hz = float(DEFAULTS["min_frequency_hz"] if min_frequency_hz is None
                     else min_frequency_hz)
    track = track_polyline(ws)
    if len(track) < 2:
        return np.zeros(0, dtype=float)
    times = track[:, 0]
    step = np.hypot(np.diff(track[:, 1]), np.diff(track[:, 2]))
    distance = np.concatenate([[0.0], np.cumsum(step)])
    # strictly increasing distance axis for np.interp (stationary spans repeat)
    distance = np.maximum.accumulate(distance)
    if spacing <= 0 or distance[-1] <= 0:
        grid = times.copy()
    else:
        targets = np.arange(0.0, distance[-1] + spacing * 0.5, spacing)
        keep = np.concatenate([[True], np.diff(distance) > 0])
        grid = np.interp(targets, distance[keep], times[keep])
    if floor_hz > 0:
        max_gap = 1.0 / floor_hz
        filled = [grid[0]] if len(grid) else []
        for t in grid[1:]:
            previous = filled[-1]
            if t - previous > max_gap * 1.001:
                n_extra = int(math.floor((t - previous) / max_gap))
                filled.extend(previous + max_gap * k for k in range(1, n_extra + 1))
            filled.append(t)
        grid = np.asarray(filled, dtype=float)
        grid = grid[(grid >= times[0]) & (grid <= times[-1])]
    grid = np.unique(np.round(np.asarray(grid, dtype=float), 3))
    return grid


def coverage_runs(desired: Sequence[float], existing: Sequence[float],
                  tolerance_s: float = 0.5) -> list[tuple[bool, list[float]]]:
    """Partition ``desired`` times into maximal contiguous covered/uncovered runs.

    A desired time is *covered* when some already-extracted frame sits within
    ``tolerance_s`` of it.  Returns ``[(covered?, [times…]), …]`` in time order,
    so each run becomes exactly one segment directory: covered runs are linked
    from the existing frames, uncovered runs go to the video decoder.
    """
    desired = np.sort(np.asarray(list(desired), dtype=float))
    pool = np.sort(np.asarray(list(existing), dtype=float))
    if not len(desired):
        return []
    if len(pool):
        idx = np.searchsorted(pool, desired)
        left = np.clip(idx - 1, 0, len(pool) - 1)
        right = np.clip(idx, 0, len(pool) - 1)
        gap = np.minimum(np.abs(pool[left] - desired), np.abs(pool[right] - desired))
        covered = gap <= float(tolerance_s)
    else:
        covered = np.zeros(desired.shape, dtype=bool)

    runs: list[tuple[bool, list[float]]] = []
    for flag, t in zip(covered.tolist(), desired.tolist()):
        if runs and runs[-1][0] == flag:
            runs[-1][1].append(t)
        else:
            runs.append((flag, [t]))
    return runs


def coalesce_runs(runs: list[tuple[bool, list[float]]], min_run: int
                  ) -> list[tuple[bool, list[float]]]:
    """Absorb runs shorter than ``min_run`` samples into their neighbour.

    Recycling against frames sampled on a DIFFERENT phase (the batch runner
    anchors its distance walk per segment, this UI anchors at the dive start)
    produces long alternating stutters of 1-2 samples — hundreds of "sub-spans"
    that are not really spans at all.  Short runs are therefore merged, with a
    covered neighbour winning: a single uncovered sample wedged between reused
    frames is dropped rather than promoted into its own extraction span (a
    frame within the reuse tolerance already images that ground), while a
    stray covered sample inside a real gap is simply re-extracted.
    """
    if min_run <= 1 or not runs:
        return runs
    work = [(flag, list(times)) for flag, times in runs]
    # Run-length smoothing: flip the SHORTEST offending run into its neighbours
    # (a covered neighbour wins), merge, repeat.  Each flip merges at least one
    # boundary away, so this terminates in at most len(runs) rounds.
    for _ in range(len(work) + 4):
        merged: list[tuple[bool, list[float]]] = []
        for flag, times in work:
            if merged and merged[-1][0] == flag:
                merged[-1][1].extend(times)
            else:
                merged.append((flag, list(times)))
        work = merged
        if len(work) < 2:
            break
        short = [i for i, (_, times) in enumerate(work) if len(times) < min_run]
        if not short:
            break
        i = min(short, key=lambda j: len(work[j][1]))
        neighbours = [work[j][0] for j in (i - 1, i + 1) if 0 <= j < len(work)]
        target = True if True in neighbours else neighbours[0]
        if target == work[i][0]:
            break
        work[i] = (target, work[i][1])
    return work


def altitude_gate_spans(times: Sequence[float], altitudes: Sequence[float],
                        alt_max_m: float = 8.0, min_frames: int = 15
                        ) -> list[list[int]]:
    """Photogrammetry altitude gate.

    Returns index runs (into the given, time-ordered arrays) of contiguous
    frames whose altitude is <= ``alt_max_m``.  Frames above the gate are
    dropped and split the span; surviving spans shorter than ``min_frames`` are
    discarded (too few images to align).  NaN altitude counts as failing the
    gate — an unknown altitude is not evidence of a near-bottom frame.
    """
    alt = np.asarray(list(altitudes), dtype=float)
    ok = np.isfinite(alt) & (alt <= float(alt_max_m))
    spans: list[list[int]] = []
    current: list[int] = []
    for i, good in enumerate(ok.tolist()):
        if good:
            current.append(i)
        elif current:
            spans.append(current)
            current = []
    if current:
        spans.append(current)
    return [s for s in spans if len(s) >= int(min_frames)]


def chunk_sizes(n: int, chunk_max: int = None, target_min: int = None) -> list[int]:
    """Split ``n`` frames into chunk sizes inside the target band.

    Policy: a chunk holds at most ``chunk_max`` (350) frames and should hold at
    least ``target_min`` (250).  So a gated span of ``n <= chunk_max`` stays one
    chunk, and a longer one is cut into ``ceil(n / chunk_max)`` *near-equal*
    parts (sizes differ by at most one frame).  That is the fewest chunks the
    cap allows, so it is also the split whose parts sit highest in — or, when
    ``n`` makes the band unreachable (n = 360 gives 180 + 180; no legal split
    reaches 250), closest to — the band.
    """
    cap = int(chunk_max if chunk_max is not None else DEFAULTS["chunk_size"])
    n = int(n)
    if n <= 0:
        return []
    if cap <= 0:                                   # 0 / negative = "unlimited"
        return [n]
    cap = max(1, cap)
    if n <= cap:
        return [n]
    # The fewest parts the cap allows also makes each part as large as possible,
    # so ``target_min`` never changes the answer — it is what the caller reports
    # a span against.
    parts = -(-n // cap)                           # ceil(n / cap)
    base, extra = divmod(n, parts)
    return [base + 1] * extra + [base] * (parts - extra)


def chunk_photos(photos: Sequence[str], chunk_size: int,
                 target_min: int = None) -> list[list[str]]:
    """``chunk_sizes`` applied to an explicit (altitude-gated) photo list.

    build_chunk_sets takes whole directories and cuts at a fixed size; this
    keeps the same sequential, never-reordered grouping but honours the
    250-350 frames-per-chunk target.
    """
    groups: list[list[str]] = []
    start = 0
    for size in chunk_sizes(len(photos), chunk_size, target_min):
        groups.append(list(photos[start:start + size]))
        start += size
    return groups


# --------------------------------------------------------------------------
# Frame sets: storage, pool scan, extraction
# --------------------------------------------------------------------------

def frame_sets_root(ws: str, create: bool = False) -> Path:
    base = Path(_resolver(ws).survey_products()) / "frame_sets"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _segment_dirs(parent: Path) -> list[Path]:
    return sorted(p for p in Path(parent).glob("segment_*")
                  if p.is_dir() and (p / "interp.csv").is_file())


def _segment_frames(segment_dir: Path) -> pd.DataFrame:
    """[unix_time, frame_filename, path] for frames that really exist on disk."""
    frames_dir = Path(segment_dir) / "frames"
    try:
        present = {p.name for p in frames_dir.iterdir()
                   if p.suffix.lower() in (".jpg", ".jpeg", ".png")}
    except OSError:
        return pd.DataFrame(columns=["unix_time", "frame_filename", "path"])
    if not present:
        return pd.DataFrame(columns=["unix_time", "frame_filename", "path"])
    try:
        df = pd.read_csv(Path(segment_dir) / "interp.csv",
                         usecols=lambda c: c in ("unix_time", "frame_filename"))
    except Exception:                                               # noqa: BLE001
        return pd.DataFrame(columns=["unix_time", "frame_filename", "path"])
    if "frame_filename" not in df.columns or "unix_time" not in df.columns:
        return pd.DataFrame(columns=["unix_time", "frame_filename", "path"])
    df = df.dropna(subset=["unix_time", "frame_filename"])
    df = df[df["frame_filename"].astype(str).isin(present)].copy()
    df["path"] = [str(frames_dir / n) for n in df["frame_filename"].astype(str)]
    df["segment_dir"] = str(segment_dir)
    return df.sort_values("unix_time").reset_index(drop=True)


def scan_frame_pool(ws: str, spacing_m: float, sampling_mode: str = "dynamic",
                    reuse_legacy: bool = True,
                    log_fn: Optional[Callable] = None) -> pd.DataFrame:
    """Every already-extracted frame at compatible sampling settings.

    Sources, in order: this UI's frame sets (matched on set_meta.json settings)
    and — when ``reuse_legacy`` — the batch runner's
    ``survey/photogrammetry/*/segment_*`` dirs, whose interp.csv manifests carry
    the same per-frame unix_time.
    """
    log = _logger(log_fn)
    frames: list[pd.DataFrame] = []
    sources = 0

    for meta_path in sorted(frame_sets_root(ws).glob("*/set_*/" + _SET_META)):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if not _set_complete(meta):
            continue                     # a build in progress / failed: never recycle
        settings = meta.get("settings") or {}
        if str(settings.get("sampling_mode", "dynamic")) != str(sampling_mode):
            continue
        if abs(float(settings.get("spacing_m", -1)) - float(spacing_m)) > 1e-9:
            continue
        for segment_dir in _segment_dirs(meta_path.parent):
            part = _segment_frames(segment_dir)
            if len(part):
                part["source"] = f"set:{meta_path.parent.name}"
                frames.append(part)
                sources += 1

    if reuse_legacy:
        photogrammetry = Path(_resolver(ws).survey_products()) / "photogrammetry"
        for segment_dir in sorted(photogrammetry.glob("*/segment_*")):
            if not (segment_dir / "interp.csv").is_file():
                continue
            part = _segment_frames(segment_dir)
            if len(part):
                part["source"] = f"legacy:{segment_dir.parent.name}"
                frames.append(part)
                sources += 1

    if not frames:
        log("  frame pool: empty (nothing to recycle)")
        return pd.DataFrame(columns=["unix_time", "frame_filename", "path",
                                     "segment_dir", "source"])
    pool = pd.concat(frames, ignore_index=True).sort_values("unix_time")
    pool = pool.drop_duplicates(subset=["frame_filename"]).reset_index(drop=True)
    log(f"  frame pool: {len(pool):,} frames from {sources} segment dir(s)")
    return pool


def _link_or_copy(src: Path, dst: Path, mode: str = "auto") -> str:
    """Materialise a reused frame without re-extracting it.

    Hard link first (the Windows Metashape subprocess can follow those on
    DrvFs/NTFS), then symlink, then a plain copy.  Returns the method used.
    """
    if dst.exists():
        return "present"
    order = {"auto": ("hardlink", "symlink", "copy"),
             "hardlink": ("hardlink", "symlink", "copy"),
             "symlink": ("symlink", "hardlink", "copy"),
             "copy": ("copy",)}.get(str(mode), ("hardlink", "symlink", "copy"))
    for method in order:
        try:
            if method == "hardlink":
                os.link(src, dst)
            elif method == "symlink":
                os.symlink(src, dst)
            else:
                shutil.copy2(src, dst)
            return method
        except (OSError, NotImplementedError):
            continue
    raise OSError(f"could not link or copy {src} -> {dst}")


def _grid_pipeline(plan: dict, log: Callable):
    """PipelineService driven by an EXPLICIT per-interval sample-time plan.

    The stock dynamic sampler anchors its distance walk at each interval's own
    start, so the same ground would get different timestamps in an interval run
    than in a whole-dive run and nothing could ever be recycled.  Overriding
    just the schedule source keeps every other line of the real extraction path
    (video mapping, JPEG writing, interp.csv construction) untouched.
    """
    from pipeline_service import PipelineService

    class _Pipeline(PipelineService):
        def _get_dynamic_sample_times(self, nav_sources, interval, config):   # noqa: D401
            key = (interval.start_time, interval.end_time)
            return sorted(plan.get(key, []))

    return _Pipeline(log_fn=lambda m: log("    " + str(m)))


def _frame_set_characteristic(meta: dict) -> str:
    # The sampling technique leads the label: a job may hold several frame sets
    # that differ ONLY in sampling, and that is what tells them apart.
    bits = [sampling_label(meta.get("settings") or {})]
    n_frames = int(meta.get("n_frames") or 0)
    if n_frames:
        reused = int(meta.get("n_reused") or 0)
        bits.append(f"{_fmt_int(n_frames)} frames"
                    + (f" ({_fmt_int(reused)} reused)" if reused else ""))
    n_int = len(meta.get("intervals") or [])
    if n_int:
        bits.append(f"{n_int} interval{'s' if n_int != 1 else ''}")
    return " · ".join(bits)


SET_STATUS_IN_PROGRESS = "in_progress"
SET_STATUS_FAILED = "failed"
SET_STATUS_COMPLETE = "complete"


def _set_complete(meta: dict) -> bool:
    """A frame set is usable only once its build finished.  Sets written
    before status markers existed carry no status and count as complete."""
    return str(meta.get("status") or SET_STATUS_COMPLETE) == SET_STATUS_COMPLETE


def _write_set_meta(set_dir: Path, meta: dict) -> None:
    from config_service import atomic_write_text
    atomic_write_text(Path(set_dir) / _SET_META, json.dumps(meta, indent=2))


def _discover_frame_set(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    for meta_path in sorted(frame_sets_root(ws).glob("*/set_*/" + _SET_META)):
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if str(meta.get("job_id") or WHOLE_TRACKLINE) != job.job_id:
            continue
        if not _set_complete(meta):
            continue                     # partial/failed set: not a product
        set_dir = meta_path.parent
        segments = _segment_dirs(set_dir)
        out.append(_instance(
            "frame_set", job.job_id, set_dir, _frame_set_characteristic(meta),
            views=[str(s / "interp.csv") for s in segments],
            created_at=float(meta.get("created_at") or _mtime(meta_path))))
    # Legacy batch frame sets (no meta) belong to the whole trackline.
    if job.is_whole:
        photogrammetry = Path(_resolver(ws).survey_products()) / "photogrammetry"
        for segment_dir in sorted(photogrammetry.glob("*/segment_*")):
            interp = segment_dir / "interp.csv"
            if not interp.is_file():
                continue
            try:
                n = sum(1 for _ in (segment_dir / "frames").iterdir())
            except OSError:
                n = 0
            out.append(_instance(
                "frame_set", WHOLE_TRACKLINE, segment_dir,
                f"legacy {segment_dir.parent.name} · {_fmt_int(n)} frames",
                views=[str(interp)]))
    return out


def _generate_frame_set(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    spacing = float(settings.get("spacing_m", DEFAULTS["spacing_m"]))
    mode = str(settings.get("sampling_mode", DEFAULTS["sampling_mode"]))
    floor_hz = float(settings.get("min_frequency_hz", DEFAULTS["min_frequency_hz"]))
    tolerance = float(settings.get("reuse_tolerance_s", DEFAULTS["reuse_tolerance_s"]))
    reuse_legacy = bool(settings.get("reuse_legacy", DEFAULTS["reuse_legacy"]))
    reuse_mode = str(settings.get("reuse_mode", "auto"))

    ensure_interp(ws, log)
    data = _ws_data(ws)
    intervals = job_intervals(ws, job)
    grid = sampling_grid(ws, spacing, floor_hz)
    desired = grid[_mask_intervals(grid, intervals)]
    # Never request a frame outside the span that has REAL positions: the
    # clock floor used to add a sample every 10 s across hours of edge-held
    # nav (review 06 P1-2 #3 — 351 such frames on J1758_rehearsal).
    track = track_polyline(ws)
    if len(track):
        inside = (desired >= track[0, 0]) & (desired <= track[-1, 0])
        if not inside.all():
            log(f"  {LOG_WARN}{int((~inside).sum()):,} requested sample time(s) outside "
                "the navigation span dropped")
        desired = desired[inside]
    log(f"  grid: {len(grid):,} whole-dive samples @ {spacing:g} m; "
        f"{len(desired):,} inside {len(intervals)} interval(s)")
    if not len(desired):
        raise RuntimeError("no sample times inside the requested intervals")

    pool = scan_frame_pool(ws, spacing, mode, reuse_legacy, log)
    raw_runs = coverage_runs(desired, pool["unix_time"].to_numpy(dtype=float)
                             if len(pool) else [], tolerance)
    min_run = int(settings.get("min_run_samples", DEFAULTS["min_run_samples"]))
    runs = coalesce_runs(raw_runs, min_run)
    n_covered = sum(len(times) for covered, times in runs if covered)
    log(f"  recycling: {_fmt_int(n_covered)} of {_fmt_int(len(desired))} samples "
        f"already on disk — {len(runs)} span(s) "
        f"(coalesced from {len(raw_runs)} at min_run={min_run})")

    created = time.time()
    set_dir = _unique_dir(frame_sets_root(ws, create=True) / job_slug(job.job_id),
                          f"set_{_stamp(created)}")
    # Partial-write marker (review 04 P1-4): until the final meta lands this
    # set is "in_progress" — never recycled, never discovered as a product.
    base_meta = {
        "status": SET_STATUS_IN_PROGRESS,
        "job_id": job.job_id,
        "job_name": job.name,
        "created_at": created,
        "intervals": [iv.as_list() for iv in intervals],
        "settings": {"sampling_mode": mode, "spacing_m": spacing,
                     "min_frequency_hz": floor_hz,
                     "reuse_tolerance_s": tolerance,
                     "reuse_legacy": reuse_legacy},
        "n_requested": int(len(desired)),
    }
    _write_set_meta(set_dir, base_meta)
    try:
        return _fill_frame_set(ws, job, set_dir, base_meta, runs, pool, data,
                               desired, intervals, spacing, floor_hz, tolerance,
                               reuse_mode, created, log)
    except BaseException as exc:
        if set_dir.is_dir():
            try:
                _write_set_meta(set_dir, dict(base_meta, status=SET_STATUS_FAILED,
                                              error=one_line(exc)))
            except OSError:
                pass
        raise


def _fill_frame_set(ws, job, set_dir, base_meta, runs, pool, data, desired,
                    intervals, spacing, floor_hz, tolerance, reuse_mode, created,
                    log) -> ProductInstance:
    from models import SelectedTimeRange
    from pipeline_service import PipelineConfig
    from video_service import VideoService

    # ---- uncovered runs: the real extraction pipeline, explicit schedule ----
    extract_runs = [times for covered, times in runs if not covered]
    n_extracted = 0
    skipped_reason = ""
    link_methods: dict[str, int] = {}
    if extract_runs:
        video_dir = data.get("video_directory") or ""
        videos: list = []
        if video_dir and Path(video_dir).is_dir():
            scanner = VideoService(data.get("filename_datetime_format") or "%Y_%m_%dT%H_%M_%S")
            videos, skipped = scanner.scan_directory(video_dir)
            log(f"  videos: {len(videos)} indexed, {len(skipped)} skipped")
            if not videos:
                skipped_reason = f"no readable videos in {video_dir}"
                log(f"  {LOG_WARN}{skipped_reason} — extraction of uncovered "
                    "spans skipped")
        else:
            skipped_reason = f"video directory unavailable ({video_dir or 'unset'})"
            log(f"  {LOG_WARN}{skipped_reason} — extraction of uncovered "
                "spans skipped")
        if videos:
            plan: dict = {}
            pipeline_intervals: list = []
            for k, times in enumerate(extract_runs, 1):
                start = naive_utc(math.floor(min(times)) - 1)
                end = naive_utc(math.ceil(max(times)) + 1)
                plan[(start, end)] = list(times)
                pipeline_intervals.append(SelectedTimeRange(
                    start_time=start, end_time=end, source=f"simple_{job_slug(job.job_id)}_{k:03d}"))
            cfg = PipelineConfig(
                video_directory=Path(video_dir),
                output_directory=set_dir,
                job_id=0,
                video_filename_time_format=data.get("filename_datetime_format", ""),
                videos=videos,
                selected_intervals=pipeline_intervals,
                navigation_file=data.get("navigation_file"),
                sensor_files=data.get("sensor_files") or [],
                depth_source=data.get("depth_source"),
                speed_source=data.get("speed_source"),
                sampling_mode="dynamic",
                dynamic_target_spacing_m=spacing,
                dynamic_min_frequency_hz=floor_hz,
                sample_images=True,
                selected_steps=["extract_frames"],
                workspace_directory=str(ws),
                frame_quality=data.get("frame_quality", "Original"),
                log_callback=lambda m: log("    " + str(m)),
            )
            log(f"  extracting {len(pipeline_intervals)} uncovered span(s), "
                f"{_fmt_int(sum(len(t) for t in extract_runs))} frames")
            _grid_pipeline(plan, log).run(cfg)
            n_extracted = sum(len(_segment_frames(d)) for d in _segment_dirs(set_dir))

    # ---- covered runs: link the existing frames into this set --------------
    n_reused = 0
    for k, (covered, times) in enumerate([r for r in runs if r[0]], 1):
        chosen = _pick_pool_frames(pool, times, tolerance)
        if chosen.empty:
            continue
        seg_name = (f"segment_r{k:02d}_"
                    f"{naive_utc(min(times)):%Y%m%dT%H%M%S}_"
                    f"{naive_utc(max(times)):%Y%m%dT%H%M%S}")
        seg_dir = set_dir / seg_name
        (seg_dir / "frames").mkdir(parents=True, exist_ok=True)
        for src in chosen["path"]:
            src_path = Path(src)
            method = _link_or_copy(src_path, seg_dir / "frames" / src_path.name, reuse_mode)
            link_methods[method] = link_methods.get(method, 0) + 1
        _reuse_interp(chosen, seg_dir / "interp.csv")
        n_reused += len(chosen)
        log(f"  reused {len(chosen):,} frames -> {seg_name}")
    if link_methods:
        log("  reuse method: " + ", ".join(f"{k}×{v}" for k, v in link_methods.items()))

    segments = _segment_dirs(set_dir)
    n_frames = sum(len(_segment_frames(d)) for d in segments)
    if not n_frames:
        # Never leave an empty set on disk: it would be discovered as a product,
        # and resolve_frame_set would then feed it to photogrammetry.
        shutil.rmtree(set_dir, ignore_errors=True)
        detail = skipped_reason or "no frames were reused or extracted"
        raise RuntimeError(f"frame set is empty — {detail}")
    meta = {
        **base_meta,
        "status": SET_STATUS_COMPLETE,
        "n_frames": n_frames,
        "n_reused": n_reused,
        "n_extracted": n_extracted,
        "n_requested": int(len(desired)),
        "reuse_methods": link_methods,
        "segments": [{"dir": d.name, "n_frames": len(_segment_frames(d))} for d in segments],
    }
    meta["characteristic"] = _frame_set_characteristic(meta)
    _write_set_meta(set_dir, meta)
    _write_meta(set_dir, "frame_set", job, meta["characteristic"], set_dir)
    log(f"  frame set: {_fmt_int(n_frames)} frames in {len(segments)} segment(s) -> {set_dir}")
    return _instance("frame_set", job.job_id, set_dir, meta["characteristic"],
                     views=[str(d / "interp.csv") for d in segments], created_at=created)


def _pick_pool_frames(pool: pd.DataFrame, times: Sequence[float],
                      tolerance: float) -> pd.DataFrame:
    """The pool frames that satisfy the given desired times (one per time)."""
    if pool.empty:
        return pool
    pool_times = pool["unix_time"].to_numpy(dtype=float)
    order = np.argsort(pool_times)
    sorted_times = pool_times[order]
    picks: list[int] = []
    for t in times:
        idx = int(np.searchsorted(sorted_times, t))
        best, best_gap = None, None
        for candidate in (idx - 1, idx):
            if 0 <= candidate < len(sorted_times):
                gap = abs(sorted_times[candidate] - t)
                if best_gap is None or gap < best_gap:
                    best, best_gap = candidate, gap
        if best is not None and best_gap is not None and best_gap <= tolerance:
            picks.append(int(order[best]))
    if not picks:
        return pool.iloc[0:0]
    return pool.iloc[sorted(set(picks))].sort_values("unix_time").reset_index(drop=True)


def _reuse_interp(chosen: pd.DataFrame, out_csv: Path) -> None:
    """Rebuild an interp.csv for reused frames from their source manifests."""
    parts: list[pd.DataFrame] = []
    for segment_dir, group in chosen.groupby("segment_dir"):
        try:
            src = pd.read_csv(Path(segment_dir) / "interp.csv")
        except Exception:                                           # noqa: BLE001
            continue
        if "frame_filename" not in src.columns:
            continue
        parts.append(src[src["frame_filename"].astype(str).isin(
            set(group["frame_filename"].astype(str)))])
    if parts:
        merged = pd.concat(parts, ignore_index=True)
        merged = (merged.drop_duplicates(subset=["frame_filename"])
                  .sort_values("unix_time").reset_index(drop=True))
    else:
        merged = chosen[["frame_filename", "unix_time"]].copy()
    merged.to_csv(out_csv, index=False)


# --------------------------------------------------------------------------
# Photogrammetry
# --------------------------------------------------------------------------

def _photogrammetry_root(ws: str, job: Job, create: bool = False) -> Path:
    base = _scope_root(ws, job) / "photogrammetry"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _CHUNK_VIEWS(chunk: Path) -> tuple[Path, ...]:
    """One chunk's viewable artefacts, cheapest-to-open first.

    Order matters: the viewer builds tabs lazily and opens the first one, so a
    small PNG leads and the 2 GB dense cloud trails.
    """
    return (chunk / "preview_ortho.png", chunk / "orthomosaic.tif",
            chunk / "dem.tif", chunk / "report.pdf",
            chunk / "sparse.ply", chunk / "mesh.obj", chunk / "dense.ply")


def _discover_photogrammetry(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    root = _photogrammetry_root(ws, job)
    if not root.is_dir():
        return out
    for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if run_dir.name in ("merged", "ortho_gallery"):
            continue
        chunks = sorted(p for p in run_dir.glob("chunk_*") if p.is_dir())
        if not chunks:
            continue
        owner = _meta_job(run_dir)
        if owner != job.job_id:
            continue
        meta = _read_meta(run_dir) or {}
        # run_status.json (photogrammetry_service): "running" before Metashape
        # starts, then "ok" | "partial" | "failed".  Crashed, still-running and
        # all-chunks-failed runs are not products (reviews 04 P1-5, 06 P1-6).
        status: dict = {}
        try:
            status = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        st = status.get("status")
        if st == "failed" or (st == "running" and not meta):
            continue
        # a run with neither outputs nor a meta is a crash left behind
        if not meta and not any((c / "orthomosaic.tif").is_file() or (c / "dem.tif").is_file()
                                for c in chunks):
            continue
        views: list[str] = []
        for chunk in chunks:
            views.extend(str(p) for p in _CHUNK_VIEWS(chunk) if p.exists())
        characteristic = meta.get("characteristic") or (
            f"{run_dir.name} · {len(chunks)} chunk{'s' if len(chunks) != 1 else ''}")
        if st == "partial":
            characteristic += (f"  [{status.get('n_ok')}/{len(chunks)} chunks "
                               "reconstructed]")
        primary = run_dir / "project.psx"
        out.append(_instance("photogrammetry", owner,
                             primary if primary.exists() else run_dir,
                             characteristic, views=views,
                             created_at=float(meta.get("created_at") or _mtime(run_dir))))
    return out


def resolve_frame_set(ws: str, job: Job, settings: dict, log: Callable) -> Path:
    """The frame set feeding a frame-consuming run (generating one if needed).

    Sampling is part of the requested product's identity, so only frame sets
    built with the SAME sampling regime qualify: asking for "fixed 1 s"
    photogrammetry inside a job that already has a "dynamic 0.25 m" frame set
    mints a second frame set rather than silently reusing the first.
    """
    requested = str(settings.get("frame_set") or "latest")
    if requested not in ("latest", "", "auto"):
        candidate = Path(requested)
        if candidate.is_dir():
            return candidate
        raise RuntimeError(f"frame set '{requested}' is not a directory")
    wanted = sampling_settings(settings)
    # Newest first, and only sets that really hold frames — an empty set would
    # otherwise shadow a usable older one.
    sets = [Path(p.path) for p in _discover_frame_set(ws, job)
            if (Path(p.path) / _SET_META).is_file()
            and any(len(_segment_frames(d)) for d in _segment_dirs(Path(p.path)))]
    matching = [s for s in sets if sampling_matches(frame_set_sampling(s), wanted)]
    if matching:
        log(f"  frame set [{sampling_label(wanted)}]: {matching[0].name}")
        return matching[0]
    if sets:
        log(f"  {LOG_WARN}{len(sets)} frame set(s) exist for this job but none "
            f"sampled '{sampling_label(wanted)}' — building one")
    else:
        log("  no frame set for this scope — generating one first")
    return Path(_generate_frame_set(ws, job, dict(DEFAULTS, **settings), log).path)


def plan_photogrammetry_chunks(set_dir, settings: dict,
                               log: Optional[Callable] = None) -> tuple[list[dict], int]:
    """Altitude gate + 250-350 frames-per-chunk planning over a frame set.

    Returns ``(specs, n_gated_out)`` where each spec is
    ``{"photos": [...], "label": str, "nav_csv": str}`` — the same shape
    build_chunk_sets produces, except the photo list has been altitude-gated
    per frame (which build_chunk_sets, taking whole directories, cannot express)
    and the cut follows the chunk band rather than a fixed size.
    """
    log = _logger(log)
    chunk_size = int(settings.get("chunk_size", DEFAULTS["chunk_size"]) or 0)
    # 350 frames/chunk is a policy ceiling, not merely a default
    chunk_size = min(chunk_size, int(DEFAULTS["chunk_size"])) if chunk_size > 0 \
        else int(DEFAULTS["chunk_size"])
    target_min = int(settings.get("chunk_target_min", DEFAULTS["chunk_target_min"]))
    target_min = min(target_min, chunk_size)
    alt_max = float(settings.get("alt_max_m", DEFAULTS["alt_max_m"]))
    min_frames = int(settings.get("min_chunk_frames", DEFAULTS["min_chunk_frames"]))

    specs: list[dict] = []
    gated_out = 0
    for i, segment_dir in enumerate(_segment_dirs(set_dir), 1):
        frames = _segment_frames(segment_dir)
        if frames.empty:
            continue
        try:
            alt_src = pd.read_csv(Path(segment_dir) / "interp.csv",
                                  usecols=lambda c: c in ("frame_filename", "alt"))
        except Exception:                                           # noqa: BLE001
            alt_src = pd.DataFrame(columns=["frame_filename", "alt"])
        names = frames["frame_filename"].astype(str).tolist()
        paths = frames["path"].tolist()
        if "alt" in alt_src.columns:
            lookup = dict(zip(alt_src["frame_filename"].astype(str), alt_src["alt"]))
            altitudes = [lookup.get(n, float("nan")) for n in names]
        else:
            log(f"    {Path(segment_dir).name}: interp.csv has no 'alt' column — "
                "altitude gate cannot be applied, keeping all frames")
            altitudes = [0.0] * len(names)
        spans = altitude_gate_spans(frames["unix_time"].tolist(), altitudes,
                                   alt_max, min_frames)
        kept = sum(len(s) for s in spans)
        gated_out += len(names) - kept
        log(f"    {Path(segment_dir).name}: {len(names)} frames -> {len(spans)} span(s), "
            f"{kept} kept, {len(names) - kept} dropped (alt > {alt_max:g} m "
            f"or span < {min_frames} frames)")
        for j, span in enumerate(spans, 1):
            groups = chunk_photos([paths[k] for k in span], chunk_size, target_min)
            if len(groups) > 1 and min(len(g) for g in groups) < target_min:
                log(f"      span {j}: {len(span)} frames cannot reach the "
                    f"{target_min}-{chunk_size} band under the cap — splitting "
                    f"near-equally into {[len(g) for g in groups]}")
            for part, group in enumerate(groups, 1):
                label = (f"seg{i:02d}span{j:02d}"
                         + (f"_part{part:02d}" if len(groups) > 1 else ""))
                specs.append({"photos": group, "label": label,
                              "nav_csv": str(Path(segment_dir) / "interp.csv")})
    return specs, gated_out


#: Generate-dialog keys that are really Metashape knobs.  The VALUES are never
#: defined here: they come from batch_service.DEFAULT_PHOTO_SETTINGS (the
#: adopted recipe1 block), which stays the single source of truth — this list
#: only says which of them the dialog is allowed to show and override.
PHOTO_SETTING_KEYS = ("align_accuracy", "quality_threshold",
                      "nav_accuracy_h", "nav_accuracy_v",
                      "nav_rotation_accuracy_deg",
                      "dense_quality", "depth_filter",
                      "build_dense", "build_mesh", "build_dem",
                      "build_orthomosaic")


def merge_photo_settings(settings: dict) -> dict:
    """recipe1 defaults, overridden by whatever the Generate dialog exposed.

    Accepts both the flat dialog keys (``dense_quality``) and a nested
    ``photo_settings`` dict, and turns the ``use_fixed_calibration`` checkbox
    into the presence/absence of the pooled fixed calibration block.
    """
    merged = dict(DEFAULTS["photo_settings"])
    merged.update(settings.get("photo_settings") or {})
    for key in PHOTO_SETTING_KEYS:
        if key in settings and settings[key] is not None:
            merged[key] = type(merged[key])(settings[key]) \
                if key in merged and not isinstance(merged.get(key), bool) \
                else settings[key]
    if "use_fixed_calibration" in settings and not bool(settings["use_fixed_calibration"]):
        merged["fixed_calibration"] = None
    return merged


def _photogrammetry_schema(ws: Optional[str] = None) -> list[tuple]:
    """The meaningful knobs, with recipe1 as the default for every one."""
    photo = DEFAULTS["photo_settings"]
    return [
        # Only "dynamic" is offered: sampling_grid + PipelineService implement
        # the distance-walk sampler alone.  sampling_label/sampling_slug already
        # render "fixed 1 s", so the identity machinery is ready the day a
        # fixed-interval sampler lands — but a choice that silently produced
        # dynamic frames would be worse than no choice.
        ("sampling_mode", "Sampling technique", "choice",
         DEFAULTS["sampling_mode"], ["dynamic"]),
        ("spacing_m", "Target spacing (m)", "float", DEFAULTS["spacing_m"], None),
        ("min_frequency_hz", "Minimum frequency (Hz)", "float",
         DEFAULTS["min_frequency_hz"], None),
        ("frame_set", "Frame set (latest = matching sampling)", "str", "latest", None),
        ("alt_max_m", "Altitude gate — drop frames above (m)", "float",
         DEFAULTS["alt_max_m"], None),
        ("chunk_size", "Chunk band — max frames per chunk", "int",
         DEFAULTS["chunk_size"], None),
        ("chunk_target_min", "Chunk band — target min frames", "int",
         DEFAULTS["chunk_target_min"], None),
        ("min_chunk_frames", "Min frames per gated span", "int",
         DEFAULTS["min_chunk_frames"], None),
        ("align_accuracy", "Alignment accuracy", "choice",
         photo.get("align_accuracy", "High"),
         ["Highest", "High", "Medium", "Low", "Lowest"]),
        ("dense_quality", "Dense cloud quality", "choice",
         photo.get("dense_quality", "Low"),
         ["Ultra high", "High", "Medium", "Low", "Lowest"]),
        ("depth_filter", "Depth filter", "choice",
         photo.get("depth_filter", "Moderate"),
         ["Aggressive", "Moderate", "Mild", "Disabled"]),
        ("nav_accuracy_h", "Nav accuracy — horizontal (m)", "float",
         photo.get("nav_accuracy_h", 0.1), None),
        ("nav_accuracy_v", "Nav accuracy — vertical (m)", "float",
         photo.get("nav_accuracy_v", 0.05), None),
        ("nav_rotation_accuracy_deg", "Rotation prior accuracy (deg)", "float",
         photo.get("nav_rotation_accuracy_deg", 30.0), None),
        ("quality_threshold", "Image quality gate (0 = off)", "float",
         photo.get("quality_threshold", 0.0), None),
        ("use_fixed_calibration", "Use the pooled fixed calibration", "bool",
         bool(photo.get("fixed_calibration")), None),
        ("build_dense", "Build dense cloud", "bool",
         bool(photo.get("build_dense", True)), None),
        ("build_mesh", "Build mesh", "bool", bool(photo.get("build_mesh", True)), None),
        ("build_dem", "Build DEM", "bool", bool(photo.get("build_dem", True)), None),
        ("build_orthomosaic", "Build orthomosaic", "bool",
         bool(photo.get("build_orthomosaic", True)), None),
    ]


def _generate_photogrammetry(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    import photogrammetry_service as photogrammetry

    if photogrammetry.metashape_driver() is None:
        raise RuntimeError(photogrammetry.metashape_unavailable_reason()
                           or "Metashape is not available")

    chunk_size = min(int(settings.get("chunk_size", DEFAULTS["chunk_size"]) or 0)
                     or int(DEFAULTS["chunk_size"]), int(DEFAULTS["chunk_size"]))
    alt_max = float(settings.get("alt_max_m", DEFAULTS["alt_max_m"]))
    min_frames = int(settings.get("min_chunk_frames", DEFAULTS["min_chunk_frames"]))

    set_dir = resolve_frame_set(ws, job, settings, log)
    # Identity follows the frames actually used, not merely what was asked for:
    # an explicit frame_set path may carry a different sampling regime.
    sampling = frame_set_sampling(set_dir) if (set_dir / _SET_META).is_file() \
        else sampling_settings(settings)
    log(f"  frames from: {set_dir} [{sampling_label(sampling)}]")
    specs, gated_out = plan_photogrammetry_chunks(set_dir, settings, log)
    if not specs:
        raise RuntimeError(f"no frames survive the altitude gate "
                           f"(alt <= {alt_max:g} m, >= {min_frames} frames per span)")

    # run the real engine: one project, many chunks
    created = time.time()
    # The sampling regime is part of this product's identity, so it is in the
    # directory name as well as the meta and the label.
    run_dir = _unique_dir(_photogrammetry_root(ws, job, create=True),
                          f"run_{_stamp(created)}__{sampling_slug(sampling)}")
    nav_csv = str(ensure_interp(ws, log))          # mirrors batch_service's recipe
    frame_sets = []
    for c, spec in enumerate(specs, 1):
        chunk_dir = run_dir / f"chunk_{c:02d}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        frame_sets.append((spec["photos"], str(chunk_dir), nav_csv, spec["label"]))
    n_photos = sum(len(s["photos"]) for s in specs)
    sizes = [len(s["photos"]) for s in specs]
    log(f"  {len(frame_sets)} chunk(s), {_fmt_int(n_photos)} photos "
        f"({min(sizes)}-{max(sizes)} per chunk, target "
        f"{DEFAULTS['chunk_target_min']}-{chunk_size}) -> {run_dir}")

    photo_settings = merge_photo_settings(settings)
    deviations = {k: v for k, v in photo_settings.items()
                  if DEFAULTS["photo_settings"].get(k) != v}
    if deviations:
        log("  settings deviating from recipe1: "
            + ", ".join(f"{k}={v}" for k, v in sorted(deviations.items())))
    result = photogrammetry.run_metashape_batch(
        str(run_dir / "project.psx"), frame_sets,
        log_fn=lambda m: log("    " + str(m)), **photo_settings)
    n_products = sum(len(v) for v in (result or {}).values())
    try:
        run_qc = json.loads((run_dir / "run_status.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        run_qc = {}

    # Merge the per-chunk orthos/DEMs into the survey-wide mosaics.  Nothing
    # else in the app called merge_products, so survey/photogrammetry/merged/
    # was never written and the survey report — which requires
    # merged/ortho_merged.tif — could not succeed on any freshly built
    # workspace.  Survey scope only: the merge reads the whole dive.
    if job.is_whole:
        superseded = None
        before = {p.name for p in _merged_outputs(ws)}
        try:
            from merge_products import build_previews, merge_survey
            # Previous mosaics move aside first (review 06 P0-1); a failed
            # merge puts them back.
            superseded = _supersede(ws, _merged_outputs(ws),
                                    "survey ortho/DEM re-merged after photogrammetry", log)
            merged = merge_survey(ws, log_fn=lambda m: log("    " + str(m)))
            previews = build_previews(ws, log_fn=lambda m: log("    " + str(m)))
            log("    merged: " + (", ".join(sorted({**merged, **previews}))
                                  or "nothing to merge"))
            lost = before - {p.name for p in _merged_outputs(ws)}
            if superseded is not None and lost:
                raise RuntimeError("merge did not reproduce " + ", ".join(sorted(lost)))
        except Exception as exc:                                    # noqa: BLE001
            log(f"    {LOG_WARN}ortho/DEM merge failed: {one_line(exc)}")
            if superseded is not None:
                _restore_superseded(superseded, log, [p for p in _merged_outputs(ws)
                                                      if p.name not in before])
    else:
        log("    ortho/DEM merge is survey-scoped — skipped for a job")

    characteristic = (f"{sampling_label(sampling)} · "
                      f"{len(frame_sets)} chunk{'s' if len(frame_sets) != 1 else ''} · "
                      f"{_fmt_int(n_photos)} photos")
    views: list[str] = []
    for spec in frame_sets:
        views.extend(str(p) for p in _CHUNK_VIEWS(Path(spec[1])) if p.exists())
    _write_meta(run_dir, "photogrammetry", job, characteristic,
                run_dir / "project.psx", views,
                extra={"created_at": created, "n_chunks": len(frame_sets),
                       "n_photos": n_photos, "gated_out": gated_out,
                       "chunk_size": chunk_size, "alt_max_m": alt_max,
                       "frame_set": str(set_dir), "n_products": n_products,
                       "sampling": sampling,
                       "sampling_label": sampling_label(sampling),
                       "photo_settings": photo_settings,
                       "chunk_qc": run_qc.get("chunks"),
                       "run_status": run_qc.get("status")})
    log(f"  photogrammetry done — {n_products} product file(s)")
    return _instance("photogrammetry", job.job_id, run_dir / "project.psx",
                     characteristic, views=views, created_at=created)


# --------------------------------------------------------------------------
# Nav trackline (GeoJSON)
# --------------------------------------------------------------------------

def _discover_nav_trackline(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    base = _scope_root(ws, job) / "nav_trackline"
    if not base.is_dir():
        return out
    for geojson in sorted(base.glob("*trackline*.geojson")):
        sidecar = geojson.with_suffix(".meta.json")
        owner, characteristic = WHOLE_TRACKLINE, None
        created = _mtime(geojson)
        if sidecar.is_file():
            try:
                meta = json.loads(sidecar.read_text(encoding="utf-8"))
                owner = str(meta.get("job_id") or WHOLE_TRACKLINE)
                characteristic = meta.get("characteristic")
                created = float(meta.get("created_at") or created)
            except (OSError, ValueError):
                pass
        if owner != job.job_id:
            continue
        if not characteristic:
            characteristic = f"{_fmt_int(_geojson_vertices(geojson))} vertices"
        out.append(_instance("nav_trackline", owner, geojson, characteristic,
                             created_at=created))
    return out


def _geojson_vertices(path) -> int:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return 0
    total = 0
    for feature in data.get("features", []):
        geom = feature.get("geometry") or {}
        coords = geom.get("coordinates") or []
        if geom.get("type") == "LineString":
            total += len(coords)
        elif geom.get("type") == "MultiLineString":
            total += sum(len(part) for part in coords)
    return total


def _generate_nav_trackline(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    ensure_interp(ws, log)
    created = time.time()
    if job.is_whole:
        from qgis_project import build_trackline_geojson
        notes: list[str] = []
        path = build_trackline_geojson(_resolver(ws), notes)
        for note in notes:
            log("  " + str(note))
        if path is None:
            raise RuntimeError("trackline not written: " + "; ".join(notes))
        path = Path(path)
    else:
        df = _interp_df(ws, ("unix_time", "easting", "northing")).dropna()
        base = _scope_root(ws, job, create=True) / "nav_trackline"
        base.mkdir(parents=True, exist_ok=True)
        path = _unique_file(base, f"trackline_{job_slug(job.job_id)}_{_stamp(created)}",
                            ".geojson")
        parts = []
        for iv in job.intervals:
            sub = df[(df["unix_time"] >= iv.t0) & (df["unix_time"] <= iv.t1)]
            coords = [[round(float(e), 3), round(float(n), 3)]
                      for e, n in zip(sub["easting"], sub["northing"])]
            if len(coords) >= 2:
                parts.append(coords)
        if not parts:
            raise RuntimeError("no track points inside the job's intervals")
        path.write_text(json.dumps({
            "type": "FeatureCollection", "name": "nav_trackline",
            "crs": {"type": "name",
                    "properties": {"name": "urn:ogc:def:crs:EPSG::32613"}},
            "features": [{"type": "Feature",
                          "geometry": {"type": "MultiLineString", "coordinates": parts},
                          "properties": {"name": f"Nav Trackline — {job.name}",
                                         "part_count": len(parts),
                                         "point_count": sum(len(p) for p in parts)}}],
        }, separators=(",", ":")), encoding="utf-8")
    characteristic = f"{_fmt_int(_geojson_vertices(path))} vertices"
    path.with_suffix(".meta.json").write_text(json.dumps(
        {"job_id": job.job_id, "job_name": job.name, "created_at": created,
         "characteristic": characteristic, "type_key": "nav_trackline"}, indent=2),
        encoding="utf-8")
    log(f"  trackline -> {path}")
    return _instance("nav_trackline", job.job_id, path, characteristic, created_at=created)


# --------------------------------------------------------------------------
# Rasters (depth + per-channel sensor)
# --------------------------------------------------------------------------

def _discover_depth_raster(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    base = _scope_root(ws, job) / "nav_depth"
    if not base.is_dir():
        return out
    for tif in sorted(base.rglob("nav_depth.tif")):
        run_dir = tif.parent
        owner = _meta_job(run_dir)
        if owner != job.job_id:
            continue
        meta = _read_meta(run_dir) or {}
        characteristic = meta.get("characteristic") or _raster_characteristic(tif, "depth")
        out.append(_instance("depth_raster", owner, tif, characteristic,
                             views=[str(tif)],
                             created_at=float(meta.get("created_at") or _mtime(tif))))
    return out


def _raster_characteristic(tif: Path, prefix: str) -> str:
    cell = None
    try:
        meta = json.loads(Path(str(tif) + ".meta.json").read_text(encoding="utf-8"))
        cell = (meta.get("settings") or {}).get("cell_size_m")
    except (OSError, ValueError):
        pass
    bits = [prefix]
    if cell:
        bits.append(f"{float(cell):g} m cell")
    bits.append(f"{tif.parent.name}")
    return " · ".join(bits)


def _generate_depth_raster(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    from output_service import OutputService
    interp = job_interp_csv(ws, job, log)
    created = time.time()
    service = OutputService(log_fn=lambda m: log("    " + str(m)))
    out_path = service.generate_nav_2d_geotiff(
        str(interp), str(_scope_root(ws, job, create=True)),
        cell_size_m=float(settings.get("cell_size_m", DEFAULTS["cell_size_m"])),
        crs_mode=str(settings.get("crs_mode", DEFAULTS["crs_mode"])))
    run_dir = Path(out_path).parent
    characteristic = (f"depth · {float(settings.get('cell_size_m', DEFAULTS['cell_size_m'])):g} m cell "
                      f"· {run_dir.name}")
    _write_meta(run_dir, "depth_raster", job, characteristic, out_path, [out_path],
                extra={"created_at": created})
    return _instance("depth_raster", job.job_id, out_path, characteristic,
                     views=[out_path], created_at=created)


def _discover_sensor_raster(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    base = _scope_root(ws, job) / "sensor_2d"
    if not base.is_dir():
        return out
    for tif in sorted(base.rglob("*_2d.tif")):
        run_dir = tif.parent
        owner = _meta_job(run_dir)
        if owner != job.job_id:
            continue
        meta = _read_meta(run_dir) or {}
        channel = meta.get("channel") or tif.stem[:-3] or run_dir.parent.name
        characteristic = meta.get("characteristic") or f"{channel} · {run_dir.name}"
        out.append(_instance("sensor_raster", owner, tif, characteristic,
                             views=[str(tif)],
                             created_at=float(meta.get("created_at") or _mtime(tif))))
    return out


def _sensor_schema(ws: str) -> list[tuple]:
    channels = sensor_channels(ws)
    return [
        ("channel", "Sensor channel", "choice", channels[0] if channels else "", channels),
        ("cell_size_m", "Cell size (m)", "float", DEFAULTS["cell_size_m"], None),
        ("crs_mode", "CRS", "choice", DEFAULTS["crs_mode"], ["utm", "wgs84"]),
        ("fill_method", "Fill", "choice", DEFAULTS["fill_method"], ["idw", "none", "rbf"]),
    ]


def _generate_sensor_raster(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    from output_service import OutputService
    channel = str(settings.get("channel") or "")
    if not channel:
        available = sensor_channels(ws)
        if not available:
            raise RuntimeError("no sensor channels in this workspace")
        channel = available[0]
    interp = job_interp_csv(ws, job, log)
    created = time.time()
    service = OutputService(log_fn=lambda m: log("    " + str(m)))
    out_path = service.generate_sensor_2d_geotiff(
        str(interp), str(_scope_root(ws, job, create=True)), channel,
        cell_size_m=float(settings.get("cell_size_m", DEFAULTS["cell_size_m"])),
        crs_mode=str(settings.get("crs_mode", DEFAULTS["crs_mode"])),
        fill_method=str(settings.get("fill_method", DEFAULTS["fill_method"])))
    run_dir = Path(out_path).parent
    characteristic = f"{channel} · {run_dir.name}"
    _write_meta(run_dir, "sensor_raster", job, characteristic, out_path, [out_path],
                extra={"created_at": created, "channel": channel})
    return _instance("sensor_raster", job.job_id, out_path, characteristic,
                     views=[out_path], created_at=created)


# --------------------------------------------------------------------------
# Anomaly detection (MATLAB detector + catalog + UTM layers + context)
# --------------------------------------------------------------------------

def _anomaly_dir(ws: str, job: Job) -> Path:
    if job.is_whole:
        return Path(_resolver(ws).anomaly_dir())
    return _scope_root(ws, job) / "anomaly"


def _discover_anomaly(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    base = _anomaly_dir(ws, job)
    windows = base / "anomaly_windows_all.csv"
    if not windows.is_file():
        return out
    if _meta_job(base) != job.job_id:
        return out
    meta = _read_meta(base) or {}
    n_windows = _csv_rows(windows)
    n_sites = _csv_rows(base / "anomalous_sites.csv")
    characteristic = meta.get("characteristic") or (
        f"{n_windows} window{'s' if n_windows != 1 else ''} · {n_sites} site"
        f"{'s' if n_sites != 1 else ''}")
    views = [str(p) for p in (
        windows, base / "anomalous_sites.csv", base / "anomalous_sites.geojson",
        base / "anomalous_sites_utm.geojson", base / "anomaly_segments_utm.geojson",
        base / "window_context.csv", base / "anomaly_combination_summary.csv",
        base / "video_review_clips.csv",
        base / "Anomaly_Site_and_Video_Review_Report.pdf") if p.exists()]
    out.append(_instance("anomaly_detection", job.job_id, windows, characteristic,
                         views=views,
                         created_at=float(meta.get("created_at") or _mtime(windows))))
    return out


def _csv_rows(path) -> int:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return 0


def _raw_nav_csv(ws: str) -> Optional[str]:
    try:
        nav = _ws_data(ws).get("navigation_file")
        source = getattr(nav, "latitude_source", None) if nav else None
        path = getattr(source, "csv_path", None)
        return str(path) if path else None
    except Exception:                                               # noqa: BLE001
        return None


def _generate_anomaly(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    import anomaly_service as anomaly

    reason = anomaly.matlab_unavailable_reason()
    if reason:
        raise RuntimeError(f"MATLAB unavailable: {reason}")

    if job.is_whole:
        interp = ensure_interp(ws, log)
    else:
        # GrapherMatrix.m opens a RELATIVE "interp_full.csv" from its cwd, so a
        # job-scoped run needs its restricted table under that exact name.
        scoped = _scope_root(ws, job, create=True) / "anomaly_input"
        scoped.mkdir(parents=True, exist_ok=True)
        interp = scoped / "interp_full.csv"
        shutil.copyfile(job_interp_csv(ws, job, log), interp)
        log(f"  job-scoped detector input: {interp}")

    out_dir = _anomaly_dir(ws, job)
    out_dir.mkdir(parents=True, exist_ok=True)
    event_root = Path(interp).resolve().parent / "grapher_matrix_figs"
    created = time.time()

    anomaly.run_detector(_REPO, interp, log_fn=lambda m: log("    " + str(m)))

    # The detector succeeded; only now move the previous catalog aside (never
    # overwrite in place — review 06 P0-1 / 04 P2-5).  A catalog failure rolls
    # back to the previous files instead of leaving a new/old mix.
    before = {p.name for p in _anomaly_outputs(out_dir)}
    superseded = _supersede(ws, _anomaly_outputs(out_dir),
                            f"anomaly detection re-run ({job.name})", log)
    try:
        # Every run-scoped path is explicit: nothing may fall back to the
        # builder's J1754 module defaults (review 03 P1-7).
        anomaly.run_catalog(_REPO, interp_csv=interp, raw_nav_csv=_raw_nav_csv(ws),
                            event_root=event_root, out_dir=out_dir,
                            ts_results=Path(_resolver(ws).inputs_dir()) / "ts_analysis_results.mat",
                            dive=_dive_name(ws),
                            log_fn=lambda m: log("    " + str(m)))
    except BaseException:
        if superseded is not None:
            _restore_superseded(superseded, log, [p for p in _anomaly_outputs(out_dir)
                                                  if p.name not in before])
        raise

    if job.is_whole:
        try:
            from anomaly_utm import build_utm_layers
            sites, segments = build_utm_layers(ws)
            log(f"    UTM layers: {Path(sites).name}, {Path(segments).name}")
        except Exception as exc:                                    # noqa: BLE001
            log(f"    {LOG_WARN}UTM layers failed: {one_line(exc)}")
        try:
            from window_context import classify_windows
            context = classify_windows(ws)
            split = context["context"].value_counts().to_dict() if len(context) else {}
            log(f"    window context: {split}")
        except Exception as exc:                                    # noqa: BLE001
            log(f"    {LOG_WARN}window context failed: {one_line(exc)}")
    else:
        log("    UTM layers / window context are survey-scoped "
            "(anomaly_utm + window_context read <ws>/survey/anomaly) — skipped for a job")

    n_windows = _csv_rows(out_dir / "anomaly_windows_all.csv")
    n_sites = _csv_rows(out_dir / "anomalous_sites.csv")
    characteristic = (f"{n_windows} window{'s' if n_windows != 1 else ''} · "
                      f"{n_sites} site{'s' if n_sites != 1 else ''}")
    views = [str(p) for p in sorted(out_dir.glob("*.csv"))] + \
            [str(p) for p in sorted(out_dir.glob("*.geojson"))] + \
            [str(p) for p in sorted(out_dir.glob("*.pdf"))]
    _write_meta(out_dir, "anomaly_detection", job, characteristic,
                out_dir / "anomaly_windows_all.csv", views,
                extra={"created_at": created})
    return _instance("anomaly_detection", job.job_id,
                     out_dir / "anomaly_windows_all.csv", characteristic,
                     views=views, created_at=created)


# --------------------------------------------------------------------------
# Figure products: spectrum trackline + anomaly trackline
# --------------------------------------------------------------------------

def _figure_dir(ws: str, name: str, create: bool = False) -> Path:
    base = Path(_resolver(ws).survey_products()) / name
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def _discover_figures(type_key: str, folder: str, ws: str, job: Job,
                      extra_views: Sequence[str] = ()) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    base = _figure_dir(ws, folder)
    if base.is_dir():
        for png in sorted(base.glob("*.png")):
            sidecar = png.with_suffix(".meta.json")
            owner, characteristic, created = WHOLE_TRACKLINE, None, _mtime(png)
            if sidecar.is_file():
                try:
                    meta = json.loads(sidecar.read_text(encoding="utf-8"))
                    owner = str(meta.get("job_id") or WHOLE_TRACKLINE)
                    characteristic = meta.get("characteristic")
                    created = float(meta.get("created_at") or created)
                except (OSError, ValueError):
                    pass
            if owner != job.job_id:
                continue
            out.append(_instance(type_key, owner, png,
                                 characteristic or png.stem,
                                 views=[str(png)] + [str(v) for v in extra_views],
                                 created_at=created))
    return out


def _discover_spectrum(ws: str, job: Job) -> list[ProductInstance]:
    return _discover_figures("spectrum_trackline", "spectrum_trackline", ws, job)


def _discover_anomaly_trackline(ws: str, job: Job) -> list[ProductInstance]:
    anomaly = _anomaly_dir(ws, job)
    geojsons = [str(p) for p in sorted(anomaly.glob("*.geojson"))] if anomaly.is_dir() else []
    out = _discover_figures("anomaly_trackline", "anomaly_trackline", ws, job, geojsons)
    # surface the anomaly geojsons even before a figure has been rendered
    if geojsons and not out and _meta_job(anomaly) == job.job_id:
        primary = next((g for g in geojsons if g.endswith("anomaly_segments_utm.geojson")),
                       geojsons[0])
        out.append(_instance("anomaly_trackline", job.job_id, primary,
                             f"{len(geojsons)} anomaly geojson layer(s)", views=geojsons))
    return out


def _figure_frame(ax, lims, title):
    ax.set_facecolor("#0e1620")
    ax.set_xlim(lims[0], lims[1])
    ax.set_ylim(lims[2], lims[3])
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10, color="#dbe4ec")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_color("#2a3846")


def _track_limits(track: np.ndarray):
    e0, e1 = float(track[:, 1].min()), float(track[:, 1].max())
    n0, n1 = float(track[:, 2].min()), float(track[:, 2].max())
    margin_e = max((e1 - e0) * 0.06, 20.0)
    margin_n = max((n1 - n0) * 0.04, 20.0)
    return (e0 - margin_e, e1 + margin_e, n0 - margin_n, n1 + margin_n)


def _generate_spectrum_trackline(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    """slide_deck.render_log_trackline's treatment (turbo on log10), single dive,
    restricted to this job's intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    requested = str(settings.get("channel") or "all")
    channels = sensor_channels(ws) if requested in ("all", "", "auto") else [requested]
    ensure_interp(ws, log)
    track = track_polyline(ws)
    columns = ["unix_time", "easting", "northing"] + list(channels)
    df = _interp_df(ws, columns)
    channels = [c for c in channels if c in df.columns]
    if not channels:
        raise RuntimeError("none of the requested channels exist in interp_full.csv")
    scoped = df[_mask_intervals(df["unix_time"].to_numpy(dtype=float), job.intervals)]
    scoped = scoped.dropna(subset=["easting", "northing"])
    if scoped.empty:
        raise RuntimeError("no interp rows inside the job's intervals")
    log(f"  {len(channels)} channel panel(s) over {_fmt_int(len(scoped))} samples")

    lims = _track_limits(track)
    n = len(channels)
    fig, axes = plt.subplots(1, n, figsize=(3.2 * n + 1.5, 6.8), dpi=150, squeeze=False)
    fig.patch.set_facecolor("#0e1620")
    for k, channel in enumerate(channels):
        ax = axes[0][k]
        ax.plot(track[:, 1], track[:, 2], color="#2b3743", lw=0.5, zorder=1)
        values = scoped[channel]
        finite = values[np.isfinite(values) & (values > 0)]
        if finite.empty:
            _figure_frame(ax, lims, f"{channel}\n(no positive values)")
            continue
        clipped = values.clip(lower=max(float(finite.min()), 1e-3))
        scatter = ax.scatter(scoped["easting"], scoped["northing"], c=clipped, s=2.5,
                            lw=0, cmap="turbo", zorder=3,
                            norm=LogNorm(vmin=float(finite.quantile(0.02)),
                                         vmax=max(float(finite.quantile(0.995)),
                                                  float(finite.quantile(0.02)) * 1.01)))
        bar = fig.colorbar(scatter, ax=ax, fraction=0.05, pad=0.02)
        bar.ax.tick_params(colors="#71828f", labelsize=6)
        _figure_frame(ax, lims, f"{channel} (log)")
    title = "Whole trackline" if job.is_whole else job.name
    fig.suptitle(f"{Path(ws).name} — log-spectrum trackline — {title}",
                 fontsize=12, color="#e8eef3")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    created = time.time()
    out = _unique_file(_figure_dir(ws, "spectrum_trackline", create=True),
                       f"spectrum_{job_slug(job.job_id)}_{_stamp(created)}", ".png")
    fig.savefig(out, dpi=150, facecolor="#0e1620", bbox_inches="tight")
    plt.close(fig)
    characteristic = f"{len(channels)} channel{'s' if len(channels) != 1 else ''} · log10 turbo"
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"job_id": job.job_id, "job_name": job.name, "created_at": created,
         "characteristic": characteristic, "channels": channels,
         "type_key": "spectrum_trackline"}, indent=2), encoding="utf-8")
    log(f"  spectrum trackline -> {out}")
    return _instance("spectrum_trackline", job.job_id, out, characteristic,
                     views=[out], created_at=created)


def _generate_anomaly_trackline(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    """multi_dive_map's tier-coloured stroke treatment, single dive, restricted
    to this job's intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.collections import LineCollection
    from matplotlib.lines import Line2D

    anomaly_dir = _anomaly_dir(ws, job)
    segments_path = anomaly_dir / "anomaly_segments_utm.geojson"
    if not segments_path.is_file():                 # a job run has no UTM layers
        segments_path = Path(_resolver(ws).anomaly_dir()) / "anomaly_segments_utm.geojson"
    if not segments_path.is_file():
        raise RuntimeError(f"missing {segments_path} — run anomaly detection "
                           "(and anomaly_utm.build_utm_layers) first")
    features = json.loads(segments_path.read_text(encoding="utf-8")).get("features", [])

    station_ids: set = set()
    if bool(settings.get("transit_only", False)):
        context_path = anomaly_dir / "window_context.csv"
        if context_path.is_file():
            context = pd.read_csv(context_path)
            station_ids = set(context.loc[context["context"] == "station", "window_id"])
            log(f"  transit-only: hiding {len(station_ids)} station window(s)")

    def in_scope(feature) -> bool:
        if not job.intervals:
            return True
        properties = feature.get("properties") or {}
        start = _parse_iso(properties.get("start_utc"))
        end = _parse_iso(properties.get("end_utc"))
        if start is None:
            return True
        end = end if end is not None else start
        return any(start <= iv.t1 and iv.t0 <= end for iv in job.intervals)

    track = track_polyline(ws)
    lims = _track_limits(track)
    fig, ax = plt.subplots(figsize=(9.5, 10), dpi=150)
    fig.patch.set_facecolor("#0e1620")
    ax.plot(track[:, 1], track[:, 2], color="#41505d", lw=0.7, alpha=0.8, zorder=2)
    if not job.is_whole:
        df = _interp_df(ws, ("unix_time", "easting", "northing")).dropna()
        for iv in job.intervals:
            sub = df[(df["unix_time"] >= iv.t0) & (df["unix_time"] <= iv.t1)]
            ax.plot(sub["easting"], sub["northing"], color="#e3ecf4", lw=2.2,
                    alpha=0.95, zorder=2.6)

    n_windows = 0
    for i, tier in enumerate(("SCREEN", "MODERATE", "HIGH")):
        strokes = [np.array(f["geometry"]["coordinates"]) for f in features
                   if (f.get("properties") or {}).get("confidence") == tier
                   and (f.get("properties") or {}).get("window_id") not in station_ids
                   and in_scope(f)]
        if strokes:
            ax.add_collection(LineCollection(strokes, colors=TIER_COLOUR[tier],
                                             linewidths=2.4, zorder=3 + i,
                                             capstyle="round"))
            n_windows += len(strokes)

    sites_path = segments_path.parent / "anomalous_sites_utm.geojson"
    n_sites = 0
    if sites_path.is_file():
        for feature in json.loads(sites_path.read_text(encoding="utf-8")).get("features", []):
            x, y = feature["geometry"]["coordinates"][:2]
            if lims[0] <= x <= lims[1] and lims[2] <= y <= lims[3]:
                ax.plot(x, y, marker="o", ms=6, mfc="none", mec="#ffffff",
                        mew=0.9, alpha=0.85, zorder=8)
                n_sites += 1
    _figure_frame(ax, lims, "")
    ax.legend(handles=[Line2D([0], [0], color=TIER_COLOUR[t], lw=3, label=f"{t.title()} window")
                       for t in ("HIGH", "MODERATE", "SCREEN")]
                      + [Line2D([0], [0], marker="o", ls="", mfc="none", mec="#fff",
                                ms=6, label="ranked site")],
              loc="upper right", fontsize=8, framealpha=0.92, facecolor="#16222e",
              edgecolor="#2a3846", labelcolor="#dbe4ec")
    title = "Whole trackline" if job.is_whole else job.name
    ax.set_title(f"{Path(ws).name} — anomaly windows — {title} "
                 f"({n_windows} window{'s' if n_windows != 1 else ''})",
                 color="#e8eef3", fontsize=12, pad=10)

    created = time.time()
    out = _unique_file(_figure_dir(ws, "anomaly_trackline", create=True),
                       f"anomaly_{job_slug(job.job_id)}_{_stamp(created)}", ".png")
    fig.savefig(out, dpi=150, facecolor="#0e1620", bbox_inches="tight")
    plt.close(fig)
    characteristic = (f"{n_windows} window{'s' if n_windows != 1 else ''} · "
                      f"{n_sites} site{'s' if n_sites != 1 else ''}")
    out.with_suffix(".meta.json").write_text(json.dumps(
        {"job_id": job.job_id, "job_name": job.name, "created_at": created,
         "characteristic": characteristic, "type_key": "anomaly_trackline"},
        indent=2), encoding="utf-8")
    log(f"  anomaly trackline -> {out}")
    views = [str(out)] + [str(p) for p in sorted(segments_path.parent.glob("*.geojson"))]
    return _instance("anomaly_trackline", job.job_id, out, characteristic,
                     views=views, created_at=created)


def _parse_iso(value) -> Optional[float]:
    """ISO timestamp (catalog style, trailing Z tolerated) -> unix seconds."""
    if not value:
        return None
    text = str(value).strip().rstrip("Z")
    try:
        return unix(datetime.fromisoformat(text))
    except ValueError:
        return None


# --------------------------------------------------------------------------
# Survey report
# --------------------------------------------------------------------------

def _discover_survey_report(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []
    if job.is_whole:
        candidates = [Path(ws) / "SURVEY_REPORT.html"] + \
            sorted(Path(ws).glob("*_Survey_Report.pdf"))
    else:
        candidates = sorted(_scope_root(ws, job).glob("SURVEY_REPORT*.html"))
    for path in candidates:
        if not path.is_file():
            continue
        size = path.stat().st_size
        out.append(_instance("survey_report", job.job_id, path,
                             f"{path.suffix.lstrip('.').upper()} · {size / 1e6:.1f} MB"))
    return out


def _generate_survey_report(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    from survey_report import build_survey_report
    # survey_report.collect() opens the merged orthomosaic unconditionally; say so
    # up front instead of surfacing a bare RasterioIOError from three frames down.
    merged = (Path(_resolver(ws).survey_products()) / "photogrammetry" / "merged"
              / "ortho_merged.tif")
    if not merged.is_file():
        raise RuntimeError(
            f"survey report needs the merged orthomosaic ({merged}); run "
            "photogrammetry (and the ortho merge) for this dive first")
    created = time.time()
    if job.is_whole:
        out_path = None
    else:
        out_path = str(_unique_file(_scope_root(ws, job, create=True),
                                    f"SURVEY_REPORT_{_stamp(created)}", ".html"))
        log("  note: survey_report.collect() is dive-wide; the job scope only "
            "changes where the file is written")
    superseded = None
    if job.is_whole:
        # The fixed-name dive report is moved aside, never overwritten in place
        # (review 06 P0-1); a failed build puts it back.
        superseded = _supersede(ws, [Path(ws) / "SURVEY_REPORT.html"],
                                "survey report regenerated", log)
    try:
        result = build_survey_report(ws, out_path=out_path,
                                     fast_mesh=bool(settings.get("fast_mesh", True)))
    except BaseException:
        if superseded is not None:
            _restore_superseded(superseded, log)
        raise
    path = Path(result)
    characteristic = f"HTML · {path.stat().st_size / 1e6:.1f} MB"
    log(f"  survey report -> {path}")
    return _instance("survey_report", job.job_id, path, characteristic,
                     views=[path], created_at=created)


# --------------------------------------------------------------------------
# Fauna computer vision (zero-shot MBARI detector over a frame set)
# --------------------------------------------------------------------------
#
# Nothing here reimplements the detector.  ``fathomnet_detect`` owns inference,
# the morphology buckets, the midwater/non-fauna exclusions, the GeoJSON and
# per-frame density tables and the in-window/out-of-window comparison;
# ``fauna_timeseries`` owns the footprint model and the density time series;
# ``fauna_occurrences`` owns the joined occurrence table.  This section only
#
#   * picks the frames — ALWAYS a recycled frame set, never a fresh extraction;
#   * redirects those modules' nav lookup at that frame set (they hard-code the
#     legacy ``survey/photogrammetry/seg*/segment_*`` layout);
#   * gives them a run-scoped workspace root so a job / sampling regime gets its
#     own products instead of overwriting the dive-wide census;
#   * degrades gracefully when anomaly detection has not run yet.
#
# The run root is a miniature workspace, because that is the interface those
# three modules take:
#
#     <scope>/fauna_runs/<DIVE>_fauna_<stamp>__<sampling>/
#         survey/fauna/…                     the products
#         survey/anomaly/window_context.csv  linked from the dive (or empty)
#         inputs/interp_full.csv             hard-linked from the dive
#
# It is named "<DIVE>_…" on purpose: both modules derive the dive label with
# ``Path(ws).name.split("_")[0]``, so figures and the occurrence table's "dive"
# column still read "J1754" rather than "run".

#: Canonical census filenames, in display order.  ``discover`` looks for these
#: under <ws>/survey/fauna (the existing full-census products on J1754/J1758)
#: and under each run's survey/fauna.
FAUNA_PRODUCT_FILES = (
    "fathomnet_detections.csv", "fauna_points_utm.geojson", "fauna_density.csv",
    "fish_frame_shortlist.csv", "occurrences.csv", "fauna_vs_anomaly.png",
    "fauna_density_timeseries_v2.png", "fauna_density_timeseries.png",
    # provenance sidecars written by fathomnet_detect / fauna_timeseries /
    # fauna_occurrences (review 02 P1-2); listed so a census replacement
    # supersedes them together with the tables they describe.
    "fauna_provenance.json", "fauna_density.meta.json", "occurrences.meta.json",
)

#: Columns the fauna modules read off window_context.csv.  An empty table with
#: exactly these columns is what "anomaly detection has not run yet" looks like.
_WINDOW_CONTEXT_COLUMNS = (
    "start_time", "end_time", "window_id", "confidence_tier", "anomaly_class",
    "channels", "context", "median_speed", "site_id", "evidence_score",
)


def _dive_name(ws: str) -> str:
    """"J1754" from "…/J1754_down.eprproj" — the label the fauna modules derive."""
    return Path(ws).name.split("_")[0] or Path(ws).name


def _fauna_runs_root(ws: str, job: Job, create: bool = False) -> Path:
    base = _scope_root(ws, job) / "fauna_runs"
    if create:
        base.mkdir(parents=True, exist_ok=True)
    return base


def fauna_thresholds(settings: Optional[dict] = None,
                     log: Optional[Callable] = None) -> dict[str, float]:
    """Per-bucket confidence thresholds for the PRODUCTION (zero-shot) model.

    From ``deploy_config.json``'s ``conf_balanced`` per bucket.  The detector
    itself runs at the MINIMUM of these so nothing is thrown away inside
    ultralytics; each bucket's own threshold is then applied afterwards, and
    what falls below it stays in the raw CSV marked ``excluded="below_conf"``
    (the same convention the midwater/non-fauna filters use).
    """
    log = _logger(log)
    settings = settings or {}
    if not bool(settings.get("fauna_bucket_thresholds",
                             DEFAULTS["fauna_bucket_thresholds"])):
        return {}
    path = Path(settings.get("fauna_deploy_config") or FAUNA_DEPLOY_CONFIG)
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
        buckets = config["models"][FAUNA_MODEL_KEY]["buckets"]
    except (OSError, ValueError, KeyError) as exc:
        log(f"  {LOG_WARN}deploy config {path} unreadable ({one_line(exc)}; set "
            f"{FAUNA_DEPLOY_CONFIG_ENV} to its path) — using a "
            f"flat conf floor of {float(settings.get('fauna_conf', DEFAULTS['fauna_conf'])):g}")
        return {}
    out = {name: float(spec.get("conf_balanced", DEFAULTS["fauna_conf"]))
           for name, spec in buckets.items()}
    log("  thresholds [" + FAUNA_MODEL_KEY + "]: "
        + ", ".join(f"{k} {v:g}" for k, v in sorted(out.items())))
    return out


def fauna_device(settings: Optional[dict] = None,
                 log: Optional[Callable] = None):
    """GPU when one is really usable, CPU otherwise — never a hard failure."""
    log = _logger(log)
    requested = str((settings or {}).get("fauna_device", DEFAULTS["fauna_device"])).lower()
    if requested in ("cpu",):
        log("  device: cpu (requested)")
        return "cpu"
    try:
        import torch
        available = bool(torch.cuda.is_available())
    except Exception as exc:                                        # noqa: BLE001
        log(f"  {LOG_WARN}torch unavailable ({one_line(exc)}) — CPU inference")
        return "cpu"
    if not available:
        if requested not in ("auto", ""):
            log(f"  {LOG_WARN}device '{requested}' requested but CUDA is not "
                "available — falling back to CPU")
        else:
            log("  device: cpu (no CUDA)")
        return "cpu"
    device = 0 if requested in ("auto", "", "cuda", "gpu") else requested
    try:
        name = torch.cuda.get_device_name(0)
        free, total = torch.cuda.mem_get_info(0)
        log(f"  device: cuda:{device} — {name}, {free / 1e9:.1f}/{total / 1e9:.1f} GB free")
    except Exception:                                               # noqa: BLE001
        log(f"  device: cuda:{device}")
    return int(device) if str(device).isdigit() else device


def fauna_frame_nav(set_dir) -> pd.DataFrame:
    """``fathomnet_detect.load_frame_nav``'s table, built from a FRAME SET.

    Same shape and column names — indexed by frame basename, carrying
    unix_time / easting / northing / alt / depth / heading / seg — so the fauna
    modules cannot tell the difference.  They hard-code the legacy
    ``survey/photogrammetry/seg*/segment_*`` glob, which a recycled frame set
    (``survey/frame_sets/<scope>/set_*/segment_*``) does not match.
    """
    wanted = ["frame_filename", "unix_time", "easting", "northing", "alt",
              "depth", "heading"]
    parts: list[pd.DataFrame] = []
    for segment_dir in _segment_dirs(Path(set_dir)):
        try:
            part = pd.read_csv(segment_dir / "interp.csv",
                               usecols=lambda c: c in wanted)
        except Exception:                                           # noqa: BLE001
            continue
        if "frame_filename" not in part.columns:
            continue
        part["seg"] = segment_dir.name
        parts.append(part)
    if not parts:
        raise FileNotFoundError(f"no usable interp.csv manifests under {set_dir}")
    nav = pd.concat(parts, ignore_index=True)
    for column in ("unix_time", "easting", "northing", "alt", "depth", "heading"):
        if column not in nav.columns:
            nav[column] = np.nan
    return nav.drop_duplicates("frame_filename").set_index("frame_filename")


class _nav_from(object):
    """Point every fauna module's ``load_frame_nav`` at one nav table.

    ``fauna_occurrences`` did ``from fathomnet_detect import load_frame_nav``,
    so its own binding has to be replaced too; both are restored on exit even
    if the body raises.
    """

    def __init__(self, nav: pd.DataFrame) -> None:
        self._nav = nav
        self._saved: list[tuple] = []

    def __enter__(self):
        import fathomnet_detect
        import fauna_occurrences
        for module in (fathomnet_detect, fauna_occurrences):
            if hasattr(module, "load_frame_nav"):
                self._saved.append((module, module.load_frame_nav))
                module.load_frame_nav = lambda _ws=None, _nav=self._nav: _nav
        return self._nav

    def __exit__(self, *_exc):
        for module, original in self._saved:
            module.load_frame_nav = original
        self._saved = []
        return False


def _is_census_run(job: Job, settings: dict) -> bool:
    """The DIVE-WIDE run at the default sampling produces the canonical census
    (<ws>/survey/fauna) that catalog_builder, fauna_ortho, worm_cover, the
    BIIGLE bridge and the angle scripts read by name."""
    return bool(job.is_whole and sampling_matches(settings, {}))


def _fauna_run_root(ws: str, job: Job, settings: dict, created: float,
                    log: Callable, temporary: list) -> tuple[Path, bool]:
    """The miniature workspace a fauna run writes into, and whether windows exist.

    EVERY run — the dive-wide census included — builds inside its own mini
    workspace (survey/fauna, survey/anomaly/window_context.csv, inputs/
    interp_full.csv), the interface fathomnet_detect / fauna_timeseries /
    fauna_occurrences take.  For the census that root is a STAGING directory
    (``survey/.fauna_staging/…``, registered in ``temporary`` for removal); the
    finished products are promoted into ``survey/fauna`` only after the whole
    build succeeded, with the previous census moved to ``_superseded/``.  So:

    * a crash or kill mid-run never leaves new detections beside old density /
      occurrence / meta files (review 06 P1-3, 04 P1-7);
    * the "no windows yet" EMPTY window_context.csv placeholder is only ever
      written inside the run root — never into the dive's own anomaly
      directory, where a kill used to leave a fabricated zero-window table
      behind (review 04 P1-6).

    ``have_windows`` is False when the dive has no anomaly window table yet.
    """
    census = _is_census_run(job, settings)
    if census:
        existing = [p.name for p in _census_outputs(ws)]
        if existing:
            log(f"  {LOG_WARN}this dive-wide run will REPLACE the census in "
                f"{Path(_resolver(ws).survey_products()) / 'fauna'} once it completes "
                f"({len(existing)} file(s): " + ", ".join(existing[:4])
                + ("…" if len(existing) > 4 else "") + "); the previous files are "
                f"moved to {Path(ws) / SUPERSEDED_DIR}/<stamp>/ first")
        parent = Path(_resolver(ws).survey_products()) / ".fauna_staging"
    else:
        parent = _fauna_runs_root(ws, job, create=True)
    # "<DIVE>_…": the fauna modules derive the dive label from the root's name.
    root = _unique_dir(parent, f"{_dive_name(ws)}_fauna_{_stamp(created)}"
                               f"__{sampling_slug(settings)}")
    if census:
        temporary.append(root)
        log(f"  staging census build in {root}")
    (root / "survey" / "fauna").mkdir(parents=True, exist_ok=True)
    (root / "survey" / "anomaly").mkdir(parents=True, exist_ok=True)
    (root / "inputs").mkdir(parents=True, exist_ok=True)

    # interp_full.csv: hard-linked (same filesystem) so a 24 MB table is not
    # copied per run; _link_or_copy falls back to symlink, then to a real copy.
    try:
        _link_or_copy(Path(ensure_interp(ws, log)), root / "inputs" / "interp_full.csv")
    except Exception as exc:                                        # noqa: BLE001
        log(f"  {LOG_WARN}could not stage interp_full.csv: {one_line(exc)} — the "
            "occurrence table will be skipped")

    source = _anomaly_dir(ws, job) / "window_context.csv"
    if not source.is_file():                       # a job run has no UTM/context
        source = Path(_resolver(ws).anomaly_dir()) / "window_context.csv"
    target = root / "survey" / "anomaly" / "window_context.csv"
    if source.is_file():
        shutil.copyfile(source, target)            # small (~260 kB), always copy
        return root, True
    pd.DataFrame(columns=list(_WINDOW_CONTEXT_COLUMNS)).to_csv(target, index=False)
    log(f"  {LOG_WARN}no anomaly window table for this workspace "
        f"({source}) — anomaly detection has not run yet. Fauna products are "
        "still built; the in/out-of-window comparison is skipped and the "
        "occurrence table's window columns stay blank.")
    return root, False


def _promote_census(ws: str, staged: Path, log: Callable) -> Path:
    """Move a finished staged census into <ws>/survey/fauna.

    The previous census files that this run replaces — AND any stale sibling
    in FAUNA_PRODUCT_FILES the new run did not produce — go to
    ``_superseded/<stamp>/`` first, so the directory never mixes generations.
    Files derived from the old census that no code here regenerates are named
    in the log.
    """
    census = Path(_resolver(ws).survey_products()) / "fauna"
    census.mkdir(parents=True, exist_ok=True)
    new_files = sorted(p for p in Path(staged).iterdir() if p.is_file())
    _supersede(ws, _census_outputs(ws), "fauna census replaced by a dive-wide run", log)
    for p in new_files:
        os.replace(p, census / p.name)
    derived = sorted(p.name for p in census.iterdir()
                     if p.is_file() and not p.name.startswith("fathomnet_detections")
                     and p.name not in {q.name for q in new_files}
                     and "_subsample" not in p.name)
    if derived:
        log(f"  {LOG_WARN}files in {census} were derived from the PREVIOUS census and "
            "are not regenerated here — rebuild them: " + ", ".join(derived[:8])
            + ("…" if len(derived) > 8 else ""))
    log(f"  census promoted: {len(new_files)} file(s) -> {census}")
    return census


def _fauna_characteristic(meta: dict) -> str:
    bits = []
    if meta.get("sampling_label"):
        bits.append(str(meta["sampling_label"]))
    n_kept = meta.get("n_detections_kept")
    if n_kept is not None:
        bits.append(f"{_fmt_int(n_kept)} detections")
    n_frames = meta.get("n_frames")
    if n_frames:
        bits.append(f"{_fmt_int(n_frames)} frames")
    by_bucket = meta.get("by_bucket_kept") or {}
    if by_bucket:
        bits.append(", ".join(f"{k} {_fmt_int(v)}" for k, v in
                              sorted(by_bucket.items(), key=lambda kv: -kv[1])[:3]))
    return " · ".join(bits) or "fauna detection"


def _fauna_views(fauna_dir: Path, suffix: str = "") -> list[str]:
    out = []
    for name in FAUNA_PRODUCT_FILES:
        stem, dot, ext = name.rpartition(".")
        candidate = fauna_dir / f"{stem}{suffix}{dot}{ext}"
        if candidate.is_file():
            out.append(str(candidate))
    return out


def _discover_fauna(ws: str, job: Job) -> list[ProductInstance]:
    out: list[ProductInstance] = []

    # (1) runs this UI made, for this scope
    root = _fauna_runs_root(ws, job)
    if root.is_dir():
        for run_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            fauna_dir = run_dir / "survey" / "fauna"
            primary = fauna_dir / "fathomnet_detections.csv"
            if not primary.is_file():
                continue
            if _meta_job(fauna_dir) != job.job_id:
                continue
            meta = _read_meta(fauna_dir) or {}
            out.append(_instance(
                "fauna_detection", job.job_id, primary,
                meta.get("characteristic") or _fauna_characteristic(meta),
                views=_fauna_views(fauna_dir),
                created_at=float(meta.get("created_at") or _mtime(primary))))

    # (2) the dive-wide census the batch/CLI runs already produced — the
    # products the PI browses on J1754 / J1758.  They predate this UI and carry
    # no meta, so they belong to the whole trackline.  One instance per
    # detections table, which is how the "_subsample" census shows up as its
    # own entry beside the full one.
    if job.is_whole:
        census = Path(_resolver(ws).survey_products()) / "fauna"
        meta = _read_meta(census) or {}          # present once this UI has run
        for primary in sorted(census.glob("fathomnet_detections*.csv")):
            suffix = primary.stem[len("fathomnet_detections"):]
            views = _fauna_views(census, suffix)
            created = None
            if not suffix and meta.get("type_key") == "fauna_detection":
                characteristic = meta.get("characteristic") or _fauna_characteristic(meta)
                created = float(meta.get("created_at") or _mtime(primary))
            else:
                n_det = _csv_rows(primary)
                density = census / f"fauna_density{suffix}.csv"
                n_frames = _csv_rows(density) if density.is_file() else 0
                label = ("full census" if not suffix
                         else suffix.lstrip("_") + " census")
                characteristic = f"{label} · {_fmt_int(n_det)} detections"
                if n_frames:
                    characteristic += f" · {_fmt_int(n_frames)} frames"
            out.append(_instance("fauna_detection", WHOLE_TRACKLINE, primary,
                                 characteristic, views=views, created_at=created))
    return out


def _fauna_schema(ws: Optional[str] = None) -> list[tuple]:
    return [
        # Only "dynamic" is offered: sampling_grid + PipelineService implement
        # the distance-walk sampler alone.  sampling_label/sampling_slug already
        # render "fixed 1 s", so the identity machinery is ready the day a
        # fixed-interval sampler lands — but a choice that silently produced
        # dynamic frames would be worse than no choice.
        ("sampling_mode", "Sampling technique", "choice",
         DEFAULTS["sampling_mode"], ["dynamic"]),
        ("spacing_m", "Target spacing (m)", "float", DEFAULTS["spacing_m"], None),
        ("min_frequency_hz", "Minimum frequency (Hz)", "float",
         DEFAULTS["min_frequency_hz"], None),
        ("frame_set", "Frame set (latest = matching sampling)", "str", "latest", None),
        ("fauna_weights", "Detector weights", "str", DEFAULTS["fauna_weights"], None),
        ("fauna_bucket_thresholds", "Per-bucket thresholds (deploy config)",
         "bool", DEFAULTS["fauna_bucket_thresholds"], None),
        ("fauna_conf", "Flat confidence floor (no deploy config)", "float",
         DEFAULTS["fauna_conf"], None),
        ("fauna_imgsz", "Inference size (px)", "int", DEFAULTS["fauna_imgsz"], None),
        ("fauna_batch", "Frames per batch", "int", DEFAULTS["fauna_batch"], None),
        ("fauna_device", "Device", "choice", DEFAULTS["fauna_device"],
         ["auto", "cuda", "cpu"]),
        ("fauna_limit", "Max frames per segment (0 = all)", "int", 0, None),
    ]


def _generate_fauna(ws: str, job: Job, settings: dict, log: Callable) -> ProductInstance:
    import fathomnet_detect as fd

    weights = Path(settings.get("fauna_weights") or FAUNA_WEIGHTS)
    if not weights.is_file():
        source = ("the Detector weights setting" if settings.get("fauna_weights")
                  and str(settings.get("fauna_weights")) != str(FAUNA_WEIGHTS)
                  else (f"${FAUNA_WEIGHTS_ENV}" if os.environ.get(FAUNA_WEIGHTS_ENV)
                        else f"the built-in fallback (${FAUNA_WEIGHTS_ENV} is not set)"))
        raise RuntimeError(
            f"detector weights not found: {weights} (from {source}). Set the "
            f"environment variable {FAUNA_WEIGHTS_ENV} to the path of "
            "mbari_315k_yolov8.pt, or enter it under Detector weights")

    # ---- frames: ALWAYS a recycled frame set, never a fresh extraction ------
    set_dir = resolve_frame_set(ws, job, settings, log)
    segments = _segment_dirs(set_dir)
    if not segments:
        raise RuntimeError(f"frame set has no segments: {set_dir}")
    nav = fauna_frame_nav(set_dir)
    log(f"  frames from: {set_dir} ({len(segments)} segment(s), "
        f"{_fmt_int(len(nav))} manifested frames)")

    thresholds = fauna_thresholds(settings, log)
    floor = (min(thresholds.values()) if thresholds
             else float(settings.get("fauna_conf", DEFAULTS["fauna_conf"])))
    device = fauna_device(settings, log)
    imgsz = int(settings.get("fauna_imgsz", DEFAULTS["fauna_imgsz"]))
    batch = max(1, int(settings.get("fauna_batch", DEFAULTS["fauna_batch"])))
    if device == "cpu":
        batch = min(batch, 2)          # a WSL VM that OOMs must not decode 8x 16 MP
        log(f"  {LOG_WARN}CPU inference — batch reduced to {batch} and this will "
            "be slow")
    limit = int(settings.get("fauna_limit", 0) or 0) or None

    # ---- inference, ONE SEGMENT AT A TIME ----------------------------------
    parts: list[pd.DataFrame] = []
    frames: list[str] = []
    runtime = 0.0
    for i, segment_dir in enumerate(segments, 1):
        frame_dir = segment_dir / "frames"
        log(f"  [{i}/{len(segments)}] {segment_dir.name}")
        try:
            part = fd.detect_frames(frame_dir, weights=weights, conf=floor,
                                    imgsz=imgsz, batch=batch, device=device,
                                    limit=limit, log=lambda m: log("    " + str(m)))
        except FileNotFoundError as exc:
            log(f"    {LOG_WARN}skipped: {one_line(exc)}")
            continue
        frames.extend(part.attrs.get("frames") or [])
        runtime += float(part.attrs.get("runtime_s") or 0.0)
        if len(part):
            parts.append(part)
    if not frames:
        raise RuntimeError(f"no .jpg frames under any segment of {set_dir}")

    columns = ["fn", "cls", "conf", "x1", "y1", "x2", "y2", "bucket", "excluded"]
    det = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=columns))
    det["excluded"] = det["excluded"].fillna("")

    # ---- per-bucket operating point ----------------------------------------
    if thresholds and len(det):
        needed = det["bucket"].map(thresholds).astype(float).fillna(0.0)
        low = det["excluded"].eq("") & (det["conf"].astype(float) < needed)
        det.loc[low, "excluded"] = "below_conf"
        log(f"  per-bucket thresholds dropped {_fmt_int(int(low.sum()))} of "
            f"{_fmt_int(len(det))} raw detections")
    det.attrs["frames"] = frames
    det.attrs["n_frames"] = len(frames)
    det.attrs["runtime_s"] = runtime

    kept = det[det["excluded"] == ""]
    log(f"  {_fmt_int(len(det))} raw detections over {_fmt_int(len(frames))} "
        f"frames in {runtime:.0f} s — {_fmt_int(len(kept))} kept")

    # ---- products -----------------------------------------------------------
    created = time.time()
    temporary: list[Path] = []
    try:
        return _build_fauna_products(ws, job, settings, det, nav, frames, kept,
                                     set_dir, weights, floor, thresholds, imgsz,
                                     batch, device, runtime, created, temporary, log)
    finally:
        # Staging roots / placeholders are removed however the build ended
        # (exception, KeyboardInterrupt, …) — nothing temporary survives.
        for path in temporary:
            try:
                if Path(path).is_dir():
                    shutil.rmtree(path, ignore_errors=True)
                else:
                    Path(path).unlink()
            except OSError:
                pass


def _build_fauna_products(ws, job, settings, det, nav, frames, kept, set_dir,
                          weights, floor, thresholds, imgsz, batch, device,
                          runtime, created, temporary, log) -> ProductInstance:
    import fathomnet_detect as fd

    run_root, have_windows = _fauna_run_root(ws, job, settings, created, log,
                                             temporary)
    out_dir = run_root / "survey" / "fauna"
    out_dir.mkdir(parents=True, exist_ok=True)
    det.to_csv(out_dir / "fathomnet_detections.csv", index=False)

    notes: list[str] = []
    with _nav_from(nav):
        # to_geojson is the one step that is NOT optional: everything below
        # consumes its per-frame density table.
        dens = fd.to_geojson(det, run_root, out_dir / "fauna_points_utm.geojson",
                             density_path=out_dir / "fauna_density.csv",
                             frames=frames, log=lambda m: log("    " + str(m)))
        _fauna_step("fish frame shortlist", notes, log,
                    lambda: fd.fish_frame_shortlist(
                        det, run_root, out_dir / "fish_frame_shortlist.csv",
                        log=lambda m: log("    " + str(m))))

        if have_windows:
            _fauna_step("density vs anomaly windows", notes, log, lambda: fd.density_vs_anomaly(
                run_root, dens, out_png=out_dir / "fauna_vs_anomaly.png",
                dive=_dive_name(ws), log=lambda m: log("    " + str(m))))
        else:
            notes.append("density-vs-anomaly comparison (no anomaly windows)")

        import fauna_timeseries as ft
        # add_density_columns owns the footprint model; it rewrites
        # fauna_density.csv in place with area_m2 + dens_* columns.
        _fauna_step("areal density columns", notes, log,
                    lambda: ft.add_density_columns(str(run_root)))
        _fauna_step("density time series", notes, log,
                    lambda: ft.build(str(run_root), log=lambda m: log("    " + str(m))))

        import fauna_occurrences as fo
        _fauna_step("occurrence table", notes, log, lambda: fo.build(
            run_root, det_csv=out_dir / "fathomnet_detections.csv",
            out_csv=out_dir / "occurrences.csv",
            log=lambda m: log("    " + str(m))))

    by_bucket = {str(k): int(v) for k, v in kept["bucket"].value_counts().items()}
    # Identity follows the frames actually used (an explicit frame_set path may
    # carry a different regime), exactly as photogrammetry does.
    sampling = (frame_set_sampling(set_dir) if (set_dir / _SET_META).is_file()
                else sampling_settings(settings))
    meta_extra = {
        "created_at": created,
        "sampling": sampling,
        "sampling_label": sampling_label(sampling),
        "frame_set": str(set_dir),
        "weights": str(weights), "model": FAUNA_MODEL_KEY,
        "conf_floor": floor, "bucket_thresholds": thresholds,
        "imgsz": imgsz, "batch": batch, "device": str(device),
        "n_frames": len(frames), "runtime_s": round(runtime, 1),
        "n_detections_raw": int(len(det)),
        "n_detections_kept": int(len(kept)),
        "by_bucket_kept": by_bucket,
        "had_anomaly_windows": have_windows,
        "skipped": notes,
    }
    try:
        meta_extra["provenance"] = json.loads(
            (out_dir / "fauna_provenance.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        pass
    characteristic = _fauna_characteristic(meta_extra)
    if _is_census_run(job, settings):
        # The whole build succeeded in staging: only now does it replace the
        # canonical census (old files -> _superseded/, review 06 P0-1 / P1-3).
        out_dir = _promote_census(ws, out_dir, log)
    views = _fauna_views(out_dir)
    # Meta lives beside the products (survey/fauna), never at a run root.
    _write_meta(out_dir, "fauna_detection", job, characteristic,
                out_dir / "fathomnet_detections.csv", views, extra=meta_extra)
    (out_dir / "fauna_run_summary.json").write_text(
        json.dumps(meta_extra, indent=2, default=str), encoding="utf-8")
    if notes:
        log(f"  {LOG_WARN}not produced: " + "; ".join(notes))
    log(f"  fauna products -> {out_dir}")
    return _instance("fauna_detection", job.job_id,
                     out_dir / "fathomnet_detections.csv", characteristic,
                     views=views, created_at=created)


def _fauna_step(name: str, notes: list[str], log: Callable, fn) -> None:
    """One optional fauna product: a failure costs that product, not the run."""
    try:
        fn()
    except Exception as exc:                                        # noqa: BLE001
        notes.append(f"{name} ({one_line(exc, 120)})")
        log(f"  {LOG_WARN}{name} skipped: {one_line(exc)}")


# --------------------------------------------------------------------------
# The registry (order = display order)
# --------------------------------------------------------------------------

PRODUCT_TYPES: list[ProductType] = [
    ProductType(
        key="frame_set", label="Frame Set",
        settings_schema=[
            ("sampling_mode", "Sampling", "choice", DEFAULTS["sampling_mode"], ["dynamic"]),
            ("spacing_m", "Target spacing (m)", "float", DEFAULTS["spacing_m"], None),
            ("min_frequency_hz", "Minimum frequency (Hz)", "float", DEFAULTS["min_frequency_hz"], None),
            ("reuse_tolerance_s", "Reuse tolerance (s)", "float", DEFAULTS["reuse_tolerance_s"], None),
            ("min_run_samples", "Min span (frames)", "int", DEFAULTS["min_run_samples"], None),
            ("reuse_legacy", "Recycle legacy batch frames", "bool", DEFAULTS["reuse_legacy"], None),
            ("reuse_mode", "Reuse by", "choice", "auto", ["auto", "hardlink", "symlink", "copy"]),
        ],
        _discover=_discover_frame_set, _generate=_generate_frame_set),
    ProductType(
        key="photogrammetry", label="Photogrammetry",
        settings_schema=_photogrammetry_schema(),
        _discover=_discover_photogrammetry, _generate=_generate_photogrammetry,
        _schema_fn=_photogrammetry_schema),
    ProductType(
        key="fauna_detection", label="Fauna Detection (CV)",
        settings_schema=_fauna_schema(),
        _discover=_discover_fauna, _generate=_generate_fauna,
        _schema_fn=_fauna_schema),
    ProductType(
        key="nav_trackline", label="Nav Trackline (GeoJSON)",
        settings_schema=[],
        _discover=_discover_nav_trackline, _generate=_generate_nav_trackline),
    ProductType(
        key="depth_raster", label="Depth Raster (GeoTIFF)",
        settings_schema=[
            ("cell_size_m", "Cell size (m)", "float", DEFAULTS["cell_size_m"], None),
            ("crs_mode", "CRS", "choice", DEFAULTS["crs_mode"], ["utm", "wgs84"]),
        ],
        _discover=_discover_depth_raster, _generate=_generate_depth_raster),
    ProductType(
        key="sensor_raster", label="Sensor Raster (GeoTIFF)",
        settings_schema=[
            ("channel", "Sensor channel", "choice", _FALLBACK_CHANNELS[0], list(_FALLBACK_CHANNELS)),
            ("cell_size_m", "Cell size (m)", "float", DEFAULTS["cell_size_m"], None),
            ("crs_mode", "CRS", "choice", DEFAULTS["crs_mode"], ["utm", "wgs84"]),
            ("fill_method", "Fill", "choice", DEFAULTS["fill_method"], ["idw", "none", "rbf"]),
        ],
        _discover=_discover_sensor_raster, _generate=_generate_sensor_raster,
        _schema_fn=_sensor_schema),
    ProductType(
        key="anomaly_detection", label="Anomaly Detection",
        settings_schema=[],
        _discover=_discover_anomaly, _generate=_generate_anomaly),
    ProductType(
        key="spectrum_trackline", label="Spectrum Trackline (PNG)",
        settings_schema=[
            ("channel", "Channel", "choice", "all", ["all"] + list(_FALLBACK_CHANNELS)),
        ],
        _discover=_discover_spectrum, _generate=_generate_spectrum_trackline,
        _schema_fn=lambda ws: [("channel", "Channel", "choice", "all",
                                ["all"] + sensor_channels(ws))]),
    ProductType(
        key="anomaly_trackline", label="Anomaly Trackline (PNG)",
        settings_schema=[
            ("transit_only", "Transit windows only", "bool", False, None),
        ],
        _discover=_discover_anomaly_trackline, _generate=_generate_anomaly_trackline),
    ProductType(
        key="survey_report", label="Survey Report",
        settings_schema=[("fast_mesh", "Fast mesh stats", "bool", True, None)],
        _discover=_discover_survey_report, _generate=_generate_survey_report),
]

PRODUCT_TYPES_BY_KEY: dict[str, ProductType] = {p.key: p for p in PRODUCT_TYPES}


def product_type(key: str) -> ProductType:
    return PRODUCT_TYPES_BY_KEY[key]


def discover_all(ws: str, job: Optional[Job] = None) -> dict[str, list[ProductInstance]]:
    """{type_key: instances} for one scope — the whole Products tree in one call."""
    scope = job or whole_job()
    return {p.key: p.discover(ws, scope) for p in PRODUCT_TYPES}


# --------------------------------------------------------------------------
# The default suite
# --------------------------------------------------------------------------

def default_run_all(ws: str, log_fn: Optional[Callable[[str], None]] = None,
                    job: Optional[Job] = None) -> None:
    """Produce every default product for a scope.

    Order: interp -> nav trackline + depth raster + one raster per sensor
    channel -> frame set -> photogrammetry -> anomaly (detector, catalog, UTM
    layers, window context) -> fauna detection -> spectrum + anomaly tracklines
    -> survey report.

    The sampling regime used throughout is the default one (WHOLE_TRACKLINE,
    dynamic 0.25 m); the three sampling-dependent steps (frame set,
    photogrammetry, fauna) therefore share ONE frame set, and the
    sampling-independent steps run exactly once.
    Every step is isolated: a failure is logged as ONE ``LOG_FAIL`` line naming
    the cause (its traceback follows as ``LOG_DETAIL`` lines the UI collapses)
    and the suite continues, so one missing dependency never kills the run.  The
    last line is always a verdict: how many steps failed, and which.
    """
    with run_lock(ws, f"default run ({(job or whole_job()).name})"):
        _default_run_all_locked(ws, log_fn, job)


def _default_run_all_locked(ws: str, log_fn: Optional[Callable[[str], None]] = None,
                            job: Optional[Job] = None) -> None:
    log = _logger(log_fn)
    scope = job or whole_job()
    started = time.time()
    log(f"################ DEFAULT RUN — {scope.name} ################")
    done: list[str] = []
    failed: list[str] = []

    def step(name: str, fn) -> None:
        done.append(name)
        try:
            log(f"--- {name} ---")
            t0 = time.time()
            result = fn()
            suffix = f" -> {Path(result.path).name}" if isinstance(result, ProductInstance) else ""
            log(f"--- {name}: ok in {time.time() - t0:.1f}s{suffix} ---")
        except Exception as exc:                                    # noqa: BLE001
            failed.append(name)
            log(f"{LOG_FAIL}{name} FAILED: {one_line(exc)}")
            for line in traceback.format_exc().rstrip().splitlines():
                log(LOG_DETAIL + line)

    step("interp_full.csv", lambda: ensure_interp(ws, log))
    step("nav trackline", lambda: product_type("nav_trackline").generate(ws, scope, {}, log))
    step("depth raster", lambda: product_type("depth_raster").generate(ws, scope, {}, log))
    for channel in sensor_channels(ws):
        step(f"sensor raster [{channel}]",
             lambda c=channel: product_type("sensor_raster").generate(
                 ws, scope, {"channel": c}, log))
    step("frame set", lambda: product_type("frame_set").generate(ws, scope, {}, log))
    step("photogrammetry", lambda: product_type("photogrammetry").generate(ws, scope, {}, log))
    step("anomaly detection", lambda: product_type("anomaly_detection").generate(ws, scope, {}, log))
    # after anomaly detection on purpose: the fauna products join detections to
    # the anomaly windows, and degrade (blank window columns) without them
    step("fauna detection", lambda: product_type("fauna_detection").generate(ws, scope, {}, log))
    step("spectrum trackline", lambda: product_type("spectrum_trackline").generate(ws, scope, {}, log))
    step("anomaly trackline", lambda: product_type("anomaly_trackline").generate(ws, scope, {}, log))
    step("survey report", lambda: product_type("survey_report").generate(ws, scope, {}, log))

    minutes = (time.time() - started) / 60
    if failed:
        log(f"{LOG_FAIL}DEFAULT RUN FINISHED WITH {len(failed)} OF {len(done)} "
            f"STEP(S) FAILED ({minutes:.1f} min): " + ", ".join(failed))
    else:
        log(f"################ DEFAULT RUN COMPLETE — {len(done)}/{len(done)} "
            f"steps ok ({minutes:.1f} min) ################")


# --------------------------------------------------------------------------
# Smoke test / self-check (read-only against a real workspace)
# --------------------------------------------------------------------------

def _selftest_gate() -> list[str]:
    """Unit-check the altitude-gate splitter on synthetic frames."""
    out = []
    times = list(range(40))
    # frames 0-14 low, 15-17 high (gate crossing), 18-39 low
    alt = [2.0] * 15 + [12.0, 9.5, 20.0] + [3.0] * 22
    spans = altitude_gate_spans(times, alt, alt_max_m=8.0, min_frames=15)
    out.append(f"  split on crossings: {[(s[0], s[-1], len(s)) for s in spans]}")
    assert len(spans) == 2, spans
    assert spans[0] == list(range(0, 15)), spans[0]
    assert spans[1] == list(range(18, 40)), spans[1]

    # a short low span is dropped (< min_frames)
    alt2 = [2.0] * 5 + [12.0] + [2.0] * 34
    spans2 = altitude_gate_spans(times, alt2, 8.0, 15)
    out.append(f"  short span dropped: {[(s[0], s[-1], len(s)) for s in spans2]}")
    assert len(spans2) == 1 and spans2[0][0] == 6, spans2

    # exactly-at-gate passes, NaN fails
    alt3 = [8.0] * 20 + [float("nan")] + [8.0] * 19
    spans3 = altitude_gate_spans(times, alt3, 8.0, 15)
    out.append(f"  alt == gate passes, NaN splits: "
               f"{[(s[0], s[-1], len(s)) for s in spans3]}")
    assert len(spans3) == 2 and len(spans3[0]) == 20 and len(spans3[1]) == 19, spans3

    # everything above the gate -> nothing survives
    assert altitude_gate_spans(times, [30.0] * 40, 8.0, 15) == []
    out.append("  all-high -> no photogrammetry spans: ok")

    return out


def _selftest_chunking() -> list[str]:
    """Unit-check the 250-350 frames-per-chunk policy."""
    out = []
    cap = int(DEFAULTS["chunk_size"])
    band_min = int(DEFAULTS["chunk_target_min"])
    assert (cap, band_min) == (350, 250), (cap, band_min)

    # a span at or under the cap is ONE chunk, even below the band
    for n in (15, 120, 249, 250, 300, 349, 350):
        assert chunk_sizes(n) == [n], (n, chunk_sizes(n))
    out.append(f"  n <= {cap} stays one chunk (15 … 350): ok")

    # over the cap: fewest parts the cap allows, near-equal, none over the cap
    cases = {351: [176, 175], 500: [250, 250], 603: [302, 301],
             700: [350, 350], 701: [234, 234, 233], 751: [251, 250, 250],
             1000: [334, 333, 333], 3500: [350] * 10, 3501: [319] * 3 + [318] * 8}
    for n, want in cases.items():
        got = chunk_sizes(n)
        assert got == want, (n, got, want)
        assert sum(got) == n and max(got) <= cap, (n, got)
        assert max(got) - min(got) <= 1, (n, got)
    out.append("  n > cap splits near-equally, never over the cap: "
               + ", ".join(f"{n}->{len(v)}×{max(v)}/{min(v)}" for n, v in
                           list(cases.items())[:4]))

    # inside the band wherever the arithmetic allows …
    for n in (500, 603, 700, 751, 1000, 3500):
        got = chunk_sizes(n)
        assert all(band_min <= s <= cap for s in got), (n, got)
    # … and closest-to-band equal splits when it cannot be reached
    assert chunk_sizes(360) == [180, 180], chunk_sizes(360)
    assert chunk_sizes(701) == [234, 234, 233], chunk_sizes(701)
    out.append("  band honoured where reachable; 360 -> [180, 180] otherwise: ok")

    # the number of chunks is minimal: one fewer would breach the cap
    for n in (351, 500, 603, 701, 1000, 3501):
        parts = len(chunk_sizes(n))
        assert parts == 1 or n / (parts - 1) > cap, (n, parts)
    out.append("  chunk count is the minimum the cap permits: ok")

    # photos are grouped in order, nothing lost or duplicated
    photos = [f"f{i:04d}.jpg" for i in range(603)]
    groups = chunk_photos(photos, cap, band_min)
    out.append(f"  chunking 603 frames (cap {cap}): {[len(g) for g in groups]}")
    assert [len(g) for g in groups] == [302, 301], groups
    assert [p for g in groups for p in g] == photos, "photo order changed"

    # an explicit smaller cap still wins, and "unlimited" stays one chunk
    assert [len(g) for g in chunk_photos(photos, 100)] == [87] + [86] * 6
    assert len(chunk_photos(photos, 0)) == 1
    assert chunk_sizes(0) == [] and chunk_photos([], cap) == []
    out.append("  explicit cap / unlimited / empty span: ok")
    return out


def _selftest_recycling() -> list[str]:
    """Unit-check the recycling coverage maths on synthetic frame manifests."""
    out = []
    desired = [float(t) for t in range(0, 100, 2)]          # 50 requested samples
    # a prior frame set covered 0-38 exactly (same deterministic grid)
    existing = [float(t) for t in range(0, 40, 2)]
    runs = coverage_runs(desired, existing, tolerance_s=0.5)
    out.append(f"  head-covered: {[(c, len(t)) for c, t in runs]}")
    assert [c for c, _ in runs] == [True, False], runs
    assert len(runs[0][1]) == 20 and len(runs[1][1]) == 30, runs
    reused = sum(len(t) for c, t in runs if c)
    out.append(f"  reuse {reused}/{len(desired)} samples, extract {len(desired) - reused}")

    # a hole in the middle -> covered / uncovered / covered
    existing2 = [float(t) for t in range(0, 100, 2) if not 40 <= t < 60]
    runs2 = coverage_runs(desired, existing2, 0.5)
    out.append(f"  middle hole: {[(c, len(t)) for c, t in runs2]}")
    assert [c for c, _ in runs2] == [True, False, True], runs2
    assert len(runs2[1][1]) == 10, runs2

    # tolerance: 0.4 s drift is still the same frame, 0.9 s is not
    assert coverage_runs([10.0], [10.4], 0.5)[0][0] is True
    assert coverage_runs([10.0], [10.9], 0.5)[0][0] is False
    out.append("  tolerance window (0.4 s reused, 0.9 s re-extracted): ok")

    # empty pool -> everything extracted; empty request -> nothing to do
    runs3 = coverage_runs(desired, [], 0.5)
    assert len(runs3) == 1 and runs3[0][0] is False and len(runs3[0][1]) == 50
    assert coverage_runs([], existing, 0.5) == []
    out.append("  empty pool / empty request: ok")

    # coalescing: a stuttering phase-mismatched pool collapses to real spans
    stutter = [(bool(i % 2), [float(i)]) for i in range(60)]        # 60 runs of 1
    stutter.append((False, [float(t) for t in range(60, 140)]))     # one real gap
    collapsed = coalesce_runs(stutter, 15)
    out.append(f"  coalesce 61 stutter runs -> {[(c, len(t)) for c, t in collapsed]}")
    assert len(collapsed) == 2, collapsed
    assert collapsed[0][0] is True and len(collapsed[0][1]) == 60, collapsed
    assert collapsed[1][0] is False and len(collapsed[1][1]) == 80, collapsed
    assert coalesce_runs([(False, [1.0, 2.0])], 15) == [(False, [1.0, 2.0])]
    out.append("  single short run (nothing to merge into) preserved: ok")

    # a whole-trackline grid composes exactly from interval grids (determinism)
    grid = np.arange(0.0, 1000.0, 2.5)
    whole = grid[_mask_intervals(grid, [Interval(0, 1000)])]
    piece_a = grid[_mask_intervals(grid, [Interval(0, 400)])]
    piece_b = grid[_mask_intervals(grid, [Interval(400.0001, 1000)])]
    runs4 = coverage_runs(whole, np.concatenate([piece_a, piece_b]), 0.5)
    assert all(covered for covered, _ in runs4), runs4
    out.append(f"  interval runs fully cover the whole-trackline grid "
               f"({len(whole)} samples, 0 re-extracted): ok")
    return out


def _smoke(ws: str) -> None:
    print(f"workspace: {ws}")
    print(f"exists: {Path(ws).is_dir()}")

    print("\n[load_jobs]")
    for job in load_jobs(ws):
        print(f"  {job.job_id:<12} {job.name:<28} {len(job.intervals)} interval(s)")

    print("\n[track_polyline]")
    track = track_polyline(ws)
    print(f"  shape={track.shape} dtype={track.dtype}")
    print(f"  t: {naive_utc(track[0, 0])} -> {naive_utc(track[-1, 0])} "
          f"({(track[-1, 0] - track[0, 0]) / 3600:.2f} h)")
    print(f"  easting  {track[:, 1].min():.1f} .. {track[:, 1].max():.1f}")
    print(f"  northing {track[:, 2].min():.1f} .. {track[:, 2].max():.1f}")

    print("\n[sampling_grid @ 0.25 m]")
    grid = sampling_grid(ws, 0.25, 0.1)
    print(f"  {len(grid):,} deterministic samples "
          f"({naive_utc(grid[0])} -> {naive_utc(grid[-1])})")

    print("\n[sensor_channels]")
    print("  " + ", ".join(sensor_channels(ws)))

    print("\n[discover — WHOLE_TRACKLINE]")
    whole = whole_job()
    total = 0
    for ptype in PRODUCT_TYPES:
        instances = ptype.discover(ws, whole)
        total += len(instances)
        print(f"  {ptype.key:<19} {len(instances):>3} instance(s)")
        for instance in instances[:2]:
            print(f"       · {instance.label}   [{len(instance.view_paths)} view(s)]")
        if len(instances) > 2:
            print(f"       … {len(instances) - 2} more")
    print(f"  TOTAL {total} instances")

    print("\n[frame pool scan]")
    pool = scan_frame_pool(ws, 0.25, "dynamic", True, log_fn=lambda m: print(" " + m))
    if len(pool):
        covered = coverage_runs(grid, pool["unix_time"].to_numpy(dtype=float), 0.5)
        spans = coalesce_runs(covered, DEFAULTS["min_run_samples"])
        reuse = sum(len(t) for c, t in spans if c)
        extract = sum(len(t) for c, t in spans if not c)
        print(f"  whole-dive grid: {reuse:,}/{len(grid):,} samples already on disk "
              f"({100.0 * reuse / max(1, len(grid)):.1f}% recyclable), "
              f"{extract:,} to extract")
        print(f"  spans: {len(covered)} raw -> {len(spans)} after coalescing "
              f"(min_run={DEFAULTS['min_run_samples']}): "
              f"{sum(1 for c, _ in spans if c)} reuse / "
              f"{sum(1 for c, _ in spans if not c)} extract")

    print("\n[unit: altitude gate]")
    for line in _selftest_gate():
        print(line)
    print("[unit: chunk band]")
    for line in _selftest_chunking():
        print(line)
    print("[unit: recycling coverage]")
    for line in _selftest_recycling():
        print(line)
    print("\nSMOKE_OK")


if __name__ == "__main__":
    import sys
    _smoke(sys.argv[1] if len(sys.argv) > 1
           else "/mnt/f/EPR_2026_PROCESSED/J1756_down.eprproj")
