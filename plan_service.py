"""
plan_service.py — turn an ordered Task list into a flat runner execution plan.

This is the orchestration core of the Task Stack, extracted from MainWindow so
it can be reasoned about and tested WITHOUT a QApplication.  It is Qt-free, like
every other service in this app.

Two stages:

  1. SCOPE RESOLUTION — each Task carries a `target` ({"kind": "full"|"job"|
     "jobs"|"all_jobs"}).  Every target expands into zero or more *scopes*:

         Scope(scope_id, interp_path, output_dir, label, job)

     "full" yields one workspace-level scope; the job kinds yield one scope per
     named job.  Jobs without intervals are dropped and reported as skips.

  2. STEP EXPANSION — each (task, scope) pair becomes one or more step dicts for
     StackWorker.  Per-channel task types fan out over the channel list, so one
     "Sensor 3D PLY" task across 5 channels and 2 jobs produces 10 steps.

IMPORTANT INVARIANT — every target is resolved BEFORE the stack runs.  A task can
therefore never target a job that a later task in the same stack creates.  That
is why One-Click materialises its anomaly job at generation time rather than
chaining it behind the detector step.

MainWindow supplies the environment through PlanContext: plain data (workspace
path, jobs, loaded videos/sensors) plus a few callables for the things that
genuinely need GUI state — notably the PipelineConfig builders, which read a
large amount of widget state and therefore stay in MainWindow.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator, Optional

from models import Job, Task, TaskStack


# Fill-method labels (as shown in the GUI) → the token the services expect.
FILL_3CH = {
    "IDW fill": "idw", "Kriging fill": "kriging", "RBF fill": "rbf",
    "No fill": "none", "Trackline only (no fill)": "none", "Trackline only": "none",
}


@dataclass(frozen=True)
class Scope:
    """One concrete execution context for a task."""
    scope_id: str            # "full" | "job_<id>"
    interp_path: str         # sensor table this scope reads
    output_dir: str          # where products are written
    label: str               # human-readable, used in step labels
    job: Optional[Job]       # None for the full-dataset scope


@dataclass
class PlanContext:
    """Everything the planner needs from the application.

    Data fields are read directly; the callables cover work that depends on GUI
    state (or on disk) and is therefore owned by MainWindow.
    """
    workspace_path: str = ""
    pending_job: Optional[Job] = None
    job_history: list = field(default_factory=list)
    sensor_files: list = field(default_factory=list)
    videos: list = field(default_factory=list)

    # Paths
    interp_full_path: Callable[[], str] = lambda: ""
    outputs_root: Callable[[], str] = lambda: ""
    filtered_interp_for_job: Callable[[Job], str] = lambda job: ""
    job_output_dirname: Callable[[Job], str] = lambda job: ""
    # Full product base directory for a job scope.  Owned by MainWindow because
    # it depends on the workspace layout (legacy flat vs .eprproj bundle).
    job_output_dir: Optional[Callable[[Job], str]] = None

    # Channel discovery (reads interp_full.csv when it exists)
    available_channels: Callable[[], list] = lambda: []

    # Config builders — these read GUI state, so MainWindow keeps them.
    # Returning None means "cannot run"; the step is skipped.
    build_interp_config: Callable[[Task], Any] = lambda task: None
    build_sampling_config: Callable[[Task, Optional[Job], str], Any] = (
        lambda task, job, scope_id: None
    )

    # Per-channel units for NetCDF metadata.
    raster_channel_units: Callable[[str], str] = lambda channel: ""


@dataclass
class PlanResult:
    """The execution plan plus everything that was dropped building it."""
    steps: list = field(default_factory=list)
    skips: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def __bool__(self) -> bool:
        return bool(self.steps)


# --------------------------------------------------------------------------
# Scope resolution
# --------------------------------------------------------------------------
def available_jobs(ctx: PlanContext) -> list[tuple]:
    """[(job_id, name), ...] for jobs that have intervals.

    After "Save Job" the pending job and its deep copy in job_history share a
    job_id — dedupe by id (pending wins) so an "All jobs" target never fans out
    to the same job twice.
    """
    jobs: list[tuple] = []
    seen: set = set()
    pending = ctx.pending_job
    if pending is not None and pending.intervals:
        jobs.append((pending.job_id, pending.name or f"Job #{pending.job_id}"))
        seen.add(pending.job_id)
    for j in ctx.job_history:
        if j.intervals and j.job_id not in seen:
            jobs.append((j.job_id, j.name or f"Job #{j.job_id}"))
            seen.add(j.job_id)
    return jobs


def find_job(ctx: PlanContext, job_id: int) -> Optional[Job]:
    pool = ([ctx.pending_job] if ctx.pending_job is not None else []) + list(ctx.job_history)
    return next((j for j in pool if j.job_id == job_id), None)


def scope_full(ctx: PlanContext) -> Scope:
    return Scope("full", ctx.interp_full_path(), ctx.outputs_root(), "Full dataset", None)


def scope_for_job(ctx: PlanContext, job: Job) -> Scope:
    # Prefer the layout-aware resolver; fall back to the legacy flat formula so
    # older callers/tests that don't supply job_output_dir keep working.
    if ctx.job_output_dir is not None:
        out_dir = str(ctx.job_output_dir(job))
    else:
        out_dir = str(Path(ctx.workspace_path) / ctx.job_output_dirname(job) / "outputs")
    return Scope(
        f"job_{job.job_id}",
        ctx.filtered_interp_for_job(job),
        out_dir,
        job.name or f"Job #{job.job_id}",
        job,
    )


def resolve_task_scopes(ctx: PlanContext, task: Task, skips: list) -> list[Scope]:
    """Expand a task's target into concrete scopes (Axis-1 batching)."""
    # interp_full.csv is workspace-level — never fan it out per job.
    if task.task_type == "build_interp":
        return [scope_full(ctx)]

    tgt = task.target or {"kind": "full"}
    kind = tgt.get("kind", "full")
    if kind == "full":
        return [scope_full(ctx)]

    job_ids: list[int] = []
    if kind == "job":
        job_ids = [int(tgt.get("job_id", -1))]
    elif kind == "jobs":
        job_ids = [int(j.get("job_id", -1)) for j in tgt.get("jobs", [])]
    elif kind == "all_jobs":
        job_ids = [jid for jid, _name in available_jobs(ctx)]

    scopes: list[Scope] = []
    for jid in job_ids:
        job = find_job(ctx, jid)
        if job is not None and job.intervals:
            scopes.append(scope_for_job(ctx, job))
        else:
            skips.append(f"{task.display_label()} — job #{jid} has no intervals (skipped)")
    return scopes


def iter_task_scopes(ctx: PlanContext, stack: TaskStack, skips: list) -> Iterator[tuple]:
    """Yield (task, scope) for every task × its resolved scopes."""
    for task in stack.tasks:
        scopes = resolve_task_scopes(ctx, task, skips)
        if not scopes:
            skips.append(f"{task.display_label()} — target resolved to no runnable scope")
        for scope in scopes:
            yield task, scope


# --------------------------------------------------------------------------
# Step construction
# --------------------------------------------------------------------------
def step(product_type: str, scope_id: str, channel, method, kwargs: dict, label: str) -> dict:
    """A plain OutputService-dispatched step."""
    return {
        "label": label, "product_type": product_type, "scope_id": scope_id,
        "channel": channel, "method": method, "engine": None, "kwargs": kwargs,
    }


def build_plan(ctx: PlanContext, stack: TaskStack) -> PlanResult:
    """Turn the ordered Task list into runner step dicts (preserving user order)."""
    plan: list[dict] = []
    skips: list[str] = []

    for task, sc in iter_task_scopes(ctx, stack, skips):
        scope_id, interp_path, output_dir, job = (
            sc.scope_id, sc.interp_path, sc.output_dir, sc.job
        )
        tag = f"  [{sc.label}]"
        t = task.task_type
        s = task.settings
        channels = task.channels or ctx.available_channels()

        if t == "build_interp":
            config = ctx.build_interp_config(task)
            if config is None:
                continue
            plan.append({
                "label": "Build interp_full.csv",
                "product_type": "build_interp", "scope_id": "full",
                "channel": None, "method": None, "engine": None,
                "kwargs": {}, "config": config,
            })

        elif t == "sampling":
            config = ctx.build_sampling_config(task, job, scope_id)
            if config is None:
                continue
            plan.append({
                "label": f"{task.type_label}{tag}",
                "product_type": "sampling", "scope_id": scope_id,
                "task_id": task.task_id,   # runner records outputs keyed by this
                "channel": None, "method": None, "engine": None,
                "kwargs": {}, "config": config,
            })

        elif t == "nav_3d":
            plan.append(step(t, scope_id, None, "generate_nav_3d_ply", {
                "interp_path": interp_path, "output_dir": output_dir,
                "cell_size": float(s.get("cell_size", 1.0)),
            }, "Nav Trackline PLY" + tag))

        elif t == "nav_2d":
            plan.append(step(t, scope_id, None, "generate_nav_2d_geotiff", {
                "interp_path": interp_path, "output_dir": output_dir,
                "cell_size_m": float(s.get("cell_size", 5.0)),
                "crs_mode": "wgs84" if s.get("crs") == "WGS84" else "utm",
            }, "Nav Depth GeoTIFF" + tag))

        elif t == "sensor_3d":
            for ch in channels:
                plan.append(step(t, scope_id, ch, "generate_sensor_3d_ply", {
                    "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                    "cell_size": float(s.get("cell_size", 1.0)),
                    "aggregation": s.get("aggregation", "mean"),
                    "fill_method": FILL_3CH.get(s.get("fill", "IDW fill"), "idw"),
                    "zero_mask_pct": float(s.get("zero_mask", 5.0)),
                }, f"Sensor 3D PLY — {ch}" + tag))

        elif t == "sensor_2d":
            for ch in channels:
                plan.append(step(t, scope_id, ch, "generate_sensor_2d_geotiff", {
                    "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                    "cell_size_m": float(s.get("cell_size", 5.0)),
                    "crs_mode": "wgs84" if s.get("crs") == "WGS84" else "utm",
                    "fill_method": FILL_3CH.get(s.get("fill", "IDW fill"), "idw"),
                }, f"Sensor 2D GeoTIFF — {ch}" + tag))

        elif t == "depth_slice_geotiffs":
            fill = ("idw" if "IDW" in s.get("fill", "IDW fill")
                    else "rbf" if "RBF" in s.get("fill", "") else "none")
            for ch in channels:
                plan.append(step(t, scope_id, ch, "generate_depth_slice_geotiffs", {
                    "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                    "altitude_step": float(s.get("altitude_step", 5.0)),
                    "cell_size_m": float(s.get("cell_size", 2.0)),
                    "fill_method": fill,
                }, f"Depth-Slice GeoTIFFs — {ch}" + tag))

        elif t == "sensor_slices":
            color = s.get("color", "viridis") or "viridis"
            local_norm = bool(s.get("local_norm", False))
            manual_range = bool(s.get("manual_range", False)) and not local_norm
            full_sensor_3d = str(Path(ctx.outputs_root()) / "sensor_3d")
            for ch in channels:
                kw: dict = {
                    "altitude_step": float(s.get("altitude_step", 5.0)),
                    "pixels_per_cell": int(s.get("ppc", 4)),
                    "color_mode": color,
                    "log_scale": bool(s.get("log_scale", False)),
                    "local_norm": local_norm,
                    "_run_glob": str(Path(output_dir) / "sensor_3d" / ch),
                }
                if manual_range:
                    # Explicit user range wins over any auto-derived scale.
                    kw["vmin"] = float(s.get("vmin", 0.0))
                    kw["vmax"] = float(s.get("vmax", 1.0))
                elif scope_id != "full":
                    # Per-job runs: point the runner at the full-dataset sensor_3d
                    # tree so it can derive a shared colour range.
                    kw["_scale_source_glob"] = str(Path(full_sensor_3d) / ch)
                plan.append(step(t, scope_id, ch, None, kw,
                                 f"PNG Depth Slices — {ch}" + tag))

        elif t == "job_interp":
            # One interp.csv per interval of the target job. Works with or
            # without video (video coverage columns are added when videos are
            # loaded).  Needs a JOB scope — "Full dataset" has no intervals.
            if job is None:
                skips.append(
                    f"{task.display_label()} — target a Job (Full dataset has no "
                    "intervals; this task writes one interp.csv per interval)."
                )
                continue
            if not job.intervals:
                skips.append(
                    f"{task.display_label()} — job '{job.name or job.job_id}' has no intervals."
                )
                continue
            ivs = [
                (calendar.timegm(iv.start_time.timetuple()),
                 calendar.timegm(iv.end_time.timetuple()))
                for iv in job.intervals
            ]
            vids = [
                (calendar.timegm(v.start_time.timetuple()),
                 calendar.timegm(v.end_time.timetuple()),
                 v.filename)
                for v in ctx.videos
            ] if (ctx.videos and s.get("annotate_video", True)) else None
            plan.append(step(t, scope_id, None, "generate_job_interval_interps", {
                "interp_path": interp_path,
                "output_dir":  output_dir,
                "intervals":   ivs,
                "videos":      vids,
                "job_name":    job.name or f"Job #{job.job_id}",
            }, f"Job Interval interp.csv set ({len(ivs)} intervals)" + tag))

        elif t == "frame_stats":
            # Frame source resolved at runtime (depends_on sampling task or
            # manual dir), same as photogrammetry.
            plan.append({
                "label": "Frame Statistics" + tag,
                "product_type": "frame_stats", "scope_id": scope_id,
                "channel": None, "method": None, "engine": None,
                "depends_on_task_id": task.depends_on,
                "kwargs": {
                    "output_dir":     output_dir,
                    "frame_dir":      s.get("frame_dir", "").strip(),
                    "sharpness_min":  float(s.get("sharpness_min", 100.0)),
                    "brightness_min": float(s.get("brightness_min", 20.0)),
                    "brightness_max": float(s.get("brightness_max", 235.0)),
                },
            })

        elif t == "photogrammetry":
            # Frame source: either linked to a sampling task (depends_on) or
            # manual.  No "dir must exist" check here because depends_on dirs are
            # resolved at runtime by the runner after sampling has produced them.
            engine = "colmap" if s.get("engine", "Metashape") == "COLMAP" else "metashape"
            plan.append({
                "label": f"Photogrammetry ({s.get('engine', 'Metashape')})" + tag,
                "product_type": "photogrammetry", "scope_id": scope_id,
                "channel": None, "method": None, "engine": engine,
                "depends_on_task_id": task.depends_on,   # None → use frame_dir
                "kwargs": {
                    # output_root + job_id: runner calls prepare_run_dir per
                    # segment.  job_id is the REAL target job's id (0 = full
                    # dataset) so the photogrammetry tree and the batch
                    # project.psx land under the right job folder.
                    "output_root": str(Path(output_dir) / "photogrammetry"),
                    "job_id":      job.job_id if job is not None else 0,
                    # Metashape single-project chunking: max images per chunk.
                    "chunk_size":  int(s.get("chunk_size", 250)),
                    # manual fallback dir (empty when using depends_on)
                    "frame_dir":   s.get("frame_dir", "").strip(),
                    "nav_csv":     interp_path if s.get("use_nav_reference", True) else None,
                    # Alignment
                    "align_accuracy":     s.get("align_accuracy", "High"),
                    "key_point_limit":    int(s.get("key_point_limit", 40000)),
                    "tie_point_limit":    int(s.get("tie_point_limit", 10000)),
                    "generic_preselect":  bool(s.get("generic_preselect", True)),
                    "reference_preselect": bool(s.get("reference_preselect", True)),
                    "adaptive_fitting":   bool(s.get("adaptive_fitting", True)),
                    "reset_cameras":      bool(s.get("reset_cameras", False)),
                    # Dense cloud
                    "build_dense":   bool(s.get("build_dense", True)),
                    "dense_quality": s.get("dense_quality", "Medium"),
                    "depth_filter":  s.get("depth_filter", "Moderate"),
                    "reuse_depth":   bool(s.get("reuse_depth", False)),
                    # Mesh
                    "build_mesh":         bool(s.get("build_mesh", False)),
                    "mesh_surface":       s.get("mesh_surface", "Arbitrary"),
                    "mesh_faces":         s.get("mesh_faces", "Medium"),
                    "mesh_source":        s.get("mesh_source", "Dense cloud"),
                    "mesh_vertex_colors": bool(s.get("mesh_vertex_colors", True)),
                    "mesh_interpolation": s.get("mesh_interpolation", "Enabled"),
                    # Texture
                    "build_texture":      bool(s.get("build_texture", False)),
                    "texture_size":       int(s.get("texture_size", 4096)),
                    "texture_blending":   s.get("texture_blending", "Mosaic"),
                    "texture_fill_holes": bool(s.get("texture_fill_holes", True)),
                    # DEM + orthomosaic (Metashape subprocess worker)
                    "build_dem":          bool(s.get("build_dem", False)),
                    "export_dem":         bool(s.get("export_dem", False)),
                    "build_orthomosaic":  bool(s.get("build_orthomosaic", False)),
                    "ortho_surface":      s.get("ortho_surface", "DEM"),
                    "make_report":        bool(s.get("make_report", True)),
                    # Export & project
                    "export_dense_ply": bool(s.get("export_dense_ply", False)),  # ARCHIVED: dense PLY export off (dense build kept for mesh/DEM/ortho)
                    "export_mesh_obj":  bool(s.get("export_mesh_obj", False)),
                    "save_project":     bool(s.get("save_project", True)),
                    # Georeference
                    "use_nav_reference": bool(s.get("use_nav_reference", True)),
                    "nav_accuracy_h":    float(s.get("nav_accuracy_h", 0.1)),
                    "nav_accuracy_v":    float(s.get("nav_accuracy_v", 0.5)),
                    # COLMAP — matching / SfM
                    "max_features":  int(s.get("max_features", 8192)),
                    "matcher":       s.get("matcher", "Exhaustive"),
                    "single_camera": bool(s.get("single_camera", True)),
                    # COLMAP — products
                    "run_mvs":                  bool(s.get("run_mvs", True)),
                    "export_camera_trajectory": bool(s.get("export_camera_trajectory", True)),
                    "export_undistorted":       bool(s.get("export_undistorted", False)),
                    "export_depth_maps":        bool(s.get("export_depth_maps", False)),
                    "build_poisson_mesh":       bool(s.get("build_poisson_mesh", False)),
                    "build_delaunay_mesh":      bool(s.get("build_delaunay_mesh", False)),
                    # COLMAP — georeference (always pass interp so it CAN georef)
                    "georeference": bool(s.get("georeference", True)),
                    "colmap_nav_csv": interp_path,
                },
            })

        elif t == "qgis_project":
            plan.append(step(t, scope_id, None, "generate_qgis_project", {
                "output_dir": output_dir,
                "project_name": s.get("project_name", "EPR Survey"),
            }, "QGIS Project" + tag))

        elif t == "anomaly_detect":
            # Workspace-level: the detector analyses interp_full.csv as a whole
            # and the catalog covers the entire survey, so this never fans out
            # per job.  Handled directly by StackWorker (no OutputService
            # method), hence method=None.
            plan.append({
                "label": "Anomaly Detection + Catalog",
                "product_type": "anomaly_detect", "scope_id": "full",
                "task_id": task.task_id,
                "channel": None, "method": None, "engine": None,
                "kwargs": {
                    "run_detector": bool(s.get("run_detector", True)),
                    "run_catalog":  bool(s.get("run_catalog", True)),
                    "workspace_dir": ctx.workspace_path,
                    "interp_path":  ctx.interp_full_path(),
                },
                "config": None,
            })

        elif t == "qc_report":
            plan.append(step(t, scope_id, None, "generate_qc_report", {
                "interp_path": interp_path,
                "output_dir": output_dir,
                "sensor_files": ctx.sensor_files,
                "channels": task.channels or None,
                "max_gap_s": float(s.get("max_gap_s", 60.0)),
            }, "Data QC Report" + tag))

        elif t == "sensor_netcdf":
            for ch in channels:
                plan.append(step(t, scope_id, ch, "generate_sensor_netcdf", {
                    "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                    "cell_size": float(s.get("cell_size", 1.0)),
                    "aggregation": s.get("aggregation", "mean"),
                    "fill_method": FILL_3CH.get(s.get("fill", "IDW fill"), "idw"),
                    "units": ctx.raster_channel_units(ch),
                }, f"Sensor NetCDF — {ch}" + tag))

    return PlanResult(steps=plan, skips=skips)
