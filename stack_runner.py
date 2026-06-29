"""
stack_runner.py — Sequential executor for a TaskStack.

The StackWorker runs a fully-resolved *plan* (a list of step dicts produced by
MainWindow) in a single background QThread.  Each step is executed
synchronously, in dependency order, by calling OutputService methods or the
photogrammetry_service directly.  Per-step signals let the UI show live
progress without spinning a thread per product.

Why a dedicated runner instead of chaining the existing per-product handlers:
the handlers (_generate_nav_2d, _generate_sensor_3d, …) each create their own
QThread and show their own completion dialogs — fine for one-off manual use,
but they cannot be composed into an ordered batch.  The runner calls the same
underlying service methods the handlers call, just sequentially and from one
worker thread.

Plan step schema (every key resolved by MainWindow before launch):

    {
        "label":        str,          # human-readable, shown in the log
        "product_type": str,          # PRODUCT_INFO key
        "scope_id":     str,          # "full" or "job_<id>"
        "channel":      str | None,   # per-channel products only
        "method":       str | None,   # OutputService method name (None ⇒ photogrammetry)
        "kwargs":       dict,         # resolved keyword args for `method`
        "engine":       str | None,   # photogrammetry engine ("metashape"/"colmap")
    }

For PNG-slice steps (product_type == "sensor_slices"/"nav_slices") the kwargs
carry a "_run_glob" key: a directory whose newest run_NNN subdir is used as the
slice target when this execution did not itself just create the matching 3D run.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from PySide6.QtCore import QObject, Signal


class StackWorker(QObject):
    """Executes a resolved product-stack plan in a background thread."""

    # index (1-based), total, label
    step_started  = Signal(int, int, str)
    # label, list[str] of output paths
    step_finished = Signal(str, list)
    # label, error message
    step_failed   = Signal(str, str)
    # individual log line from the underlying service
    log           = Signal(str)
    # per-step sub-progress 0–100 (e.g. frame extraction within a sampling step)
    sub_progress  = Signal(int)
    # per-step detail label (e.g. "video_003.MP4 | frame 42/120")
    sub_status    = Signal(str)
    # summary dict: {"completed": int, "failed": int, "skipped": int, "paths": list,
    #                "report": list[dict]}
    finished      = Signal(dict)
    # fatal error that aborts the whole stack
    error         = Signal(str)

    def __init__(self, plan: list[dict], log_file: Optional[str] = None) -> None:
        super().__init__()
        self._plan = plan
        # (scope_id, product_type, channel) -> run_dir created this execution
        self._created_runs: dict[tuple, str] = {}
        # (task_id, scope_id) -> output dirs produced by that sampling task+scope.
        # Keyed by scope so batched (multi-job) sampling feeds each job's own
        # photogrammetry step the correct frames.
        self._sampling_outputs: dict[tuple, list[str]] = {}
        self._abort = False
        # Path of the complete, uncapped task log file (None ⇒ no file logging).
        self._log_file = log_file
        self._log_fh = None

    def abort(self) -> None:
        """Request graceful abort after the current step completes."""
        self._abort = True

    # -----------------------------------------------------------------------

    @staticmethod
    def _ts() -> str:
        """Wall-clock HH:MM:SS prefix for log lines."""
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    def _file_write(self, msg: str = "") -> None:
        """Append one line to the task log file (uncapped).  No-op without a file.

        All file writes happen on the worker thread (the only thread that runs
        the pipeline), so there is no cross-thread file-handle contention.
        """
        if self._log_fh is not None:
            try:
                self._log_fh.write(msg + "\n")
            except Exception:  # noqa: BLE001 — never let logging break the run
                pass

    def _emit(self, msg: str = "") -> None:
        """Send a line to BOTH the task log file (uncapped) and the GUI log."""
        self._file_write(msg)
        self.log.emit(msg)

    def _sampling_status(self, text: str) -> None:
        """Sampling high-level phase → GUI detail label + task log file."""
        self.sub_status.emit(text)
        self._file_write(f"      [sampling] {text}")

    def _sampling_substatus(self, text: str) -> None:
        """Sampling per-file detail → GUI detail label + task log file (not GUI log)."""
        self.sub_status.emit(text)
        self._file_write(f"      [sampling] {text}")

    def _logln(self, msg: str = "") -> None:
        """Emit one log line (no timestamp — used for service passthrough)."""
        self._emit(msg)

    def _log_event(self, msg: str) -> None:
        """Emit a timestamped top-level event line."""
        self._emit(f"[{self._ts()}] {msg}")

    def run(self) -> None:
        # Open the task log file (best-effort; failure just disables file logging).
        if self._log_file:
            try:
                self._log_fh = open(self._log_file, "w", encoding="utf-8")
                from datetime import datetime
                self._log_fh.write(
                    f"EPR Imaging — task stack log\n"
                    f"started: {datetime.now().isoformat(timespec='seconds')}\n"
                    f"{'=' * 60}\n"
                )
                self.log.emit(f"Full task log → {self._log_file}")
            except Exception as exc:  # noqa: BLE001
                self._log_fh = None
                self.log.emit(f"⚠ Could not open task log file ({exc}); GUI log only.")
        try:
            from output_service import OutputService
            svc = OutputService(log_fn=self._emit)

            total     = len(self._plan)
            completed = 0
            failed    = 0
            skipped   = 0
            all_paths: list[str] = []
            report: list[dict] = []   # per-step record for the final report

            self._log_event(f"═══ Stack run started — {total} step(s) queued ═══")

            for i, step in enumerate(self._plan, start=1):
                if self._abort:
                    self._log_event("Stack aborted by user.")
                    break

                label = step.get("label", step.get("product_type", "step"))
                self.step_started.emit(i, total, label)
                self.sub_progress.emit(0)
                self._log_step_header(i, total, step, label)

                try:
                    paths = self._run_step(svc, step)
                    if paths is None:
                        skipped += 1
                        self._log_event(f"⊘ [{i}/{total}] SKIPPED — {label}")
                        report.append({"label": label, "status": "skipped", "paths": []})
                        continue
                    completed += 1
                    all_paths.extend(paths)
                    self._log_outputs(label, paths)
                    self.sub_progress.emit(100)
                    self._log_event(f"✓ [{i}/{total}] DONE — {label} ({len(paths)} file(s) written)")
                    self.step_finished.emit(label, paths)
                    report.append({"label": label, "status": "completed", "paths": list(paths)})
                except Exception as exc:  # noqa: BLE001 — one failed product must
                    failed += 1            # not abort the rest of the stack
                    self.step_failed.emit(label, str(exc))
                    self._log_event(f"✗ [{i}/{total}] FAILED — {label}: {exc}")
                    report.append({"label": label, "status": "failed", "paths": [], "error": str(exc)})

            self._log_event(
                f"═══ Stack run finished — {completed} completed, "
                f"{failed} failed, {skipped} skipped ═══"
            )
            self.finished.emit({
                "completed": completed,
                "failed":    failed,
                "skipped":   skipped,
                "paths":     all_paths,
                "report":    report,
            })
        except Exception as exc:  # noqa: BLE001 — unexpected fatal error
            self._file_write(f"FATAL: {exc}")
            self.error.emit(str(exc))
        finally:
            if self._log_fh is not None:
                try:
                    self._log_fh.flush()
                    self._log_fh.close()
                except Exception:  # noqa: BLE001
                    pass
                self._log_fh = None

    # -----------------------------------------------------------------------
    # Logging helpers
    # -----------------------------------------------------------------------

    def _log_step_header(self, i: int, total: int, step: dict, label: str) -> None:
        """Emit a descriptive header announcing a step and its resolved paths."""
        self._logln("")
        self._log_event(f"▶ [{i}/{total}] {label}")
        product = step.get("product_type", "")
        kwargs  = step.get("kwargs", {})
        scope   = step.get("scope_id")
        if scope:
            self._logln(f"      target scope : {scope}")
        if product == "photogrammetry":
            engine = step.get("engine", "?")
            self._logln(f"      engine       : {engine}")
            dep = step.get("depends_on_task_id")
            if dep is not None:
                self._logln(f"      frame source : sampling task #{dep} (resolved at run time)")
            elif kwargs.get("frame_dir"):
                self._logln(f"      frame source : {kwargs['frame_dir']}")
            self._logln(f"      output root  : {kwargs.get('output_root', '?')}")
        else:
            # Most output_service steps carry interp_path / output_dir in kwargs.
            if kwargs.get("interp_path"):
                self._logln(f"      input        : {kwargs['interp_path']}")
            if kwargs.get("output_dir"):
                self._logln(f"      output dir   : {kwargs['output_dir']}")
            ch = step.get("channel")
            if ch:
                self._logln(f"      channel      : {ch}")

    def _log_outputs(self, label: str, paths: list) -> None:
        """List every output path the step produced, each on its own line."""
        if not paths:
            self._logln("      (no files written)")
            return
        self._logln(f"      outputs ({len(paths)}):")
        for p in paths:
            self._logln(f"        → {p}")

    # -----------------------------------------------------------------------
    # Step dispatch
    # -----------------------------------------------------------------------

    def _run_step(self, svc, step: dict) -> Optional[list[str]]:
        """Execute one step.  Return list of output paths, or None to mark skipped."""
        product = step["product_type"]
        kwargs  = dict(step.get("kwargs", {}))

        if product == "sampling":
            return self._run_sampling(step)

        if product == "photogrammetry":
            return self._run_photogrammetry(step)

        if product in ("sensor_slices", "nav_slices"):
            return self._run_slices(svc, step, kwargs)

        method_name = step["method"]
        method = getattr(svc, method_name)
        result = method(**kwargs)

        # Capture 3D run directories so dependent slice steps can attach to them.
        if product in ("sensor_3d", "nav_3d") and isinstance(result, str):
            key = (step["scope_id"], product, step.get("channel"))
            self._created_runs[key] = result

        return self._as_path_list(result)

    def _run_slices(self, svc, step: dict, kwargs: dict) -> Optional[list[str]]:
        """Resolve the target 3D run directory, then generate PNG slices from it."""
        product      = step["product_type"]
        parent_3d    = "sensor_3d" if product == "sensor_slices" else "nav_3d"
        run_glob     = kwargs.pop("_run_glob", None)
        scale_glob   = kwargs.pop("_scale_source_glob", None)

        # Prefer a run created earlier in THIS execution; else newest on disk.
        key     = (step["scope_id"], parent_3d, step.get("channel"))
        run_dir = self._created_runs.get(key)
        if not run_dir and run_glob:
            run_dir = self._latest_run_dir(run_glob)

        if not run_dir or not Path(run_dir).exists():
            self._emit(
                f"  ⚠ {step.get('label', product)}: no 3D run found to slice "
                f"(enable the matching 3D product, or generate it first). Skipping."
            )
            return None

        # Inject the full-dataset colour range so interval slices stay on the
        # same scale as the full trackline (and as each other).
        if product == "sensor_slices" and scale_glob and "vmin" not in kwargs:
            scale_run = self._latest_run_dir(scale_glob)
            if scale_run and scale_run != run_dir:
                vmin, vmax = self._grid_scalar_range(scale_run)
                if vmin is not None:
                    kwargs["vmin"] = vmin
                    kwargs["vmax"] = vmax
                    self._emit(
                        f"  Colour scale: full-dataset range "
                        f"[{vmin:.4g}, {vmax:.4g}]"
                    )

        method = getattr(svc, "generate_sensor_slices_from_run"
                         if product == "sensor_slices"
                         else "generate_nav_slices_from_run")
        result = method(run_dir=run_dir, **kwargs)
        return self._as_path_list(result)

    def _grid_scalar_range(self, run_dir: str) -> tuple:
        """Return (min, max) raw scalar values from a grid.csv.gz, or (None, None)."""
        grid_path = Path(run_dir) / "grid.csv.gz"
        if not grid_path.exists():
            return None, None
        try:
            import pandas as pd
            df = pd.read_csv(str(grid_path))
            coord_cols = {"x", "y", "z", "ix", "iy", "iz"}
            scalar_cols = [c for c in df.columns if c not in coord_cols]
            if not scalar_cols:
                return None, None
            vals = df[scalar_cols[0]].dropna().values
            if len(vals) == 0:
                return None, None
            return float(vals.min()), float(vals.max())
        except Exception:
            return None, None

    def _run_sampling(self, step: dict) -> list[str]:
        """Run a frame-extraction pipeline pass from a prebuilt PipelineConfig.

        MainWindow constructs the PipelineConfig (it owns the video/nav/sensor
        context) and attaches it to the step under "config"; the worker just
        drives PipelineService synchronously.  Output dirs are recorded in
        _sampling_outputs keyed by task_id so downstream photogrammetry steps
        can resolve the frame directory without manual path entry.
        """
        from pipeline_service import PipelineService
        config = step.get("config")
        if config is None:
            raise ValueError("Sampling step has no PipelineConfig attached.")

        # Route the pipeline's rich play-by-play to the GUI.  Without this the
        # pipeline's _emit_log/_emit_status (which include every file path it
        # writes) only reach the Python logger during a stack run.
        config.log_callback         = self._emit             # file + GUI play-by-play
        config.status_callback      = self._sampling_status  # phase label (GUI) + file
        config.substatus_callback   = self._sampling_substatus  # per-file detail label (+file)
        config.progress_callback    = self.sub_progress.emit # overall step progress
        config.subprogress_callback = self.sub_progress.emit # per-video progress

        self._emit(f"      videos       : {len(config.videos)} loaded")
        self._emit(f"      intervals    : {len(config.selected_intervals)}")
        self._emit(f"      mode         : {config.sampling_mode}"
                      + (f" @ {config.frame_rate} Hz" if config.sampling_mode == 'fixed'
                         else f" @ ~{config.dynamic_target_spacing_m} m spacing"))
        self._emit(f"      steps        : {', '.join(config.selected_steps)}")
        self._emit(f"      output dir   : {config.output_directory}")

        svc = PipelineService(log_fn=self._emit)
        out_dirs = svc.run(config)
        paths = [str(p) for p in (out_dirs or [])]
        task_id = step.get("task_id")
        if task_id is not None:
            self._sampling_outputs[(task_id, step.get("scope_id"))] = paths
        return paths

    def _run_photogrammetry(self, step: dict) -> list[str]:
        import photogrammetry_service as ps

        kwargs  = dict(step.get("kwargs", {}))
        engine  = step.get("engine", "metashape")

        # Resolve frame directories.  depends_on_task_id → use the sampling
        # task's output dirs (each contains a "frames" subdir).  Fallback to
        # the manual frame_dir from kwargs when no dependency is specified.
        #
        # On any resolution failure we RAISE so the step is reported as FAILED
        # with a precise reason — never silently "completed / 0 files".
        depends_on_task_id = step.get("depends_on_task_id")
        frame_dirs: list[str] = []

        if depends_on_task_id is not None:
            # Match the sampling output for THIS scope (batching: one sampling
            # task fans into per-job outputs keyed by (task_id, scope_id)).
            scope_id = step.get("scope_id")
            sampling_dirs = self._sampling_outputs.get((depends_on_task_id, scope_id), [])
            self._emit(
                f"      frame source : sampling task #{depends_on_task_id} "
                f"[{scope_id}] → {len(sampling_dirs)} output dir(s) recorded this run"
            )
            if not sampling_dirs:
                raise RuntimeError(
                    f"Frame source is sampling task #{depends_on_task_id} for scope "
                    f"'{scope_id}', but it produced no output this run (recorded: "
                    f"{sorted(self._sampling_outputs.keys()) or 'none'}). "
                    "Ensure the sampling task ran successfully and is ordered ABOVE this "
                    "photogrammetry task, or switch the frame source to a manual directory."
                )
            for seg_dir in sampling_dirs:
                frames_sub = Path(seg_dir) / "frames"
                if frames_sub.is_dir() and any(frames_sub.iterdir()):
                    frame_dirs.append(str(frames_sub))
                    self._emit(f"        ✓ frames: {frames_sub}")
                elif frames_sub.is_dir():
                    self._emit(f"        ⚠ empty frames dir (skipped): {frames_sub}")
                elif Path(seg_dir).is_dir():
                    frame_dirs.append(seg_dir)  # fallback: images may sit directly here
                    self._emit(f"        ✓ frames (dir root): {seg_dir}")
                else:
                    self._emit(f"        ⚠ missing output dir (skipped): {seg_dir}")
            if not frame_dirs:
                raise RuntimeError(
                    f"Sampling task #{depends_on_task_id} produced output dir(s) but none "
                    "contained extracted frames. Check that the sampling task actually "
                    "extracted images (its 'frames' subfolder is non-empty)."
                )
        else:
            manual = kwargs.get("frame_dir", "").strip()
            self._emit(f"      frame source : manual directory → {manual or '(none set)'}")
            if not manual:
                raise RuntimeError(
                    "No frame source configured. Edit the photogrammetry task and either "
                    "link it to a sampling task in the stack, or set a manual frame directory."
                )
            if not Path(manual).is_dir():
                raise RuntimeError(f"Manual frame directory does not exist: {manual}")
            frame_dirs = [manual]

        self._emit(f"      frame sets   : {len(frame_dirs)} to process")

        all_products: list[str] = []
        for seg_i, frame_dir in enumerate(frame_dirs, start=1):
            run_dir = ps.prepare_run_dir(kwargs["output_root"], job_id=kwargs.get("job_id", 0))
            self._emit(
                f"[{self._ts()}] [photogrammetry] frame set {seg_i}/{len(frame_dirs)} — {engine}"
            )
            self._emit(f"      frames in    : {frame_dir}")
            self._emit(f"      run dir      : {run_dir}")

            if engine == "metashape":
                products = ps.run_metashape(
                    run_dir=run_dir,
                    frame_dir=frame_dir,
                    align_accuracy=kwargs.get("align_accuracy", "High"),
                    key_point_limit=int(kwargs.get("key_point_limit", 40000)),
                    tie_point_limit=int(kwargs.get("tie_point_limit", 10000)),
                    generic_preselect=bool(kwargs.get("generic_preselect", True)),
                    reference_preselect=bool(kwargs.get("reference_preselect", True)),
                    adaptive_fitting=bool(kwargs.get("adaptive_fitting", True)),
                    reset_cameras=bool(kwargs.get("reset_cameras", False)),
                    build_dense=bool(kwargs.get("build_dense", True)),
                    dense_quality=kwargs.get("dense_quality", "Medium"),
                    depth_filter=kwargs.get("depth_filter", "Moderate"),
                    reuse_depth=bool(kwargs.get("reuse_depth", False)),
                    build_mesh=bool(kwargs.get("build_mesh", False)),
                    mesh_surface=kwargs.get("mesh_surface", "Arbitrary"),
                    mesh_faces=kwargs.get("mesh_faces", "Medium"),
                    mesh_source=kwargs.get("mesh_source", "Dense cloud"),
                    mesh_vertex_colors=bool(kwargs.get("mesh_vertex_colors", True)),
                    build_texture=bool(kwargs.get("build_texture", False)),
                    texture_size=int(kwargs.get("texture_size", 4096)),
                    texture_blending=kwargs.get("texture_blending", "Mosaic"),
                    texture_fill_holes=bool(kwargs.get("texture_fill_holes", True)),
                    export_dense_ply=bool(kwargs.get("export_dense_ply", True)),
                    export_mesh_obj=bool(kwargs.get("export_mesh_obj", False)),
                    nav_csv=kwargs.get("nav_csv"),
                    use_nav_reference=bool(kwargs.get("use_nav_reference", True)),
                    nav_accuracy_h=float(kwargs.get("nav_accuracy_h", 0.1)),
                    nav_accuracy_v=float(kwargs.get("nav_accuracy_v", 0.5)),
                    save_project=bool(kwargs.get("save_project", True)),
                    log_fn=self._emit,
                )
            else:
                # COLMAP's dense (MVS) stage is driven by its own "Run dense
                # reconstruction (MVS)" toggle (run_mvs), NOT the Metashape-only
                # Dense Cloud group (build_dense, which stays at its default for
                # COLMAP tasks).  Fall back to build_dense for old saved stacks.
                run_mvs = bool(kwargs.get("run_mvs", kwargs.get("build_dense", True)))
                # Prefer the per-segment interp.csv beside the frames (it carries
                # this segment's frame_filename → position rows) for georeferencing;
                # fall back to the global interp passed from the plan.
                seg_nav = Path(frame_dir).parent / "interp.csv"
                nav_csv = str(seg_nav) if seg_nav.exists() else kwargs.get("colmap_nav_csv")
                self._emit(f"      nav for georef: {nav_csv or '(none — georef disabled)'}")
                products = ps.run_colmap(
                    run_dir=run_dir,
                    frame_dir=frame_dir,
                    nav_csv=nav_csv,
                    single_camera=bool(kwargs.get("single_camera", True)),
                    matcher=kwargs.get("matcher", "Exhaustive"),
                    max_features=int(kwargs.get("max_features", 8192)),
                    georeference=bool(kwargs.get("georeference", True)),
                    build_dense=run_mvs,
                    export_camera_trajectory=bool(kwargs.get("export_camera_trajectory", True)),
                    export_undistorted=bool(kwargs.get("export_undistorted", False)),
                    export_depth_maps=bool(kwargs.get("export_depth_maps", False)),
                    build_poisson_mesh=bool(kwargs.get("build_poisson_mesh", False)),
                    build_delaunay_mesh=bool(kwargs.get("build_delaunay_mesh", False)),
                    colmap_bin=kwargs.get("colmap_bin", "colmap"),
                    log_fn=self._emit,            # GUI (capped) + file
                    file_log_fn=self._file_write,  # full COLMAP output → file only
                )

            # Log each photogrammetry product, labelled by friendly name + path.
            self._emit(f"[{self._ts()}] [photogrammetry] products from frame set {seg_i}:")
            for key, path in products.items():
                self._emit(f"        [photogrammetry] {self._PHOTO_PRODUCT_LABELS.get(key, key):<22}: {path}")
            all_products.extend(products.values())

        return all_products

    # Friendly labels for photogrammetry product keys returned by the engines.
    _PHOTO_PRODUCT_LABELS = {
        "sparse_ply":            "Sparse cloud (PLY)",
        "dense_ply":             "Dense cloud (PLY)",
        "camera_trajectory_ply": "Camera trajectory (PLY)",
        "mesh_poisson_ply":      "Poisson mesh (PLY)",
        "mesh_delaunay_ply":     "Delaunay mesh (PLY)",
        "mesh_obj":              "Mesh (OBJ)",
        "mesh_textured_obj":     "Textured mesh (OBJ)",
        "texture_png":           "Texture (PNG)",
        "cameras_json":          "Camera poses (JSON)",
        "georef_txt":            "Georeference geo.txt",
        "undistorted_dir":       "Undistorted frames (dir)",
        "depth_maps_dir":        "Depth maps (dir)",
        "normal_maps_dir":       "Normal maps (dir)",
        "report_pdf":            "Processing report (PDF)",
        "metashape_psx":         "Metashape project (PSX)",
    }

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _as_path_list(result) -> list[str]:
        if result is None:
            return []
        if isinstance(result, str):
            return [result]
        return list(result)

    @staticmethod
    def _latest_run_dir(parent_glob: str) -> Optional[str]:
        """Return the newest run_NNN subdirectory under parent_glob, or None."""
        parent = Path(parent_glob)
        if not parent.exists():
            return None
        runs = sorted(
            (p for p in parent.glob("run_*") if p.is_dir()),
            key=lambda p: p.name,
        )
        return str(runs[-1]) if runs else None
