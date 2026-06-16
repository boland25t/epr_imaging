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
    # summary dict: {"completed": int, "failed": int, "skipped": int, "paths": list}
    finished      = Signal(dict)
    # fatal error that aborts the whole stack
    error         = Signal(str)

    def __init__(self, plan: list[dict]) -> None:
        super().__init__()
        self._plan = plan
        # (scope_id, product_type, channel) -> run_dir created this execution
        self._created_runs: dict[tuple, str] = {}
        # task_id -> list of output dirs produced by that sampling task this execution
        self._sampling_outputs: dict[int, list[str]] = {}
        self._abort = False

    def abort(self) -> None:
        """Request graceful abort after the current step completes."""
        self._abort = True

    # -----------------------------------------------------------------------

    def run(self) -> None:
        try:
            from output_service import OutputService
            svc = OutputService(log_fn=self.log.emit)

            total     = len(self._plan)
            completed = 0
            failed    = 0
            skipped   = 0
            all_paths: list[str] = []

            for i, step in enumerate(self._plan, start=1):
                if self._abort:
                    self.log.emit("Stack aborted by user.")
                    break

                label = step.get("label", step.get("product_type", "step"))
                self.step_started.emit(i, total, label)

                try:
                    paths = self._run_step(svc, step)
                    if paths is None:
                        skipped += 1
                        continue
                    completed += 1
                    all_paths.extend(paths)
                    self.step_finished.emit(label, paths)
                except Exception as exc:  # noqa: BLE001 — one failed product must
                    failed += 1            # not abort the rest of the stack
                    self.step_failed.emit(label, str(exc))
                    self.log.emit(f"  ✗ {label}: {exc}")

            self.finished.emit({
                "completed": completed,
                "failed":    failed,
                "skipped":   skipped,
                "paths":     all_paths,
            })
        except Exception as exc:  # noqa: BLE001 — unexpected fatal error
            self.error.emit(str(exc))

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
        product   = step["product_type"]
        parent_3d = "sensor_3d" if product == "sensor_slices" else "nav_3d"
        run_glob  = kwargs.pop("_run_glob", None)

        # Prefer a run created earlier in THIS execution; else newest on disk.
        key     = (step["scope_id"], parent_3d, step.get("channel"))
        run_dir = self._created_runs.get(key)
        if not run_dir and run_glob:
            run_dir = self._latest_run_dir(run_glob)

        if not run_dir or not Path(run_dir).exists():
            self.log.emit(
                f"  ⚠ {step.get('label', product)}: no 3D run found to slice "
                f"(enable the matching 3D product, or generate it first). Skipping."
            )
            return None

        method = getattr(svc, "generate_sensor_slices_from_run"
                         if product == "sensor_slices"
                         else "generate_nav_slices_from_run")
        result = method(run_dir=run_dir, **kwargs)
        return self._as_path_list(result)

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
        svc = PipelineService(log_fn=self.log.emit)
        out_dirs = svc.run(config)
        paths = [str(p) for p in (out_dirs or [])]
        task_id = step.get("task_id")
        if task_id is not None:
            self._sampling_outputs[task_id] = paths
        return paths

    def _run_photogrammetry(self, step: dict) -> list[str]:
        import photogrammetry_service as ps

        kwargs  = dict(step.get("kwargs", {}))
        engine  = step.get("engine", "metashape")

        # Resolve frame directories.  depends_on_task_id → use the sampling
        # task's output dirs (each contains a "frames" subdir).  Fallback to
        # the manual frame_dir from kwargs when no dependency is specified.
        depends_on_task_id = step.get("depends_on_task_id")
        frame_dirs: list[str] = []

        if depends_on_task_id is not None:
            sampling_dirs = self._sampling_outputs.get(depends_on_task_id, [])
            for seg_dir in sampling_dirs:
                frames_sub = Path(seg_dir) / "frames"
                if frames_sub.exists():
                    frame_dirs.append(str(frames_sub))
                elif Path(seg_dir).exists():
                    frame_dirs.append(seg_dir)  # fallback: use the dir itself
            if not frame_dirs:
                self.log.emit(
                    f"  ⚠ Photogrammetry: linked sampling task #{depends_on_task_id} "
                    "produced no output this run.  Run the sampling task first, "
                    "or switch to a manual frame directory."
                )
                return []
        else:
            manual = kwargs.get("frame_dir", "").strip()
            if manual and Path(manual).is_dir():
                frame_dirs = [manual]
            else:
                self.log.emit(
                    "  ⚠ Photogrammetry: no frame directory specified or directory "
                    "does not exist.  Configure a frame source in the task settings."
                )
                return []

        all_products: list[str] = []
        for frame_dir in frame_dirs:
            run_dir = ps.prepare_run_dir(kwargs["output_root"], job_id=kwargs.get("job_id", 0))
            self.log.emit(f"Photogrammetry: processing {frame_dir}  →  {run_dir}")

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
                    log_fn=self.log.emit,
                )
            else:
                # COLMAP's dense (MVS) stage is driven by its own "Run dense
                # reconstruction (MVS)" toggle (run_mvs), NOT the Metashape-only
                # Dense Cloud group (build_dense, which stays at its default for
                # COLMAP tasks).  Fall back to build_dense for old saved stacks.
                run_mvs = bool(kwargs.get("run_mvs", kwargs.get("build_dense", True)))
                products = ps.run_colmap(
                    run_dir=run_dir,
                    frame_dir=frame_dir,
                    build_dense=run_mvs,
                    max_features=int(kwargs.get("max_features", 8192)),
                    matcher=kwargs.get("matcher", "Exhaustive"),
                    colmap_bin=kwargs.get("colmap_bin", "colmap"),
                    log_fn=self.log.emit,
                )

            all_products.extend(products.values())

        return all_products

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
