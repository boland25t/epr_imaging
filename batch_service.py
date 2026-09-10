"""Full-dive batch orchestration — the headless equivalent of driving the app
to produce every product for a dive.

Composes the existing Qt-free services rather than reimplementing them:
  ConfigService.load_workspace   -> nav / sensor / depth config as real objects
  VideoService.scan_directory    -> VideoRecords (start/end times)
  nav_segments.detect_moving_segments -> traverse segments == photogrammetry scopes
  PipelineService.run            -> distance-sampled frames per segment
  photogrammetry_service         -> align + dense + mesh + DEM + ortho per chunk
  OutputService                  -> trackline / sensor rasters / netCDF (survey scope)
  anomaly_service                -> anomaly detection + site catalogue

The GUI can expose this behind a single "Process entire dive" action; the same
call drives an overnight headless run.  Resumable (per-scope DONE markers) and
isolated (a failure in one scope is logged and the batch continues).
"""
from __future__ import annotations

import json
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

import nav_segments


# The settled J1756 photogrammetry recipe (see product_tree_ui memory): Medium
# dense (not Low — Low makes stretched depth-map curtains), Height Field surface
# with interpolation Disabled (nadir gaps stay honest holes), reference
# preselection on via nav, ortho on the DEM, pressure-depth Z.
DEFAULT_PHOTO_SETTINGS = dict(
    quality_threshold=0.5,
    align_accuracy="High",
    use_nav_reference=True,
    build_dense=True, dense_quality="Medium", depth_filter="Moderate",
    export_dense_ply=True,
    build_mesh=True, mesh_source="Dense cloud", mesh_surface="Height Field",
    # interpolation ENABLED: Disabled leaves a pitted/broken surface on traverse
    # meshes (fills nothing between sparse dense points).  Enabled gives a smooth
    # continuous surface (it also skirts a bit beyond the data — trim if needed).
    mesh_faces="Medium", mesh_interpolation="Enabled", export_mesh_obj=True,
    build_dem=True, export_dem=True, build_orthomosaic=True, ortho_surface="DEM",
    make_report=True, save_project=True,
)

SENSOR_CHANNELS = ["CO2 Concentration", "CH4 Concentration", "O2 Concentration",
                   "Salinity", "Temperature"]


def _naive_utc(ts: float) -> datetime:
    """Unix seconds -> naive-UTC datetime (the app's datetime convention)."""
    return datetime.fromtimestamp(ts, tz=timezone.utc).replace(tzinfo=None)


def segments_to_intervals(segments):
    """Moving traverse segments -> SelectedTimeRange intervals (photogrammetry
    scopes).  Mirrors anomaly_service.windows_to_intervals so segments register
    as first-class scopes the same way anomaly windows do."""
    from models import SelectedTimeRange
    out = []
    for i, s in enumerate(segments, 1):
        out.append(SelectedTimeRange(start_time=_naive_utc(s.t_start),
                                     end_time=_naive_utc(s.t_end),
                                     source=f"segment_{i:02d}"))
    return out


@dataclass
class DiveBatchConfig:
    workspace_dir: str
    spacing_m: float = 0.25
    v_min: float = 0.08
    chunk_size: int = 350
    photo_settings: dict = field(default_factory=lambda: dict(DEFAULT_PHOTO_SETTINGS))
    do_sensors: bool = True
    do_anomaly: bool = True
    do_photogrammetry: bool = True
    resume: bool = True


class BatchOrchestrator:
    def __init__(self, config: DiveBatchConfig, log_fn: Optional[Callable[[str], None]] = None):
        self.cfg = config
        self._log = log_fn or print
        from workspace_paths import PathResolver
        self.resolver = PathResolver(config.workspace_dir)
        self.interp = str(self.resolver.interp_full())

    def log(self, m: str) -> None:
        self._log(f"[{datetime.now():%m-%d %H:%M:%S}] {m}")

    # ---- workspace / inputs ----------------------------------------------
    def _load_workspace(self) -> dict:
        from config_service import ConfigService
        ws_json = Path(self.cfg.workspace_dir) / "workspace.json"
        return ConfigService.load_workspace(str(ws_json))

    def _scan_videos(self, data: dict):
        from video_service import VideoService
        fmt = data.get("filename_datetime_format") or "%Y_%m_%dT%H_%M_%S"
        vs = VideoService(fmt)
        videos, skipped = vs.scan_directory(data["video_directory"])
        self.log(f"videos: {len(videos)} indexed, {len(skipped)} skipped")
        return videos

    # ---- fast survey-scope products --------------------------------------
    def run_sensor_products(self) -> None:
        from output_service import OutputService
        svc = OutputService(log_fn=lambda m: self.log("    " + str(m)))
        sp = Path(self.resolver.survey_products())
        self.log("=== survey sensor / trackline / netCDF products ===")
        jobs = [
            ("nav_trackline", lambda d: svc.generate_nav_3d_ply(self.interp, d)),
            ("nav_depth",     lambda d: svc.generate_nav_2d_geotiff(self.interp, d)),
        ]
        for name, fn in jobs:
            self._safe(name, fn, sp / name)
        for ch in SENSOR_CHANNELS:
            slug = ch.split()[0].lower()
            self._safe(f"sensor_2d/{slug}",
                       lambda d, c=ch: svc.generate_sensor_2d_geotiff(self.interp, d, c),
                       sp / "sensor_2d")
            self._safe(f"sensor_netcdf/{slug}",
                       lambda d, c=ch: svc.generate_sensor_netcdf(self.interp, d, c),
                       sp / "sensor_netcdf")

    def _safe(self, label, fn, out_dir) -> None:
        try:
            Path(out_dir).mkdir(parents=True, exist_ok=True)
            p = fn(str(out_dir))
            self.log(f"    ok {label}: {Path(p).name if p else '(done)'}")
        except Exception as e:                                      # noqa: BLE001
            self.log(f"    !! {label} failed: {e!r}")

    def run_anomaly(self) -> None:
        import anomaly_service as anomaly
        reason = anomaly.matlab_unavailable_reason()
        if reason:
            self.log(f"=== anomaly: SKIP — MATLAB unavailable ({reason}) ===")
            return
        self.log("=== anomaly detection (GrapherMatrix) ===")
        repo = str(Path(__file__).resolve().parent)
        out_dir = str(self.resolver.anomaly_dir())
        # The detector writes its event CSVs next to interp_full.csv (MATLAB cwd);
        # the raw Renav CSV comes from the workspace's navigation config.
        event_root = Path(self.interp).resolve().parent / "grapher_matrix_figs"
        raw_nav = None
        nav = getattr(self, "_ws_data", {}).get("navigation_file")
        lat_src = getattr(nav, "latitude_source", None) if nav else None
        if lat_src is not None and getattr(lat_src, "csv_path", None):
            raw_nav = lat_src.csv_path
        try:
            det = anomaly.run_detector(repo, self.interp,
                                       log_fn=lambda m: self.log("    " + str(m)))
            anomaly.run_catalog(repo, interp_csv=self.interp,
                                raw_nav_csv=raw_nav, event_root=event_root,
                                out_dir=out_dir,
                                log_fn=lambda m: self.log("    " + str(m)))
            self.log(f"    anomaly products -> {out_dir}")
        except Exception as e:                                     # noqa: BLE001
            self.log(f"    !! anomaly failed: {e!r}")

    # ---- photogrammetry per moving segment -------------------------------
    def run_photogrammetry(self, videos) -> None:
        from pipeline_service import PipelineService, PipelineConfig
        import photogrammetry_service as ps
        data = self._ws_data

        reason = ps.metashape_unavailable_reason()
        # A subprocess (Windows exe) path is used when in-proc import fails; only
        # a hard "no engine at all" should abort.
        if reason and ps.metashape_driver() is None:
            self.log(f"=== photogrammetry: ABORT — {reason} ===")
            return

        segs = nav_segments.detect_moving_segments(self.interp, v_min=self.cfg.v_min)
        intervals = segments_to_intervals(segs)
        # keep only segments overlapping video coverage
        vcov = [(v.start_time, v.end_time) for v in videos]
        def covered(iv):
            return any(a <= iv.end_time and iv.start_time <= b for a, b in vcov)
        pairs = [(i + 1, s, iv) for i, (s, iv) in enumerate(zip(segs, intervals)) if covered(iv)]
        self.log(f"=== photogrammetry: {len(pairs)}/{len(segs)} segments have video coverage ===")

        root = Path(self.resolver.photogrammetry_root(self.resolver.survey_products()))
        done = skipped = failed = 0
        t0 = datetime.now()
        for idx, (si, seg, iv) in enumerate(pairs, 1):
            sd = root / f"seg{si:02d}"
            marker = sd / "DONE"
            if self.cfg.resume and marker.exists():
                self.log(f"seg{si:02d}: DONE — skip"); skipped += 1; continue
            self.log(f"--- seg{si:02d} {seg.iso_start}->{seg.iso_end} {seg.path_m:.0f} m "
                     f"({idx}/{len(pairs)}) ---")
            try:
                # 1) sample frames via the real pipeline (dynamic distance mode)
                cfg = PipelineConfig(
                    video_directory=Path(data["video_directory"]),
                    output_directory=sd,
                    job_id=si,
                    video_filename_time_format=data.get("filename_datetime_format", ""),
                    videos=videos,
                    selected_intervals=[iv],
                    navigation_file=data["navigation_file"],
                    sensor_files=data.get("sensor_files") or [],
                    depth_source=data.get("depth_source"),
                    speed_source=data.get("speed_source"),
                    sampling_mode="dynamic",
                    dynamic_target_spacing_m=float(self.cfg.spacing_m),
                    dynamic_min_frequency_hz=float(data.get("dynamic_min_frequency_hz", 0.1)),
                    sample_images=True,
                    selected_steps=["extract_frames"],
                    workspace_directory=self.cfg.workspace_dir,
                )
                # reuse already-sampled frames on resume; else run the real sampler
                frame_dirs = [str(p) for p in sd.glob("segment_*/frames")
                              if any(p.glob("*.jpg"))]
                if frame_dirs:
                    nfr = sum(len(list(Path(fd).glob("*.jpg"))) for fd in frame_dirs)
                    self.log(f"    reusing {nfr} already-sampled frames")
                else:
                    seg_dirs = PipelineService().run(cfg)
                    frame_dirs = [str(Path(d) / "frames") for d in seg_dirs
                                  if (Path(d) / "frames").is_dir()]
                    nfr = sum(len(list(Path(fd).glob("*.jpg"))) for fd in frame_dirs)
                    self.log(f"    sampled {nfr} frames into {len(frame_dirs)} interval dir(s)")
                if nfr < 15:
                    self.log(f"    <15 frames — skip seg{si:02d}"); skipped += 1; continue
                # 2) chunk + photogrammetry via the real engine.  Each chunk gets a
                # clean segNN/chunk_CC run dir (matches the validated fix_test layout).
                specs = ps.build_chunk_sets(frame_dirs, self.cfg.chunk_size)
                frame_sets = []
                for c, spec in enumerate(specs, 1):
                    run_dir = sd / f"chunk_{c:02d}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    frame_sets.append((spec["photos"], str(run_dir), self.interp, spec["label"]))
                self.log(f"    {len(frame_sets)} chunk(s)")
                res = ps.run_metashape_batch(
                    str(sd / "project.psx"), frame_sets,
                    log_fn=lambda m: self.log("      " + str(m)),
                    **self.cfg.photo_settings)
                nprod = sum(len(v) for v in (res or {}).values())
                marker.write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S} {nprod} products\n")
                self.log(f"--- seg{si:02d} done — {nprod} products ---")
                done += 1
            except Exception as e:                                 # noqa: BLE001
                failed += 1
                self.log(f"!! seg{si:02d} FAILED: {e!r}")
                self.log(traceback.format_exc())
            self.log(f"    progress: {done} done / {skipped} skipped / {failed} failed, "
                     f"{len(pairs)-idx} left, {(datetime.now()-t0).total_seconds()/3600:.1f} h")
        self.log(f"=== photogrammetry complete: {done} done, {skipped} skipped, {failed} failed ===")

    # ---- top level -------------------------------------------------------
    def run(self) -> None:
        self.log("################ FULL-DIVE BATCH ################")
        self._ws_data = self._load_workspace()
        # Only photogrammetry consumes the video index; skip the scan otherwise
        # (probing videos mid-upload can block on incomplete files).
        videos = self._scan_videos(self._ws_data) if self.cfg.do_photogrammetry else []
        if self.cfg.do_sensors:
            self.run_sensor_products()
        if self.cfg.do_anomaly:
            self.run_anomaly()
        if self.cfg.do_photogrammetry:
            self.run_photogrammetry(videos)
        self.log("################ BATCH COMPLETE ################")


def process_dive(workspace_dir, log_fn=None, **kwargs) -> None:
    """Convenience entry: produce every product for a dive.  kwargs override
    DiveBatchConfig fields (spacing_m, v_min, chunk_size, photo_settings,
    do_sensors, do_anomaly, do_photogrammetry, resume)."""
    cfg = DiveBatchConfig(workspace_dir=str(workspace_dir), **kwargs)
    BatchOrchestrator(cfg, log_fn=log_fn).run()


if __name__ == "__main__":
    import sys
    ws = sys.argv[1] if len(sys.argv) > 1 else "."
    process_dive(ws)
