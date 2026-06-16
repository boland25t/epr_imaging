# config_service.py — Workspace JSON serialisation and deserialisation
#
# A "workspace" is a JSON file that captures the complete state of the GUI:
# paths to video and sensor files, navigation configuration, selected time
# intervals, pipeline settings, and which processing steps have already run.
# Saving and loading workspaces lets the user resume a project without
# re-entering all settings from scratch.
#
# Design decisions:
#   - All file paths in the JSON are stored *relative* to the workspace file.
#     This makes the workspace portable: the user can move the project folder
#     to a different drive or share it with a colleague.
#   - Absolute paths are kept if the file is outside the workspace directory
#     (e.g. a video drive mounted separately).
#   - Numbers, booleans, and strings are stored verbatim; datetimes are stored
#     as ISO 8601 strings.
#
# There are two separate save/load methods:
#   save_json / (no load counterpart) — older format written alongside output;
#                                        kept for backward compatibility.
#   save_workspace / load_workspace   — full round-trip format used by the GUI.

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from models import (
    AnnotationConfig,
    Job,
    NavigationConfig,
    SegmentRecord,
    SelectedTimeRange,
    SensorChannel,
    SensorFileConfig,
    ThresholdConfig,
    TimeValueSourceConfig,
    VideoRecord,
)


# ---------------------------------------------------------------------------
# Module-level path utilities
# ---------------------------------------------------------------------------

def _to_relative(path: str | Path | None, base: Path) -> str | None:
    """Return path as a string relative to base directory, or None if path is None/empty.

    If path is not under base (e.g. a video drive mounted at a different root),
    the absolute path string is returned unchanged so the file can still be
    found after loading.

    Args:
        path: The path to convert (may be absolute or relative, or None).
        base: The directory to compute the relative path from (typically the
              directory containing the workspace .json file).

    Returns:
        A relative path string, an absolute path string, or None.
    """
    if not path:
        return None
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        # Path is not under base — return absolute so the file remains locatable.
        return str(path)


def _resolve(path: str | None, base: Path) -> Path | None:
    """Resolve a (possibly relative) path string against base directory.

    Relative paths are joined with base and resolved to an absolute path.
    Absolute paths are returned as-is.  None/empty input returns None.

    Args:
        path: A path string read from the workspace JSON (may be relative).
        base: The directory of the workspace file, used as the resolution root.

    Returns:
        An absolute Path, or None if path is empty/None.
    """
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    # Resolve turns ".." components into canonical absolute paths.
    return (base / p).resolve()


# ---------------------------------------------------------------------------
# ConfigService
# ---------------------------------------------------------------------------

class ConfigService:
    """Static-method collection for workspace file I/O.

    All methods are @staticmethod because there is no instance state; the class
    acts as a namespace.  Callers use ConfigService.save_workspace() and
    ConfigService.load_workspace() directly.
    """

    # ---------------------------------------------------------------------------
    # Legacy JSON save (alongside pipeline output)
    # ---------------------------------------------------------------------------

    @staticmethod
    def save_json(
        output_path: str | Path,
        video_directory: str | Path,
        video_filename_time_format: str,
        videos: list[VideoRecord],
        sensor_files: list[SensorFileConfig],
        selected_intervals: list[SelectedTimeRange],
        navigation_file: NavigationConfig | None = None,
        output_directory: str | Path | None = None,
        frame_rate: float | None = None,
        generate_sensor_rasters: bool | None = None,
        annotate_frames: bool | None = None,
        depth_source: SensorFileConfig | None = None,
        speed_source: SensorFileConfig | None = None,
        altitude_threshold: float | None = None,
        depth_threshold: float | None = None,
        speed_threshold: float | None = None,
        min_segment_frames: int | None = None,
    ) -> None:
        """Write a pipeline configuration JSON alongside the output directory.

        This is a legacy format written by the pipeline into the output folder
        so the exact settings used for a run can always be recovered later.
        File paths are stored as absolute strings (not relative to the JSON
        file) because the output directory may be far from the source data.

        Args:
            output_path: Where to write the JSON file.
            All other parameters: pipeline settings to serialise.
        """
        payload = {
            "video_directory": str(video_directory),
            "video_filename_time_format": video_filename_time_format,
            "output_directory": str(output_directory) if output_directory else None,
            "frame_rate": frame_rate,
            "generate_sensor_rasters": generate_sensor_rasters,
            "annotate_frames": annotate_frames,
            "altitude_threshold": altitude_threshold,
            "depth_threshold": depth_threshold,
            "speed_threshold": speed_threshold,
            "min_segment_frames": min_segment_frames,
            "depth_source": depth_source.to_dict() if depth_source else None,
            "speed_source": speed_source.to_dict() if speed_source else None,
            "videos": [video.to_dict() for video in videos],
            "navigation_file": navigation_file.to_dict() if navigation_file else None,
            "sensor_files": [sensor.to_dict() for sensor in sensor_files],
            "selected_intervals": [interval.to_dict() for interval in selected_intervals],
        }
        output_path = Path(output_path)
        # indent=2 produces a human-readable file; no minification needed here.
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Workspace save
    # ---------------------------------------------------------------------------

    @staticmethod
    def save_workspace(
        path: str | Path,
        *,
        video_directory: str,
        filename_datetime_format: str,
        navigation_file: NavigationConfig | None,
        sensor_files: list[SensorFileConfig],
        pending_job: "Job | None",
        next_job_id: int,
        segment_history: "list[SegmentRecord]",
        job_history: "list[Job] | None" = None,
        threshold_history: "list[ThresholdConfig] | None" = None,
        annotation_config: "AnnotationConfig | None" = None,
        frame_rate: float,
        generate_sensor_tiffs: bool,
        annotate_frames: bool,
        frame_quality: str = "Original",
        depth_source: SensorFileConfig | None = None,
        speed_source: SensorFileConfig | None = None,
        altitude_threshold: float | None = None,
        depth_threshold: float | None = None,
        speed_threshold: float | None = None,
        min_segment_frames: int = 1,
        sampling_mode: str = "fixed",
        dynamic_target_spacing_m: float = 2.0,
        dynamic_min_frequency_hz: float = 0.1,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
        workspace_path: str = "",
        # Outputs tab
        out_nav_2d_cell_size: float = 5.0,
        out_nav_2d_crs: str = "utm",
        out_nav_3d_cell_size: float = 1.0,
        out_nav_slices_step: float = 5.0,
        out_nav_slices_ppc: int = 4,
        out_sensor_2d_cell_size: float = 5.0,
        out_sensor_2d_crs: str = "utm",
        out_sensor_2d_fill: str = "IDW fill",
        out_sensor_3d_cell_size: float = 1.0,
        out_sensor_3d_agg: str = "mean",
        out_sensor_3d_fill: str = "IDW fill",
        out_sensor_slices_step: float = 5.0,
        out_sensor_slices_ppc: int = 4,
        out_sensor_slices_color: str = "rgb (viridis)",
        out_sensor_slices_log: bool = False,
        out_sensor_slices_pct: float = 100.0,
        **extra_settings: Any,
    ) -> None:
        """Serialise the complete GUI state to a portable workspace JSON file.

        The workspace directory (parent of this file) serves as the root for
        all job output: job_001/, job_002/, etc.  File paths inside the JSON
        are made relative to this directory wherever possible so the project
        folder can be moved or shared without breaking references.

        Args:
            path: Destination path for the workspace .json file.
            All keyword-only arguments: current GUI settings to save.
            **extra_settings: Any additional flat (JSON-serialisable) GUI
                settings.  These are merged into the payload verbatim, which
                lets the GUI add new persisted settings (e.g. task_stack,
                photogrammetry options) without this signature having to track
                every one.  load_workspace() passes them back through
                unchanged.  Only use this for plain scalars / dicts / lists —
                anything needing path-relativisation or object reconstruction
                must be an explicit parameter handled above.
        """
        base = Path(path).resolve().parent

        def rel_time_value_source(src: TimeValueSourceConfig) -> dict:
            d = src.to_dict()
            d["csv_path"] = _to_relative(src.csv_path, base)
            return d

        def rel_sensor_file(sf: SensorFileConfig) -> dict:
            d = sf.to_dict()
            d["csv_path"] = _to_relative(sf.csv_path, base)
            return d

        def rel_nav(nav: NavigationConfig) -> dict:
            d = nav.to_dict()
            d["latitude_source"]  = rel_time_value_source(nav.latitude_source)
            d["longitude_source"] = rel_time_value_source(nav.longitude_source)
            for key, src in (
                ("altitude_source", nav.altitude_source),
                ("depth_source",    nav.depth_source),
                ("heading_source",  nav.heading_source),
                ("pitch_source",    nav.pitch_source),
                ("roll_source",     nav.roll_source),
            ):
                d[key] = rel_time_value_source(src) if src is not None else None
            return d

        payload = {
            "video_directory":          _to_relative(video_directory, base) or video_directory,
            "filename_datetime_format": filename_datetime_format,
            "navigation_file":          rel_nav(navigation_file) if navigation_file else None,
            "sensor_files":             [rel_sensor_file(sf) for sf in sensor_files],

            # Job structure
            "next_job_id":        next_job_id,
            "pending_job":        pending_job.to_dict() if pending_job else None,
            "segment_history":    [r.to_dict() for r in segment_history],
            "job_history":        [j.to_dict() for j in (job_history or [])],
            "threshold_history":  [t.to_dict() for t in (threshold_history or [])],
            "annotation_config":  annotation_config.to_dict() if annotation_config else None,

            # Pipeline settings
            "frame_rate":            frame_rate,
            "generate_sensor_tiffs": generate_sensor_tiffs,
            "annotate_frames":       annotate_frames,
            "frame_quality":         frame_quality,

            # Threshold filters
            "altitude_threshold": altitude_threshold,
            "depth_threshold":    depth_threshold,
            "speed_threshold":    speed_threshold,
            "min_segment_frames": min_segment_frames,

            # Optional depth and speed sensor sources
            "depth_source": rel_sensor_file(depth_source) if depth_source else None,
            "speed_source": rel_sensor_file(speed_source) if speed_source else None,

            # Dynamic sampling parameters
            "sampling_mode":            sampling_mode,
            "dynamic_target_spacing_m": dynamic_target_spacing_m,
            "dynamic_min_frequency_hz": dynamic_min_frequency_hz,

            # CLAHE contrast-enhancement parameters
            "clahe_clip_limit":     clahe_clip_limit,
            "clahe_tile_grid_size": clahe_tile_grid_size,

            # Original workspace file path (used to restore output dir root after
            # auto-restore from last_session.json on next startup)
            "workspace_path": workspace_path,

            # Outputs tab settings — persisted so the user doesn't re-enter
            # preferred cell sizes, fill methods, etc. every session.
            "out_nav_2d_cell_size":    out_nav_2d_cell_size,
            "out_nav_2d_crs":          out_nav_2d_crs,
            "out_nav_3d_cell_size":    out_nav_3d_cell_size,
            "out_nav_slices_step":     out_nav_slices_step,
            "out_nav_slices_ppc":      out_nav_slices_ppc,
            "out_sensor_2d_cell_size": out_sensor_2d_cell_size,
            "out_sensor_2d_crs":       out_sensor_2d_crs,
            "out_sensor_2d_fill":      out_sensor_2d_fill,
            "out_sensor_3d_cell_size": out_sensor_3d_cell_size,
            "out_sensor_3d_agg":       out_sensor_3d_agg,
            "out_sensor_3d_fill":      out_sensor_3d_fill,
            "out_sensor_slices_step":  out_sensor_slices_step,
            "out_sensor_slices_ppc":   out_sensor_slices_ppc,
            "out_sensor_slices_color": out_sensor_slices_color,
            "out_sensor_slices_log":   out_sensor_slices_log,
            "out_sensor_slices_pct":   out_sensor_slices_pct,
        }

        # Merge forward-compatible flat settings (task_stack, photogrammetry
        # options, geo-slice settings, etc.).  Explicit payload keys win so a
        # stray duplicate in extra_settings can't clobber a relativised path.
        for key, value in extra_settings.items():
            payload.setdefault(key, value)

        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    # ---------------------------------------------------------------------------
    # Workspace load
    # ---------------------------------------------------------------------------

    @staticmethod
    def load_workspace(path: str | Path) -> dict:
        """Read a workspace JSON file and return a dict of GUI-ready values.

        Relative paths inside the JSON are resolved back to absolute paths
        using the workspace file's parent directory as the base.  Missing or
        unknown keys receive sensible defaults so old workspace files remain
        loadable after the schema gains new fields.

        Args:
            path: Path to the workspace .json file.

        Returns:
            A dict whose keys match the keyword arguments of save_workspace()
            plus the resolved config objects (NavigationConfig, SensorFileConfig
            lists, etc.).
        """
        path = Path(path).resolve()
        base = path.parent  # Resolution root for relative paths inside the file.
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"Could not load workspace file '{path.name}': {exc}\n"
                "The file may be corrupted or from an incompatible version."
            ) from exc

        # --- Reconstruct NavigationConfig ---
        navigation_file = None
        if data.get("navigation_file"):
            nav = data["navigation_file"]
            def _opt_src(key: str) -> "TimeValueSourceConfig | None":
                return (ConfigService._load_time_value_source(nav[key], base)
                        if nav.get(key) else None)
            navigation_file = NavigationConfig(
                latitude_source=ConfigService._load_time_value_source(nav["latitude_source"], base),
                longitude_source=ConfigService._load_time_value_source(nav["longitude_source"], base),
                altitude_source=_opt_src("altitude_source"),
                depth_source=_opt_src("depth_source"),
                negate_depth=bool(nav.get("negate_depth", False)),
                heading_source=_opt_src("heading_source"),
                pitch_source=_opt_src("pitch_source"),
                roll_source=_opt_src("roll_source"),
            )

        # --- Reconstruct sensor file list ---
        sensor_files = [
            ConfigService._load_sensor_file(item, base)
            for item in data.get("sensor_files", [])
        ]

        # --- Reconstruct job structure ---
        next_job_id = int(data.get("next_job_id", 1))

        pending_job: Job | None = None
        if data.get("pending_job"):
            pj = data["pending_job"]
            pending_job = Job(
                job_id=int(pj["job_id"]),
                name=pj.get("name", ""),
                intervals=[ConfigService._load_interval(i) for i in pj.get("intervals", [])],
                status=pj.get("status", "pending"),
                settings_snapshot=pj.get("settings_snapshot", {}),
            )

        segment_history: list[SegmentRecord] = []
        for rec in data.get("segment_history", []):
            segment_history.append(SegmentRecord(
                job_id=int(rec["job_id"]),
                job_name=rec.get("job_name", ""),
                interval=ConfigService._load_interval(rec["interval"]),
                output_path=rec.get("output_path", ""),
                status=rec.get("status", "completed"),
                processed_at=ConfigService._parse_dt(rec.get("processed_at")) or datetime.utcnow(),
                settings_snapshot=rec.get("settings_snapshot", {}),
            ))

        job_history: list[Job] = []
        for jh in data.get("job_history", []):
            job_history.append(Job(
                job_id=int(jh["job_id"]),
                name=jh.get("name", ""),
                intervals=[ConfigService._load_interval(i) for i in jh.get("intervals", [])],
                status=jh.get("status", "pending"),
                settings_snapshot=jh.get("settings_snapshot", {}),
            ))

        threshold_history: list[ThresholdConfig] = [
            ThresholdConfig.from_dict(t) for t in data.get("threshold_history", [])
        ]

        # --- Resolve video directory path ---
        video_dir = _resolve(data.get("video_directory"), base)

        # --- Optional depth and speed sources ---
        depth_source = ConfigService._load_sensor_file(data["depth_source"], base) if data.get("depth_source") else None
        speed_source = ConfigService._load_sensor_file(data["speed_source"], base) if data.get("speed_source") else None

        # --- Build and return the complete settings dict ---
        result = {
            "video_directory":          str(video_dir) if video_dir else "",
            "filename_datetime_format": data.get("filename_datetime_format", ""),
            "navigation_file":          navigation_file,
            "sensor_files":             sensor_files,
            "next_job_id":              next_job_id,
            "pending_job":              pending_job,
            "segment_history":          segment_history,
            "job_history":              job_history,
            "threshold_history":        threshold_history,
            "annotation_config":        AnnotationConfig.from_dict(data["annotation_config"])
                                        if data.get("annotation_config") else None,
            "frame_rate":               float(data.get("frame_rate", 1.0)),
            "generate_sensor_tiffs":    bool(data.get("generate_sensor_tiffs", True)),
            "annotate_frames":          bool(data.get("annotate_frames", False)),
            "frame_quality":            data.get("frame_quality", "Original"),
            "altitude_threshold":       data.get("altitude_threshold"),
            "depth_threshold":          data.get("depth_threshold"),
            "speed_threshold":          data.get("speed_threshold"),
            "min_segment_frames":       int(data.get("min_segment_frames", 1)),
            "depth_source":             depth_source,
            "speed_source":             speed_source,
            "sampling_mode":            data.get("sampling_mode", "fixed"),
            "dynamic_target_spacing_m": float(data.get("dynamic_target_spacing_m", 2.0)),
            "dynamic_min_frequency_hz": float(data.get("dynamic_min_frequency_hz", 0.1)),
            "clahe_clip_limit":         float(data.get("clahe_clip_limit", 2.0)),
            "clahe_tile_grid_size":     int(data.get("clahe_tile_grid_size", 8)),
            "workspace_path":           data.get("workspace_path", ""),
            # Outputs tab
            "out_nav_2d_cell_size":     float(data.get("out_nav_2d_cell_size", 5.0)),
            "out_nav_2d_crs":           data.get("out_nav_2d_crs", "utm"),
            "out_nav_3d_cell_size":     float(data.get("out_nav_3d_cell_size", 1.0)),
            "out_nav_slices_step":      float(data.get("out_nav_slices_step", 5.0)),
            "out_nav_slices_ppc":       int(data.get("out_nav_slices_ppc", 4)),
            "out_sensor_2d_cell_size":  float(data.get("out_sensor_2d_cell_size", 5.0)),
            "out_sensor_2d_crs":        data.get("out_sensor_2d_crs", "utm"),
            "out_sensor_2d_fill":       data.get("out_sensor_2d_fill", "IDW fill"),
            "out_sensor_3d_cell_size":  float(data.get("out_sensor_3d_cell_size", 1.0)),
            "out_sensor_3d_agg":        data.get("out_sensor_3d_agg", "mean"),
            "out_sensor_3d_fill":       data.get("out_sensor_3d_fill", "IDW fill"),
            "out_sensor_slices_step":   float(data.get("out_sensor_slices_step", 5.0)),
            "out_sensor_slices_ppc":    int(data.get("out_sensor_slices_ppc", 4)),
            "out_sensor_slices_color":  data.get("out_sensor_slices_color", "rgb (viridis)"),
            "out_sensor_slices_log":    bool(data.get("out_sensor_slices_log", False)),
            "out_sensor_slices_pct":    float(data.get("out_sensor_slices_pct", 100.0)),
        }

        # Pass through any remaining flat settings saved via save_workspace's
        # **extra_settings catch-all (task_stack, photo_* options, geo-slice
        # settings, qgis_project_name, …).  setdefault keeps the reconstructed
        # objects above authoritative; only genuinely unhandled keys are added.
        for key, value in data.items():
            result.setdefault(key, value)

        return result

    # ---------------------------------------------------------------------------
    # Private deserialization helpers
    # ---------------------------------------------------------------------------

    @staticmethod
    def _load_time_value_source(data: dict, base: Path) -> TimeValueSourceConfig:
        """Reconstruct a TimeValueSourceConfig from its JSON dict representation.

        Args:
            data: The dict loaded from JSON (from to_dict()).
            base: Workspace directory used to resolve relative csv_path values.
        """
        return TimeValueSourceConfig(
            csv_path=_resolve(data["csv_path"], base),
            timestamp_column=data["timestamp_column"],
            value_column=data["value_column"],
            date_column=data.get("date_column"),
            start_time=ConfigService._parse_dt(data.get("start_time")),
            end_time=ConfigService._parse_dt(data.get("end_time")),
            no_header=bool(data.get("no_header", False)),
        )

    @staticmethod
    def _load_sensor_file(data: dict, base: Path) -> SensorFileConfig:
        """Reconstruct a SensorFileConfig (with its channels list) from JSON.

        Args:
            data: The dict loaded from JSON (from SensorFileConfig.to_dict()).
            base: Workspace directory for resolving relative csv_path.
        """
        # Reconstruct each SensorChannel from its own sub-dict.
        channels = [
            SensorChannel(
                source_column=channel["source_column"],
                display_name=channel["display_name"],
                units=channel.get("units", ""),
                use_header_name=channel.get("use_header_name", True),
            )
            for channel in data.get("channels", [])
        ]
        return SensorFileConfig(
            csv_path=_resolve(data["csv_path"], base),
            timestamp_column=data["timestamp_column"],
            date_column=data.get("date_column"),
            channels=channels,
            start_time=ConfigService._parse_dt(data.get("start_time")),
            end_time=ConfigService._parse_dt(data.get("end_time")),
            no_header=bool(data.get("no_header", False)),
        )

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        """Parse an ISO 8601 datetime string, returning None on failure or empty input.

        Used to reconstruct start_time and end_time fields from JSON without
        crashing on missing or malformed values in older workspace files.
        """
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _load_interval(data: dict) -> SelectedTimeRange:
        """Reconstruct a SelectedTimeRange from its JSON dict representation."""
        return SelectedTimeRange(
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
            source=data.get("source", "manual"),
            threshold_desc=data.get("threshold_desc", ""),
        )

    # ---------------------------------------------------------------------------
    # Output directory inspection
    # ---------------------------------------------------------------------------

    @staticmethod
    def _detect_applied_steps(output_dir: Path) -> list[str]:
        """Inspect the output directory and infer which pipeline steps have run.

        Rather than relying solely on the "applied_steps" key in the workspace
        JSON (which could be stale), this method checks the actual contents of
        the output directory at load time and returns a fresh list.

        Steps detected:
          - "extract_frames"       — at least one segment has a non-empty frames/ dir
          - "generate_sensor_rasters" — at least one segment has .tif files, or has
                                       a master.csv (which implies rasters were at
                                       least attempted)
          - "annotate_frames"      — at least one segment has frames_annotated/ with JPEGs

        Args:
            output_dir: The root output directory configured in the workspace.

        Returns:
            A sorted list of detected step name strings.
        """
        applied = set()

        if not output_dir.exists():
            return []

        # Walk immediate subdirectories that match the segment naming convention.
        for segment_dir in output_dir.iterdir():
            if not (segment_dir.is_dir() and segment_dir.name.startswith("segment_")):
                continue

            frames_dir    = segment_dir / "frames"
            sensors_dir   = segment_dir / "sensors"
            annotated_dir = segment_dir / "frames_annotated"
            interp_csv    = segment_dir / "interp.csv"

            # Frame extraction evidence: the frames/ directory has at least one JPEG.
            if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
                applied.add("extract_frames")

            # Raster evidence: .tif files in sensors/, or a master.csv (implying the
            # raster step ran and produced the CSV even if no .tif was written).
            if interp_csv.exists():
                applied.add("generate_sensor_rasters")
            if sensors_dir.exists() and any(sensors_dir.glob("*.tif")):
                applied.add("generate_sensor_rasters")

            # Annotation evidence: annotated JPEG frames exist.
            if annotated_dir.exists() and any(annotated_dir.glob("*.jpg")):
                applied.add("annotate_frames")

        return sorted(applied)
