# pipeline_service.py — Core processing pipeline for the EPR Imaging application
#
# Responsibilities:
#   1. Accept a fully configured PipelineConfig object and orchestrate every
#      processing step end-to-end.
#   2. Support three distinct frame-extraction modes:
#       a. Sensor-only  — no images; write a dense interp.csv for later threshold use.
#       b. Threshold    — read a pre-built sensor-only interp.csv, apply user-defined
#                         threshold filters, split into contiguous groups, and extract
#                         frames only for windows that pass all thresholds.
#       c. Normal       — extract frames at a fixed rate (or dynamically spaced by
#                         GPS distance), merge nav + sensor data, and write all outputs.
#   3. Build interp.csv — one row per extracted frame; columns include unix_time,
#      ISO timestamp, lat, lon, alt, and all sensor channels (sensor values are
#      obtained by linearly interpolating each sensor timeseries at each frame time).
#   4. Produce optional derived outputs: sensor GeoTIFFs, annotated frames,
#      CLAHE-enhanced frames, and a WebODM-compatible geo.txt file.
#
# Threading notes:
#   PipelineService.run() is designed to be called from a background thread.
#   All progress/status updates are delivered via callbacks stored in
#   PipelineConfig rather than emitting Qt signals directly, so the service
#   layer stays decoupled from the GUI framework.

from __future__ import annotations

# ---------------------------------------------------------------------------
# Standard-library imports
# ---------------------------------------------------------------------------
import calendar    # calendar.timegm() converts a naive (UTC) timetuple to a Unix timestamp
                   # without applying any local-timezone offset — critical for correctness
from dataclasses import dataclass, field  # dataclass decorator; field() for mutable defaults
from datetime import datetime              # Used for wall-clock arithmetic and UTC conversions
from pathlib import Path                   # Cross-platform path handling throughout
from typing import Callable               # Used in the LogFn type alias
import logging                             # Module-level logger; mirrors progress to the Python log
import math                                # math.floor() for computing integer sample counts

# ---------------------------------------------------------------------------
# Third-party imports
# ---------------------------------------------------------------------------
import cv2          # OpenCV — open video files, seek to frames, read and write JPEG images
import numpy as np  # Vectorised array operations for grid generation and NaN-fill logic
import pandas as pd # DataFrame used as the primary in-memory table for interp.csv construction
import rasterio     # Write georeferenced GeoTIFF files for each sensor raster
from rasterio.transform import from_bounds  # Compute affine transform from bounding box + pixel size
from scipy.interpolate import griddata      # Scattered-point spatial interpolation for sensor rasters

# ---------------------------------------------------------------------------
# Internal imports
# ---------------------------------------------------------------------------
from dynamicsampling import create_dynamic_sample_schedule  # Distance-adaptive frame schedule
from models import AnnotationConfig, NavigationConfig, SegmentRecord, SelectedTimeRange, SensorFileConfig, ThresholdConfig, VideoRecord
from sensor_service import SensorService   # Timeseries loading and linear interpolation helpers
from video_service import VideoService     # Directory scan + filename-based start-time parsing

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

# LogFn is a callable that accepts a single string message and returns nothing.
# It is used in PipelineService.__init__() to inject a logging sink without
# creating a hard dependency on the Qt GUI logging widget.
LogFn = Callable[[str], None]


# ===========================================================================
# PipelineConfig — the single configuration object passed to PipelineService.run()
# ===========================================================================

@dataclass
class PipelineConfig:
    """All parameters required to run the extraction pipeline for one session.

    This dataclass acts as an explicit, inspectable API boundary between the
    MainWindow (which reads the form fields and constructs this object) and
    PipelineService (which consumes it).  Keeping everything here avoids
    threading a large number of keyword arguments through every helper method.

    Fields are grouped below by logical concern:
      — I/O paths
      — Pre-loaded data (videos, intervals)
      — Navigation & sensor sources
      — Threshold filters
      — Extraction settings
      — Optional output steps
      — CLAHE parameters
      — UI callbacks
    """

    # -----------------------------------------------------------------------
    # I/O paths
    # -----------------------------------------------------------------------

    # Root folder that contains the raw video files on disk.  Scanned by
    # VideoService.scan_directory() if `videos` is empty.
    video_directory: Path

    # Job output directory: <workspace_dir>/job_<job_id:03d>/.  Created by
    # MainWindow before calling run(); segment subdirectories go inside it.
    output_directory: Path

    # strptime-style format string the user typed into the "Filename format"
    # field (e.g. "%Y_%m_%dT%H_%M_%S").  Passed directly to VideoService so
    # it can try the user's format before falling back to COMMON_PATTERNS.
    # An empty string means "use auto-detection only".
    video_filename_time_format: str

    # Serial number of the job being run.  Stored in SegmentRecord so the
    # history can group segments by job.
    job_id: int = 0

    # -----------------------------------------------------------------------
    # Pre-loaded data
    # -----------------------------------------------------------------------

    # List of VideoRecord objects produced by a prior VideoService scan.
    # When non-empty, the pipeline uses this list directly and skips the
    # directory scan.  MainWindow populates this from its own cached scan so
    # the user doesn't have to wait for another full directory walk.
    videos: list[VideoRecord] = field(default_factory=list)

    # Time windows selected by the user on the Pipeline tab.  Each window
    # becomes one output segment directory.  When empty, the pipeline defaults
    # to the bounding box of all videos (one single interval covering
    # everything).
    selected_intervals: list[SelectedTimeRange] = field(default_factory=list)

    # -----------------------------------------------------------------------
    # Navigation & sensor sources
    # -----------------------------------------------------------------------

    # Holds three TimeValueSourceConfig objects pointing at the CSV columns
    # for latitude, longitude, and optionally altitude.  None if the user
    # hasn't configured navigation.
    navigation_file: NavigationConfig | None = None

    # List of additional sensor CSV files (e.g. temperature, salinity).
    # Each entry is a SensorFileConfig that names the timestamp column and
    # the data channels to extract.
    sensor_files: list[SensorFileConfig] = field(default_factory=list)

    # Optional dedicated depth source.  When set, the depth channel is loaded
    # from this file rather than from sensor_files; it is appended to the
    # sensor_frames list and treated identically to other sensor channels.
    depth_source: SensorFileConfig | None = None

    # Optional dedicated speed source, same mechanics as depth_source.
    speed_source: SensorFileConfig | None = None

    # -----------------------------------------------------------------------
    # Threshold filters
    # -----------------------------------------------------------------------
    # Threshold filters are only applied in the "threshold-based sampling"
    # path, which requires a pre-built sensor-only interp.csv.  Each filter
    # is expressed as a scalar limit; None means "no filter for this channel".

    # Maximum altitude (metres) — rows above this are excluded.
    altitude_threshold: float | None = None

    # Minimum depth (metres) — rows shallower than this are excluded.
    depth_threshold: float | None = None

    # Minimum speed (m/s or knots, user's units) — rows slower than this are excluded.
    speed_threshold: float | None = None

    # Per-sensor-channel thresholds: column_name → (min_value, max_value).
    # Either bound can be None (meaning no bound on that side).
    # Example: {"Temperature": (2.0, 30.0)} keeps only rows where 2 ≤ temp ≤ 30.
    sensor_thresholds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)

    # Minimum number of consecutive threshold-passing rows required for a
    # contiguous group to be extracted as a distinct range.  Prevents very
    # short gaps from producing near-empty output segments.
    min_segment_frames: int = 1

    # -----------------------------------------------------------------------
    # Extraction settings
    # -----------------------------------------------------------------------

    # Number of frames extracted per second of video when using fixed-rate
    # sampling.  1.0 = one frame per second; 0.5 = one frame every two seconds.
    frame_rate: float = 1.0

    # Controls whether frames are extracted by fixed clock rate or by GPS
    # distance.  "fixed" uses frame_rate; "dynamic" calls
    # create_dynamic_sample_schedule() to place frames at equal ground-track
    # intervals (target_spacing_m), with a minimum-frequency floor.
    sampling_mode: str = "fixed"  # "fixed" or "dynamic"

    # Target ground spacing between frames in dynamic mode (metres).
    dynamic_target_spacing_m: float = 2.0

    # Minimum frame frequency in dynamic mode (Hz).  Prevents the schedule
    # from becoming completely silent on long straight sections where the GPS
    # distance criterion would space frames many seconds apart.
    dynamic_min_frequency_hz: float = 0.1

    # When False, the pipeline skips all image extraction and only produces
    # a interp.csv (sensor-only mode).  Useful for building a dense sensor
    # record first, then applying thresholds, before committing to extracting
    # potentially thousands of frames.
    sample_images: bool = True

    # Whether to generate GeoTIFF rasters for each sensor channel.
    # Only applies when sample_images is True and frames have been extracted.
    generate_sensor_rasters: bool = True

    # Whether to burn nav + sensor overlays onto each extracted frame.
    annotate_frames: bool = False

    # List of processing step names that are enabled for this run.
    # Possible values: "extract_frames", "generate_sensor_rasters",
    #   "annotate_frames", "apply_clahe", "generate_geo_txt",
    #   "update_master", "interpolate_only", "build_full_interp".
    # The pipeline checks membership in this list to gate each step, which
    # allows re-running only specific steps on existing output directories
    # without re-extracting frames.
    selected_steps: list[str] = field(default_factory=lambda: ["extract_frames", "generate_sensor_rasters", "annotate_frames"])

    # Sample rate (Hz) used when building the full-dataset interp_full.csv.
    # Only used when "build_full_interp" is in selected_steps.
    full_interp_sample_hz: float = 1.0

    # Workspace directory path — interp_full.csv is written here so it lives
    # alongside workspace.json rather than inside a per-job subfolder.
    workspace_directory: str = ""

    # -----------------------------------------------------------------------
    # CLAHE parameters
    # -----------------------------------------------------------------------

    # Contrast Limited Adaptive Histogram Equalization clip limit.
    # Higher values allow stronger contrast enhancement but risk noise
    # amplification.  OpenCV default is 40.0; 2.0 is a conservative value
    # for underwater imagery.
    clahe_clip_limit: float = 2.0

    # Tile grid size for CLAHE (applied as tileGridSize × tileGridSize).
    # Smaller tiles produce more localised contrast enhancement; larger
    # tiles behave more like global histogram equalisation.
    clahe_tile_grid_size: int = 8

    # -----------------------------------------------------------------------
    # UI callbacks
    # -----------------------------------------------------------------------
    # All callbacks are optional (None means "no-op").  They are called from
    # the background thread that runs PipelineService.run(), so Qt connections
    # must route them through a signal/slot mechanism or use
    # QMetaObject.invokeMethod() to stay thread-safe.  The MainWindow wires
    # these up to its Qt signals before calling run().

    # Called with an integer 0–100 to update the top-level progress bar.
    progress_callback: callable | None = None

    # Called with a string to update the main status label.
    status_callback: callable | None = None

    # Called with a string to append a line to the log text view.
    log_callback: callable | None = None

    # Called with an integer 0–100 to update the per-video sub-progress bar.
    subprogress_callback: callable | None = None

    # Called with a string to update the sub-status label (e.g. current filename).
    substatus_callback: callable | None = None

    # Called with a SegmentRecord when each segment completes (or fails).
    # MainWindow connects this to append the record to segment_history and
    # auto-save the workspace.
    segment_completed_callback: callable | None = None

    # Controls which fields are burned onto annotated frames, where the text
    # block is placed, and how it looks.  None falls back to legacy defaults.
    annotation_config: AnnotationConfig | None = None

    # JPEG quality used for all cv2.imwrite() calls.
    # "Original" copies frames without re-encoding (saves at 95 JPEG quality).
    # Other options compress to the specified quality level.
    frame_quality: str = "Original"


# ===========================================================================
# PipelineService — stateless orchestrator that drives all pipeline stages
# ===========================================================================

class PipelineService:
    """Stateless processing service that drives every pipeline stage.

    The service itself holds almost no state — only a logger and an optional
    log function injected at construction time.  All per-run state travels
    in the PipelineConfig argument.  This design makes PipelineService safe
    to reuse across multiple run() calls from the same GUI session.

    Typical call sequence (from a background QThread):
        service = PipelineService(log_fn=self.log_signal.emit)
        produced = service.run(config)
        # produced is a list of Path objects for the directories that were written
    """

    def __init__(self, log_fn: LogFn | None = None):
        """Initialise with an optional external log sink.

        Args:
            log_fn: Called with a string message whenever log() is invoked
                    directly on the service.  Falls back to a no-op lambda so
                    callers never need to guard against None.
        """
        # Wrap None into a no-op so all log() calls inside this class are
        # unconditional — no if-guards needed at every call site.
        self.log_fn = log_fn or (lambda message: None)

        # Python standard logger for module-level output.  Mirrors _emit_status
        # messages to the configured logging handlers (e.g. file logs, console).
        self.logger = logging.getLogger("PipelineService")

    def log(self, message: str) -> None:
        """Forward a message through the injected log sink.

        Used by methods that need to log a detail that doesn't warrant
        a full status-bar update.
        """
        self.log_fn(message)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def run(self, config: PipelineConfig) -> list[Path]:
        """Execute the complete pipeline for one session.

        This is the single public entry point.  It orchestrates every step
        end-to-end:

          1. Validate and set up output directories.
          2. Scan for videos (or use pre-loaded list).
          3. Derive the interval list (or default to the full video span).
          4. Determine which steps are enabled (extract, rasters, annotate…).
          5. Load all navigation and sensor CSV data.
          6. For each selected time interval, choose one of three paths:
               a. Sensor-only  — build a dense interp.csv without images.
               b. Threshold    — read the sensor-only CSV, apply filters, extract
                                 frames only for passing windows.
               c. Normal       — extract frames (fixed or dynamic rate), build
                                 interp.csv, and run any enabled downstream steps.

        Args:
            config: Fully populated PipelineConfig describing what to do.

        Returns:
            A list of Path objects, one per output directory that was written
            (one directory per interval, or per threshold range).

        Raises:
            ValueError:   If no videos are found or no steps are enabled.
            RuntimeError: If a later step is requested but its prerequisite
                          output (e.g. frames_dir) doesn't exist.
        """
        # Announce the start and reset all progress bars to 0.
        self._emit_progress(config, 0)
        self._emit_status(config, "Starting pipeline...")
        self._emit_subprogress(config, 0)
        self._emit_substatus(config, "Idle.")

        # Create the root output directory now so any error in the early setup
        # doesn't leave the GUI waiting for a directory that never appears.
        config.output_directory.mkdir(parents=True, exist_ok=True)

        # -----------------------------------------------------------------
        # Step 1: Resolve the video list
        # If MainWindow pre-loaded VideoRecords, use those directly.  If not
        # (e.g. running from a saved workspace), scan the directory now.
        # -----------------------------------------------------------------
        # build_full_interp operates entirely on nav/sensor data — no videos needed.
        if config.selected_steps == ["build_full_interp"]:
            videos = []
        else:
            videos = config.videos or VideoService(config.video_filename_time_format).scan_directory(config.video_directory)[0]
            if not videos:
                raise ValueError("No videos available to process.")

        # -----------------------------------------------------------------
        # Step 2: Resolve the interval list
        # If the user drew intervals on the timeline, use those.  If not,
        # synthesise a single interval spanning all videos so the pipeline
        # processes everything.
        # -----------------------------------------------------------------
        if config.selected_intervals:
            intervals = config.selected_intervals
        elif videos:
            intervals = [
                SelectedTimeRange(
                    start_time=min(video.start_time for video in videos),
                    end_time=max(video.end_time for video in videos),
                )
            ]
        else:
            intervals = []

        # -----------------------------------------------------------------
        # Step 3: Determine which pipeline steps are enabled
        # The selected_steps list is the single source of truth.  Each flag
        # below is computed once so per-interval logic doesn't re-evaluate
        # the list membership on every iteration.
        # -----------------------------------------------------------------
        selected_steps = config.selected_steps or ["extract_frames", "generate_sensor_rasters", "annotate_frames"]

        # sensor_only: build a dense interp.csv at sample_rate with no image
        # extraction or image post-processing.  Triggered explicitly via the
        # "interpolate_only" step (set by _build_pipeline_config when
        # sample_images is unchecked on the main Execute path).
        # Post-processing runs (CLAHE, annotate, etc.) must NOT set sensor_only
        # even when sample_images is False — those steps operate on existing frames.
        sensor_only = "interpolate_only" in selected_steps

        # Individual step flags; sensor_only overrides image-related steps.
        run_extract       = "extract_frames"          in selected_steps and not sensor_only
        run_update_master = "update_master"           in selected_steps or sensor_only
        run_rasters       = "generate_sensor_rasters" in selected_steps and not sensor_only
        run_annotate      = "annotate_frames"         in selected_steps and not sensor_only
        run_clahe         = "apply_clahe"             in selected_steps and not sensor_only
        run_geo_txt       = "generate_geo_txt"        in selected_steps and not sensor_only
        run_full_interp   = "build_full_interp"       in selected_steps

        # Guard: at least one step must be active, otherwise there's nothing to do.
        if not any((run_extract, run_update_master, run_rasters, run_annotate, run_clahe, run_geo_txt, sensor_only, run_full_interp)):
            raise ValueError("At least one processing step must be selected.")

        # -----------------------------------------------------------------
        # Step 4: Detect whether any threshold filters are configured
        # has_thresholds enables the threshold-based extraction path when a
        # sensor-only interp.csv already exists.
        # -----------------------------------------------------------------
        has_thresholds = (
            config.altitude_threshold is not None or
            config.depth_threshold    is not None or
            config.speed_threshold    is not None or
            bool(config.sensor_thresholds)
        )

        # -----------------------------------------------------------------
        # Step 5: Load navigation sources
        # Each coordinate (lat, lon, alt) is loaded into a separate DataFrame
        # with columns ["unix_time", "value"].  They live in a dict so the
        # rest of the pipeline can look up each coordinate by name without
        # needing to know which file or column it came from.
        # -----------------------------------------------------------------
        self._emit_progress(config, 5)
        self._emit_status(config, "Preparing inputs...")
        nav_sources: dict[str, pd.DataFrame] = {}
        if config.navigation_file:
            self._emit_log(config, f"Loading latitude source: {config.navigation_file.latitude_source.csv_path}")
            nav_sources["lat"] = SensorService.load_time_value_dataframe(config.navigation_file.latitude_source)

            self._emit_log(config, f"Loading longitude source: {config.navigation_file.longitude_source.csv_path}")
            nav_sources["lon"] = SensorService.load_time_value_dataframe(config.navigation_file.longitude_source)

            if config.navigation_file.altitude_source:
                self._emit_log(config, f"Loading altitude source: {config.navigation_file.altitude_source.csv_path}")
                nav_sources["alt"] = SensorService.load_time_value_dataframe(config.navigation_file.altitude_source)

            if config.navigation_file.depth_source:
                self._emit_log(config, f"Loading depth source: {config.navigation_file.depth_source.csv_path}")
                nav_sources["water_depth"] = SensorService.load_time_value_dataframe(
                    config.navigation_file.depth_source,
                    negate=getattr(config.navigation_file, "negate_depth", False),
                )

            if config.navigation_file.heading_source:
                self._emit_log(config, f"Loading heading source: {config.navigation_file.heading_source.csv_path}")
                nav_sources["heading"] = SensorService.load_time_value_dataframe(config.navigation_file.heading_source)

            if config.navigation_file.pitch_source:
                self._emit_log(config, f"Loading pitch source: {config.navigation_file.pitch_source.csv_path}")
                nav_sources["pitch"] = SensorService.load_time_value_dataframe(config.navigation_file.pitch_source)

            if config.navigation_file.roll_source:
                self._emit_log(config, f"Loading roll source: {config.navigation_file.roll_source.csv_path}")
                nav_sources["roll"] = SensorService.load_time_value_dataframe(config.navigation_file.roll_source)

        self._emit_progress(config, 15)
        self._emit_status(config, "Navigation sources loaded.")

        # -----------------------------------------------------------------
        # Step 6: Load sensor data files
        # sensor_frames is a list of (config, dataframe) pairs.  Each entry
        # gives us a timeseries we can interpolate at any unix timestamp.
        # depth_source and speed_source are appended here so they go through
        # the same interpolation path as regular sensor channels.
        # -----------------------------------------------------------------
        sensor_frames: list[tuple[SensorFileConfig, pd.DataFrame]] = []
        for sensor_cfg in config.sensor_files:
            self._emit_log(config, f"Loading sensor file: {sensor_cfg.csv_path}")
            sensor_frames.append((sensor_cfg, SensorService.load_sensor_dataframe(sensor_cfg)))

        if config.depth_source:
            self._emit_log(config, f"Loading depth source: {config.depth_source.csv_path}")
            sensor_frames.append((config.depth_source, SensorService.load_sensor_dataframe(config.depth_source)))

        if config.speed_source:
            self._emit_log(config, f"Loading speed source: {config.speed_source.csv_path}")
            sensor_frames.append((config.speed_source, SensorService.load_sensor_dataframe(config.speed_source)))

        self._emit_progress(config, 25)
        self._emit_status(config, "Sensor sources loaded.")

        # -----------------------------------------------------------------
        # Step 7: Main loop — process each selected time interval
        # produced_dirs accumulates the output paths that were written so
        # the caller can open them or report them to the user.
        # -----------------------------------------------------------------
        produced_dirs: list[Path] = []
        total_intervals = len(intervals)

        for idx, interval in enumerate(intervals):
            self._emit_status(config, f"Processing interval {idx + 1} of {total_intervals}...")

            # Build a human-readable directory name that encodes the interval
            # boundaries so output folders sort chronologically and are
            # unambiguous even after re-runs with different settings.
            segment_name = (
                f"segment_{idx + 1:03d}_"
                f"{interval.start_time.strftime('%Y%m%dT%H%M%S')}_"
                f"{interval.end_time.strftime('%Y%m%dT%H%M%S')}"
            )

            # Pre-build all expected sub-directory paths so each branch of
            # the conditional below can reference them without recomputing.
            segment_dir   = config.output_directory / segment_name
            frames_dir    = segment_dir / "frames"
            sensors_dir   = segment_dir / "sensors"
            annotated_dir = segment_dir / "frames_annotated"
            clahe_dir     = segment_dir / "frames_clahe"
            interp_csv    = segment_dir / "interp.csv"

            # ==============================================================
            # PATH A: Sensor-only mode
            # Build a interp.csv populated at sample_rate with sensor values
            # interpolated at each sample time.  No images are extracted.
            # This is the prerequisite for threshold-based sampling on a
            # subsequent run.
            # ==============================================================
            if sensor_only:
                segment_dir.mkdir(parents=True, exist_ok=True)
                sensors_dir.mkdir(parents=True, exist_ok=True)

                # If real frames already exist in this segment, re-interpolate
                # on the existing interp.csv rows so frame_filename is preserved.
                # Only build from synthetic timestamps when no frames exist yet.
                frames_exist = frames_dir.exists() and any(frames_dir.glob("*.jpg"))
                if frames_exist and interp_csv.exists():
                    master_df = pd.read_csv(interp_csv)
                    master_df = self._update_master_dataframe(master_df, nav_sources, sensor_frames)
                    master_df.to_csv(interp_csv, index=False)
                    self._emit_log(config, f"  interp.csv → {interp_csv} (updated, frame_filename preserved)")
                    self._emit_status(config, f"interp.csv updated ({len(master_df)} rows).")
                else:
                    # Build a synthetic frame_df — a table of unix_times at which
                    # to evaluate all sensor timeseries.  No actual frames are read.
                    if config.sampling_mode == "dynamic" and nav_sources:
                        self._emit_substatus(config, "Computing dynamic sample schedule...")
                        dynamic_times = self._get_dynamic_sample_times(nav_sources, interval, config)
                        self._emit_log(config, f"Dynamic sampling: {len(dynamic_times)} target frames for interval {idx + 1}")
                        frame_df = self._create_dynamic_sample_df(videos, dynamic_times)
                    else:
                        frame_df = self._create_sampled_frame_df(videos, interval, config.frame_rate)

                    if frame_df.empty:
                        self._emit_log(config, f"No sample times for interval {idx + 1}; skipping.")
                        continue

                    master_df = self._build_master_dataframe(frame_df, nav_sources, sensor_frames)
                    master_df.to_csv(interp_csv, index=False)
                    self._emit_log(config, f"  interp.csv → {interp_csv}")
                    self._emit_status(config, f"Sensor-only interp.csv written ({len(master_df)} rows, no images).")

                self._emit_progress(config, 60)
                produced_dirs.append(segment_dir)
                self._emit_segment_completed(config, interval, segment_dir, "completed")
                continue  # Jump to the next interval; no more steps in this path.

            # ==============================================================
            # PATH B: Threshold-based sampling
            # Triggered when:
            #   — extract_frames is requested
            #   — at least one threshold filter is active
            #   — a sensor-only interp.csv already exists
            #   — no frames have been extracted yet (frames_dir is empty)
            # Reads the dense interp.csv, applies threshold masks, splits the
            # passing rows into contiguous groups, and extracts one set of
            # frames per group.
            # ==============================================================
            frames_exist  = frames_dir.exists() and any(frames_dir.glob("*.jpg"))
            master_exists = interp_csv.exists()

            if run_extract and has_thresholds and master_exists and not frames_exist:
                self._emit_progress(config, 35)
                self._emit_status(config, "Applying thresholds to sensor-only data, splitting into ranges...")

                # Load the dense interp.csv produced by the sensor-only run.
                dense_df = pd.read_csv(interp_csv)

                # Build a boolean mask: True for rows that pass all thresholds.
                mask = self._apply_thresholds(dense_df, config)

                # Trim to passing rows, then group contiguous windows.  A gap
                # wider than 2.5× the sample interval is treated as a break.
                valid_df = dense_df[mask].reset_index(drop=True)
                groups   = self._split_contiguous_groups(valid_df, config.frame_rate, config.min_segment_frames)
                self._emit_log(
                    config,
                    f"Interval {idx + 1}: {mask.sum()} of {len(dense_df)} rows pass thresholds "
                    f"→ {len(groups)} range(s)."
                )

                # Process each contiguous group as an independent range output.
                for grp_idx, grp_df in enumerate(groups):
                    range_name       = f"{segment_name}_range_{grp_idx + 1:03d}"
                    range_dir        = config.output_directory / range_name
                    range_frames_dir = range_dir / "frames"
                    range_interp_csv = range_dir / "interp.csv"
                    range_dir.mkdir(parents=True, exist_ok=True)
                    range_frames_dir.mkdir(parents=True, exist_ok=True)
                    self._emit_substatus(config, f"Extracting range {grp_idx + 1}/{len(groups)}: {len(grp_df)} frames...")

                    # Collect the target unix timestamps from the dense CSV
                    # then distribute them to whichever video file covers each.
                    target_times = grp_df["unix_time"].tolist()
                    frame_parts  = []
                    for video in videos:
                        vstart = self._file_dt_to_unix(video.start_time)
                        vend   = self._file_dt_to_unix(video.end_time)
                        vtimes = [t for t in target_times if vstart <= t <= vend]
                        if vtimes:
                            part = self._extract_frames_for_timestamps(video, vtimes, range_frames_dir, config)
                            if not part.empty:
                                frame_parts.append(part)

                    if not frame_parts:
                        self._emit_log(config, f"No frames extracted for range {grp_idx + 1}; skipping.")
                        continue

                    # Merge frame parts, re-sort by time, and build a range-level interp.csv.
                    frame_df = (
                        pd.concat(frame_parts, ignore_index=True)
                        .sort_values("unix_time")
                        .reset_index(drop=True)
                    )
                    range_master_df = self._build_master_dataframe(frame_df, nav_sources, sensor_frames)
                    range_master_df.to_csv(range_interp_csv, index=False)
                    self._emit_log(config, f"  frames      → {range_frames_dir} ({len(frame_df)} frames)")
                    self._emit_log(config, f"  interp.csv  → {range_interp_csv}")

                    # Optionally write a WebODM geo.txt for this range.
                    if run_geo_txt and range_frames_dir.exists() and any(range_frames_dir.glob("*.jpg")):
                        geo_txt_path = range_dir / "geo.txt"
                        n = self._write_geo_txt(range_dir, range_master_df)
                        if n:
                            self._emit_log(config, f"  geo.txt     → {geo_txt_path} ({n} frames)")
                        else:
                            self._emit_log(config, f"  geo.txt skipped for range {grp_idx + 1}: no lat/lon data in interp.csv")

                    produced_dirs.append(range_dir)
                    self._emit_segment_completed(config, interval, range_dir, "completed")
                    self._emit_log(config, f"Range {grp_idx + 1}: {len(frame_df)} frames → {range_name}")

                self._emit_progress(config, 90)
                self._emit_status(config, f"Threshold sampling complete: {len(groups)} range(s) produced.")
                continue  # Jump to the next interval.

            # ==============================================================
            # PATH C: Normal extraction
            # Frame extraction at fixed or dynamic rate, followed by any
            # enabled downstream steps.
            # ==============================================================

            # Create output directories only if the relevant steps will run.
            if run_extract:
                segment_dir.mkdir(parents=True, exist_ok=True)
                frames_dir.mkdir(parents=True, exist_ok=True)
            if run_rasters or run_update_master:
                sensors_dir.mkdir(parents=True, exist_ok=True)

            # Cache existence checks so we can decide whether to skip steps.
            rasters_exist     = sensors_dir.exists()    and any(sensors_dir.glob("*.tif"))
            annotations_exist = annotated_dir.exists()  and any(annotated_dir.glob("*.jpg"))

            # Declare as local variables; assigned by whichever branch below executes.
            frame_df:  pd.DataFrame
            master_df: pd.DataFrame

            # ------------------------------------------------------------------
            # Sub-step: Frame extraction (skip if frames already exist)
            # ------------------------------------------------------------------
            if run_extract and not frames_exist:
                self._emit_progress(config, 35)
                self._emit_status(config, "Extracting frames...")
                self._emit_subprogress(config, 0)
                self._emit_substatus(config, "Waiting to start frame extraction...")
                frame_parts = []

                if config.sampling_mode == "dynamic" and nav_sources:
                    # --- Dynamic sampling: GPS-distance-adaptive schedule ---
                    self._emit_substatus(config, "Computing dynamic sample schedule...")
                    dynamic_times = self._get_dynamic_sample_times(nav_sources, interval, config)
                    self._emit_log(config, f"Dynamic sampling: {len(dynamic_times)} target frames for interval {idx + 1}")

                    # Distribute target times to the video that covers each timestamp.
                    for video in videos:
                        vstart = self._file_dt_to_unix(video.start_time)
                        vend   = self._file_dt_to_unix(video.end_time)
                        vtimes = [t for t in dynamic_times if vstart <= t <= vend]
                        if vtimes:
                            part = self._extract_frames_for_timestamps(video, vtimes, frames_dir, config)
                            if not part.empty:
                                frame_parts.append(part)
                else:
                    # --- Fixed-rate sampling: one frame every 1/frame_rate seconds ---
                    for video in videos:
                        # Only process videos whose time range overlaps the interval.
                        if self._intervals_overlap(video.start_time, video.end_time, interval.start_time, interval.end_time):
                            part = self._extract_frames_for_interval(
                                video,
                                interval.start_time,
                                interval.end_time,
                                frames_dir,
                                config.frame_rate,
                                config,
                            )
                            if not part.empty:
                                frame_parts.append(part)

                if not frame_parts:
                    self._emit_log(config, f"No frames extracted for interval {idx + 1}; skipping interval.")
                    continue

                # Merge results from all video files for this interval and sort
                # chronologically so interp.csv rows are in time order.
                frame_df = (
                    pd.concat(frame_parts, ignore_index=True)
                    .sort_values("unix_time")
                    .reset_index(drop=True)
                )
                self._emit_subprogress(config, 0)
                self._emit_substatus(config, "Frame extraction complete.")

                # Build and write interp.csv immediately after extraction so
                # that a crash during a later step doesn't lose the frame list.
                master_df = self._build_master_dataframe(frame_df, nav_sources, sensor_frames)
                master_df.to_csv(interp_csv, index=False)
                self._emit_log(config, f"  frames      → {frames_dir} ({len(frame_df)} frames)")
                self._emit_log(config, f"  interp.csv  → {interp_csv}")
                self._emit_progress(config, 60)
                self._emit_status(config, "interp.csv created.")

            elif run_extract and frames_exist:
                # Frames exist from a previous run; load the existing interp.csv.
                self._emit_log(config, f"Frames already exist for interval {idx + 1}; skipping extraction.")
                self._emit_log(config, f"  frames      → {frames_dir}")
                if not master_exists:
                    self._emit_log(config, f"interp.csv missing for existing frames in interval {idx + 1}; cannot proceed.")
                    continue
                master_df = pd.read_csv(interp_csv)
                self._emit_log(config, f"  interp.csv  → {interp_csv} (loaded existing)")
                self._emit_progress(config, 60)
                self._emit_status(config, "Loaded existing interp.csv.")

            else:
                # extract_frames step is not selected; load an existing interp.csv
                # that was produced by a prior run.  This branch handles the case
                # where the user re-runs only downstream steps (rasters, annotate…).
                if not segment_dir.exists() or not interp_csv.exists():
                    raise RuntimeError("Frame extraction output must already exist before running later steps.")
                master_df = pd.read_csv(interp_csv)
                self._emit_log(config, f"  interp.csv  → {interp_csv}")
                self._emit_progress(config, 60)

                if run_update_master:
                    # Update the interp.csv by re-interpolating nav + sensor columns.
                    # Useful when new sensor files have been added since the initial run.
                    self._emit_status(config, "Updating existing interp.csv with new sources...")
                    master_df = self._update_master_dataframe(master_df, nav_sources, sensor_frames)
                    master_df.to_csv(interp_csv, index=False)
                    self._emit_log(config, f"  interp.csv  → {interp_csv} (updated)")
                    self._emit_status(config, "interp.csv updated.")
                else:
                    self._emit_status(config, "Loaded existing interp.csv.")

            # ------------------------------------------------------------------
            # Sub-step: Sensor raster generation (GeoTIFFs)
            # ------------------------------------------------------------------
            if run_rasters and not rasters_exist:
                # Generate one GeoTIFF for each non-navigation sensor column.
                for column in self._sensor_value_columns(master_df):
                    safe_name  = self._safe_name(column)
                    output_tif = sensors_dir / f"{safe_name}.tif"
                    self._create_sensor_raster(master_df, column, output_tif)
                    self._emit_log(config, f"  raster      → {output_tif}")
                self._emit_progress(config, 80)
                self._emit_status(config, "Sensor TIFF generation complete.")
            elif run_rasters and rasters_exist:
                self._emit_log(config, f"Raster TIFFs already exist for interval {idx + 1}; skipping generation.")
                self._emit_log(config, f"  sensors dir → {sensors_dir}")
                self._emit_progress(config, 80)
                self._emit_status(config, "Sensor TIFFs already exist.")

            # ------------------------------------------------------------------
            # Sub-step: Frame annotation (burn text overlays onto frames)
            # ------------------------------------------------------------------
            if run_annotate and not annotations_exist:
                self._annotate_frames(frames_dir, annotated_dir, master_df, config.annotation_config, config.frame_quality)
                self._emit_log(config, f"  annotated   → {annotated_dir}")
                self._emit_progress(config, 93)
                self._emit_status(config, "Annotation complete.")
            elif run_annotate and annotations_exist:
                self._emit_log(config, f"Annotated frames already exist for interval {idx + 1}; skipping annotation.")
                self._emit_log(config, f"  annotated   → {annotated_dir}")
                self._emit_progress(config, 93)
                self._emit_status(config, "Annotated frames already exist.")

            # ------------------------------------------------------------------
            # Sub-step: CLAHE contrast enhancement
            # ------------------------------------------------------------------
            clahe_exist = clahe_dir.exists() and any(clahe_dir.glob("*.jpg"))
            if run_clahe and not clahe_exist:
                self._apply_clahe_to_frames(frames_dir, clahe_dir, config)
                self._emit_log(config, f"  clahe       → {clahe_dir}")
                self._emit_progress(config, 97)
                self._emit_status(config, "CLAHE complete.")
            elif run_clahe and clahe_exist:
                self._emit_log(config, f"CLAHE frames already exist for interval {idx + 1}; skipping.")
                self._emit_log(config, f"  clahe       → {clahe_dir}")
                self._emit_progress(config, 97)
                self._emit_status(config, "CLAHE frames already exist.")

            # ------------------------------------------------------------------
            # Sub-step: Write WebODM geo.txt
            # The geo.txt format lists each image filename and its GPS coords so
            # WebODM can georeference the photogrammetry reconstruction.
            # ------------------------------------------------------------------
            if run_geo_txt:
                if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
                    geo_txt_path = segment_dir / "geo.txt"
                    n_written    = self._write_geo_txt(segment_dir, master_df)
                    self._emit_progress(config, 98)
                    if n_written:
                        self._emit_log(config, f"  geo.txt     → {geo_txt_path} ({n_written} frames)")
                        self._emit_status(config, f"WebODM geo.txt written ({n_written} frames).")
                    else:
                        self._emit_status(config, "WebODM geo.txt skipped — no lat/lon data in interp.csv.")
                        self._emit_log(
                            config,
                            f"geo.txt not written for interval {idx + 1}: interp.csv has no valid lat/lon rows. "
                            "Ensure navigation (lat/lon) is configured and re-run with 'Update interp.csv' checked."
                        )
                else:
                    self._emit_log(config, f"No frames found for interval {idx + 1}; skipping geo.txt.")

            produced_dirs.append(segment_dir)
            self._emit_segment_completed(config, interval, segment_dir, "completed")

        # Build the full-dataset interp_full.csv spanning the entire nav range.
        if run_full_interp:
            ws_dir = Path(config.workspace_directory) if config.workspace_directory else config.output_directory
            self._build_full_interp_csv(config, nav_sources, sensor_frames, ws_dir)

        # Pipeline complete — reset progress indicators.
        self._emit_progress(config, 100)
        self._emit_status(config, "Pipeline complete.")
        self._emit_subprogress(config, 0)
        self._emit_substatus(config, "Idle.")
        return produced_dirs

    # -----------------------------------------------------------------------
    # Threshold interval analysis (no pipeline run required)
    # -----------------------------------------------------------------------

    @staticmethod
    def find_threshold_intervals(
        navigation_file: "NavigationConfig | None",
        sensor_files: "list[SensorFileConfig]",
        constraints: list[dict],
        min_duration_s: float = 2.0,
    ) -> list[SelectedTimeRange]:
        """Scan raw sensor/nav data and return intervals where ALL constraints are met.

        Each constraint is a dict with keys:
            channel   — display_name of a sensor channel OR a nav key
                        ("alt", "water_depth", "heading", "pitch", "roll")
            min_val   — float or None (no lower bound)
            max_val   — float or None (no upper bound)

        Constraints are combined with logical AND: every constraint must be
        satisfied simultaneously for a data point to be included.  Contiguous
        runs of passing points that span ≥ min_duration_s become one interval.

        Args:
            navigation_file: Configured nav sources (may be None).
            sensor_files:    List of sensor CSV configs.
            constraints:     List of {channel, min_val, max_val} dicts.
            min_duration_s:  Minimum interval length in seconds.

        Returns:
            List of SelectedTimeRange objects with source="threshold".
        """
        if not constraints:
            return []

        # ------------------------------------------------------------------
        # Step 1: load all relevant timeseries
        # ------------------------------------------------------------------
        series: dict[str, pd.DataFrame] = {}   # channel_name → (unix_time, value) df

        # Navigation channels
        _NAV_KEYS = {"alt", "water_depth", "heading", "pitch", "roll"}
        if navigation_file is not None:
            _nav_srcs = {
                "alt":         navigation_file.altitude_source,
                "water_depth": navigation_file.depth_source,
                "heading":     navigation_file.heading_source,
                "pitch":       navigation_file.pitch_source,
                "roll":        navigation_file.roll_source,
            }
            for key, src in _nav_srcs.items():
                if src is not None:
                    try:
                        df = SensorService.load_time_value_dataframe(
                            src,
                            negate=(key == "water_depth" and getattr(navigation_file, "negate_depth", False)),
                        )
                        series[key] = df
                    except Exception:
                        pass

        # Sensor channels
        for sf in sensor_files:
            try:
                raw = SensorService.load_sensor_dataframe(sf)
                for ch in sf.channels:
                    name = ch.display_name or ch.source_column
                    if ch.source_column in raw.columns:
                        ch_df = raw[["unix_time", ch.source_column]].copy()
                        ch_df = ch_df.rename(columns={ch.source_column: "value"})
                        series[name] = ch_df
            except Exception:
                pass

        if not series:
            return []

        # ------------------------------------------------------------------
        # Step 2: build a common time axis via union of all timestamps
        # ------------------------------------------------------------------
        relevant_channels = {c["channel"] for c in constraints}
        missing = relevant_channels - set(series.keys())
        if missing:
            return []   # required channel not available

        # Merge on unix_time using outer join, interpolate linearly
        merged: pd.DataFrame | None = None
        for ch in relevant_channels:
            df = series[ch].rename(columns={"value": ch})
            merged = df if merged is None else pd.merge_asof(
                merged.sort_values("unix_time"),
                df.sort_values("unix_time"),
                on="unix_time",
                direction="nearest",
                tolerance=5.0,   # 5-second tolerance for matching timestamps
            )
        if merged is None or merged.empty:
            return []

        merged = merged.sort_values("unix_time").reset_index(drop=True)

        # ------------------------------------------------------------------
        # Step 3: build a boolean mask — True where ALL constraints pass
        # ------------------------------------------------------------------
        mask = pd.Series(True, index=merged.index)
        for c in constraints:
            col = c["channel"]
            if col not in merged.columns:
                return []
            vals = pd.to_numeric(merged[col], errors="coerce")
            if c.get("min_val") is not None:
                mask &= vals >= float(c["min_val"])
            if c.get("max_val") is not None:
                mask &= vals <= float(c["max_val"])

        # ------------------------------------------------------------------
        # Step 4: find contiguous True runs of sufficient duration
        # ------------------------------------------------------------------
        times = merged["unix_time"].to_numpy(dtype=float)
        m     = mask.to_numpy(dtype=bool)

        intervals: list[SelectedTimeRange] = []
        i = 0
        while i < len(m):
            if m[i]:
                # Find end of this True run
                j = i
                while j < len(m) and m[j]:
                    j += 1
                duration = times[j - 1] - times[i] if j > i else 0.0
                if duration >= min_duration_s:
                    start_dt = datetime.utcfromtimestamp(float(times[i]))
                    end_dt   = datetime.utcfromtimestamp(float(times[j - 1]))
                    desc_parts = []
                    for c in constraints:
                        lo = f"≥{c['min_val']:.3g}" if c.get("min_val") is not None else ""
                        hi = f"≤{c['max_val']:.3g}" if c.get("max_val") is not None else ""
                        desc_parts.append(f"{c['channel']} {lo}{hi}".strip())
                    intervals.append(SelectedTimeRange(
                        start_time=start_dt,
                        end_time=end_dt,
                        source="threshold",
                        threshold_desc="  &  ".join(desc_parts),
                    ))
                i = j
            else:
                i += 1

        return intervals

    # -----------------------------------------------------------------------
    # Static utilities
    # -----------------------------------------------------------------------

    @staticmethod
    def _intervals_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
        """Return True if two half-open time intervals [a_start, a_end) and [b_start, b_end) overlap.

        Uses the standard two-interval overlap test: they overlap if and only if
        neither one ends before the other begins.  Touching at a single point
        (a_end == b_start) is treated as non-overlapping (strict less-than).

        Args:
            a_start: Start of the first interval (inclusive).
            a_end:   End of the first interval (exclusive).
            b_start: Start of the second interval (inclusive).
            b_end:   End of the second interval (exclusive).

        Returns:
            True if the intervals share at least one instant of time.
        """
        return a_start < b_end and b_start < a_end

    @staticmethod
    def _file_dt_to_unix(dt: datetime) -> float:
        """Convert a naive UTC datetime to a Unix timestamp (seconds since epoch).

        Uses calendar.timegm() rather than datetime.timestamp() because
        datetime.timestamp() interprets naive datetimes as local time, which
        would introduce a timezone offset on machines not set to UTC.
        calendar.timegm() always treats the input as UTC, matching the
        application-wide convention that all naive datetimes are UTC.

        Args:
            dt: A naive datetime (no tzinfo) representing a UTC wall-clock time.

        Returns:
            Float seconds since the Unix epoch (1970-01-01T00:00:00 UTC).
        """
        return float(calendar.timegm(dt.timetuple()))

    # -----------------------------------------------------------------------
    # Frame extraction — fixed-rate
    # -----------------------------------------------------------------------

    def _extract_frames_for_interval(
        self,
        video: VideoRecord,
        interval_start: datetime,
        interval_end: datetime,
        output_dir: Path,
        sample_hz: float,
        config: PipelineConfig,
    ) -> pd.DataFrame:
        """Extract frames from one video file at a fixed sampling rate.

        Computes the intersection of [interval_start, interval_end] and
        [video.start_time, video.end_time], then evenly spaces target frame
        indices across that window at sample_hz frames per second.

        Strategy:
          1. Clip the interval to the video's actual time coverage.
          2. Convert the clipped window to offsets in seconds from video start.
          3. Generate evenly-spaced offset values at sample_hz.
          4. Convert each offset to a frame index: idx = round(offset * fps).
          5. Seek the video to each index, read the frame, and write a JPEG.
          6. Build a DataFrame row for each successfully saved frame.

        Args:
            video:          VideoRecord describing the video file.
            interval_start: Wall-clock start of the desired extraction window.
            interval_end:   Wall-clock end of the desired extraction window.
            output_dir:     Directory to write extracted JPEGs into.
            sample_hz:      Frames to extract per second of video time.
            config:         PipelineConfig for callback access.

        Returns:
            A DataFrame with columns [frame_filename, unix_time,
            video_filename, frame_index], one row per extracted frame.
            Returns an empty DataFrame if the video doesn't overlap the
            interval or if no frames could be read.

        Raises:
            ValueError:   If sample_hz ≤ 0.
            RuntimeError: If the video file cannot be opened, FPS is invalid,
                          or a frame write fails.
        """
        if sample_hz <= 0:
            raise ValueError("Frame rate must be > 0")

        # Clip the desired extraction window to the actual video coverage.
        # This handles videos that start before or end after the interval.
        effective_start = max(video.start_time, interval_start)
        effective_end   = min(video.end_time,   interval_end)
        if effective_start >= effective_end:
            return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video.path}")

        try:
            # Read FPS and frame count from the container header.
            video_fps = video.fps or cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or video_fps <= 0:
                raise RuntimeError(f"Could not determine FPS for {video.path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                raise RuntimeError(f"Could not determine frame count for {video.path}")

            # Convert the clipped window endpoints to offsets from the video's
            # own start time.  max(0, ...) guards against tiny floating-point
            # negatives when effective_start equals video.start_time.
            start_offset_s = max(0.0, (effective_start - video.start_time).total_seconds())
            end_offset_s   = max(0.0, (effective_end   - video.start_time).total_seconds())
            if end_offset_s < start_offset_s:
                return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

            # Generate evenly-spaced sample offsets.  +1 in sample_count so
            # that the last sample falls at or very near end_offset_s rather
            # than stopping one step short.  The 1e-9 tolerance guards against
            # floating-point rounding that would include a spurious extra sample.
            duration_s   = end_offset_s - start_offset_s
            sample_count = max(1, int(math.floor(duration_s * sample_hz)) + 1)
            target_times = [start_offset_s + (i / sample_hz) for i in range(sample_count)]
            target_times = [t for t in target_times if t <= end_offset_s + 1e-9]

            if not target_times:
                return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

            # Convert offsets to integer frame indices (zero-based).
            raw_target_frames = [int(round(t * video_fps)) for t in target_times]

            # Deduplicate frame indices to avoid writing the same frame twice
            # when sample_hz is high relative to the video's actual FPS, and
            # clamp all indices to [0, frame_count - 1].
            target_frames: list[int] = []
            seen: set[int] = set()
            for idx in raw_target_frames:
                idx = max(0, min(idx, frame_count - 1))
                if idx not in seen:
                    seen.add(idx)
                    target_frames.append(idx)

            if not target_frames:
                return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

            self._emit_substatus(config, f"Extracting frames from {video.filename}")
            self._emit_subprogress(config, 0)

            rows: list[dict] = []
            total_targets    = len(target_frames)

            # Pre-compute the video start as a Unix float so the per-frame
            # unix_time can be computed with a simple addition.
            video_start_raw = self._file_dt_to_unix(video.start_time)

            for saved_idx, frame_idx in enumerate(target_frames):
                # Seek to the target frame.  Some containers don't support
                # exact seeking; the frame read will be from the nearest keyframe.
                ok_seek = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                if not ok_seek:
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                # Compute the wall-clock time of this frame and verify it
                # falls within the requested interval (safeguard against
                # seeks landing on a slightly different frame due to keyframes).
                frame_offset_s = frame_idx / video_fps
                frame_dt       = datetime.utcfromtimestamp(video_start_raw + frame_offset_s)
                if not (interval_start <= frame_dt < interval_end):
                    continue
                unix_time = video_start_raw + frame_offset_s

                # Build the output filename: [video_stem]_[YYYYMMDDTHHMMSS_mmm].jpg
                ts_dt = datetime.utcfromtimestamp(unix_time)
                ts_str = ts_dt.strftime("%Y%m%dT%H%M%S") + f"_{ts_dt.microsecond // 1000:03d}"
                fname = f"{video.path.stem}_{ts_str}.jpg"
                fpath = output_dir / fname

                ok_write = cv2.imwrite(
                    str(fpath),
                    frame,
                    self._jpeg_params(config.frame_quality),
                )
                if not ok_write:
                    raise RuntimeError(f"Failed to write extracted frame to {fpath}")

                rows.append({
                    "frame_filename": fname,
                    "unix_time":      unix_time,
                    "video_filename": video.filename,
                    "frame_index":    frame_idx,
                })

                # Update the sub-progress bar and label every frame so the
                # user can see the extraction is progressing.
                progress = int(round(((saved_idx + 1) / total_targets) * 100))
                self._emit_subprogress(config, progress)
                detail = (
                    f"Extracting frames from {video.filename} | "
                    f"saved {saved_idx + 1}/{total_targets}: {fname}"
                )
                self._emit_substatus(config, detail)

            self._emit_subprogress(config, 100)
            self._emit_substatus(config, f"Finished extracting frames from {video.filename} ({len(rows)} frames saved)")
            return pd.DataFrame(rows)

        finally:
            # Always release the OpenCV capture handle, even on exception,
            # to avoid leaking file descriptors.
            cap.release()

    # -----------------------------------------------------------------------
    # Sampling helpers
    # -----------------------------------------------------------------------

    def _create_sampled_frame_df(
        self, videos: list[VideoRecord], interval: SelectedTimeRange, frame_rate: float
    ) -> pd.DataFrame:
        """Build a synthetic frame DataFrame for sensor-only (no-image) mode.

        Rather than reading actual video frames, this creates a table of
        evenly-spaced unix timestamps at frame_rate Hz across the interval.
        The sensor interpolation logic in _build_master_dataframe() doesn't
        care whether the timestamps came from real frames or a synthetic grid,
        so the same path works for both cases.

        Args:
            videos:     List of VideoRecord objects (used only to populate the
                        video_filename column with a plausible value).
            interval:   The time window to sample.
            frame_rate: Number of sample points per second.

        Returns:
            DataFrame with columns [unix_time, frame_filename, video_filename,
            frame_index].  frame_filename is an empty string because no image
            is written to disk.
        """
        start_ts   = self._file_dt_to_unix(interval.start_time)
        end_ts     = self._file_dt_to_unix(interval.end_time)
        duration   = end_ts - start_ts
        num_samples = max(1, int(math.floor(duration * frame_rate)) + 1)

        # np.linspace produces exactly num_samples evenly-spaced values from
        # start_ts to end_ts (inclusive on both ends).
        unix_times = np.linspace(start_ts, end_ts, num_samples)
        rows = []
        for i, unix_time in enumerate(unix_times):
            rows.append({
                "unix_time":      float(unix_time),
                "frame_filename": "",  # No image written in sensor-only mode
                "video_filename": videos[0].filename if videos else "",
                "frame_index":    i,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _build_nav_df(nav_sources: dict[str, pd.DataFrame], t_start: float, t_end: float) -> pd.DataFrame:
        """Extract and join nav data within a time window (with padding) for dynamic sampling.

        Merges lat and lon onto a shared time grid derived from the lat timeseries,
        and optionally adds altitude.  Used by _get_dynamic_sample_times() to
        provide the GPS track that create_dynamic_sample_schedule() needs.

        A 60-second pad on each side prevents the edge effects of clamped
        interpolation from distorting the GPS track near interval boundaries.

        Args:
            nav_sources: Dict with keys "lat", "lon", optionally "alt"; each value
                         is a DataFrame with ["unix_time", "value"] columns.
            t_start:     Unix timestamp for the start of the interval.
            t_end:       Unix timestamp for the end of the interval.

        Returns:
            A DataFrame with columns [unix_time, lat, lon] and optionally [alt].
            Returns an empty DataFrame if lat or lon sources are missing or
            if no nav data overlaps the padded window.
        """
        if "lat" not in nav_sources or "lon" not in nav_sources:
            return pd.DataFrame()

        # 60-second padding ensures the dynamic schedule has good context at
        # the edges of the interval rather than running off the end of the data.
        pad    = 60.0
        lat_df = nav_sources["lat"]
        lat_df = lat_df[
            (lat_df["unix_time"] >= t_start - pad) &
            (lat_df["unix_time"] <= t_end   + pad)
        ].copy()

        if lat_df.empty:
            return pd.DataFrame()

        lon_df = nav_sources["lon"]

        # Use the lat timeseries as the master time axis and interpolate lon
        # onto it to avoid dealing with different GPS log rates for each channel.
        result = pd.DataFrame({
            "unix_time": lat_df["unix_time"].values,
            "lat":       lat_df["value"].values,
        })
        result["lon"] = SensorService.interpolate_series(
            result["unix_time"], lon_df["unix_time"], lon_df["value"]
        )

        if "alt" in nav_sources:
            alt_df = nav_sources["alt"]
            result["alt"] = SensorService.interpolate_series(
                result["unix_time"], alt_df["unix_time"], alt_df["value"]
            )
        else:
            # Default to sea level (0.0) when no altitude source is configured.
            result["alt"] = 0.0

        # Drop rows where lat or lon couldn't be interpolated (i.e. the lat
        # track has timestamps outside the lon coverage window).
        return result.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    def _get_dynamic_sample_times(
        self,
        nav_sources: dict[str, pd.DataFrame],
        interval: SelectedTimeRange,
        config: PipelineConfig,
    ) -> list[float]:
        """Compute GPS-distance-adaptive frame sample times for one interval.

        Delegates to create_dynamic_sample_schedule() (dynamicsampling.py),
        which walks the GPS track and fires a sample whenever the cumulative
        ground distance reaches target_spacing_m, or when the minimum
        frequency floor requires one.

        Args:
            nav_sources: Dict of nav DataFrames (lat, lon, optionally alt).
            interval:    The time window to compute samples for.
            config:      PipelineConfig for target spacing and min-frequency parameters.

        Returns:
            Sorted list of Unix timestamps at which to extract frames.
            Returns [] if nav data is unavailable or the interval is too short.
        """
        t_start = self._file_dt_to_unix(interval.start_time)
        t_end   = self._file_dt_to_unix(interval.end_time)

        # Build the nav track for this interval (with padding for context).
        nav_df = self._build_nav_df(nav_sources, t_start, t_end)
        if nav_df.empty:
            return []

        # Clip the nav track to the exact interval so the schedule doesn't
        # extend beyond the user's selection.
        nav_df = nav_df[
            (nav_df["unix_time"] >= t_start) &
            (nav_df["unix_time"] <= t_end)
        ].reset_index(drop=True)

        # Need at least two nav points to compute any distance.
        if len(nav_df) < 2:
            return []

        alt_col = "alt" if "alt" in nav_sources else None

        try:
            schedule = create_dynamic_sample_schedule(
                nav_df,
                time_col="unix_time",
                lat_col="lat",
                lon_col="lon",
                alt_col=alt_col,
                target_spacing_m=config.dynamic_target_spacing_m,
                min_frequency_hz=config.dynamic_min_frequency_hz,
            )
        except ValueError as exc:
            # Non-monotone or duplicate timestamps in the nav CSV raise ValueError.
            # Log and fall back to an empty schedule for this interval rather than
            # crashing the entire pipeline run.
            self._emit_log(config,
                f"  WARNING: Dynamic sampling skipped for this interval: {exc}. "
                "Check that navigation timestamps are strictly increasing.")
            return []

        if schedule.empty:
            return []

        # Return only the scheduled timestamps; the caller maps each to a video.
        return schedule["sample_time_s"].tolist()

    @staticmethod
    def _create_dynamic_sample_df(videos: list[VideoRecord], sample_times: list[float]) -> pd.DataFrame:
        """Wrap a list of dynamic sample times in a DataFrame for sensor-only use.

        Analogous to _create_sampled_frame_df() but fed from the dynamic
        schedule rather than from a regular grid.  Used when sensor_only is
        True and sampling_mode is "dynamic".

        Args:
            videos:       Video list — used only for the video_filename column.
            sample_times: Unix timestamps produced by _get_dynamic_sample_times().

        Returns:
            DataFrame with columns [unix_time, frame_filename, video_filename,
            frame_index].  frame_filename is empty (no image written).
        """
        if not sample_times:
            return pd.DataFrame(columns=["unix_time", "frame_filename", "video_filename", "frame_index"])

        rows = [
            {
                "unix_time":      float(t),
                "frame_filename": "",
                "video_filename": videos[0].filename if videos else "",
                "frame_index":    i,
            }
            for i, t in enumerate(sorted(sample_times))
        ]
        return pd.DataFrame(rows)

    # -----------------------------------------------------------------------
    # Threshold logic
    # -----------------------------------------------------------------------

    def _apply_thresholds(self, df: pd.DataFrame, config: PipelineConfig) -> "pd.Series[bool]":
        """Build a boolean mask that is True for rows passing all configured thresholds.

        Each threshold is applied as an independent condition; all conditions
        are AND-ed together.  Columns that don't exist in df are silently
        skipped so that a config with thresholds for a channel not present
        in this particular file doesn't crash.

        Args:
            df:     DataFrame to filter, typically a sensor-only interp.csv.
            config: PipelineConfig holding the threshold values.

        Returns:
            A boolean pd.Series aligned to df.index; True = row passes all filters.
        """
        # Start with all rows passing; AND in each active threshold below.
        mask = pd.Series(True, index=df.index)

        # Altitude: rows must be at or below the altitude ceiling.
        if config.altitude_threshold is not None and "alt" in df.columns:
            mask &= df["alt"] <= config.altitude_threshold

        # Depth: rows must be at or deeper than the threshold.
        # The column may be named "depth" (lower) depending on how navigation was imported.
        # When negate_depth is True, stored values are negative; the comparison must flip
        # so that "deeper than X metres" means a more-negative value in the column.
        depth_col = next((c for c in ("water_depth", "depth", "Depth") if c in df.columns), None)
        if config.depth_threshold is not None and depth_col:
            negate = getattr(config.navigation_file, "negate_depth", False) if config.navigation_file else False
            if negate:
                # e.g. threshold=-2000 → keep rows where depth ≤ -2000 (deeper)
                mask &= df[depth_col] <= config.depth_threshold
            else:
                # positive depth convention → keep rows where depth ≥ threshold
                mask &= df[depth_col] >= config.depth_threshold

        # Speed: rows must meet or exceed the minimum speed.
        if config.speed_threshold is not None and "Speed" in df.columns:
            mask &= df["Speed"] >= config.speed_threshold

        # Per-channel range filters: apply min and max independently.
        for col_name, (min_val, max_val) in (config.sensor_thresholds or {}).items():
            if col_name in df.columns:
                if min_val is not None:
                    mask &= df[col_name] >= min_val
                if max_val is not None:
                    mask &= df[col_name] <= max_val

        return mask

    def _split_contiguous_groups(
        self, df: pd.DataFrame, frame_rate: float, min_frames: int
    ) -> list[pd.DataFrame]:
        """Split a filtered DataFrame into contiguous time groups.

        After applying thresholds, the remaining rows may not be consecutive
        in time — some rows in between were excluded.  This function detects
        those gaps and splits the DataFrame wherever the time gap between
        adjacent rows exceeds 2.5× the nominal sample interval.

        The factor 2.5 is a heuristic: it is small enough to catch genuine
        data gaps but large enough to tolerate minor timing jitter from the
        GPS logger or sensor clock.

        Args:
            df:         Threshold-filtered DataFrame with a "unix_time" column.
            frame_rate: The nominal sample rate in Hz (e.g. 1.0 fps).
            min_frames: Minimum number of rows a group must have to be kept.
                        Groups smaller than this threshold are discarded.

        Returns:
            List of sub-DataFrames, one per contiguous group.  Each has
            reset integer indices starting at 0.
        """
        if df.empty:
            return []

        # The expected time between consecutive samples; × 2.5 = gap threshold.
        gap_threshold = (1.0 / max(frame_rate, 1e-6)) * 2.5
        times         = df["unix_time"].to_numpy()
        groups: list[pd.DataFrame] = []
        start_i = 0

        # Walk the time array and emit a group whenever a gap is detected.
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > gap_threshold:
                grp = df.iloc[start_i:i]
                if len(grp) >= min_frames:
                    groups.append(grp.reset_index(drop=True))
                start_i = i

        # Flush the final group after the loop ends.
        grp = df.iloc[start_i:]
        if len(grp) >= min_frames:
            groups.append(grp.reset_index(drop=True))

        return groups

    # -----------------------------------------------------------------------
    # Frame extraction — timestamp-targeted
    # -----------------------------------------------------------------------

    def _extract_frames_for_timestamps(
        self,
        video: VideoRecord,
        target_unix_times: list[float],
        output_dir: Path,
        config: PipelineConfig,
    ) -> pd.DataFrame:
        """Extract frames from a video at a specific list of wall-clock timestamps.

        Used by both dynamic-sampling mode and threshold-based sampling.  Unlike
        _extract_frames_for_interval(), which generates a regular grid of times,
        this method accepts an arbitrary list of unix timestamps and seeks to
        the corresponding frame in the video for each one.

        Frame index calculation:
            frame_idx = round((unix_time - video_start_unix) * fps)

        This assumes the video's internal frame timestamps are monotonic and
        closely follow the clock time implied by the filename.

        Args:
            video:             VideoRecord for the video file.
            target_unix_times: List of Unix timestamps (seconds since epoch)
                               for which frames should be extracted.  Should
                               all fall within [video.start_time, video.end_time].
            output_dir:        Directory to write JPEGs into.
            config:            PipelineConfig for callback access.

        Returns:
            DataFrame with columns [frame_filename, unix_time, video_filename,
            frame_index].  Rows are in the order of sorted target_unix_times.

        Raises:
            RuntimeError: If the video cannot be opened or FPS is invalid.
        """
        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video.path}")

        try:
            video_fps = video.fps or cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or video_fps <= 0:
                raise RuntimeError(f"Could not determine FPS for {video.path}")

            frame_count      = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_start_ts   = self._file_dt_to_unix(video.start_time)

            # Sort so frame seeks are approximately monotonically increasing,
            # which tends to be faster for most container formats.
            sorted_times  = sorted(target_unix_times)
            total_targets = len(sorted_times)
            self._emit_substatus(config, f"Extracting frames from {video.filename}")

            rows: list[dict] = []
            for saved_idx, unix_time in enumerate(sorted_times):
                # Convert the absolute timestamp to a frame index.
                offset_s  = unix_time - video_start_ts
                frame_idx = int(round(offset_s * video_fps))
                frame_idx = max(0, min(frame_idx, frame_count - 1))

                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                ts_dt = datetime.utcfromtimestamp(unix_time)
                ts_str = ts_dt.strftime("%Y%m%dT%H%M%S") + f"_{ts_dt.microsecond // 1000:03d}"
                fname = f"{video.path.stem}_{ts_str}.jpg"
                fpath = output_dir / fname

                ok_write = cv2.imwrite(str(fpath), frame, self._jpeg_params(config.frame_quality))
                if not ok_write:
                    raise RuntimeError(f"Failed to write extracted frame to {fpath}")

                rows.append({
                    "frame_filename": fname,
                    "unix_time":      unix_time,
                    "video_filename": video.filename,
                    "frame_index":    frame_idx,
                })

                # Report per-frame progress to the sub-progress bar.
                self._emit_subprogress(config, int(round(((saved_idx + 1) / total_targets) * 100)))
                self._emit_substatus(
                    config,
                    f"Extracting frames from {video.filename} | {saved_idx + 1}/{total_targets}: {fname}"
                )

            self._emit_subprogress(config, 100)
            self._emit_substatus(config, f"Finished extracting frames from {video.filename} ({len(rows)} frames saved)")
            return pd.DataFrame(rows)

        finally:
            cap.release()

    # -----------------------------------------------------------------------
    # interp.csv construction
    # -----------------------------------------------------------------------

    def _build_full_interp_csv(
        self,
        config: "PipelineConfig",
        nav_sources: dict[str, pd.DataFrame],
        sensor_frames: list[tuple["SensorFileConfig", pd.DataFrame]],
        output_dir: Path,
    ) -> None:
        """Build interp_full.csv spanning the entire navigation time range.

        Unlike the per-segment interp.csv files, this output covers every
        timestamp in the navigation data at a uniform sample rate, regardless
        of job intervals.  It contains no frame-extraction columns (no
        frame_filename, video_filename, or frame_index).

        The file is written to output_dir/interp_full.csv, overwriting any
        existing file.

        Args:
            config:        Pipeline config (used for full_interp_sample_hz and logging).
            nav_sources:   Loaded nav timeseries (lat, lon, alt, water_depth, …).
            sensor_frames: Loaded sensor (config, dataframe) pairs.
            output_dir:    Job-level output directory.
        """
        # Collect the time extent of every nav and sensor source, then take the
        # union so interp_full.csv covers the full data coverage window without
        # being bounded by any single source or by video file timestamps.
        all_mins: list[float] = []
        all_maxs: list[float] = []
        for df in nav_sources.values():
            all_mins.append(float(df["unix_time"].min()))
            all_maxs.append(float(df["unix_time"].max()))
        for _, df in sensor_frames:
            if "unix_time" in df.columns:
                all_mins.append(float(df["unix_time"].min()))
                all_maxs.append(float(df["unix_time"].max()))

        if not all_mins:
            self._emit_log(config, "interp_full.csv skipped: no navigation or sensor sources configured.")
            return

        t_min = min(all_mins)
        t_max = max(all_maxs)
        hz      = max(config.full_interp_sample_hz, 0.001)
        step    = 1.0 / hz
        times   = np.arange(t_min, t_max + step * 0.5, step)

        frame_df = pd.DataFrame({"unix_time": times})
        full_df  = self._build_master_dataframe(frame_df, nav_sources, sensor_frames)

        # Drop frame-specific columns — they carry no meaning here.
        for col in ("frame_filename", "video_filename", "frame_index"):
            if col in full_df.columns:
                full_df = full_df.drop(columns=[col])

        output_dir.mkdir(parents=True, exist_ok=True)
        out_path = output_dir / "interp_full.csv"
        full_df.to_csv(out_path, index=False)
        self._emit_log(
            config,
            f"  interp_full.csv → {out_path}  ({len(full_df)} rows @ {hz:.3g} Hz)"
        )
        self._emit_status(config, f"interp_full.csv written ({len(full_df)} rows).")

    def _build_master_dataframe(
        self,
        frame_df: pd.DataFrame,
        nav_sources: dict[str, pd.DataFrame],
        sensor_frames: list[tuple[SensorFileConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Build the interp.csv DataFrame by merging frame times with all data sources.

        Takes the raw frame table (one row per frame, with unix_time) and
        enriches it with nav and sensor values interpolated at each frame's
        unix_time.  All interpolation is done with numpy.interp() (via
        SensorService.interpolate_series()), which performs linear interpolation
        and clamps extrapolated values to the boundary values.

        Column ordering in the result:
          frame_filename, timestamp_iso, unix_time, lat, lon, alt,
          video_filename, frame_index, <sensor columns...>

        A non-overlap warning is emitted when the frame timestamps fall
        entirely outside the nav data's time range, since all coordinates
        would then be extrapolated constants, which is almost certainly wrong.

        Args:
            frame_df:      DataFrame from _extract_frames_for_interval() or
                           _create_sampled_frame_df(), with a unix_time column.
            nav_sources:   Dict keyed by "lat", "lon", optionally "alt".
            sensor_frames: List of (SensorFileConfig, DataFrame) pairs loaded
                           from sensor CSV files.

        Returns:
            The enriched DataFrame ready to write to interp.csv.
        """
        master = frame_df.copy()
        master = master.sort_values("unix_time").reset_index(drop=True)

        # Add a human-readable ISO timestamp column alongside unix_time.
        # pd.to_datetime() treats the unit="s" as seconds since epoch (UTC)
        # without any timezone conversion, matching our naive-UTC convention.
        master["timestamp_iso"] = (
            pd.to_datetime(master["unix_time"], unit="s")
            .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
        )

        # Warn if the frame time range and nav time range don't overlap.
        # Extrapolated coordinates will be the boundary value rather than
        # genuinely useful positions, which would silently corrupt the output.
        if nav_sources and "lat" in nav_sources and not master.empty:
            nav_min   = nav_sources["lat"]["unix_time"].min()
            nav_max   = nav_sources["lat"]["unix_time"].max()
            frame_min = master["unix_time"].min()
            frame_max = master["unix_time"].max()
            if frame_max < nav_min or frame_min > nav_max:
                self.log(
                    f"  WARNING: Frame time range ({pd.to_datetime(frame_min, unit='s').isoformat()} – "
                    f"{pd.to_datetime(frame_max, unit='s').isoformat()}) does not overlap with navigation data "
                    f"({pd.to_datetime(nav_min, unit='s').isoformat()} – {pd.to_datetime(nav_max, unit='s').isoformat()}). "
                    f"Coordinates will be constant. Check that your video filenames and navigation data share the same time reference."
                )

        # Interpolate each nav channel at every frame timestamp.
        # If a source isn't configured, fill with NaN (or 0 for altitude) so
        # downstream code doesn't crash on missing columns.
        if "lat" in nav_sources:
            master["lat"] = SensorService.interpolate_series(
                master["unix_time"], nav_sources["lat"]["unix_time"], nav_sources["lat"]["value"]
            )
        else:
            master["lat"] = np.nan

        if "lon" in nav_sources:
            master["lon"] = SensorService.interpolate_series(
                master["unix_time"], nav_sources["lon"]["unix_time"], nav_sources["lon"]["value"]
            )
        else:
            master["lon"] = np.nan

        if "alt" in nav_sources:
            master["alt"] = SensorService.interpolate_series(
                master["unix_time"], nav_sources["alt"]["unix_time"], nav_sources["alt"]["value"]
            )
        else:
            # Default to 0.0 so alt is always a valid column even without data.
            master["alt"] = 0.0

        # Interpolate optional nav channels when configured.
        for key in ("water_depth", "heading", "pitch", "roll"):
            if key in nav_sources:
                master[key] = SensorService.interpolate_series(
                    master["unix_time"], nav_sources[key]["unix_time"], nav_sources[key]["value"]
                )

        # Interpolate each sensor channel at every frame timestamp.
        # The display_name from the channel config becomes the column header.
        for sensor_cfg, sensor_df in sensor_frames:
            for channel in sensor_cfg.channels:
                display_name = channel.display_name or channel.source_column
                master[display_name] = SensorService.interpolate_series(
                    master["unix_time"], sensor_df["unix_time"], sensor_df[channel.source_column]
                )

        master = self._add_utm_columns(master)

        # Reorder columns so the most important ones come first, making the
        # CSV easy to read and predictable for downstream tooling.
        preferred = ["frame_filename", "timestamp_iso", "unix_time", "lat", "lon", "alt",
                     "water_depth", "heading", "pitch", "roll",
                     "easting", "northing", "depth", "utm_zone",
                     "video_filename", "frame_index"]
        ordered   = [c for c in preferred if c in master.columns]
        remaining = [col for col in master.columns if col not in set(ordered)]
        return master[ordered + remaining]

    def _update_master_dataframe(
        self,
        master_df: pd.DataFrame,
        nav_sources: dict[str, pd.DataFrame],
        sensor_frames: list[tuple[SensorFileConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        """Re-interpolate nav and sensor columns on an existing master DataFrame.

        Called when the user selects "Update interp.csv" after adding new sensor
        files or changing navigation sources.  Unlike _build_master_dataframe(),
        this method preserves the existing frame_filename and frame_index values
        rather than generating them anew — only the interpolated columns are
        refreshed.

        Args:
            master_df:     The existing interp.csv loaded into a DataFrame.
            nav_sources:   New or updated nav source DataFrames.
            sensor_frames: New or updated sensor source pairs.

        Returns:
            The updated DataFrame with re-interpolated nav and sensor columns.
        """
        master = master_df.copy()

        # Ensure timestamp_iso exists (may be absent in very old workspace files).
        if "timestamp_iso" not in master.columns and "unix_time" in master.columns:
            master["timestamp_iso"] = (
                pd.to_datetime(master["unix_time"], unit="s")
                .dt.strftime("%Y-%m-%dT%H:%M:%S.%f")
            )

        # Re-interpolate whichever nav channels are now available.
        # Channels whose sources were not provided are left unchanged.
        if "lat" in nav_sources:
            master["lat"] = SensorService.interpolate_series(
                master["unix_time"], nav_sources["lat"]["unix_time"], nav_sources["lat"]["value"]
            )
        if "lon" in nav_sources:
            master["lon"] = SensorService.interpolate_series(
                master["unix_time"], nav_sources["lon"]["unix_time"], nav_sources["lon"]["value"]
            )
        if "alt" in nav_sources:
            master["alt"] = SensorService.interpolate_series(
                master["unix_time"], nav_sources["alt"]["unix_time"], nav_sources["alt"]["value"]
            )
        elif "alt" not in master.columns:
            master["alt"] = 0.0

        # Re-interpolate optional nav channels when sources are available.
        for key in ("water_depth", "heading", "pitch", "roll"):
            if key in nav_sources:
                master[key] = SensorService.interpolate_series(
                    master["unix_time"], nav_sources[key]["unix_time"], nav_sources[key]["value"]
                )

        # Re-interpolate all sensor channels.
        for sensor_cfg, sensor_df in sensor_frames:
            for channel in sensor_cfg.channels:
                display_name = channel.display_name or channel.source_column
                master[display_name] = SensorService.interpolate_series(
                    master["unix_time"], sensor_df["unix_time"], sensor_df[channel.source_column]
                )

        master = self._add_utm_columns(master)

        # Restore canonical column order.
        preferred = ["frame_filename", "timestamp_iso", "unix_time", "lat", "lon", "alt",
                     "water_depth", "heading", "pitch", "roll",
                     "easting", "northing", "depth", "utm_zone",
                     "video_filename", "frame_index"]
        ordered   = [c for c in preferred if c in master.columns]
        remaining = [col for col in master.columns if col not in set(ordered)]
        return master[ordered + remaining]

    # -----------------------------------------------------------------------
    # UTM coordinate computation
    # -----------------------------------------------------------------------

    @staticmethod
    def _add_utm_columns(master: pd.DataFrame) -> pd.DataFrame:
        """Append utm_x, utm_y, utm_z, and utm_zone columns to *master*.

        utm_x / utm_y are UTM easting / northing in metres (WGS84), projected
        from lat/lon using the `utm` library.  All rows are forced to the zone
        determined from the track centroid to prevent discontinuities at zone
        boundaries.

        utm_z = -depth (depth is positive-down from surface → negative Z).
        utm_zone is the full designation string, e.g. "18T" or "34S".

        All four columns are NaN / empty when required source columns are absent.
        """
        import utm as _utm

        if "lat" not in master.columns or "lon" not in master.columns:
            return master

        lat_arr = master["lat"].to_numpy(dtype=float)
        lon_arr = master["lon"].to_numpy(dtype=float)

        valid = ~(np.isnan(lat_arr) | np.isnan(lon_arr))
        if not valid.any():
            master = master.copy()
            master["easting"]  = np.nan
            master["northing"] = np.nan
            master["depth"]    = np.nan
            master["utm_zone"] = ""
            return master

        # Pin zone to centroid so the whole track shares one coordinate space.
        median_lat = float(np.median(lat_arr[valid]))
        median_lon = float(np.median(lon_arr[valid]))
        _, _, zone_number, zone_letter = _utm.from_latlon(median_lat, median_lon)

        easting  = np.full(len(lat_arr), np.nan)
        northing = np.full(len(lat_arr), np.nan)
        e, n, _, _ = _utm.from_latlon(
            lat_arr[valid], lon_arr[valid],
            force_zone_number=zone_number,
            force_zone_letter=zone_letter,
        )
        easting[valid]  = e
        northing[valid] = n

        master = master.copy()
        master["easting"]  = easting
        master["northing"] = northing
        master["depth"]    = -master["water_depth"] if "water_depth" in master.columns else np.nan
        master["utm_zone"] = f"{zone_number}{zone_letter}"
        return master

    # -----------------------------------------------------------------------
    # Sensor raster generation
    # -----------------------------------------------------------------------

    @staticmethod
    def _sensor_value_columns(master_df: pd.DataFrame) -> list[str]:
        """Return the list of sensor data column names in a master DataFrame.

        Filters out the standard infrastructure columns so that downstream
        raster and annotation code only iterates over actual measurement data
        (temperature, salinity, turbidity, etc.).

        Args:
            master_df: The interp.csv DataFrame with any number of sensor columns.

        Returns:
            List of column name strings that are not in the fixed exclusion set.
        """
        excluded = {"frame_filename", "timestamp_iso", "unix_time",
                    "lat", "lon", "alt", "water_depth", "heading", "pitch", "roll",
                    "easting", "northing", "depth", "utm_zone",
                    "video_filename", "frame_index"}
        return [col for col in master_df.columns if col not in excluded]

    def _create_sensor_raster(self, df: pd.DataFrame, value_column: str, output_path: Path) -> None:
        """Interpolate scattered GPS+sensor data onto a regular grid and write a GeoTIFF.

        The survey track is typically a curved line through geographic space.
        To produce a raster image suitable for GIS use, this method spatially
        interpolates the sensor values from those scattered track points onto a
        regular longitude/latitude grid using scipy.griddata().

        Interpolation strategy:
          1. Try "linear" (Delaunay triangulation) first — best quality.
          2. If that produces a fully-NaN grid (e.g. data is nearly collinear),
             fall back to "nearest" (Voronoi cells) which always succeeds.
          3. If "linear" partially succeeds (NaN holes outside the convex hull),
             fill the holes with "nearest" values.

        Grid size is computed dynamically based on the number of unique
        coordinate values, clamped to [200, 1200] pixels per axis.

        Args:
            df:           interp.csv DataFrame; must contain "lat", "lon", and
                          the named value column.
            value_column: Name of the sensor column to rasterise.
            output_path:  Where to write the GeoTIFF file.

        Raises:
            ValueError: If fewer than two valid data points exist for this column.
        """
        # Drop rows where any of the three required values is NaN.
        valid = df[["lon", "lat", value_column]].dropna()
        if len(valid) < 2:
            raise ValueError(f"Not enough valid points to create raster for {value_column}")

        lons   = valid["lon"].to_numpy(dtype=float)
        lats   = valid["lat"].to_numpy(dtype=float)
        values = valid[value_column].to_numpy(dtype=float)

        # Compute grid dimensions proportional to the data density, then clamp
        # to avoid degenerate (1×1) or enormous (>1200px) rasters.
        x_unique = np.unique(lons)
        y_unique = np.unique(lats)
        width    = int(np.clip(max(200, len(x_unique) * 4), 200, 1200))
        height   = int(np.clip(max(200, len(y_unique) * 4), 200, 1200))

        # Build the regular output grid from bounding-box extents.
        grid_x, grid_y = np.meshgrid(
            np.linspace(lons.min(), lons.max(), width),
            np.linspace(lats.min(), lats.max(), height),
        )

        # First attempt: linear interpolation (Delaunay triangulation).
        try:
            grid_z = griddata((lons, lats), values, (grid_x, grid_y), method="linear")
        except Exception:
            grid_z = None

        if grid_z is None or np.isnan(grid_z).all():
            # Linear interpolation produced no usable output; use nearest-neighbour.
            grid_z = griddata((lons, lats), values, (grid_x, grid_y), method="nearest")
        elif np.isnan(grid_z).any():
            # Linear interpolation has holes (outside the convex hull); fill with nearest.
            nearest = griddata((lons, lats), values, (grid_x, grid_y), method="nearest")
            grid_z  = np.where(np.isnan(grid_z), nearest, grid_z)

        # Compute the affine transform that maps pixel coordinates to geographic
        # coordinates (EPSG:4326) so GIS tools can correctly overlay the raster.
        transform = from_bounds(
            float(lons.min()), float(lats.min()),
            float(lons.max()), float(lats.max()),
            width, height,
        )

        # Write a single-band float32 GeoTIFF.
        with rasterio.open(
            output_path, "w",
            driver="GTiff",
            height=grid_z.shape[0],
            width=grid_z.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(grid_z.astype("float32"), 1)

    # -----------------------------------------------------------------------
    # Frame post-processing
    # -----------------------------------------------------------------------

    def _annotate_frames(
        self,
        frames_dir: Path,
        annotated_dir: Path,
        master_df: pd.DataFrame,
        ann: AnnotationConfig | None = None,
        frame_quality: str = "Original",
    ) -> None:
        """Burn configurable telemetry overlays onto each extracted frame.

        Rendering strategy:
          1. Build the list of text lines from ann.enabled_items.
          2. Measure the text block dimensions with cv2.getTextSize().
          3. Place the block in the chosen corner with a fixed padding margin.
          4. Optionally draw a semi-transparent dark rectangle behind the block
             (ann.use_background).  This is the most reliable visibility fix
             across all frame content types.
          5. Draw a thin dark outline around each character, then draw the
             user-specified fill colour on top.

        If ann is None, falls back to the legacy behaviour (white text,
        top-left, all nav + sensor channels).

        Args:
            frames_dir:    Directory containing the source JPEG frames.
            annotated_dir: Directory to write annotated copies into.
            master_df:     interp.csv DataFrame, one row per frame.
            ann:           AnnotationConfig controlling content and style.
        """
        annotated_dir.mkdir(parents=True, exist_ok=True)

        font      = cv2.FONT_HERSHEY_SIMPLEX
        pad       = 10   # px padding between block edge and frame edge / text

        if ann is None:
            ann = AnnotationConfig()

        scale     = float(ann.font_scale)
        r, g, b   = (int(c) for c in ann.color_rgb)
        fill_bgr  = (b, g, r)        # OpenCV uses BGR
        # Outline in the opposite luminance direction for contrast
        luma      = 0.299 * r + 0.587 * g + 0.114 * b
        out_bgr   = (0, 0, 0) if luma > 128 else (255, 255, 255)
        fill_th   = max(1, int(scale * 1.5))
        out_th    = fill_th + 2

        for _, row in master_df.iterrows():
            src = frames_dir / str(row["frame_filename"])
            if not src.exists():
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue

            h_img, w_img = img.shape[:2]

            # --- Build text lines ---
            lines: list[str] = []
            for ident in ann.enabled_items:
                if ident == "filename":
                    lines.append(str(row.get("frame_filename", "")))
                elif ident == "timestamp":
                    lines.append(str(row.get("timestamp_iso", "")))
                elif ident == "lat":
                    v = row.get("lat")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        lines.append(f"Lat:   {float(v):.6f}°")
                elif ident == "lon":
                    v = row.get("lon")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        lines.append(f"Lon:   {float(v):.6f}°")
                elif ident == "alt":
                    v = row.get("alt")
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        lines.append(f"Alt:   {float(v):.2f} m")
                elif ident in master_df.columns:
                    v = row.get(ident)
                    if v is not None:
                        try:
                            fv = float(v)
                            if not np.isnan(fv):
                                lines.append(f"{ident}: {fv:.3f}")
                        except (TypeError, ValueError):
                            lines.append(f"{ident}: {v}")

            if not lines:
                if not cv2.imwrite(str(annotated_dir / src.name), img, self._jpeg_params(frame_quality)):
                    self.log_fn(f"Warning: failed to write annotated frame {src.name}")
                continue

            # --- Measure text block ---
            line_sizes = [
                cv2.getTextSize(ln, font, scale, fill_th)[0] for ln in lines
            ]
            block_w   = max(sz[0] for sz in line_sizes) + pad * 2
            line_h    = max(sz[1] for sz in line_sizes) + int(scale * 6)
            block_h   = len(lines) * line_h + pad * 2

            # --- Compute corner origin ---
            pos = ann.position
            if pos == "top_left":
                bx, by = pad, pad
            elif pos == "top_right":
                bx, by = w_img - block_w - pad, pad
            elif pos == "bottom_left":
                bx, by = pad, h_img - block_h - pad
            else:  # bottom_right
                bx, by = w_img - block_w - pad, h_img - block_h - pad

            bx = max(0, min(bx, w_img - block_w))
            by = max(0, min(by, h_img - block_h))

            # --- Optional background rectangle ---
            if ann.use_background and ann.bg_opacity > 0.0:
                x1, y1 = bx, by
                x2, y2 = bx + block_w, by + block_h
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_img, x2), min(h_img, y2)
                overlay = img.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 0, 0), -1)
                cv2.addWeighted(overlay, ann.bg_opacity, img, 1.0 - ann.bg_opacity, 0, img)

            # --- Draw text lines ---
            for i, (line, (tw, th)) in enumerate(zip(lines, line_sizes)):
                tx = bx + pad
                ty = by + pad + (i + 1) * line_h
                # Outline pass
                cv2.putText(img, line, (tx, ty), font, scale, out_bgr, out_th, cv2.LINE_AA)
                # Fill pass
                cv2.putText(img, line, (tx, ty), font, scale, fill_bgr, fill_th, cv2.LINE_AA)

            if not cv2.imwrite(str(annotated_dir / src.name), img, self._jpeg_params(frame_quality)):
                self.log_fn(f"Warning: failed to write annotated frame {src.name}")

    def _apply_clahe_to_frames(self, frames_dir: Path, clahe_dir: Path, config: PipelineConfig) -> None:
        """Apply CLAHE contrast enhancement to all frames and write results.

        CLAHE (Contrast Limited Adaptive Histogram Equalization) is applied in
        the LAB colour space, modifying only the L (luminance) channel.  This
        preserves hue and saturation while improving local contrast, which is
        important for underwater or low-light imagery where the colour balance
        must not shift.

        Processing:
          1. Convert BGR → LAB.
          2. Apply CLAHE to the L channel only.
          3. Convert LAB → BGR.
          4. Write the result as JPEG at quality 90.

        Args:
            frames_dir: Directory containing the source JPEG frames.
            clahe_dir:  Directory to write CLAHE-enhanced copies into
                        (created if it doesn't exist).
            config:     PipelineConfig for clip_limit, tile_grid_size, and callbacks.
        """
        clahe_dir.mkdir(parents=True, exist_ok=True)

        # Create the CLAHE object once; reuse it for all frames in the directory.
        # clipLimit controls the contrast amplification ceiling per tile.
        # tileGridSize divides the image into non-overlapping tiles for local processing.
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )

        frame_paths = sorted(frames_dir.glob("*.jpg"))
        total       = len(frame_paths)
        if total == 0:
            self._emit_log(config, "No frames found for CLAHE processing.")
            return

        self._emit_subprogress(config, 0)
        for i, src in enumerate(frame_paths):
            img = cv2.imread(str(src))
            if img is None:
                continue

            # Work in LAB so CLAHE only affects luminance, not chrominance.
            lab           = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b       = cv2.split(lab)
            lab_result    = cv2.merge([clahe.apply(l), a, b])
            result        = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)

            if not cv2.imwrite(str(clahe_dir / src.name), result, self._jpeg_params(config.frame_quality)):
                self._emit_log(config, f"Warning: failed to write CLAHE frame {src.name}")

            self._emit_subprogress(config, int(round(((i + 1) / total) * 100)))
            self._emit_substatus(config, f"Applying CLAHE: {src.name} ({i + 1}/{total})")

        self._emit_subprogress(config, 100)
        self._emit_substatus(config, f"CLAHE complete ({total} frames written to {clahe_dir.name})")

    # -----------------------------------------------------------------------
    # WebODM geo.txt writer
    # -----------------------------------------------------------------------

    @staticmethod
    def _write_geo_txt(output_dir: Path, master_df: pd.DataFrame) -> int:
        """Write a WebODM-compatible geo.txt geolocation file.

        The geo.txt format is a plain-text table that WebODM reads to
        georeference images during the photogrammetry reconstruction.
        Format:
          Line 1: CRS identifier (EPSG:4326 — WGS 84 geographic coordinates)
          Subsequent lines: filename<TAB>longitude<TAB>latitude<TAB>altitude

        Filenames with spaces are percent-encoded ("%20") because WebODM's
        parser splits on whitespace.

        Only rows with non-empty frame_filename AND valid (non-NaN) lon and
        lat values are written; rows without GPS coverage are skipped.

        Args:
            output_dir: Directory to write geo.txt into.
            master_df:  The segment's interp.csv DataFrame.

        Returns:
            The number of image rows written.  0 means no geo.txt was
            created (no valid data).
        """
        # Guard: all three required columns must exist in the DataFrame.
        missing = [c for c in ("frame_filename", "lon", "lat") if c not in master_df.columns]
        if missing:
            return 0

        # Use the altitude column if it exists; otherwise default all rows to 0.
        alt_col = master_df["alt"] if "alt" in master_df.columns else pd.Series(0.0, index=master_df.index)

        rows = master_df[["frame_filename", "lon", "lat"]].copy()
        rows["alt"] = alt_col

        # Filter out rows with no filename and rows without valid coordinates.
        rows = rows[rows["frame_filename"].astype(str).str.strip() != ""]
        rows = rows.dropna(subset=["lon", "lat"])

        if rows.empty:
            return 0

        geo_txt_path = output_dir / "geo.txt"
        with geo_txt_path.open("w") as f:
            f.write("EPSG:4326\n")  # WebODM CRS header
            for _, row in rows.iterrows():
                alt   = 0.0 if pd.isna(row["alt"]) else float(row["alt"])
                fname = str(row["frame_filename"]).replace(" ", "%20")
                f.write(f"{fname}\t{float(row['lon']):.10f}\t{float(row['lat']):.10f}\t{alt:.4f}\n")

        return len(rows)

    # -----------------------------------------------------------------------
    # Utility helpers
    # -----------------------------------------------------------------------

    @staticmethod
    def _safe_name(value: str) -> str:
        """Convert an arbitrary string to a filesystem-safe identifier.

        Replaces any character that is not alphanumeric, a dash, or an
        underscore with an underscore.  Strips leading/trailing underscores.
        Falls back to "sensor" if the result is empty (e.g. the input was
        all punctuation).

        Used to turn sensor column names (which may contain spaces, slashes,
        or non-ASCII characters) into safe filenames for GeoTIFF output.

        Examples:
            "Temperature (°C)" → "Temperature___C"
            "/Salinity"        → "Salinity"
            "___"              → "sensor"

        Args:
            value: The raw string to sanitise.

        Returns:
            A filesystem-safe string.
        """
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "sensor"

    # -----------------------------------------------------------------------
    # Progress / status callback helpers
    # -----------------------------------------------------------------------
    # All five helpers follow the same pattern: check whether the callback was
    # supplied and call it if so.  This avoids repeating the None-guard at
    # every call site, keeping the business logic clean.

    @staticmethod
    def _jpeg_params(frame_quality: str) -> list[int]:
        """Return cv2.imwrite() JPEG parameter list for the given quality string.

        "Original" maps to quality 95 (high fidelity, reasonable file size).
        Named presets map to standard JPEG quality values.
        """
        _MAP = {
            "Original":     95,
            "High (90%)":   90,
            "Medium (75%)": 75,
            "Low (50%)":    50,
        }
        q = _MAP.get(frame_quality, 95)
        return [int(cv2.IMWRITE_JPEG_QUALITY), q]

    def _emit_progress(self, config: PipelineConfig, value: int) -> None:
        """Update the top-level progress bar (0–100)."""
        if config.progress_callback:
            config.progress_callback(value)

    def _emit_subprogress(self, config: PipelineConfig, value: int) -> None:
        """Update the per-video sub-progress bar (0–100)."""
        if config.subprogress_callback:
            config.subprogress_callback(value)

    def _emit_status(self, config: PipelineConfig, text: str) -> None:
        """Update the main status label, log to the Python logger, and append to the GUI log.

        Note: _emit_status() sends to ALL three sinks (Qt status label, Python
        logger, GUI log view) because high-level status messages are always
        worth recording.  Sub-step details use _emit_substatus() instead, which
        only goes to the sub-status label and doesn't flood the log.
        """
        if config.status_callback:
            config.status_callback(text)
        self.logger.info(text)
        if config.log_callback:
            config.log_callback(text)

    def _emit_substatus(self, config: PipelineConfig, text: str) -> None:
        """Update the per-video sub-status label (does NOT write to the log view)."""
        if config.substatus_callback:
            config.substatus_callback(text)

    def _emit_log(self, config: PipelineConfig, text: str) -> None:
        """Append a line to the GUI log view and log it to the Python logger.

        Used for file-level detail messages (paths written, counts, warnings)
        that are useful in the log but don't need to appear in the status bar.
        """
        self.logger.info(text)
        if config.log_callback:
            config.log_callback(text)

    def _emit_segment_completed(
        self,
        config: PipelineConfig,
        interval: SelectedTimeRange,
        output_dir: Path,
        status: str,
    ) -> None:
        """Fire the segment_completed_callback with a SegmentRecord.

        Called immediately after each segment directory is finalised so
        MainWindow can append the record to segment_history and auto-save
        the workspace without waiting for the full job to finish.
        """
        if config.segment_completed_callback:
            record = SegmentRecord(
                job_id=config.job_id,
                interval=interval,
                output_path=str(output_dir),
                status=status,
                processed_at=datetime.utcnow(),
            )
            config.segment_completed_callback(record)
