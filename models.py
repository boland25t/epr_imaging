# models.py — Pure data model classes
#
# This module contains only dataclasses and lightweight value objects.  There is
# no I/O, no Qt, and no processing logic here.  Every other module imports from
# this file; it must remain dependency-free so the import graph stays clean.
#
# Conceptual hierarchy:
#   VideoRecord            — one video file on disk
#   SensorChannel          — one column of sensor data to extract from a CSV
#   SensorFileConfig       — one sensor CSV file plus its timestamp + channel config
#   TimeValueSourceConfig  — one CSV column that represents a single scalar timeseries
#                            (used for latitude, longitude, altitude)
#   NavigationConfig       — bundles lat, lon, and optionally alt sources together
#   SelectedTimeRange      — a half-open [start, end] interval chosen by the user
#                            for frame extraction

from __future__ import annotations

from dataclasses import dataclass, field  # dataclass decorator + field() for mutable defaults
from datetime import datetime             # Used to store parsed timestamps as naive UTC datetimes
from pathlib import Path                  # Cross-platform file-path handling
from typing import Any                    # Used in to_dict() return types for JSON-serialisable dicts


# ---------------------------------------------------------------------------
# VideoRecord
# ---------------------------------------------------------------------------

@dataclass
class VideoRecord:
    """Represents a single video file that has been scanned from disk.

    VideoService.scan_directory() produces these objects.  The pipeline uses
    them to know which files to open, when each file starts/ends in absolute
    time, and how fast to seek through frames.

    All datetime values are naive (no timezone).  The convention throughout
    the application is to treat every naive datetime as UTC.
    """

    # Absolute path to the video file on disk.
    path: Path

    # Just the filename portion (e.g. "2026_01_18T15_36_28.MP4").  Kept
    # separately so we can display/log it without having to call path.name.
    filename: str

    # Wall-clock moment when the first frame of the video was captured.
    # Parsed from the filename if a matching pattern is found; falls back to
    # the file's modification time if no pattern matches.
    start_time: datetime

    # Derived from start_time + duration_s.  Marks the moment the last frame
    # was captured, which is used to check which GPS fixes overlap this video.
    end_time: datetime

    # Total recording duration in seconds, computed from frame_count / fps
    # as reported by OpenCV.
    duration_s: float

    # Frames per second as reported by the video container.  Optional because
    # a small number of containers omit this metadata; the pipeline falls back
    # to cap.get(CAP_PROP_FPS) at extraction time if this is None.
    fps: float | None = None

    # Human-readable tag that records how start_time was determined.
    # Possible values: "filename:<fmt>", "auto:<fmt>", "filesystem_mtime".
    # Useful for debugging time-alignment issues.
    time_source: str = "filename"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON storage in workspace files.

        datetime and Path fields are converted to strings because JSON has no
        native representation for those types.
        """
        return {
            "filename": self.filename,
            "path": str(self.path),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "time_source": self.time_source,
        }


# ---------------------------------------------------------------------------
# SensorChannel
# ---------------------------------------------------------------------------

@dataclass
class SensorChannel:
    """Describes one data column that should be extracted from a sensor CSV.

    A SensorFileConfig can contain multiple SensorChannel entries — one per
    measurement column the user wants to include in master.csv and GeoTIFFs
    (e.g. temperature, salinity, turbidity).
    """

    # The exact column header in the CSV as it appears on disk.
    source_column: str

    # The name that will appear in master.csv and raster filenames.  Allows
    # the user to rename an awkward raw column header to something readable.
    display_name: str

    # Physical units for display purposes (e.g. "°C", "PSU", "NTU").
    # Not used in computation; purely informational.
    units: str = ""

    # When True, the display_name is taken directly from the CSV column header
    # rather than the manually entered display_name field.  The import dialog
    # sets this to True when the user hasn't overridden the name.
    use_header_name: bool = True

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON workspace storage."""
        return {
            "source_column": self.source_column,
            "display_name": self.display_name,
            "units": self.units,
            "use_header_name": self.use_header_name,
        }


# ---------------------------------------------------------------------------
# SensorFileConfig
# ---------------------------------------------------------------------------

@dataclass
class SensorFileConfig:
    """Configuration for a single sensor CSV file.

    Captures everything needed to load the file, locate timestamps, and
    extract the desired sensor channels.  Used both in the pipeline (for
    merging sensor data into master.csv and producing GeoTIFFs) and in the
    import dialog (for previewing/validating the file).

    start_time / end_time are pre-computed from the CSV's timestamp range so
    the UI can show coverage without re-reading the file.
    """

    # Path to the CSV (or .ppi) file on disk.
    csv_path: Path

    # Name of the column that contains timestamps.  Combined with date_column
    # when timestamps and dates are in separate columns.
    timestamp_column: str

    # Optional separate date column.  When set, SensorService combines this
    # column with timestamp_column to produce absolute unix timestamps.  Needed
    # for instruments that log date and time in separate fields.
    date_column: str | None = None

    # Ordered list of data channels to extract from this file.
    channels: list[SensorChannel] = field(default_factory=list)

    # Earliest timestamp found in the file; None if the file hasn't been read yet.
    start_time: datetime | None = None

    # Latest timestamp found in the file; None if the file hasn't been read yet.
    end_time: datetime | None = None

    # True when the source CSV has no header row (first row is data).
    no_header: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "csv_path": str(self.csv_path),
            "timestamp_column": self.timestamp_column,
            "date_column": self.date_column,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "channels": [channel.to_dict() for channel in self.channels],
            "no_header": self.no_header,
        }


# ---------------------------------------------------------------------------
# TimeValueSourceConfig
# ---------------------------------------------------------------------------

@dataclass
class TimeValueSourceConfig:
    """Configuration for a single scalar time-series column inside a CSV.

    This is a specialised version of SensorFileConfig used exclusively for
    navigation data (latitude, longitude, altitude).  Rather than extracting
    multiple named channels, it targets exactly one value column (e.g. the
    latitude column) and exposes it as a simple (time, value) series.

    NavigationConfig holds three of these — one each for lat, lon, and
    optionally alt — allowing each coordinate to come from a different file
    or column.
    """

    # Path to the source CSV file.
    csv_path: Path

    # Column that holds the timestamp for each row.
    timestamp_column: str

    # Column that holds the scalar measurement (e.g. "latitude", "Lat_deg").
    value_column: str

    # Optional separate date column, same semantics as SensorFileConfig.
    date_column: str | None = None

    # Pre-computed time bounds for UI display without re-reading the file.
    start_time: datetime | None = None
    end_time: datetime | None = None

    # True when the source CSV has no header row.  Column names are auto-
    # generated as "0", "1", "2", … and the file is loaded with header=None.
    no_header: bool = False

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "csv_path": str(self.csv_path),
            "timestamp_column": self.timestamp_column,
            "value_column": self.value_column,
            "date_column": self.date_column,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "no_header": self.no_header,
        }


# ---------------------------------------------------------------------------
# NavigationConfig
# ---------------------------------------------------------------------------

@dataclass
class NavigationConfig:
    """Bundles navigation time-series (lat, lon, and optional orientation/depth) into one object.

    Each channel can come from a different CSV file or column, which is
    common for instruments that log navigation in separate files.  All fields
    beyond latitude and longitude are optional.

    The start_time and end_time properties return the tightest bounding window
    across all configured sources, used to display the navigation coverage on
    the timeline and to clip the nav data when loading.
    """

    # TimeValueSourceConfig pointing at the latitude column.
    latitude_source: TimeValueSourceConfig

    # TimeValueSourceConfig pointing at the longitude column.
    longitude_source: TimeValueSourceConfig

    # Altitude above the substrate (acoustic altimeter).  None when not configured.
    altitude_source: TimeValueSourceConfig | None = None

    # Depth below the water surface (pressure sensor).  None when not configured.
    depth_source: TimeValueSourceConfig | None = None

    # When True, depth values are multiplied by -1 after loading so that depths
    # below the surface are negative (Z-up convention).  Use this when the
    # instrument records depth as a positive number (e.g. 45 m) and you need
    # values in the range (-∞, 0] for coordinate calculations.
    negate_depth: bool = False

    # Vehicle heading / yaw (degrees true, 0 = North, clockwise).  None when not configured.
    heading_source: TimeValueSourceConfig | None = None

    # Vehicle pitch angle (degrees, positive = nose up).  None when not configured.
    pitch_source: TimeValueSourceConfig | None = None

    # Vehicle roll angle (degrees, positive = starboard down).  None when not configured.
    roll_source: TimeValueSourceConfig | None = None

    @property
    def _all_sources(self) -> list[TimeValueSourceConfig]:
        """All configured sources as a flat list, skipping None entries."""
        return [
            src for src in (
                self.latitude_source, self.longitude_source,
                self.altitude_source, self.depth_source,
                self.heading_source, self.pitch_source, self.roll_source,
            )
            if src is not None
        ]

    @property
    def start_time(self) -> datetime | None:
        """Earliest timestamp across all configured navigation sources."""
        starts = [s.start_time for s in self._all_sources if s.start_time is not None]
        return min(starts) if starts else None

    @property
    def end_time(self) -> datetime | None:
        """Latest timestamp across all configured navigation sources."""
        ends = [s.end_time for s in self._all_sources if s.end_time is not None]
        return max(ends) if ends else None

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage.

        Includes redundant start_time / end_time keys so a workspace file
        reader can display coverage without re-loading the CSV files.
        """
        def _src(s: TimeValueSourceConfig | None) -> dict | None:
            return s.to_dict() if s is not None else None

        return {
            "latitude_source":  self.latitude_source.to_dict(),
            "longitude_source": self.longitude_source.to_dict(),
            "altitude_source":  _src(self.altitude_source),
            "depth_source":     _src(self.depth_source),
            "negate_depth":     self.negate_depth,
            "heading_source":   _src(self.heading_source),
            "pitch_source":     _src(self.pitch_source),
            "roll_source":      _src(self.roll_source),
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time":   self.end_time.isoformat()   if self.end_time   else None,
        }


# ---------------------------------------------------------------------------
# AnnotationConfig
# ---------------------------------------------------------------------------

@dataclass
class AnnotationConfig:
    """Controls what text is burned onto annotated frames and how it looks.

    enabled_items is an ordered list of field identifiers.  Each identifier
    maps to one text line on the frame:
        "filename"   → frame filename
        "timestamp"  → ISO UTC timestamp
        "lat"        → latitude
        "lon"        → longitude
        "alt"        → altitude above substrate
        "depth"      → depth below surface (if column exists in interp.csv)
        "pitch"      → vehicle pitch (if column exists)
        "roll"       → vehicle roll (if column exists)
        <any other>  → looked up as a column name in interp.csv (sensor channels)

    color_rgb is stored as a 3-element list/tuple of 0–255 integers (R, G, B).
    When use_background is True, a semi-transparent dark rectangle is drawn
    behind the text block — this is the most reliable way to keep annotations
    readable over any frame content.
    """

    enabled_items: list[str] = field(default_factory=lambda: [
        "filename", "timestamp", "lat", "lon", "alt"
    ])
    position:       str   = "top_left"   # "top_left" | "top_right" | "bottom_left" | "bottom_right"
    font_scale:     float = 0.65
    color_rgb:      tuple = (255, 255, 255)  # white default; stored as (R, G, B)
    use_background: bool  = True
    bg_opacity:     float = 0.55          # 0.0 (transparent) → 1.0 (opaque)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "enabled_items": list(self.enabled_items),
            "position":       self.position,
            "font_scale":     self.font_scale,
            "color_rgb":      list(self.color_rgb),
            "use_background": self.use_background,
            "bg_opacity":     self.bg_opacity,
        }

    @staticmethod
    def from_dict(d: dict) -> "AnnotationConfig":
        """Reconstruct from a JSON dict, applying safe defaults for missing keys."""
        return AnnotationConfig(
            enabled_items=d.get("enabled_items", ["filename", "timestamp", "lat", "lon", "alt"]),
            position=d.get("position", "top_left"),
            font_scale=float(d.get("font_scale", 0.65)),
            color_rgb=tuple(d.get("color_rgb", [255, 255, 255])),
            use_background=bool(d.get("use_background", True)),
            bg_opacity=float(d.get("bg_opacity", 0.55)),
        )


# ---------------------------------------------------------------------------
# SelectedTimeRange
# ---------------------------------------------------------------------------

@dataclass
class SelectedTimeRange:
    """A user-defined time window for frame extraction.

    The user creates one or more of these on the Pipeline tab.  The pipeline
    iterates over them in order, extracting frames and sensor data only within
    each window.  Intervals are stored in the workspace file and round-trip
    through ConfigService.save_workspace / load_workspace.
    """

    # Inclusive start of the extraction window (naive UTC datetime).
    start_time: datetime

    # Inclusive end of the extraction window (naive UTC datetime).
    end_time: datetime

    # Where the interval came from.
    source: str = "manual"   # "manual" (trackline pick) | "threshold" (sensor analysis)

    # Human-readable summary of threshold criteria, when source="threshold".
    threshold_desc: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "start_time":     self.start_time.isoformat(),
            "end_time":       self.end_time.isoformat(),
            "source":         self.source,
            "threshold_desc": self.threshold_desc,
        }


# ---------------------------------------------------------------------------
# ThresholdConstraint / ThresholdConfig
# ---------------------------------------------------------------------------

@dataclass
class ThresholdConstraint:
    """One channel + value-range criterion used in threshold interval analysis.

    Multiple constraints may be stacked (logical AND: all must be satisfied).
    channel is the display_name of the sensor channel or a nav channel key
    such as "alt", "depth", "heading".
    """

    channel:   str
    min_val:   float | None = None   # None = no lower bound
    max_val:   float | None = None   # None = no upper bound

    def to_dict(self) -> dict[str, Any]:
        return {"channel": self.channel, "min_val": self.min_val, "max_val": self.max_val}

    @staticmethod
    def from_dict(d: dict) -> "ThresholdConstraint":
        return ThresholdConstraint(
            channel=d["channel"],
            min_val=d.get("min_val"),
            max_val=d.get("max_val"),
        )


@dataclass
class ThresholdConfig:
    """A saved set of threshold constraints plus the intervals they produced.

    Stored in threshold_history so the user can reload and re-apply a previous
    threshold analysis without re-entering all the values.
    """

    constraints:  list[ThresholdConstraint] = field(default_factory=list)
    result_count: int      = 0
    created_at:   datetime = field(default_factory=datetime.now)

    def label(self) -> str:
        """Short human-readable summary for display in a list widget."""
        parts = []
        for c in self.constraints:
            bounds = []
            if c.min_val is not None:
                bounds.append(f"≥{c.min_val:.3g}")
            if c.max_val is not None:
                bounds.append(f"≤{c.max_val:.3g}")
            parts.append(f"{c.channel} {' '.join(bounds)}" if bounds else c.channel)
        return "  &  ".join(parts) + f"  ({self.result_count} intervals)"

    def to_dict(self) -> dict[str, Any]:
        return {
            "constraints":  [c.to_dict() for c in self.constraints],
            "result_count": self.result_count,
            "created_at":   self.created_at.isoformat(),
        }

    @staticmethod
    def from_dict(d: dict) -> "ThresholdConfig":
        return ThresholdConfig(
            constraints=[ThresholdConstraint.from_dict(c) for c in d.get("constraints", [])],
            result_count=int(d.get("result_count", 0)),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.utcnow(),
        )


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A named unit of pipeline work: a collection of intervals to sample.

    The output directory is derived at run time from the workspace file path:
        <workspace_dir> / f"job_{job_id:03d}"
    so the workspace must be saved before a job can be executed.

    job_id is an auto-incremented serial number.  name is required before the
    job can be submitted to the pipeline.  settings_snapshot records the
    pipeline settings at save/execute time for the history record.
    """

    job_id:            int
    name:              str  = ""
    intervals:         list[SelectedTimeRange] = field(default_factory=list)
    status:            str  = "pending"   # "pending" | "saved" | "completed" | "failed"
    settings_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "job_id":            self.job_id,
            "name":              self.name,
            "intervals":         [i.to_dict() for i in self.intervals],
            "status":            self.status,
            "settings_snapshot": self.settings_snapshot,
        }


# ---------------------------------------------------------------------------
# SegmentRecord
# ---------------------------------------------------------------------------

@dataclass
class SegmentRecord:
    """A record of one segment that the pipeline has attempted to process.

    Written to the workspace JSON when a segment completes (or fails) so the
    interval history panels can display previously-sampled regions and allow
    them to be added to new jobs.
    """

    job_id:            int                # Which job produced this segment
    interval:          SelectedTimeRange  # The time window that was sampled
    output_path:       str                # Absolute path to the segment directory
    status:            str                # "completed" | "failed" | "partial"
    processed_at:      datetime           # Wall-clock time the pipeline ran it
    job_name:          str  = ""          # Set by MainWindow after the record is created
    settings_snapshot: dict = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "job_id":            self.job_id,
            "job_name":          self.job_name,
            "interval":          self.interval.to_dict(),
            "output_path":       self.output_path,
            "status":            self.status,
            "processed_at":      self.processed_at.isoformat(),
            "settings_snapshot": self.settings_snapshot,
        }


# ---------------------------------------------------------------------------
# PhotogrammetryRun
# ---------------------------------------------------------------------------

@dataclass
class PhotogrammetryRun:
    """A record of one photogrammetry processing run for a job.

    Stored in the workspace JSON so the Workspace panel can display previously
    generated photogrammetry products alongside sensor and nav outputs.

    products maps product type keys to absolute file paths:
        "sparse_ply"    → sparse SfM point cloud
        "dense_ply"     → dense MVS point cloud
        "mesh_obj"      → triangulated mesh
        "texture_png"   → texture image (paired with mesh_obj)
        "cameras_json"  → per-camera pose in nav coordinates
        "report_pdf"    → Metashape processing report
        "metashape_psx" → Agisoft Metashape project file

    quality is one of "draft" | "normal" | "high" | "highest" and maps to
    engine-specific accuracy/quality constants at run time.
    """

    run_id:       int
    job_id:       int
    engine:       str                        # "metashape" | "colmap"
    quality:      str                        # "draft" | "normal" | "high" | "highest"
    frame_dir:    str                        # source directory of extracted frames
    output_dir:   str                        # run_NNN directory written to disk
    products:     dict[str, str] = field(default_factory=dict)
    status:       str            = "pending" # "pending"|"running"|"complete"|"failed"
    created_at:   datetime       = field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None
    error_msg:    str            = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id":       self.run_id,
            "job_id":       self.job_id,
            "engine":       self.engine,
            "quality":      self.quality,
            "frame_dir":    self.frame_dir,
            "output_dir":   self.output_dir,
            "products":     self.products,
            "status":       self.status,
            "created_at":   self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "error_msg":    self.error_msg,
        }

    @staticmethod
    def from_dict(d: dict) -> "PhotogrammetryRun":
        return PhotogrammetryRun(
            run_id=int(d["run_id"]),
            job_id=int(d["job_id"]),
            engine=d.get("engine", "metashape"),
            quality=d.get("quality", "normal"),
            frame_dir=d.get("frame_dir", ""),
            output_dir=d.get("output_dir", ""),
            products=dict(d.get("products", {})),
            status=d.get("status", "pending"),
            created_at=datetime.fromisoformat(d["created_at"]) if d.get("created_at") else datetime.utcnow(),
            completed_at=datetime.fromisoformat(d["completed_at"]) if d.get("completed_at") else None,
            error_msg=d.get("error_msg", ""),
        )


# ---------------------------------------------------------------------------
# Task / TaskStack — instance-based product pipeline
# ---------------------------------------------------------------------------
#
# A Task is one user-created unit of work: a chosen task TYPE plus its own
# settings and its own data TARGET.  Unlike the earlier fixed-card model, the
# user can create as many instances of any type as they like — e.g. two
# "sampling" tasks over the same data at different frequencies.  The TaskStack
# is an ordered list of these instances; the runner executes them top-to-bottom
# in exactly the order the user arranged them.

# Per-type metadata.
#   label       — display name in the Create-Task menu and task rows
#   requires    — data inputs the type needs (see flags below)
#   per_channel — True if the type fans out over the selected sensor channels
#   category    — grouping for the Create-Task menu
#
# Requirement flags:
#   "video"  → at least one video file is loaded
#   "nav"    → a NavigationConfig is configured
#   "sensor" → at least one SensorFileConfig is configured
#   "interp" → interp_full.csv exists (the build step has run)
TASK_INFO: dict[str, dict] = {
    "sampling":             {"label": "Sampling (frame extraction)", "requires": ["video"],               "per_channel": False, "category": "Sampling"},
    "nav_3d":               {"label": "Nav Trackline PLY",           "requires": ["interp"],               "per_channel": False, "category": "Outputs"},
    "nav_2d":               {"label": "Nav Depth GeoTIFF",           "requires": ["interp"],               "per_channel": False, "category": "Outputs"},
    "sensor_3d":            {"label": "Sensor 3D PLY",               "requires": ["interp", "sensor"],     "per_channel": True,  "category": "Outputs"},
    "sensor_2d":            {"label": "Sensor 2D GeoTIFF",           "requires": ["interp", "sensor"],     "per_channel": True,  "category": "Outputs"},
    "depth_slice_geotiffs": {"label": "Depth-Slice GeoTIFFs",        "requires": ["interp", "sensor"],     "per_channel": True,  "category": "Outputs"},
    "sensor_slices":        {"label": "PNG Depth Slices",            "requires": ["interp", "sensor"],     "per_channel": True,  "category": "Outputs"},
    "photogrammetry":       {"label": "Photogrammetry",              "requires": ["video"],                "per_channel": False, "category": "Photogrammetry"},
    "qgis_project":         {"label": "QGIS Project (.qgs)",         "requires": ["interp"],               "per_channel": False, "category": "Export"},
}

# Order the Create-Task menu groups appear in.
TASK_CATEGORIES: list[str] = ["Sampling", "Outputs", "Photogrammetry", "Export"]


@dataclass
class Task:
    """One instance of a task type, with its own target and settings.

    task_id    — stable identity within a TaskStack (assigned on creation).
    task_type  — key into TASK_INFO.
    target     — where it runs: {"kind": "full"} or
                 {"kind": "job", "job_id": int, "name": str}.
    settings   — type-specific parameters as a JSON-serialisable dict.
    channels   — sensor channels to fan out over (per-channel types only);
                 empty means "all channels found in the target's interp CSV".
    depends_on — task_id of an upstream task whose output feeds this one.
                 Currently used by photogrammetry tasks to consume frames from
                 a sampling task in the same stack without specifying the
                 directory manually.
    """

    task_id:    int
    task_type:  str
    target:     dict = field(default_factory=lambda: {"kind": "full"})
    settings:   dict = field(default_factory=dict)
    channels:   list[str] = field(default_factory=list)
    depends_on: int | None = None

    @property
    def type_label(self) -> str:
        return TASK_INFO.get(self.task_type, {}).get("label", self.task_type)

    @property
    def per_channel(self) -> bool:
        return bool(TASK_INFO.get(self.task_type, {}).get("per_channel", False))

    @property
    def requires(self) -> list[str]:
        return list(TASK_INFO.get(self.task_type, {}).get("requires", []))

    def target_label(self) -> str:
        if self.target.get("kind") == "job":
            return self.target.get("name") or f"Job #{self.target.get('job_id')}"
        return "Full dataset"

    def display_label(self) -> str:
        """One-line summary used in the stack list, e.g. 'Sampling @ 5 Hz — Full dataset'."""
        bits = [self.type_label]
        s = self.settings
        if self.task_type == "sampling":
            if s.get("mode") == "dynamic":
                bits.append(f"~{s.get('spacing_m', 1.0):g} m spacing")
            else:
                bits.append(f"@ {s.get('frame_rate', 1.0):g} Hz")
        elif self.task_type == "photogrammetry":
            bits.append(f"{s.get('engine', 'Metashape')} / {s.get('quality', 'Normal')}")
            if self.depends_on is not None:
                bits.append(f"← Task #{self.depends_on}")
        elif self.per_channel and self.channels:
            bits.append(", ".join(self.channels))
        suffix = "  —  " + self.target_label()
        return "  ·  ".join(bits) + suffix

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id":    self.task_id,
            "task_type":  self.task_type,
            "target":     dict(self.target),
            "settings":   dict(self.settings),
            "channels":   list(self.channels),
            "depends_on": self.depends_on,
        }

    @staticmethod
    def from_dict(d: dict) -> "Task":
        return Task(
            task_id=int(d.get("task_id", 0)),
            task_type=d["task_type"],
            target=dict(d.get("target", {"kind": "full"})),
            settings=dict(d.get("settings", {})),
            channels=list(d.get("channels", [])),
            depends_on=d.get("depends_on"),
        )


@dataclass
class TaskStack:
    """An ordered list of Task instances, executed top-to-bottom on run."""

    tasks:        list[Task] = field(default_factory=list)
    _next_task_id: int = 1

    def new_id(self) -> int:
        tid = self._next_task_id
        self._next_task_id += 1
        return tid

    def add(self, task: Task) -> None:
        self.tasks.append(task)

    def remove(self, task_id: int) -> None:
        self.tasks = [t for t in self.tasks if t.task_id != task_id]

    def move(self, task_id: int, delta: int) -> None:
        """Move the task with task_id up (delta<0) or down (delta>0) by |delta|."""
        idx = next((i for i, t in enumerate(self.tasks) if t.task_id == task_id), None)
        if idx is None:
            return
        new_idx = max(0, min(len(self.tasks) - 1, idx + delta))
        self.tasks.insert(new_idx, self.tasks.pop(idx))

    def get(self, task_id: int) -> "Task | None":
        return next((t for t in self.tasks if t.task_id == task_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "tasks":         [t.to_dict() for t in self.tasks],
            "_next_task_id": self._next_task_id,
        }

    @staticmethod
    def from_dict(d: dict) -> "TaskStack":
        tasks = [Task.from_dict(t) for t in d.get("tasks", []) if t.get("task_type") in TASK_INFO]
        next_id = int(d.get("_next_task_id", 0)) or (max((t.task_id for t in tasks), default=0) + 1)
        return TaskStack(tasks=tasks, _next_task_id=next_id)
