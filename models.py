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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "csv_path": str(self.csv_path),
            "timestamp_column": self.timestamp_column,
            "date_column": self.date_column,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "channels": [channel.to_dict() for channel in self.channels],
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "csv_path": str(self.csv_path),
            "timestamp_column": self.timestamp_column,
            "value_column": self.value_column,
            "date_column": self.date_column,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
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
                self.pitch_source, self.roll_source,
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

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }


# ---------------------------------------------------------------------------
# Job
# ---------------------------------------------------------------------------

@dataclass
class Job:
    """A named unit of pipeline work: a collection of intervals to sample.

    The output directory is derived at run time from the workspace file path:
        <workspace_dir> / f"job_{job_id:03d}"
    so the workspace must be saved before a job can be executed.

    job_id is an auto-incremented serial number managed by MainWindow and
    stored in the workspace JSON as next_job_id.
    """

    job_id:    int
    intervals: list[SelectedTimeRange] = field(default_factory=list)
    status:    str = "pending"   # "pending" | "completed" | "failed"

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "job_id":    self.job_id,
            "intervals": [i.to_dict() for i in self.intervals],
            "status":    self.status,
        }


# ---------------------------------------------------------------------------
# SegmentRecord
# ---------------------------------------------------------------------------

@dataclass
class SegmentRecord:
    """A record of one segment that the pipeline has attempted to process.

    Written to the workspace JSON when a segment completes (or fails) so the
    Visualization tab can show previously-sampled regions and the history
    dropdown can offer them for re-use.
    """

    job_id:       int                # Which job produced this segment
    interval:     SelectedTimeRange  # The time window that was sampled
    output_path:  str                # Absolute path to the segment directory
    status:       str                # "completed" | "failed" | "partial"
    processed_at: datetime           # Wall-clock time the pipeline ran it

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a plain dict for JSON workspace storage."""
        return {
            "job_id":       self.job_id,
            "interval":     self.interval.to_dict(),
            "output_path":  self.output_path,
            "status":       self.status,
            "processed_at": self.processed_at.isoformat(),
        }
