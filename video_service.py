# video_service.py — Video directory scanning and start-time parsing
#
# Responsibilities:
#   1. Walk a directory and find all recognised video files.
#   2. Parse each filename to extract the recording start time.
#   3. Open each file with OpenCV to read duration and frame rate.
#   4. Return a list of VideoRecord objects (and a list of skipped filenames
#      with error messages for display in the UI).
#
# Time-parsing strategy (in priority order):
#   a. User-supplied custom format (set in the "Filename format" field).
#   b. A table of common auto-detected patterns (COMMON_PATTERNS).
#   c. File modification time as a last resort (least reliable).
#
# All datetime values produced here are naive (no timezone).  The application
# convention is to treat every naive datetime as UTC.

from __future__ import annotations

from datetime import datetime, timedelta  # datetime for start/end times; timedelta to add duration
from pathlib import Path                  # Cross-platform path handling
import re                                 # Regex used to match timestamp substrings in filenames

import cv2  # OpenCV — used only to query video FPS and frame count, not to decode pixels

from models import VideoRecord  # The dataclass we populate and return


class VideoScanError(Exception):
    """Raised when a video directory is missing or a file cannot be opened.

    Callers catch this to report problems in the UI rather than crashing.
    """
    pass


class VideoService:
    """Scans a directory of video files, parses start times, and reads metadata.

    One instance is created per scan (because the datetime_format can change
    between scans).  The two main entry points are:
      - scan_directory()   — returns (list[VideoRecord], list[str skipped])
      - _parse_start_time() — internal, but documented below for reference
    """

    # ---------------------------------------------------------------------------
    # Auto-detection pattern table
    # ---------------------------------------------------------------------------

    # Each entry is (compiled_regex, strptime_format).  Patterns are tried in
    # order; the first match wins.  The regex must contain exactly one capture
    # group that isolates the timestamp substring so it can be passed directly
    # to datetime.strptime().
    #
    # Ordering matters: more specific patterns (longer, more constrained) come
    # before less specific ones to avoid false positives.  For example, the
    # compact 12-digit pattern at the end could accidentally match inside a
    # longer timestamp, so it goes last.
    COMMON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        # 20260118_153628 — compact date + compact time, underscore separator
        (re.compile(r"(20\d{6}_\d{6})"), "%Y%m%d_%H%M%S"),

        # 20260118-153628 — compact date + compact time, dash separator
        (re.compile(r"(20\d{6}-\d{6})"), "%Y%m%d-%H%M%S"),

        # 2026_01_18T15_36_28 — ISO-style with underscores throughout (GoPro/survey cams)
        (re.compile(r"(20\d{2}_\d{2}_\d{2}T\d{2}_\d{2}_\d{2})"), "%Y_%m_%dT%H_%M_%S"),

        # 2026_01_18_15_36_28 — all underscores, no T separator
        (re.compile(r"(20\d{2}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2})"), "%Y_%m_%d_%H_%M_%S"),

        # 2026-01-18_15-36-28 — ISO date, dash-separated time
        (re.compile(r"(20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"), "%Y-%m-%d_%H-%M-%S"),

        # 2026-01-18T15-36-28 — ISO date, T separator, dash-separated time
        (re.compile(r"(20\d{2}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})"), "%Y-%m-%dT%H-%M-%S"),

        # 2026-01-18T15:36:28 — full ISO 8601 with colons
        (re.compile(r"(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"), "%Y-%m-%dT%H:%M:%S"),

        # 2026-01-18_15:36:28 — ISO date + colon-separated time
        (re.compile(r"(20\d{2}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"), "%Y-%m-%d_%H:%M:%S"),

        # 2026-01-18 15:36:28 — ISO date + space + colon-separated time
        (re.compile(r"(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"), "%Y-%m-%d %H:%M:%S"),

        # 20260118153628 — fully compact 14-digit form, no separators (least specific, goes last)
        (re.compile(r"(20\d{2}\d{2}\d{2}\d{2}\d{2}\d{2})"), "%Y%m%d%H%M%S"),
    ]

    # File extensions considered as video files (both lower and upper case so
    # the set works on case-sensitive Linux filesystems).
    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

    def __init__(self, datetime_format: str):
        """Initialise the service with an optional user-supplied format string.

        Args:
            datetime_format: A strptime-style format string (e.g.
                             "%Y_%m_%dT%H_%M_%S") that the user typed into the
                             "Filename format" field.  Pass an empty string to
                             skip custom-format matching and rely only on
                             COMMON_PATTERNS.
        """
        # Strip surrounding whitespace so a paste with a trailing space doesn't
        # silently break parsing.
        self.datetime_format = datetime_format.strip()

        # Pre-compile the regex derived from the user's format so we don't
        # recompile it for every file in the directory.  None means no custom
        # format was provided.
        self.regex = self._build_regex_from_format(self.datetime_format) if self.datetime_format else None

    # ---------------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------------

    def scan_directory(self, directory: str | Path) -> tuple[list[VideoRecord], list[str]]:
        """Scan a directory and return (valid_records, skipped_files).

        Iterates over every file whose extension is in VIDEO_EXTENSIONS,
        attempts to parse its start time and read its duration, and builds a
        VideoRecord for each success.  Files that fail for any reason are added
        to skipped_files with a descriptive error message so the UI can warn
        the user without aborting the whole scan.

        Args:
            directory: Path to the folder containing video files.

        Returns:
            A 2-tuple of:
              - list[VideoRecord]: successfully parsed records, sorted by filename
              - list[str]: "filename - error message" strings for failed files

        Raises:
            VideoScanError: If the directory doesn't exist or is not a directory.
        """
        root = Path(directory)

        # Validate early so the user gets a clear error rather than an empty list.
        if not root.exists() or not root.is_dir():
            raise VideoScanError(f"Invalid video directory: {root}")

        valid_records: list[VideoRecord] = []
        skipped_files: list[str] = []

        # Collect and sort paths so records are in a deterministic order
        # regardless of OS filesystem ordering.
        paths = [
            p
            for p in sorted(root.iterdir())
            if p.is_file()
            and p.suffix in self.VIDEO_EXTENSIONS
            and not p.name.startswith('.')  # Skip hidden/system files on macOS
        ]

        for path in paths:
            try:
                # Parse the recording start time from the filename (or mtime).
                start_time, time_source = self._parse_start_time(path)

                # Ask OpenCV for duration and fps (requires opening the file).
                duration_s, fps = self._read_video_duration(path)

                # Derive end_time by adding duration to start_time so interval
                # overlap checks in the pipeline can use plain datetime math.
                end_time = start_time + timedelta(seconds=duration_s)

                valid_records.append(
                    VideoRecord(
                        path=path,
                        filename=path.name,
                        start_time=start_time,
                        end_time=end_time,
                        duration_s=duration_s,
                        fps=fps,
                        time_source=time_source,
                    )
                )
            except Exception as exc:
                # Catch-all so a single bad file doesn't stop the whole scan.
                skipped_files.append(f"{path.name} - {exc}")

        return valid_records, skipped_files

    # ---------------------------------------------------------------------------
    # Internal helpers
    # ---------------------------------------------------------------------------

    def _parse_start_time(self, path: Path) -> tuple[datetime, str]:
        """Extract a recording start time from a video filename.

        Tries, in order:
          1. User-supplied custom format (self.regex / self.datetime_format).
          2. Each entry in COMMON_PATTERNS.
          3. File modification time (least reliable; used as a fallback).

        Returns:
            A 2-tuple of (start_datetime, source_tag) where source_tag is a
            human-readable string identifying which strategy succeeded.
        """
        filename = path.name

        # --- Strategy 1: custom user format ---
        if self.regex is not None:
            match = self.regex.search(filename)
            if match:
                try:
                    # match.group(0) is the full match (the regex was built to
                    # match exactly the timestamp substring).
                    return (
                        datetime.strptime(match.group(0), self.datetime_format),
                        f"filename:{self.datetime_format}",
                    )
                except ValueError:
                    # strptime can fail if the regex matched a substring that
                    # looks like a timestamp but doesn't conform to the format
                    # (e.g. month "13").  Fall through to the common patterns.
                    pass

        # --- Strategy 2: auto-detection from COMMON_PATTERNS ---
        for pattern, dt_format in self.COMMON_PATTERNS:
            match = pattern.search(filename)
            if match:
                try:
                    # group(1) isolates the timestamp substring inside the
                    # capturing group; group(0) would include any surrounding
                    # context captured by the outer regex.
                    return datetime.strptime(match.group(1), dt_format), f"auto:{dt_format}"
                except ValueError:
                    # Pattern matched but strptime failed (e.g. invalid date
                    # value); try the next pattern.
                    continue

        # --- Strategy 3: filesystem modification time (last resort) ---
        # datetime.fromtimestamp() returns local time; the rest of the
        # application treats all naive datetimes as UTC, so this is only
        # reliable when the system clock is set to UTC (true of most survey
        # computers and WSL2 environments).
        return datetime.fromtimestamp(path.stat().st_mtime), "filesystem_mtime"

    def _read_video_duration(self, path: Path) -> tuple[float, float | None]:
        """Open a video file with OpenCV and read its duration and frame rate.

        We compute duration as frame_count / fps rather than using the
        container-level duration tag because some cameras write incorrect
        container durations but have accurate frame counts.

        Args:
            path: Path to the video file.

        Returns:
            A 2-tuple of (duration_seconds, fps).

        Raises:
            VideoScanError: If the file can't be opened or lacks usable metadata.
        """
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoScanError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

        # Release immediately; we only needed the header metadata.
        cap.release()

        if fps is None or fps <= 0 or frame_count is None or frame_count < 0:
            raise VideoScanError(f"Could not determine duration for: {path}")

        # Integer frame counts are returned as float by OpenCV.
        duration_s = frame_count / fps
        return float(duration_s), float(fps)

    @staticmethod
    def _build_regex_from_format(datetime_format: str) -> re.Pattern[str]:
        """Convert a strptime format string into a compiled regex pattern.

        Each strptime token (%Y, %m, %d, %H, %M, %S) is replaced with a
        digit-count pattern, and all other characters are escaped so they
        match literally.  The resulting regex can search() a filename and
        find the embedded timestamp substring.

        Example:
            "%Y_%m_%dT%H_%M_%S"
            → re.compile(r"[0-9]{4}_[0-9]{2}_[0-9]{2}T[0-9]{2}_[0-9]{2}_[0-9]{2}")

        Args:
            datetime_format: The user-supplied strptime format string.

        Returns:
            A compiled regex whose group(0) matches the full timestamp
            substring in a filename.
        """
        # Map each strptime token to a fixed-width digit pattern.
        token_map = {
            "%Y": r"\d{4}",  # 4-digit year
            "%m": r"\d{2}",  # 2-digit month
            "%d": r"\d{2}",  # 2-digit day
            "%H": r"\d{2}",  # 2-digit hour (24-hour)
            "%M": r"\d{2}",  # 2-digit minute
            "%S": r"\d{2}",  # 2-digit second
        }

        # re.escape() makes literal characters (underscores, dashes, T, etc.)
        # safe to embed in a regex without accidentally acting as metacharacters.
        escaped = re.escape(datetime_format)

        # Replace each escaped token with its digit pattern.  We must escape
        # the token itself before searching because re.escape already escaped
        # the % sign.
        for token, pattern in token_map.items():
            escaped = escaped.replace(re.escape(token), pattern)

        return re.compile(escaped)
