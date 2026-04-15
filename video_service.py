from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path
import re

import cv2

from models import VideoRecord


class VideoScanError(Exception):
    pass


class VideoService:
    """Scan a directory of video files, parse start times robustly, and read file durations."""

    COMMON_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(r"(20\d{6}_\d{6})"), "%Y%m%d_%H%M%S"),
        (re.compile(r"(20\d{6}-\d{6})"), "%Y%m%d-%H%M%S"),
        (re.compile(r"(20\d{2}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})"), "%Y-%m-%d_%H-%M-%S"),
        (re.compile(r"(20\d{2}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2})"), "%Y-%m-%dT%H-%M-%S"),
        (re.compile(r"(20\d{2}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})"), "%Y-%m-%dT%H:%M:%S"),
        (re.compile(r"(20\d{2}-\d{2}-\d{2}_\d{2}:\d{2}:\d{2})"), "%Y-%m-%d_%H:%M:%S"),
        (re.compile(r"(20\d{2}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"), "%Y-%m-%d %H:%M:%S"),
        (re.compile(r"(20\d{2}\d{2}\d{2}\d{2}\d{2}\d{2})"), "%Y%m%d%H%M%S"),
    ]

    VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".MP4", ".MOV", ".AVI", ".MKV"}

    def __init__(self, datetime_format: str):
        self.datetime_format = datetime_format.strip()
        self.regex = self._build_regex_from_format(self.datetime_format) if self.datetime_format else None

    def scan_directory(self, directory: str | Path) -> tuple[list[VideoRecord], list[str]]:
        root = Path(directory)
        if not root.exists() or not root.is_dir():
            raise VideoScanError(f"Invalid video directory: {root}")

        valid_records: list[VideoRecord] = []
        skipped_files: list[str] = []

        paths = [
            p
            for p in sorted(root.iterdir())
            if p.is_file()
            and p.suffix in self.VIDEO_EXTENSIONS
            and not p.name.startswith('.')
        ]
        for path in paths:
            try:
                start_time, time_source = self._parse_start_time(path)
                duration_s, fps = self._read_video_duration(path)
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
                skipped_files.append(f"{path.name} - {exc}")

        return valid_records, skipped_files

    def _parse_start_time(self, path: Path) -> tuple[datetime, str]:
        filename = path.name
        if self.regex is not None:
            match = self.regex.search(filename)
            if match:
                try:
                    return datetime.strptime(match.group(0), self.datetime_format), f"filename:{self.datetime_format}"
                except ValueError:
                    pass

        for pattern, dt_format in self.COMMON_PATTERNS:
            match = pattern.search(filename)
            if match:
                try:
                    return datetime.strptime(match.group(1), dt_format), f"auto:{dt_format}"
                except ValueError:
                    continue

        return datetime.fromtimestamp(path.stat().st_mtime), "filesystem_mtime"

    def _read_video_duration(self, path: Path) -> tuple[float, float | None]:
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise VideoScanError(f"Could not open video: {path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()

        if fps is None or fps <= 0 or frame_count is None or frame_count < 0:
            raise VideoScanError(f"Could not determine duration for: {path}")

        duration_s = frame_count / fps
        return float(duration_s), float(fps)

    @staticmethod
    def _build_regex_from_format(datetime_format: str) -> re.Pattern[str]:
        token_map = {
            "%Y": r"\d{4}",
            "%m": r"\d{2}",
            "%d": r"\d{2}",
            "%H": r"\d{2}",
            "%M": r"\d{2}",
            "%S": r"\d{2}",
        }

        escaped = re.escape(datetime_format)
        for token, pattern in token_map.items():
            escaped = escaped.replace(re.escape(token), pattern)

        return re.compile(escaped)
