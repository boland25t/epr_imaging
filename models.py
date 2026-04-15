from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class VideoRecord:
    path: Path
    filename: str
    start_time: datetime
    end_time: datetime
    duration_s: float
    fps: float | None = None
    time_source: str = "filename"

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "path": str(self.path),
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "duration_s": self.duration_s,
            "fps": self.fps,
            "time_source": self.time_source,
        }


@dataclass
class SensorChannel:
    source_column: str
    display_name: str
    units: str = ""
    use_header_name: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_column": self.source_column,
            "display_name": self.display_name,
            "units": self.units,
            "use_header_name": self.use_header_name,
        }


@dataclass
class SensorFileConfig:
    csv_path: Path
    timestamp_column: str
    channels: list[SensorChannel] = field(default_factory=list)
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "csv_path": str(self.csv_path),
            "timestamp_column": self.timestamp_column,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "channels": [channel.to_dict() for channel in self.channels],
        }


@dataclass
class TimeValueSourceConfig:
    csv_path: Path
    timestamp_column: str
    value_column: str
    start_time: datetime | None = None
    end_time: datetime | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "csv_path": str(self.csv_path),
            "timestamp_column": self.timestamp_column,
            "value_column": self.value_column,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class NavigationConfig:
    latitude_source: TimeValueSourceConfig
    longitude_source: TimeValueSourceConfig
    altitude_source: TimeValueSourceConfig | None = None

    @property
    def start_time(self) -> datetime | None:
        starts = [
            src.start_time
            for src in (self.latitude_source, self.longitude_source, self.altitude_source)
            if src is not None and src.start_time is not None
        ]
        return min(starts) if starts else None

    @property
    def end_time(self) -> datetime | None:
        ends = [
            src.end_time
            for src in (self.latitude_source, self.longitude_source, self.altitude_source)
            if src is not None and src.end_time is not None
        ]
        return max(ends) if ends else None

    def to_dict(self) -> dict[str, Any]:
        return {
            "latitude_source": self.latitude_source.to_dict(),
            "longitude_source": self.longitude_source.to_dict(),
            "altitude_source": self.altitude_source.to_dict() if self.altitude_source else None,
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
        }


@dataclass
class SelectedTimeRange:
    start_time: datetime
    end_time: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
        }
