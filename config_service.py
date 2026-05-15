from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from models import (
    NavigationConfig,
    SelectedTimeRange,
    SensorChannel,
    SensorFileConfig,
    TimeValueSourceConfig,
    VideoRecord,
)


def _to_relative(path: str | Path | None, base: Path) -> str | None:
    """Return path as a string relative to base directory, or None if path is None/empty."""
    if not path:
        return None
    try:
        return str(Path(path).relative_to(base))
    except ValueError:
        return str(path)  # not under base — keep absolute


def _resolve(path: str | None, base: Path) -> Path | None:
    """Resolve a (possibly relative) path string against base directory."""
    if not path:
        return None
    p = Path(path)
    if p.is_absolute():
        return p
    return (base / p).resolve()


class ConfigService:
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
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def save_workspace(
        path: str | Path,
        *,
        video_directory: str,
        filename_datetime_format: str,
        navigation_file: NavigationConfig | None,
        sensor_files: list[SensorFileConfig],
        selected_intervals: list[SelectedTimeRange],
        output_directory: str,
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
        applied_steps: list[str] | None = None,
        sampling_mode: str = "fixed",
        dynamic_target_spacing_m: float = 2.0,
        dynamic_min_frequency_hz: float = 0.1,
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid_size: int = 8,
    ) -> None:
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
            d["latitude_source"] = rel_time_value_source(nav.latitude_source)
            d["longitude_source"] = rel_time_value_source(nav.longitude_source)
            if nav.altitude_source:
                d["altitude_source"] = rel_time_value_source(nav.altitude_source)
            return d

        payload = {
            "video_directory": _to_relative(video_directory, base) or video_directory,
            "filename_datetime_format": filename_datetime_format,
            "navigation_file": rel_nav(navigation_file) if navigation_file else None,
            "sensor_files": [rel_sensor_file(sf) for sf in sensor_files],
            "selected_intervals": [interval.to_dict() for interval in selected_intervals],
            "output_directory": _to_relative(output_directory, base) or output_directory,
            "frame_rate": frame_rate,
            "generate_sensor_tiffs": generate_sensor_tiffs,
            "annotate_frames": annotate_frames,
            "frame_quality": frame_quality,
            "altitude_threshold": altitude_threshold,
            "depth_threshold": depth_threshold,
            "speed_threshold": speed_threshold,
            "min_segment_frames": min_segment_frames,
            "depth_source": rel_sensor_file(depth_source) if depth_source else None,
            "speed_source": rel_sensor_file(speed_source) if speed_source else None,
            "applied_steps": applied_steps or [],
            "sampling_mode": sampling_mode,
            "dynamic_target_spacing_m": dynamic_target_spacing_m,
            "dynamic_min_frequency_hz": dynamic_min_frequency_hz,
            "clahe_clip_limit": clahe_clip_limit,
            "clahe_tile_grid_size": clahe_tile_grid_size,
        }
        Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @staticmethod
    def load_workspace(path: str | Path) -> dict:
        path = Path(path).resolve()
        base = path.parent
        data = json.loads(path.read_text(encoding="utf-8"))

        navigation_file = None
        if data.get("navigation_file"):
            nav = data["navigation_file"]
            navigation_file = NavigationConfig(
                latitude_source=ConfigService._load_time_value_source(nav["latitude_source"], base),
                longitude_source=ConfigService._load_time_value_source(nav["longitude_source"], base),
                altitude_source=ConfigService._load_time_value_source(nav["altitude_source"], base) if nav.get("altitude_source") else None,
            )
        sensor_files = [ConfigService._load_sensor_file(item, base) for item in data.get("sensor_files", [])]
        selected_intervals = [ConfigService._load_interval(item) for item in data.get("selected_intervals", [])]

        video_dir = _resolve(data.get("video_directory"), base)
        output_dir = _resolve(data.get("output_directory"), base)

        depth_source = ConfigService._load_sensor_file(data["depth_source"], base) if data.get("depth_source") else None
        speed_source = ConfigService._load_sensor_file(data["speed_source"], base) if data.get("speed_source") else None

        return {
            "video_directory": str(video_dir) if video_dir else "",
            "filename_datetime_format": data.get("filename_datetime_format", ""),
            "navigation_file": navigation_file,
            "sensor_files": sensor_files,
            "selected_intervals": selected_intervals,
            "output_directory": str(output_dir) if output_dir else "",
            "frame_rate": float(data.get("frame_rate", 1.0)),
            "generate_sensor_tiffs": bool(data.get("generate_sensor_tiffs", True)),
            "annotate_frames": bool(data.get("annotate_frames", False)),
            "frame_quality": data.get("frame_quality", "Original"),
            "altitude_threshold": data.get("altitude_threshold"),
            "depth_threshold": data.get("depth_threshold"),
            "speed_threshold": data.get("speed_threshold"),
            "min_segment_frames": int(data.get("min_segment_frames", 1)),
            "depth_source": depth_source,
            "speed_source": speed_source,
            "applied_steps": ConfigService._detect_applied_steps(output_dir) if output_dir else data.get("applied_steps", []),
            "sampling_mode": data.get("sampling_mode", "fixed"),
            "dynamic_target_spacing_m": float(data.get("dynamic_target_spacing_m", 2.0)),
            "dynamic_min_frequency_hz": float(data.get("dynamic_min_frequency_hz", 0.1)),
            "clahe_clip_limit": float(data.get("clahe_clip_limit", 2.0)),
            "clahe_tile_grid_size": int(data.get("clahe_tile_grid_size", 8)),
        }

    @staticmethod
    def _load_time_value_source(data: dict, base: Path) -> TimeValueSourceConfig:
        return TimeValueSourceConfig(
            csv_path=_resolve(data["csv_path"], base),
            timestamp_column=data["timestamp_column"],
            value_column=data["value_column"],
            start_time=ConfigService._parse_dt(data.get("start_time")),
            end_time=ConfigService._parse_dt(data.get("end_time")),
        )

    @staticmethod
    def _load_sensor_file(data: dict, base: Path) -> SensorFileConfig:
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
            channels=channels,
            start_time=ConfigService._parse_dt(data.get("start_time")),
            end_time=ConfigService._parse_dt(data.get("end_time")),
        )

    @staticmethod
    def _parse_dt(value: str | None) -> datetime | None:
        if not value:
            return None
        try:
            return datetime.fromisoformat(value)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _load_interval(data: dict) -> SelectedTimeRange:
        return SelectedTimeRange(
            start_time=datetime.fromisoformat(data["start_time"]),
            end_time=datetime.fromisoformat(data["end_time"]),
        )

    @staticmethod
    def _detect_applied_steps(output_dir: Path) -> list[str]:
        applied = set()
        if not output_dir.exists():
            return []
        for segment_dir in output_dir.iterdir():
            if segment_dir.is_dir() and segment_dir.name.startswith("segment_"):
                frames_dir = segment_dir / "frames"
                sensors_dir = segment_dir / "sensors"
                annotated_dir = segment_dir / "frames_annotated"
                master_csv = segment_dir / "master.csv"
                if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
                    applied.add("extract_frames")
                if master_csv.exists():
                    applied.add("generate_sensor_rasters")  # assuming master implies rasters
                if sensors_dir.exists() and any(sensors_dir.glob("*.tif")):
                    applied.add("generate_sensor_rasters")
                if annotated_dir.exists() and any(annotated_dir.glob("*.jpg")):
                    applied.add("annotate_frames")
        return sorted(applied)
