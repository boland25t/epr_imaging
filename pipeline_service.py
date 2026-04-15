from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
import logging
import math
import subprocess

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

from models import NavigationConfig, SelectedTimeRange, SensorFileConfig, VideoRecord
from sensor_service import SensorService
from video_service import VideoService

LogFn = Callable[[str], None]


@dataclass
class PipelineConfig:
    video_directory: Path
    output_directory: Path
    video_filename_time_format: str
    videos: list[VideoRecord] = field(default_factory=list)
    selected_intervals: list[SelectedTimeRange] = field(default_factory=list)
    navigation_file: NavigationConfig | None = None
    sensor_files: list[SensorFileConfig] = field(default_factory=list)
    frame_rate: float = 1.0
    run_metashape: bool = False
    metashape_exec: str | None = None
    generate_sensor_rasters: bool = True
    annotate_frames: bool = False
    progress_callback: callable | None = None
    status_callback: callable | None = None
    log_callback: callable | None = None
    subprogress_callback: callable | None = None
    substatus_callback: callable | None = None
    frame_quality: str = "Original"

class PipelineService:
    def __init__(self, log_fn: LogFn | None = None):
        self.log_fn = log_fn or (lambda message: None)
        self.logger = logging.getLogger("PipelineService")

    def log(self, message: str) -> None:
        self.log_fn(message)

    def run(self, config: PipelineConfig) -> list[Path]:
        self._emit_progress(config, 0)
        self._emit_status(config, "Starting pipeline...")
        self._emit_subprogress(config, 0)
        self._emit_substatus(config, "Idle.")
        config.output_directory.mkdir(parents=True, exist_ok=True)

        videos = config.videos or VideoService(config.video_filename_time_format).scan_directory(config.video_directory)[0]
        if not videos:
            raise ValueError("No videos available to process.")

        intervals = config.selected_intervals or [
            SelectedTimeRange(
                start_time=min(video.start_time for video in videos),
                end_time=max(video.end_time for video in videos),
            )
        ]

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
        self._emit_progress(config, 15)
        self._emit_status(config, "Navigation sources loaded.")

        sensor_frames: list[tuple[SensorFileConfig, pd.DataFrame]] = []
        for sensor_cfg in config.sensor_files:
            self._emit_log(config, f"Loading sensor file: {sensor_cfg.csv_path}")
            sensor_frames.append((sensor_cfg, SensorService.load_sensor_dataframe(sensor_cfg)))
        self._emit_progress(config, 25)
        self._emit_status(config, "Sensor sources loaded.")

        produced_dirs: list[Path] = []
        total_intervals = len(intervals)
        for idx, interval in enumerate(intervals):
            self._emit_status(config, f"Processing interval {idx + 1} of {total_intervals}...")
            segment_name = f"segment_{idx + 1:03d}_{interval.start_time.strftime('%Y%m%dT%H%M%S')}_{interval.end_time.strftime('%Y%m%dT%H%M%S')}"
            segment_dir = config.output_directory / segment_name
            frames_dir = segment_dir / "frames"
            sensors_dir = segment_dir / "sensors"
            annotated_dir = segment_dir / "frames_annotated"
            segment_dir.mkdir(parents=True, exist_ok=True)
            frames_dir.mkdir(parents=True, exist_ok=True)
            if config.generate_sensor_rasters:
                sensors_dir.mkdir(parents=True, exist_ok=True)

            self._emit_progress(config, 35)
            self._emit_status(config, "Extracting frames...")
            self._emit_subprogress(config, 0)
            self._emit_substatus(config, "Waiting to start frame extraction...")
            frame_parts: list[pd.DataFrame] = []
            for video in videos:
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

            frame_df = pd.concat(frame_parts, ignore_index=True).sort_values("unix_time").reset_index(drop=True)
            self._emit_subprogress(config, 0)
            self._emit_substatus(config, "Frame extraction complete.")

            master_df = self._build_master_dataframe(frame_df, nav_sources, sensor_frames)
            master_csv = segment_dir / "master.csv"
            master_df.to_csv(master_csv, index=False)

            reference_csv = segment_dir / "metashape_reference.csv"
            master_df[["frame_filename", "lon", "lat", "alt"]].to_csv(reference_csv, index=False)
            self._emit_progress(config, 60)
            self._emit_status(config, "Master CSV created.")

            if config.generate_sensor_rasters:
                for column in self._sensor_value_columns(master_df):
                    safe_name = self._safe_name(column)
                    output_tif = sensors_dir / f"{safe_name}.tif"
                    self._emit_log(config, f"Generating raster for {column} -> {output_tif.name}")
                    self._create_sensor_raster(master_df, column, output_tif)
                self._emit_progress(config, 80)
                self._emit_status(config, "Sensor TIFF generation complete.")

            if config.annotate_frames:
                self._annotate_frames(frames_dir, annotated_dir, master_df)
                self._emit_progress(config, 95)
                self._emit_status(config, "Annotation complete.")

            if config.run_metashape:
                if not config.metashape_exec:
                    raise ValueError("Run Metashape is enabled, but no executable path was provided.")
                self._emit_progress(config, 85)
                self._emit_status(config, "Running Metashape...")
                self._run_metashape(frames_dir, master_csv, segment_dir, config.metashape_exec)

            produced_dirs.append(segment_dir)

        self._emit_progress(config, 100)
        self._emit_status(config, "Pipeline complete.")
        self._emit_subprogress(config, 0)
        self._emit_substatus(config, "Idle.")
        return produced_dirs

    @staticmethod
    def _intervals_overlap(a_start: datetime, a_end: datetime, b_start: datetime, b_end: datetime) -> bool:
        return a_start < b_end and b_start < a_end

    def _extract_frames_for_interval(
        self,
        video: VideoRecord,
        interval_start: datetime,
        interval_end: datetime,
        output_dir: Path,
        sample_hz: float,
        config: PipelineConfig,
    ) -> pd.DataFrame:
        if sample_hz <= 0:
            raise ValueError("Frame rate must be > 0")

        effective_start = max(video.start_time, interval_start)
        effective_end = min(video.end_time, interval_end)
        if effective_start >= effective_end:
            return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video.path}")

        try:
            video_fps = video.fps or cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or video_fps <= 0:
                raise RuntimeError(f"Could not determine FPS for {video.path}")

            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if frame_count <= 0:
                raise RuntimeError(f"Could not determine frame count for {video.path}")

            start_offset_s = max(0.0, (effective_start - video.start_time).total_seconds())
            end_offset_s = max(0.0, (effective_end - video.start_time).total_seconds())
            if end_offset_s < start_offset_s:
                return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

            duration_s = end_offset_s - start_offset_s
            sample_count = max(1, int(math.floor(duration_s * sample_hz)) + 1)

            target_times = [start_offset_s + (i / sample_hz) for i in range(sample_count)]
            target_times = [t for t in target_times if t <= end_offset_s + 1e-9]

            if not target_times:
                return pd.DataFrame(columns=["frame_filename", "unix_time", "video_filename", "frame_index"])

            raw_target_frames = [int(round(t * video_fps)) for t in target_times]

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
            total_targets = len(target_frames)

            for saved_idx, frame_idx in enumerate(target_frames):
                ok_seek = cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                if not ok_seek:
                    continue

                ok, frame = cap.read()
                if not ok or frame is None:
                    continue

                unix_time = video.start_time.timestamp() + (frame_idx / video_fps)
                frame_dt = datetime.fromtimestamp(unix_time)
                if not (interval_start <= frame_dt <= interval_end):
                    continue

                fname = f"{video.path.stem}_fig_{saved_idx:05d}.jpg"
                fpath = output_dir / fname

                ok_write = cv2.imwrite(
                    str(fpath),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 90],
                )
                if not ok_write:
                    raise RuntimeError(f"Failed to write extracted frame to {fpath}")

                rows.append(
                    {
                        "frame_filename": fname,
                        "unix_time": unix_time,
                        "video_filename": video.filename,
                        "frame_index": frame_idx,
                    }
                )

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
            cap.release()

    def _build_master_dataframe(
        self,
        frame_df: pd.DataFrame,
        nav_sources: dict[str, pd.DataFrame],
        sensor_frames: list[tuple[SensorFileConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        master = frame_df.copy()
        master = master.sort_values("unix_time").reset_index(drop=True)
        master["timestamp_iso"] = pd.to_datetime(master["unix_time"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

        if "lat" in nav_sources:
            master["lat"] = SensorService.interpolate_series(master["unix_time"], nav_sources["lat"]["unix_time"], nav_sources["lat"]["value"])
        else:
            master["lat"] = np.nan
        if "lon" in nav_sources:
            master["lon"] = SensorService.interpolate_series(master["unix_time"], nav_sources["lon"]["unix_time"], nav_sources["lon"]["value"])
        else:
            master["lon"] = np.nan
        if "alt" in nav_sources:
            master["alt"] = SensorService.interpolate_series(master["unix_time"], nav_sources["alt"]["unix_time"], nav_sources["alt"]["value"])
        else:
            master["alt"] = 0.0

        for sensor_cfg, sensor_df in sensor_frames:
            for channel in sensor_cfg.channels:
                display_name = channel.display_name or channel.source_column
                master[display_name] = SensorService.interpolate_series(master["unix_time"], sensor_df["unix_time"], sensor_df[channel.source_column])

        ordered = ["frame_filename", "timestamp_iso", "unix_time", "lat", "lon", "alt", "video_filename", "frame_index"]
        remaining = [col for col in master.columns if col not in ordered]
        return master[ordered + remaining]

    @staticmethod
    def _sensor_value_columns(master_df: pd.DataFrame) -> list[str]:
        excluded = {"frame_filename", "timestamp_iso", "unix_time", "lat", "lon", "alt", "video_filename", "frame_index"}
        return [col for col in master_df.columns if col not in excluded]

    def _create_sensor_raster(self, df: pd.DataFrame, value_column: str, output_path: Path) -> None:
        valid = df[["lon", "lat", value_column]].dropna()
        if len(valid) < 2:
            raise ValueError(f"Not enough valid points to create raster for {value_column}")

        lons = valid["lon"].to_numpy(dtype=float)
        lats = valid["lat"].to_numpy(dtype=float)
        values = valid[value_column].to_numpy(dtype=float)

        x_unique = np.unique(lons)
        y_unique = np.unique(lats)
        width = int(np.clip(max(200, len(x_unique) * 4), 200, 1200))
        height = int(np.clip(max(200, len(y_unique) * 4), 200, 1200))
        grid_x, grid_y = np.meshgrid(np.linspace(lons.min(), lons.max(), width), np.linspace(lats.min(), lats.max(), height))

        try:
            grid_z = griddata((lons, lats), values, (grid_x, grid_y), method="linear")
        except Exception:
            grid_z = None
        if grid_z is None or np.isnan(grid_z).all():
            grid_z = griddata((lons, lats), values, (grid_x, grid_y), method="nearest")
        elif np.isnan(grid_z).any():
            nearest = griddata((lons, lats), values, (grid_x, grid_y), method="nearest")
            grid_z = np.where(np.isnan(grid_z), nearest, grid_z)

        transform = from_bounds(float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()), width, height)
        with rasterio.open(
            output_path,
            "w",
            driver="GTiff",
            height=grid_z.shape[0],
            width=grid_z.shape[1],
            count=1,
            dtype="float32",
            crs="EPSG:4326",
            transform=transform,
        ) as dst:
            dst.write(grid_z.astype("float32"), 1)

    def _annotate_frames(self, frames_dir: Path, annotated_dir: Path, master_df: pd.DataFrame) -> None:
        annotated_dir.mkdir(parents=True, exist_ok=True)
        for _, row in master_df.iterrows():
            src = frames_dir / str(row["frame_filename"])
            if not src.exists():
                continue
            img = cv2.imread(str(src))
            if img is None:
                continue

            overlay_lines = [
                f"Time: {row['timestamp_iso']}",
                f"Lat: {row['lat']:.6f}  Lon: {row['lon']:.6f}  Alt: {row['alt']:.2f}",
            ]
            for col in self._sensor_value_columns(master_df):
                overlay_lines.append(f"{col}: {row[col]:.3f}")

            y = 30
            for line in overlay_lines:
                cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
                cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
                y += 28

            cv2.imwrite(str(annotated_dir / src.name), img)

    def _run_metashape(self, frames_dir: Path, master_csv: Path, output_dir: Path, metashape_exec: str) -> None:
        script_path = output_dir / "metashape_script.py"
        project_path = output_dir / "metashape_project.psx"
        ortho_path = output_dir / "orthomosaic.tif"
        reference_path = output_dir / "metashape_reference.csv"

        master = pd.read_csv(master_csv)
        ref = master[["frame_filename", "lon", "lat", "alt"]].dropna().copy()
        ref.to_csv(reference_path, index=False)

        script_text = f'''import os\nimport Metashape\n\ndoc = Metashape.Document()\ndoc.save(r"{project_path}")\nchunk = doc.addChunk()\nphotos = [os.path.join(r"{frames_dir}", f) for f in sorted(os.listdir(r"{frames_dir}")) if f.lower().endswith(".jpg")]\nchunk.addPhotos(photos)\nchunk.importReference(path=r"{reference_path}", format=Metashape.ReferenceFormatCSV, columns="nxyz", delimiter=",")\nchunk.matchPhotos(downscale=1, generic_preselection=True, reference_preselection=True)\nchunk.alignCameras()\nchunk.buildDepthMaps(downscale=2)\nchunk.buildDenseCloud()\nchunk.buildDem(source_data=Metashape.DenseCloudData)\nchunk.buildOrthomosaic(surface_data=Metashape.ElevationData)\nchunk.exportRaster(r"{ortho_path}")\ndoc.save()\n'''
        script_path.write_text(script_text, encoding="utf-8")
        subprocess.run([metashape_exec, "-r", str(script_path)], check=True)

    @staticmethod
    def _safe_name(value: str) -> str:
        return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in value).strip("_") or "sensor"

    def _emit_progress(self, config: PipelineConfig, value: int) -> None:
        if config.progress_callback:
            config.progress_callback(value)

    def _emit_subprogress(self, config: PipelineConfig, value: int) -> None:
        if config.subprogress_callback:
            config.subprogress_callback(value)

    def _emit_status(self, config: PipelineConfig, text: str) -> None:
        if config.status_callback:
            config.status_callback(text)
        self.logger.info(text)
        if config.log_callback:
            config.log_callback(text)

    def _emit_substatus(self, config: PipelineConfig, text: str) -> None:
        if config.substatus_callback:
            config.substatus_callback(text)

    def _emit_log(self, config: PipelineConfig, text: str) -> None:
        self.logger.info(text)
        if config.log_callback:
            config.log_callback(text)