from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable
import logging
import math
import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import griddata

from dynamicsampling import create_dynamic_sample_schedule
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
    depth_source: SensorFileConfig | None = None
    speed_source: SensorFileConfig | None = None
    altitude_threshold: float | None = None
    depth_threshold: float | None = None
    speed_threshold: float | None = None
    sensor_thresholds: dict[str, tuple[float | None, float | None]] = field(default_factory=dict)
    min_segment_frames: int = 1
    frame_rate: float = 1.0
    sampling_mode: str = "fixed"  # "fixed" or "dynamic"
    dynamic_target_spacing_m: float = 2.0
    dynamic_min_frequency_hz: float = 0.1
    sample_images: bool = True
    generate_sensor_rasters: bool = True
    annotate_frames: bool = False
    selected_steps: list[str] = field(default_factory=lambda: ["extract_frames", "generate_sensor_rasters", "annotate_frames"])
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
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

        selected_steps = config.selected_steps or ["extract_frames", "generate_sensor_rasters", "annotate_frames"]
        sensor_only = not config.sample_images or "interpolate_only" in selected_steps
        run_extract = "extract_frames" in selected_steps and not sensor_only
        run_update_master = "update_master" in selected_steps or sensor_only
        run_rasters = "generate_sensor_rasters" in selected_steps and not sensor_only
        run_annotate = "annotate_frames" in selected_steps and not sensor_only
        run_clahe = "apply_clahe" in selected_steps and not sensor_only
        if not any((run_extract, run_update_master, run_rasters, run_annotate, run_clahe, sensor_only)):
            raise ValueError("At least one processing step must be selected.")

        has_thresholds = (
            bool(config.altitude_threshold) or
            bool(config.depth_threshold) or
            bool(config.speed_threshold) or
            bool(config.sensor_thresholds)
        )

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
        if config.depth_source:
            self._emit_log(config, f"Loading depth source: {config.depth_source.csv_path}")
            sensor_frames.append((config.depth_source, SensorService.load_sensor_dataframe(config.depth_source)))
        if config.speed_source:
            self._emit_log(config, f"Loading speed source: {config.speed_source.csv_path}")
            sensor_frames.append((config.speed_source, SensorService.load_sensor_dataframe(config.speed_source)))
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
            clahe_dir = segment_dir / "frames_clahe"
            master_csv = segment_dir / "master.csv"

            # ------------------------------------------------------------------
            # Sensor-only mode: interpolate all sensors at sample rate, no images
            # ------------------------------------------------------------------
            if sensor_only:
                segment_dir.mkdir(parents=True, exist_ok=True)
                sensors_dir.mkdir(parents=True, exist_ok=True)
                master_exists = master_csv.exists()
                if master_exists and not run_update_master:
                    self._emit_log(config, f"Sensor-only master already exists for interval {idx + 1}; updating.")
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
                master_df.to_csv(master_csv, index=False)
                self._emit_progress(config, 60)
                self._emit_status(config, f"Sensor-only master CSV written ({len(master_df)} rows, no images).")
                produced_dirs.append(segment_dir)
                continue

            # ------------------------------------------------------------------
            # Threshold-based sampling: read dense master.csv, apply thresholds,
            # split into contiguous ranges, extract frames for each range
            # ------------------------------------------------------------------
            frames_exist = frames_dir.exists() and any(frames_dir.glob("*.jpg"))
            master_exists = master_csv.exists()

            if run_extract and has_thresholds and master_exists and not frames_exist:
                self._emit_progress(config, 35)
                self._emit_status(config, "Applying thresholds to sensor-only data, splitting into ranges...")
                dense_df = pd.read_csv(master_csv)
                mask = self._apply_thresholds(dense_df, config)
                valid_df = dense_df[mask].reset_index(drop=True)
                groups = self._split_contiguous_groups(valid_df, config.frame_rate, config.min_segment_frames)
                self._emit_log(config, f"Interval {idx + 1}: {mask.sum()} of {len(dense_df)} rows pass thresholds → {len(groups)} range(s).")

                for grp_idx, grp_df in enumerate(groups):
                    range_name = f"{segment_name}_range_{grp_idx + 1:03d}"
                    range_dir = config.output_directory / range_name
                    range_frames_dir = range_dir / "frames"
                    range_master_csv = range_dir / "master.csv"
                    range_dir.mkdir(parents=True, exist_ok=True)
                    range_frames_dir.mkdir(parents=True, exist_ok=True)
                    self._emit_substatus(config, f"Extracting range {grp_idx + 1}/{len(groups)}: {len(grp_df)} frames...")

                    target_times = grp_df["unix_time"].tolist()
                    frame_parts = []
                    for video in videos:
                        vstart = video.start_time.timestamp()
                        vend = video.end_time.timestamp()
                        vtimes = [t for t in target_times if vstart <= t <= vend]
                        if vtimes:
                            part = self._extract_frames_for_timestamps(video, vtimes, range_frames_dir, config)
                            if not part.empty:
                                frame_parts.append(part)

                    if not frame_parts:
                        self._emit_log(config, f"No frames extracted for range {grp_idx + 1}; skipping.")
                        continue

                    frame_df = pd.concat(frame_parts, ignore_index=True).sort_values("unix_time").reset_index(drop=True)
                    range_master_df = self._build_master_dataframe(frame_df, nav_sources, sensor_frames)
                    range_master_df.to_csv(range_master_csv, index=False)
                    produced_dirs.append(range_dir)
                    self._emit_log(config, f"Range {grp_idx + 1}: {len(frame_df)} frames → {range_name}")

                self._emit_progress(config, 90)
                self._emit_status(config, f"Threshold sampling complete: {len(groups)} range(s) produced.")
                continue

            # ------------------------------------------------------------------
            # Normal extraction path
            # ------------------------------------------------------------------
            if run_extract:
                segment_dir.mkdir(parents=True, exist_ok=True)
                frames_dir.mkdir(parents=True, exist_ok=True)
            if run_rasters or run_update_master:
                sensors_dir.mkdir(parents=True, exist_ok=True)

            rasters_exist = sensors_dir.exists() and any(sensors_dir.glob("*.tif"))
            annotations_exist = annotated_dir.exists() and any(annotated_dir.glob("*.jpg"))

            frame_df: pd.DataFrame
            master_df: pd.DataFrame

            if run_extract and not frames_exist:
                self._emit_progress(config, 35)
                self._emit_status(config, "Extracting frames...")
                self._emit_subprogress(config, 0)
                self._emit_substatus(config, "Waiting to start frame extraction...")
                frame_parts = []
                if config.sampling_mode == "dynamic" and nav_sources:
                    self._emit_substatus(config, "Computing dynamic sample schedule...")
                    dynamic_times = self._get_dynamic_sample_times(nav_sources, interval, config)
                    self._emit_log(config, f"Dynamic sampling: {len(dynamic_times)} target frames for interval {idx + 1}")
                    for video in videos:
                        vstart = video.start_time.timestamp()
                        vend = video.end_time.timestamp()
                        vtimes = [t for t in dynamic_times if vstart <= t <= vend]
                        if vtimes:
                            part = self._extract_frames_for_timestamps(video, vtimes, frames_dir, config)
                            if not part.empty:
                                frame_parts.append(part)
                else:
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
                master_df.to_csv(master_csv, index=False)
                self._emit_progress(config, 60)
                self._emit_status(config, "Master CSV created.")
            elif run_extract and frames_exist:
                self._emit_log(config, f"Frames already exist for interval {idx + 1}; skipping extraction.")
                if not master_exists:
                    self._emit_log(config, f"Master CSV missing for existing frames in interval {idx + 1}; cannot proceed.")
                    continue
                master_df = pd.read_csv(master_csv)
                self._emit_progress(config, 60)
                self._emit_status(config, "Loaded existing master CSV.")
            else:
                if not segment_dir.exists() or not master_csv.exists():
                    raise RuntimeError("Frame extraction output must already exist before running later steps.")
                master_df = pd.read_csv(master_csv)
                self._emit_progress(config, 60)
                if run_update_master:
                    self._emit_status(config, "Updating existing master CSV with new sources...")
                    master_df = self._update_master_dataframe(master_df, nav_sources, sensor_frames)
                    master_df.to_csv(master_csv, index=False)
                    self._emit_log(config, f"Updated master CSV for interval {idx + 1}.")
                    self._emit_status(config, "Master CSV updated.")
                else:
                    self._emit_status(config, "Loaded existing master CSV.")

            if run_rasters and not rasters_exist:
                for column in self._sensor_value_columns(master_df):
                    safe_name = self._safe_name(column)
                    output_tif = sensors_dir / f"{safe_name}.tif"
                    self._emit_log(config, f"Generating raster for {column} -> {output_tif.name}")
                    self._create_sensor_raster(master_df, column, output_tif)
                self._emit_progress(config, 80)
                self._emit_status(config, "Sensor TIFF generation complete.")
            elif run_rasters and rasters_exist:
                self._emit_log(config, f"Raster TIFFs already exist for interval {idx + 1}; skipping generation.")
                self._emit_progress(config, 80)
                self._emit_status(config, "Sensor TIFFs already exist.")

            if run_annotate and not annotations_exist:
                self._annotate_frames(frames_dir, annotated_dir, master_df)
                self._emit_progress(config, 93)
                self._emit_status(config, "Annotation complete.")
            elif run_annotate and annotations_exist:
                self._emit_log(config, f"Annotated frames already exist for interval {idx + 1}; skipping annotation.")
                self._emit_progress(config, 93)
                self._emit_status(config, "Annotated frames already exist.")

            clahe_exist = clahe_dir.exists() and any(clahe_dir.glob("*.jpg"))
            if run_clahe and not clahe_exist:
                self._apply_clahe_to_frames(frames_dir, clahe_dir, config)
                self._emit_progress(config, 97)
                self._emit_status(config, "CLAHE complete.")
            elif run_clahe and clahe_exist:
                self._emit_log(config, f"CLAHE frames already exist for interval {idx + 1}; skipping.")
                self._emit_progress(config, 97)
                self._emit_status(config, "CLAHE frames already exist.")

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

    def _create_sampled_frame_df(self, videos: list[VideoRecord], interval: SelectedTimeRange, frame_rate: float) -> pd.DataFrame:
        start_ts = interval.start_time.timestamp()
        end_ts = interval.end_time.timestamp()
        duration = end_ts - start_ts
        num_samples = max(1, int(math.floor(duration * frame_rate)) + 1)
        unix_times = np.linspace(start_ts, end_ts, num_samples)
        rows = []
        for i, unix_time in enumerate(unix_times):
            rows.append({
                "unix_time": float(unix_time),
                "frame_filename": "",
                "video_filename": videos[0].filename if videos else "",
                "frame_index": i,
            })
        return pd.DataFrame(rows)

    @staticmethod
    def _build_nav_df(nav_sources: dict[str, pd.DataFrame], t_start: float, t_end: float) -> pd.DataFrame:
        if "lat" not in nav_sources or "lon" not in nav_sources:
            return pd.DataFrame()
        pad = 60.0
        lat_df = nav_sources["lat"]
        lat_df = lat_df[(lat_df["unix_time"] >= t_start - pad) & (lat_df["unix_time"] <= t_end + pad)].copy()
        if lat_df.empty:
            return pd.DataFrame()
        lon_df = nav_sources["lon"]
        result = pd.DataFrame({"unix_time": lat_df["unix_time"].values, "lat": lat_df["value"].values})
        result["lon"] = SensorService.interpolate_series(result["unix_time"], lon_df["unix_time"], lon_df["value"])
        if "alt" in nav_sources:
            alt_df = nav_sources["alt"]
            result["alt"] = SensorService.interpolate_series(result["unix_time"], alt_df["unix_time"], alt_df["value"])
        else:
            result["alt"] = 0.0
        return result.dropna(subset=["lat", "lon"]).reset_index(drop=True)

    def _get_dynamic_sample_times(
        self,
        nav_sources: dict[str, pd.DataFrame],
        interval: SelectedTimeRange,
        config: PipelineConfig,
    ) -> list[float]:
        t_start = interval.start_time.timestamp()
        t_end = interval.end_time.timestamp()
        nav_df = self._build_nav_df(nav_sources, t_start, t_end)
        if nav_df.empty:
            return []
        nav_df = nav_df[(nav_df["unix_time"] >= t_start) & (nav_df["unix_time"] <= t_end)].reset_index(drop=True)
        if len(nav_df) < 2:
            return []
        alt_col = "alt" if "alt" in nav_sources else None
        schedule = create_dynamic_sample_schedule(
            nav_df,
            time_col="unix_time",
            lat_col="lat",
            lon_col="lon",
            alt_col=alt_col,
            target_spacing_m=config.dynamic_target_spacing_m,
            min_frequency_hz=config.dynamic_min_frequency_hz,
        )
        if schedule.empty:
            return []
        return schedule["sample_time_s"].tolist()

    @staticmethod
    def _create_dynamic_sample_df(videos: list[VideoRecord], sample_times: list[float]) -> pd.DataFrame:
        if not sample_times:
            return pd.DataFrame(columns=["unix_time", "frame_filename", "video_filename", "frame_index"])
        rows = [
            {
                "unix_time": float(t),
                "frame_filename": "",
                "video_filename": videos[0].filename if videos else "",
                "frame_index": i,
            }
            for i, t in enumerate(sorted(sample_times))
        ]
        return pd.DataFrame(rows)

    def _apply_thresholds(self, df: pd.DataFrame, config: PipelineConfig) -> "pd.Series[bool]":
        mask = pd.Series(True, index=df.index)
        if config.altitude_threshold and "alt" in df.columns:
            mask &= df["alt"] <= config.altitude_threshold
        if config.depth_threshold and "Depth" in df.columns:
            mask &= df["Depth"] >= config.depth_threshold
        if config.speed_threshold and "Speed" in df.columns:
            mask &= df["Speed"] >= config.speed_threshold
        for col_name, (min_val, max_val) in (config.sensor_thresholds or {}).items():
            if col_name in df.columns:
                if min_val is not None:
                    mask &= df[col_name] >= min_val
                if max_val is not None:
                    mask &= df[col_name] <= max_val
        return mask

    def _split_contiguous_groups(self, df: pd.DataFrame, frame_rate: float, min_frames: int) -> list[pd.DataFrame]:
        if df.empty:
            return []
        gap_threshold = (1.0 / max(frame_rate, 1e-6)) * 2.5
        times = df["unix_time"].to_numpy()
        groups: list[pd.DataFrame] = []
        start_i = 0
        for i in range(1, len(times)):
            if times[i] - times[i - 1] > gap_threshold:
                grp = df.iloc[start_i:i]
                if len(grp) >= min_frames:
                    groups.append(grp.reset_index(drop=True))
                start_i = i
        grp = df.iloc[start_i:]
        if len(grp) >= min_frames:
            groups.append(grp.reset_index(drop=True))
        return groups

    def _extract_frames_for_timestamps(
        self,
        video: VideoRecord,
        target_unix_times: list[float],
        output_dir: Path,
        config: PipelineConfig,
    ) -> pd.DataFrame:
        cap = cv2.VideoCapture(str(video.path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video {video.path}")
        try:
            video_fps = video.fps or cap.get(cv2.CAP_PROP_FPS)
            if not video_fps or video_fps <= 0:
                raise RuntimeError(f"Could not determine FPS for {video.path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            video_start_ts = video.start_time.timestamp()
            sorted_times = sorted(target_unix_times)
            total_targets = len(sorted_times)
            self._emit_substatus(config, f"Extracting frames from {video.filename}")
            rows: list[dict] = []
            for saved_idx, unix_time in enumerate(sorted_times):
                offset_s = unix_time - video_start_ts
                frame_idx = int(round(offset_s * video_fps))
                frame_idx = max(0, min(frame_idx, frame_count - 1))
                if not cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx):
                    continue
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                fname = f"{video.path.stem}_fig_{saved_idx:05d}.jpg"
                fpath = output_dir / fname
                cv2.imwrite(str(fpath), frame, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                rows.append({
                    "frame_filename": fname,
                    "unix_time": unix_time,
                    "video_filename": video.filename,
                    "frame_index": frame_idx,
                })
                self._emit_subprogress(config, int(round(((saved_idx + 1) / total_targets) * 100)))
                self._emit_substatus(config, f"Extracting frames from {video.filename} | {saved_idx + 1}/{total_targets}: {fname}")
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

    def _update_master_dataframe(
        self,
        master_df: pd.DataFrame,
        nav_sources: dict[str, pd.DataFrame],
        sensor_frames: list[tuple[SensorFileConfig, pd.DataFrame]],
    ) -> pd.DataFrame:
        master = master_df.copy()
        if "timestamp_iso" not in master.columns and "unix_time" in master.columns:
            master["timestamp_iso"] = pd.to_datetime(master["unix_time"], unit="s").dt.strftime("%Y-%m-%dT%H:%M:%S.%f")

        if "lat" in nav_sources:
            master["lat"] = SensorService.interpolate_series(master["unix_time"], nav_sources["lat"]["unix_time"], nav_sources["lat"]["value"])
        if "lon" in nav_sources:
            master["lon"] = SensorService.interpolate_series(master["unix_time"], nav_sources["lon"]["unix_time"], nav_sources["lon"]["value"])
        if "alt" in nav_sources:
            master["alt"] = SensorService.interpolate_series(master["unix_time"], nav_sources["alt"]["unix_time"], nav_sources["alt"]["value"])
        elif "alt" not in master.columns:
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

    def _apply_clahe_to_frames(self, frames_dir: Path, clahe_dir: Path, config: PipelineConfig) -> None:
        clahe_dir.mkdir(parents=True, exist_ok=True)
        clahe = cv2.createCLAHE(
            clipLimit=config.clahe_clip_limit,
            tileGridSize=(config.clahe_tile_grid_size, config.clahe_tile_grid_size),
        )
        frame_paths = sorted(frames_dir.glob("*.jpg"))
        total = len(frame_paths)
        if total == 0:
            self._emit_log(config, "No frames found for CLAHE processing.")
            return
        self._emit_subprogress(config, 0)
        for i, src in enumerate(frame_paths):
            img = cv2.imread(str(src))
            if img is None:
                continue
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            lab_result = cv2.merge([clahe.apply(l), a, b])
            result = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
            cv2.imwrite(str(clahe_dir / src.name), result, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
            self._emit_subprogress(config, int(round(((i + 1) / total) * 100)))
            self._emit_substatus(config, f"Applying CLAHE: {src.name} ({i + 1}/{total})")
        self._emit_subprogress(config, 100)
        self._emit_substatus(config, f"CLAHE complete ({total} frames written to {clahe_dir.name})")

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