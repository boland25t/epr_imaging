from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from config_service import ConfigService
from models import NavigationConfig, SelectedTimeRange, SensorFileConfig, TimeValueSourceConfig, VideoRecord
from pipeline_service import PipelineConfig, PipelineService
from video_service import VideoScanError, VideoService
from widgets.navigation_import_dialog import NavigationImportDialog
from widgets.sensor_import_dialog import SensorImportDialog
from widgets.timeline_widget import TimelineWidget


class PipelineWorker(QObject):
    finished = Signal(list)
    error = Signal(str)
    log = Signal(str)
    progress = Signal(int)
    status = Signal(str)
    subprogress = Signal(int)
    substatus = Signal(str)

    def __init__(self, config: PipelineConfig):
        super().__init__()
        self.config = config
        self.service = PipelineService(log_fn=self.log.emit)

    def run(self) -> None:
        try:
            self.config.progress_callback = self.progress.emit
            self.config.status_callback = self.status.emit
            self.config.log_callback = self.log.emit
            self.config.subprogress_callback = self.subprogress.emit
            self.config.substatus_callback = self.substatus.emit
            outputs = self.service.run(self.config)
            self.finished.emit([str(path) for path in outputs])
        except Exception as exc:
            self.error.emit(str(exc))


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video + Sensor Processing Tool")
        self.resize(1500, 900)

        self.video_directory: str = ""
        self.video_datetime_format: str = "%Y%m%d_%H%M%S"
        self.videos: list[VideoRecord] = []
        self.skipped_videos: list[str] = []
        self.sensor_files: list[SensorFileConfig] = []
        self.navigation_file: NavigationConfig | None = None
        self.selected_intervals: list[SelectedTimeRange] = []
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None

        self._build_ui()
        self._wire_signals()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Idle.")
        self.subprogress_bar.setValue(0)
        self.subprogress_label.setText("Idle.")
        self._refresh_all_views()

    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        splitter.addWidget(self._build_controls_panel())
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_summary_panel())
        splitter.setSizes([450, 720, 420])

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        video_group = QGroupBox("Video Inputs")
        video_layout = QVBoxLayout(video_group)
        video_form = QFormLayout()
        self.video_dir_edit = QLineEdit()
        self.video_format_edit = QLineEdit(self.video_datetime_format)
        video_form.addRow("Video directory:", self.video_dir_edit)
        video_form.addRow("Filename datetime format:", self.video_format_edit)
        video_layout.addLayout(video_form)

        video_button_row = QHBoxLayout()
        self.browse_video_button = QPushButton("Browse...")
        self.scan_videos_button = QPushButton("Scan Videos")
        video_button_row.addWidget(self.browse_video_button)
        video_button_row.addWidget(self.scan_videos_button)
        video_layout.addLayout(video_button_row)

        hint = QLabel(
            "Example format: %Y_%m_%dT%H_%M_%S\n"
            "If the format misses, the scanner also tries several common datetime patterns and finally falls back to the file modified time."
        )
        hint.setWordWrap(True)
        video_layout.addWidget(hint)
        layout.addWidget(video_group)

        nav_group = QGroupBox("Navigation Input")
        nav_layout = QVBoxLayout(nav_group)
        self.add_navigation_button = QPushButton("Configure Navigation Sources")
        self.clear_navigation_button = QPushButton("Clear Navigation")
        self.navigation_summary = QTextEdit()
        self.navigation_summary.setReadOnly(True)
        self.navigation_summary.setMinimumHeight(130)
        nav_layout.addWidget(self.add_navigation_button)
        nav_layout.addWidget(self.clear_navigation_button)
        nav_layout.addWidget(self.navigation_summary)
        layout.addWidget(nav_group)

        sensor_group = QGroupBox("Sensor Inputs")
        sensor_layout = QVBoxLayout(sensor_group)
        self.add_sensor_button = QPushButton("Add Sensor")
        self.remove_sensor_button = QPushButton("Remove Selected Sensor")
        self.sensor_list = QListWidget()
        sensor_layout.addWidget(self.add_sensor_button)
        sensor_layout.addWidget(self.remove_sensor_button)
        sensor_layout.addWidget(self.sensor_list)
        layout.addWidget(sensor_group)

        interval_group = QGroupBox("Selected Time Intervals")
        interval_layout = QVBoxLayout(interval_group)
        self.interval_table = QTableWidget(0, 2)
        self.interval_table.setHorizontalHeaderLabels(["Start (ISO)", "End (ISO)"])
        self.interval_table.horizontalHeader().setStretchLastSection(True)
        self.add_interval_button = QPushButton("Add Interval From Coverage")
        self.clear_intervals_button = QPushButton("Clear Intervals")
        interval_layout.addWidget(self.interval_table)
        interval_button_row = QHBoxLayout()
        interval_button_row.addWidget(self.add_interval_button)
        interval_button_row.addWidget(self.clear_intervals_button)
        interval_layout.addLayout(interval_button_row)
        layout.addWidget(interval_group)

        process_group = QGroupBox("Processing")
        process_layout = QFormLayout(process_group)
        self.output_dir_edit = QLineEdit()
        self.browse_output_button = QPushButton("Browse...")
        out_row = QWidget()
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.addWidget(self.output_dir_edit)
        out_layout.addWidget(self.browse_output_button)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.01, 120.0)
        self.frame_rate_spin.setValue(1.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setSuffix(" Hz")

        self.frame_quality_combo = QComboBox()
        self.frame_quality_combo.addItems([
            "Original",
            "1080p",
            "720p",
            "480p",
            "360p",
        ])
        self.frame_quality_combo.setCurrentText("Original")

        self.run_metashape_check = QCheckBox("Run Metashape")
        self.generate_rasters_check = QCheckBox("Generate sensor TIFFs")
        self.generate_rasters_check.setChecked(True)
        self.annotate_frames_check = QCheckBox("Annotate extracted frames")

        self.metashape_exec_edit = QLineEdit()
        self.browse_metashape_button = QPushButton("Browse...")
        meta_row = QWidget()
        meta_layout = QHBoxLayout(meta_row)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.addWidget(self.metashape_exec_edit)
        meta_layout.addWidget(self.browse_metashape_button)

        process_layout.addRow("Output directory:", out_row)
        process_layout.addRow("Frame extraction rate:", self.frame_rate_spin)
        process_layout.addRow("Frame quality:", self.frame_quality_combo)
        process_layout.addRow(self.run_metashape_check)
        process_layout.addRow("Metashape executable:", meta_row)
        process_layout.addRow(self.generate_rasters_check)
        process_layout.addRow(self.annotate_frames_check)
        layout.addWidget(process_group)

        self.progress_label = QLabel("Idle.")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)

        self.subprogress_label = QLabel("Idle.")
        self.subprogress_label.setWordWrap(True)
        self.subprogress_bar = QProgressBar()
        self.subprogress_bar.setRange(0, 100)
        self.subprogress_bar.setValue(0)

        layout.addWidget(QLabel("Overall Progress"))
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress_bar)
        layout.addWidget(QLabel("Current Step Progress"))
        layout.addWidget(self.subprogress_label)
        layout.addWidget(self.subprogress_bar)

        self.run_button = QPushButton("Run Pipeline")
        self.save_workspace_button = QPushButton("Save Workspace")
        self.load_workspace_button = QPushButton("Load Workspace")
        self.save_button = QPushButton("Save Configuration")
        layout.addWidget(self.run_button)
        layout.addWidget(self.save_workspace_button)
        layout.addWidget(self.load_workspace_button)
        layout.addWidget(self.save_button)
        layout.addStretch(1)
        return panel

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        title = QLabel("Coverage Timeline")
        title.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(title)
        subtitle = QLabel(
            "Base row shows video coverage. Navigation coverage appears next. Each selected sensor channel gets its own row."
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        self.timeline_widget = TimelineWidget()
        layout.addWidget(self.timeline_widget, stretch=1)
        return panel

    def _build_summary_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)

        summary_group = QGroupBox("Project Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)
        layout.addWidget(summary_group)

        warning_group = QGroupBox("Skipped / Warnings")
        warning_layout = QVBoxLayout(warning_group)
        self.warning_text = QTextEdit()
        self.warning_text.setReadOnly(True)
        warning_layout.addWidget(self.warning_text)
        layout.addWidget(warning_group)

        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)
        return panel

    def _wire_signals(self) -> None:
        self.browse_video_button.clicked.connect(self._browse_video_directory)
        self.scan_videos_button.clicked.connect(self._scan_videos)
        self.add_navigation_button.clicked.connect(self._add_navigation_file)
        self.clear_navigation_button.clicked.connect(self._clear_navigation_file)
        self.add_sensor_button.clicked.connect(self._add_sensor_file)
        self.remove_sensor_button.clicked.connect(self._remove_selected_sensor)
        self.add_interval_button.clicked.connect(self._add_default_interval)
        self.clear_intervals_button.clicked.connect(self._clear_intervals)
        self.browse_output_button.clicked.connect(self._browse_output_directory)
        self.browse_metashape_button.clicked.connect(self._browse_metashape_executable)
        self.run_button.clicked.connect(self._run_pipeline)
        self.save_button.clicked.connect(self._save_configuration)
        self.save_workspace_button.clicked.connect(self._save_workspace)
        self.load_workspace_button.clicked.connect(self._load_workspace)

    def _browse_video_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if selected:
            self.video_dir_edit.setText(selected)

    def _browse_output_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if selected:
            self.output_dir_edit.setText(selected)

    def _browse_metashape_executable(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(self, "Select Metashape Executable")
        if selected:
            self.metashape_exec_edit.setText(selected)

    def _scan_videos(self) -> None:
        self.video_directory = self.video_dir_edit.text().strip()
        self.video_datetime_format = self.video_format_edit.text().strip()
        if not self.video_directory:
            QMessageBox.warning(self, "Missing directory", "Select a video directory first.")
            return

        try:
            service = VideoService(self.video_datetime_format)
            self.videos, self.skipped_videos = service.scan_directory(self.video_directory)
        except VideoScanError as exc:
            QMessageBox.critical(self, "Video scan failed", str(exc))
            return
        except Exception as exc:
            QMessageBox.critical(self, "Unexpected error", str(exc))
            return

        self._refresh_all_views()

    def _add_navigation_file(self) -> None:
        dialog = NavigationImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                self.navigation_file = result
                self._refresh_all_views()

    def _clear_navigation_file(self) -> None:
        self.navigation_file = None
        self._refresh_all_views()

    def _add_sensor_file(self) -> None:
        dialog = SensorImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                self.sensor_files.append(result)
                self._refresh_all_views()

    def _remove_selected_sensor(self) -> None:
        current_row = self.sensor_list.currentRow()
        if current_row < 0:
            return
        del self.sensor_files[current_row]
        self._refresh_all_views()

    def _add_default_interval(self) -> None:
        coverage = self._compute_combined_video_coverage()
        if coverage is None:
            QMessageBox.warning(self, "No video coverage", "Scan videos first before creating an interval.")
            return
        self.selected_intervals.append(SelectedTimeRange(start_time=coverage[0], end_time=coverage[1]))
        self._refresh_interval_table()
        self._refresh_summary()

    def _clear_intervals(self) -> None:
        self.selected_intervals.clear()
        self._refresh_interval_table()
        self._refresh_summary()

    def _build_pipeline_config(self) -> PipelineConfig:
        video_dir = self.video_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not video_dir:
            raise ValueError("Select a video directory.")
        if not output_dir:
            raise ValueError("Select an output directory.")
        if not self.videos:
            raise ValueError("Scan videos before running the pipeline.")

        return PipelineConfig(
            video_directory=Path(video_dir),
            output_directory=Path(output_dir),
            video_filename_time_format=self.video_format_edit.text().strip(),
            videos=self.videos,
            selected_intervals=self.selected_intervals,
            navigation_file=self.navigation_file,
            sensor_files=self.sensor_files,
            frame_rate=float(self.frame_rate_spin.value()),
            frame_quality=self.frame_quality_combo.currentText(),  # 👈 NEW
            run_metashape=self.run_metashape_check.isChecked(),
            metashape_exec=self.metashape_exec_edit.text().strip() or None,
            generate_sensor_rasters=self.generate_rasters_check.isChecked(),
            annotate_frames=self.annotate_frames_check.isChecked(),
        )
        

    def _run_pipeline(self) -> None:
        try:
            config = self._build_pipeline_config()
        except Exception as exc:
            QMessageBox.warning(self, "Cannot run pipeline", str(exc))
            return

        self.log_text.clear()
        self.progress_bar.setValue(0)
        self.progress_label.setText("Starting pipeline...")
        self.subprogress_bar.setValue(0)
        self.subprogress_label.setText("Waiting to start frame extraction...")
        self._set_processing_enabled(False)
        self.worker_thread = QThread(self)
        self.worker = PipelineWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self.progress_bar.setValue)
        self.worker.status.connect(self.progress_label.setText)
        self.worker.subprogress.connect(self.subprogress_bar.setValue)
        self.worker.substatus.connect(self.subprogress_label.setText)
        self.worker.error.connect(self._on_pipeline_error)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_pipeline_finished(self, output_dirs: list[str]) -> None:
        self._set_processing_enabled(True)
        self.progress_bar.setValue(100)
        self.progress_label.setText("Pipeline complete.")
        self.subprogress_bar.setValue(0)
        self.subprogress_label.setText("Idle.")
        self.log_text.append("Pipeline complete.")
        if output_dirs:
            QMessageBox.information(self, "Pipeline complete", f"Outputs written to:\n" + "\n".join(output_dirs))
        else:
            QMessageBox.warning(self, "Pipeline complete", "The pipeline finished, but no output folders were generated.")

    def _on_pipeline_error(self, message: str) -> None:
        self._set_processing_enabled(True)
        self.progress_label.setText("Pipeline failed.")
        self.subprogress_bar.setValue(0)
        self.subprogress_label.setText("Idle.")
        self.log_text.append(f"Pipeline failed: {message}")
        QMessageBox.critical(self, "Pipeline failed", message)

    def _set_processing_enabled(self, enabled: bool) -> None:
        self.run_button.setEnabled(enabled)
        self.scan_videos_button.setEnabled(enabled)
        self.save_button.setEnabled(enabled)
        self.save_workspace_button.setEnabled(enabled)
        self.load_workspace_button.setEnabled(enabled)

    def _save_configuration(self) -> None:
        if not self.videos:
            QMessageBox.warning(self, "No videos", "Scan videos before saving a configuration.")
            return
        output_path, _ = QFileDialog.getSaveFileName(self, "Save Configuration JSON", filter="JSON Files (*.json)")
        if not output_path:
            return
        try:
            ConfigService.save_json(
                output_path=output_path,
                video_directory=self.video_dir_edit.text().strip(),
                video_filename_time_format=self.video_format_edit.text().strip(),
                videos=self.videos,
                sensor_files=self.sensor_files,
                selected_intervals=self.selected_intervals,
                navigation_file=self.navigation_file,
                output_directory=self.output_dir_edit.text().strip() or None,
                frame_rate=float(self.frame_rate_spin.value()),
                run_metashape=self.run_metashape_check.isChecked(),
                generate_sensor_rasters=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                metashape_exec=self.metashape_exec_edit.text().strip() or None,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Save failed", str(exc))
            return
        QMessageBox.information(self, "Saved", f"Configuration saved to:\n{output_path}")

    def _save_workspace(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "Save Workspace", filter="JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            ConfigService.save_workspace(
                path=path,
                video_directory=self.video_dir_edit.text().strip(),
                filename_datetime_format=self.video_format_edit.text().strip(),
                navigation_file=self.navigation_file,
                sensor_files=self.sensor_files,
                selected_intervals=self.selected_intervals,
                output_directory=self.output_dir_edit.text().strip(),
                frame_rate=float(self.frame_rate_spin.value()),
                run_metashape=self.run_metashape_check.isChecked(),
                generate_sensor_tiffs=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                metashape_executable=self.metashape_exec_edit.text().strip(),
                frame_quality=self.frame_quality_combo.currentText(),
            )
            self.log_text.append("Workspace saved.")
        except Exception as exc:
            QMessageBox.critical(self, "Workspace save failed", str(exc))

    def _load_workspace(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Workspace", filter="JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            data = ConfigService.load_workspace(path)
            self.video_dir_edit.setText(data["video_directory"])
            self.video_format_edit.setText(data["filename_datetime_format"])
            self.navigation_file = data["navigation_file"]
            self.sensor_files = data["sensor_files"]
            self.selected_intervals = data["selected_intervals"]
            self.output_dir_edit.setText(data["output_directory"])
            self.frame_rate_spin.setValue(data["frame_rate"])
            self.run_metashape_check.setChecked(data["run_metashape"])
            self.generate_rasters_check.setChecked(data["generate_sensor_tiffs"])
            self.annotate_frames_check.setChecked(data["annotate_frames"])
            self.metashape_exec_edit.setText(data["metashape_executable"])
            quality = data.get("frame_quality", "Original")
            idx = self.frame_quality_combo.findText(quality)
            self.frame_quality_combo.setCurrentIndex(idx if idx >= 0 else 0)
            self.videos = []
            self.skipped_videos = []
            if self.video_dir_edit.text().strip():
                try:
                    service = VideoService(self.video_format_edit.text().strip())
                    self.videos, self.skipped_videos = service.scan_directory(self.video_dir_edit.text().strip())
                except Exception as exc:
                    self.log_text.append(f"Video rescan after workspace load failed: {exc}")

            self._refresh_all_views()
            self.log_text.append("Workspace loaded.")
        except Exception as exc:
            QMessageBox.critical(self, "Workspace load failed", str(exc))

    def _append_log(self, message: str) -> None:
        self.log_text.append(message)

    def _refresh_all_views(self) -> None:
        self._refresh_sensor_list()
        self._refresh_navigation_summary()
        self._refresh_interval_table()
        self._refresh_timeline()
        self._refresh_summary()
        self._refresh_warnings()

    def _refresh_sensor_list(self) -> None:
        self.sensor_list.clear()
        for sensor in self.sensor_files:
            channel_summary = ", ".join(channel.display_name for channel in sensor.channels)
            coverage = ""
            if sensor.start_time and sensor.end_time:
                coverage = f" [{sensor.start_time.isoformat()} -> {sensor.end_time.isoformat()}]"
            self.sensor_list.addItem(f"{Path(sensor.csv_path).name}: {channel_summary}{coverage}")

    def _refresh_navigation_summary(self) -> None:
        if self.navigation_file is None:
            self.navigation_summary.setPlainText("No navigation sources configured.")
            return

        def source_lines(label: str, source: TimeValueSourceConfig | None) -> list[str]:
            if source is None:
                return [f"{label}: <None>"]
            lines = [
                f"{label}:",
                f"  File: {source.csv_path}",
                f"  Timestamp: {source.timestamp_column}",
                f"  Value: {source.value_column}",
            ]
            if source.start_time and source.end_time:
                lines.append(f"  Coverage: {source.start_time.isoformat()} -> {source.end_time.isoformat()}")
            return lines

        lines: list[str] = []
        lines.extend(source_lines("Latitude source", self.navigation_file.latitude_source))
        lines.append("")
        lines.extend(source_lines("Longitude source", self.navigation_file.longitude_source))
        lines.append("")
        lines.extend(source_lines("Altitude source", self.navigation_file.altitude_source))
        if self.navigation_file.start_time and self.navigation_file.end_time:
            lines.append("")
            lines.append(
                f"Overall navigation coverage: {self.navigation_file.start_time.isoformat()} -> {self.navigation_file.end_time.isoformat()}"
            )
        self.navigation_summary.setPlainText("\n".join(lines))

    def _refresh_interval_table(self) -> None:
        self.interval_table.setRowCount(len(self.selected_intervals))
        for row, interval in enumerate(self.selected_intervals):
            self.interval_table.setItem(row, 0, self._table_item(interval.start_time.isoformat()))
            self.interval_table.setItem(row, 1, self._table_item(interval.end_time.isoformat()))
        self.interval_table.resizeColumnsToContents()

    def _refresh_timeline(self) -> None:
        timeline_items: list[SensorFileConfig] = []
        if self.navigation_file:
            nav_sources = [
                ("Navigation - Latitude", self.navigation_file.latitude_source),
                ("Navigation - Longitude", self.navigation_file.longitude_source),
                ("Navigation - Altitude", self.navigation_file.altitude_source),
            ]
            for label, source in nav_sources:
                if source is None:
                    continue
                timeline_items.append(
                    SensorFileConfig(
                        csv_path=source.csv_path,
                        timestamp_column=source.timestamp_column,
                        channels=[],
                        start_time=source.start_time,
                        end_time=source.end_time,
                    )
                )
        timeline_items.extend(self.sensor_files)
        self.timeline_widget.set_data(self.videos, timeline_items)

    def _refresh_summary(self) -> None:
        lines = [
            f"Videos: {len(self.videos)}",
            f"Navigation loaded: {'Yes' if self.navigation_file else 'No'}",
            f"Sensor CSVs: {len(self.sensor_files)}",
            f"Selected intervals: {len(self.selected_intervals)}",
            f"Output directory: {self.output_dir_edit.text().strip() or '<not set>'}",
            f"Frame quality: {self.frame_quality_combo.currentText()}",
            f"Run Metashape: {'Yes' if self.run_metashape_check.isChecked() else 'No'}",
            f"Generate sensor TIFFs: {'Yes' if self.generate_rasters_check.isChecked() else 'No'}",
            f"Annotate frames: {'Yes' if self.annotate_frames_check.isChecked() else 'No'}",
            f"Progress: {self.progress_bar.value()}%",
            "",
        ]
        if self.videos:
            lines.append("Video coverage:")
            for video in self.videos:
                lines.append(
                    f"- {video.filename}: {video.start_time.isoformat()} -> {video.end_time.isoformat()} ({video.duration_s:.2f}s, time source={video.time_source})"
                )
        self.summary_text.setPlainText("\n".join(lines))

    def _refresh_warnings(self) -> None:
        warnings = []
        if self.skipped_videos:
            warnings.append("Skipped videos:")
            warnings.extend(f"- {name}" for name in self.skipped_videos)
        if not warnings:
            warnings.append("No warnings.")
        self.warning_text.setPlainText("\n".join(warnings))

    def _compute_combined_video_coverage(self):
        if not self.videos:
            return None
        start = min(video.start_time for video in self.videos)
        end = max(video.end_time for video in self.videos)
        return start, end

    @staticmethod
    def _table_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item
