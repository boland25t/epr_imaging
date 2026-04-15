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
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QToolBar,
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
        self.setWindowTitle("EPR Video + Sensor Processing Tool")
        self.setMinimumSize(1000, 650)

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
        self._reset_progress()
        self._refresh_all_views()

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        self._build_toolbar()
        self._build_status_bar()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        self.controls_tabs = QTabWidget()
        self.controls_tabs.addTab(self._build_inputs_tab(), "Inputs")
        self.controls_tabs.addTab(self._build_processing_tab(), "Processing")
        self.controls_tabs.setMinimumWidth(340)
        self.controls_tabs.setMaximumWidth(480)

        splitter.addWidget(self.controls_tabs)
        splitter.addWidget(self._build_center_panel())
        splitter.addWidget(self._build_summary_panel())

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([400, 800, 400])

    def _build_toolbar(self) -> None:
        tb = QToolBar("Main Toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextOnly)
        tb.setStyleSheet("QToolBar { spacing: 6px; padding: 4px; }")
        self.addToolBar(tb)

        self.run_action = tb.addAction("▶  Run Pipeline", self._run_pipeline)
        self.run_action.setToolTip("Extract frames and run the processing pipeline")

        tb.addSeparator()

        self.load_workspace_action = tb.addAction("Load Workspace", self._load_workspace)
        self.save_workspace_action = tb.addAction("Save Workspace", self._save_workspace)

        tb.addSeparator()

        self.save_config_action = tb.addAction("Save Config JSON", self._save_configuration)
        self.save_config_action.setToolTip("Export a standalone pipeline configuration file")

    def _build_status_bar(self) -> None:
        sb = self.statusBar()
        sb.setSizeGripEnabled(True)

        self._status_label = QLabel("Idle.")
        self._status_label.setMinimumWidth(200)

        separator = QLabel(" | ")
        separator.setStyleSheet("color: gray;")

        self._substatus_label = QLabel("")
        self._substatus_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        self._subprogress_bar = QProgressBar()
        self._subprogress_bar.setRange(0, 100)
        self._subprogress_bar.setFixedWidth(150)
        self._subprogress_bar.setFixedHeight(16)
        self._subprogress_bar.setFormat("Step: %p%")

        self._progress_bar = QProgressBar()
        self._progress_bar.setRange(0, 100)
        self._progress_bar.setFixedWidth(150)
        self._progress_bar.setFixedHeight(16)
        self._progress_bar.setFormat("Overall: %p%")

        sb.addWidget(self._status_label)
        sb.addWidget(separator)
        sb.addWidget(self._substatus_label, 1)
        sb.addPermanentWidget(self._subprogress_bar)
        sb.addPermanentWidget(self._progress_bar)

    def _reset_progress(self) -> None:
        self._progress_bar.setValue(0)
        self._status_label.setText("Idle.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("")

    def _build_inputs_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        tab_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # Video Inputs
        video_group = QGroupBox("Video Inputs")
        video_layout = QVBoxLayout(video_group)
        video_form = QFormLayout()
        self.video_dir_edit = QLineEdit()
        self.video_format_edit = QLineEdit(self.video_datetime_format)
        video_form.addRow("Directory:", self.video_dir_edit)
        video_form.addRow("Filename format:", self.video_format_edit)
        video_layout.addLayout(video_form)
        video_btn_row = QHBoxLayout()
        self.browse_video_button = QPushButton("Browse…")
        self.scan_videos_button = QPushButton("Scan Videos")
        video_btn_row.addWidget(self.browse_video_button)
        video_btn_row.addWidget(self.scan_videos_button)
        video_layout.addLayout(video_btn_row)
        hint = QLabel("e.g. %Y_%m_%dT%H_%M_%S  —  Falls back through common patterns, then file modified time.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        video_layout.addWidget(hint)
        layout.addWidget(video_group)

        # Navigation
        nav_group = QGroupBox("Navigation Input")
        nav_layout = QVBoxLayout(nav_group)
        nav_btn_row = QHBoxLayout()
        self.add_navigation_button = QPushButton("Configure Sources…")
        self.clear_navigation_button = QPushButton("Clear")
        self.clear_navigation_button.setMaximumWidth(60)
        nav_btn_row.addWidget(self.add_navigation_button)
        nav_btn_row.addWidget(self.clear_navigation_button)
        nav_layout.addLayout(nav_btn_row)
        self.navigation_summary = QTextEdit()
        self.navigation_summary.setReadOnly(True)
        self.navigation_summary.setMinimumHeight(60)
        self.navigation_summary.setMaximumHeight(110)
        nav_layout.addWidget(self.navigation_summary)
        layout.addWidget(nav_group)

        # Sensor Inputs
        sensor_group = QGroupBox("Sensor Inputs")
        sensor_layout = QVBoxLayout(sensor_group)
        sensor_btn_row = QHBoxLayout()
        self.add_sensor_button = QPushButton("Add Sensor…")
        self.remove_sensor_button = QPushButton("Remove Selected")
        sensor_btn_row.addWidget(self.add_sensor_button)
        sensor_btn_row.addWidget(self.remove_sensor_button)
        sensor_layout.addLayout(sensor_btn_row)
        self.sensor_list = QListWidget()
        self.sensor_list.setMinimumHeight(60)
        self.sensor_list.setMaximumHeight(120)
        sensor_layout.addWidget(self.sensor_list)
        layout.addWidget(sensor_group)

        # Time Intervals
        interval_group = QGroupBox("Selected Time Intervals")
        interval_layout = QVBoxLayout(interval_group)
        self.interval_table = QTableWidget(0, 2)
        self.interval_table.setHorizontalHeaderLabels(["Start (ISO)", "End (ISO)"])
        self.interval_table.horizontalHeader().setStretchLastSection(True)
        self.interval_table.setMinimumHeight(60)
        self.interval_table.setMaximumHeight(120)
        self.interval_table.verticalHeader().setVisible(False)
        interval_btn_row = QHBoxLayout()
        self.add_interval_button = QPushButton("Add from Video Coverage")
        self.clear_intervals_button = QPushButton("Clear")
        self.clear_intervals_button.setMaximumWidth(60)
        interval_btn_row.addWidget(self.add_interval_button)
        interval_btn_row.addWidget(self.clear_intervals_button)
        interval_layout.addWidget(self.interval_table)
        interval_layout.addLayout(interval_btn_row)
        layout.addWidget(interval_group)

        layout.addStretch()
        return tab

    def _build_processing_tab(self) -> QWidget:
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        tab_layout.addWidget(scroll)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # Output
        output_group = QGroupBox("Output")
        output_form = QFormLayout(output_group)
        self.output_dir_edit = QLineEdit()
        self.browse_output_button = QPushButton("Browse…")
        out_row = QWidget()
        out_layout = QHBoxLayout(out_row)
        out_layout.setContentsMargins(0, 0, 0, 0)
        out_layout.addWidget(self.output_dir_edit)
        out_layout.addWidget(self.browse_output_button)
        output_form.addRow("Directory:", out_row)
        layout.addWidget(output_group)

        # Frame extraction
        frame_group = QGroupBox("Frame Extraction")
        frame_form = QFormLayout(frame_group)
        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.01, 120.0)
        self.frame_rate_spin.setValue(1.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setSuffix(" Hz")
        self.frame_quality_combo = QComboBox()
        self.frame_quality_combo.addItems(["Original", "1080p", "720p", "480p", "360p"])
        self.frame_quality_combo.setCurrentText("Original")
        frame_form.addRow("Frame rate:", self.frame_rate_spin)
        frame_form.addRow("Quality:", self.frame_quality_combo)
        layout.addWidget(frame_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        self.generate_rasters_check = QCheckBox("Generate sensor GeoTIFFs")
        self.generate_rasters_check.setChecked(True)
        self.annotate_frames_check = QCheckBox("Annotate extracted frames")
        options_layout.addWidget(self.generate_rasters_check)
        options_layout.addWidget(self.annotate_frames_check)
        layout.addWidget(options_group)

        # Metashape
        metashape_group = QGroupBox("Metashape")
        metashape_layout = QVBoxLayout(metashape_group)
        self.run_metashape_check = QCheckBox("Run Metashape after extraction")
        metashape_form = QFormLayout()
        self.metashape_exec_edit = QLineEdit()
        self.browse_metashape_button = QPushButton("Browse…")
        meta_row = QWidget()
        meta_layout = QHBoxLayout(meta_row)
        meta_layout.setContentsMargins(0, 0, 0, 0)
        meta_layout.addWidget(self.metashape_exec_edit)
        meta_layout.addWidget(self.browse_metashape_button)
        metashape_form.addRow("Executable:", meta_row)
        metashape_layout.addWidget(self.run_metashape_check)
        metashape_layout.addLayout(metashape_form)
        layout.addWidget(metashape_group)

        layout.addStretch()
        return tab

    def _build_center_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Coverage Timeline")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        subtitle = QLabel("Blue = video  ·  Yellow = navigation  ·  Green = sensor channels")
        subtitle.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.timeline_widget = TimelineWidget()
        layout.addWidget(self.timeline_widget, stretch=1)
        return panel

    def _build_summary_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(280)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        summary_group = QGroupBox("Project Summary")
        summary_layout = QVBoxLayout(summary_group)
        self.summary_text = QTextEdit()
        self.summary_text.setReadOnly(True)
        summary_layout.addWidget(self.summary_text)

        warning_group = QGroupBox("Skipped / Warnings")
        warning_layout = QVBoxLayout(warning_group)
        self.warning_text = QTextEdit()
        self.warning_text.setReadOnly(True)
        warning_layout.addWidget(self.warning_text)

        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)

        layout.addWidget(summary_group, stretch=2)
        layout.addWidget(warning_group, stretch=1)
        layout.addWidget(log_group, stretch=3)
        return panel

    # ------------------------------------------------------------------
    # Signals
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # User actions
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Pipeline
    # ------------------------------------------------------------------

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
            frame_quality=self.frame_quality_combo.currentText(),
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
        self._progress_bar.setValue(0)
        self._status_label.setText("Starting pipeline…")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("Waiting to start frame extraction…")
        self._set_processing_enabled(False)

        self.worker_thread = QThread(self)
        self.worker = PipelineWorker(config)
        self.worker.moveToThread(self.worker_thread)
        self.worker_thread.started.connect(self.worker.run)
        self.worker.log.connect(self._append_log)
        self.worker.progress.connect(self._progress_bar.setValue)
        self.worker.status.connect(self._status_label.setText)
        self.worker.subprogress.connect(self._subprogress_bar.setValue)
        self.worker.substatus.connect(self._substatus_label.setText)
        self.worker.error.connect(self._on_pipeline_error)
        self.worker.finished.connect(self._on_pipeline_finished)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _on_pipeline_finished(self, output_dirs: list[str]) -> None:
        self._set_processing_enabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText("Pipeline complete.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("Idle.")
        self.log_text.append("Pipeline complete.")
        if output_dirs:
            QMessageBox.information(self, "Pipeline complete", "Outputs written to:\n" + "\n".join(output_dirs))
        else:
            QMessageBox.warning(self, "Pipeline complete", "The pipeline finished but no output folders were generated.")

    def _on_pipeline_error(self, message: str) -> None:
        self._set_processing_enabled(True)
        self._status_label.setText("Pipeline failed.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("")
        self.log_text.append(f"Pipeline failed: {message}")
        QMessageBox.critical(self, "Pipeline failed", message)

    def _set_processing_enabled(self, enabled: bool) -> None:
        self.run_action.setEnabled(enabled)
        self.scan_videos_button.setEnabled(enabled)
        self.save_config_action.setEnabled(enabled)
        self.save_workspace_action.setEnabled(enabled)
        self.load_workspace_action.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

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
            idx = self.frame_quality_combo.findText(data.get("frame_quality", "Original"))
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

    # ------------------------------------------------------------------
    # View refresh
    # ------------------------------------------------------------------

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
                coverage = (
                    f" [{sensor.start_time.strftime('%Y-%m-%d %H:%M')} → "
                    f"{sensor.end_time.strftime('%H:%M')}]"
                )
            self.sensor_list.addItem(f"{Path(sensor.csv_path).name}: {channel_summary}{coverage}")

    def _refresh_navigation_summary(self) -> None:
        if self.navigation_file is None:
            self.navigation_summary.setPlainText("No navigation sources configured.")
            return

        def source_line(label: str, source: TimeValueSourceConfig | None) -> str:
            if source is None:
                return f"{label}: <not set>"
            coverage = ""
            if source.start_time and source.end_time:
                coverage = (
                    f"  [{source.start_time.strftime('%Y-%m-%d %H:%M')} → "
                    f"{source.end_time.strftime('%H:%M')}]"
                )
            return f"{label}: {source.csv_path.name}  [{source.value_column}]{coverage}"

        lines = [
            source_line("Lat", self.navigation_file.latitude_source),
            source_line("Lon", self.navigation_file.longitude_source),
            source_line("Alt", self.navigation_file.altitude_source),
        ]
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
            for label, source in [
                ("NAV: Latitude", self.navigation_file.latitude_source),
                ("NAV: Longitude", self.navigation_file.longitude_source),
                ("NAV: Altitude", self.navigation_file.altitude_source),
            ]:
                if source is None:
                    continue
                timeline_items.append(SensorFileConfig(
                    csv_path=source.csv_path,
                    timestamp_column=source.timestamp_column,
                    channels=[],
                    start_time=source.start_time,
                    end_time=source.end_time,
                ))
        timeline_items.extend(self.sensor_files)
        self.timeline_widget.set_data(self.videos, timeline_items)

    def _refresh_summary(self) -> None:
        lines = [
            f"Videos: {len(self.videos)}",
            f"Navigation: {'configured' if self.navigation_file else 'not set'}",
            f"Sensor CSVs: {len(self.sensor_files)}",
            f"Intervals: {len(self.selected_intervals)}",
            f"Output: {self.output_dir_edit.text().strip() or '<not set>'}",
            f"Frame rate: {self.frame_rate_spin.value():.2f} Hz",
            f"Quality: {self.frame_quality_combo.currentText()}",
            f"Metashape: {'enabled' if self.run_metashape_check.isChecked() else 'off'}",
            f"Sensor TIFFs: {'yes' if self.generate_rasters_check.isChecked() else 'no'}",
            f"Annotate frames: {'yes' if self.annotate_frames_check.isChecked() else 'no'}",
            "",
        ]
        if self.videos:
            lines.append("Video coverage:")
            for video in self.videos:
                lines.append(
                    f"  {video.filename}: "
                    f"{video.start_time.strftime('%Y-%m-%d %H:%M:%S')} → "
                    f"{video.end_time.strftime('%H:%M:%S')}  "
                    f"({video.duration_s:.0f}s, src={video.time_source})"
                )
        self.summary_text.setPlainText("\n".join(lines))

    def _refresh_warnings(self) -> None:
        if self.skipped_videos:
            warnings = ["Skipped videos:"] + [f"  {name}" for name in self.skipped_videos]
            self.warning_text.setPlainText("\n".join(warnings))
        else:
            self.warning_text.setPlainText("No warnings.")

    def _compute_combined_video_coverage(self):
        if not self.videos:
            return None
        return min(v.start_time for v in self.videos), max(v.end_time for v in self.videos)

    @staticmethod
    def _table_item(text: str) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item
