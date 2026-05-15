from __future__ import annotations

import math
from pathlib import Path

import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, Signal, Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSpinBox,
    QStackedWidget,
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
from sensor_service import SensorService
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
        self.depth_source: SensorFileConfig | None = None
        self.speed_source: SensorFileConfig | None = None
        self.selected_intervals: list[SelectedTimeRange] = []
        self.altitude_threshold: float | None = None
        self.depth_threshold: float | None = None
        self.speed_threshold: float | None = None
        self.sensor_thresholds: dict[str, tuple[float | None, float | None]] = {}
        self.min_segment_frames: int = 1
        self.applied_steps: list[str] = []
        self._current_run_steps: list[str] = []
        self._sensor_only_run: bool = False
        self.worker_thread: QThread | None = None
        self.worker: PipelineWorker | None = None
        self.workspace_saved: bool = False
        self.workspace_path: str = ""
        self.sensor_threshold_widgets: dict[str, tuple[QCheckBox, QDoubleSpinBox, QCheckBox, QDoubleSpinBox]] = {}

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
        self.controls_tabs.addTab(self._build_postprocessing_tab(), "Post-Processing")
        self.controls_tabs.setMinimumWidth(340)
        self.controls_tabs.setMaximumWidth(480)

        splitter.addWidget(self.controls_tabs)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self._build_timeline_panel())
        self.center_stack.addWidget(self._build_postprocessing_graph_panel())
        splitter.addWidget(self.center_stack)
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
        self.clear_workspace_action = tb.addAction("Clear Workspace", self._clear_workspace)
        self.clear_workspace_action.setToolTip("Save the current workspace then reset all fields")

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

        # Depth / Speed Inputs
        depth_speed_group = QGroupBox("Depth / Speed Sources")
        depth_speed_layout = QVBoxLayout(depth_speed_group)
        depth_speed_layout.setContentsMargins(8, 8, 8, 8)

        depth_btn_row = QHBoxLayout()
        self.add_depth_button = QPushButton("Select Depth Source…")
        self.clear_depth_button = QPushButton("Clear")
        self.clear_depth_button.setMaximumWidth(60)
        depth_btn_row.addWidget(self.add_depth_button)
        depth_btn_row.addWidget(self.clear_depth_button)
        depth_speed_layout.addLayout(depth_btn_row)
        self.depth_summary = QLabel("No depth source configured.")
        self.depth_summary.setWordWrap(True)
        self.depth_summary.setStyleSheet("color: gray; font-size: 11px;")
        depth_speed_layout.addWidget(self.depth_summary)

        speed_btn_row = QHBoxLayout()
        self.add_speed_button = QPushButton("Select Speed Source…")
        self.clear_speed_button = QPushButton("Clear")
        self.clear_speed_button.setMaximumWidth(60)
        speed_btn_row.addWidget(self.add_speed_button)
        speed_btn_row.addWidget(self.clear_speed_button)
        depth_speed_layout.addLayout(speed_btn_row)
        self.speed_summary = QLabel("No speed source configured.")
        self.speed_summary.setWordWrap(True)
        self.speed_summary.setStyleSheet("color: gray; font-size: 11px;")
        depth_speed_layout.addWidget(self.speed_summary)

        layout.addWidget(depth_speed_group)

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

        self.sampling_mode_combo = QComboBox()
        self.sampling_mode_combo.addItems(["Fixed rate", "Dynamic spacing"])
        frame_form.addRow("Mode:", self.sampling_mode_combo)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.01, 120.0)
        self.frame_rate_spin.setValue(1.0)
        self.frame_rate_spin.setDecimals(2)
        self.frame_rate_spin.setSuffix(" Hz")
        frame_form.addRow("Frame rate:", self.frame_rate_spin)

        self.dynamic_spacing_spin = QDoubleSpinBox()
        self.dynamic_spacing_spin.setRange(0.1, 10000.0)
        self.dynamic_spacing_spin.setValue(2.0)
        self.dynamic_spacing_spin.setDecimals(2)
        self.dynamic_spacing_spin.setSuffix(" m")
        self.dynamic_spacing_spin.setEnabled(False)
        self.dynamic_spacing_spin.setToolTip("Target ground distance between sampled frames. Requires navigation data.")
        frame_form.addRow("Target spacing:", self.dynamic_spacing_spin)

        self.dynamic_min_freq_spin = QDoubleSpinBox()
        self.dynamic_min_freq_spin.setRange(0.001, 10.0)
        self.dynamic_min_freq_spin.setValue(0.1)
        self.dynamic_min_freq_spin.setDecimals(3)
        self.dynamic_min_freq_spin.setSuffix(" Hz")
        self.dynamic_min_freq_spin.setEnabled(False)
        self.dynamic_min_freq_spin.setToolTip("Minimum sampling frequency — guarantees a frame even when the vehicle is slow or stationary.")
        frame_form.addRow("Min frequency:", self.dynamic_min_freq_spin)

        self.frame_quality_combo = QComboBox()
        self.frame_quality_combo.addItems(["Original", "1080p", "720p", "480p", "360p"])
        self.frame_quality_combo.setCurrentText("Original")
        frame_form.addRow("Quality:", self.frame_quality_combo)
        layout.addWidget(frame_group)

        # Options
        options_group = QGroupBox("Options")
        options_layout = QVBoxLayout(options_group)
        self.sample_images_check = QCheckBox("Sample images")
        self.sample_images_check.setChecked(True)
        self.sample_images_check.setToolTip(
            "When unchecked, interpolates all sensor channels at the sample rate without extracting any images. "
            "Use this for large video sets to preview sensor data and set thresholds, then re-run with sampling enabled."
        )
        self.generate_rasters_check = QCheckBox("Generate sensor GeoTIFFs")
        self.generate_rasters_check.setChecked(True)
        self.annotate_frames_check = QCheckBox("Annotate extracted frames")
        options_layout.addWidget(self.sample_images_check)
        options_layout.addWidget(self.generate_rasters_check)
        options_layout.addWidget(self.annotate_frames_check)
        layout.addWidget(options_group)

        layout.addStretch()
        return tab

    def _build_postprocessing_tab(self) -> QWidget:
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

        info = QLabel(
            "Run individual processing steps after frame extraction. "
            "Set thresholds below then re-run with 'Sample images' enabled to produce per-range orthophoto inputs."
        )
        info.setWordWrap(True)
        info.setStyleSheet("color: gray; font-size: 11px;")
        layout.addWidget(info)

        steps_group = QGroupBox("Processing Steps")
        steps_layout = QVBoxLayout(steps_group)
        self.postprocess_extract_check = QCheckBox("Extract frames")
        self.postprocess_generate_rasters_check = QCheckBox("Generate sensor GeoTIFFs")
        self.postprocess_annotate_check = QCheckBox("Annotate extracted frames")
        self.postprocess_clahe_check = QCheckBox("Apply CLAHE enhancement")
        self.postprocess_update_master_check = QCheckBox("Update master CSV with newly added sources")
        self.postprocess_extract_check.setChecked(True)
        self.postprocess_generate_rasters_check.setChecked(True)
        self.postprocess_annotate_check.setChecked(False)
        self.postprocess_clahe_check.setChecked(False)
        self.postprocess_update_master_check.setChecked(True)
        steps_layout.addWidget(self.postprocess_extract_check)
        steps_layout.addWidget(self.postprocess_generate_rasters_check)
        steps_layout.addWidget(self.postprocess_annotate_check)
        steps_layout.addWidget(self.postprocess_clahe_check)
        steps_layout.addWidget(self.postprocess_update_master_check)

        clahe_group = QGroupBox("CLAHE Parameters")
        clahe_form = QFormLayout(clahe_group)
        self.clahe_clip_limit_spin = QDoubleSpinBox()
        self.clahe_clip_limit_spin.setRange(0.1, 40.0)
        self.clahe_clip_limit_spin.setValue(2.0)
        self.clahe_clip_limit_spin.setDecimals(1)
        self.clahe_clip_limit_spin.setSingleStep(0.5)
        self.clahe_clip_limit_spin.setToolTip("Threshold for contrast limiting. Higher values allow more contrast enhancement.")
        self.clahe_tile_size_spin = QSpinBox()
        self.clahe_tile_size_spin.setRange(1, 64)
        self.clahe_tile_size_spin.setValue(8)
        self.clahe_tile_size_spin.setSuffix(" px")
        self.clahe_tile_size_spin.setToolTip("Size of each tile in the grid (applied as N×N). Smaller tiles produce more localized enhancement.")
        clahe_form.addRow("Contrast Limit:", self.clahe_clip_limit_spin)
        clahe_form.addRow("Tile Grid Size:", self.clahe_tile_size_spin)
        steps_layout.addWidget(clahe_group)

        layout.addWidget(steps_group)

        thresholds_group = QGroupBox("Threshold Settings (0 = disabled)")
        thresholds_layout = QFormLayout(thresholds_group)
        self.altitude_threshold_spin = QDoubleSpinBox()
        self.altitude_threshold_spin.setRange(-1e6, 1e6)
        self.altitude_threshold_spin.setDecimals(3)
        self.altitude_threshold_spin.setValue(0.0)
        self.altitude_threshold_spin.setSuffix(" m")
        self.altitude_threshold_spin.setToolTip("Maximum altitude — only frames at or below this altitude are kept. 0 = disabled.")

        self.depth_threshold_spin = QDoubleSpinBox()
        self.depth_threshold_spin.setRange(-1e6, 1e6)
        self.depth_threshold_spin.setDecimals(3)
        self.depth_threshold_spin.setValue(0.0)
        self.depth_threshold_spin.setSuffix(" m")
        self.depth_threshold_spin.setToolTip("Minimum depth. 0 = disabled.")

        self.speed_threshold_spin = QDoubleSpinBox()
        self.speed_threshold_spin.setRange(0, 1e6)
        self.speed_threshold_spin.setDecimals(3)
        self.speed_threshold_spin.setValue(0.0)
        self.speed_threshold_spin.setSuffix(" m/s")
        self.speed_threshold_spin.setToolTip("Minimum speed. 0 = disabled.")

        self.min_segment_spin = QSpinBox()
        self.min_segment_spin.setRange(1, 10000)
        self.min_segment_spin.setValue(1)
        self.min_segment_spin.setToolTip("Minimum frames in a contiguous range to produce a segment.")

        thresholds_layout.addRow("Max altitude:", self.altitude_threshold_spin)
        thresholds_layout.addRow("Min depth:", self.depth_threshold_spin)
        thresholds_layout.addRow("Min speed:", self.speed_threshold_spin)
        thresholds_layout.addRow("Min segment frames:", self.min_segment_spin)
        layout.addWidget(thresholds_group)

        self.sensor_thresholds_group = QGroupBox("Sensor Thresholds")
        self.sensor_thresholds_layout = QFormLayout(self.sensor_thresholds_group)
        hint = QLabel("Sensor thresholds appear here once sensors are loaded. Enable a checkbox to activate that bound.")
        hint.setWordWrap(True)
        hint.setStyleSheet("color: gray; font-size: 10px;")
        self.sensor_thresholds_layout.addRow(hint)
        layout.addWidget(self.sensor_thresholds_group)

        self.postprocess_run_button = QPushButton("Run Selected Steps")
        self.postprocess_full_button = QPushButton("Run Full Sequence")
        layout.addWidget(self.postprocess_run_button)
        layout.addWidget(self.postprocess_full_button)

        layout.addStretch()
        return tab

    def _build_timeline_panel(self) -> QWidget:
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

    def _build_postprocessing_graph_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        title = QLabel("Post-Processing Graphs")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        subtitle = QLabel("Sensor data trends — set thresholds on the Post-Processing tab then re-run with sampling")
        subtitle.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        graphs_scroll = QScrollArea()
        graphs_scroll.setWidgetResizable(True)
        graphs_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.graphs_container = QWidget()
        self.graphs_layout = QVBoxLayout(self.graphs_container)
        self.graphs_layout.setContentsMargins(0, 0, 0, 0)
        self.graphs_layout.setSpacing(4)
        graphs_scroll.setWidget(self.graphs_container)
        layout.addWidget(graphs_scroll, stretch=1)
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
        self.add_depth_button.clicked.connect(self._add_depth_source)
        self.clear_depth_button.clicked.connect(self._clear_depth_source)
        self.add_speed_button.clicked.connect(self._add_speed_source)
        self.clear_speed_button.clicked.connect(self._clear_speed_source)
        self.add_interval_button.clicked.connect(self._add_default_interval)
        self.clear_intervals_button.clicked.connect(self._clear_intervals)
        self.browse_output_button.clicked.connect(self._browse_output_directory)
        self.postprocess_run_button.clicked.connect(self._run_selected_postprocessing)
        self.postprocess_full_button.clicked.connect(self._run_full_postprocessing)
        self.controls_tabs.currentChanged.connect(self._update_center_panel)
        self.altitude_threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.depth_threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.speed_threshold_spin.valueChanged.connect(self._on_threshold_changed)
        self.sampling_mode_combo.currentIndexChanged.connect(self._refresh_sampling_mode_ui)
        self.postprocess_clahe_check.toggled.connect(self.clahe_clip_limit_spin.setEnabled)
        self.postprocess_clahe_check.toggled.connect(self.clahe_tile_size_spin.setEnabled)
        self.postprocess_clahe_check.toggled.emit(self.postprocess_clahe_check.isChecked())

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

    def _add_depth_source(self) -> None:
        dialog = SensorImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                if result.channels:
                    result.channels[0].display_name = "Depth"
                self.depth_source = result
                self._refresh_all_views()

    def _clear_depth_source(self) -> None:
        self.depth_source = None
        self._refresh_all_views()

    def _add_speed_source(self) -> None:
        dialog = SensorImportDialog(self)
        dialog.setWindowTitle("Select Speed Source")
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                if result.channels:
                    result.channels[0].display_name = "Speed"
                self.speed_source = result
                self._refresh_all_views()

    def _clear_speed_source(self) -> None:
        self.speed_source = None
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

    def _build_pipeline_config(self, selected_steps: list[str] | None = None) -> PipelineConfig:
        video_dir = self.video_dir_edit.text().strip()
        output_dir = self.output_dir_edit.text().strip()
        if not video_dir:
            raise ValueError("Select a video directory.")
        if not output_dir:
            raise ValueError("Select an output directory.")
        if not self.videos:
            raise ValueError("Scan videos before running the pipeline.")

        self.altitude_threshold = float(self.altitude_threshold_spin.value())
        self.depth_threshold = float(self.depth_threshold_spin.value())
        self.speed_threshold = float(self.speed_threshold_spin.value())
        self.min_segment_frames = int(self.min_segment_spin.value())
        self.sensor_thresholds = self._collect_sensor_thresholds()

        sampling_mode = "dynamic" if self.sampling_mode_combo.currentText() == "Dynamic spacing" else "fixed"
        if sampling_mode == "dynamic" and self.navigation_file is None:
            raise ValueError("Dynamic spacing requires navigation data. Configure navigation sources on the Inputs tab first.")

        config = PipelineConfig(
            video_directory=Path(video_dir),
            output_directory=Path(output_dir),
            video_filename_time_format=self.video_format_edit.text().strip(),
            videos=self.videos,
            selected_intervals=self.selected_intervals,
            navigation_file=self.navigation_file,
            sensor_files=self.sensor_files,
            depth_source=self.depth_source,
            speed_source=self.speed_source,
            altitude_threshold=self.altitude_threshold,
            depth_threshold=self.depth_threshold,
            speed_threshold=self.speed_threshold,
            sensor_thresholds=self.sensor_thresholds,
            min_segment_frames=self.min_segment_frames,
            frame_rate=float(self.frame_rate_spin.value()),
            sampling_mode=sampling_mode,
            dynamic_target_spacing_m=float(self.dynamic_spacing_spin.value()),
            dynamic_min_frequency_hz=float(self.dynamic_min_freq_spin.value()),
            frame_quality=self.frame_quality_combo.currentText(),
            sample_images=self.sample_images_check.isChecked(),
            generate_sensor_rasters=self.generate_rasters_check.isChecked(),
            annotate_frames=self.annotate_frames_check.isChecked(),
            clahe_clip_limit=float(self.clahe_clip_limit_spin.value()),
            clahe_tile_grid_size=int(self.clahe_tile_size_spin.value()),
        )
        if selected_steps is not None:
            config.selected_steps = selected_steps
        return config

    def _run_pipeline(self, checked: bool = False, selected_steps: list[str] | None = None) -> None:
        try:
            config = self._build_pipeline_config(selected_steps)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot run pipeline", str(exc))
            return

        # If re-extraction would hit existing frames, require a new workspace
        if config.sample_images and "extract_frames" in config.selected_steps:
            if self._frames_exist_for_config(config):
                if not self._handle_reextraction_workspace():
                    return
                try:
                    config = self._build_pipeline_config(selected_steps)
                except Exception as exc:
                    QMessageBox.warning(self, "Cannot run pipeline", str(exc))
                    return

        # Show frame count preview and let user confirm before starting
        if config.sample_images and "extract_frames" in config.selected_steps:
            if not self._show_extraction_preview(config):
                return

        # Auto-save workspace if not saved
        if not self.workspace_saved:
            self._auto_save_workspace(config.output_directory)

        self._current_run_steps = config.selected_steps
        self._sensor_only_run = not config.sample_images or "interpolate_only" in config.selected_steps
        self.log_text.clear()
        self._progress_bar.setValue(0)
        self._status_label.setText("Starting pipeline…")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("Waiting…")
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

    def _run_selected_postprocessing(self) -> None:
        selected_steps: list[str] = []
        if self.postprocess_extract_check.isChecked():
            selected_steps.append("extract_frames")
        if self.postprocess_generate_rasters_check.isChecked():
            selected_steps.append("generate_sensor_rasters")
        if self.postprocess_annotate_check.isChecked():
            selected_steps.append("annotate_frames")
        if self.postprocess_clahe_check.isChecked():
            selected_steps.append("apply_clahe")
        if self.postprocess_update_master_check.isChecked():
            selected_steps.append("update_master")
        if not selected_steps:
            QMessageBox.warning(self, "No steps selected", "Select at least one post-processing step to run.")
            return
        self._run_pipeline(selected_steps=selected_steps)

    def _run_full_postprocessing(self) -> None:
        self.postprocess_extract_check.setChecked(True)
        self.postprocess_generate_rasters_check.setChecked(True)
        self.postprocess_annotate_check.setChecked(True)
        self.postprocess_clahe_check.setChecked(True)
        self._run_pipeline(selected_steps=["extract_frames", "generate_sensor_rasters", "annotate_frames", "apply_clahe"])

    def _compute_frame_preview(self, config: PipelineConfig) -> list[dict]:
        intervals = list(config.selected_intervals)
        if not intervals and config.videos:
            intervals = [SelectedTimeRange(
                start_time=min(v.start_time for v in config.videos),
                end_time=max(v.end_time for v in config.videos),
            )]
        if not intervals:
            return []

        nav_sources: dict = {}
        if config.sampling_mode == "dynamic" and config.navigation_file:
            try:
                nav_sources["lat"] = SensorService.load_time_value_dataframe(config.navigation_file.latitude_source)
                nav_sources["lon"] = SensorService.load_time_value_dataframe(config.navigation_file.longitude_source)
                if config.navigation_file.altitude_source:
                    nav_sources["alt"] = SensorService.load_time_value_dataframe(config.navigation_file.altitude_source)
            except Exception:
                nav_sources = {}

        svc = PipelineService()
        results = []
        for idx, interval in enumerate(intervals):
            duration_s = (interval.end_time - interval.start_time).total_seconds()
            if config.sampling_mode == "dynamic" and nav_sources:
                times = svc._get_dynamic_sample_times(nav_sources, interval, config)
                count = len(times)
                mode_label = f"Dynamic spacing — {config.dynamic_target_spacing_m:.2f} m, f_min = {config.dynamic_min_frequency_hz:.3f} Hz"
            else:
                count = max(1, int(math.floor(duration_s * config.frame_rate)) + 1)
                mode_label = f"Fixed rate — {config.frame_rate:.2f} Hz"
            results.append({
                "idx": idx + 1,
                "start": interval.start_time,
                "end": interval.end_time,
                "duration_s": duration_s,
                "frame_count": count,
                "mode": mode_label,
            })
        return results

    def _show_extraction_preview(self, config: PipelineConfig) -> bool:
        preview = self._compute_frame_preview(config)
        if not preview:
            return True

        total = sum(p["frame_count"] for p in preview)

        lines = []
        for p in preview:
            h, rem = divmod(int(p["duration_s"]), 3600)
            m, s = divmod(rem, 60)
            dur_str = f"{h}h {m:02d}m {s:02d}s" if h else f"{m}m {s:02d}s"
            lines.append(
                f"Interval {p['idx']}:  "
                f"{p['start'].strftime('%Y-%m-%d %H:%M:%S')}  →  "
                f"{p['end'].strftime('%H:%M:%S')}  ({dur_str})"
            )
            lines.append(f"  Mode:    {p['mode']}")
            lines.append(f"  Frames:  {p['frame_count']:,}")
            lines.append("")
        lines.append(f"Total:  {total:,} frame(s) across {len(preview)} interval(s)")

        dialog = QDialog(self)
        dialog.setWindowTitle("Confirm Frame Extraction")
        dialog.setMinimumWidth(480)
        layout = QVBoxLayout(dialog)

        header = QLabel("Review the extraction plan below and confirm to proceed.")
        header.setWordWrap(True)
        layout.addWidget(header)

        preview_text = QTextEdit()
        preview_text.setReadOnly(True)
        preview_text.setPlainText("\n".join(lines))
        preview_text.setMinimumHeight(140)
        preview_text.setMaximumHeight(280)
        layout.addWidget(preview_text)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.button(QDialogButtonBox.Ok).setText("Proceed")
        buttons.button(QDialogButtonBox.Cancel).setText("Cancel")
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)

        return dialog.exec() == QDialog.Accepted

    def _frames_exist_for_config(self, config: PipelineConfig) -> bool:
        if not config.output_directory.exists():
            return False
        if config.selected_intervals:
            for idx, interval in enumerate(config.selected_intervals):
                segment_name = (
                    f"segment_{idx + 1:03d}_"
                    f"{interval.start_time.strftime('%Y%m%dT%H%M%S')}_"
                    f"{interval.end_time.strftime('%Y%m%dT%H%M%S')}"
                )
                frames_dir = config.output_directory / segment_name / "frames"
                if frames_dir.exists() and any(frames_dir.glob("*.jpg")):
                    return True
        else:
            for d in config.output_directory.iterdir():
                if d.is_dir() and d.name.startswith("segment_"):
                    if (d / "frames").exists() and any((d / "frames").glob("*.jpg")):
                        return True
        return False

    def _handle_reextraction_workspace(self) -> bool:
        msg = QMessageBox(self)
        msg.setWindowTitle("Frames Already Exist")
        msg.setIcon(QMessageBox.Warning)
        msg.setText("Extracted frames already exist in the current output directory.")
        msg.setInformativeText(
            "Re-extraction requires a new workspace and output directory.\n\n"
            "Your video, navigation, and sensor sources will be carried over. "
            "Applied post-processing steps will not follow.\n\n"
            "Click OK to choose a new output directory and save a new workspace, "
            "or Cancel to abort."
        )
        msg.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        msg.setDefaultButton(QMessageBox.Cancel)
        if msg.exec() != QMessageBox.Ok:
            return False

        new_output = QFileDialog.getExistingDirectory(self, "Select New Output Directory")
        if not new_output:
            return False
        self.output_dir_edit.setText(new_output)
        self.applied_steps = []

        path, _ = QFileDialog.getSaveFileName(
            self, "Save New Workspace", filter="JSON Files (*.json)"
        )
        if not path:
            return False

        try:
            ConfigService.save_workspace(
                path=path,
                video_directory=self.video_dir_edit.text().strip(),
                filename_datetime_format=self.video_format_edit.text().strip(),
                navigation_file=self.navigation_file,
                sensor_files=self.sensor_files,
                selected_intervals=self.selected_intervals,
                output_directory=new_output,
                frame_rate=float(self.frame_rate_spin.value()),
                generate_sensor_tiffs=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                frame_quality=self.frame_quality_combo.currentText(),
                depth_source=self.depth_source,
                speed_source=self.speed_source,
                altitude_threshold=float(self.altitude_threshold_spin.value()),
                depth_threshold=float(self.depth_threshold_spin.value()),
                speed_threshold=float(self.speed_threshold_spin.value()),
                min_segment_frames=int(self.min_segment_spin.value()),
                applied_steps=[],
                sampling_mode="dynamic" if self.sampling_mode_combo.currentText() == "Dynamic spacing" else "fixed",
                dynamic_target_spacing_m=float(self.dynamic_spacing_spin.value()),
                dynamic_min_frequency_hz=float(self.dynamic_min_freq_spin.value()),
                clahe_clip_limit=float(self.clahe_clip_limit_spin.value()),
                clahe_tile_grid_size=int(self.clahe_tile_size_spin.value()),
            )
            self.workspace_saved = True
            self.workspace_path = path
            self.log_text.append(f"New workspace saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Workspace save failed", str(exc))
            return False

        return True

    def _on_pipeline_finished(self, output_dirs: list[str]) -> None:
        self._set_processing_enabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText("Pipeline complete.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("Idle.")
        self.log_text.append("Pipeline complete.")
        self.applied_steps = list(dict.fromkeys(self.applied_steps + self._current_run_steps))
        self._refresh_summary()

        if self._sensor_only_run:
            # Switch to Post-Processing tab and show graphs so user can set thresholds
            pp_idx = next(
                (i for i in range(self.controls_tabs.count()) if self.controls_tabs.tabText(i) == "Post-Processing"),
                -1,
            )
            if pp_idx >= 0:
                self.controls_tabs.setCurrentIndex(pp_idx)
            msg = (
                "Sensor interpolation complete. Review the graphs, set thresholds on the "
                "Post-Processing tab, then re-run with 'Sample images' enabled to extract frames "
                "and produce one orthophoto segment per continuous valid range."
            )
            QMessageBox.information(self, "Sensor-only run complete", msg)
        elif output_dirs:
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
                generate_sensor_rasters=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                depth_source=self.depth_source,
                speed_source=self.speed_source,
                altitude_threshold=float(self.altitude_threshold_spin.value()),
                depth_threshold=float(self.depth_threshold_spin.value()),
                speed_threshold=float(self.speed_threshold_spin.value()),
                min_segment_frames=int(self.min_segment_spin.value()),
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
                generate_sensor_tiffs=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                frame_quality=self.frame_quality_combo.currentText(),
                depth_source=self.depth_source,
                speed_source=self.speed_source,
                altitude_threshold=float(self.altitude_threshold_spin.value()),
                depth_threshold=float(self.depth_threshold_spin.value()),
                speed_threshold=float(self.speed_threshold_spin.value()),
                min_segment_frames=int(self.min_segment_spin.value()),
                applied_steps=self.applied_steps,
                sampling_mode="dynamic" if self.sampling_mode_combo.currentText() == "Dynamic spacing" else "fixed",
                dynamic_target_spacing_m=float(self.dynamic_spacing_spin.value()),
                dynamic_min_frequency_hz=float(self.dynamic_min_freq_spin.value()),
                clahe_clip_limit=float(self.clahe_clip_limit_spin.value()),
                clahe_tile_grid_size=int(self.clahe_tile_size_spin.value()),
            )
            self.log_text.append("Workspace saved.")
            self.workspace_saved = True
            self.workspace_path = path
        except Exception as exc:
            QMessageBox.critical(self, "Workspace save failed", str(exc))

    def _clear_workspace(self) -> None:
        self._save_workspace()
        if not self.workspace_saved:
            return  # User cancelled the save dialog

        self.video_directory = ""
        self.video_datetime_format = "%Y%m%d_%H%M%S"
        self.videos = []
        self.skipped_videos = []
        self.sensor_files = []
        self.navigation_file = None
        self.depth_source = None
        self.speed_source = None
        self.selected_intervals = []
        self.altitude_threshold = None
        self.depth_threshold = None
        self.speed_threshold = None
        self.sensor_thresholds = {}
        self.min_segment_frames = 1
        self.applied_steps = []
        self.workspace_saved = False
        self.workspace_path = ""

        self.video_dir_edit.setText("")
        self.video_format_edit.setText(self.video_datetime_format)
        self.output_dir_edit.setText("")
        self.frame_rate_spin.setValue(1.0)
        self.sampling_mode_combo.setCurrentIndex(0)
        self.dynamic_spacing_spin.setValue(2.0)
        self.dynamic_min_freq_spin.setValue(0.1)
        self.frame_quality_combo.setCurrentIndex(0)
        self.sample_images_check.setChecked(True)
        self.generate_rasters_check.setChecked(True)
        self.annotate_frames_check.setChecked(False)
        self.altitude_threshold_spin.setValue(0.0)
        self.depth_threshold_spin.setValue(0.0)
        self.speed_threshold_spin.setValue(0.0)
        self.min_segment_spin.setValue(1)
        self.postprocess_clahe_check.setChecked(False)
        self.clahe_clip_limit_spin.setValue(2.0)
        self.clahe_tile_size_spin.setValue(8)
        self.log_text.clear()

        self._reset_progress()
        self._refresh_all_views()

    def _load_workspace(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Load Workspace", filter="JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            data = ConfigService.load_workspace(path)
            self.video_dir_edit.setText(data["video_directory"])
            self.video_format_edit.setText(data["filename_datetime_format"])
            self.navigation_file = data["navigation_file"]
            self.depth_source = data.get("depth_source")
            self.speed_source = data.get("speed_source")
            self.sensor_files = data["sensor_files"]
            self.selected_intervals = data["selected_intervals"]
            self.output_dir_edit.setText(data["output_directory"])
            self.frame_rate_spin.setValue(data["frame_rate"])
            self.generate_rasters_check.setChecked(data["generate_sensor_tiffs"])
            self.annotate_frames_check.setChecked(data["annotate_frames"])
            self.altitude_threshold = data.get("altitude_threshold")
            self.depth_threshold = data.get("depth_threshold")
            self.speed_threshold = data.get("speed_threshold")
            self.min_segment_frames = int(data.get("min_segment_frames", 1))
            self.applied_steps = data.get("applied_steps", [])
            self.altitude_threshold_spin.setValue(self.altitude_threshold if self.altitude_threshold is not None else 0.0)
            self.depth_threshold_spin.setValue(self.depth_threshold if self.depth_threshold is not None else 0.0)
            self.speed_threshold_spin.setValue(self.speed_threshold if self.speed_threshold is not None else 0.0)
            self.min_segment_spin.setValue(self.min_segment_frames)
            idx = self.frame_quality_combo.findText(data.get("frame_quality", "Original"))
            self.frame_quality_combo.setCurrentIndex(idx if idx >= 0 else 0)
            mode_text = "Dynamic spacing" if data.get("sampling_mode", "fixed") == "dynamic" else "Fixed rate"
            self.sampling_mode_combo.setCurrentIndex(self.sampling_mode_combo.findText(mode_text))
            self.dynamic_spacing_spin.setValue(float(data.get("dynamic_target_spacing_m", 2.0)))
            self.dynamic_min_freq_spin.setValue(float(data.get("dynamic_min_frequency_hz", 0.1)))
            self.clahe_clip_limit_spin.setValue(float(data.get("clahe_clip_limit", 2.0)))
            self.clahe_tile_size_spin.setValue(int(data.get("clahe_tile_grid_size", 8)))
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
            self.workspace_saved = True
            self.workspace_path = path
        except Exception as exc:
            QMessageBox.critical(self, "Workspace load failed", str(exc))

    def _auto_save_workspace(self, output_directory: Path) -> None:
        auto_path = output_directory / "auto_saved_workspace.json"
        try:
            ConfigService.save_workspace(
                path=str(auto_path),
                video_directory=self.video_dir_edit.text().strip(),
                filename_datetime_format=self.video_format_edit.text().strip(),
                navigation_file=self.navigation_file,
                sensor_files=self.sensor_files,
                selected_intervals=self.selected_intervals,
                output_directory=self.output_dir_edit.text().strip(),
                frame_rate=float(self.frame_rate_spin.value()),
                generate_sensor_tiffs=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                frame_quality=self.frame_quality_combo.currentText(),
                depth_source=self.depth_source,
                speed_source=self.speed_source,
                altitude_threshold=float(self.altitude_threshold_spin.value()),
                depth_threshold=float(self.depth_threshold_spin.value()),
                speed_threshold=float(self.speed_threshold_spin.value()),
                min_segment_frames=int(self.min_segment_spin.value()),
                applied_steps=self.applied_steps,
                sampling_mode="dynamic" if self.sampling_mode_combo.currentText() == "Dynamic spacing" else "fixed",
                dynamic_target_spacing_m=float(self.dynamic_spacing_spin.value()),
                dynamic_min_frequency_hz=float(self.dynamic_min_freq_spin.value()),
                clahe_clip_limit=float(self.clahe_clip_limit_spin.value()),
                clahe_tile_grid_size=int(self.clahe_tile_size_spin.value()),
            )
            self.workspace_saved = True
            self.workspace_path = str(auto_path)
            self.log_text.append(f"Auto-saved workspace to {auto_path}")
        except Exception as exc:
            self.log_text.append(f"Auto-save workspace failed: {exc}")
            # Don't block processing for this

    def _append_log(self, message: str) -> None:
        self.log_text.append(message)

    def _refresh_all_views(self) -> None:
        self._refresh_sensor_list()
        self._refresh_navigation_summary()
        self._refresh_depth_summary()
        self._refresh_speed_summary()
        self._refresh_interval_table()
        self._refresh_timeline()
        self._refresh_sensor_thresholds_ui()
        self._refresh_sampling_mode_ui()
        self._update_center_panel()
        self._refresh_summary()
        self._refresh_warnings()

    def _refresh_sampling_mode_ui(self) -> None:
        is_dynamic = self.sampling_mode_combo.currentText() == "Dynamic spacing"
        nav_ok = self.navigation_file is not None
        self.frame_rate_spin.setEnabled(not is_dynamic)
        self.dynamic_spacing_spin.setEnabled(is_dynamic)
        self.dynamic_min_freq_spin.setEnabled(is_dynamic)
        if is_dynamic and not nav_ok:
            warn = "Navigation data must be configured before running dynamic sampling."
            self.dynamic_spacing_spin.setToolTip(warn)
            self.dynamic_min_freq_spin.setToolTip(warn)
        else:
            self.dynamic_spacing_spin.setToolTip("Target ground distance between sampled frames.")
            self.dynamic_min_freq_spin.setToolTip("Minimum sampling frequency — guarantees a frame even when the vehicle is slow or stationary.")

    def _on_threshold_changed(self) -> None:
        if self.center_stack.currentIndex() == 1:
            self._refresh_postprocessing_graphs()

    def _update_center_panel(self, index: int | None = None) -> None:
        if index is None:
            index = self.controls_tabs.currentIndex()
        if self.controls_tabs.tabText(index) == "Post-Processing":
            self.center_stack.setCurrentIndex(1)
            self._refresh_postprocessing_graphs()
        else:
            self.center_stack.setCurrentIndex(0)

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

    def _refresh_depth_summary(self) -> None:
        if self.depth_source is None:
            self.depth_summary.setText("No depth source configured.")
            return
        channel = self.depth_source.channels[0] if self.depth_source.channels else None
        if channel:
            coverage = ""
            if self.depth_source.start_time and self.depth_source.end_time:
                coverage = (
                    f"  [{self.depth_source.start_time.strftime('%Y-%m-%d %H:%M')} → "
                    f"{self.depth_source.end_time.strftime('%H:%M')}]"
                )
            self.depth_summary.setText(f"{Path(self.depth_source.csv_path).name}: {channel.display_name}  [{channel.source_column}]{coverage}")
        else:
            self.depth_summary.setText("Depth source configured but no channels.")

    def _refresh_speed_summary(self) -> None:
        if self.speed_source is None:
            self.speed_summary.setText("No speed source configured.")
            return
        channel = self.speed_source.channels[0] if self.speed_source.channels else None
        if channel:
            coverage = ""
            if self.speed_source.start_time and self.speed_source.end_time:
                coverage = (
                    f"  [{self.speed_source.start_time.strftime('%Y-%m-%d %H:%M')} → "
                    f"{self.speed_source.end_time.strftime('%H:%M')}]"
                )
            self.speed_summary.setText(f"{Path(self.speed_source.csv_path).name}: {channel.display_name}  [{channel.source_column}]{coverage}")
        else:
            self.speed_summary.setText("Speed source configured but no channels.")

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
        if self.depth_source:
            timeline_items.append(self.depth_source)
        if self.speed_source:
            timeline_items.append(self.speed_source)
        self.timeline_widget.set_data(self.videos, timeline_items)

    def _refresh_summary(self) -> None:
        lines = [
            f"Videos: {len(self.videos)}",
            f"Navigation: {'configured' if self.navigation_file else 'not set'}",
            f"Depth source: {'configured' if self.depth_source else 'not set'}",
            f"Speed source: {'configured' if self.speed_source else 'not set'}",
            f"Sensor CSVs: {len(self.sensor_files)}",
            f"Intervals: {len(self.selected_intervals)}",
            f"Output: {self.output_dir_edit.text().strip() or '<not set>'}",
            f"Sampling: {self.sampling_mode_combo.currentText()}",
            f"Frame rate: {self.frame_rate_spin.value():.2f} Hz" if self.sampling_mode_combo.currentText() == "Fixed rate" else f"Target spacing: {self.dynamic_spacing_spin.value():.2f} m  |  f_min: {self.dynamic_min_freq_spin.value():.3f} Hz",
            f"Quality: {self.frame_quality_combo.currentText()}",
            f"Sensor TIFFs: {'yes' if self.generate_rasters_check.isChecked() else 'no'}",
            f"Annotate frames: {'yes' if self.annotate_frames_check.isChecked() else 'no'}",
            f"Applied steps: {', '.join(self.applied_steps) if self.applied_steps else 'none'}",
            f"Altitude threshold: {self.altitude_threshold if self.altitude_threshold is not None else 'none'}",
            f"Depth threshold: {self.depth_threshold if self.depth_threshold is not None else 'none'}",
            f"Speed threshold: {self.speed_threshold if self.speed_threshold is not None else 'none'}",
            f"Min segment frames: {self.min_segment_frames}",
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

    def _refresh_postprocessing_graphs(self) -> None:
        while self.graphs_layout.count():
            item = self.graphs_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        alt_thresh = float(self.altitude_threshold_spin.value())
        depth_thresh = float(self.depth_threshold_spin.value())
        speed_thresh = float(self.speed_threshold_spin.value())

        if self.navigation_file and self.navigation_file.altitude_source:
            try:
                df = SensorService.load_time_value_dataframe(self.navigation_file.altitude_source)
                plot = self._create_timeseries_plot("Altitude (m)")
                self._plot_series(plot, df["unix_time"].tolist(), df["value"].tolist(), color="#1f77b4")
                if alt_thresh != 0.0:
                    plot.addItem(pg.InfiniteLine(pos=alt_thresh, angle=0, pen=pg.mkPen("#e63946", width=1, style=Qt.DashLine), label=f"max {alt_thresh:.3g}", labelOpts={"color": "#e63946"}))
                self.graphs_layout.addWidget(plot)
            except Exception:
                pass

        if self.depth_source and self.depth_source.channels:
            try:
                df = SensorService.load_sensor_dataframe(self.depth_source)
                col = self.depth_source.channels[0].source_column
                plot = self._create_timeseries_plot("Depth (m)")
                self._plot_series(plot, df["unix_time"].tolist(), df[col].astype(float).tolist(), color="#2ca02c")
                if depth_thresh != 0.0:
                    plot.addItem(pg.InfiniteLine(pos=depth_thresh, angle=0, pen=pg.mkPen("#e63946", width=1, style=Qt.DashLine), label=f"min {depth_thresh:.3g}", labelOpts={"color": "#e63946"}))
                self.graphs_layout.addWidget(plot)
            except Exception:
                pass

        if self.speed_source and self.speed_source.channels:
            try:
                df = SensorService.load_sensor_dataframe(self.speed_source)
                col = self.speed_source.channels[0].source_column
                plot = self._create_timeseries_plot("Speed (m/s)")
                self._plot_series(plot, df["unix_time"].tolist(), df[col].astype(float).tolist(), color="#ff7f0e")
                if speed_thresh != 0.0:
                    plot.addItem(pg.InfiniteLine(pos=speed_thresh, angle=0, pen=pg.mkPen("#e63946", width=1, style=Qt.DashLine), label=f"min {speed_thresh:.3g}", labelOpts={"color": "#e63946"}))
                self.graphs_layout.addWidget(plot)
            except Exception:
                pass

        colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]
        color_idx = 0
        for sensor in self.sensor_files:
            try:
                df = SensorService.load_sensor_dataframe(sensor)
                for channel in sensor.channels:
                    name = channel.display_name or channel.source_column
                    plot = self._create_timeseries_plot(name)
                    self._plot_series(plot, df["unix_time"].tolist(), df[channel.source_column].astype(float).tolist(), color=colors[color_idx % len(colors)])
                    if name in self.sensor_threshold_widgets:
                        min_en, min_spin, max_en, max_spin = self.sensor_threshold_widgets[name]
                        if min_en.isChecked():
                            plot.addItem(pg.InfiniteLine(pos=min_spin.value(), angle=0, pen=pg.mkPen("#e63946", width=1, style=Qt.DashLine), label=f"min {min_spin.value():.3g}", labelOpts={"color": "#e63946"}))
                        if max_en.isChecked():
                            plot.addItem(pg.InfiniteLine(pos=max_spin.value(), angle=0, pen=pg.mkPen("#e67e22", width=1, style=Qt.DashLine), label=f"max {max_spin.value():.3g}", labelOpts={"color": "#e67e22"}))
                    self.graphs_layout.addWidget(plot)
                    color_idx += 1
            except Exception:
                pass

        if self.graphs_layout.count() == 0:
            placeholder = QLabel("No sensor data loaded. Add navigation or sensor sources on the Inputs tab.")
            placeholder.setWordWrap(True)
            placeholder.setStyleSheet("color: gray; font-size: 12px; padding: 20px;")
            self.graphs_layout.addWidget(placeholder)

    def _refresh_sensor_thresholds_ui(self) -> None:
        # Collect all sensor channel display names (excluding depth/speed which have built-in thresholds)
        channel_names: list[str] = []
        for sensor in self.sensor_files:
            for channel in sensor.channels:
                name = channel.display_name or channel.source_column
                if name not in channel_names:
                    channel_names.append(name)

        # Preserve existing threshold values keyed by name
        preserved: dict[str, tuple[bool, float, bool, float]] = {}
        for name, (min_en, min_spin, max_en, max_spin) in self.sensor_threshold_widgets.items():
            preserved[name] = (min_en.isChecked(), min_spin.value(), max_en.isChecked(), max_spin.value())

        # Clear layout
        while self.sensor_thresholds_layout.count():
            item = self.sensor_thresholds_layout.takeAt(0)
            w = item.widget()
            if w:
                w.deleteLater()

        self.sensor_threshold_widgets.clear()

        if not channel_names:
            hint = QLabel("Sensor thresholds appear here once sensors are loaded.")
            hint.setWordWrap(True)
            hint.setStyleSheet("color: gray; font-size: 10px;")
            self.sensor_thresholds_layout.addRow(hint)
            return

        for name in channel_names:
            prev = preserved.get(name, (False, 0.0, False, 0.0))

            min_en = QCheckBox("Min:")
            min_en.setChecked(prev[0])
            min_spin = QDoubleSpinBox()
            min_spin.setRange(-1e9, 1e9)
            min_spin.setDecimals(4)
            min_spin.setValue(prev[1])
            min_spin.setEnabled(prev[0])
            min_en.toggled.connect(min_spin.setEnabled)

            max_en = QCheckBox("Max:")
            max_en.setChecked(prev[2])
            max_spin = QDoubleSpinBox()
            max_spin.setRange(-1e9, 1e9)
            max_spin.setDecimals(4)
            max_spin.setValue(prev[3])
            max_spin.setEnabled(prev[2])
            max_en.toggled.connect(max_spin.setEnabled)

            row = QWidget()
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(0, 0, 0, 0)
            row_layout.addWidget(min_en)
            row_layout.addWidget(min_spin)
            row_layout.addSpacing(8)
            row_layout.addWidget(max_en)
            row_layout.addWidget(max_spin)

            self.sensor_thresholds_layout.addRow(name + ":", row)
            self.sensor_threshold_widgets[name] = (min_en, min_spin, max_en, max_spin)

    def _collect_sensor_thresholds(self) -> dict[str, tuple[float | None, float | None]]:
        result: dict[str, tuple[float | None, float | None]] = {}
        for name, (min_en, min_spin, max_en, max_spin) in self.sensor_threshold_widgets.items():
            min_val = min_spin.value() if min_en.isChecked() else None
            max_val = max_spin.value() if max_en.isChecked() else None
            if min_val is not None or max_val is not None:
                result[name] = (min_val, max_val)
        return result

    def _create_timeseries_plot(self, title: str) -> pg.PlotWidget:
        axis = pg.DateAxisItem(orientation="bottom")
        plot = pg.PlotWidget(axisItems={"bottom": axis})
        plot.setBackground("w")
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setLabel("left", title)
        plot.setLabel("bottom", "Time")
        plot.setTitle(title)
        plot.setMinimumHeight(220)
        plot.setMaximumHeight(320)
        return plot

    @staticmethod
    def _plot_series(plot: pg.PlotWidget, data_x: list[float], data_y: list[float], color: str = "#0077cc") -> None:
        plot.clear()
        if not data_x or not data_y:
            text_item = pg.TextItem("No data available", color="gray", anchor=(0, 0))
            plot.addItem(text_item)
            text_item.setPos(0, 0)
            return
        plot.plot(data_x, data_y, pen=pg.mkPen(color=color, width=2))
        plot.setXRange(min(data_x), max(data_x), padding=0.02)

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
