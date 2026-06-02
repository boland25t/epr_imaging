# main_window.py — Top-level GUI application window for the EPR Imaging tool
#
# Responsibilities:
#   1. Build and manage the entire application UI: four left-panel tabs
#      (Inputs, Processing, Post-Processing, Visualization) plus a central
#      display area (Timeline / Graphs / Map) and a right-side summary panel.
#   2. React to user interactions on the Inputs tab: scan video directories,
#      configure navigation and sensor CSV sources, add/remove time intervals.
#   3. Run the extraction pipeline in a background QThread via PipelineWorker,
#      routing progress/status/log callbacks back to the GUI thread via signals.
#   4. Load and save workspace JSON files (via ConfigService) so the user's
#      configuration persists across sessions.
#   5. Drive the Visualization tab: load master.csv data, display the GPS
#      trackline and frame scatter on MapWidget, manage named segments, support
#      two-click interval picking directly on the map.
#   6. Drive the Post-Processing tab: display live sensor timeseries graphs with
#      threshold overlays; manage per-channel min/max threshold widgets.
#
# Threading model:
#   The pipeline runs in a QThread; PipelineWorker emits Qt signals from that
#   thread.  All signal handlers in MainWindow execute on the main (GUI) thread
#   via Qt's automatic cross-thread signal delivery, making the callbacks safe
#   to use for QWidget updates without additional locking.
#
# State management:
#   All session state (videos, sensor configs, intervals, thresholds, …) lives
#   as instance attributes on MainWindow.  _refresh_all_views() rebuilds every
#   dependent widget from that state in one call so partial refreshes can't leave
#   the UI inconsistent.

from __future__ import annotations

import calendar   # calendar.timegm() for UTC datetime → Unix timestamp without local-tz offset
import math       # math.floor() used in frame count estimates
import shutil     # shutil.copy2() used when exporting segment frames
from datetime import datetime  # datetime.utcfromtimestamp() for Unix → UTC wall clock
from pathlib import Path       # Cross-platform path handling throughout

import numpy as np
import pandas as pd
import pyqtgraph as pg
from PySide6.QtCore import QObject, QThread, QTimer, Signal, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QScrollArea,
    QSlider,
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
from models import AnnotationConfig, Job, NavigationConfig, SegmentRecord, SelectedTimeRange, SensorFileConfig, TimeValueSourceConfig, VideoRecord
from pipeline_service import PipelineConfig, PipelineService
from sensor_service import SensorService
from video_service import VideoScanError, VideoService
from widgets.annotation_settings_dialog import AnnotationSettingsDialog
from widgets.map_widget import MapWidget, SEGMENT_COLORS, haversine_km
from widgets.navigation_import_dialog import NavigationImportDialog
from widgets.sensor_import_dialog import SensorImportDialog
from widgets.timeline_widget import TimelineWidget


# ===========================================================================
# PipelineWorker — thin QObject wrapper that runs PipelineService in a thread
# ===========================================================================

class PipelineWorker(QObject):
    """Runs PipelineService.run() in a background QThread and relays results via signals.

    MainWindow creates one PipelineWorker per pipeline invocation, moves it to a
    fresh QThread, and connects its signals to GUI-updating slots.  Because all
    signals cross the thread boundary, Qt delivers them on the main thread
    automatically — no manual locking is required.

    Signals:
        finished(list[str])  — emitted with string paths of produced directories on success
        error(str)           — emitted with the exception message on failure
        log(str)             — individual log line for the log text view
        progress(int)        — 0–100 for the top-level progress bar
        status(str)          — human-readable step description for the status label
        subprogress(int)     — 0–100 for the per-video sub-progress bar
        substatus(str)       — detail text for the sub-status label
    """

    finished           = Signal(list)    # list[str] — string paths of output directories
    error              = Signal(str)     # exception message string
    log                = Signal(str)     # one log line
    progress           = Signal(int)     # 0–100 overall progress
    status             = Signal(str)     # current step description
    subprogress        = Signal(int)     # 0–100 per-video sub-progress
    substatus          = Signal(str)     # current filename / frame detail
    segment_completed  = Signal(object)  # SegmentRecord emitted after each segment

    def __init__(self, config: PipelineConfig):
        """Store the config and create the service with the log signal as its sink."""
        super().__init__()
        self.config  = config
        # Inject the log signal so PipelineService.log() calls end up in the GUI log.
        self.service = PipelineService(log_fn=self.log.emit)

    def run(self) -> None:
        """Entry point called by QThread.started.  Wires callbacks then runs the pipeline.

        All five PipelineConfig callbacks are wired to the corresponding Qt signals
        so they cross the thread boundary safely.  On completion, finished or error
        is emitted to notify MainWindow.
        """
        try:
            # Wire all five PipelineConfig callback slots to our Qt signals.
            # These will be called from this background thread but Qt ensures
            # they are delivered to connected slots on the correct thread.
            self.config.progress_callback          = self.progress.emit
            self.config.status_callback            = self.status.emit
            self.config.log_callback               = self.log.emit
            self.config.subprogress_callback       = self.subprogress.emit
            self.config.substatus_callback         = self.substatus.emit
            self.config.segment_completed_callback = self.segment_completed.emit

            outputs = self.service.run(self.config)
            # Convert Path objects to strings so the signal payload is JSON-safe.
            self.finished.emit([str(path) for path in outputs])
        except Exception as exc:
            self.error.emit(str(exc))


# ===========================================================================
# MainWindow — the top-level application window
# ===========================================================================

class MainWindow(QMainWindow):
    """Top-level application window for the EPR Video + Sensor Processing Tool.

    Layout (left → right):
      1. Left panel: QTabWidget with four tabs
           — Inputs:          Video dir, navigation, sensors, time intervals
           — Processing:      Output dir, sampling mode, frame rate, options
           — Post-Processing: Per-step checkboxes, CLAHE params, thresholds
           — Visualization:   CSV load, box-select, interval pick, segments
      2. Centre: QStackedWidget cycling between
           — Timeline panel   (visible when Inputs / Processing tab active)
           — Graphs panel     (visible when Post-Processing tab active)
           — Map panel        (visible when Visualization tab active)
      3. Right panel: Project Summary, Skipped/Warnings, Processing Log

    The three panels sit inside a QSplitter so the user can resize them.

    Central state (instance attributes):
      videos, sensor_files, navigation_file, depth_source, speed_source,
      selected_intervals, altitude/depth/speed_threshold, sensor_thresholds,
      applied_steps — all maintained in __init__ and mutated by user actions.
    """

    def __init__(self):
        """Initialise all session state, build the UI, wire signals, and do an initial refresh."""
        super().__init__()
        self.setWindowTitle("EPR Video + Sensor Processing Tool")
        self.setMinimumSize(1000, 650)

        # -----------------------------------------------------------------------
        # Session state — modified by user actions and pipeline events
        # -----------------------------------------------------------------------

        # Last scanned video directory path (string).
        self.video_directory: str = ""

        # strptime format string used by VideoService for filename parsing.
        self.video_datetime_format: str = "%Y%m%d_%H%M%S"

        # VideoRecord list produced by the most recent scan; empty until Scan Videos is clicked.
        self.videos: list[VideoRecord] = []

        # Filenames of video files that were skipped during the last scan (with error reasons).
        self.skipped_videos: list[str] = []

        # SensorFileConfig objects added by the user on the Inputs tab.
        self.sensor_files: list[SensorFileConfig] = []

        # NavigationConfig produced by NavigationImportDialog; None if not configured.
        self.navigation_file: NavigationConfig | None = None

        # Optional dedicated depth channel (shown separately in the Inputs tab).
        self.depth_source: SensorFileConfig | None = None

        # Optional dedicated speed channel (shown separately in the Inputs tab).
        self.speed_source: SensorFileConfig | None = None

        # -----------------------------------------------------------------------
        # Job and segment history state
        # -----------------------------------------------------------------------

        # Auto-incrementing serial counter; the next new job gets this ID.
        self.next_job_id: int = 1

        # The job currently being built in the Visualization tab.
        # Always non-None: a fresh Job is initialized here and after each run.
        self.pending_job: Job = Job(job_id=1)

        # All segments the pipeline has attempted, across all jobs.
        # Persisted in the workspace JSON; drives the history overlay mode.
        self.segment_history: list[SegmentRecord] = []

        # Scalar threshold values read from the Post-Processing spinboxes.
        # None means "threshold is disabled (0.0 was entered)".
        self.altitude_threshold: float | None = None
        self.depth_threshold:    float | None = None
        self.speed_threshold:    float | None = None

        # Per-channel threshold ranges: column_name → (min_val | None, max_val | None).
        # Populated by _collect_sensor_thresholds() just before a pipeline run.
        self.sensor_thresholds: dict[str, tuple[float | None, float | None]] = {}

        # Minimum number of consecutive threshold-passing rows to form a range segment.
        self.min_segment_frames: int = 1

        # Steps that were requested in the most recent pipeline run (used in _on_pipeline_finished).
        self._current_run_steps: list[str] = []

        # Annotation settings — None uses pipeline defaults; overridden via the dialog.
        self.annotation_config: AnnotationConfig = AnnotationConfig()

        # True when the last pipeline run was sensor-only (no image extraction).
        # Controls which message is shown after completion.
        self._sensor_only_run: bool = False

        # Background thread and worker for pipeline execution.
        # Set to None between runs; created fresh for each run.
        self.worker_thread: QThread | None = None
        self.worker:        PipelineWorker | None = None

        # Workspace persistence flags; used to avoid re-saving when not needed.
        self.workspace_saved: bool = False
        self.workspace_path:  str  = ""

        # dict mapping channel display name → (min_checkbox, min_spinbox, max_checkbox, max_spinbox)
        # for the dynamically-generated per-sensor threshold rows on the Post-Processing tab.
        self.sensor_threshold_widgets: dict[str, tuple[QCheckBox, QDoubleSpinBox, QCheckBox, QDoubleSpinBox]] = {}

        # -----------------------------------------------------------------------
        # Visualization state — managed by the Visualization tab handlers
        # -----------------------------------------------------------------------

        # Sensor raster DataFrame (sensor-native timestamps, GPS-geolocated).
        # Populated by _load_sensor_rasters(); None until loaded.
        self._raster_df: pd.DataFrame | None = None

        # Combined DataFrame from one or more master.csv files; None if no data loaded.
        self._viz_df: pd.DataFrame | None = None

        # Display path for the viz_csv_label (single file path or parent directory).
        self._viz_csv_path: str = ""

        # Named segments: list of {name: str, indices: list[int], color: str}.
        # indices are row indices into _viz_df.
        self._viz_segments: list[dict] = []

        # Row indices currently selected by the box-select tool but not yet named.
        # Cleared when the user saves a segment or clears the selection.
        self._viz_pending_indices: list[int] = []

        # -----------------------------------------------------------------------
        # Build UI, wire interactions, and do initial display refresh
        # -----------------------------------------------------------------------
        self._build_ui()
        self._wire_signals()
        self._reset_progress()
        self._refresh_all_views()
        # Restore the last session once the event loop starts so all widgets are ready.
        QTimer.singleShot(0, self._restore_last_session)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------
    # The _build_* methods create and return QWidget trees; they do not wire
    # any signal connections (that is done exclusively in _wire_signals()).
    # Each method is responsible for storing widget references on self so
    # the action handlers can read and update them later.
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the complete application window layout.

        Creates the toolbar, status bar, and then the three-panel splitter:
        left (tabs), centre (stacked display), right (summary).
        """
        self._build_toolbar()
        self._build_status_bar()

        central = QWidget()
        self.setCentralWidget(central)
        root_layout = QHBoxLayout(central)
        root_layout.setContentsMargins(4, 4, 4, 4)

        splitter = QSplitter(Qt.Horizontal)
        root_layout.addWidget(splitter)

        self.controls_tabs = QTabWidget()
        self.controls_tabs.addTab(self._build_inputs_tab(),          "Inputs")
        self.controls_tabs.addTab(self._build_visualization_tab(),   "Visualization")
        self.controls_tabs.addTab(self._build_processing_tab(),      "Processing")
        self.controls_tabs.addTab(self._build_postprocessing_tab(),  "Post-Processing")
        self.controls_tabs.setMinimumWidth(340)
        self.controls_tabs.setMaximumWidth(480)

        splitter.addWidget(self.controls_tabs)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self._build_timeline_panel())
        self.center_stack.addWidget(self._build_postprocessing_graph_panel())
        self.center_stack.addWidget(self._build_map_panel())
        splitter.addWidget(self.center_stack)
        splitter.addWidget(self._build_summary_panel())

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([400, 800, 400])

    def _build_toolbar(self) -> None:
        """Create the main toolbar with Run, workspace management, and config export actions."""
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
        """Build the status bar with two progress bars and two status labels.

        Layout (left → right):
          _status_label     — high-level step name ("Extracting frames…")
          separator         — grey " | "
          _substatus_label  — per-file detail ("video_001.MP4 | frame 4/120")
          _subprogress_bar  — per-video 0–100% bar (150 px wide)
          _progress_bar     — overall 0–100% bar (150 px wide)
        """
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
        """Reset all four progress/status widgets to their idle defaults."""
        self._progress_bar.setValue(0)
        self._status_label.setText("Idle.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("")

    def _build_inputs_tab(self) -> QWidget:
        """Build the Inputs tab: video dir, navigation, sensor files, time intervals.

        Returns a scrollable QWidget containing group boxes for each input category.
        All interactive widgets are stored on self for later access by action handlers.
        """
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
        self.navigation_summary.setMaximumHeight(160)
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

        # Interval management has moved to the Visualization tab (Job Builder panel).

        layout.addStretch()
        return tab

    def _build_processing_tab(self) -> QWidget:
        """Build the Processing tab: output directory, sampling mode, frame rate, options.

        Contains:
          — Output group: output directory path + Browse button
          — Frame Extraction group: sampling mode combo (Fixed / Dynamic),
            frame_rate_spin, dynamic_spacing_spin, dynamic_min_freq_spin, quality combo
          — Options group: sample_images_check, generate_rasters_check, annotate_frames_check
        """
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

        # Output directory is now derived automatically:
        # <workspace_dir>/job_<N:03d>/ — save the workspace first, then run.

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
        annotate_row = QHBoxLayout()
        annotate_row.addWidget(self.annotate_frames_check)
        self.annotate_configure_button = QPushButton("Configure…")
        self.annotate_configure_button.setMaximumWidth(85)
        self.annotate_configure_button.setToolTip(
            "Choose which fields to annotate, font size, text color, and text position."
        )
        annotate_row.addWidget(self.annotate_configure_button)
        options_layout.addWidget(self.sample_images_check)
        options_layout.addWidget(self.generate_rasters_check)
        options_layout.addLayout(annotate_row)
        layout.addWidget(options_group)

        layout.addStretch()
        return tab

    def _build_postprocessing_tab(self) -> QWidget:
        """Build the Post-Processing tab: per-step checkboxes, CLAHE params, thresholds.

        Contains:
          — Steps group: six checkboxes for individual pipeline steps
                         (extract, rasters, annotate, clahe, update_master, geo_txt)
            plus a CLAHE sub-group for clip_limit and tile_grid_size.
          — Thresholds group: altitude/depth/speed spinboxes (0 = disabled).
          — Sensor Thresholds group: dynamically populated by
            _refresh_sensor_thresholds_ui() once sensor channels are loaded.
          — Run buttons: "Run Selected Steps" and "Run Full Sequence".
        """
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
        self.postprocess_annotate_configure_button = QPushButton("Configure…")
        self.postprocess_annotate_configure_button.setMaximumWidth(85)
        self.postprocess_annotate_configure_button.setToolTip(
            "Choose which fields to annotate, font size, text color, and text position."
        )
        self.postprocess_clahe_check = QCheckBox("Apply CLAHE enhancement")
        self.postprocess_update_master_check = QCheckBox("Update master CSV with newly added sources")
        self.postprocess_geo_txt_check = QCheckBox("Generate WebODM geo.txt")
        self.postprocess_geo_txt_check.setToolTip(
            "Write a geo.txt image geolocation file into each frames/ directory. "
            "Upload this file alongside the extracted frames in WebODM to georeference the images."
        )
        self.postprocess_extract_check.setChecked(True)
        self.postprocess_generate_rasters_check.setChecked(True)
        self.postprocess_annotate_check.setChecked(False)
        self.postprocess_clahe_check.setChecked(False)
        self.postprocess_update_master_check.setChecked(True)
        self.postprocess_geo_txt_check.setChecked(False)
        postprocess_annotate_row = QHBoxLayout()
        postprocess_annotate_row.addWidget(self.postprocess_annotate_check)
        postprocess_annotate_row.addWidget(self.postprocess_annotate_configure_button)
        steps_layout.addWidget(self.postprocess_extract_check)
        steps_layout.addWidget(self.postprocess_generate_rasters_check)
        steps_layout.addLayout(postprocess_annotate_row)
        steps_layout.addWidget(self.postprocess_clahe_check)
        steps_layout.addWidget(self.postprocess_update_master_check)
        steps_layout.addWidget(self.postprocess_geo_txt_check)

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

    def _build_visualization_tab(self) -> QWidget:
        """Build the Visualization tab: data source, selection tools, interval picker, segments.

        Contains:
          — Data Source group: label, Browse / Scan / Clear buttons, PPI→CSV converter.
          — Selection group: selection count label, Box Select / Fit View / Clear buttons.
          — Pick Interval group: checkable Pick Interval button + status label.
          — Segments group: name field, Save button, segment list, Export / Remove buttons.
        """
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

        # Data source
        source_group = QGroupBox("Data Source")
        source_layout = QVBoxLayout(source_group)
        self.viz_csv_label = QLabel("No data loaded.")
        self.viz_csv_label.setWordWrap(True)
        self.viz_csv_label.setStyleSheet("color: gray; font-size: 11px;")
        source_layout.addWidget(self.viz_csv_label)
        source_btn_row = QHBoxLayout()
        self.viz_browse_button = QPushButton("Browse interp.csv…")
        self.viz_scan_button = QPushButton("Scan Output Dir")
        self.viz_scan_button.setToolTip("Find and merge all interp.csv files in the configured output directory")
        self.viz_clear_data_button = QPushButton("Clear")
        self.viz_clear_data_button.setMaximumWidth(55)
        source_btn_row.addWidget(self.viz_browse_button)
        source_btn_row.addWidget(self.viz_scan_button)
        source_btn_row.addWidget(self.viz_clear_data_button)
        source_layout.addLayout(source_btn_row)

        ppi_row = QHBoxLayout()
        self.viz_ppi_convert_button = QPushButton("Convert PPI → CSV…")
        self.viz_ppi_convert_button.setToolTip(
            "Select one or more .ppi navigation files and convert them to CSV "
            "in the same directory (preserving all columns)."
        )
        ppi_row.addWidget(self.viz_ppi_convert_button)
        ppi_row.addStretch()
        source_layout.addLayout(ppi_row)
        self.viz_stats_label = QLabel("")
        self.viz_stats_label.setStyleSheet("font-size: 11px;")
        source_layout.addWidget(self.viz_stats_label)
        layout.addWidget(source_group)

        # Sensor Rasters — independent of pipeline; loads from Inputs tab sources
        raster_group = QGroupBox("Sensor Rasters")
        raster_layout = QVBoxLayout(raster_group)
        raster_layout.setSpacing(4)

        self.raster_status_label = QLabel("No raster loaded.")
        self.raster_status_label.setWordWrap(True)
        self.raster_status_label.setStyleSheet("color: gray; font-size: 11px;")
        raster_layout.addWidget(self.raster_status_label)

        raster_btn_row = QHBoxLayout()
        self.raster_load_button = QPushButton("Load from Inputs")
        self.raster_load_button.setToolTip(
            "Build sensor rasters directly from the configured navigation and\n"
            "sensor sources (no pipeline run required). Each sensor reading is\n"
            "geolocated using the nav data at its native timestamp."
        )
        self.raster_clear_button = QPushButton("Clear")
        self.raster_clear_button.setMaximumWidth(55)
        raster_btn_row.addWidget(self.raster_load_button)
        raster_btn_row.addWidget(self.raster_clear_button)
        raster_layout.addLayout(raster_btn_row)

        # Channel + scale row
        raster_chan_row = QHBoxLayout()
        raster_chan_row.addWidget(QLabel("Channel:"))
        self.raster_channel_combo = QComboBox()
        self.raster_channel_combo.addItem("None")
        self.raster_channel_combo.setToolTip("Sensor channel to color the raster trackline")
        raster_chan_row.addWidget(self.raster_channel_combo, stretch=1)
        raster_chan_row.addSpacing(6)
        self.raster_scale_combo = QComboBox()
        self.raster_scale_combo.addItems(["Linear", "Logarithmic"])
        self.raster_scale_combo.setFixedWidth(105)
        raster_chan_row.addWidget(self.raster_scale_combo)
        raster_layout.addLayout(raster_chan_row)

        # Max spinbox (blue / high end)
        raster_max_row = QHBoxLayout()
        self.raster_max_label = QLabel("Max (blue):")
        raster_max_row.addWidget(self.raster_max_label)
        self.raster_max_spin = QDoubleSpinBox()
        self.raster_max_spin.setRange(-1e9, 1e9)
        self.raster_max_spin.setDecimals(4)
        self.raster_max_spin.setValue(1.0)
        self.raster_max_spin.setEnabled(False)
        raster_max_row.addWidget(self.raster_max_spin, stretch=1)
        raster_layout.addLayout(raster_max_row)

        # Min spinbox (red / low end)
        raster_min_row = QHBoxLayout()
        self.raster_min_label = QLabel("Min (red):")
        raster_min_row.addWidget(self.raster_min_label)
        self.raster_min_spin = QDoubleSpinBox()
        self.raster_min_spin.setRange(-1e9, 1e9)
        self.raster_min_spin.setDecimals(4)
        self.raster_min_spin.setValue(0.0)
        self.raster_min_spin.setEnabled(False)
        raster_min_row.addWidget(self.raster_min_spin, stretch=1)
        raster_layout.addLayout(raster_min_row)

        self.raster_reset_button = QPushButton("Reset to data range")
        self.raster_reset_button.setEnabled(False)
        raster_layout.addWidget(self.raster_reset_button)

        self.raster_recolor_button = QPushButton("Recolor")
        self.raster_recolor_button.setEnabled(False)
        raster_layout.addWidget(self.raster_recolor_button)

        raster_export_row = QHBoxLayout()
        self.raster_export_map_button = QPushButton("Export Map…")
        self.raster_export_map_button.setToolTip(
            "Export a print-quality map of the sensor raster trackline with\n"
            "correct geographic scale, colorbar, scale bar, and north arrow."
        )
        self.raster_export_map_button.setEnabled(False)
        self.raster_export_csv_button = QPushButton("Export CSV…")
        self.raster_export_csv_button.setToolTip(
            "Save the sensor raster as a CSV with unix_time, lat, lon, and all\n"
            "loaded channel columns."
        )
        self.raster_export_csv_button.setEnabled(False)
        raster_export_row.addWidget(self.raster_export_map_button)
        raster_export_row.addWidget(self.raster_export_csv_button)
        raster_layout.addLayout(raster_export_row)

        self.raster_export_geojson_button = QPushButton("Export QGIS GeoJSON…")
        self.raster_export_geojson_button.setToolTip(
            "Export the color-spectrum trackline as a GeoJSON file for QGIS.\n"
            "Each track segment becomes a LineString feature with a 'sensor_value'\n"
            "property — apply Graduated symbology in QGIS to reproduce the color spectrum."
        )
        self.raster_export_geojson_button.setEnabled(False)
        raster_layout.addWidget(self.raster_export_geojson_button)

        self.raster_export_analysis_button = QPushButton("Export Sensor Analysis…")
        self.raster_export_analysis_button.setToolTip(
            "Generate a full spatial analysis bundle for the active channel:\n"
            "  • Enhanced point layer (anomaly, z-score, hotspot flags)\n"
            "  • IDW-interpolated raster\n"
            "  • Confidence/distance raster\n"
            "  • Anomaly raster (value − global median)\n"
            "All outputs are QGIS-ready GeoTIFFs and GeoJSON."
        )
        self.raster_export_analysis_button.setEnabled(False)
        raster_layout.addWidget(self.raster_export_analysis_button)

        layout.addWidget(raster_group)

        # Track display options
        display_group = QGroupBox("Track Coloring")
        display_layout = QVBoxLayout(display_group)
        display_layout.setSpacing(4)

        # Channel + scale row
        chan_row = QHBoxLayout()
        chan_row.addWidget(QLabel("Color by:"))
        self.viz_channel_combo = QComboBox()
        self.viz_channel_combo.addItem("None")
        self.viz_channel_combo.setToolTip(
            "Select which channel colors the GPS trackline.\n"
            "Sensor channels come from the loaded master.csv columns."
        )
        chan_row.addWidget(self.viz_channel_combo, stretch=1)
        chan_row.addSpacing(6)
        self.viz_color_scale_combo = QComboBox()
        self.viz_color_scale_combo.addItems(["Linear", "Logarithmic"])
        self.viz_color_scale_combo.setFixedWidth(105)
        self.viz_color_scale_combo.setToolTip(
            "Linear: uniform spacing between min and max.\n"
            "Logarithmic: compresses high values, expands low — useful for\n"
            "data spanning multiple orders of magnitude."
        )
        chan_row.addWidget(self.viz_color_scale_combo)
        display_layout.addLayout(chan_row)

        # Max spinbox — blue (high) end
        max_row = QHBoxLayout()
        self.viz_sensor_max_label = QLabel("Max (blue):")
        max_row.addWidget(self.viz_sensor_max_label)
        self.viz_alt_max_spin = QDoubleSpinBox()
        self.viz_alt_max_spin.setRange(-1e9, 1e9)
        self.viz_alt_max_spin.setDecimals(4)
        self.viz_alt_max_spin.setValue(15.0)
        self.viz_alt_max_spin.setEnabled(False)
        max_row.addWidget(self.viz_alt_max_spin, stretch=1)
        display_layout.addLayout(max_row)

        # Min spinbox — red (low) end
        min_row = QHBoxLayout()
        self.viz_sensor_min_label = QLabel("Min (red):")
        min_row.addWidget(self.viz_sensor_min_label)
        self.viz_alt_min_spin = QDoubleSpinBox()
        self.viz_alt_min_spin.setRange(-1e9, 1e9)
        self.viz_alt_min_spin.setDecimals(4)
        self.viz_alt_min_spin.setValue(0.0)
        self.viz_alt_min_spin.setEnabled(False)
        min_row.addWidget(self.viz_alt_min_spin, stretch=1)
        display_layout.addLayout(min_row)

        self.viz_alt_reset_button = QPushButton("Reset to data range")
        self.viz_alt_reset_button.setEnabled(False)
        display_layout.addWidget(self.viz_alt_reset_button)

        self.viz_alt_recolor_button = QPushButton("Recolor")
        self.viz_alt_recolor_button.setEnabled(False)
        display_layout.addWidget(self.viz_alt_recolor_button)

        self.viz_export_track_button = QPushButton("Export Trackline CSV…")
        self.viz_export_track_button.setToolTip(
            "Save the full GPS trackline as a CSV with lat, lon, unix_time, and the\n"
            "selected sensor column."
        )
        self.viz_export_track_button.setEnabled(False)
        display_layout.addWidget(self.viz_export_track_button)

        self.viz_export_geojson_button = QPushButton("Export QGIS GeoJSON…")
        self.viz_export_geojson_button.setToolTip(
            "Export the color-spectrum trackline as a GeoJSON file for QGIS.\n"
            "Each track segment becomes a LineString feature with a 'sensor_value'\n"
            "property — apply Graduated symbology in QGIS to reproduce the color spectrum."
        )
        self.viz_export_geojson_button.setEnabled(False)
        display_layout.addWidget(self.viz_export_geojson_button)

        # Track width slider
        width_row = QHBoxLayout()
        width_row.addWidget(QLabel("Track width:"))
        self.viz_track_width_slider = QSlider(Qt.Horizontal)
        self.viz_track_width_slider.setRange(1, 10)
        self.viz_track_width_slider.setValue(2)
        self.viz_track_width_slider.setTickPosition(QSlider.TicksBelow)
        self.viz_track_width_slider.setTickInterval(1)
        self.viz_track_width_label = QLabel("2 px")
        self.viz_track_width_label.setFixedWidth(32)
        width_row.addWidget(self.viz_track_width_slider, stretch=1)
        width_row.addWidget(self.viz_track_width_label)
        display_layout.addLayout(width_row)

        layout.addWidget(display_group)

        # Selection tools
        select_group = QGroupBox("Selection")
        select_layout = QVBoxLayout(select_group)
        self.viz_selection_label = QLabel("0 frames selected")
        self.viz_selection_label.setStyleSheet("font-size: 11px;")
        select_layout.addWidget(self.viz_selection_label)
        select_btn_row = QHBoxLayout()
        self.viz_box_select_button = QPushButton("Box Select")
        self.viz_box_select_button.setCheckable(True)
        self.viz_box_select_button.setToolTip("Drag a rectangle on the map to select frames")
        self.viz_fit_button = QPushButton("Fit View")
        self.viz_clear_select_button = QPushButton("Clear Selection")
        select_btn_row.addWidget(self.viz_box_select_button)
        select_btn_row.addWidget(self.viz_fit_button)
        select_btn_row.addWidget(self.viz_clear_select_button)
        select_layout.addLayout(select_btn_row)
        layout.addWidget(select_group)

        # Track interval picker
        pick_group = QGroupBox("Pick Interval from Track")
        pick_layout = QVBoxLayout(pick_group)
        self.viz_pick_interval_button = QPushButton("Pick Interval")
        self.viz_pick_interval_button.setCheckable(True)
        self.viz_pick_interval_button.setToolTip(
            "Click two points on the red trackline to define a time interval.\n"
            "The interval is added to the Pipeline tab's interval list."
        )
        self.viz_pick_status_label = QLabel("Click to activate, then click two points on the track.")
        self.viz_pick_status_label.setStyleSheet("font-size: 10px; color: gray;")
        self.viz_pick_status_label.setWordWrap(True)
        pick_layout.addWidget(self.viz_pick_interval_button)
        pick_layout.addWidget(self.viz_pick_status_label)
        layout.addWidget(pick_group)

        # Segment management
        seg_group = QGroupBox("Segments")
        seg_layout = QVBoxLayout(seg_group)
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.viz_segment_name_edit = QLineEdit()
        self.viz_segment_name_edit.setPlaceholderText("e.g. Transect 1")
        name_row.addWidget(self.viz_segment_name_edit, stretch=1)
        self.viz_save_segment_button = QPushButton("Save Segment")
        name_row.addWidget(self.viz_save_segment_button)
        seg_layout.addLayout(name_row)
        self.viz_segment_list = QListWidget()
        self.viz_segment_list.setMinimumHeight(80)
        self.viz_segment_list.setMaximumHeight(180)
        seg_layout.addWidget(self.viz_segment_list)
        seg_btn_row = QHBoxLayout()
        self.viz_export_segment_button = QPushButton("Export Selected Segment")
        self.viz_remove_segment_button = QPushButton("Remove")
        self.viz_remove_segment_button.setMaximumWidth(70)
        seg_btn_row.addWidget(self.viz_export_segment_button)
        seg_btn_row.addWidget(self.viz_remove_segment_button)
        seg_layout.addLayout(seg_btn_row)
        layout.addWidget(seg_group)

        # Job Builder — define intervals for the next pipeline run
        job_group = QGroupBox("Job Builder")
        job_layout = QVBoxLayout(job_group)
        job_layout.setSpacing(4)

        job_header = QHBoxLayout()
        self.job_id_label = QLabel("Job #1")
        self.job_id_label.setStyleSheet("font-weight: bold; font-size: 12px;")
        self.job_new_button = QPushButton("New Job")
        self.job_new_button.setToolTip("Clear the current job's interval list and start a new job.")
        self.job_new_button.setMaximumWidth(75)
        job_header.addWidget(self.job_id_label)
        job_header.addStretch()
        job_header.addWidget(self.job_new_button)
        job_layout.addLayout(job_header)

        self.job_interval_list = QListWidget()
        self.job_interval_list.setMinimumHeight(80)
        self.job_interval_list.setMaximumHeight(150)
        self.job_interval_list.setToolTip(
            "Intervals queued for the next pipeline run.\n"
            "Pick intervals using 'Pick Interval' above or add from history."
        )
        job_layout.addWidget(self.job_interval_list)

        job_remove_row = QHBoxLayout()
        self.job_remove_interval_button = QPushButton("Remove Selected")
        self.job_remove_interval_button.setEnabled(False)
        job_remove_row.addWidget(self.job_remove_interval_button)
        job_layout.addLayout(job_remove_row)

        # History dropdown — previously processed segments
        history_sep = QLabel("Add from history:")
        history_sep.setStyleSheet("font-size: 10px; color: #555; margin-top: 4px;")
        job_layout.addWidget(history_sep)

        hist_row = QHBoxLayout()
        self.job_history_combo = QComboBox()
        self.job_history_combo.setToolTip(
            "Previously processed segments.  Select one and click Add to\n"
            "re-add it to the current job (e.g. to re-sample with different settings)."
        )
        self.job_history_combo.addItem("— no history —")
        self.job_add_from_history_button = QPushButton("Add")
        self.job_add_from_history_button.setMaximumWidth(45)
        self.job_open_folder_button = QPushButton("📂")
        self.job_open_folder_button.setMaximumWidth(32)
        self.job_open_folder_button.setToolTip("Open the folder for the selected history segment.")
        hist_row.addWidget(self.job_history_combo, stretch=1)
        hist_row.addWidget(self.job_add_from_history_button)
        hist_row.addWidget(self.job_open_folder_button)
        job_layout.addLayout(hist_row)

        layout.addWidget(job_group)

        # History overlay mode toggle
        history_group = QGroupBox("Track History Mode")
        history_layout = QVBoxLayout(history_group)
        self.viz_history_mode_button = QPushButton("Show Sampled Regions")
        self.viz_history_mode_button.setCheckable(True)
        self.viz_history_mode_button.setToolTip(
            "Toggle history overlay: blue = previously sampled regions,\n"
            "red = unsampled.  Hover a blue segment to see job/time info."
        )
        history_layout.addWidget(self.viz_history_mode_button)
        layout.addWidget(history_group)

        layout.addStretch()
        return tab

    def _build_map_panel(self) -> QWidget:
        """Build the centre map panel containing the MapWidget and nav status label.

        The nav status label shows a colour-coded summary of what nav track and frame
        data is currently visible (green = OK, red = wrong columns / missing data).
        """
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header_row = QHBoxLayout()
        title = QLabel("Frame Location Map")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        header_row.addWidget(title)
        header_row.addStretch()
        self.viz_utm_zone_label = QLabel("UTM: —")
        self.viz_utm_zone_label.setStyleSheet(
            "font-size: 10px; color: #555; padding: 0 8px;"
        )
        self.viz_utm_zone_label.setToolTip(
            "UTM zone derived from the mean longitude of the loaded GPS trackline"
        )
        header_row.addWidget(self.viz_utm_zone_label)
        header_row.addWidget(QLabel("Scale:"))
        self.map_scale_combo = QComboBox()
        self.map_scale_combo.addItems(["° Lat / Lon", "m Meters"])
        self.map_scale_combo.setFixedWidth(120)
        self.map_scale_combo.setToolTip("Switch between raw coordinates and a metres-from-centroid projection")
        header_row.addWidget(self.map_scale_combo)
        self.viz_print_button = QPushButton("Export Map…")
        self.viz_print_button.setToolTip(
            "Export the current map as a print-quality PDF, PNG, or SVG with\n"
            "correct geographic scale, scale bar, colorbar, and axis labels."
        )
        header_row.addWidget(self.viz_print_button)
        layout.addLayout(header_row)

        subtitle = QLabel(
            "Colored line = GPS track by altitude  ·  Blue dots = extracted frames  ·"
            "  Hover track for GPS time / video timestamp  ·  Pick Interval: click two track points"
        )
        subtitle.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(subtitle)

        self.map_nav_status_label = QLabel("")
        self.map_nav_status_label.setWordWrap(True)
        self.map_nav_status_label.setTextFormat(Qt.RichText)
        self.map_nav_status_label.setStyleSheet("font-size: 10px; padding: 2px 4px;")
        layout.addWidget(self.map_nav_status_label)

        # Map area: thin altitude key on the left, MapWidget on the right.
        map_area = QHBoxLayout()
        map_area.setSpacing(4)

        # Altitude colour key — thin vertical gradient bar with min/max labels.
        self.viz_alt_key_widget = QWidget()
        self.viz_alt_key_widget.setFixedWidth(42)
        self.viz_alt_key_widget.setVisible(False)
        key_layout = QVBoxLayout(self.viz_alt_key_widget)
        key_layout.setContentsMargins(2, 0, 2, 0)
        key_layout.setSpacing(2)

        self.viz_alt_key_max_label = QLabel("—")
        self.viz_alt_key_max_label.setAlignment(Qt.AlignHCenter)
        self.viz_alt_key_max_label.setStyleSheet("font-size: 9px; color: #00c;")
        self.viz_alt_key_max_label.setWordWrap(True)

        self.viz_alt_key_gradient = QLabel()
        self.viz_alt_key_gradient.setStyleSheet(
            "background: qlineargradient("
            "x1:0, y1:1, x2:0, y2:0, "
            "stop:0    rgba(220,  0,  0,255), "
            "stop:0.25 rgba(220,220,  0,255), "
            "stop:0.5  rgba(  0,200,  0,255), "
            "stop:0.75 rgba(  0,210,210,255), "
            "stop:1    rgba(  0,  0,210,255)"
            ");"
            "border: 1px solid #999;"
            "border-radius: 2px;"
        )
        self.viz_alt_key_gradient.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Expanding)
        self.viz_alt_key_gradient.setFixedWidth(18)

        self.viz_alt_key_min_label = QLabel("—")
        self.viz_alt_key_min_label.setAlignment(Qt.AlignHCenter)
        self.viz_alt_key_min_label.setStyleSheet("font-size: 9px; color: #c00;")
        self.viz_alt_key_min_label.setWordWrap(True)

        key_layout.addWidget(self.viz_alt_key_max_label)
        key_layout.addWidget(self.viz_alt_key_gradient, stretch=1)
        key_layout.addWidget(self.viz_alt_key_min_label)

        map_area.addWidget(self.viz_alt_key_widget)

        self.map_widget = MapWidget()
        map_area.addWidget(self.map_widget, stretch=1)
        layout.addLayout(map_area, stretch=1)
        return panel

    def _build_timeline_panel(self) -> QWidget:
        """Build the centre timeline panel containing the TimelineWidget.

        Visible when the Inputs or Processing tab is active.  Displays coloured
        horizontal bars showing the temporal coverage of videos (blue), navigation
        (yellow), and sensor channels (green).
        """
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
        """Build the centre graphs panel for the Post-Processing tab.

        Contains a vertical scrollable area (graphs_container / graphs_layout)
        that is populated dynamically by _refresh_postprocessing_graphs().  Each
        loaded data source gets one pyqtgraph timeseries plot with optional
        threshold reference lines.
        """
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
        """Build the right-side summary panel with three read-only text areas.

        — Project Summary: key configuration values and video coverage.
        — Skipped / Warnings: videos that failed to scan (populated by _refresh_warnings).
        — Processing Log: pipeline output lines appended by _append_log.
        """
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

    # -----------------------------------------------------------------------
    # Signal wiring
    # -----------------------------------------------------------------------

    def _wire_signals(self) -> None:
        """Connect all button/combo/spin signals to their handler methods.

        Called once in __init__ after _build_ui().  Grouping all connections
        here makes it easy to see the complete interaction model in one place
        without having to hunt through the _build_* methods.
        """
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
        self.annotate_configure_button.clicked.connect(self._open_annotation_settings)
        self.postprocess_annotate_configure_button.clicked.connect(self._open_annotation_settings)
        self.job_new_button.clicked.connect(self._job_new)
        self.job_remove_interval_button.clicked.connect(self._job_remove_selected_interval)
        self.job_interval_list.currentRowChanged.connect(
            lambda row: self.job_remove_interval_button.setEnabled(row >= 0)
        )
        self.job_add_from_history_button.clicked.connect(self._job_add_from_history)
        self.job_open_folder_button.clicked.connect(self._job_open_history_folder)
        self.viz_history_mode_button.toggled.connect(self._on_history_mode_toggled)
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

        # Visualization
        self.viz_browse_button.clicked.connect(self._viz_browse_csv)
        self.viz_ppi_convert_button.clicked.connect(self._viz_convert_ppi)
        self.viz_scan_button.clicked.connect(self._viz_scan_output_dir)
        self.viz_clear_data_button.clicked.connect(self._viz_clear_data)
        self.viz_box_select_button.toggled.connect(self._viz_toggle_select_mode)
        self.viz_fit_button.clicked.connect(self.map_widget.fit_view)
        self.viz_clear_select_button.clicked.connect(self._viz_clear_selection)
        self.viz_pick_interval_button.toggled.connect(self._viz_toggle_pick_mode)
        self.map_widget.segment_created.connect(self._on_track_interval_picked)
        self.map_widget.pick_point_placed.connect(self._on_track_pick_point_placed)
        self.viz_save_segment_button.clicked.connect(self._viz_save_segment)
        self.viz_export_segment_button.clicked.connect(self._viz_export_segment)
        self.viz_remove_segment_button.clicked.connect(self._viz_remove_segment)
        self.map_widget.selection_changed.connect(self._on_viz_selection_changed)
        self.map_scale_combo.currentTextChanged.connect(self._on_map_scale_changed)
        self.viz_channel_combo.currentTextChanged.connect(self._on_channel_combo_changed)
        self.viz_color_scale_combo.currentTextChanged.connect(self._on_color_scale_changed)
        self.viz_alt_recolor_button.clicked.connect(self._on_sensor_range_changed)
        self.viz_alt_reset_button.clicked.connect(self._on_sensor_range_reset)
        self.viz_export_track_button.clicked.connect(self._viz_export_track_csv)
        self.viz_export_geojson_button.clicked.connect(self._viz_export_qgis_geojson)
        self.viz_print_button.clicked.connect(self._viz_export_map)
        self.viz_track_width_slider.valueChanged.connect(self._on_trackline_width_changed)

        # Sensor Rasters
        self.raster_load_button.clicked.connect(self._load_sensor_rasters)
        self.raster_clear_button.clicked.connect(self._clear_sensor_rasters)
        self.raster_channel_combo.currentTextChanged.connect(self._on_raster_channel_changed)
        self.raster_scale_combo.currentTextChanged.connect(self._on_raster_scale_changed)
        self.raster_reset_button.clicked.connect(self._on_raster_range_reset)
        self.raster_recolor_button.clicked.connect(self._on_raster_recolor)
        self.raster_export_map_button.clicked.connect(self._raster_export_map)
        self.raster_export_csv_button.clicked.connect(self._raster_export_csv)
        self.raster_export_geojson_button.clicked.connect(self._raster_export_qgis_geojson)
        self.raster_export_analysis_button.clicked.connect(self._export_sensor_analysis)

    # -----------------------------------------------------------------------
    # User action handlers
    # -----------------------------------------------------------------------
    # Each handler corresponds to one user interaction.  They mutate session
    # state (self.videos, self.sensor_files, etc.) then call _refresh_all_views()
    # so every dependent widget updates in one shot.
    # -----------------------------------------------------------------------

    def _browse_video_directory(self) -> None:
        """Open a directory picker and populate the video directory text field."""
        selected = QFileDialog.getExistingDirectory(self, "Select Video Directory")
        if selected:
            self.video_dir_edit.setText(selected)

    def _scan_videos(self) -> None:
        """Read the video directory field, scan it with VideoService, and refresh views.

        Stores the resulting VideoRecord list in self.videos and skipped filenames
        in self.skipped_videos so _refresh_warnings() can display them.
        """
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
        """Open NavigationImportDialog and store the resulting NavigationConfig."""
        dialog = NavigationImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                self.navigation_file = result
                self._refresh_all_views()

    def _clear_navigation_file(self) -> None:
        """Clear the configured navigation source and refresh all dependent views."""
        self.navigation_file = None
        self._refresh_all_views()

    def _add_sensor_file(self) -> None:
        """Open SensorImportDialog and append the resulting SensorFileConfig to sensor_files."""
        dialog = SensorImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                self.sensor_files.append(result)
                self._refresh_all_views()

    def _add_depth_source(self) -> None:
        """Open SensorImportDialog for the depth channel; force its display name to "Depth"."""
        dialog = SensorImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                if result.channels:
                    result.channels[0].display_name = "Depth"
                self.depth_source = result
                self._refresh_all_views()

    def _clear_depth_source(self) -> None:
        """Clear the depth sensor source and refresh all dependent views."""
        self.depth_source = None
        self._refresh_all_views()

    def _add_speed_source(self) -> None:
        """Open SensorImportDialog for the speed channel; force its display name to "Speed"."""
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
        """Clear the speed sensor source and refresh all dependent views."""
        self.speed_source = None
        self._refresh_all_views()

    def _open_annotation_settings(self) -> None:
        """Open the Annotation Settings dialog and update annotation_config on accept."""
        dlg = AnnotationSettingsDialog(
            parent=self,
            current_config=self.annotation_config,
            navigation_file=self.navigation_file,
            sensor_files=self.sensor_files,
        )
        if dlg.exec() == QDialog.Accepted:
            self.annotation_config = dlg.get_result()

    def _remove_selected_sensor(self) -> None:
        """Remove the sensor_files entry highlighted in sensor_list and refresh views.

        Does nothing if no row is currently selected (currentRow() == -1).
        """
        current_row = self.sensor_list.currentRow()
        if current_row < 0:
            return
        del self.sensor_files[current_row]
        self._refresh_all_views()


    # -----------------------------------------------------------------------
    # Pipeline management
    # -----------------------------------------------------------------------

    def _build_pipeline_config(self, selected_steps: list[str] | None = None) -> PipelineConfig:
        """Read all form fields and construct a PipelineConfig object.

        Validates that the video directory and output directory are set and that
        videos have been scanned.  Reads sampling mode, frame rate, thresholds,
        and all step checkboxes.

        Args:
            selected_steps: If provided, overrides the default step list derived
                            from the Processing tab checkboxes.

        Returns:
            A fully populated PipelineConfig ready to pass to PipelineWorker.

        Raises:
            ValueError: If a required field is empty.
        """
        video_dir = self.video_dir_edit.text().strip()
        if not video_dir:
            raise ValueError("Select a video directory on the Inputs tab.")
        if not self.videos:
            raise ValueError("Scan videos before running the pipeline.")
        if not self.pending_job.intervals:
            raise ValueError("Add at least one interval to the current job (Visualization tab → Job Builder).")
        if not self.workspace_path:
            raise ValueError(
                "Save the workspace first (toolbar → Save Workspace).\n"
                "The job output directory is derived from the workspace location."
            )
        output_dir = str(Path(self.workspace_path).parent / f"job_{self.pending_job.job_id:03d}")

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
            job_id=self.pending_job.job_id,
            video_filename_time_format=self.video_format_edit.text().strip(),
            videos=self.videos,
            selected_intervals=list(self.pending_job.intervals),
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
            annotation_config=self.annotation_config,
        )
        if selected_steps is not None:
            config.selected_steps = selected_steps
        return config

    def _run_pipeline(self, checked: bool = False, selected_steps: list[str] | None = None) -> None:
        """Validate config, show the extraction preview dialog, then start a background pipeline run.

        Checks for existing frames (to avoid silently overwriting output), offers to
        save a new workspace if needed, then creates a QThread + PipelineWorker pair and
        starts them.  All signal connections are made before the thread starts.

        Args:
            checked:        Unused bool from toolbar action signal; included for signal compatibility.
            selected_steps: Optional list of step names; passed through to _build_pipeline_config.
        """
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

        # Mark the pending job as running
        self.pending_job.status = "running"

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
        self.worker.segment_completed.connect(self._on_segment_completed)
        self.worker.finished.connect(self.worker_thread.quit)
        self.worker.error.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker.deleteLater)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.start()

    def _run_selected_postprocessing(self) -> None:
        """Read the Post-Processing tab checkboxes and run the checked steps only."""
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
        if self.postprocess_geo_txt_check.isChecked():
            selected_steps.append("generate_geo_txt")
        if not selected_steps:
            QMessageBox.warning(self, "No steps selected", "Select at least one post-processing step to run.")
            return
        self._run_pipeline(selected_steps=selected_steps)

    def _run_full_postprocessing(self) -> None:
        """Check all four image-related post-processing steps and run the full sequence."""
        self.postprocess_extract_check.setChecked(True)
        self.postprocess_generate_rasters_check.setChecked(True)
        self.postprocess_annotate_check.setChecked(True)
        self.postprocess_clahe_check.setChecked(True)
        self._run_pipeline(selected_steps=["extract_frames", "generate_sensor_rasters", "annotate_frames", "apply_clahe"])

    def _compute_frame_preview(self, config: PipelineConfig) -> list[dict]:
        """Estimate the frame count for each interval without extracting any frames.

        For fixed-rate mode, uses floor(duration × frame_rate) + 1.
        For dynamic mode, calls PipelineService._get_dynamic_sample_times() with
        the loaded nav data.

        Returns:
            List of dicts with keys: idx, start, end, duration_s, frame_count, mode.
        """
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
        """Show a modal preview dialog with per-interval frame counts and ask for confirmation.

        Returns:
            True if the user clicked Proceed; False if they clicked Cancel.
        """
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
        """Return True if any existing frames directory contains JPEG files for this config.

        Used to guard against silently overwriting output from a previous run.
        """
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
        """Prompt the user to choose a new output directory and workspace file before re-extraction.

        When a re-extraction is requested and frames already exist in the output directory,
        this method shows an explanation dialog, then opens two file dialogs (output directory
        and workspace file).  If either dialog is cancelled, the re-extraction is aborted.

        Returns:
            True if a new workspace was saved and the output directory was updated.
            False if the user cancelled at any point.
        """
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

        # New job for the new workspace
        self.next_job_id += 1
        self.pending_job = Job(job_id=self.next_job_id)

        path, _ = QFileDialog.getSaveFileName(
            self, "Save New Workspace", filter="JSON Files (*.json)"
        )
        if not path:
            return False

        try:
            ConfigService.save_workspace(path=path, **self._workspace_save_kwargs())
            self.workspace_saved = True
            self.workspace_path = path
            self.log_text.append(f"New workspace saved to {path}")
        except Exception as exc:
            QMessageBox.critical(self, "Workspace save failed", str(exc))
            return False

        return True

    def _on_segment_completed(self, record: SegmentRecord) -> None:
        """Record a finished segment and auto-save the workspace.

        Called from the main thread via the segment_completed signal after each
        segment the pipeline processes.  Appends to segment_history, refreshes
        the history dropdown in the Job Builder, and saves the workspace so the
        record survives a crash before the full job finishes.
        """
        self.segment_history.append(record)
        self._refresh_job_history_dropdown()
        self._auto_save_workspace()

    def _on_pipeline_finished(self, output_dirs: list[str]) -> None:
        """Handle successful pipeline completion: advance job, re-enable controls, show result."""
        self._set_processing_enabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText("Pipeline complete.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("Idle.")
        self.log_text.append("Pipeline complete.")

        # Mark the completed job and create the next pending job.
        self.pending_job.status = "completed"
        self.next_job_id += 1
        self.pending_job = Job(job_id=self.next_job_id)
        self._refresh_job_builder_ui()
        self._auto_save_workspace()
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
        """Handle pipeline failure: re-enable controls and display the error message."""
        self._set_processing_enabled(True)
        self._status_label.setText("Pipeline failed.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("")
        self.pending_job.status = "pending"  # Reset so the job can be re-run or edited
        self.log_text.append(f"Pipeline failed: {message}")
        QMessageBox.critical(self, "Pipeline failed", message)

    def _set_processing_enabled(self, enabled: bool) -> None:
        """Enable or disable all controls that must be blocked while the pipeline is running."""
        self.run_action.setEnabled(enabled)
        self.scan_videos_button.setEnabled(enabled)
        self.save_config_action.setEnabled(enabled)
        self.save_workspace_action.setEnabled(enabled)
        self.load_workspace_action.setEnabled(enabled)

    # -----------------------------------------------------------------------
    # Workspace and configuration persistence
    # -----------------------------------------------------------------------

    def _save_configuration(self) -> None:
        """Export a standalone pipeline configuration JSON (not a full workspace).

        The config JSON records video paths, sensor files, and pipeline parameters
        but does not include applied_steps or workspace bookkeeping fields.
        Intended for sharing or scripting — not for session restore.
        """
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
                selected_intervals=list(self.pending_job.intervals),
                navigation_file=self.navigation_file,
                output_directory=str(Path(self.workspace_path).parent / f"job_{self.pending_job.job_id:03d}") if self.workspace_path else None,
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

    # -----------------------------------------------------------------------
    # Last-session persistence (auto-save on close, auto-restore on open)
    # -----------------------------------------------------------------------

    @staticmethod
    def _last_session_path() -> Path:
        """Return the fixed path used for automatic session save/restore.

        Written on every close; read on every startup.  Stored in the user's
        config directory so it survives reboots and is independent of any
        specific output directory.
        """
        cfg_dir = Path.home() / ".config" / "epr_imaging"
        cfg_dir.mkdir(parents=True, exist_ok=True)
        return cfg_dir / "last_session.json"

    def closeEvent(self, event) -> None:
        """Auto-save the current workspace when the window is closed.

        Writes the full session state to a fixed path in ~/.config/epr_imaging/
        so it can be silently restored the next time the app opens.  Errors are
        swallowed so a save failure never blocks the app from closing.
        """
        try:
            ConfigService.save_workspace(
                path=str(self._last_session_path()),
                **self._workspace_save_kwargs(),
            )
        except Exception:
            pass  # Never block close due to a save failure
        event.accept()

    def _restore_last_session(self) -> None:
        """Load the last-session workspace on startup if one exists.

        Called via QTimer.singleShot(0, ...) so it runs after the event loop
        starts and all widgets are fully initialised.  Silent on success; logs a
        single warning line if the file exists but cannot be parsed.
        """
        path = self._last_session_path()
        if not path.exists():
            return
        try:
            data = ConfigService.load_workspace(str(path))
            self._apply_workspace_data(data)
            self.log_text.append(f"Last session restored from {path}")
        except Exception as exc:
            self.log_text.append(f"Could not restore last session: {exc}")

    def _save_workspace(self) -> None:
        """Save the complete session state to a JSON workspace file.

        Workspace files store all paths relative to the workspace file location
        so they remain valid when the project is moved to a different directory.
        Uses ConfigService.save_workspace() for serialisation.
        """
        path, _ = QFileDialog.getSaveFileName(self, "Save Workspace", filter="JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            ConfigService.save_workspace(path=path, **self._workspace_save_kwargs())
            self.log_text.append("Workspace saved.")
            self.workspace_saved = True
            self.workspace_path = path
        except Exception as exc:
            QMessageBox.critical(self, "Workspace save failed", str(exc))

    def _clear_workspace(self) -> None:
        """Save the current workspace then reset all fields to their defaults.

        Forces a save first so no work is lost.  If the user cancels the save dialog,
        the clear is aborted.  After saving, all session state variables and all
        form widgets are reset to their initial values.
        """
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
        self.next_job_id = 1
        self.pending_job = Job(job_id=1)
        self.segment_history = []
        self.annotation_config = AnnotationConfig()
        self.altitude_threshold = None
        self.depth_threshold = None
        self.speed_threshold = None
        self.sensor_thresholds = {}
        self.min_segment_frames = 1
        self.workspace_saved = False
        self.workspace_path = ""

        self.video_dir_edit.setText("")
        self.video_format_edit.setText(self.video_datetime_format)
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

    def _apply_workspace_data(self, data: dict) -> None:
        """Apply a loaded workspace dict to all form fields and session state.

        Shared by _load_workspace (manual load) and _restore_last_session
        (automatic startup restore).  Re-scans the video directory so the
        VideoRecord list is populated before _refresh_all_views() runs.

        Args:
            data: Dict returned by ConfigService.load_workspace().
        """
        self.video_dir_edit.setText(data["video_directory"])
        self.video_format_edit.setText(data["filename_datetime_format"])
        self.navigation_file = data["navigation_file"]
        self.depth_source = data.get("depth_source")
        self.speed_source = data.get("speed_source")
        self.sensor_files = data["sensor_files"]
        self.next_job_id = int(data.get("next_job_id", 1))
        self.pending_job = data.get("pending_job") or Job(job_id=self.next_job_id)
        self.segment_history = data.get("segment_history") or []
        self.frame_rate_spin.setValue(data["frame_rate"])
        self.generate_rasters_check.setChecked(data["generate_sensor_tiffs"])
        self.annotate_frames_check.setChecked(data["annotate_frames"])
        self.altitude_threshold = data.get("altitude_threshold")
        self.depth_threshold = data.get("depth_threshold")
        self.speed_threshold = data.get("speed_threshold")
        self.min_segment_frames = int(data.get("min_segment_frames", 1))
        ann_data = data.get("annotation_config")
        self.annotation_config = AnnotationConfig.from_dict(ann_data) if ann_data else AnnotationConfig()
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

    def _load_workspace(self) -> None:
        """Load a previously saved workspace JSON file and restore all form fields.

        After loading, re-scans the video directory to rebuild the VideoRecord list
        (so the pipeline can use pre-loaded records without a redundant scan).
        Calls _refresh_all_views() to update every dependent widget.
        """
        path, _ = QFileDialog.getOpenFileName(self, "Load Workspace", filter="JSON Files (*.json);;All Files (*)")
        if not path:
            return
        try:
            data = ConfigService.load_workspace(path)
            self._apply_workspace_data(data)
            self.log_text.append("Workspace loaded.")
            self.workspace_saved = True
            self.workspace_path = path
        except Exception as exc:
            QMessageBox.critical(self, "Workspace load failed", str(exc))

    def _workspace_save_kwargs(self) -> dict:
        """Return the keyword dict for ConfigService.save_workspace() from current state."""
        return dict(
            video_directory=self.video_dir_edit.text().strip(),
            filename_datetime_format=self.video_format_edit.text().strip(),
            navigation_file=self.navigation_file,
            sensor_files=self.sensor_files,
            pending_job=self.pending_job,
            next_job_id=self.next_job_id,
            segment_history=self.segment_history,
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
            sampling_mode="dynamic" if self.sampling_mode_combo.currentText() == "Dynamic spacing" else "fixed",
            dynamic_target_spacing_m=float(self.dynamic_spacing_spin.value()),
            dynamic_min_frequency_hz=float(self.dynamic_min_freq_spin.value()),
            clahe_clip_limit=float(self.clahe_clip_limit_spin.value()),
            clahe_tile_grid_size=int(self.clahe_tile_size_spin.value()),
            annotation_config=self.annotation_config,
        )

    def _auto_save_workspace(self) -> None:
        """Silently save the workspace to workspace_path (if set) or the last-session path.

        Called after each segment completes and after job completion so the
        segment history survives crashes.  Failures are logged but never block.
        """
        save_path = self.workspace_path or str(self._last_session_path())
        try:
            ConfigService.save_workspace(path=save_path, **self._workspace_save_kwargs())
            if self.workspace_path:
                self.workspace_saved = True
        except Exception as exc:
            self.log_text.append(f"Auto-save workspace failed: {exc}")

    def _append_log(self, message: str) -> None:
        """Append one line to the Processing Log text area (connected to PipelineWorker.log)."""
        self.log_text.append(message)

    def _refresh_all_views(self) -> None:
        """Rebuild every dependent UI widget from current session state.

        Called after any state change that could affect multiple widgets.
        The order matches the visual top-to-bottom layout so the most visible
        elements (sensor list, nav summary) update first.
        """
        self._refresh_sensor_list()
        self._refresh_navigation_summary()
        self._refresh_depth_summary()
        self._refresh_speed_summary()
        self._refresh_job_builder_ui()
        self._refresh_timeline()
        self._refresh_sensor_thresholds_ui()
        self._refresh_sampling_mode_ui()
        self._update_center_panel()
        self._refresh_summary()
        self._refresh_warnings()

    def _refresh_sampling_mode_ui(self) -> None:
        """Enable/disable the correct spinboxes for the selected sampling mode.

        In Fixed mode: frame_rate_spin is enabled, dynamic spinboxes are disabled.
        In Dynamic mode: the two dynamic spinboxes are enabled, frame_rate_spin is disabled.
        Also updates tooltips to explain the "navigation required" constraint when
        dynamic mode is selected without a navigation source configured.
        """
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
        """Refresh the sensor graphs when a threshold spinbox value changes.

        Only triggers the refresh if the Post-Processing graphs panel is currently
        visible (centre stack index 1) to avoid unnecessary data reads.
        """
        if self.center_stack.currentIndex() == 1:
            self._refresh_postprocessing_graphs()

    def _update_center_panel(self, index: int | None = None) -> None:
        """Switch the centre QStackedWidget to match the currently active left tab.

        Post-Processing tab → graphs panel (index 1, also triggers graph refresh).
        Visualization tab   → map panel   (index 2, also triggers map refresh).
        Any other tab       → timeline    (index 0).
        """
        if index is None:
            index = self.controls_tabs.currentIndex()
        tab_text = self.controls_tabs.tabText(index)
        if tab_text == "Post-Processing":
            self.center_stack.setCurrentIndex(1)
            self._refresh_postprocessing_graphs()
        elif tab_text == "Visualization":
            self.center_stack.setCurrentIndex(2)
            self._refresh_map()
        else:
            self.center_stack.setCurrentIndex(0)

    def _refresh_sensor_list(self) -> None:
        """Rebuild the sensor_list QListWidget from the current sensor_files state."""
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
        """Update the navigation_summary QTextEdit with file names, columns, and time coverage."""
        if self.navigation_file is None:
            self.navigation_summary.setPlainText("No navigation sources configured.")
            return

        def source_line(label: str, source: TimeValueSourceConfig | None) -> str:
            """Format a single navigation source as "Label: filename [column]  [start→end]"."""
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
            source_line("Lat",   self.navigation_file.latitude_source),
            source_line("Lon",   self.navigation_file.longitude_source),
            source_line("Alt",   self.navigation_file.altitude_source),
            source_line("Depth", self.navigation_file.depth_source),
            source_line("Pitch", self.navigation_file.pitch_source),
            source_line("Roll",  self.navigation_file.roll_source),
        ]
        # Omit "<not set>" lines for optional sources that aren't configured.
        lines = [l for l in lines if "<not set>" not in l or l.startswith("Lat") or l.startswith("Lon")]
        self.navigation_summary.setPlainText("\n".join(lines))

    def _refresh_depth_summary(self) -> None:
        """Update the depth_summary label with the current depth source's file and channel info.

        Shows the filename, channel display name, source column, and UTC coverage
        window.  Falls back to a "not configured" message when depth_source is None.
        """
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
        """Update the speed_summary label with the current speed source's file and channel info.

        Mirrors _refresh_depth_summary() — shows filename, channel name, source
        column name, and UTC coverage window.
        """
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
        """Compatibility shim — delegates to the new Job Builder UI refresh."""
        self._refresh_job_builder_ui()

    def _refresh_timeline(self) -> None:
        """Build the timeline_items list and push it to the TimelineWidget.

        Navigation sources are wrapped in no-channel SensorFileConfig objects so
        the TimelineWidget can display them alongside real sensor files using the
        same _draw_segment() code path.
        """
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
        """Rebuild the Project Summary text area with the current session state snapshot."""
        lines = [
            f"Videos: {len(self.videos)}",
            f"Navigation: {'configured' if self.navigation_file else 'not set'}",
            f"Depth source: {'configured' if self.depth_source else 'not set'}",
            f"Speed source: {'configured' if self.speed_source else 'not set'}",
            f"Sensor CSVs: {len(self.sensor_files)}",
            f"Job #{self.pending_job.job_id}: {len(self.pending_job.intervals)} interval(s)",
            f"Completed segments: {len(self.segment_history)}",
            f"Output: <workspace dir>/job_{self.pending_job.job_id:03d}/" if self.workspace_path else "Output: <save workspace first>",
            f"Sampling: {self.sampling_mode_combo.currentText()}",
            f"Frame rate: {self.frame_rate_spin.value():.2f} Hz" if self.sampling_mode_combo.currentText() == "Fixed rate" else f"Target spacing: {self.dynamic_spacing_spin.value():.2f} m  |  f_min: {self.dynamic_min_freq_spin.value():.3f} Hz",
            f"Quality: {self.frame_quality_combo.currentText()}",
            f"Sensor TIFFs: {'yes' if self.generate_rasters_check.isChecked() else 'no'}",
            f"Annotate frames: {'yes' if self.annotate_frames_check.isChecked() else 'no'}",
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
        """Clear and rebuild all sensor timeseries plots in the Post-Processing graphs panel.

        Creates one pyqtgraph PlotWidget per data source (altitude, depth, speed,
        and each sensor channel).  Each plot shows the full timeseries and, if
        a non-zero threshold is set, adds a dashed horizontal InfiniteLine at
        the threshold value so the user can visually identify valid windows.
        """
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
        """Rebuild the per-channel threshold rows in the sensor_thresholds_group.

        Called whenever sensor_files changes.  Preserves existing threshold
        values by key so adding or removing a sensor file doesn't reset the
        values for channels that are still present.

        Each channel gets a row:  [Min checkbox] [min spinbox]  [Max checkbox] [max spinbox]
        The spinboxes are disabled until their respective checkboxes are ticked.
        """
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
        """Read all per-channel threshold widgets and return the active (non-None) values.

        Returns:
            Dict mapping channel display name → (min_val | None, max_val | None).
            Only channels with at least one enabled bound are included.
        """
        result: dict[str, tuple[float | None, float | None]] = {}
        for name, (min_en, min_spin, max_en, max_spin) in self.sensor_threshold_widgets.items():
            min_val = min_spin.value() if min_en.isChecked() else None
            max_val = max_spin.value() if max_en.isChecked() else None
            if min_val is not None or max_val is not None:
                result[name] = (min_val, max_val)
        return result

    # -----------------------------------------------------------------------
    # Visualization tab handlers
    # -----------------------------------------------------------------------
    # These methods manage the Visualization tab state: loading CSVs, toggling
    # selection/pick modes, managing named segments, and triggering map refreshes.
    # -----------------------------------------------------------------------

    def _viz_browse_csv(self) -> None:
        """Open a file picker for a single interp.csv and load it for visualization."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select interp.csv", filter="CSV Files (*.csv);;All Files (*)"
        )
        if not path:
            return
        self._viz_load_csv_files([Path(path)])

    def _viz_convert_ppi(self) -> None:
        """Convert one or more .ppi navigation files to .csv in the same directory.

        Uses SensorService._read_ppi() to parse the binary PPI format.  Writes
        each converted file alongside the source with a .csv extension.  Reports
        a summary of successes and failures in an information or warning dialog.
        """
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Select PPI files to convert",
            filter="PPI Navigation Files (*.ppi);;All Files (*)"
        )
        if not paths:
            return

        converted: list[str] = []
        failed: list[str] = []
        for path_str in paths:
            src = Path(path_str)
            dst = src.with_suffix(".csv")
            try:
                df = SensorService._read_ppi(src)
                df.to_csv(dst, index=False)
                converted.append(dst.name)
            except Exception as exc:
                failed.append(f"{src.name}: {exc}")

        lines = []
        if converted:
            lines.append(f"Converted {len(converted)} file(s):")
            lines.extend(f"  {n}" for n in converted)
        if failed:
            lines.append("")
            lines.append(f"Failed ({len(failed)}):")
            lines.extend(f"  {e}" for e in failed)

        if converted:
            QMessageBox.information(self, "PPI → CSV conversion", "\n".join(lines))
        else:
            QMessageBox.warning(self, "PPI → CSV conversion", "\n".join(lines))

    def _viz_scan_output_dir(self) -> None:
        """Recursively find all interp.csv files under the output directory and load them.

        Uses rglob("interp.csv") so it finds files in segment subdirectories at any depth.
        All found CSVs are concatenated by _viz_load_csv_files() into a single DataFrame
        with a "_source_csv" column tracking which file each row came from.
        """
        if not self.workspace_path:
            QMessageBox.warning(self, "No workspace", "Save the workspace first — output lives next to the workspace file.")
            return
        root = Path(self.workspace_path).parent
        if not root.exists():
            QMessageBox.warning(self, "Directory not found", f"Workspace directory does not exist:\n{root}")
            return
        csv_files = sorted(root.rglob("interp.csv"))
        if not csv_files:
            QMessageBox.information(self, "No data found", "No interp.csv files found under the output directory.")
            return
        self._viz_load_csv_files(csv_files)

    def _viz_load_csv_files(self, csv_files: list[Path]) -> None:
        """Load, concatenate, and store the given interp.csv files as _viz_df.

        Adds a "_source_csv" column to each DataFrame before concatenation so
        segment export can locate the source frames/ directory for each row.
        Sorts by unix_time if that column exists.  Triggers a map refresh.

        Args:
            csv_files: List of paths to master.csv files to load and merge.
        """
        frames: list[pd.DataFrame] = []
        for f in csv_files:
            try:
                df = pd.read_csv(f)
                df["_source_csv"] = str(f)
                frames.append(df)
            except Exception as exc:
                self.log_text.append(f"Visualization: failed to load {f}: {exc}")
        if not frames:
            QMessageBox.warning(self, "Load failed", "Could not load any CSV data.")
            return
        combined = pd.concat(frames, ignore_index=True)
        if "unix_time" in combined.columns:
            combined = combined.sort_values("unix_time").reset_index(drop=True)
        self._viz_df = combined
        self._viz_csv_path = str(csv_files[0]) if len(csv_files) == 1 else str(csv_files[0].parent)
        self._viz_segments = []
        self._viz_pending_indices = []
        self._refresh_viz_stats()
        self._refresh_viz_segments_list()
        self._populate_viz_channel_combo()
        if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
            self._refresh_map()

    _VIZ_STANDARD_COLS = frozenset({
        "frame_filename", "timestamp_iso", "unix_time", "lat", "lon", "alt",
        "video_filename", "frame_index", "_source_csv",
    })

    def _populate_viz_channel_combo(self) -> None:
        """Refresh the Track Coloring channel dropdown from the current master.csv.

        Always offers "None" and, when an altitude source is configured, "Altitude".
        Any numeric column in _viz_df that isn't a standard master.csv field is
        treated as a sensor channel and added in column order.
        """
        combo = self.viz_channel_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None")
        if self.navigation_file is not None and self.navigation_file.altitude_source is not None:
            combo.addItem("Altitude")
        if self._viz_df is not None:
            for col in self._viz_df.columns:
                if col not in self._VIZ_STANDARD_COLS and pd.api.types.is_numeric_dtype(self._viz_df[col]):
                    combo.addItem(col)
        idx = combo.findText(current)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)
        # Re-evaluate enabled state without triggering a map refresh
        enabled = (combo.currentText() != "None")
        for w in (self.viz_alt_min_spin, self.viz_alt_max_spin,
                  self.viz_alt_reset_button, self.viz_alt_recolor_button,
                  self.viz_export_track_button, self.viz_export_geojson_button,
                  self.viz_color_scale_combo):
            w.setEnabled(enabled)
        self.viz_alt_key_widget.setVisible(enabled)

    def _viz_export_track_csv(self) -> None:
        """Export the full GPS trackline with the active sensor channel as a CSV.

        Writes columns: unix_time, lat, lon, <channel> — suitable for QGIS
        'color by value' import.  Sensor values are the same interpolated array
        that drives the on-screen colour gradient.
        """
        channel = self.viz_channel_combo.currentText()
        if channel == "None" or self.navigation_file is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Select a channel in Track Coloring first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Trackline CSV",
            filter="CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return

        try:
            lat_src = self.navigation_file.latitude_source
            lon_src = self.navigation_file.longitude_source
            lat_df  = SensorService.load_time_value_dataframe(lat_src).sort_values("unix_time")
            lon_df  = SensorService.load_time_value_dataframe(lon_src).sort_values("unix_time")
            nav_times = lat_df["unix_time"].to_numpy(dtype=float)
            nav_lats  = lat_df["value"].to_numpy(dtype=float)
            nav_lons  = SensorService.interpolate_series(
                lat_df["unix_time"], lon_df["unix_time"], lon_df["value"]
            )
            mask = np.isfinite(nav_lats) & np.isfinite(nav_lons)
            nav_times = nav_times[mask]
            nav_lats  = nav_lats[mask]
            nav_lons  = nav_lons[mask]

            if channel == "Altitude" and self.navigation_file.altitude_source is not None:
                alt_df   = SensorService.load_time_value_dataframe(
                    self.navigation_file.altitude_source
                )
                sensor_v = SensorService.interpolate_series(
                    lat_df["unix_time"], alt_df["unix_time"], alt_df["value"]
                )[mask]
            elif self._viz_df is not None and channel in self._viz_df.columns:
                frame_times = self._viz_df["unix_time"].to_numpy(dtype=float)
                frame_vals  = pd.to_numeric(
                    self._viz_df[channel], errors="coerce"
                ).to_numpy(dtype=float)
                sensor_v = SensorService.interpolate_series(
                    pd.Series(nav_times), pd.Series(frame_times), pd.Series(frame_vals)
                )
            else:
                QMessageBox.warning(self, "Export failed", f"Channel '{channel}' not available.")
                return

            out = pd.DataFrame({
                "unix_time": nav_times,
                "lat":       nav_lats,
                "lon":       nav_lons,
                channel:     sensor_v,
            })
            out.to_csv(path, index=False)
            self.log_text.append(
                f"Trackline exported: {len(out):,} pts → {Path(path).name}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ------------------------------------------------------------------
    # Sensor Raster handlers
    # ------------------------------------------------------------------

    def _load_sensor_rasters(self) -> None:
        """Load sensor rasters directly from the configured nav + sensor sources.

        Requires navigation and at least one sensor channel to be configured on
        the Inputs tab.  No pipeline run is needed — sensor values are preserved
        at their native timestamps, and each reading is geolocated by interpolating
        the nav data to that moment.
        """
        if self.navigation_file is None:
            QMessageBox.warning(self, "Navigation required",
                                "Configure navigation sources on the Inputs tab first.")
            return
        active = [sf for sf in self.sensor_files if sf.channels]
        if not active:
            QMessageBox.warning(self, "Sensors required",
                                "Add at least one sensor file with channels on the Inputs tab first.")
            return
        try:
            self._raster_df = SensorService.build_sensor_raster_dataframe(
                self.navigation_file, active
            )
        except Exception as exc:
            QMessageBox.critical(self, "Raster load failed", str(exc))
            return

        n = len(self._raster_df)
        channels = [c for c in self._raster_df.columns
                    if c not in ("unix_time", "lat", "lon", "alt")]
        dur = ""
        if n > 0:
            t0 = self._raster_df["unix_time"].iloc[0]
            t1 = self._raster_df["unix_time"].iloc[-1]
            mins = (t1 - t0) / 60.0
            dur = f" · {mins:.1f} min"
        self.raster_status_label.setText(
            f"{n:,} readings · {len(channels)} channel{'s' if len(channels) != 1 else ''}{dur}"
        )
        self.raster_status_label.setStyleSheet("color: #27ae60; font-size: 11px;")
        self._populate_raster_channel_combo()

        # Auto-select the first sensor channel so color appears immediately.
        combo = self.raster_channel_combo
        if combo.currentText() == "None" and combo.count() > 1:
            combo.blockSignals(True)
            combo.setCurrentIndex(1)
            combo.blockSignals(False)
            self._set_raster_controls_enabled(True)

        # Reset the colour range to the data extent of the auto-selected channel.
        self._on_raster_range_reset()

    def _clear_sensor_rasters(self) -> None:
        """Discard the loaded sensor raster data and reset the raster controls."""
        self._raster_df = None
        self.raster_status_label.setText("No raster loaded.")
        self.raster_status_label.setStyleSheet("color: gray; font-size: 11px;")
        self._populate_raster_channel_combo()
        if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
            self._refresh_map()

    def _populate_raster_channel_combo(self) -> None:
        """Rebuild the raster channel dropdown from the current _raster_df columns."""
        combo = self.raster_channel_combo
        current = combo.currentText()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("None")
        if self._raster_df is not None:
            for col in self._raster_df.columns:
                if col not in ("unix_time", "lat", "lon", "alt"):
                    combo.addItem(col)
        idx = combo.findText(current)
        combo.setCurrentIndex(max(0, idx))
        combo.blockSignals(False)
        self._set_raster_controls_enabled(combo.currentText() != "None")

    def _set_raster_controls_enabled(self, enabled: bool) -> None:
        """Enable/disable raster color controls based on whether a channel is active."""
        for w in (self.raster_min_spin, self.raster_max_spin,
                  self.raster_reset_button, self.raster_recolor_button,
                  self.raster_scale_combo, self.raster_export_map_button,
                  self.raster_export_csv_button, self.raster_export_geojson_button,
                  self.raster_export_analysis_button):
            w.setEnabled(enabled)

    def _on_raster_channel_changed(self, text: str) -> None:
        """Handle raster channel selection change: update controls and refresh map."""
        enabled = (text != "None")
        self._set_raster_controls_enabled(enabled)
        if enabled:
            self._on_raster_range_reset()
        else:
            if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
                self._refresh_map()

    def _on_raster_scale_changed(self, _text: str) -> None:
        """Re-render the raster trackline when the linear/log scale toggle changes."""
        if self.raster_channel_combo.currentText() != "None":
            if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
                self._refresh_map()

    def _on_raster_range_reset(self) -> None:
        """Reset the raster colour scale to the actual data min/max."""
        channel = self.raster_channel_combo.currentText()
        arr = self._get_raster_array_for_channel(channel)
        if arr is not None and len(arr) > 0:
            finite = arr[np.isfinite(arr)]
            if len(finite) > 0:
                self.raster_min_spin.blockSignals(True)
                self.raster_max_spin.blockSignals(True)
                self.raster_min_spin.setValue(float(finite.min()))
                self.raster_max_spin.setValue(float(finite.max()))
                self.raster_min_spin.blockSignals(False)
                self.raster_max_spin.blockSignals(False)
        self._refresh_raster_key_labels()
        if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
            self._refresh_map()

    def _on_raster_recolor(self) -> None:
        """Apply new raster spinbox bounds and refresh the map."""
        if self.raster_channel_combo.currentText() != "None":
            self._refresh_raster_key_labels()
            if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
                self._refresh_map()

    def _get_raster_array_for_channel(self, channel: str) -> np.ndarray | None:
        """Return raw raster values for the named channel."""
        if channel == "None" or self._raster_df is None:
            return None
        if channel in self._raster_df.columns:
            return pd.to_numeric(
                self._raster_df[channel], errors="coerce"
            ).to_numpy(dtype=float)
        return None

    def _get_raster_channel_label(self, channel: str) -> str:
        """Return a display label (name + units) for the named raster channel."""
        for sf in self.sensor_files:
            for ch in sf.channels:
                if ch.display_name == channel or ch.source_column == channel:
                    return f"{ch.display_name} ({ch.units})" if ch.units else ch.display_name
        return channel

    def _get_raster_channel_units(self, channel: str) -> str:
        """Return the units string for a raster channel."""
        for sf in self.sensor_files:
            for ch in sf.channels:
                if ch.display_name == channel or ch.source_column == channel:
                    return ch.units or ""
        return ""

    def _refresh_raster_key_labels(self) -> None:
        """Sync the color key min/max labels when raster mode is active."""
        lo = self.raster_min_spin.value()
        hi = self.raster_max_spin.value()
        units = self._get_raster_channel_units(self.raster_channel_combo.currentText())
        self.viz_alt_key_max_label.setText(f"{hi:.4g}\n{units}")
        self.viz_alt_key_min_label.setText(f"{lo:.4g}\n{units}")

    def _raster_export_csv(self) -> None:
        """Export the full sensor raster DataFrame as a CSV file.

        Writes unix_time, lat, lon, alt, and all sensor channel columns so the
        file can be imported in QGIS (or similar tools) for spatial color styling.
        """
        if self._raster_df is None or len(self._raster_df) == 0:
            QMessageBox.information(self, "Nothing to export",
                                    "Load sensor rasters first.")
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Sensor Raster CSV",
            filter="CSV Files (*.csv);;All Files (*)",
        )
        if not path:
            return
        try:
            self._raster_df.to_csv(path, index=False)
            self.log_text.append(
                f"Raster CSV exported: {len(self._raster_df):,} rows → {Path(path).name}"
            )
        except Exception as exc:
            QMessageBox.critical(self, "Export failed", str(exc))

    # ------------------------------------------------------------------
    # QGIS GeoJSON export (vector trackline with sensor values)
    # ------------------------------------------------------------------

    def _build_trackline_geojson(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        times: np.ndarray,
        sensor_v: np.ndarray,
        channel: str,
        sensor_label: str,
        v_min: float | None = None,
        v_max: float | None = None,
    ) -> dict:
        """Build a GeoJSON FeatureCollection for a sensor-colored trackline.

        Each consecutive pair of GPS points becomes one LineString Feature with
        three value properties:
          sensor_value  — raw reading, never modified.
          sensor_style  — raw value hard-clipped to [v_min, v_max] in original
                          units.  Use this for QGIS Graduated symbology: outliers
                          outside the bounds share the end color instead of
                          compressing the rest of the ramp.
          value_norm    — sensor_style rescaled to 0–1.

        The FeatureCollection includes a 'metadata' property recording the export
        bounds so QGIS can be configured to match the on-screen color scale.

        Args:
            lons:         Longitude array (decimal degrees, WGS84).
            lats:         Latitude array, same length as lons.
            times:        Unix timestamp array, same length as lons.
            sensor_v:     Sensor value at each GPS point, same length as lons.
            channel:      Column name of the sensor channel.
            sensor_label: Human-readable label string.
            v_min:        Low bound for color normalization (red end).
                          Auto-derived from finite data values when None.
            v_max:        High bound for color normalization (blue end).
                          Auto-derived from finite data values when None.

        Returns:
            A GeoJSON-compatible dict ready to pass to json.dump().
        """
        # Resolve bounds from data when not supplied or degenerate.
        finite = sensor_v[np.isfinite(sensor_v)] if len(sensor_v) > 0 else np.array([])
        if v_min is None or v_max is None or abs((v_max or 0) - (v_min or 0)) < 1e-9:
            v_min = float(finite.min()) if len(finite) > 0 else 0.0
            v_max = float(finite.max()) if len(finite) > 0 else 1.0
            if abs(v_max - v_min) < 1e-9:
                v_max = v_min + 1.0

        span = v_max - v_min  # guaranteed > 0

        features = []
        n = len(lons)
        for i in range(n - 1):
            if not (np.isfinite(lons[i]) and np.isfinite(lats[i])
                    and np.isfinite(lons[i + 1]) and np.isfinite(lats[i + 1])):
                continue
            sv0 = float(sensor_v[i])     if (i < len(sensor_v)     and np.isfinite(sensor_v[i]))     else None
            sv1 = float(sensor_v[i + 1]) if (i + 1 < len(sensor_v) and np.isfinite(sensor_v[i + 1])) else None
            if sv0 is not None and sv1 is not None:
                seg_val = (sv0 + sv1) / 2.0
            else:
                seg_val = sv0 if sv0 is not None else sv1

            if seg_val is not None:
                style_val = round(max(v_min, min(v_max, seg_val)), 6)
                norm_val  = round((style_val - v_min) / span, 6)
            else:
                style_val = None
                norm_val  = None

            t_mid = float(times[i]) if i < len(times) else None
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [
                        [float(lons[i]),     float(lats[i])],
                        [float(lons[i + 1]), float(lats[i + 1])],
                    ],
                },
                "properties": {
                    "sensor_value": round(seg_val, 6) if seg_val is not None else None,
                    "sensor_style": style_val,
                    "value_norm":   norm_val,
                    "channel":      channel,
                    "unix_time":    t_mid,
                },
            })
        return {
            "type": "FeatureCollection",
            "name": f"trackline_{channel}",
            "crs": {
                "type": "name",
                "properties": {"name": "urn:ogc:def:crs:EPSG::4326"},
            },
            "metadata": {
                "channel":      channel,
                "sensor_label": sensor_label,
                "v_min":        round(v_min, 6),
                "v_max":        round(v_max, 6),
            },
            "features": features,
        }

    def _viz_export_qgis_geojson(self) -> None:
        """Export the Track Coloring trackline as a QGIS-ready GeoJSON file.

        Builds the same sensor-interpolated GPS track shown on-screen and writes
        one LineString Feature per consecutive GPS point pair.  Load in QGIS with
        Layer → Add Vector Layer, then style with Graduated symbology on
        'sensor_value' to reproduce the color spectrum.
        """
        channel = self.viz_channel_combo.currentText()
        if channel == "None" or self.navigation_file is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Select a sensor channel in Track Coloring first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export QGIS GeoJSON",
            filter="GeoJSON (*.geojson);;All Files (*)",
        )
        if not path:
            return

        import json

        try:
            lat_src  = self.navigation_file.latitude_source
            lon_src  = self.navigation_file.longitude_source
            lat_df   = SensorService.load_time_value_dataframe(lat_src).sort_values("unix_time")
            lon_df   = SensorService.load_time_value_dataframe(lon_src).sort_values("unix_time")
            nav_times = lat_df["unix_time"].to_numpy(dtype=float)
            nav_lats  = lat_df["value"].to_numpy(dtype=float)
            nav_lons  = SensorService.interpolate_series(
                lat_df["unix_time"], lon_df["unix_time"], lon_df["value"]
            )
            mask = np.isfinite(nav_lats) & np.isfinite(nav_lons)
            nav_times = nav_times[mask]
            nav_lats  = nav_lats[mask]
            nav_lons  = nav_lons[mask]

            if channel == "Altitude" and self.navigation_file.altitude_source is not None:
                alt_df   = SensorService.load_time_value_dataframe(
                    self.navigation_file.altitude_source
                )
                sensor_v = SensorService.interpolate_series(
                    lat_df["unix_time"], alt_df["unix_time"], alt_df["value"]
                )[mask]
            elif self._viz_df is not None and channel in self._viz_df.columns:
                frame_times = self._viz_df["unix_time"].to_numpy(dtype=float)
                frame_vals  = pd.to_numeric(
                    self._viz_df[channel], errors="coerce"
                ).to_numpy(dtype=float)
                sensor_v = SensorService.interpolate_series(
                    pd.Series(nav_times), pd.Series(frame_times), pd.Series(frame_vals)
                )
            else:
                QMessageBox.warning(self, "Export failed",
                                    f"Channel '{channel}' not available.")
                return

            sensor_label = self._get_channel_label(channel)
            v_min = self.viz_alt_min_spin.value()
            v_max = self.viz_alt_max_spin.value()
            geojson = self._build_trackline_geojson(
                nav_lons, nav_lats, nav_times, sensor_v, channel, sensor_label,
                v_min=v_min, v_max=v_max,
            )
            with open(path, "w") as f:
                json.dump(geojson, f, separators=(",", ":"))

            meta  = geojson["metadata"]
            lo, hi = meta["v_min"], meta["v_max"]
            n = len(geojson["features"])
            msg = (
                f"Saved {n:,} track segments to:\n{Path(path).name}\n\n"
                f"Color bounds: {lo:.4g} – {hi:.4g}  ({sensor_label})\n\n"
                f"In QGIS:\n"
                f"  1. Layer → Add Layer → Add Vector Layer → select the .geojson\n"
                f"  2. Layer Properties → Symbology → Graduated\n"
                f"  3. Column: sensor_style  (outliers clipped to your bounds)\n"
                f"     Range: {lo:.4g} – {hi:.4g}    ← set these as Min/Max in QGIS\n"
                f"     'sensor_value' = untouched raw data if you need it"
            )
            self.log_text.append(msg)
            QMessageBox.information(self, "QGIS GeoJSON exported", msg)
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Export failed",
                                 f"{exc}\n\n{traceback.format_exc()[-600:]}")

    def _raster_export_qgis_geojson(self) -> None:
        """Export the Sensor Rasters trackline as a QGIS-ready GeoJSON file.

        Uses _raster_df (sensor-native timestamps, GPS-geolocated) to build one
        LineString Feature per consecutive GPS point pair.  Load in QGIS and
        style with Graduated symbology on 'sensor_value'.
        """
        if self._raster_df is None or len(self._raster_df) == 0:
            QMessageBox.information(self, "Nothing to export",
                                    "Load sensor rasters first.")
            return

        channel = self.raster_channel_combo.currentText()
        if channel == "None":
            QMessageBox.information(self, "No channel selected",
                                    "Select a channel in the Sensor Rasters panel first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export QGIS GeoJSON",
            filter="GeoJSON (*.geojson);;All Files (*)",
        )
        if not path:
            return

        import json

        try:
            rdf      = self._raster_df
            mask     = np.isfinite(rdf["lat"].to_numpy(dtype=float)) & \
                       np.isfinite(rdf["lon"].to_numpy(dtype=float))
            r_lats   = rdf["lat"].to_numpy(dtype=float)[mask]
            r_lons   = rdf["lon"].to_numpy(dtype=float)[mask]
            r_times  = rdf["unix_time"].to_numpy(dtype=float)[mask]
            sensor_v = pd.to_numeric(rdf[channel], errors="coerce").to_numpy(dtype=float)[mask]

            sensor_label = self._get_raster_channel_label(channel)
            v_min = self.raster_min_spin.value()
            v_max = self.raster_max_spin.value()
            geojson = self._build_trackline_geojson(
                r_lons, r_lats, r_times, sensor_v, channel, sensor_label,
                v_min=v_min, v_max=v_max,
            )
            with open(path, "w") as f:
                json.dump(geojson, f, separators=(",", ":"))

            meta  = geojson["metadata"]
            lo, hi = meta["v_min"], meta["v_max"]
            n = len(geojson["features"])
            msg = (
                f"Saved {n:,} track segments to:\n{Path(path).name}\n\n"
                f"Color bounds: {lo:.4g} – {hi:.4g}  ({sensor_label})\n\n"
                f"In QGIS:\n"
                f"  1. Layer → Add Layer → Add Vector Layer → select the .geojson\n"
                f"  2. Layer Properties → Symbology → Graduated\n"
                f"  3. Column: sensor_style  (outliers clipped to your bounds)\n"
                f"     Range: {lo:.4g} – {hi:.4g}    ← set these as Min/Max in QGIS\n"
                f"     'sensor_value' = untouched raw data if you need it"
            )
            self.log_text.append(msg)
            QMessageBox.information(self, "QGIS GeoJSON exported", msg)
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Export failed",
                                 f"{exc}\n\n{traceback.format_exc()[-600:]}")

    # ------------------------------------------------------------------
    # Sensor analysis export (IDW raster, anomaly, confidence, points)
    # ------------------------------------------------------------------

    def _export_sensor_analysis(self) -> None:
        """Show the sensor analysis export dialog and dispatch the computation."""
        if self._raster_df is None or len(self._raster_df) == 0:
            QMessageBox.information(self, "Nothing to export", "Load sensor rasters first.")
            return
        channel = self.raster_channel_combo.currentText()
        if channel == "None":
            QMessageBox.information(self, "No channel selected",
                                    "Select a sensor channel in the Sensor Rasters panel first.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Export Sensor Analysis — {channel}")
        layout = QVBoxLayout(dlg)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        cell_mode = QComboBox()
        cell_mode.addItems(["Dynamic (from track spacing)", "Fixed"])
        form.addRow("Cell size:", cell_mode)

        cell_spin = QDoubleSpinBox()
        cell_spin.setRange(0.1, 100.0)
        cell_spin.setValue(2.0)
        cell_spin.setSuffix(" m")
        cell_spin.setEnabled(False)
        cell_spin.setToolTip("Grid cell side length for the interpolated rasters.")
        cell_mode.currentTextChanged.connect(lambda t: cell_spin.setEnabled(t == "Fixed"))
        form.addRow("Fixed cell size:", cell_spin)

        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0.5, 200.0)
        conf_spin.setValue(5.0)
        conf_spin.setSuffix(" m")
        conf_spin.setToolTip(
            "Raster cells farther than this distance from any sensor reading\n"
            "are set to nodata — prevents misleading extrapolation."
        )
        form.addRow("Confidence radius:", conf_spin)

        layout.addLayout(form)

        dir_row = QHBoxLayout()
        dir_edit = QLineEdit()
        dir_edit.setPlaceholderText("Output folder…")
        dir_btn = QPushButton("Browse…")
        dir_btn.clicked.connect(
            lambda: dir_edit.setText(
                QFileDialog.getExistingDirectory(dlg, "Select Output Folder") or dir_edit.text()
            )
        )
        dir_row.addWidget(dir_edit)
        dir_row.addWidget(dir_btn)
        layout.addLayout(dir_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        layout.addWidget(btns)

        if dlg.exec() != QDialog.Accepted:
            return

        out_dir = dir_edit.text().strip()
        if not out_dir:
            QMessageBox.warning(self, "No output folder", "Please select an output folder.")
            return

        fixed_cell = cell_spin.value() if cell_mode.currentText() == "Fixed" else None
        try:
            self._run_sensor_analysis_export(
                channel=channel,
                out_dir=Path(out_dir),
                fixed_cell_m=fixed_cell,
                conf_radius_m=conf_spin.value(),
            )
        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Analysis export failed",
                                 f"{exc}\n\n{traceback.format_exc()[-800:]}")

    def _run_sensor_analysis_export(
        self,
        channel: str,
        out_dir: Path,
        fixed_cell_m: float | None,
        conf_radius_m: float,
    ) -> None:
        """Compute and write the four-file sensor analysis bundle.

        Outputs (all in out_dir):
          <channel>_points.geojson  — point layer with anomaly/zscore/hotspot fields
          <channel>_IDW.tif         — IDW-interpolated sensor raster
          <channel>_confidence.tif  — distance-to-nearest-sample raster
          <channel>_anomaly.tif     — IDW raster minus global survey median

        The grid is computed in local metre space (equirectangular, centred at the
        data centroid) so IDW weights are isotropic, then written to WGS84 GeoTIFFs.

        Args:
            channel:       Sensor column name from _raster_df.
            out_dir:       Destination directory (created if absent).
            fixed_cell_m:  Fixed cell size in metres; None → derive from track spacing.
            conf_radius_m: Max distance (m) from a sample; cells beyond → nodata.
        """
        import json
        import rasterio
        from rasterio.transform import from_bounds
        from scipy.spatial import cKDTree

        NODATA    = np.float32(-9999.0)
        MAX_DIM   = 2000
        IDW_POWER = 2
        IDW_K     = 16

        # --- Filter to finite positions + finite sensor values ---
        rdf  = self._raster_df
        geo_ok = np.isfinite(rdf["lat"].to_numpy(dtype=float)) & np.isfinite(rdf["lon"].to_numpy(dtype=float))
        lats_raw = rdf["lat"].to_numpy(dtype=float)[geo_ok]
        lons_raw = rdf["lon"].to_numpy(dtype=float)[geo_ok]
        t_raw    = rdf["unix_time"].to_numpy(dtype=float)[geo_ok]
        sv_raw   = pd.to_numeric(rdf[channel], errors="coerce").to_numpy(dtype=float)[geo_ok]

        sv_ok = np.isfinite(sv_raw)
        lats = lats_raw[sv_ok];  lons = lons_raw[sv_ok]
        times = t_raw[sv_ok];    sensor_v = sv_raw[sv_ok]

        if len(sensor_v) < 4:
            raise ValueError(
                f"Need ≥ 4 finite {channel} readings; found {len(sensor_v)}."
            )

        sensor_label = self._get_raster_channel_label(channel)
        v_min = self.raster_min_spin.value()
        v_max = self.raster_max_spin.value()
        if abs(v_max - v_min) < 1e-9:
            finite = sensor_v[np.isfinite(sensor_v)]
            v_min, v_max = float(finite.min()), float(finite.max())
            if abs(v_max - v_min) < 1e-9:
                v_max = v_min + 1.0

        # --- Project to local metres ---
        ref_lat = float(np.mean(lats))
        ref_lon = float(np.mean(lons))
        cos_lat = np.cos(np.radians(ref_lat))
        M_PER_DEG = 111_319.5
        x = (lons - ref_lon) * cos_lat * M_PER_DEG
        y = (lats - ref_lat) * M_PER_DEG

        # --- Cell size ---
        if fixed_cell_m is not None:
            cell_m = float(fixed_cell_m)
            cell_mode_str = f"fixed {cell_m:.3g} m"
        else:
            spacings = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
            spacings = spacings[spacings > 0.01]
            raw = float(np.median(spacings)) / 2.0 if len(spacings) else 2.0
            cell_m = max(0.5, min(raw, 20.0))
            cell_mode_str = f"dynamic {cell_m:.3g} m (median track spacing ÷ 2)"

        # --- Grid extents with padding ---
        pad   = cell_m * 3
        xmin, xmax = x.min() - pad, x.max() + pad
        ymin, ymax = y.min() - pad, y.max() + pad
        ncols = max(2, int(np.ceil((xmax - xmin) / cell_m)))
        nrows = max(2, int(np.ceil((ymax - ymin) / cell_m)))

        if max(ncols, nrows) > MAX_DIM:
            scale  = max(ncols, nrows) / MAX_DIM
            cell_m = cell_m * scale
            ncols  = max(2, int(np.ceil((xmax - xmin) / cell_m)))
            nrows  = max(2, int(np.ceil((ymax - ymin) / cell_m)))
            cell_mode_str += f" → auto-scaled to {cell_m:.3g} m (capped at {MAX_DIM}²)"
            self.log_text.append(
                f"  Note: {channel} raster capped at {MAX_DIM}² — cell size scaled to {cell_m:.3g} m"
            )

        # Grid columns W→E, rows N→S (row 0 = northernmost, matching GeoTIFF convention)
        xi = np.linspace(xmin + cell_m / 2, xmax - cell_m / 2, ncols)
        yi = np.linspace(ymax - cell_m / 2, ymin + cell_m / 2, nrows)
        gx, gy = np.meshgrid(xi, yi)
        grid_pts   = np.column_stack([gx.ravel(), gy.ravel()])
        sample_pts = np.column_stack([x, y])

        # --- IDW interpolation ---
        K    = min(IDW_K, len(sensor_v))
        tree = cKDTree(sample_pts)
        d_k, idx_k = tree.query(grid_pts, k=K)
        d_k = np.maximum(d_k, 1e-3)
        w   = 1.0 / (d_k ** IDW_POWER)
        idw_flat = np.sum(w * sensor_v[idx_k], axis=1) / np.sum(w, axis=1)
        idw_grid = idw_flat.reshape(nrows, ncols).astype(np.float32)

        # --- Confidence raster ---
        d_nearest, _ = tree.query(grid_pts, k=1)
        conf_grid = d_nearest.reshape(nrows, ncols).astype(np.float32)

        # --- Anomaly raster ---
        global_median = float(np.nanmedian(sensor_v))
        anom_grid = (idw_grid - global_median).astype(np.float32)

        # Apply nodata beyond confidence radius
        far = conf_grid > conf_radius_m
        idw_out  = idw_grid.copy();  idw_out[far]  = NODATA
        anom_out = anom_grid.copy(); anom_out[far] = NODATA

        # --- GeoTIFF affine transform (WGS84) ---
        lon_w = ref_lon + xmin / (cos_lat * M_PER_DEG)
        lon_e = ref_lon + xmax / (cos_lat * M_PER_DEG)
        lat_s = ref_lat + ymin / M_PER_DEG
        lat_n = ref_lat + ymax / M_PER_DEG
        transform = from_bounds(lon_w, lat_s, lon_e, lat_n, ncols, nrows)
        raster_kw = dict(driver="GTiff", dtype="float32", width=ncols, height=nrows,
                         count=1, crs="EPSG:4326", transform=transform, nodata=NODATA)

        out_dir.mkdir(parents=True, exist_ok=True)
        safe_ch = "".join(c if (c.isalnum() or c == "_") else "_" for c in channel)
        idw_path  = out_dir / f"{safe_ch}_IDW.tif"
        conf_path = out_dir / f"{safe_ch}_confidence.tif"
        anom_path = out_dir / f"{safe_ch}_anomaly.tif"
        pts_path  = out_dir / f"{safe_ch}_points.geojson"

        with rasterio.open(str(idw_path), "w", **raster_kw) as dst:
            dst.write(idw_out, 1)
        with rasterio.open(str(conf_path), "w", **{**raster_kw, "nodata": None}) as dst:
            dst.write(conf_grid, 1)
        with rasterio.open(str(anom_path), "w", **raster_kw) as dst:
            dst.write(anom_out, 1)

        # --- Enhanced point layer ---
        global_std = float(np.nanstd(sensor_v))

        # Cumulative distance along track (metres)
        cum_dist    = np.zeros(len(x))
        seg_lengths = np.sqrt(np.diff(x) ** 2 + np.diff(y) ** 2)
        cum_dist[1:] = np.cumsum(seg_lengths)

        # Distance-based rolling mean (window = 5× cell size, min 5 m)
        roll_m = max(cell_m * 5.0, 5.0)
        rolling_mean = np.empty(len(sensor_v))
        for i in range(len(sensor_v)):
            lo = int(np.searchsorted(cum_dist, cum_dist[i] - roll_m / 2))
            hi = int(np.searchsorted(cum_dist, cum_dist[i] + roll_m / 2, side="right"))
            rolling_mean[i] = float(np.mean(sensor_v[lo:hi]))

        anom_v   = sensor_v - global_median
        zscore_v = anom_v / global_std if global_std > 1e-9 else np.zeros_like(anom_v)
        style_v  = np.clip(sensor_v, v_min, v_max)

        features = [
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [round(float(lons[i]), 8), round(float(lats[i]), 8)],
                },
                "properties": {
                    channel:                   round(float(sensor_v[i]), 6),
                    f"{channel}_style":        round(float(style_v[i]), 6),
                    f"{channel}_anomaly":      round(float(anom_v[i]), 6),
                    f"{channel}_zscore":       round(float(zscore_v[i]), 4),
                    f"{channel}_rolling_mean": round(float(rolling_mean[i]), 6),
                    "hotspot_flag":            int(zscore_v[i] >  2.0),
                    "coldspot_flag":           int(zscore_v[i] < -2.0),
                    "dist_along_track_m":      round(float(cum_dist[i]), 2),
                    "unix_time":               round(float(times[i]), 3),
                },
            }
            for i in range(len(lons))
        ]

        geojson = {
            "type": "FeatureCollection",
            "name": f"{safe_ch}_analysis_points",
            "crs": {"type": "name", "properties": {"name": "urn:ogc:def:crs:EPSG::4326"}},
            "metadata": {
                "channel":          channel,
                "sensor_label":     sensor_label,
                "n_points":         len(features),
                "global_median":    round(global_median, 6),
                "global_std":       round(global_std, 6),
                "v_min":            round(v_min, 6),
                "v_max":            round(v_max, 6),
                "cell_size_m":      round(cell_m, 4),
                "cell_mode":        cell_mode_str,
                "conf_radius_m":    conf_radius_m,
                "rolling_window_m": round(roll_m, 2),
            },
            "features": features,
        }

        with open(str(pts_path), "w") as f:
            json.dump(geojson, f, separators=(",", ":"))

        msg = (
            f"Sensor analysis exported — {channel} ({sensor_label})\n"
            f"  Points:         {len(features):,}\n"
            f"  Raster:         {nrows} × {ncols} px  ·  {cell_mode_str}\n"
            f"  Global median:  {global_median:.4g}  ·  std: {global_std:.4g}\n"
            f"  Conf. radius:   {conf_radius_m:.1f} m\n\n"
            f"Files written to {out_dir}:\n"
            f"  {pts_path.name}\n"
            f"  {idw_path.name}\n"
            f"  {conf_path.name}\n"
            f"  {anom_path.name}\n\n"
            f"In QGIS:\n"
            f"  {idw_path.name}  → Graduated, range {v_min:.4g} – {v_max:.4g}\n"
            f"  {anom_path.name} → Diverging color ramp centered at 0\n"
            f"  {conf_path.name} → mask / transparency where > {conf_radius_m:.1f} m\n"
            f"  {pts_path.name}  → Graduate on '{channel}_style' or 'hotspot_flag'"
        )
        self.log_text.append(msg)
        QMessageBox.information(self, "Sensor analysis exported", msg)

    # ------------------------------------------------------------------
    # Map export (matplotlib, print-quality)
    # ------------------------------------------------------------------

    @staticmethod
    def _map_scale_bar(ax, ref_lat: float) -> None:
        """Draw a geographic scale bar in the lower-left corner of ax.

        Picks a round-number bar length that is roughly 15 % of the current
        view width, converts to longitude degrees for placement, and annotates
        in km (or m when < 1 km).
        """
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        km_per_lon = 111.319 * np.cos(np.radians(ref_lat))

        target_km = (x1 - x0) * km_per_lon * 0.15
        nice = [0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000]
        bar_km = min(nice, key=lambda s: abs(s - target_km))
        bar_deg = bar_km / km_per_lon

        bx  = x0 + (x1 - x0) * 0.05
        by  = y0 + (y1 - y0) * 0.04
        tick_h = (y1 - y0) * 0.006

        ax.plot([bx, bx + bar_deg], [by, by],  color="k", lw=2.5, solid_capstyle="butt", zorder=10)
        for px in (bx, bx + bar_deg):
            ax.plot([px, px], [by - tick_h, by + tick_h], color="k", lw=1.5, zorder=10)

        label = f"{bar_km:.3g} km" if bar_km >= 1.0 else f"{bar_km * 1000:.3g} m"
        ax.text(bx + bar_deg / 2, by + tick_h * 2.5, label,
                ha="center", va="bottom", fontsize=8, zorder=10,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=1))

    @staticmethod
    def _map_north_arrow(ax) -> None:
        """Draw a simple north arrow in the upper-right corner of ax."""
        x0, x1 = ax.get_xlim()
        y0, y1 = ax.get_ylim()
        ax_w   = x1 - x0
        ax_h   = y1 - y0
        cx = x1 - ax_w * 0.05
        cy = y1 - ax_h * 0.07
        dy = ax_h * 0.055
        ax.annotate(
            "", xy=(cx, cy + dy), xytext=(cx, cy - dy),
            arrowprops=dict(arrowstyle="-|>", color="black", lw=1.5),
            zorder=10,
        )
        ax.text(cx, cy + dy * 1.5, "N", ha="center", va="bottom",
                fontsize=9, fontweight="bold", zorder=10)

    def _raster_export_map(self) -> None:
        """Export the sensor raster trackline as a print-quality PDF, PNG, or SVG.

        Builds a matplotlib figure with correct geographic aspect ratio
        (1° lon = cos(lat) × 1° lat), colorbar, scale bar, and north arrow.
        The gray nav trackline is drawn underneath the colored raster trackline
        when nav data is available so full survey coverage is always visible.
        """
        if self._raster_df is None or len(self._raster_df) == 0:
            QMessageBox.information(self, "Nothing to export",
                                    "Load sensor rasters first.")
            return

        channel = self.raster_channel_combo.currentText()
        if channel == "None":
            QMessageBox.information(self, "No channel selected",
                                    "Select a channel in the Sensor Rasters panel first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Raster Map",
            filter="PDF (*.pdf);;PNG 300 dpi (*.png);;SVG (*.svg);;All Files (*)",
        )
        if not path:
            return

        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
            from matplotlib.cm import ScalarMappable
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                 "matplotlib is required for map export.\n"
                                 "Install it with: pip install matplotlib")
            return

        try:
            cmap = LinearSegmentedColormap.from_list("epr_sensor", [
                (0.00, (220 / 255,   0 / 255,   0 / 255)),
                (0.25, (220 / 255, 220 / 255,   0 / 255)),
                (0.50, (  0 / 255, 200 / 255,   0 / 255)),
                (0.75, (  0 / 255, 210 / 255, 210 / 255)),
                (1.00, (  0 / 255,   0 / 255, 210 / 255)),
            ])

            rdf = self._raster_df
            r_lats  = rdf["lat"].to_numpy(dtype=float)
            r_lons  = rdf["lon"].to_numpy(dtype=float)
            mask    = np.isfinite(r_lats) & np.isfinite(r_lons)
            r_lats  = r_lats[mask]
            r_lons  = r_lons[mask]

            sensor_v = pd.to_numeric(
                rdf[channel], errors="coerce"
            ).to_numpy(dtype=float)[mask]

            v_min = self.raster_min_spin.value()
            v_max = self.raster_max_spin.value()
            if abs(v_max - v_min) < 1e-9:
                finite = sensor_v[np.isfinite(sensor_v)]
                if len(finite) > 0:
                    v_min, v_max = float(finite.min()), float(finite.max())
                if abs(v_max - v_min) < 1e-9:
                    v_max = v_min + 1.0
            log_scale = self.raster_scale_combo.currentText() == "Logarithmic"
            sensor_label = self._get_raster_channel_label(channel)

            # Optional nav background track
            nav_bg_lats = nav_bg_lons = np.array([])
            if self.navigation_file is not None:
                try:
                    lat_src = self.navigation_file.latitude_source
                    lon_src = self.navigation_file.longitude_source
                    lat_df  = SensorService.load_time_value_dataframe(lat_src).sort_values("unix_time")
                    lon_df  = SensorService.load_time_value_dataframe(lon_src).sort_values("unix_time")
                    bg_lats = lat_df["value"].to_numpy(dtype=float)
                    bg_lons = SensorService.interpolate_series(
                        lat_df["unix_time"], lon_df["unix_time"], lon_df["value"]
                    )
                    bg_mask = np.isfinite(bg_lats) & np.isfinite(bg_lons)
                    nav_bg_lats = bg_lats[bg_mask]
                    nav_bg_lons = bg_lons[bg_mask]
                except Exception:
                    pass

            all_lats = np.concatenate([a for a in [r_lats, nav_bg_lats] if len(a) > 0]) if (len(r_lats) > 0 or len(nav_bg_lats) > 0) else np.array([0.0])
            all_lons = np.concatenate([a for a in [r_lons, nav_bg_lons] if len(a) > 0]) if (len(r_lons) > 0 or len(nav_bg_lons) > 0) else np.array([0.0])
            ref_lat  = float(np.mean(all_lats))

            fig, ax = plt.subplots(figsize=(11, 8.5))

            # Nav background track (thin gray)
            if len(nav_bg_lons) >= 2:
                ax.plot(nav_bg_lons, nav_bg_lats, "-",
                        color="#cccccc", linewidth=0.7, alpha=0.8,
                        label="Nav track", zorder=2)

            # Colored raster trackline
            if len(r_lons) >= 2:
                if log_scale and v_min > 0 and v_max > 0:
                    norm = LogNorm(vmin=max(v_min, 1e-10), vmax=v_max)
                else:
                    norm = Normalize(vmin=v_min, vmax=v_max)
                pts  = np.array([r_lons, r_lats]).T.reshape(-1, 1, 2)
                segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                seg_vals = (sensor_v[:-1] + sensor_v[1:]) / 2.0
                lc = LineCollection(segs, cmap=cmap, norm=norm,
                                    linewidth=1.5, alpha=0.95, zorder=3)
                lc.set_array(seg_vals)
                ax.add_collection(lc)
                sm = ScalarMappable(cmap=cmap, norm=norm)
                sm.set_array([])
                cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
                cbar.set_label(sensor_label, fontsize=10)
                cbar.ax.tick_params(labelsize=8)
            else:
                ax.plot(r_lons, r_lats, "-", color="#e63946",
                        linewidth=1.0, alpha=0.9, zorder=3)

            # Extent + padding
            lon_pad = (all_lons.max() - all_lons.min()) * 0.06 + 1e-4
            lat_pad = (all_lats.max() - all_lats.min()) * 0.06 + 1e-4
            ax.set_xlim(all_lons.min() - lon_pad, all_lons.max() + lon_pad)
            ax.set_ylim(all_lats.min() - lat_pad, all_lats.max() + lat_pad)

            ax.set_aspect(1.0 / np.cos(np.radians(ref_lat)))
            ax.set_xlabel("Longitude (°)", fontsize=11)
            ax.set_ylabel("Latitude (°)", fontsize=11)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.6)

            self._map_scale_bar(ax, ref_lat)
            self._map_north_arrow(ax)

            # Title: channel name + time range
            n = len(rdf)
            ts_range = ""
            if n > 0:
                t0 = datetime.utcfromtimestamp(float(rdf["unix_time"].iloc[0]))
                t1 = datetime.utcfromtimestamp(float(rdf["unix_time"].iloc[-1]))
                ts_range = f"  |  {t0.strftime('%Y-%m-%d %H:%M')} – {t1.strftime('%H:%M')}"
            ax.set_title(f"Sensor Raster: {sensor_label}{ts_range}", fontsize=12, pad=10)

            if len(nav_bg_lons) >= 2:
                handles, labels_ = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles=handles, labels=labels_,
                              loc="lower right", fontsize=8, framealpha=0.85)

            ext = Path(path).suffix.lower()
            dpi = 300 if ext == ".png" else 150
            plt.tight_layout()
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            self.log_text.append(
                f"Raster map exported: {n:,} pts · channel={channel} → {Path(path).name}"
            )
            QMessageBox.information(self, "Raster map exported",
                                    f"Saved to {Path(path).name}\n{n:,} sensor readings")

        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Export failed",
                                 f"{exc}\n\n{traceback.format_exc()[-800:]}")

    def _viz_export_map(self) -> None:
        """Render the current map to a print-quality PDF, PNG, or SVG file.

        Uses matplotlib so the output has correct geographic aspect ratio
        (1° lon = cos(lat) × 1° lat in physical distance), a scale bar, a
        north arrow, and — when a sensor channel is active — a colorbar.
        """
        if self.navigation_file is None and self._viz_df is None:
            QMessageBox.information(self, "Nothing to export",
                                    "Load data or configure a navigation source first.")
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export Map",
            filter="PDF (*.pdf);;PNG 300 dpi (*.png);;SVG (*.svg);;All Files (*)",
        )
        if not path:
            return

        # Lazy matplotlib import — Agg backend avoids any Qt/display conflicts.
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.collections import LineCollection
            from matplotlib.colors import LinearSegmentedColormap, Normalize, LogNorm
            from matplotlib.cm import ScalarMappable
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                 "matplotlib is required for map export.\n"
                                 "Install it with: pip install matplotlib")
            return

        try:
            # ---- Build the custom red → yellow → green → cyan → blue colormap ----
            cmap = LinearSegmentedColormap.from_list("epr_sensor", [
                (0.00, (220 / 255,   0 / 255,   0 / 255)),
                (0.25, (220 / 255, 220 / 255,   0 / 255)),
                (0.50, (  0 / 255, 200 / 255,   0 / 255)),
                (0.75, (  0 / 255, 210 / 255, 210 / 255)),
                (1.00, (  0 / 255,   0 / 255, 210 / 255)),
            ])

            # ---- Load GPS trackline ----
            nav_lats = nav_lons = nav_times = np.array([])
            nav_alt  = None
            if self.navigation_file is not None:
                lat_src = self.navigation_file.latitude_source
                lon_src = self.navigation_file.longitude_source
                lat_df  = SensorService.load_time_value_dataframe(lat_src).sort_values("unix_time")
                lon_df  = SensorService.load_time_value_dataframe(lon_src).sort_values("unix_time")
                nav_times_raw = lat_df["unix_time"].to_numpy(dtype=float)
                nav_lats_raw  = lat_df["value"].to_numpy(dtype=float)
                nav_lons_raw  = SensorService.interpolate_series(
                    lat_df["unix_time"], lon_df["unix_time"], lon_df["value"]
                )
                mask      = np.isfinite(nav_lats_raw) & np.isfinite(nav_lons_raw)
                nav_lats  = nav_lats_raw[mask]
                nav_lons  = nav_lons_raw[mask]
                nav_times = nav_times_raw[mask]
                if self.navigation_file.altitude_source is not None:
                    try:
                        alt_df  = SensorService.load_time_value_dataframe(
                            self.navigation_file.altitude_source
                        )
                        nav_alt = SensorService.interpolate_series(
                            lat_df["unix_time"], alt_df["unix_time"], alt_df["value"]
                        )[mask]
                    except Exception:
                        pass

            # ---- Sensor values for colouring ----
            channel      = self.viz_channel_combo.currentText()
            log_scale    = self.viz_color_scale_combo.currentText() == "Logarithmic"
            v_min        = self.viz_alt_min_spin.value()
            v_max        = self.viz_alt_max_spin.value()
            sensor_v     = None
            sensor_label = ""

            if channel == "Altitude" and nav_alt is not None:
                sensor_v     = nav_alt
                sensor_label = "Altitude (m)"
            elif (channel not in ("None", "Altitude")
                  and self._viz_df is not None
                  and channel in self._viz_df.columns
                  and len(nav_times) > 0):
                frame_times = self._viz_df["unix_time"].to_numpy(dtype=float)
                frame_vals  = pd.to_numeric(
                    self._viz_df[channel], errors="coerce"
                ).to_numpy(dtype=float)
                sensor_v     = SensorService.interpolate_series(
                    pd.Series(nav_times),
                    pd.Series(frame_times),
                    pd.Series(frame_vals),
                )
                sensor_label = self._get_channel_label(channel)

            # Auto-set color range from data when the spinbox values are degenerate
            # (e.g. v_min == v_max because the user never clicked Reset).
            if sensor_v is not None and abs(v_max - v_min) < 1e-9:
                finite = sensor_v[np.isfinite(sensor_v)]
                if len(finite) > 0:
                    v_min = float(finite.min())
                    v_max = float(finite.max())
                    if abs(v_max - v_min) < 1e-9:
                        v_max = v_min + 1.0

            # ---- Frame scatter ----
            frame_lons = frame_lats = np.array([])
            if (self._viz_df is not None
                    and "lat" in self._viz_df.columns
                    and "lon" in self._viz_df.columns):
                fl = self._viz_df["lat"].to_numpy(dtype=float)
                fo = self._viz_df["lon"].to_numpy(dtype=float)
                v  = np.isfinite(fl) & np.isfinite(fo)
                frame_lats = fl[v]
                frame_lons = fo[v]

            # ---- Reference latitude for aspect correction ----
            all_lats_combined = np.concatenate(
                [a for a in [nav_lats, frame_lats] if len(a) > 0]
            ) if (len(nav_lats) > 0 or len(frame_lats) > 0) else np.array([0.0])
            ref_lat = float(np.mean(all_lats_combined))

            # ---- Figure ----
            has_colorbar = (sensor_v is not None and channel != "None")
            fig, ax = plt.subplots(figsize=(11, 8.5))  # US letter landscape

            # ---- GPS track ----
            if len(nav_lons) >= 2:
                if has_colorbar and sensor_v is not None and len(sensor_v) == len(nav_lons):
                    if log_scale and v_min > 0 and v_max > 0:
                        norm = LogNorm(vmin=max(v_min, 1e-10), vmax=v_max)
                    else:
                        norm = Normalize(vmin=v_min, vmax=v_max)
                    pts  = np.array([nav_lons, nav_lats]).T.reshape(-1, 1, 2)
                    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
                    # Colour each segment by the midpoint sensor value
                    seg_vals = (sensor_v[:-1] + sensor_v[1:]) / 2.0
                    lc = LineCollection(segs, cmap=cmap, norm=norm,
                                        linewidth=1.5, alpha=0.9, zorder=3)
                    lc.set_array(seg_vals)
                    ax.add_collection(lc)
                    sm = ScalarMappable(cmap=cmap, norm=norm)
                    sm.set_array([])
                    cbar = fig.colorbar(sm, ax=ax, shrink=0.7, pad=0.02)
                    cbar.set_label(sensor_label, fontsize=10)
                    cbar.ax.tick_params(labelsize=8)
                else:
                    ax.plot(nav_lons, nav_lats, "-", color="#e63946",
                            linewidth=1.0, alpha=0.85, label="GPS track", zorder=3)

            # ---- Frame scatter ----
            if len(frame_lons) > 0:
                ax.scatter(frame_lons, frame_lats, s=8, c="#1565c0",
                           alpha=0.85, zorder=5, label=f"Frames ({len(frame_lons):,})")

            # ---- Set view extent with padding ----
            all_lons_combined = np.concatenate(
                [a for a in [nav_lons, frame_lons] if len(a) > 0]
            ) if (len(nav_lons) > 0 or len(frame_lons) > 0) else np.array([0.0])
            lon_pad = (all_lons_combined.max() - all_lons_combined.min()) * 0.06 + 1e-4
            lat_pad = (all_lats_combined.max() - all_lats_combined.min()) * 0.06 + 1e-4
            ax.set_xlim(all_lons_combined.min() - lon_pad, all_lons_combined.max() + lon_pad)
            ax.set_ylim(all_lats_combined.min() - lat_pad, all_lats_combined.max() + lat_pad)

            # ---- Correct geographic aspect ratio ----
            # 1° longitude = cos(lat) × 1° latitude in physical distance.
            ax.set_aspect(1.0 / np.cos(np.radians(ref_lat)))

            # ---- Labels, ticks, grid ----
            ax.set_xlabel("Longitude (°)", fontsize=11)
            ax.set_ylabel("Latitude (°)", fontsize=11)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(True, linestyle=":", alpha=0.5, linewidth=0.6)

            # ---- Scale bar and north arrow ----
            self._map_scale_bar(ax, ref_lat)
            self._map_north_arrow(ax)

            # ---- Title ----
            dive = Path(self._viz_csv_path).stem if self._viz_csv_path else "Map Export"
            ts_range = ""
            if self._viz_df is not None and "timestamp_iso" in self._viz_df.columns:
                ts = self._viz_df["timestamp_iso"].dropna()
                if len(ts) > 0:
                    ts_range = f"  |  {str(ts.iloc[0])[:10]} – {str(ts.iloc[-1])[:10]}"
            ax.set_title(f"{dive}{ts_range}", fontsize=12, pad=10)

            # ---- Legend (only when track is solid) ----
            if not has_colorbar:
                handles, labels = ax.get_legend_handles_labels()
                if handles:
                    ax.legend(handles=handles, labels=labels,
                              loc="lower right", fontsize=8, framealpha=0.85)

            # ---- Save ----
            ext = Path(path).suffix.lower()
            dpi = 300 if ext == ".png" else 150
            plt.tight_layout()
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
            plt.close(fig)

            n_track = len(nav_lons)
            n_frames = len(frame_lons)
            self.log_text.append(
                f"Map exported: {n_track:,} track pts, {n_frames:,} frames → {Path(path).name}"
            )
            QMessageBox.information(self, "Map exported",
                                    f"Saved to {Path(path).name}\n"
                                    f"Track: {n_track:,} pts  |  Frames: {n_frames:,}")

        except Exception as exc:
            import traceback
            QMessageBox.critical(self, "Export failed",
                                 f"{exc}\n\n{traceback.format_exc()[-800:]}")

    def _viz_clear_data(self) -> None:
        """Reset all visualization state: DataFrame, path, segments, pending selection.

        Called when the user clears the loaded CSV or loads a new set of files.
        Resets every visualization widget to its empty/placeholder state and
        triggers a map refresh if the Visualization tab is currently active.
        """
        self._viz_df = None
        self._viz_csv_path = ""
        self._viz_segments = []
        self._viz_pending_indices = []
        self.viz_csv_label.setText("No data loaded.")
        self.viz_stats_label.setText("")
        self.viz_selection_label.setText("0 frames selected")
        self.viz_utm_zone_label.setText("UTM: —")
        self._refresh_viz_segments_list()
        self._populate_viz_channel_combo()
        if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
            self._refresh_map()

    def _viz_toggle_select_mode(self, enabled: bool) -> None:
        """Toggle rubber-band box-select mode on the map; deactivates pick mode if on."""
        if enabled:
            self.viz_pick_interval_button.setChecked(False)
        self.map_widget.set_select_mode(enabled)
        self.viz_box_select_button.setText("Box Select  ✓" if enabled else "Box Select")

    def _viz_toggle_pick_mode(self, enabled: bool) -> None:
        """Toggle two-click trackline pick mode; deactivates box-select if on."""
        if enabled:
            self.viz_box_select_button.setChecked(False)
            self.viz_pick_status_label.setText("Click first point on the track...")
            self.viz_pick_status_label.setStyleSheet("font-size: 10px; color: #e67e22; font-weight: bold;")
        else:
            self.viz_pick_status_label.setText("Click to activate, then click two points on the track.")
            self.viz_pick_status_label.setStyleSheet("font-size: 10px; color: gray;")
        self.map_widget.set_pick_mode(enabled)
        self.viz_pick_interval_button.setText("Pick Interval  ✓" if enabled else "Pick Interval")

    def _on_track_pick_point_placed(self, unix_t: float) -> None:
        """Handle the first click in pick mode: update the status label with the start time."""
        dt_str = datetime.utcfromtimestamp(unix_t).strftime("%H:%M:%S")
        self.viz_pick_status_label.setText(f"Start: {dt_str} — now click the end point on the track.")
        self.viz_pick_status_label.setStyleSheet("font-size: 10px; color: #e67e22; font-weight: bold;")

    def _on_track_interval_picked(self, t_start: float, t_end: float) -> None:
        """Handle the completed two-click interval pick: add interval to the pending job."""
        self.viz_pick_interval_button.setChecked(False)
        start_dt = datetime.utcfromtimestamp(t_start)
        end_dt   = datetime.utcfromtimestamp(t_end)
        self.pending_job.intervals.append(SelectedTimeRange(start_time=start_dt, end_time=end_dt))
        self._refresh_job_builder_ui()
        self._refresh_summary()
        self.viz_pick_status_label.setText(
            f"Added to Job #{self.pending_job.job_id}: {start_dt.strftime('%H:%M:%S')} → {end_dt.strftime('%H:%M:%S')}"
        )
        self.viz_pick_status_label.setStyleSheet("font-size: 10px; color: #27ae60; font-weight: bold;")

    # -----------------------------------------------------------------------
    # Job Builder handlers
    # -----------------------------------------------------------------------

    def _refresh_job_builder_ui(self) -> None:
        """Rebuild the Job Builder list widget from pending_job.intervals."""
        self.job_id_label.setText(f"Job #{self.pending_job.job_id}")
        self.job_interval_list.clear()
        for interval in self.pending_job.intervals:
            label = (
                f"{interval.start_time.strftime('%Y-%m-%d %H:%M:%S')}  →  "
                f"{interval.end_time.strftime('%H:%M:%S')}"
            )
            self.job_interval_list.addItem(label)

    def _refresh_job_history_dropdown(self) -> None:
        """Rebuild the history combo box from segment_history."""
        self.job_history_combo.blockSignals(True)
        self.job_history_combo.clear()
        if not self.segment_history:
            self.job_history_combo.addItem("— no history —")
        else:
            for rec in reversed(self.segment_history):  # most recent first
                label = (
                    f"Job #{rec.job_id}  "
                    f"{rec.interval.start_time.strftime('%H:%M:%S')} → "
                    f"{rec.interval.end_time.strftime('%H:%M:%S')}  "
                    f"[{rec.status}]"
                )
                self.job_history_combo.addItem(label)
        self.job_history_combo.blockSignals(False)

    def _job_new(self) -> None:
        """Clear the current job's intervals and start a new job with the next ID."""
        self.next_job_id += 1
        self.pending_job = Job(job_id=self.next_job_id)
        self._refresh_job_builder_ui()
        self._refresh_summary()

    def _job_remove_selected_interval(self) -> None:
        """Remove the selected interval from the pending job."""
        row = self.job_interval_list.currentRow()
        if 0 <= row < len(self.pending_job.intervals):
            del self.pending_job.intervals[row]
            self._refresh_job_builder_ui()
            self._refresh_summary()

    def _job_add_from_history(self) -> None:
        """Copy the selected history segment's interval into the pending job."""
        idx = self.job_history_combo.currentIndex()
        if not self.segment_history or idx < 0:
            return
        # History is displayed in reverse order (most recent first).
        history_idx = len(self.segment_history) - 1 - idx
        if 0 <= history_idx < len(self.segment_history):
            rec = self.segment_history[history_idx]
            self.pending_job.intervals.append(
                SelectedTimeRange(start_time=rec.interval.start_time, end_time=rec.interval.end_time)
            )
            self._refresh_job_builder_ui()
            self._refresh_summary()

    def _job_open_history_folder(self) -> None:
        """Open the output folder of the selected history segment in the file manager."""
        import subprocess
        idx = self.job_history_combo.currentIndex()
        if not self.segment_history or idx < 0:
            return
        history_idx = len(self.segment_history) - 1 - idx
        if 0 <= history_idx < len(self.segment_history):
            folder = self.segment_history[history_idx].output_path
            if Path(folder).exists():
                subprocess.Popen(["xdg-open", folder])
            else:
                QMessageBox.warning(self, "Folder not found", f"The folder no longer exists:\n{folder}")

    def _on_history_mode_toggled(self, enabled: bool) -> None:
        """Switch the map trackline between normal mode and history overlay mode."""
        self.viz_history_mode_button.setText(
            "History Mode  ✓" if enabled else "Show Sampled Regions"
        )
        self._refresh_map()

    def _viz_clear_selection(self) -> None:
        """Discard the current box-select pending indices and reset the map selection highlight."""
        self._viz_pending_indices = []
        self.viz_box_select_button.setChecked(False)
        self.map_widget.clear_selection()

    def _on_viz_selection_changed(self, indices: list[int]) -> None:
        """Receive the updated rubber-band selection from MapWidget and update the count label.

        Connected to MapWidget.selection_changed.  Stores the selected indices in
        _viz_pending_indices so _viz_save_segment() can retrieve them.

        Args:
            indices: Valid-coordinate row indices into _viz_df for the selected frames.
        """
        self._viz_pending_indices = indices
        n = len(indices)
        self.viz_selection_label.setText(f"{n} frame{'s' if n != 1 else ''} selected")

    def _viz_save_segment(self) -> None:
        """Save the pending box-selection as a named segment.

        Reads the name from viz_segment_name_edit, assigns the next color from
        SEGMENT_COLORS (cycling modulo), appends the segment to _viz_segments,
        clears the pending selection, and pushes the updated segment list to MapWidget.

        Shows a warning dialog if no frames are selected or the name is empty.
        """
        if not self._viz_pending_indices:
            QMessageBox.warning(self, "Nothing selected", "Use Box Select on the map to select frames first.")
            return
        name = self.viz_segment_name_edit.text().strip()
        if not name:
            QMessageBox.warning(self, "Name required", "Enter a name for the segment.")
            return
        color = SEGMENT_COLORS[len(self._viz_segments) % len(SEGMENT_COLORS)]
        self._viz_segments.append({
            "name": name,
            "indices": list(self._viz_pending_indices),
            "color": color,
        })
        self.viz_segment_name_edit.clear()
        self._viz_pending_indices = []
        self.viz_selection_label.setText("0 frames selected")
        self.viz_box_select_button.setChecked(False)
        self._refresh_viz_segments_list()
        self.map_widget.set_selection([])
        self.map_widget.set_segments(self._viz_segments)
        self._refresh_trackline_segments()

    def _viz_remove_segment(self) -> None:
        """Delete the segment selected in viz_segment_list and reassign remaining colors.

        Colors are reassigned sequentially after deletion so the color order always
        matches SEGMENT_COLORS[0], SEGMENT_COLORS[1], … regardless of which segment
        was deleted.
        """
        row = self.viz_segment_list.currentRow()
        if row < 0 or row >= len(self._viz_segments):
            return
        del self._viz_segments[row]
        # Reassign colors to keep them sequential
        for i, seg in enumerate(self._viz_segments):
            seg["color"] = SEGMENT_COLORS[i % len(SEGMENT_COLORS)]
        self._refresh_viz_segments_list()
        self.map_widget.set_segments(self._viz_segments)
        self._refresh_trackline_segments()

    def _viz_export_segment(self) -> None:
        """Export the selected segment's frames and CSV data to a user-chosen folder.

        Workflow:
          1. Reads the selected segment from _viz_segments.
          2. Prompts the user to choose an output directory.
          3. Creates <output>/<segment_name>/frames/ and copies each frame image
             from its source interp.csv's sibling frames/ directory.
          4. Saves a segment_interp.csv with the segment's rows (minus internal
             _source_csv column) into <output>/<segment_name>/.
          5. Reports how many frames were copied vs. skipped (missing source files).

        The internal _source_csv column is dropped before writing segment_interp.csv
        so it does not appear in the user-facing output.
        """
        row = self.viz_segment_list.currentRow()
        if row < 0 or row >= len(self._viz_segments):
            QMessageBox.warning(self, "No segment selected", "Select a segment from the list to export.")
            return
        if self._viz_df is None:
            return
        seg = self._viz_segments[row]
        indices = seg["indices"]
        seg_df = self._viz_df.iloc[indices].copy()

        out_dir = QFileDialog.getExistingDirectory(
            self, f"Export '{seg['name']}' — choose output folder"
        )
        if not out_dir:
            return

        seg_name_safe = seg["name"].replace(" ", "_").replace("/", "-")
        export_dir = Path(out_dir) / seg_name_safe
        frames_out = export_dir / "frames"
        frames_out.mkdir(parents=True, exist_ok=True)

        copied = 0
        skipped = 0
        for _, row_data in seg_df.iterrows():
            frame_fn = row_data.get("frame_filename", "")
            if not frame_fn:
                skipped += 1
                continue
            source_csv = Path(row_data.get("_source_csv", ""))
            frames_dir = source_csv.parent / "frames" if source_csv.exists() else None
            if frames_dir is None or not frames_dir.exists():
                skipped += 1
                continue
            src = frames_dir / frame_fn
            if src.exists():
                shutil.copy2(src, frames_out / frame_fn)
                copied += 1
            else:
                skipped += 1

        # Drop internal column before saving
        csv_out = seg_df.drop(columns=["_source_csv"], errors="ignore")
        csv_out.to_csv(export_dir / "segment_interp.csv", index=False)

        msg = f"Exported {copied} frames to:\n{export_dir}"
        if skipped:
            msg += f"\n({skipped} frames skipped — source files not found)"
        QMessageBox.information(self, "Export complete", msg)

    def _refresh_viz_stats(self) -> None:
        """Update the viz_csv_label and viz_stats_label with row count and total track distance.

        Computes the cumulative haversine distance along the lat/lon track using
        consecutive valid (non-NaN) coordinate pairs.  Both labels are cleared to
        placeholder text when no DataFrame is loaded.
        """
        if self._viz_df is None:
            self.viz_csv_label.setText("No data loaded.")
            self.viz_stats_label.setText("")
            return
        df = self._viz_df
        n = len(df)
        path_display = self._viz_csv_path if self._viz_csv_path else "?"
        self.viz_csv_label.setText(path_display)

        # Compute total track distance
        dist_km = 0.0
        if "lat" in df.columns and "lon" in df.columns:
            lats = df["lat"].to_numpy(dtype=float)
            lons = df["lon"].to_numpy(dtype=float)
            valid = ~(np.isnan(lats) | np.isnan(lons))
            v_lats = lats[valid]
            v_lons = lons[valid]
            for i in range(1, len(v_lats)):
                dist_km += haversine_km(v_lats[i - 1], v_lons[i - 1], v_lats[i], v_lons[i])
        self.viz_stats_label.setText(
            f"{n:,} frames  ·  track distance: {dist_km:.3f} km"
        )

    def _refresh_viz_segments_list(self) -> None:
        """Rebuild viz_segment_list from _viz_segments, coloring each item by segment color."""
        self.viz_segment_list.clear()
        for seg in self._viz_segments:
            n = len(seg["indices"])
            self.viz_segment_list.addItem(f"  {seg['name']}  ({n} frames)")
            item = self.viz_segment_list.item(self.viz_segment_list.count() - 1)
            # Color indicator via foreground
            item.setForeground(QColor(seg["color"]))

    def _on_map_scale_changed(self, text: str) -> None:
        """Translate the scale-combo text to a MapWidget display mode and apply it.

        "meters (projected)" → "meters"  (equirectangular projection, x/y in metres)
        Any other text      → "latlon"   (raw longitude/latitude axes)

        Args:
            text: The currently selected text from the map scale combo box.
        """
        mode = "meters" if text.startswith("m") else "latlon"
        self.map_widget.set_display_mode(mode)

    def _on_channel_combo_changed(self, text: str) -> None:
        """Enable/disable colour controls when the channel selection changes."""
        enabled = (text != "None")
        for w in (self.viz_alt_min_spin, self.viz_alt_max_spin,
                  self.viz_alt_reset_button, self.viz_alt_recolor_button,
                  self.viz_export_track_button, self.viz_export_geojson_button,
                  self.viz_color_scale_combo):
            w.setEnabled(enabled)
        self.viz_alt_key_widget.setVisible(enabled)
        if enabled:
            self._on_sensor_range_reset()
        else:
            self._refresh_map()

    def _on_color_scale_changed(self, _text: str) -> None:
        """Re-render the trackline when the linear/log scale toggle changes."""
        if self.viz_channel_combo.currentText() != "None":
            self._refresh_map()

    def _on_sensor_range_changed(self) -> None:
        """Apply the current spinbox bounds and recolor the trackline."""
        if self.viz_channel_combo.currentText() != "None":
            self._update_sensor_key_labels()
            self._refresh_map()

    def _on_sensor_range_reset(self) -> None:
        """Reset the colour scale to the actual data min/max for the active channel."""
        channel = self.viz_channel_combo.currentText()
        data_arr = self._get_sensor_array_for_channel(channel)
        if data_arr is not None and len(data_arr) > 0:
            finite = data_arr[np.isfinite(data_arr)]
            if len(finite) > 0:
                self.viz_alt_min_spin.blockSignals(True)
                self.viz_alt_max_spin.blockSignals(True)
                self.viz_alt_min_spin.setValue(float(finite.min()))
                self.viz_alt_max_spin.setValue(float(finite.max()))
                self.viz_alt_min_spin.blockSignals(False)
                self.viz_alt_max_spin.blockSignals(False)
        self._update_sensor_key_labels()
        self._refresh_map()

    def _get_sensor_array_for_channel(self, channel: str) -> np.ndarray | None:
        """Return raw values for the named channel (used by range-reset and export)."""
        if channel == "None":
            return None
        if channel == "Altitude":
            if self.navigation_file and self.navigation_file.altitude_source:
                try:
                    df = SensorService.load_time_value_dataframe(
                        self.navigation_file.altitude_source
                    )
                    return df["value"].to_numpy(dtype=float)
                except Exception:
                    pass
            return None
        if self._viz_df is not None and channel in self._viz_df.columns:
            return pd.to_numeric(self._viz_df[channel], errors="coerce").to_numpy(dtype=float)
        return None

    def _get_channel_label(self, channel: str) -> str:
        """Return a display label (name + units) for the active channel."""
        if channel == "Altitude":
            return "Altitude (m)"
        for sf in self.sensor_files:
            for ch in sf.channels:
                if ch.source_column == channel or ch.display_name == channel:
                    return f"{ch.display_name} ({ch.units})" if ch.units else ch.display_name
        return channel

    def _get_channel_units(self, channel: str) -> str:
        """Return the units string for a channel (empty string if unknown)."""
        if channel == "Altitude":
            return "m"
        for sf in self.sensor_files:
            for ch in sf.channels:
                if ch.source_column == channel or ch.display_name == channel:
                    return ch.units or ""
        return ""

    def _update_sensor_key_labels(self) -> None:
        """Sync the in-map colorbar min/max labels to the current spinbox values."""
        lo = self.viz_alt_min_spin.value()
        hi = self.viz_alt_max_spin.value()
        units = self._get_channel_units(self.viz_channel_combo.currentText())
        self.viz_alt_key_max_label.setText(f"{hi:.4g}\n{units}")
        self.viz_alt_key_min_label.setText(f"{lo:.4g}\n{units}")

    def _on_trackline_width_changed(self, value: int) -> None:
        """Apply a new trackline width from the slider and update the pixel label.

        Calls MapWidget.set_trackline_width() directly so only the trackline is
        redrawn — no full _refresh_map() needed.

        Args:
            value: Slider integer value (1–10), used directly as pixel width.
        """
        self.viz_track_width_label.setText(f"{value} px")
        self.map_widget.set_trackline_width(float(value))

    @staticmethod
    def _is_geographic(lats: np.ndarray, lons: np.ndarray) -> tuple[bool, str]:
        """Return (ok, warning_message). ok=True when values look like geographic degrees."""
        if len(lats) == 0 or len(lons) == 0:
            return True, ""
        lat_ok = bool(np.all(lats >= -90) and np.all(lats <= 90))
        lon_ok = bool(np.all(lons >= -180) and np.all(lons <= 180))
        if lat_ok and lon_ok:
            return True, ""
        parts = []
        if not lat_ok:
            parts.append(f"lat range {lats.min():.3g}–{lats.max():.3g} (expected −90 to 90)")
        if not lon_ok:
            parts.append(f"lon range {lons.min():.3g}–{lons.max():.3g} (expected −180 to 180)")
        return False, "Non-geographic values: " + "; ".join(parts)

    def _refresh_trackline_segments(self) -> None:
        """Push segment time-range highlights to the map trackline.

        For each saved segment, finds the min/max unix_time of its frames in
        _viz_df and passes (start_unix, end_unix, color) to the map widget so
        it can draw a colored band over the corresponding section of the trackline.
        """
        if self._viz_df is None or "unix_time" not in self._viz_df.columns:
            self.map_widget.set_trackline_segment_ranges([])
            return
        ranges: list[tuple[float, float, str]] = []
        for seg in self._viz_segments:
            idxs = seg.get("indices", [])
            if not idxs:
                continue
            times = self._viz_df.iloc[idxs]["unix_time"].dropna()
            if times.empty:
                continue
            ranges.append((float(times.min()), float(times.max()), seg["color"]))
        self.map_widget.set_trackline_segment_ranges(ranges)

    def _refresh_history_overlay(self) -> None:
        """Push previously-sampled regions to the map as colored trackline bands.

        In history mode, each SegmentRecord in segment_history is drawn as a blue
        band over the nav trackline.  In normal mode, the overlay is cleared so the
        standard trackline segment coloring is shown instead.
        """
        if not getattr(self, "viz_history_mode_button", None):
            return
        if not self.viz_history_mode_button.isChecked():
            self.map_widget.set_history_ranges([])
            return
        ranges: list[tuple[float, float, str, str]] = []
        for rec in self.segment_history:
            t0 = float(calendar.timegm(rec.interval.start_time.timetuple()))
            t1 = float(calendar.timegm(rec.interval.end_time.timetuple()))
            tooltip = (
                f"Job #{rec.job_id}  ·  "
                f"{rec.interval.start_time.strftime('%H:%M:%S')} → {rec.interval.end_time.strftime('%H:%M:%S')}"
                f"  ·  {rec.status}"
            )
            ranges.append((t0, t1, "#2196f3", tooltip))
        self.map_widget.set_history_ranges(ranges)

    def _refresh_map(self) -> None:
        """Rebuild the map display: full GPS trackline, video coverage, and frame scatter.

        Execution order:
          1. Load raw nav lat/lon arrays from the configured NavigationConfig, interpolate
             longitude onto the latitude time axis, and filter out NaN pairs.  Validates
             that the resulting values are plausible geographic coordinates (lat ∈ [−90,90],
             lon ∈ [−180,180]); logs a red warning if not.
          2. Push the trackline (lons, lats, unix_times) to MapWidget.set_full_trackline()
             and video coverage windows to MapWidget.set_videos().
          3. If _viz_df is not loaded or has no lat/lon columns, pass empty arrays to
             MapWidget.load_data() and return early.
          4. Extract valid (non-NaN) frame coordinates, build per-frame tooltip labels,
             remap segment indices from _viz_df row space → valid-coordinate index space,
             and push the frame scatter + segments to the map.

        All status/warning HTML fragments are collected in status_parts and written to
        map_nav_status_label at the end.
        """
        status_parts: list[str] = []

        # --- Full navigation trackline from raw nav CSVs ---
        nav_lons: np.ndarray = np.array([])
        nav_lats: np.ndarray = np.array([])
        nav_alt_values: np.ndarray | None = None
        if self.navigation_file is not None:
            lat_src = self.navigation_file.latitude_source
            lon_src = self.navigation_file.longitude_source
            try:
                lat_df = SensorService.load_time_value_dataframe(lat_src)
                lon_df = SensorService.load_time_value_dataframe(lon_src)
                lat_df = lat_df.sort_values("unix_time").reset_index(drop=True)
                lon_df = lon_df.sort_values("unix_time").reset_index(drop=True)
                nav_unix_raw = lat_df["unix_time"].to_numpy(dtype=float)
                nav_lats = lat_df["value"].to_numpy(dtype=float)
                nav_lons = SensorService.interpolate_series(
                    lat_df["unix_time"], lon_df["unix_time"], lon_df["value"]
                )
                mask = np.isfinite(nav_lats) & np.isfinite(nav_lons)
                nav_lats = nav_lats[mask]
                nav_lons = nav_lons[mask]
                nav_unix_times = nav_unix_raw[mask]

                if self.navigation_file.altitude_source is not None:
                    try:
                        alt_src = self.navigation_file.altitude_source
                        alt_df  = SensorService.load_time_value_dataframe(alt_src)
                        raw_alt = SensorService.interpolate_series(
                            lat_df["unix_time"], alt_df["unix_time"], alt_df["value"]
                        )
                        nav_alt_values = raw_alt[mask]
                    except Exception as alt_exc:
                        self.log_text.append(
                            f"Visualization: altitude load failed: {alt_exc}"
                        )

                geo_ok, geo_warn = self._is_geographic(nav_lats, nav_lons)
                if not geo_ok:
                    msg = (
                        f"NAV WARNING: {geo_warn}\n"
                        f"  Lat col: '{lat_src.value_column}'  in  {Path(lat_src.csv_path).name}\n"
                        f"  Lon col: '{lon_src.value_column}'  in  {Path(lon_src.csv_path).name}\n"
                        f"  Check Navigation config — wrong columns selected?"
                    )
                    self.log_text.append(msg)
                    status_parts.append(
                        f"<span style='color:#c0392b; font-weight:bold;'>NAV: wrong columns? "
                        f"{geo_warn}</span>"
                    )
                    nav_lats = np.array([])
                    nav_lons = np.array([])
                    nav_unix_times = np.array([])
                    nav_alt_values = None
                else:
                    lat_csv = Path(lat_src.csv_path).name
                    lon_csv = Path(lon_src.csv_path).name
                    status_parts.append(
                        f"<span style='color:#27ae60;'>NAV: {len(nav_lats):,} pts "
                        f"· lat {nav_lats.min():.4f}→{nav_lats.max():.4f} "
                        f"({lat_src.value_column} / {lat_csv})"
                        f" · lon {nav_lons.min():.4f}→{nav_lons.max():.4f} "
                        f"({lon_src.value_column} / {lon_csv})</span>"
                    )
            except Exception as exc:
                self.log_text.append(f"Visualization: could not load nav trackline: {exc}")
                status_parts.append(f"<span style='color:#c0392b;'>NAV error: {exc}</span>")
                nav_unix_times = np.array([])
        else:
            status_parts.append("<span style='color:gray;'>NAV: not configured</span>")
            nav_unix_times = np.array([])

        # --- UTM zone indicator ---
        if len(nav_lons) > 0 and len(nav_lats) > 0:
            mean_lon = float(np.mean(nav_lons))
            mean_lat = float(np.mean(nav_lats))
            zone_num  = math.floor((mean_lon + 180.0) / 6.0) + 1
            hemi      = "N" if mean_lat >= 0 else "S"
            self.viz_utm_zone_label.setText(f"UTM: {zone_num}{hemi}")
        else:
            self.viz_utm_zone_label.setText("UTM: —")

        # --- Determine whether raster mode is active ---
        raster_channel = self.raster_channel_combo.currentText()
        raster_active  = (
            self._raster_df is not None
            and raster_channel != "None"
            and raster_channel in self._raster_df.columns
        )

        if raster_active:
            # === RASTER MODE ===
            # Primary trackline = sensor-geolocated raster (sensor native timestamps).
            # Nav trackline becomes a thin gray background underlay.
            rdf = self._raster_df
            r_lats  = rdf["lat"].to_numpy(dtype=float)
            r_lons  = rdf["lon"].to_numpy(dtype=float)
            r_times = rdf["unix_time"].to_numpy(dtype=float)
            r_alt   = rdf["alt"].to_numpy(dtype=float) if "alt" in rdf.columns else None

            r_mask  = np.isfinite(r_lats) & np.isfinite(r_lons)
            r_lats  = r_lats[r_mask]
            r_lons  = r_lons[r_mask]
            r_times = r_times[r_mask]
            r_alt   = r_alt[r_mask] if r_alt is not None else None

            sensor_v = pd.to_numeric(
                rdf[raster_channel], errors="coerce"
            ).to_numpy(dtype=float)[r_mask]
            raster_log   = self.raster_scale_combo.currentText() == "Logarithmic"
            raster_label = self._get_raster_channel_label(raster_channel)

            self.map_widget.set_trackline_sensor_range(
                self.raster_min_spin.value(), self.raster_max_spin.value()
            )
            self.map_widget.set_full_trackline(
                r_lons, r_lats, r_times,
                alt_values=r_alt,
                sensor_values=sensor_v,
                sensor_coloring_enabled=True,
                sensor_log_scale=raster_log,
                sensor_label=raster_label,
            )
            # Show the full nav trackline as a gray background reference.
            self.map_widget.set_nav_background_track(nav_lons, nav_lats)

            # Update the shared color key to show raster range.
            self._refresh_raster_key_labels()
            self.viz_alt_key_widget.setVisible(True)

        else:
            # === PIPELINE / NAV MODE (existing behavior) ===
            self.map_widget.set_nav_background_track(np.array([]), np.array([]))

            channel   = self.viz_channel_combo.currentText()
            # Sync the key widget visibility with the pipeline channel selection.
            self.viz_alt_key_widget.setVisible(channel != "None")
            log_scale = self.viz_color_scale_combo.currentText() == "Logarithmic"
            sensor_values: np.ndarray | None = None
            sensor_label: str = ""

            if channel == "Altitude":
                sensor_values = nav_alt_values
                sensor_label  = "Altitude (m)"
            elif channel != "None" and self._viz_df is not None and channel in self._viz_df.columns:
                if len(nav_unix_times) > 0 and "unix_time" in self._viz_df.columns:
                    frame_times = self._viz_df["unix_time"].to_numpy(dtype=float)
                    frame_vals  = pd.to_numeric(
                        self._viz_df[channel], errors="coerce"
                    ).to_numpy(dtype=float)
                    raw_sensor  = SensorService.interpolate_series(
                        pd.Series(nav_unix_times),
                        pd.Series(frame_times),
                        pd.Series(frame_vals),
                    )
                    sensor_values = raw_sensor
                    sensor_label  = self._get_channel_label(channel)

            if channel != "None":
                self.map_widget.set_trackline_sensor_range(
                    self.viz_alt_min_spin.value(), self.viz_alt_max_spin.value()
                )

            self.map_widget.set_full_trackline(
                nav_lons, nav_lats, nav_unix_times,
                alt_values=nav_alt_values,
                sensor_values=sensor_values,
                sensor_coloring_enabled=(channel != "None"),
                sensor_log_scale=log_scale,
                sensor_label=sensor_label,
            )
        self.map_widget.set_videos([
            (float(calendar.timegm(v.start_time.timetuple())),
             float(calendar.timegm(v.end_time.timetuple())),
             v.filename)
            for v in self.videos
        ])

        # --- Frame scatter from master CSV ---
        if self._viz_df is None or len(self._viz_df) == 0:
            self.map_widget.load_data(np.array([]), np.array([]), [])
            self.map_nav_status_label.setText("<br>".join(status_parts))
            return

        df = self._viz_df
        if "lat" not in df.columns or "lon" not in df.columns:
            self.map_widget.load_data(np.array([]), np.array([]), [])
            self.log_text.append("Visualization: master.csv has no lat/lon columns.")
            status_parts.append("<span style='color:#c0392b;'>Frames: no lat/lon columns in master.csv</span>")
            self.map_nav_status_label.setText("<br>".join(status_parts))
            return

        lats = df["lat"].to_numpy(dtype=float)
        lons = df["lon"].to_numpy(dtype=float)
        valid = ~(np.isnan(lats) | np.isnan(lons))
        valid_indices = np.where(valid)[0]
        v_lons = lons[valid_indices]
        v_lats = lats[valid_indices]

        if len(v_lats) > 0:
            frame_geo_ok, frame_geo_warn = self._is_geographic(v_lats, v_lons)
            if not frame_geo_ok:
                status_parts.append(
                    f"<span style='color:#c0392b;'>Frames: {frame_geo_warn}</span>"
                )
            else:
                status_parts.append(
                    f"<span style='color:#2980b9;'>Frames: {len(v_lats):,} pts "
                    f"· lat {v_lats.min():.4f}→{v_lats.max():.4f} "
                    f"· lon {v_lons.min():.4f}→{v_lons.max():.4f}</span>"
                )
        else:
            status_parts.append("<span style='color:gray;'>Frames: no valid lat/lon (all NaN)</span>")

        self.map_nav_status_label.setText("<br>".join(status_parts))

        labels: list[str] = []
        for orig_idx in valid_indices:
            row = df.iloc[orig_idx]
            frame_fn = str(row.get("frame_filename", ""))
            video_fn = str(row.get("video_filename", ""))
            ts = str(row.get("timestamp_iso", ""))
            labels.append(f"Frame: {frame_fn}\nVideo: {video_fn}\nTime:  {ts}")

        orig_to_valid: dict[int, int] = {int(orig): vi for vi, orig in enumerate(valid_indices)}
        remapped_segments = []
        for seg in self._viz_segments:
            remapped = [orig_to_valid[i] for i in seg["indices"] if i in orig_to_valid]
            remapped_segments.append({**seg, "indices": remapped})

        self.map_widget.load_data(v_lons, v_lats, labels)
        self.map_widget.set_segments(remapped_segments)
        self._refresh_trackline_segments()
        self._refresh_history_overlay()
        self.map_widget.fit_view()

    def _create_timeseries_plot(self, title: str) -> pg.PlotWidget:
        """Create a pyqtgraph PlotWidget with a date-aware X axis and standard styling.

        Used by _refresh_postprocessing_graphs() to build each per-channel sensor
        timeseries graph.  The resulting widget is sized between 220–320 px tall so
        three or four plots fit comfortably in the scrollable graph panel.

        Args:
            title: Axis label (left) and plot title displayed at the top of the widget.

        Returns:
            A configured PlotWidget ready to be added to a layout.
        """
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
        """Clear the plot and draw a new line series, or show a "No data" placeholder.

        Args:
            plot:   The PlotWidget to draw into (cleared first).
            data_x: Unix timestamps for the X axis.
            data_y: Sensor values for the Y axis.
            color:  Hex color string for the line pen (default medium blue).
        """
        plot.clear()
        if not data_x or not data_y:
            text_item = pg.TextItem("No data available", color="gray", anchor=(0, 0))
            plot.addItem(text_item)
            text_item.setPos(0, 0)
            return
        plot.plot(data_x, data_y, pen=pg.mkPen(color=color, width=2))
        plot.setXRange(min(data_x), max(data_x), padding=0.02)

    def _refresh_warnings(self) -> None:
        """Update the warning_text panel with skipped video filenames.

        Shows a bulleted list of skipped filenames when any videos were excluded
        during the last scan, or "No warnings." when the list is empty.
        """
        if self.skipped_videos:
            warnings = ["Skipped videos:"] + [f"  {name}" for name in self.skipped_videos]
            self.warning_text.setPlainText("\n".join(warnings))
        else:
            self.warning_text.setPlainText("No warnings.")

    def _compute_combined_video_coverage(self):
        """Return the (earliest_start, latest_end) datetime pair spanning all loaded videos.

        Returns:
            A (datetime, datetime) tuple of the combined coverage window, or None
            if no videos are loaded.
        """
        if not self.videos:
            return None
        return min(v.start_time for v in self.videos), max(v.end_time for v in self.videos)

    @staticmethod
    def _table_item(text: str) -> QTableWidgetItem:
        """Create a read-only QTableWidgetItem with the given text.

        Strips the Qt.ItemIsEditable flag so the user cannot accidentally edit
        table cells that are meant to display data only.

        Args:
            text: The display string for the table cell.

        Returns:
            A non-editable QTableWidgetItem.
        """
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemIsEditable)
        return item
