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
import copy
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
    QButtonGroup,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QDockWidget,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QRadioButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QStackedWidget,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
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

from chat_panel import ChatPanel
from claude_service import save_api_key, load_api_key
from config_service import ConfigService
from workspace_panel import WorkspacePanel
from models import AnnotationConfig, Job, NavigationConfig, Task, TaskStack, SegmentRecord, SelectedTimeRange, SensorFileConfig, ThresholdConfig, TimeValueSourceConfig, VideoRecord
from pipeline_service import PipelineConfig, PipelineService
from sensor_service import SensorService
from video_service import VideoScanError, VideoService
from widgets.annotation_settings_dialog import AnnotationSettingsDialog
from widgets.map_widget import MapWidget, SEGMENT_COLORS, haversine_km
from widgets.navigation_import_dialog import NavigationImportDialog
from widgets.sensor_import_dialog import SensorImportDialog
from widgets.timeline_widget import TimelineWidget


# ===========================================================================
# WorkspaceStartupDialog — shown on every launch; user must pick an action
# ===========================================================================

class WorkspaceStartupDialog(QDialog):
    """Modal startup dialog that forces the user to open or create a workspace.

    Has no close button and ignores the Escape key so the app cannot be used
    without a workspace selected.  Shows a "Continue last session" option only
    when a previous session with a still-valid workspace directory is found.
    """

    OPEN     = "open"
    NEW      = "new"
    CONTINUE = "continue"

    def __init__(
        self,
        has_last_session: bool = False,
        last_session_path: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.choice: str | None = None

        self.setWindowTitle("Sampling Tool — Select Workspace")
        # Remove the ? help button and the close (X) button.
        self.setWindowFlags(
            Qt.Dialog | Qt.WindowTitleHint | Qt.CustomizeWindowHint
        )
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(28, 24, 28, 24)

        title = QLabel("Sampling Tool")
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)

        sub = QLabel("Select a workspace to begin.")
        sub.setStyleSheet("color: #555; margin-bottom: 6px;")
        layout.addWidget(sub)

        if has_last_session and last_session_path:
            ws_name = Path(last_session_path).name
            cont_btn = QPushButton(f"Continue last session  —  {ws_name}")
            cont_btn.setStyleSheet(
                "text-align: left; padding: 9px 14px; font-weight: bold; "
                "background: #e8f4fd; border: 1px solid #90caf9; border-radius: 4px;"
            )
            cont_btn.setToolTip(last_session_path)
            cont_btn.clicked.connect(lambda: self._choose(self.CONTINUE))
            layout.addWidget(cont_btn)

        open_btn = QPushButton("Open existing workspace…")
        open_btn.setStyleSheet("padding: 8px 14px;")
        open_btn.clicked.connect(lambda: self._choose(self.OPEN))
        layout.addWidget(open_btn)

        new_btn = QPushButton("Create new workspace…")
        new_btn.setStyleSheet("padding: 8px 14px;")
        new_btn.clicked.connect(lambda: self._choose(self.NEW))
        layout.addWidget(new_btn)

    def _choose(self, choice: str) -> None:
        self.choice = choice
        self.accept()

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key_Escape:
            return  # Block Escape
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        event.ignore()  # Block the window-close button


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
# OutputWorker — runs OutputService tasks in a background QThread
# ===========================================================================

class OutputWorker(QObject):
    """Runs a single OutputService method in a background QThread.

    Signals:
        finished(list[str]) — list of output file paths on success
        error(str)          — exception message on failure
        log(str)            — individual log line
        progress(int)       — 0–100 (currently unused; reserved for future use)
        status(str)         — human-readable step description
    """

    finished = Signal(list)
    error    = Signal(str)
    log      = Signal(str)
    progress = Signal(int)
    status   = Signal(str)

    def __init__(self, task: str, kwargs: dict) -> None:
        super().__init__()
        self.task   = task
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            import inspect
            from output_service import OutputService
            svc    = OutputService(log_fn=self.log.emit)
            method = getattr(svc, self.task)
            sig    = inspect.signature(method)
            # Only pass kwargs the method actually accepts (avoids TypeError when
            # _run_output_task injects interp_path/output_dir for tasks that don't need them)
            accepted = set(sig.parameters)
            filtered = {k: v for k, v in self.kwargs.items() if k in accepted}
            result = method(**filtered)
            self.finished.emit([result] if isinstance(result, str) else list(result))
        except Exception as exc:
            self.error.emit(str(exc))


# ===========================================================================
# PhotogrammetryWorker — runs photogrammetry_service tasks in a QThread
# ===========================================================================

class PhotogrammetryWorker(QObject):
    """Runs Metashape or COLMAP pipelines in a background QThread.

    Signals mirror OutputWorker so the same UI plumbing handles both.
    """

    finished = Signal(dict)   # products dict {type: path}
    error    = Signal(str)
    log      = Signal(str)
    progress = Signal(int)
    status   = Signal(str)

    def __init__(self, engine: str, kwargs: dict) -> None:
        super().__init__()
        self.engine = engine
        self.kwargs = kwargs

    def run(self) -> None:
        try:
            import photogrammetry_service as ps
            run_dir  = Path(self.kwargs["run_dir"])
            frame_dir = self.kwargs["frame_dir"]
            quality   = self.kwargs.get("quality", "normal")
            kw = {
                "run_dir":       run_dir,
                "frame_dir":     frame_dir,
                "quality":       quality,
                "build_dense":   self.kwargs.get("build_dense", True),
                "build_mesh":    self.kwargs.get("build_mesh", True),
                "build_texture": self.kwargs.get("build_texture", False),
                "log_fn":        self.log.emit,
            }
            if self.engine == "metashape":
                kw["nav_csv"] = self.kwargs.get("nav_csv")
                products = ps.run_metashape(**kw)
            else:
                kw["colmap_bin"] = self.kwargs.get("colmap_bin", "colmap")
                kw.pop("build_mesh", None)
                kw.pop("build_texture", None)
                kw.pop("nav_csv", None)
                products = ps.run_colmap(**kw)
            self.finished.emit(products)
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

    # Maximum number of points passed to the map widget for display.
    # Dense sensor data (10–20 Hz) is downsampled to this cap before rendering
    # so the colored trackline stays responsive.  Exports bypass this limit
    # and always use the full-resolution data from _raster_df.
    _MAX_DISPLAY_PTS: int = 5_000

    @staticmethod
    def _downsample_track(
        *arrays: "np.ndarray | None",
        max_pts: int = 5_000,
    ) -> "tuple[np.ndarray | None, ...]":
        """Uniformly stride a set of aligned arrays so the longest is ≤ max_pts.

        All arrays are sliced with the same stride so positional alignment is
        preserved.  None entries pass through unchanged.  Arrays already shorter
        than max_pts are returned as-is.
        """
        lengths = [len(a) for a in arrays if a is not None]
        if not lengths:
            return arrays
        n = max(lengths)
        if n <= max_pts:
            return arrays
        stride = max(1, n // max_pts)
        return tuple(a[::stride] if a is not None else None for a in arrays)

    def __init__(self):
        """Initialise all session state, build the UI, wire signals, and do an initial refresh."""
        super().__init__()
        self.setWindowTitle("Sampling Tool")
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

        # The job currently being built.  Always non-None.
        self.pending_job: Job = Job(job_id=1)

        # All jobs ever saved or executed (separate from pending_job).
        self.job_history: list[Job] = []

        # All segments the pipeline has actually sampled (the interval history).
        # Persisted in the workspace JSON; drives the history overlay mode.
        self.segment_history: list[SegmentRecord] = []

        # Saved threshold configurations (for reloading in the Threshold tab).
        self.threshold_history: list[ThresholdConfig] = []

        # Threshold intervals computed from the most recent analysis run.
        # Not persisted — recomputed when the user clicks Calculate.
        self._threshold_intervals: list[SelectedTimeRange] = []

        # True when the current pending_job was loaded from a completed history entry.
        # Cleared after execution or when a new job is created.
        self._loaded_from_completed_job: bool = False

        # Intervals staged in the Manual Interval Selection tab (not yet in a job).
        self._manual_staged: list[SelectedTimeRange] = []

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

        # Background thread and worker for output generation (GeoTIFF / PLY).
        self.output_worker_thread: QThread | None = None
        self.output_worker:        OutputWorker | None = None

        # Background thread and worker for photogrammetry (Metashape / COLMAP).
        self.photo_worker_thread: QThread | None = None
        self.photo_worker:        PhotogrammetryWorker | None = None

        # Cached engine detection results (populated on first sub-tab visit).
        self._photo_engines: dict[str, str | None] | None = None

        # Task stack (Stack dock) and its background runner.
        self._task_stack: TaskStack = TaskStack()
        self._stack_panel = None      # StackPanel | None (created in _build_docks)
        self.stack_worker_thread: QThread | None = None
        self.stack_worker = None  # StackWorker | None

        # Set to True when a build_full_interp pipeline run just completed,
        # so _on_pipeline_finished can show the right message and refresh channels.
        self._build_full_interp_run: bool = False

        # Workspace persistence flags; used to avoid re-saving when not needed.
        self.workspace_saved: bool = False
        self.workspace_path:  str  = ""

        # Currently selected 3D model run directories — tracked separately so that
        # clicking a sensor_3d run does not overwrite the nav target, and vice versa.
        self._selected_nav_run:    str | None = None
        self._selected_sensor_run: str | None = None

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
        # Show the workspace selection dialog once the event loop starts.
        QTimer.singleShot(0, self._show_startup_dialog)

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
        self.controls_tabs.addTab(self._build_inputs_tab(),              "Inputs")
        self.controls_tabs.addTab(self._build_threshold_tab(),           "Threshold Intervals")
        self.controls_tabs.addTab(self._build_manual_interval_tab(),     "Manual Intervals")
        # Jobs tab: still the place to review a job and add/delete its intervals.
        # (Product generation moved to the Task Stack dock, but job/interval
        # *management* still lives here.)
        self._jobs_tab_widget    = self._build_jobs_tab()
        self.controls_tabs.addTab(self._jobs_tab_widget,                 "Jobs")
        # Outputs is fully superseded by the Task Stack dock.  Its widget is
        # still constructed (the sampling / output / photogrammetry machinery
        # reads its controls) but kept off the tab bar.  The ref is held on self
        # so Qt does not garbage-collect its child widgets.
        self._outputs_tab_widget = self._build_outputs_tab()
        self.controls_tabs.setMinimumWidth(340)
        self.controls_tabs.setMaximumWidth(680)

        splitter.addWidget(self.controls_tabs)
        self.center_stack = QStackedWidget()
        self.center_stack.addWidget(self._build_timeline_panel())              # 0
        self.center_stack.addWidget(self._build_postprocessing_graph_panel()) # 1
        self.center_stack.addWidget(self._build_map_panel())                  # 2
        self.center_stack.addWidget(self._build_manual_map_panel())           # 3
        self.center_stack.addWidget(self._build_threshold_graph_panel())      # 4
        self.center_stack.addWidget(QWidget())                                # 5 — blank
        splitter.addWidget(self.center_stack)
        splitter.addWidget(self._build_summary_panel())

        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        splitter.setSizes([580, 520, 400])

        # Task-stack dock — bottom-LEFT (the primary orchestration surface).
        from stack_panel import StackPanel
        self._stack_panel = StackPanel(self._task_stack)
        self._stack_panel.set_providers(
            self._stack_available_jobs,
            self._stack_available_channels,
            self._stack_data_availability,
        )
        self._stack_panel.run_requested.connect(self._run_stack)
        self._stack_panel.tasks_changed.connect(self._on_tasks_changed)
        stack_dock = QDockWidget("Task Stack", self)
        stack_dock.setObjectName("StackDock")
        stack_dock.setWidget(self._stack_panel)
        stack_dock.setMinimumWidth(320)
        self.addDockWidget(Qt.BottomDockWidgetArea, stack_dock)

        # File tree / Workspace dock — bottom-RIGHT.
        self._workspace_panel = WorkspacePanel()
        self._workspace_panel.run_selected.connect(self._on_3d_run_selected)
        self._workspace_panel.make_slices_requested.connect(self._on_make_slices_requested)
        self._workspace_panel.open_in_viewer_requested.connect(self._open_ply_in_viewer)
        self._workspace_panel.open_in_metashape_requested.connect(self._open_in_metashape_gui)
        workspace_dock = QDockWidget("Workspace", self)
        workspace_dock.setObjectName("WorkspaceDock")
        workspace_dock.setWidget(self._workspace_panel)
        workspace_dock.setMinimumWidth(280)
        self.addDockWidget(Qt.BottomDockWidgetArea, workspace_dock)
        # Place the file tree to the RIGHT of the task stack.
        self.splitDockWidget(stack_dock, workspace_dock, Qt.Horizontal)

        # Claude AI assistant panel — right-side dock
        self._chat_panel = ChatPanel(context_fn=self._build_claude_context)
        chat_dock = QDockWidget("", self)
        chat_dock.setObjectName("ClaudeDock")
        chat_dock.setWidget(self._chat_panel)
        chat_dock.setMinimumWidth(300)
        chat_dock.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable
        )
        self.addDockWidget(Qt.RightDockWidgetArea, chat_dock)

    def _build_toolbar(self) -> None:
        """Build the File menu bar (replaces the old toolbar Run button)."""
        menu_bar  = self.menuBar()
        file_menu = menu_bar.addMenu("File")

        self.save_workspace_action  = file_menu.addAction("Save Workspace",   self._save_workspace)
        self.load_workspace_action  = file_menu.addAction("Load Workspace",   self._load_workspace)
        self.clear_workspace_action = file_menu.addAction("Clear Workspace",  self._clear_workspace)
        file_menu.addSeparator()
        self.save_config_action = file_menu.addAction("Save Config JSON", self._save_configuration)
        file_menu.addSeparator()
        file_menu.addAction("Export Raster Map…", self._raster_export_map)
        file_menu.addAction("Export Map…",        self._viz_export_map)
        file_menu.addSeparator()
        file_menu.addAction("Claude API Key…", self._show_api_key_dialog)

        # Keep a no-op run_action attribute so existing code that references it
        # (e.g., _set_processing_enabled) doesn't crash.  It is never displayed.
        from PySide6.QtGui import QAction  # type: ignore
        self.run_action = QAction("Run", self)

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

        # Workspace path display
        self.workspace_path_label = QLabel()
        self.workspace_path_label.setWordWrap(True)
        self.workspace_path_label.setStyleSheet("color: gray; font-size: 10px; padding: 2px 0px;")
        layout.addWidget(self.workspace_path_label)

        # Full-dataset interp_full.csv
        full_interp_row = QHBoxLayout()
        self.full_interp_hz_spin = QDoubleSpinBox()
        self.full_interp_hz_spin.setRange(0.001, 100.0)
        self.full_interp_hz_spin.setValue(1.0)
        self.full_interp_hz_spin.setDecimals(3)
        self.full_interp_hz_spin.setSuffix(" Hz")
        self.full_interp_hz_spin.setMaximumWidth(90)
        self.full_interp_build_btn = QPushButton("Build interp_full.csv")
        self.full_interp_build_btn.setToolTip(
            "Interpolates all nav and sensor channels across the entire navigation time range "
            "at the specified sample rate.  Saved as interp_full.csv in the workspace folder "
            "alongside workspace.json.  No frames are extracted."
        )
        full_interp_row.addWidget(self.full_interp_hz_spin)
        full_interp_row.addWidget(self.full_interp_build_btn)
        layout.addLayout(full_interp_row)

        layout.addStretch()
        return tab

    # -----------------------------------------------------------------------
    # Threshold Intervals tab
    # -----------------------------------------------------------------------

    def _build_threshold_tab(self) -> QWidget:
        """Threshold Interval Selection tab — sidebar only.
        Sensor graphs live in the center panel (_build_threshold_graph_panel).
        """
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        tab_layout.addWidget(scroll, stretch=1)
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # --- Constraint builder ---
        constraint_group = QGroupBox("Threshold Constraints  (all must be satisfied)")
        self._constraint_group_layout = QVBoxLayout(constraint_group)
        self._constraint_group_layout.setSpacing(4)
        self._constraint_rows: list[dict] = []   # each entry: {row_widget, channel_combo, min_edit, max_edit, enabled_check}

        add_constraint_btn = QPushButton("+ Add constraint")
        add_constraint_btn.clicked.connect(self._threshold_add_constraint)
        self._constraint_group_layout.addWidget(add_constraint_btn)
        layout.addWidget(constraint_group)

        # Add first constraint row immediately
        self._threshold_add_constraint()

        # Min duration
        dur_row = QHBoxLayout()
        dur_row.addWidget(QLabel("Min interval duration:"))
        self.threshold_min_dur_spin = QDoubleSpinBox()
        self.threshold_min_dur_spin.setRange(0.1, 3600.0)
        self.threshold_min_dur_spin.setValue(2.0)
        self.threshold_min_dur_spin.setSuffix(" s")
        self.threshold_min_dur_spin.setMaximumWidth(100)
        dur_row.addWidget(self.threshold_min_dur_spin)
        dur_row.addStretch()
        layout.addLayout(dur_row)

        # Calculate button
        self.threshold_calculate_btn = QPushButton("Calculate Intervals")
        self.threshold_calculate_btn.setStyleSheet(
            "font-weight: bold; padding: 6px 16px;"
        )
        self.threshold_calculate_btn.clicked.connect(self._threshold_calculate)
        layout.addWidget(self.threshold_calculate_btn)

        # Results list
        results_group = QGroupBox("Qualifying Intervals")
        results_layout = QVBoxLayout(results_group)
        self.threshold_results_label = QLabel("0 intervals found.")
        self.threshold_results_label.setStyleSheet("font-size: 11px; color: gray;")
        results_layout.addWidget(self.threshold_results_label)
        self.threshold_results_list = QListWidget()
        self.threshold_results_list.setMinimumHeight(100)
        self.threshold_results_list.setMaximumHeight(180)
        self.threshold_results_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.threshold_results_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.threshold_results_list.customContextMenuRequested.connect(
            lambda pos: self._interval_list_context_menu(self.threshold_results_list, pos, source="threshold_results")
        )
        results_layout.addWidget(self.threshold_results_list)
        layout.addWidget(results_group)

        # Saved threshold history
        history_group = QGroupBox("Saved Threshold Configs")
        history_layout = QVBoxLayout(history_group)
        self.threshold_history_list = QListWidget()
        self.threshold_history_list.setMaximumHeight(120)
        self.threshold_history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.threshold_history_list.customContextMenuRequested.connect(
            lambda pos: self._threshold_history_context_menu(pos)
        )
        history_layout.addWidget(self.threshold_history_list)
        load_saved_btn = QPushButton("Load selected config")
        load_saved_btn.clicked.connect(self._threshold_load_saved)
        history_layout.addWidget(load_saved_btn)
        layout.addWidget(history_group)

        layout.addStretch()

        # Add to Job button pinned at bottom
        self.threshold_add_to_job_btn = QPushButton("Add Selected Intervals to Job")
        self.threshold_add_to_job_btn.setStyleSheet(
            "background: #1565c0; color: white; font-weight: bold; padding: 8px;"
        )
        self.threshold_add_to_job_btn.clicked.connect(self._threshold_add_to_job)
        tab_layout.addWidget(self.threshold_add_to_job_btn)

        return tab

    # -----------------------------------------------------------------------
    # Manual Interval Selection tab
    # -----------------------------------------------------------------------

    def _build_manual_interval_tab(self) -> QWidget:
        """Manual Interval Selection tab — sidebar only.  Map in center panel."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)

        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(6)
        scroll.setWidget(sidebar)

        # Two-click pick controls
        pick_group = QGroupBox("Pick Interval from Track")
        pick_layout = QVBoxLayout(pick_group)
        self.manual_pick_button = QPushButton("Pick Interval")
        self.manual_pick_button.setCheckable(True)
        self.manual_pick_button.setToolTip(
            "Click two points on the trackline to define an interval."
        )
        self.manual_pick_status = QLabel("Click to activate, then click two points on the track.")
        self.manual_pick_status.setWordWrap(True)
        self.manual_pick_status.setStyleSheet("font-size: 10px; color: gray;")
        pick_layout.addWidget(self.manual_pick_button)
        pick_layout.addWidget(self.manual_pick_status)
        side_layout.addWidget(pick_group)

        # Staged interval list
        staged_group = QGroupBox("Staged Intervals (not yet in job)")
        staged_layout = QVBoxLayout(staged_group)
        self.manual_staged_list = QListWidget()
        self.manual_staged_list.setMinimumHeight(80)
        self.manual_staged_list.setMaximumHeight(160)
        self.manual_staged_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.manual_staged_list.customContextMenuRequested.connect(
            lambda pos: self._interval_list_context_menu(self.manual_staged_list, pos, source="manual_staged")
        )
        staged_layout.addWidget(self.manual_staged_list)
        clear_staged_btn = QPushButton("Clear Staged")
        clear_staged_btn.clicked.connect(self._manual_clear_staged)
        staged_layout.addWidget(clear_staged_btn)
        side_layout.addWidget(staged_group)

        # Recorded interval history
        rec_group = QGroupBox("Recorded Intervals (previously sampled)")
        rec_layout = QVBoxLayout(rec_group)
        self.manual_history_list = QListWidget()
        self.manual_history_list.setMinimumHeight(80)
        self.manual_history_list.setMaximumHeight(180)
        self.manual_history_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.manual_history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.manual_history_list.customContextMenuRequested.connect(
            lambda pos: self._interval_list_context_menu(self.manual_history_list, pos, source="manual_history")
        )
        self.manual_history_list.itemSelectionChanged.connect(
            self._manual_history_selection_changed
        )
        rec_layout.addWidget(self.manual_history_list)
        add_from_hist_btn = QPushButton("Stage Selected from History")
        add_from_hist_btn.clicked.connect(self._manual_add_from_history)
        rec_layout.addWidget(add_from_hist_btn)
        side_layout.addWidget(rec_group)

        # --- Sensor Tracklines panel ---
        raster_group = QGroupBox("Sensor Tracklines")
        raster_layout = QVBoxLayout(raster_group)
        raster_layout.setSpacing(4)

        self.raster_status_label = QLabel("No trackline loaded.")
        self.raster_status_label.setWordWrap(True)
        self.raster_status_label.setStyleSheet("color: gray; font-size: 11px;")
        raster_layout.addWidget(self.raster_status_label)

        raster_btn_row = QHBoxLayout()
        self.raster_load_button = QPushButton("Load from Inputs")
        self.raster_load_button.setToolTip(
            "Build sensor tracklines from the configured navigation and sensor sources."
        )
        self.raster_clear_button = QPushButton("Clear")
        self.raster_clear_button.setMaximumWidth(55)
        raster_btn_row.addWidget(self.raster_load_button)
        raster_btn_row.addWidget(self.raster_clear_button)
        raster_layout.addLayout(raster_btn_row)

        raster_chan_row = QHBoxLayout()
        raster_chan_row.addWidget(QLabel("Channel:"))
        self.raster_channel_combo = QComboBox()
        self.raster_channel_combo.addItem("None")
        raster_chan_row.addWidget(self.raster_channel_combo, stretch=1)
        raster_chan_row.addSpacing(4)
        self.raster_scale_combo = QComboBox()
        self.raster_scale_combo.addItems(["Linear", "Logarithmic"])
        self.raster_scale_combo.setFixedWidth(100)
        raster_chan_row.addWidget(self.raster_scale_combo)
        raster_layout.addLayout(raster_chan_row)

        raster_max_row = QHBoxLayout()
        raster_max_row.addWidget(QLabel("Max (blue):"))
        self.raster_max_spin = QDoubleSpinBox()
        self.raster_max_spin.setRange(-1e9, 1e9)
        self.raster_max_spin.setDecimals(4)
        self.raster_max_spin.setValue(1.0)
        self.raster_max_spin.setEnabled(False)
        raster_max_row.addWidget(self.raster_max_spin, stretch=1)
        raster_layout.addLayout(raster_max_row)

        raster_min_row = QHBoxLayout()
        raster_min_row.addWidget(QLabel("Min (red):"))
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
        self.raster_export_map_button.setEnabled(False)
        self.raster_export_csv_button = QPushButton("Export CSV…")
        self.raster_export_csv_button.setEnabled(False)
        raster_export_row.addWidget(self.raster_export_map_button)
        raster_export_row.addWidget(self.raster_export_csv_button)
        raster_layout.addLayout(raster_export_row)

        self.raster_export_geojson_button = QPushButton("Export QGIS GeoJSON…")
        self.raster_export_geojson_button.setEnabled(False)
        raster_layout.addWidget(self.raster_export_geojson_button)

        self.raster_export_analysis_button = QPushButton("Export Sensor Analysis…")
        self.raster_export_analysis_button.setEnabled(False)
        raster_layout.addWidget(self.raster_export_analysis_button)

        side_layout.addWidget(raster_group)

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
        side_layout.addLayout(width_row)

        side_layout.addStretch()

        # Add to Job button pinned at bottom of the tab (outside scroll)
        self.manual_add_to_job_btn = QPushButton("Add Staged Intervals to Job")
        self.manual_add_to_job_btn.setStyleSheet(
            "background: #1565c0; color: white; font-weight: bold; padding: 8px;"
        )
        self.manual_add_to_job_btn.clicked.connect(self._manual_add_to_job)
        outer.addWidget(self.manual_add_to_job_btn)

        return tab

    # -----------------------------------------------------------------------
    # Jobs tab
    # -----------------------------------------------------------------------

    def _build_jobs_tab(self) -> QWidget:
        """Jobs tab: sidebar-only.  The map lives in the center panel."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)

        sidebar = QWidget()
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(6, 6, 6, 6)
        side_layout.setSpacing(6)
        scroll.setWidget(sidebar)

        # Current job
        job_group = QGroupBox("Current Job")
        job_layout = QVBoxLayout(job_group)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name:"))
        self.job_name_edit = QLineEdit()
        self.job_name_edit.setPlaceholderText("Required before execution…")
        self.job_name_edit.setStyleSheet("border: 1px solid #aaa;")
        name_row.addWidget(self.job_name_edit, stretch=1)
        job_layout.addLayout(name_row)

        self.job_interval_list = QListWidget()
        self.job_interval_list.setMinimumHeight(100)
        self.job_interval_list.setMaximumHeight(200)
        self.job_interval_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.job_interval_list.customContextMenuRequested.connect(
            lambda pos: self._interval_list_context_menu(self.job_interval_list, pos, source="pending_job")
        )
        job_layout.addWidget(self.job_interval_list)

        job_btn_row = QHBoxLayout()
        self.job_clear_btn = QPushButton("Clear Job")
        self.job_new_btn = QPushButton("New Job")
        self.job_save_btn = QPushButton("Save Job")
        self.job_save_btn.setToolTip("Save this job to history without executing.")
        job_btn_row.addWidget(self.job_clear_btn)
        job_btn_row.addWidget(self.job_new_btn)
        job_btn_row.addWidget(self.job_save_btn)
        job_layout.addLayout(job_btn_row)
        side_layout.addWidget(job_group)

        # Job history
        hist_group = QGroupBox("Job History")
        hist_layout = QVBoxLayout(hist_group)
        self.job_history_list = QListWidget()
        self.job_history_list.setMinimumHeight(100)
        self.job_history_list.setMaximumHeight(200)
        self.job_history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.job_history_list.customContextMenuRequested.connect(
            self._job_history_context_menu
        )
        hist_layout.addWidget(self.job_history_list)
        load_job_btn = QPushButton("Load Selected Job")
        load_job_btn.clicked.connect(self._job_load_from_history)
        hist_layout.addWidget(load_job_btn)
        side_layout.addWidget(hist_group)

        # Interval histories (both types, for reference / adding to job)
        int_hist_group = QGroupBox("Interval Histories")
        int_hist_layout = QVBoxLayout(int_hist_group)

        int_hist_layout.addWidget(QLabel("Manual (blue):"))
        self.jobs_manual_history_list = QListWidget()
        self.jobs_manual_history_list.setMaximumHeight(110)
        self.jobs_manual_history_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.jobs_manual_history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.jobs_manual_history_list.customContextMenuRequested.connect(
            lambda pos: self._interval_list_context_menu(self.jobs_manual_history_list, pos, source="manual_history")
        )
        self.jobs_manual_history_list.itemSelectionChanged.connect(self._jobs_history_selection_changed)
        int_hist_layout.addWidget(self.jobs_manual_history_list)

        add_manual_btn = QPushButton("Add selected manual intervals to job")
        add_manual_btn.clicked.connect(self._jobs_add_manual_history_to_job)
        int_hist_layout.addWidget(add_manual_btn)

        int_hist_layout.addWidget(QLabel("Threshold (green):"))
        self.jobs_threshold_history_list = QListWidget()
        self.jobs_threshold_history_list.setMaximumHeight(110)
        self.jobs_threshold_history_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.jobs_threshold_history_list.setContextMenuPolicy(Qt.CustomContextMenu)
        self.jobs_threshold_history_list.customContextMenuRequested.connect(
            lambda pos: self._interval_list_context_menu(self.jobs_threshold_history_list, pos, source="threshold_history")
        )
        self.jobs_threshold_history_list.itemSelectionChanged.connect(self._jobs_history_selection_changed)
        int_hist_layout.addWidget(self.jobs_threshold_history_list)

        add_thresh_btn = QPushButton("Add selected threshold intervals to job")
        add_thresh_btn.clicked.connect(self._jobs_add_threshold_history_to_job)
        int_hist_layout.addWidget(add_thresh_btn)

        side_layout.addWidget(int_hist_group)
        side_layout.addStretch()
        return tab

    # -----------------------------------------------------------------------
    # Outputs tab (Navigation / Sensors / Video sub-tabs)
    # -----------------------------------------------------------------------

    def _build_outputs_tab(self) -> QWidget:
        """Top-level Outputs tab containing three sub-tabs: Navigation, Sensors, Video."""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(4, 4, 4, 0)
        layout.setSpacing(4)

        # Data source selector — determines which interp CSV drives all generation
        src_group = QGroupBox("Data source")
        src_row = QHBoxLayout(src_group)
        src_row.setContentsMargins(8, 4, 8, 6)
        src_row.setSpacing(6)
        src_label = QLabel("Source:")
        src_label.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)
        src_row.addWidget(src_label)
        self.output_source_combo = QComboBox()
        self.output_source_combo.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.output_source_combo.setToolTip(
            "Full dataset: uses interp_full.csv covering the entire survey.\n"
            "Segment entries: use the interp.csv from a specific job interval,\n"
            "  restricting all outputs to that interval's data."
        )
        self.output_source_combo.addItem("Full dataset (interp_full.csv)", userData=None)
        src_row.addWidget(self.output_source_combo, stretch=1)
        refresh_src_btn = QPushButton("↻")
        refresh_src_btn.setFixedWidth(28)
        refresh_src_btn.setToolTip("Re-scan workspace for completed segments")
        refresh_src_btn.clicked.connect(self._refresh_output_source_combo)
        src_row.addWidget(refresh_src_btn)
        layout.addWidget(src_group)

        self.outputs_subtabs = QTabWidget()
        self.outputs_subtabs.addTab(self._build_nav_outputs_subtab(),         "Navigation")
        self.outputs_subtabs.addTab(self._build_sensor_outputs_subtab(),      "Sensors")
        self.outputs_subtabs.addTab(self._build_video_processing_subtab(),    "Video")
        self.outputs_subtabs.addTab(self._build_photogrammetry_subtab(),      "Photogrammetry")
        layout.addWidget(self.outputs_subtabs, stretch=1)
        return tab

    # -----------------------------------------------------------------------
    # Task stack — providers, plan building, execution
    # -----------------------------------------------------------------------

    def _stack_available_jobs(self) -> list[tuple]:
        """Return [(job_id, name), ...] for jobs that have intervals."""
        jobs = []
        if self.pending_job.intervals:
            jobs.append((self.pending_job.job_id, self.pending_job.name or f"Job #{self.pending_job.job_id}"))
        for j in self.job_history:
            if j.intervals:
                jobs.append((j.job_id, j.name or f"Job #{j.job_id}"))
        return jobs

    def _stack_available_channels(self) -> list[str]:
        """Return sensor channels found in interp_full.csv (empty if not built)."""
        interp = self._interp_full_path()
        if not interp or not Path(interp).exists():
            return []
        try:
            from output_service import sensor_channels_from_csv
            return sensor_channels_from_csv(interp)
        except Exception:
            return []

    def _stack_data_availability(self) -> dict:
        """Return which data inputs are currently loaded (gates the Create menu)."""
        interp = self._interp_full_path()
        return {
            "interp": bool(interp) and Path(interp).exists(),
            "sensor": bool(self.sensor_files),
            "nav":    self.navigation_file is not None,
            "video":  bool(self.videos),
        }

    def _find_job(self, job_id: int) -> "Job | None":
        return next(
            (j for j in [self.pending_job] + list(self.job_history) if j.job_id == job_id),
            None,
        )

    def _resolve_task_target(self, task: "Task") -> tuple:
        """Resolve a task's target to (scope_id, interp_path, output_dir, label)."""
        tgt = task.target
        if tgt.get("kind") == "job":
            job = self._find_job(int(tgt.get("job_id", -1)))
            if job is not None and job.intervals:
                interp  = self._get_filtered_interp_for_job(job)
                out_dir = str(Path(self.workspace_path) / self._job_output_dirname(job) / "outputs")
                return (f"job_{job.job_id}", interp, out_dir, job.name or f"Job #{job.job_id}")
        return ("full", self._interp_full_path(), self._outputs_root(), "Full dataset")

    # ------------------------------------------------------------------

    _FILL_3CH = {"IDW fill": "idw", "Kriging fill": "kriging",
                 "RBF fill": "rbf", "No fill": "none",
                 "Trackline only (no fill)": "none", "Trackline only": "none"}

    def _build_task_plan(self) -> list[dict]:
        """Turn the ordered Task list into runner step dicts (preserving user order)."""
        plan: list[dict] = []
        for task in self._task_stack.tasks:
            scope_id, interp_path, output_dir, tlabel = self._resolve_task_target(task)
            tag = f"  [{tlabel}]"
            t = task.task_type
            s = task.settings
            channels = task.channels or self._stack_available_channels()

            if t == "sampling":
                config = self._build_sampling_config(task)
                if config is None:
                    continue
                plan.append({
                    "label": f"{task.type_label}{tag}",
                    "product_type": "sampling", "scope_id": scope_id,
                    "task_id": task.task_id,   # runner records outputs keyed by this
                    "channel": None, "method": None, "engine": None,
                    "kwargs": {}, "config": config,
                })

            elif t == "nav_3d":
                plan.append(self._step(t, scope_id, None, "generate_nav_3d_ply", {
                    "interp_path": interp_path, "output_dir": output_dir,
                    "cell_size": float(s.get("cell_size", 1.0)),
                }, "Nav Trackline PLY" + tag))

            elif t == "nav_2d":
                plan.append(self._step(t, scope_id, None, "generate_nav_2d_geotiff", {
                    "interp_path": interp_path, "output_dir": output_dir,
                    "cell_size_m": float(s.get("cell_size", 5.0)),
                    "crs_mode": "wgs84" if s.get("crs") == "WGS84" else "utm",
                }, "Nav Depth GeoTIFF" + tag))

            elif t == "sensor_3d":
                for ch in channels:
                    plan.append(self._step(t, scope_id, ch, "generate_sensor_3d_ply", {
                        "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                        "cell_size": float(s.get("cell_size", 1.0)),
                        "aggregation": s.get("aggregation", "mean"),
                        "fill_method": self._FILL_3CH.get(s.get("fill", "IDW fill"), "idw"),
                        "zero_mask_pct": float(s.get("zero_mask", 5.0)),
                    }, f"Sensor 3D PLY — {ch}" + tag))

            elif t == "sensor_2d":
                for ch in channels:
                    plan.append(self._step(t, scope_id, ch, "generate_sensor_2d_geotiff", {
                        "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                        "cell_size_m": float(s.get("cell_size", 5.0)),
                        "crs_mode": "wgs84" if s.get("crs") == "WGS84" else "utm",
                        "fill_method": self._FILL_3CH.get(s.get("fill", "IDW fill"), "idw"),
                    }, f"Sensor 2D GeoTIFF — {ch}" + tag))

            elif t == "depth_slice_geotiffs":
                fill = ("idw" if "IDW" in s.get("fill", "IDW fill")
                        else "rbf" if "RBF" in s.get("fill", "") else "none")
                for ch in channels:
                    plan.append(self._step(t, scope_id, ch, "generate_depth_slice_geotiffs", {
                        "interp_path": interp_path, "output_dir": output_dir, "channel": ch,
                        "altitude_step": float(s.get("altitude_step", 5.0)),
                        "cell_size_m": float(s.get("cell_size", 2.0)),
                        "fill_method": fill,
                    }, f"Depth-Slice GeoTIFFs — {ch}" + tag))

            elif t == "sensor_slices":
                color = "grayscale" if "grayscale" in s.get("color", "") else "rgb"
                for ch in channels:
                    plan.append(self._step(t, scope_id, ch, None, {
                        "altitude_step": float(s.get("altitude_step", 5.0)),
                        "pixels_per_cell": int(s.get("ppc", 4)),
                        "color_mode": color,
                        "local_norm": bool(s.get("local_norm", False)),
                        "_run_glob": str(Path(output_dir) / "sensor_3d" / ch),
                    }, f"PNG Depth Slices — {ch}" + tag))

            elif t == "photogrammetry":
                # Frame source: either linked to a sampling task (depends_on) or manual.
                # We skip the "dir must exist" check here because depends_on dirs are
                # resolved at runtime by the runner after sampling has produced them.
                engine = "colmap" if s.get("engine", "Metashape") == "COLMAP" else "metashape"
                plan.append({
                    "label": f"Photogrammetry ({s.get('engine', 'Metashape')})" + tag,
                    "product_type": "photogrammetry", "scope_id": scope_id,
                    "channel": None, "method": None, "engine": engine,
                    "depends_on_task_id": task.depends_on,   # None → use frame_dir
                    "kwargs": {
                        # output_root + job_id: runner calls prepare_run_dir per segment
                        "output_root": str(Path(output_dir) / "photogrammetry"),
                        "job_id":      0,
                        # manual fallback dir (empty when using depends_on)
                        "frame_dir":   s.get("frame_dir", "").strip(),
                        "nav_csv":     interp_path if s.get("use_nav_reference", True) else None,
                        # Alignment
                        "align_accuracy":     s.get("align_accuracy", "High"),
                        "key_point_limit":    int(s.get("key_point_limit", 40000)),
                        "tie_point_limit":    int(s.get("tie_point_limit", 10000)),
                        "generic_preselect":  bool(s.get("generic_preselect", True)),
                        "reference_preselect": bool(s.get("reference_preselect", True)),
                        "adaptive_fitting":   bool(s.get("adaptive_fitting", True)),
                        "reset_cameras":      bool(s.get("reset_cameras", False)),
                        # Dense cloud
                        "build_dense":   bool(s.get("build_dense", True)),
                        "dense_quality": s.get("dense_quality", "Medium"),
                        "depth_filter":  s.get("depth_filter", "Moderate"),
                        "reuse_depth":   bool(s.get("reuse_depth", False)),
                        # Mesh
                        "build_mesh":         bool(s.get("build_mesh", False)),
                        "mesh_surface":       s.get("mesh_surface", "Arbitrary"),
                        "mesh_faces":         s.get("mesh_faces", "Medium"),
                        "mesh_source":        s.get("mesh_source", "Dense cloud"),
                        "mesh_vertex_colors": bool(s.get("mesh_vertex_colors", True)),
                        # Texture
                        "build_texture":      bool(s.get("build_texture", False)),
                        "texture_size":       int(s.get("texture_size", 4096)),
                        "texture_blending":   s.get("texture_blending", "Mosaic"),
                        "texture_fill_holes": bool(s.get("texture_fill_holes", True)),
                        # Export & project
                        "export_dense_ply": bool(s.get("export_dense_ply", True)),
                        "export_mesh_obj":  bool(s.get("export_mesh_obj", False)),
                        "save_project":     bool(s.get("save_project", True)),
                        # Georeference
                        "use_nav_reference": bool(s.get("use_nav_reference", True)),
                        "nav_accuracy_h":    float(s.get("nav_accuracy_h", 0.1)),
                        "nav_accuracy_v":    float(s.get("nav_accuracy_v", 0.5)),
                        # COLMAP
                        "max_features": int(s.get("max_features", 8192)),
                        "matcher":      s.get("matcher", "Exhaustive"),
                        "run_mvs":      bool(s.get("run_mvs", True)),
                    },
                })

            elif t == "qgis_project":
                plan.append(self._step(t, scope_id, None, "generate_qgis_project", {
                    "output_dir": output_dir,
                    "project_name": s.get("project_name", "EPR Survey"),
                }, "QGIS Project" + tag))

        return plan

    def _build_sampling_config(self, task: "Task"):
        """Build a PipelineConfig for a sampling task (frame extraction)."""
        from pipeline_service import PipelineConfig
        video_dir = self.video_dir_edit.text().strip()
        if not video_dir or not self.videos:
            self.log_text.append("Stack: sampling task skipped — no videos loaded.")
            return None

        # Intervals come from the target job, or span all video coverage for 'full'.
        tgt = task.target
        if tgt.get("kind") == "job":
            job = self._find_job(int(tgt.get("job_id", -1)))
            intervals = list(job.intervals) if job else []
            out_name  = f"sampling_{task.task_id}_{self._job_output_dirname(job)}" if job else f"sampling_{task.task_id}"
        else:
            starts = [v.start_time for v in self.videos]
            ends   = [v.end_time for v in self.videos]
            intervals = [SelectedTimeRange(start_time=min(starts), end_time=max(ends), source="stack_full")]
            out_name  = f"sampling_{task.task_id}_full"

        if not intervals:
            self.log_text.append(f"Stack: sampling task {task.task_id} has no intervals — skipping.")
            return None

        s = task.settings
        steps = ["extract_frames"]
        if s.get("rasters"):
            steps.append("generate_sensor_rasters")
        if s.get("annotate"):
            steps.append("annotate_frames")

        out_dir = str(Path(self.workspace_path) / out_name)
        return PipelineConfig(
            video_directory=Path(video_dir),
            output_directory=Path(out_dir),
            job_id=task.task_id,
            video_filename_time_format=self.video_format_edit.text().strip(),
            videos=self.videos,
            selected_intervals=intervals,
            navigation_file=self.navigation_file,
            sensor_files=self.sensor_files,
            depth_source=self.depth_source,
            speed_source=self.speed_source,
            frame_rate=float(s.get("frame_rate", 1.0)),
            sampling_mode="dynamic" if s.get("mode") == "dynamic" else "fixed",
            dynamic_target_spacing_m=float(s.get("spacing_m", 1.0)),
            frame_quality=s.get("quality", "high"),
            sample_images=True,
            generate_sensor_rasters=bool(s.get("rasters", False)),
            annotate_frames=bool(s.get("annotate", False)),
            annotation_config=self.annotation_config,
            selected_steps=steps,
            workspace_directory=self.workspace_path,
        )

    @staticmethod
    def _step(product_type, scope_id, channel, method, kwargs, label) -> dict:
        return {
            "label": label, "product_type": product_type, "scope_id": scope_id,
            "channel": channel, "method": method, "engine": None, "kwargs": kwargs,
        }

    def _on_tasks_changed(self) -> None:
        """Persist the stack after any add/edit/remove/reorder (if a workspace exists)."""
        self.workspace_saved = False

    def _run_stack(self) -> None:
        if not self.workspace_path:
            QMessageBox.warning(self, "No workspace", "Save the workspace first.")
            return
        if self._stack_worker_is_running() or self._output_worker_is_running():
            QMessageBox.warning(self, "Busy", "A task is already running.")
            return
        if not self._task_stack.tasks:
            QMessageBox.information(self, "Empty stack",
                                    "Create at least one task before running.")
            return

        plan = self._build_task_plan()
        if not plan:
            QMessageBox.information(self, "Nothing to run",
                                    "No runnable steps — check task targets and data.")
            return

        self.log_text.clear()
        self.log_text.append(f"Stack: {len(plan)} step(s) queued.")
        self._status_label.setText("Running stack…")
        self._progress_bar.setValue(0)
        self._stack_result_paths: list[str] = []
        if self._stack_panel is not None:
            self._stack_panel.set_running(True)

        from stack_runner import StackWorker
        thread = QThread(self)
        worker = StackWorker(plan)
        self.stack_worker_thread = thread
        self.stack_worker        = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.step_started.connect(self._on_stack_step_started)
        worker.step_finished.connect(self._on_stack_step_finished)
        worker.step_failed.connect(self._on_stack_step_failed)
        worker.finished.connect(self._on_stack_finished)
        worker.error.connect(self._on_stack_error)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_stack_worker)
        thread.start()

    def _stack_worker_is_running(self) -> bool:
        try:
            return bool(self.stack_worker_thread and self.stack_worker_thread.isRunning())
        except RuntimeError:
            self.stack_worker_thread = None
            self.stack_worker = None
            return False

    def _clear_stack_worker(self) -> None:
        self.stack_worker_thread = None
        self.stack_worker = None
        if self._stack_panel is not None:
            self._stack_panel.set_running(False)

    def _on_stack_step_started(self, index: int, total: int, label: str) -> None:
        self._status_label.setText(f"Stack {index}/{total}: {label}")
        pct = int((index - 1) / total * 100) if total else 0
        self._progress_bar.setValue(pct)
        self.log_text.append(f"▶ [{index}/{total}] {label}")

    def _on_stack_step_finished(self, label: str, paths: list) -> None:
        self.log_text.append(f"  ✓ {label} — {len(paths)} file(s)")
        self._stack_result_paths.extend(paths)
        self._workspace_panel.refresh()

    def _on_stack_step_failed(self, label: str, message: str) -> None:
        self.log_text.append(f"  ✗ {label}: {message}")

    def _on_stack_finished(self, summary: dict) -> None:
        self._progress_bar.setValue(100)
        c, f, sk = summary.get("completed", 0), summary.get("failed", 0), summary.get("skipped", 0)
        msg = f"Stack finished: {c} completed, {f} failed, {sk} skipped."
        self._status_label.setText(msg)
        self.log_text.append(msg)
        self._workspace_panel.refresh()
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "Stack complete", msg))

    def _on_stack_error(self, message: str) -> None:
        self._status_label.setText("Stack failed.")
        self.log_text.append(f"Stack failed: {message}")
        QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Stack failed", message))


    def _build_nav_outputs_subtab(self) -> QWidget:
        """Navigation sub-tab: 2D GeoTIFF and 3D trackline PLY from interp_full.csv."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # Prerequisite warning
        self.nav_outputs_warning = QLabel()
        self.nav_outputs_warning.setWordWrap(True)
        self.nav_outputs_warning.setStyleSheet(
            "color: #c0392b; font-size: 11px; padding: 4px 6px; "
            "background: #fdecea; border: 1px solid #f5c6cb; border-radius: 4px;"
        )
        self.nav_outputs_warning.setVisible(False)
        layout.addWidget(self.nav_outputs_warning)

        # CRS group
        crs_group = QGroupBox("Coordinate Reference System")
        crs_layout = QHBoxLayout(crs_group)
        self.nav_crs_group = QButtonGroup(self)
        self.nav_crs_utm_radio = QRadioButton("UTM (auto-detected from data)")
        self.nav_crs_wgs84_radio = QRadioButton("WGS84 (lat/lon)")
        self.nav_crs_utm_radio.setChecked(True)
        self.nav_crs_group.addButton(self.nav_crs_utm_radio)
        self.nav_crs_group.addButton(self.nav_crs_wgs84_radio)
        crs_layout.addWidget(self.nav_crs_utm_radio)
        crs_layout.addWidget(self.nav_crs_wgs84_radio)
        crs_layout.addStretch()
        layout.addWidget(crs_group)

        # 2D GeoTIFF
        nav_2d_group = QGroupBox("2D Output — GeoTIFF")
        nav_2d_layout = QHBoxLayout(nav_2d_group)
        nav_2d_layout.addWidget(QLabel("Cell size:"))
        self.nav_2d_cell_spin = QDoubleSpinBox()
        self.nav_2d_cell_spin.setRange(0.1, 1000.0)
        self.nav_2d_cell_spin.setValue(5.0)
        self.nav_2d_cell_spin.setSuffix(" m")
        self.nav_2d_cell_spin.setMaximumWidth(90)
        nav_2d_layout.addWidget(self.nav_2d_cell_spin)
        self.nav_2d_generate_btn = QPushButton("Generate Nav GeoTIFF")
        nav_2d_layout.addWidget(self.nav_2d_generate_btn, stretch=1)
        layout.addWidget(nav_2d_group)

        # 3D PLY
        nav_3d_group = QGroupBox("3D Output — PLY (UTM, colour = water depth)")
        nav_3d_layout = QHBoxLayout(nav_3d_group)
        nav_3d_layout.addWidget(QLabel("Cell size:"))
        self.nav_3d_cell_spin = QDoubleSpinBox()
        self.nav_3d_cell_spin.setRange(0.1, 100.0)
        self.nav_3d_cell_spin.setValue(1.0)
        self.nav_3d_cell_spin.setSuffix(" m")
        self.nav_3d_cell_spin.setMaximumWidth(90)
        nav_3d_layout.addWidget(self.nav_3d_cell_spin)
        self.nav_3d_generate_btn = QPushButton("Generate Trackline PLY")
        nav_3d_layout.addWidget(self.nav_3d_generate_btn, stretch=1)
        layout.addWidget(nav_3d_group)

        # 3D Raster Slices
        nav_slices_group = QGroupBox("3D Raster Slices — PNG  (one image per depth band)")
        nav_slices_layout = QFormLayout(nav_slices_group)
        self.nav_slices_target_label = QLabel("(click a run in the Workspace panel)")
        self.nav_slices_target_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 10px;"
        )
        self.nav_slices_target_label.setWordWrap(True)
        nav_slices_layout.addRow("Target run:", self.nav_slices_target_label)
        self.nav_slices_inherited_label = QLabel("")
        self.nav_slices_inherited_label.setStyleSheet(
            "color: #666; font-size: 10px; font-style: italic;"
        )
        self.nav_slices_inherited_label.setVisible(False)
        nav_slices_layout.addRow("From target:", self.nav_slices_inherited_label)
        self.nav_slices_step_spin = QDoubleSpinBox()
        self.nav_slices_step_spin.setRange(0.1, 1000.0)
        self.nav_slices_step_spin.setValue(5.0)
        self.nav_slices_step_spin.setSuffix(" m")
        self.nav_slices_step_spin.setToolTip(
            "Vertical depth interval per slice.\n"
            "Smaller values = more slices, finer depth resolution."
        )
        nav_slices_layout.addRow("Altitude step:", self.nav_slices_step_spin)
        self.nav_slices_ppc_spin = QSpinBox()
        self.nav_slices_ppc_spin.setRange(1, 20)
        self.nav_slices_ppc_spin.setValue(4)
        self.nav_slices_ppc_spin.setToolTip(
            "Output pixels per grid cell. Use >1 for larger, blockier images."
        )
        nav_slices_layout.addRow("Pixels per cell:", self.nav_slices_ppc_spin)
        self.nav_slices_generate_btn = QPushButton("Generate Depth Slices")
        self.nav_slices_generate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        nav_slices_layout.addRow(self.nav_slices_generate_btn)
        layout.addWidget(nav_slices_group)

        layout.addStretch()
        return tab

    def _build_sensor_outputs_subtab(self) -> QWidget:
        """Sensors sub-tab: per-channel 2D GeoTIFF and 3D PLY from interp_full.csv."""
        tab = QWidget()
        outer = QVBoxLayout(tab)
        outer.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        outer.addWidget(scroll, stretch=1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # Prerequisite warning
        self.sensor_outputs_warning = QLabel()
        self.sensor_outputs_warning.setWordWrap(True)
        self.sensor_outputs_warning.setStyleSheet(
            "color: #c0392b; font-size: 11px; padding: 4px 6px; "
            "background: #fdecea; border: 1px solid #f5c6cb; border-radius: 4px;"
        )
        self.sensor_outputs_warning.setVisible(False)
        layout.addWidget(self.sensor_outputs_warning)

        # Regenerate reminder
        self.sensor_regen_hint = QLabel(
            "ℹ  Regenerate interp_full.csv after adding new sensor channels."
        )
        self.sensor_regen_hint.setWordWrap(True)
        self.sensor_regen_hint.setStyleSheet(
            "color: #555; font-size: 10px; padding: 2px 4px;"
        )
        layout.addWidget(self.sensor_regen_hint)

        # CRS group
        crs_group = QGroupBox("Coordinate Reference System  (2D outputs)")
        crs_layout = QHBoxLayout(crs_group)
        self.sensor_crs_group = QButtonGroup(self)
        self.sensor_crs_utm_radio = QRadioButton("UTM (auto-detected)")
        self.sensor_crs_wgs84_radio = QRadioButton("WGS84 (lat/lon)")
        self.sensor_crs_utm_radio.setChecked(True)
        self.sensor_crs_group.addButton(self.sensor_crs_utm_radio)
        self.sensor_crs_group.addButton(self.sensor_crs_wgs84_radio)
        crs_layout.addWidget(self.sensor_crs_utm_radio)
        crs_layout.addWidget(self.sensor_crs_wgs84_radio)
        crs_layout.addStretch()
        layout.addWidget(crs_group)

        # Channel selection
        chan_group = QGroupBox("Channel Selection")
        chan_layout = QVBoxLayout(chan_group)
        chan_header = QHBoxLayout()
        chan_header.addWidget(QLabel("Select channels to include in outputs:"))
        self.sensor_refresh_channels_btn = QPushButton("Refresh")
        self.sensor_refresh_channels_btn.setMaximumWidth(70)
        self.sensor_refresh_channels_btn.setToolTip(
            "Re-read interp_full.csv to pick up any new channels."
        )
        chan_header.addWidget(self.sensor_refresh_channels_btn)
        chan_layout.addLayout(chan_header)

        self.sensor_channel_list = QListWidget()
        self.sensor_channel_list.setMinimumHeight(80)
        self.sensor_channel_list.setMaximumHeight(140)
        chan_layout.addWidget(self.sensor_channel_list)

        chan_btn_row = QHBoxLayout()
        self.sensor_select_all_btn = QPushButton("Select All")
        self.sensor_select_none_btn = QPushButton("Select None")
        self.sensor_select_all_btn.setMaximumWidth(80)
        self.sensor_select_none_btn.setMaximumWidth(80)
        chan_btn_row.addWidget(self.sensor_select_all_btn)
        chan_btn_row.addWidget(self.sensor_select_none_btn)
        chan_btn_row.addStretch()
        chan_layout.addLayout(chan_btn_row)
        layout.addWidget(chan_group)

        # 2D GeoTIFF
        s2d_group = QGroupBox("2D Outputs — GeoTIFF  (one file per selected channel)")
        s2d_layout = QFormLayout(s2d_group)
        self.sensor_2d_cell_spin = QDoubleSpinBox()
        self.sensor_2d_cell_spin.setRange(0.1, 1000.0)
        self.sensor_2d_cell_spin.setValue(5.0)
        self.sensor_2d_cell_spin.setSuffix(" m")
        s2d_layout.addRow("Cell size:", self.sensor_2d_cell_spin)
        self.sensor_2d_fill_combo = QComboBox()
        self.sensor_2d_fill_combo.addItems(["IDW fill", "Kriging fill", "RBF fill", "Trackline only (no fill)"])
        s2d_layout.addRow("Fill method:", self.sensor_2d_fill_combo)
        self.sensor_2d_generate_btn = QPushButton("Generate 2D GeoTIFF(s)")
        self.sensor_2d_generate_btn.setStyleSheet(
            "font-weight: bold; padding: 6px;"
        )
        s2d_layout.addRow(self.sensor_2d_generate_btn)
        layout.addWidget(s2d_group)

        # 3D PLY
        s3d_group = QGroupBox("3D Outputs — PLY  (UTM; one file per selected channel)")
        s3d_layout = QFormLayout(s3d_group)
        self.sensor_3d_cell_spin = QDoubleSpinBox()
        self.sensor_3d_cell_spin.setRange(0.1, 100.0)
        self.sensor_3d_cell_spin.setValue(1.0)
        self.sensor_3d_cell_spin.setSuffix(" m")
        s3d_layout.addRow("Cell size:", self.sensor_3d_cell_spin)
        self.sensor_3d_agg_combo = QComboBox()
        self.sensor_3d_agg_combo.addItems(["mean", "min", "max"])
        s3d_layout.addRow("Aggregation:", self.sensor_3d_agg_combo)
        self.sensor_3d_fill_combo = QComboBox()
        self.sensor_3d_fill_combo.addItems(["IDW fill", "Kriging fill", "RBF fill", "No fill"])
        self.sensor_3d_fill_combo.setToolTip(
            "IDW fill: fast, recommended for most datasets.\n"
            "Kriging fill: Ordinary Kriging per depth layer — honours spatial structure,\n"
            "  slower than IDW; requires pykrige.\n"
            "RBF fill: smooth but may fail when depth is nearly constant\n"
            "  (polynomial basis becomes rank-deficient for 1-D tracklines in 3-D)."
        )
        s3d_layout.addRow("Fill method:", self.sensor_3d_fill_combo)
        self.sensor_3d_zero_spin = QDoubleSpinBox()
        self.sensor_3d_zero_spin.setRange(0.0, 50.0)
        self.sensor_3d_zero_spin.setValue(5.0)
        self.sensor_3d_zero_spin.setSuffix(" %")
        self.sensor_3d_zero_spin.setSingleStep(1.0)
        self.sensor_3d_zero_spin.setDecimals(0)
        self.sensor_3d_zero_spin.setToolTip(
            "Generate a second *_signal PLY where the bottom N % of values\n"
            "are hidden, making near-zero background invisible.\n"
            "Set to 0 to skip the signal PLY."
        )
        s3d_layout.addRow("Near-zero mask:", self.sensor_3d_zero_spin)
        self.sensor_3d_generate_btn = QPushButton("Generate 3D PLY(s)")
        self.sensor_3d_generate_btn.setStyleSheet(
            "font-weight: bold; padding: 6px;"
        )
        s3d_layout.addRow(self.sensor_3d_generate_btn)
        layout.addWidget(s3d_group)

        # 3D Raster Slices — target-based; cell/agg/fill come from the selected 3D run
        s_slices_group = QGroupBox(
            "3D Raster Slices — PNG  (select a 3D run in the Workspace panel)"
        )
        s_slices_layout = QFormLayout(s_slices_group)
        self.sensor_slices_target_label = QLabel("(click a run in the Workspace panel)")
        self.sensor_slices_target_label.setStyleSheet(
            "color: #888; font-style: italic; font-size: 10px;"
        )
        self.sensor_slices_target_label.setWordWrap(True)
        s_slices_layout.addRow("Target run:", self.sensor_slices_target_label)
        self.sensor_slices_inherited_label = QLabel("")
        self.sensor_slices_inherited_label.setStyleSheet(
            "color: #666; font-size: 10px; font-style: italic;"
        )
        self.sensor_slices_inherited_label.setVisible(False)
        s_slices_layout.addRow("From target:", self.sensor_slices_inherited_label)
        self.sensor_slices_step_spin = QDoubleSpinBox()
        self.sensor_slices_step_spin.setRange(0.1, 1000.0)
        self.sensor_slices_step_spin.setValue(5.0)
        self.sensor_slices_step_spin.setSuffix(" m")
        self.sensor_slices_step_spin.setToolTip(
            "Vertical depth interval per slice.\n"
            "Smaller values = more slices, finer depth resolution."
        )
        s_slices_layout.addRow("Altitude step:", self.sensor_slices_step_spin)
        self.sensor_slices_ppc_spin = QSpinBox()
        self.sensor_slices_ppc_spin.setRange(1, 20)
        self.sensor_slices_ppc_spin.setValue(4)
        self.sensor_slices_ppc_spin.setToolTip(
            "Output pixels per grid cell. Use >1 for larger, blockier images."
        )
        s_slices_layout.addRow("Pixels per cell:", self.sensor_slices_ppc_spin)
        self.sensor_slices_color_combo = QComboBox()
        self.sensor_slices_color_combo.addItems(["rgb (viridis)", "grayscale"])
        s_slices_layout.addRow("Color mode:", self.sensor_slices_color_combo)
        self.sensor_slices_log_check = QCheckBox("Log scale")
        self.sensor_slices_log_check.setToolTip(
            "Apply log scaling before colour mapping.\n"
            "Useful for data with a large dynamic range."
        )
        s_slices_layout.addRow("", self.sensor_slices_log_check)
        self.sensor_slices_local_norm_check = QCheckBox("Local normalization")
        self.sensor_slices_local_norm_check.setToolTip(
            "Normalize each slice to its own min/max value.\n"
            "Use when all values are in a narrow absolute range (e.g. low CO₂):\n"
            "reveals spatial variation even when the signal is uniformly weak.\n\n"
            "Leave unchecked to keep slices colour-comparable across depths."
        )
        s_slices_layout.addRow("", self.sensor_slices_local_norm_check)
        self.sensor_slices_pct_spin = QDoubleSpinBox()
        self.sensor_slices_pct_spin.setRange(50.0, 100.0)
        self.sensor_slices_pct_spin.setValue(100.0)
        self.sensor_slices_pct_spin.setSuffix(" %")
        self.sensor_slices_pct_spin.setDecimals(1)
        self.sensor_slices_pct_spin.setToolTip(
            "Clip values above this percentile before colour mapping.\n"
            "Set below 100 to suppress outliers (e.g. 99 or 95)."
        )
        s_slices_layout.addRow("Percentile cap:", self.sensor_slices_pct_spin)
        self.sensor_slices_generate_btn = QPushButton("Generate Depth Slices")
        self.sensor_slices_generate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        s_slices_layout.addRow(self.sensor_slices_generate_btn)
        layout.addWidget(s_slices_group)

        # Depth-slice GeoTIFFs — geo-referenced plan-view rasters per depth band
        s_geo_slices_group = QGroupBox(
            "Depth-Slice GeoTIFFs  (Float32, UTM; one .tif per depth band)"
        )
        s_geo_slices_group.setToolTip(
            "Produces geo-referenced GeoTIFF rasters, one per depth band.\n"
            "Unlike the PNG slices above, these carry actual sensor values (not\n"
            "colour-mapped pixels) and a UTM spatial reference — load them\n"
            "directly into QGIS and apply any colormap."
        )
        s_geo_layout = QFormLayout(s_geo_slices_group)
        self.sensor_geo_slices_step_spin = QDoubleSpinBox()
        self.sensor_geo_slices_step_spin.setRange(0.1, 1000.0)
        self.sensor_geo_slices_step_spin.setValue(5.0)
        self.sensor_geo_slices_step_spin.setSuffix(" m")
        s_geo_layout.addRow("Altitude step:", self.sensor_geo_slices_step_spin)
        self.sensor_geo_slices_cell_spin = QDoubleSpinBox()
        self.sensor_geo_slices_cell_spin.setRange(0.1, 100.0)
        self.sensor_geo_slices_cell_spin.setValue(2.0)
        self.sensor_geo_slices_cell_spin.setSuffix(" m")
        s_geo_layout.addRow("Cell size:", self.sensor_geo_slices_cell_spin)
        self.sensor_geo_slices_fill_combo = QComboBox()
        self.sensor_geo_slices_fill_combo.addItems(["IDW fill", "RBF fill", "Trackline only"])
        s_geo_layout.addRow("Fill method:", self.sensor_geo_slices_fill_combo)
        self.sensor_geo_slices_generate_btn = QPushButton("Generate Depth-Slice GeoTIFFs")
        self.sensor_geo_slices_generate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        s_geo_layout.addRow(self.sensor_geo_slices_generate_btn)
        layout.addWidget(s_geo_slices_group)

        # QGIS project export
        qgis_group  = QGroupBox("QGIS Project Export")
        qgis_layout = QFormLayout(qgis_group)
        self.qgis_project_name_edit = QLineEdit()
        self.qgis_project_name_edit.setText("EPR Survey")
        self.qgis_project_name_edit.setPlaceholderText("Project name shown in QGIS")
        qgis_layout.addRow("Project name:", self.qgis_project_name_edit)
        self.qgis_generate_btn = QPushButton("Generate QGIS Project (.qgs)")
        self.qgis_generate_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.qgis_generate_btn.setToolTip(
            "Scans the outputs directory for all GeoTIFFs and writes a .qgs\n"
            "project file.  Open it in QGIS 3.x to see all 2D and depth-slice\n"
            "rasters pre-loaded and styled — no additional importing needed."
        )
        qgis_layout.addRow(self.qgis_generate_btn)
        layout.addWidget(qgis_group)

        layout.addStretch()
        return tab

    # -----------------------------------------------------------------------
    # Video Processing sub-tab (was: Combined Processing tab)
    # -----------------------------------------------------------------------

    def _build_video_processing_subtab(self) -> QWidget:
        """Video / frame extraction and post-processing sub-tab (formerly the Processing tab).

        All pipeline settings appear here.  The Execute button is at the bottom.
        """
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        tab_layout.addWidget(scroll, stretch=1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # ── Frame Extraction ─────────────────────────────────────────────
        # This group is disabled when frames already exist for the current job.
        # The user must click "Create New Sampling Job…" to re-enable it.
        self.frame_extract_group = QGroupBox("Frame Extraction")
        fex_layout = QVBoxLayout(self.frame_extract_group)
        fex_layout.setSpacing(6)

        # Sampling status label — always visible, shows whether frames have been extracted
        self.sampling_status_label = QLabel()
        self.sampling_status_label.setWordWrap(True)
        self.sampling_status_label.setStyleSheet(
            "font-size: 11px; color: #6c757d; padding: 2px 0px;"
        )
        fex_layout.addWidget(self.sampling_status_label)

        # Sampling settings form
        fex_form = QFormLayout()
        fex_form.setContentsMargins(0, 0, 0, 0)

        self.sample_images_check = QCheckBox("Sample images  (uncheck for sensor-only run)")
        self.sample_images_check.setChecked(True)
        fex_form.addRow(self.sample_images_check)

        self.sampling_mode_combo = QComboBox()
        self.sampling_mode_combo.addItems(["Fixed rate", "Dynamic spacing"])
        fex_form.addRow("Mode:", self.sampling_mode_combo)

        self.frame_rate_spin = QDoubleSpinBox()
        self.frame_rate_spin.setRange(0.01, 30.0)
        self.frame_rate_spin.setSingleStep(0.5)
        self.frame_rate_spin.setValue(1.0)
        self.frame_rate_spin.setSuffix(" Hz")
        fex_form.addRow("Frame rate:", self.frame_rate_spin)

        self.dynamic_spacing_spin = QDoubleSpinBox()
        self.dynamic_spacing_spin.setRange(0.1, 100.0)
        self.dynamic_spacing_spin.setValue(2.0)
        self.dynamic_spacing_spin.setSuffix(" m")
        fex_form.addRow("Target spacing:", self.dynamic_spacing_spin)

        self.dynamic_min_freq_spin = QDoubleSpinBox()
        self.dynamic_min_freq_spin.setRange(0.001, 10.0)
        self.dynamic_min_freq_spin.setValue(0.1)
        self.dynamic_min_freq_spin.setSuffix(" Hz")
        fex_form.addRow("Min frequency:", self.dynamic_min_freq_spin)

        self.frame_quality_combo = QComboBox()
        self.frame_quality_combo.addItems(["Original", "High (90%)", "Medium (75%)", "Low (50%)"])
        fex_form.addRow("Quality:", self.frame_quality_combo)

        fex_layout.addLayout(fex_form)

        # "Create New Sampling Job" button — only visible when extraction is locked
        self.new_sampling_job_button = QPushButton("Create New Sampling Job…")
        self.new_sampling_job_button.setStyleSheet(
            "background: #e65100; color: white; font-weight: bold; padding: 6px;"
        )
        self.new_sampling_job_button.setToolTip(
            "Frames already exist for this job.  Click to create a new job "
            "(same intervals, new name) and re-enable frame extraction."
        )
        self.new_sampling_job_button.hide()
        self.new_sampling_job_button.clicked.connect(self._on_new_sampling_job_clicked)
        fex_layout.addWidget(self.new_sampling_job_button)

        layout.addWidget(self.frame_extract_group)

        # ── Post-Processing Steps ────────────────────────────────────────
        # Always accessible — these operate on already-extracted frames.
        steps_group = QGroupBox("Post-Processing Steps")
        steps_layout = QVBoxLayout(steps_group)

        self.postprocess_extract_check = QCheckBox("Extract frames")
        self.postprocess_extract_check.setVisible(False)   # controlled by frame_extract_group
        self.postprocess_generate_rasters_check = QCheckBox("Generate sensor GeoTIFFs")
        self.generate_rasters_check = self.postprocess_generate_rasters_check  # alias

        self.postprocess_annotate_check = QCheckBox("Annotate extracted frames")
        self.annotate_frames_check = self.postprocess_annotate_check           # alias
        postprocess_annotate_row = QHBoxLayout()
        postprocess_annotate_row.addWidget(self.postprocess_annotate_check)
        self.annotate_configure_button = QPushButton("Configure…")
        self.annotate_configure_button.setMaximumWidth(85)
        self.postprocess_annotate_configure_button = self.annotate_configure_button  # alias
        postprocess_annotate_row.addWidget(self.annotate_configure_button)

        self.postprocess_clahe_check = QCheckBox("Apply CLAHE enhancement")
        self.postprocess_update_master_check = QCheckBox("Rebuild interp.csv from current inputs")
        self.postprocess_update_master_check.setToolTip(
            "Re-interpolates all configured nav channels (lat, lon, alt, depth, heading, pitch, roll) "
            "and sensor channels onto the existing frame timestamps.  "
            "Run this alone (uncheck other steps) to add or change CSV columns without re-sampling."
        )
        self.postprocess_geo_txt_check = QCheckBox("Generate WebODM geo.txt")

        self.postprocess_generate_rasters_check.setChecked(True)
        self.postprocess_annotate_check.setChecked(False)
        self.postprocess_clahe_check.setChecked(False)
        self.postprocess_update_master_check.setChecked(True)
        self.postprocess_geo_txt_check.setChecked(False)

        steps_layout.addWidget(self.postprocess_generate_rasters_check)
        steps_layout.addLayout(postprocess_annotate_row)
        steps_layout.addWidget(self.postprocess_clahe_check)
        steps_layout.addWidget(self.postprocess_update_master_check)
        steps_layout.addWidget(self.postprocess_geo_txt_check)
        layout.addWidget(steps_group)

        # --- CLAHE Parameters ---
        clahe_group = QGroupBox("CLAHE Parameters")
        clahe_form = QFormLayout(clahe_group)
        self.clahe_clip_limit_spin = QDoubleSpinBox()
        self.clahe_clip_limit_spin.setRange(0.5, 40.0)
        self.clahe_clip_limit_spin.setValue(2.0)
        clahe_form.addRow("Clip limit:", self.clahe_clip_limit_spin)
        self.clahe_tile_size_spin = QSpinBox()
        self.clahe_tile_size_spin.setRange(4, 64)
        self.clahe_tile_size_spin.setValue(8)
        clahe_form.addRow("Tile grid size:", self.clahe_tile_size_spin)
        self.clahe_clip_limit_spin.setEnabled(False)
        self.clahe_tile_size_spin.setEnabled(False)
        layout.addWidget(clahe_group)

        # Threshold-based pipeline filtering (altitude/depth/speed/per-channel) has
        # been superseded by the Threshold Intervals tab, which lets the user define
        # intervals visually from sensor graphs.  The state variables below are kept
        # for workspace round-trip compatibility but are not surfaced in the UI.

        layout.addStretch()

        # --- Execute button pinned at bottom ---
        self.execute_job_btn = QPushButton("Execute Job")
        self.execute_job_btn.setStyleSheet(
            "background: #2e7d32; color: white; font-weight: bold; font-size: 14px; padding: 10px;"
        )
        self.execute_job_btn.setToolTip(
            "Run the pipeline for the current job.  The job must be named first."
        )
        self.execute_job_btn.clicked.connect(self._run_pipeline)
        tab_layout.addWidget(self.execute_job_btn)

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
        self.viz_utm_zone_label = QLabel("UTM: —")
        self.viz_utm_zone_label.setStyleSheet(
            "font-size: 10px; color: #555; padding: 0 8px;"
        )
        self.viz_utm_zone_label.setToolTip(
            "UTM zone derived from the mean longitude of the loaded GPS trackline"
        )
        header_row.addWidget(self.viz_utm_zone_label)
        header_row.addStretch()
        self.viz_history_mode_button = QPushButton("History Overlay")
        self.viz_history_mode_button.setCheckable(True)
        self.viz_history_mode_button.setToolTip(
            "Toggle interval history overlay on the trackline.\n"
            "Yellow = pending job  ·  Blue = manual history  ·  Green = threshold history"
        )
        header_row.addWidget(self.viz_history_mode_button)
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

    def _build_threshold_graph_panel(self) -> QWidget:
        """Center panel for the Threshold Intervals tab: scrollable sensor graphs."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel("Sensor Data")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        header.addWidget(title)
        subtitle = QLabel("Hover over a graph to read the value at any point in time.")
        subtitle.setStyleSheet("color: gray; font-size: 10px;")
        header.addWidget(subtitle)
        header.addStretch()
        layout.addLayout(header)

        # Scrollable graph area — populated by _refresh_threshold_graphs()
        graph_scroll = QScrollArea()
        graph_scroll.setWidgetResizable(True)
        graph_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        graph_container = QWidget()
        self._threshold_graph_layout = QVBoxLayout(graph_container)
        self._threshold_graph_layout.setSpacing(6)
        self._threshold_graph_layout.setContentsMargins(0, 0, 0, 0)
        self.threshold_graph_placeholder = QLabel(
            "Configure navigation and sensor sources on the Inputs tab,\n"
            "then switch to this tab to view graphs and set thresholds."
        )
        self.threshold_graph_placeholder.setStyleSheet("color: gray; font-style: italic;")
        self.threshold_graph_placeholder.setAlignment(Qt.AlignCenter)
        self._threshold_graph_layout.addWidget(self.threshold_graph_placeholder)
        self._threshold_graph_layout.addStretch()
        graph_container.setLayout(self._threshold_graph_layout)
        graph_scroll.setWidget(graph_container)
        layout.addWidget(graph_scroll, stretch=1)
        return panel

    def _build_manual_map_panel(self) -> QWidget:
        """Center panel for the Manual Intervals tab: trackline + two-click picking."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(4)

        header = QHBoxLayout()
        title = QLabel("Manual Interval Selection")
        title.setStyleSheet("font-size: 15px; font-weight: bold;")
        header.addWidget(title)
        header.addStretch()
        layout.addLayout(header)

        subtitle = QLabel(
            "Click 'Pick Interval' in the sidebar, then click two points on the trackline to define an interval."
        )
        subtitle.setStyleSheet("color: gray; font-size: 10px;")
        layout.addWidget(subtitle)

        self.manual_nav_status_label = QLabel("")
        self.manual_nav_status_label.setWordWrap(True)
        self.manual_nav_status_label.setStyleSheet("font-size: 10px; padding: 2px 4px;")
        layout.addWidget(self.manual_nav_status_label)

        self.manual_map_widget = MapWidget()
        layout.addWidget(self.manual_map_widget, stretch=1)
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

    # -----------------------------------------------------------------------
    # Photogrammetry sub-tab
    # -----------------------------------------------------------------------

    def _build_photogrammetry_subtab(self) -> QWidget:
        """Photogrammetry sub-tab: Metashape / COLMAP reconstruction from extracted frames."""
        tab = QWidget()
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        tab_layout.addWidget(scroll, stretch=1)

        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        scroll.setWidget(content)

        # ── Engine ────────────────────────────────────────────────────────────
        engine_group = QGroupBox("Photogrammetry Engine")
        engine_form  = QFormLayout(engine_group)

        self.photo_engine_meta_radio   = QRadioButton("Metashape Professional")
        self.photo_engine_colmap_radio = QRadioButton("COLMAP  (free / open source)")
        self.photo_engine_meta_radio.setChecked(True)
        engine_form.addRow("Engine:", self.photo_engine_meta_radio)
        engine_form.addRow("",        self.photo_engine_colmap_radio)

        self.photo_meta_status_label = QLabel("Not detected")
        self.photo_meta_status_label.setStyleSheet("color: #888; font-size: 10px;")
        engine_form.addRow("Metashape:", self.photo_meta_status_label)

        self.photo_colmap_status_label = QLabel("Not detected")
        self.photo_colmap_status_label.setStyleSheet("color: #888; font-size: 10px;")
        engine_form.addRow("COLMAP:", self.photo_colmap_status_label)

        detect_btn = QPushButton("Detect Engines")
        detect_btn.clicked.connect(self._detect_photo_engines)
        engine_form.addRow(detect_btn)
        layout.addWidget(engine_group)

        # ── Source ────────────────────────────────────────────────────────────
        source_group = QGroupBox("Frame Source")
        source_form  = QFormLayout(source_group)

        self.photo_frame_dir_edit = QLineEdit()
        self.photo_frame_dir_edit.setPlaceholderText(
            "Path to extracted frames directory…"
        )
        self.photo_frame_dir_edit.setToolTip(
            "Directory containing JPEG/PNG frames extracted by the pipeline.\n"
            "Typically: job_NNN/segment_NNN/frames/"
        )
        browse_btn = QPushButton("Browse…")
        browse_btn.clicked.connect(self._browse_photo_frame_dir)
        frame_row = QWidget()
        frame_row_layout = QHBoxLayout(frame_row)
        frame_row_layout.setContentsMargins(0, 0, 0, 0)
        frame_row_layout.addWidget(self.photo_frame_dir_edit)
        frame_row_layout.addWidget(browse_btn)
        source_form.addRow("Frames dir:", frame_row)

        self.photo_frame_count_label = QLabel("—")
        self.photo_frame_count_label.setStyleSheet("color: #666; font-size: 10px;")
        source_form.addRow("Frames found:", self.photo_frame_count_label)
        self.photo_frame_dir_edit.textChanged.connect(self._refresh_photo_frame_count)

        self.photo_nav_seed_check = QCheckBox("Pre-seed camera positions from nav data")
        self.photo_nav_seed_check.setChecked(True)
        self.photo_nav_seed_check.setToolTip(
            "Use lat/lon/alt from interp_full.csv to initialise camera reference\n"
            "locations before alignment. Improves accuracy and reduces alignment\n"
            "failures for long tracklines. Metashape engine only."
        )
        source_form.addRow("", self.photo_nav_seed_check)
        layout.addWidget(source_group)

        # ── Settings ──────────────────────────────────────────────────────────
        settings_group = QGroupBox("Processing Settings")
        settings_form  = QFormLayout(settings_group)

        self.photo_quality_combo = QComboBox()
        self.photo_quality_combo.addItems(["Draft", "Normal", "High", "Highest"])
        self.photo_quality_combo.setCurrentIndex(1)
        self.photo_quality_combo.setToolTip(
            "Draft:   fastest, lowest detail — useful for quick alignment check\n"
            "Normal:  balanced speed/quality for most dives\n"
            "High:    slower; use for close-range high-detail scenes\n"
            "Highest: very slow; GPU strongly recommended"
        )
        settings_form.addRow("Quality:", self.photo_quality_combo)

        self.photo_dense_check = QCheckBox("Dense point cloud")
        self.photo_dense_check.setChecked(True)
        self.photo_dense_check.setToolTip(
            "Build MVS dense reconstruction (slow; GPU strongly recommended).\n"
            "Without this, only the sparse SfM cloud is produced."
        )
        settings_form.addRow("", self.photo_dense_check)

        self.photo_mesh_check = QCheckBox("Mesh  (requires dense cloud)")
        self.photo_mesh_check.setChecked(True)
        self.photo_mesh_check.setToolTip(
            "Build a triangulated mesh from the dense cloud.\n"
            "Metashape engine only."
        )
        settings_form.addRow("", self.photo_mesh_check)

        self.photo_texture_check = QCheckBox("Texture  (requires mesh)")
        self.photo_texture_check.setChecked(False)
        self.photo_texture_check.setToolTip(
            "Project original frame colours onto the mesh.\n"
            "Metashape engine only. Adds significant processing time."
        )
        settings_form.addRow("", self.photo_texture_check)

        # Keep mesh/texture checkboxes in sync with dense toggle
        self.photo_dense_check.toggled.connect(
            lambda on: self.photo_mesh_check.setEnabled(on)
        )
        self.photo_mesh_check.toggled.connect(
            lambda on: self.photo_texture_check.setEnabled(on and self.photo_dense_check.isChecked())
        )

        layout.addWidget(settings_group)

        # ── Actions ───────────────────────────────────────────────────────────
        actions_group = QGroupBox("Actions")
        actions_layout = QVBoxLayout(actions_group)

        self.photo_run_btn = QPushButton("Run Photogrammetry")
        self.photo_run_btn.setStyleSheet("font-weight: bold; padding: 6px;")
        self.photo_run_btn.setToolTip(
            "Run the selected engine headlessly using the settings above.\n"
            "Progress appears in the Processing Log panel."
        )
        self.photo_run_btn.clicked.connect(self._run_photogrammetry)
        actions_layout.addWidget(self.photo_run_btn)

        self.photo_open_gui_btn = QPushButton("Open Project in Metashape / COLMAP GUI")
        self.photo_open_gui_btn.setToolTip(
            "Open the most recent photogrammetry project in its native GUI.\n"
            "For Metashape: opens the .psx with full editing capabilities.\n"
            "For COLMAP: opens the GUI with the existing database.\n\n"
            "Use this when automatic alignment fails or you need fine control."
        )
        self.photo_open_gui_btn.setEnabled(False)
        self.photo_open_gui_btn.clicked.connect(self._open_photo_in_gui)
        actions_layout.addWidget(self.photo_open_gui_btn)

        layout.addWidget(actions_group)

        # Track most-recent run dir for the "Open in GUI" button
        self._last_photo_run_dir: str = ""

        layout.addStretch()
        return tab

    # -----------------------------------------------------------------------
    # Photogrammetry helpers
    # -----------------------------------------------------------------------

    def _detect_photo_engines(self) -> None:
        """Probe for Metashape and COLMAP and update status labels."""
        import photogrammetry_service as ps
        self._photo_engines = ps.detect_engines()

        meta_path  = self._photo_engines.get("metashape")
        colmap_cmd = self._photo_engines.get("colmap")

        if meta_path:
            self.photo_meta_status_label.setText(f"Found — {meta_path}")
            self.photo_meta_status_label.setStyleSheet("color: #2a9d2a; font-size: 10px;")
        else:
            # Distinguish "installed on the Windows host but not importable from
            # WSL" from "not installed at all" so the message is actionable.
            reason = ps.metashape_unavailable_reason() or (
                "Not found — install Metashape and ensure its Python API is importable"
            )
            self.photo_meta_status_label.setText(reason)
            self.photo_meta_status_label.setWordWrap(True)
            self.photo_meta_status_label.setStyleSheet("color: #c0392b; font-size: 10px;")

        if colmap_cmd:
            self.photo_colmap_status_label.setText(f"Found — {colmap_cmd}")
            self.photo_colmap_status_label.setStyleSheet("color: #2a9d2a; font-size: 10px;")
        else:
            self.photo_colmap_status_label.setText("Not found — install COLMAP and add to PATH")
            self.photo_colmap_status_label.setStyleSheet("color: #c0392b; font-size: 10px;")

    def _browse_photo_frame_dir(self) -> None:
        d = QFileDialog.getExistingDirectory(self, "Select Frame Directory")
        if d:
            self.photo_frame_dir_edit.setText(d)

    def _refresh_photo_frame_count(self) -> None:
        d = self.photo_frame_dir_edit.text().strip()
        if not d or not Path(d).is_dir():
            self.photo_frame_count_label.setText("—")
            return
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff"}
        n = sum(1 for f in Path(d).iterdir() if f.suffix.lower() in exts)
        self.photo_frame_count_label.setText(f"{n} image{'s' if n != 1 else ''}")

    def _photo_worker_is_running(self) -> bool:
        try:
            return bool(self.photo_worker_thread and self.photo_worker_thread.isRunning())
        except RuntimeError:
            self.photo_worker_thread = None
            self.photo_worker = None
            return False

    def _clear_photo_worker(self) -> None:
        self.photo_worker_thread = None
        self.photo_worker = None

    def _run_photogrammetry(self) -> None:
        if not self.workspace_path:
            QMessageBox.warning(self, "No workspace", "Save the workspace first.")
            return
        if self._photo_worker_is_running():
            QMessageBox.warning(self, "Busy", "A photogrammetry run is already in progress.")
            return

        frame_dir = self.photo_frame_dir_edit.text().strip()
        if not frame_dir or not Path(frame_dir).is_dir():
            QMessageBox.warning(self, "No frames", "Select a valid frames directory first.")
            return

        engine = "metashape" if self.photo_engine_meta_radio.isChecked() else "colmap"

        # Detect engines on first run if not cached
        if self._photo_engines is None:
            self._detect_photo_engines()
        if not self._photo_engines.get(engine):
            QMessageBox.critical(
                self, "Engine not available",
                f"{engine.title()} was not detected on this system.\n"
                "Run 'Detect Engines' for details."
            )
            return

        import photogrammetry_service as ps
        output_dir = str(Path(self._outputs_root()) / "photogrammetry")
        run_dir    = ps.prepare_run_dir(output_dir, job_id=0)

        quality = self.photo_quality_combo.currentText().lower()
        kwargs  = {
            "run_dir":       str(run_dir),
            "frame_dir":     frame_dir,
            "quality":       quality,
            "build_dense":   self.photo_dense_check.isChecked(),
            "build_mesh":    self.photo_mesh_check.isChecked() and self.photo_mesh_check.isEnabled(),
            "build_texture": self.photo_texture_check.isChecked() and self.photo_texture_check.isEnabled(),
        }
        if engine == "metashape" and self.photo_nav_seed_check.isChecked():
            kwargs["nav_csv"] = self._interp_full_path()
        if engine == "colmap" and self._photo_engines.get("colmap"):
            kwargs["colmap_bin"] = self._photo_engines["colmap"]

        self._last_photo_run_dir = str(run_dir)
        self.log_text.clear()
        self._status_label.setText("Photogrammetry running…")
        self._progress_bar.setValue(0)

        thread = QThread(self)
        worker = PhotogrammetryWorker(engine, kwargs)
        self.photo_worker_thread = thread
        self.photo_worker        = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.error.connect(self._on_photo_error)
        worker.finished.connect(self._on_photo_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_photo_worker)
        thread.start()

    def _on_photo_finished(self, products: dict) -> None:
        self._status_label.setText("Photogrammetry complete.")
        self._progress_bar.setValue(100)
        self.photo_open_gui_btn.setEnabled(True)
        lines = [f"  {k}: {v}" for k, v in products.items()]
        msg   = "Photogrammetry products:\n" + "\n".join(lines)
        self.log_text.append(msg)
        self._workspace_panel.refresh()
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "Photogrammetry complete", msg))

    def _on_photo_error(self, message: str) -> None:
        self._status_label.setText("Photogrammetry failed.")
        self._progress_bar.setValue(0)
        self.log_text.append(f"Photogrammetry failed: {message}")
        QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Photogrammetry failed", message))

    def _open_photo_in_gui(self) -> None:
        """Open the most recent photogrammetry project in its native GUI."""
        import photogrammetry_service as ps
        run_dir = Path(self._last_photo_run_dir)
        engine  = "metashape" if self.photo_engine_meta_radio.isChecked() else "colmap"

        if engine == "metashape":
            psx = run_dir / "project.psx"
            if not psx.exists():
                QMessageBox.warning(self, "No project", f"project.psx not found in:\n{run_dir}")
                return
            exe = (self._photo_engines or {}).get("metashape") or ps._find_metashape_exe()
            ps.launch_in_metashape(str(psx), exe=exe)
        else:
            db = run_dir / "colmap" / "database.db"
            if not db.exists():
                QMessageBox.warning(self, "No database", f"COLMAP database not found in:\n{run_dir}")
                return
            colmap_bin = (self._photo_engines or {}).get("colmap") or "colmap"
            ps.launch_in_colmap_gui(str(db), colmap_bin=colmap_bin)

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
        self.annotate_configure_button.clicked.connect(self._open_annotation_settings)
        self.postprocess_annotate_configure_button.clicked.connect(self._open_annotation_settings)
        self.controls_tabs.currentChanged.connect(self._update_center_panel)
        # Threshold spinboxes removed from UI — no signals to wire.
        self.sampling_mode_combo.currentIndexChanged.connect(self._refresh_sampling_mode_ui)
        self.sample_images_check.stateChanged.connect(self._refresh_sample_images_fields)
        self.postprocess_clahe_check.toggled.connect(self.clahe_clip_limit_spin.setEnabled)
        self.postprocess_clahe_check.toggled.connect(self.clahe_tile_size_spin.setEnabled)
        self.postprocess_clahe_check.toggled.emit(self.postprocess_clahe_check.isChecked())
        self.full_interp_build_btn.clicked.connect(self._build_full_interp)

        # Jobs tab
        self.job_new_btn.clicked.connect(self._job_new)
        self.job_clear_btn.clicked.connect(self._job_clear)
        self.job_save_btn.clicked.connect(self._job_save)
        self.viz_history_mode_button.toggled.connect(self._on_history_mode_toggled)
        self.viz_print_button.clicked.connect(self._viz_export_map)
        self.viz_track_width_slider.valueChanged.connect(self._on_trackline_width_changed)
        self.map_scale_combo.currentTextChanged.connect(self._on_map_scale_changed)

        # Manual interval tab
        self.manual_pick_button.toggled.connect(self._manual_toggle_pick_mode)
        self.manual_map_widget.segment_created.connect(self._on_manual_interval_picked)
        self.manual_map_widget.pick_point_placed.connect(self._on_manual_pick_point_placed)

        # Sensor Tracklines (now in Jobs tab sidebar)
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

        # Outputs tab — sub-tab changes also update the center panel
        self.outputs_subtabs.currentChanged.connect(
            lambda _: self._update_center_panel()
        )

        # Output data source selector
        self.output_source_combo.currentIndexChanged.connect(self._refresh_outputs_status)

        # Navigation outputs
        self.nav_2d_generate_btn.clicked.connect(self._generate_nav_2d)
        self.nav_3d_generate_btn.clicked.connect(self._generate_nav_3d)

        # Sensor outputs
        self.sensor_refresh_channels_btn.clicked.connect(self._refresh_outputs_channels)
        self.sensor_select_all_btn.clicked.connect(self._sensor_select_all_channels)
        self.sensor_select_none_btn.clicked.connect(self._sensor_select_no_channels)
        self.sensor_2d_generate_btn.clicked.connect(self._generate_sensor_2d)
        self.sensor_3d_generate_btn.clicked.connect(self._generate_sensor_3d)
        self.nav_slices_generate_btn.clicked.connect(self._generate_nav_slices)
        self.sensor_slices_generate_btn.clicked.connect(self._generate_sensor_slices)
        self.sensor_geo_slices_generate_btn.clicked.connect(self._generate_sensor_geo_slices)
        self.qgis_generate_btn.clicked.connect(self._generate_qgis_project)

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
                self._auto_save_workspace()

    def _clear_navigation_file(self) -> None:
        """Clear the configured navigation source and refresh all dependent views."""
        self.navigation_file = None
        self._refresh_all_views()
        self._auto_save_workspace()

    def _add_sensor_file(self) -> None:
        """Open SensorImportDialog and append the resulting SensorFileConfig to sensor_files."""
        dialog = SensorImportDialog(self)
        if dialog.exec():
            result = dialog.get_result()
            if result is not None:
                self.sensor_files.append(result)
                self._refresh_all_views()
                self._auto_save_workspace()

    def _build_full_interp(self) -> None:
        """Run the pipeline with only the build_full_interp step."""
        self._run_pipeline(selected_steps=["build_full_interp"])

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
            self._auto_save_workspace()

    def _remove_selected_sensor(self) -> None:
        """Remove the sensor_files entry highlighted in sensor_list and refresh views.

        Does nothing if no row is currently selected (currentRow() == -1).
        """
        current_row = self.sensor_list.currentRow()
        if current_row < 0:
            return
        del self.sensor_files[current_row]
        self._refresh_all_views()
        self._auto_save_workspace()


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
        # build_full_interp only needs nav/sensor sources and a workspace path —
        # skip all video/interval/job-name requirements for that case.
        if selected_steps == ["build_full_interp"]:
            if not self.workspace_path:
                raise ValueError(
                    "Save the workspace first.\n"
                    "The workspace directory is where interp_full.csv will be written."
                )
            return PipelineConfig(
                video_directory=Path(self.workspace_path),
                output_directory=Path(self.workspace_path),
                video_filename_time_format="",
                videos=[],
                selected_intervals=[],
                navigation_file=self.navigation_file,
                sensor_files=self.sensor_files,
                sample_images=False,
                selected_steps=["build_full_interp"],
                full_interp_sample_hz=float(self.full_interp_hz_spin.value()),
                workspace_directory=self.workspace_path,
            )

        video_dir = self.video_dir_edit.text().strip()
        if not video_dir:
            raise ValueError("Select a video directory on the Inputs tab.")
        if not self.videos:
            raise ValueError("Scan videos before running the pipeline.")
        if not self.pending_job.intervals:
            raise ValueError(
                "Add at least one interval to the current job.\n"
                "Use the Manual Intervals or Threshold Intervals tab."
            )
        # Require a job name
        job_name = self.job_name_edit.text().strip() if hasattr(self, "job_name_edit") else ""
        if not job_name:
            if hasattr(self, "job_name_edit"):
                self.job_name_edit.setStyleSheet("border: 2px solid #c0392b;")
            raise ValueError(
                "Give the job a name before executing.\n"
                "Enter a name in the Jobs tab → Current Job → Name field."
            )
        self.pending_job.name = job_name
        if hasattr(self, "job_name_edit"):
            self.job_name_edit.setStyleSheet("")
        if not self.workspace_path:
            raise ValueError(
                "Save the workspace first (toolbar → Save Workspace).\n"
                "The job output directory is derived from the workspace location."
            )
        output_dir = str(Path(self.workspace_path) / self._job_output_dirname())

        # Threshold values come from state (not UI widgets); sensor thresholds disabled.
        self.sensor_thresholds = {}

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
            full_interp_sample_hz=float(self.full_interp_hz_spin.value()),
            workspace_directory=self.workspace_path,
        )
        if selected_steps is not None:
            config.selected_steps = selected_steps
        else:
            # Build explicit steps from the current UI state so downstream code
            # never has to infer intent from the sample_images flag.
            if self.sample_images_check.isChecked():
                steps = ["extract_frames"]
                if self.generate_rasters_check.isChecked():
                    steps.append("generate_sensor_rasters")
                if self.annotate_frames_check.isChecked():
                    steps.append("annotate_frames")
            else:
                # Sample images is off — don't extract frames, but run every
                # post-processing step that is currently checked.
                steps = []
                if self.postprocess_update_master_check.isChecked():
                    steps.append("update_master")
                if self.generate_rasters_check.isChecked():
                    steps.append("generate_sensor_rasters")
                if self.postprocess_annotate_check.isChecked():
                    steps.append("annotate_frames")
                if self.postprocess_clahe_check.isChecked():
                    steps.append("apply_clahe")
                if self.postprocess_geo_txt_check.isChecked():
                    steps.append("generate_geo_txt")
                if not steps:
                    # Nothing checked — fall back to a sensor-only CSV scan.
                    steps = ["interpolate_only"]
            config.selected_steps = steps
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
        # Frame extraction is locked via the UI when a completed job is loaded.
        # _on_new_sampling_job_clicked handles versioned job creation before we get here.

        try:
            config = self._build_pipeline_config(selected_steps)
        except Exception as exc:
            QMessageBox.warning(self, "Cannot run pipeline", str(exc))
            return

        # Frames exist check — only relevant for jobs that haven't been through the
        # "Create New Sampling Job" flow.  For completed loaded jobs, extraction is
        # already locked in the UI so this path is unreachable.
        if config.sample_images and "extract_frames" in config.selected_steps:
            if self._frames_exist_for_config(config):
                reply = QMessageBox.question(
                    self, "Frames already exist",
                    "Extracted frames already exist for this job's output directory.\n\n"
                    "Re-running extraction will overwrite them.  Continue?",
                    QMessageBox.Yes | QMessageBox.No,
                )
                if reply != QMessageBox.Yes:
                    return

        # Show frame count preview and let user confirm before starting
        if config.sample_images and "extract_frames" in config.selected_steps:
            if not self._show_extraction_preview(config):
                return

        # Mark the pending job as running
        self.pending_job.status = "running"

        self._current_run_steps = config.selected_steps
        self._sensor_only_run = not config.sample_images or "interpolate_only" in config.selected_steps
        self._build_full_interp_run = config.selected_steps == ["build_full_interp"]
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

        import re as _re
        name, ok = QInputDialog.getText(
            self, "Create New Workspace", "New workspace folder name:", text="workspace"
        )
        if not ok:
            return False
        name = _re.sub(r"[^\w\-]", "_", name.strip()).strip("_") or "workspace"
        parent = QFileDialog.getExistingDirectory(self, "Select location for new workspace folder")
        if not parent:
            return False
        ws_dir = Path(parent) / name
        try:
            ws_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Workspace creation failed", f"Could not create folder:\n{exc}")
            return False

        self.workspace_path = str(ws_dir)
        self._copy_inputs_to_workspace()
        try:
            ConfigService.save_workspace(path=str(ws_dir / "workspace.json"), **self._workspace_save_kwargs())
            self.workspace_saved = True
            self.log_text.append(f"New workspace created at {ws_dir}")
            self._refresh_workspace_path_label()
        except Exception as exc:
            self.workspace_path = ""
            QMessageBox.critical(self, "Workspace save failed", str(exc))
            return False

        return True

    def _on_segment_completed(self, record: SegmentRecord) -> None:
        """Record a finished segment and auto-save the workspace.

        The record is enriched with the job name and settings snapshot before
        being appended to segment_history (the persistent interval history).
        """
        record.job_name          = self.pending_job.name
        record.settings_snapshot = self._collect_settings_snapshot()
        self.segment_history.append(record)
        self._refresh_job_history_dropdown()
        self._auto_save_workspace()

    def _on_pipeline_finished(self, output_dirs: list[str]) -> None:
        """Handle successful pipeline completion: record job, advance, re-enable controls."""
        self._set_processing_enabled(True)
        self._progress_bar.setValue(100)
        self._status_label.setText("Pipeline complete.")
        self._subprogress_bar.setValue(0)
        self._substatus_label.setText("Idle.")
        self.log_text.append("Pipeline complete.")

        # Mark the completed job and save it to job_history.
        self.pending_job.status            = "completed"
        self.pending_job.name              = self.pending_job.name or f"Job #{self.pending_job.job_id}"
        self.pending_job.settings_snapshot = self._collect_settings_snapshot()
        completed = copy.deepcopy(self.pending_job)
        existing = next((i for i, j in enumerate(self.job_history)
                         if j.job_id == completed.job_id), None)
        if existing is not None:
            self.job_history[existing] = completed
        else:
            self.job_history.append(completed)

        # Also save any threshold configs used in this run
        if self._threshold_intervals:
            constraints_used = [
                {"channel": r["channel_combo"].currentText(),
                 "min_val": float(r["min_edit"].text()) if r["min_edit"].text() else None,
                 "max_val": float(r["max_edit"].text()) if r["max_edit"].text() else None}
                for r in self._constraint_rows if r["enabled_check"].isChecked()
            ]
            if constraints_used:
                from models import ThresholdConfig as TC
                from models import ThresholdConstraint as TConstr
                cfg = TC(
                    constraints=[TConstr(**c) for c in constraints_used],
                    result_count=len(self._threshold_intervals),
                )
                self.threshold_history.append(cfg)
                self._refresh_threshold_history_list()

        # Advance to the next job.
        self.next_job_id += 1
        self.pending_job = Job(job_id=self.next_job_id)
        self._loaded_from_completed_job = False
        if hasattr(self, "job_name_edit"):
            self.job_name_edit.clear()
        self._refresh_extraction_section_state()
        self._refresh_job_builder_ui()
        self._refresh_job_history_dropdown()
        self._auto_save_workspace()
        self._refresh_summary()
        self._refresh_output_source_combo()

        if self._build_full_interp_run:
            # Refresh channel list on the Sensors sub-tab and switch to Outputs tab.
            self._refresh_outputs_channels()
            outputs_idx = next(
                (i for i in range(self.controls_tabs.count())
                 if self.controls_tabs.tabText(i) == "Outputs"),
                -1,
            )
            if outputs_idx >= 0:
                self.controls_tabs.setCurrentIndex(outputs_idx)
                # Select the Sensors sub-tab so the user sees the freshly populated list
                if hasattr(self, "outputs_subtabs"):
                    sensors_idx = next(
                        (i for i in range(self.outputs_subtabs.count())
                         if self.outputs_subtabs.tabText(i) == "Sensors"),
                        -1,
                    )
                    if sensors_idx >= 0:
                        self.outputs_subtabs.setCurrentIndex(sensors_idx)
            QMessageBox.information(
                self, "interp_full.csv built",
                "interp_full.csv has been written to the workspace directory.\n\n"
                "The Sensors tab has been updated with available channels.\n"
                "Remember to regenerate interp_full.csv if you add new sensor channels.",
            )
        elif self._sensor_only_run:
            # Switch to the Outputs / Video sub-tab so the user can see the graphs
            outputs_idx = next(
                (i for i in range(self.controls_tabs.count())
                 if self.controls_tabs.tabText(i) == "Outputs"),
                -1,
            )
            if outputs_idx >= 0:
                self.controls_tabs.setCurrentIndex(outputs_idx)
                if hasattr(self, "outputs_subtabs"):
                    video_idx = next(
                        (i for i in range(self.outputs_subtabs.count())
                         if self.outputs_subtabs.tabText(i) == "Video"),
                        -1,
                    )
                    if video_idx >= 0:
                        self.outputs_subtabs.setCurrentIndex(video_idx)
            msg = (
                "Sensor interpolation complete. Review the graphs on the Outputs → Video tab, "
                "set thresholds on the Threshold Intervals tab, then re-run with "
                "'Sample images' enabled to extract frames."
            )
            QMessageBox.information(self, "Sensor-only run complete", msg)
        elif output_dirs:
            QMessageBox.information(self, "Pipeline complete", "Outputs written to:\n" + "\n".join(output_dirs))
        else:
            QMessageBox.warning(self, "Pipeline complete", "The pipeline finished but no output folders were generated.")

    def _on_pipeline_error(self, message: str) -> None:
        """Handle pipeline failure: re-enable controls, reset bars, and display the error."""
        self._set_processing_enabled(True)
        self._progress_bar.setValue(0)
        self._subprogress_bar.setValue(0)
        self._status_label.setText("Pipeline failed.")
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
    # Claude AI assistant
    # -----------------------------------------------------------------------

    def _build_claude_context(self) -> str:
        """Return a plain-text workspace snapshot injected into Claude's system prompt."""
        lines: list[str] = []
        if not self.workspace_path:
            lines.append("No workspace is currently loaded.")
            return "\n".join(lines)

        lines.append(f"Workspace: {self.workspace_path}")

        # interp_full.csv
        from pathlib import Path as _Path
        interp = _Path(self.workspace_path) / "interp_full.csv"
        if interp.exists():
            stat = interp.stat()
            size_mb = stat.st_size / 1024 ** 2
            lines.append(f"interp_full.csv: present ({size_mb:.1f} MB)")
            try:
                import pandas as _pd, numpy as _np
                _NAV = frozenset({
                    "unix_time","lat","lon","easting","northing","depth",
                    "water_depth","alt","heading","pitch","roll","utm_zone","frame_filename",
                })
                _df = _pd.read_csv(str(interp), nrows=1)
                channels = [c for c in _df.select_dtypes(include=[_np.number]).columns if c not in _NAV]
                if channels:
                    lines.append(f"Sensor channels ({len(channels)}): {', '.join(channels)}")
            except Exception:
                pass
        else:
            lines.append("interp_full.csv: not yet built")

        # Selected run targets
        if self._selected_nav_run:
            lines.append(f"Selected nav run target: {self._selected_nav_run}")
        if self._selected_sensor_run:
            lines.append(f"Selected sensor run target: {self._selected_sensor_run}")

        # Output product inventory from workspace panel
        ws_summary = self._workspace_panel.get_context_summary()
        if ws_summary:
            lines.append("")
            lines.append("Output products:")
            lines.append(ws_summary)

        # Last few lines of the processing log
        if hasattr(self, "log_text"):
            log_tail = self.log_text.toPlainText()[-600:].strip()
            if log_tail:
                lines.append("")
                lines.append(f"Recent processing log:\n{log_tail}")

        return "\n".join(lines)

    def _show_api_key_dialog(self) -> None:
        """Show a tutorial + entry dialog for the Anthropic API key."""
        from PySide6.QtWidgets import QFrame
        from PySide6.QtGui import QDesktopServices
        from PySide6.QtCore import QUrl

        dlg = QDialog(self)
        dlg.setWindowTitle("Claude API Key Setup")
        dlg.setMinimumWidth(460)
        dlg.setMaximumWidth(520)
        layout = QVBoxLayout(dlg)
        layout.setSpacing(10)
        layout.setContentsMargins(18, 16, 18, 16)

        # --- Header ---
        header = QLabel("Connect the AI Assistant")
        header.setStyleSheet("font-size: 13px; font-weight: bold;")
        layout.addWidget(header)

        sub = QLabel(
            "The Claude Assistant panel is powered by Anthropic's Claude AI. "
            "To use it, you need a free API key from Anthropic."
        )
        sub.setWordWrap(True)
        sub.setStyleSheet("color: #555;")
        layout.addWidget(sub)

        # --- Divider ---
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet("color: #ddd;")
        layout.addWidget(line)

        # --- Steps ---
        steps_label = QLabel("How to get your API key:")
        steps_label.setStyleSheet("font-weight: bold; margin-top: 4px;")
        layout.addWidget(steps_label)

        steps = QLabel(
            "1.  Go to  <a href='https://console.anthropic.com'>console.anthropic.com</a>"
            "  and sign in or create a free account.<br><br>"
            "2.  Click <b>API Keys</b> in the left sidebar, then <b>Create Key</b>.<br><br>"
            "3.  Give the key a name (e.g. <i>Sampling Tool</i>), copy it, "
            "and paste it into the field below.<br><br>"
            "4.  Click <b>Save</b>.  The key is stored only on this computer and "
            "is never shared with anyone."
        )
        steps.setWordWrap(True)
        steps.setOpenExternalLinks(True)
        steps.setTextFormat(Qt.RichText)
        layout.addWidget(steps)

        # --- Divider ---
        line2 = QFrame()
        line2.setFrameShape(QFrame.HLine)
        line2.setStyleSheet("color: #ddd;")
        layout.addWidget(line2)

        # --- Key entry ---
        key_label = QLabel("API Key:")
        key_label.setStyleSheet("font-weight: bold;")
        layout.addWidget(key_label)

        key_row = QHBoxLayout()
        key_edit = QLineEdit()
        key_edit.setEchoMode(QLineEdit.Password)
        key_edit.setText(load_api_key())
        key_edit.setPlaceholderText("sk-ant-…")
        key_row.addWidget(key_edit)

        show_btn = QPushButton("Show")
        show_btn.setFixedWidth(50)
        show_btn.setCheckable(True)
        show_btn.toggled.connect(
            lambda checked: key_edit.setEchoMode(
                QLineEdit.Normal if checked else QLineEdit.Password
            )
        )
        key_row.addWidget(show_btn)
        layout.addLayout(key_row)

        storage_note = QLabel("Stored in  ~/.config/epr_imaging/config.json")
        storage_note.setStyleSheet("font-size: 10px; color: #888;")
        layout.addWidget(storage_note)

        # --- Buttons ---
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Cancel)
        buttons.accepted.connect(dlg.accept)
        buttons.rejected.connect(dlg.reject)
        layout.addWidget(buttons)

        if dlg.exec() == QDialog.Accepted:
            save_api_key(key_edit.text().strip())
            self._chat_panel.refresh_status()

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
                output_directory=str(Path(self.workspace_path) / self._job_output_dirname()) if self.workspace_path else None,
                frame_rate=float(self.frame_rate_spin.value()),
                generate_sensor_rasters=self.generate_rasters_check.isChecked(),
                annotate_frames=self.annotate_frames_check.isChecked(),
                depth_source=self.depth_source,
                speed_source=self.speed_source,
                altitude_threshold=float(self.altitude_threshold or 0.0),
                depth_threshold=float(self.depth_threshold or 0.0),
                speed_threshold=float(self.speed_threshold or 0.0),
                min_segment_frames=int(self.min_segment_frames),
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

    def _show_startup_dialog(self) -> None:
        """Show the workspace selection dialog on startup.

        Loops until the user has a valid workspace_path — either by continuing
        the last session, opening an existing workspace, or creating a new one.
        Cancelling an open/create dialog re-shows the startup screen rather than
        leaving the app in an unusable state.
        """
        # Inspect last_session.json to decide whether to offer "Continue" option.
        last_session_label = ""
        last_session_path  = self._last_session_path()
        if last_session_path.exists():
            try:
                data = ConfigService.load_workspace(str(last_session_path))
                saved = data.get("workspace_path", "")
                if saved:
                    p = Path(saved)
                    ws_dir = p if p.is_dir() else (p.parent if p.is_file() else None)
                    if ws_dir and ws_dir.is_dir():
                        last_session_label = str(ws_dir)
            except Exception:
                pass

        while not self.workspace_path:
            dlg = WorkspaceStartupDialog(
                has_last_session=bool(last_session_label),
                last_session_path=last_session_label,
                parent=self,
            )
            dlg.exec()

            if dlg.choice == WorkspaceStartupDialog.CONTINUE:
                self._restore_last_session()
                # If the stored workspace path is gone, workspace_path stays
                # empty and the loop repeats so the user can choose another option.
            elif dlg.choice == WorkspaceStartupDialog.OPEN:
                self._load_workspace()
            elif dlg.choice == WorkspaceStartupDialog.NEW:
                self._save_workspace()

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
            saved_ws_path = data.get("workspace_path", "")
            if saved_ws_path:
                p = Path(saved_ws_path)
                # New format: workspace_path is a directory.
                # Old format: workspace_path was the .json file — use its parent.
                ws_dir = p if p.is_dir() else (p.parent if p.is_file() else None)
                if ws_dir and ws_dir.is_dir():
                    self.workspace_path = str(ws_dir)
                    self.workspace_saved = True
                    self._refresh_workspace_path_label()
                    self._refresh_output_source_combo()
            self.log_text.append(f"Last session restored from {path}")
        except Exception as exc:
            import traceback
            self.log_text.append(f"Could not restore last session: {exc}\n{traceback.format_exc()}")

    def _save_workspace(self) -> None:
        """Save session state to the workspace directory.

        If a workspace directory is already set, saves workspace.json in place.
        Otherwise prompts for a parent location and folder name, creates the
        directory, and saves workspace.json inside it.
        """
        if self.workspace_path:
            # Existing workspace — copy inputs then save in place.
            self._copy_inputs_to_workspace()
            try:
                ConfigService.save_workspace(
                    path=str(self._workspace_json_path()), **self._workspace_save_kwargs()
                )
                self.log_text.append("Workspace saved.")
                self.workspace_saved = True
                self._refresh_workspace_path_label()
            except Exception as exc:
                QMessageBox.critical(self, "Workspace save failed", str(exc))
            return

        # No workspace yet — ask for a name and parent location.
        import re as _re
        name, ok = QInputDialog.getText(
            self, "Create Workspace", "Workspace folder name:", text="workspace"
        )
        if not ok:
            return
        name = _re.sub(r"[^\w\-]", "_", name.strip()).strip("_") or "workspace"
        parent = QFileDialog.getExistingDirectory(self, "Select location for workspace folder")
        if not parent:
            return
        ws_dir = Path(parent) / name
        try:
            ws_dir.mkdir(parents=True, exist_ok=True)
        except Exception as exc:
            QMessageBox.critical(self, "Workspace creation failed", f"Could not create folder:\n{exc}")
            return
        self.workspace_path = str(ws_dir)
        self._copy_inputs_to_workspace()
        try:
            ConfigService.save_workspace(
                path=str(ws_dir / "workspace.json"), **self._workspace_save_kwargs()
            )
            self.workspace_saved = True
            self._refresh_workspace_path_label()
            self.log_text.append(f"Workspace created at {ws_dir}")
        except Exception as exc:
            self.workspace_path = ""
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
        self.altitude_threshold  = None
        self.depth_threshold     = None
        self.speed_threshold     = None
        self.min_segment_frames  = 1
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
        self.next_job_id       = int(data.get("next_job_id", 1))
        self.pending_job       = data.get("pending_job") or Job(job_id=self.next_job_id)
        self.segment_history   = data.get("segment_history") or []
        self.job_history       = data.get("job_history") or []
        self.threshold_history = data.get("threshold_history") or []
        self.frame_rate_spin.setValue(data["frame_rate"])
        self.generate_rasters_check.setChecked(data["generate_sensor_tiffs"])
        self.annotate_frames_check.setChecked(data["annotate_frames"])
        self.altitude_threshold = data.get("altitude_threshold")
        self.depth_threshold = data.get("depth_threshold")
        self.speed_threshold = data.get("speed_threshold")
        self.min_segment_frames = int(data.get("min_segment_frames", 1))
        ann_data = data.get("annotation_config")
        if isinstance(ann_data, AnnotationConfig):
            self.annotation_config = ann_data          # already converted by load_workspace
        elif isinstance(ann_data, dict):
            self.annotation_config = AnnotationConfig.from_dict(ann_data)
        else:
            self.annotation_config = AnnotationConfig()
        # Threshold spinboxes removed from Processing tab — state kept in memory only.
        idx = self.frame_quality_combo.findText(data.get("frame_quality", "Original"))
        self.frame_quality_combo.setCurrentIndex(idx if idx >= 0 else 0)
        mode_text = "Dynamic spacing" if data.get("sampling_mode", "fixed") == "dynamic" else "Fixed rate"
        self.sampling_mode_combo.setCurrentIndex(self.sampling_mode_combo.findText(mode_text))
        self.dynamic_spacing_spin.setValue(float(data.get("dynamic_target_spacing_m", 2.0)))
        self.dynamic_min_freq_spin.setValue(float(data.get("dynamic_min_frequency_hz", 0.1)))
        self.clahe_clip_limit_spin.setValue(float(data.get("clahe_clip_limit", 2.0)))
        self.clahe_tile_size_spin.setValue(int(data.get("clahe_tile_grid_size", 8)))

        # Outputs tab
        def _set_combo(combo, key, default=""):
            val = data.get(key, default)
            idx = combo.findText(val)
            if idx >= 0:
                combo.setCurrentIndex(idx)

        if hasattr(self, "nav_2d_cell_spin"):
            self.nav_2d_cell_spin.setValue(float(data.get("out_nav_2d_cell_size", 5.0)))
            crs = data.get("out_nav_2d_crs", "utm")
            self.nav_crs_wgs84_radio.setChecked(crs == "wgs84")
            self.nav_crs_utm_radio.setChecked(crs != "wgs84")
            self.nav_3d_cell_spin.setValue(float(data.get("out_nav_3d_cell_size", 1.0)))
            self.nav_slices_step_spin.setValue(float(data.get("out_nav_slices_step", 5.0)))
            self.nav_slices_ppc_spin.setValue(int(data.get("out_nav_slices_ppc", 4)))
            self.sensor_2d_cell_spin.setValue(float(data.get("out_sensor_2d_cell_size", 5.0)))
            crs2 = data.get("out_sensor_2d_crs", "utm")
            self.sensor_crs_wgs84_radio.setChecked(crs2 == "wgs84")
            self.sensor_crs_utm_radio.setChecked(crs2 != "wgs84")
            _set_combo(self.sensor_2d_fill_combo, "out_sensor_2d_fill", "IDW fill")
            self.sensor_3d_cell_spin.setValue(float(data.get("out_sensor_3d_cell_size", 1.0)))
            _set_combo(self.sensor_3d_agg_combo,  "out_sensor_3d_agg",  "mean")
            _set_combo(self.sensor_3d_fill_combo, "out_sensor_3d_fill", "IDW fill")
            self.sensor_3d_zero_spin.setValue(float(data.get("out_sensor_3d_zero_mask", 5.0)))
            self.sensor_slices_step_spin.setValue(float(data.get("out_sensor_slices_step", 5.0)))
            self.sensor_slices_ppc_spin.setValue(int(data.get("out_sensor_slices_ppc", 4)))
            _set_combo(self.sensor_slices_color_combo, "out_sensor_slices_color", "rgb (viridis)")
            self.sensor_slices_log_check.setChecked(bool(data.get("out_sensor_slices_log", False)))
            self.sensor_slices_local_norm_check.setChecked(bool(data.get("out_sensor_slices_local_norm", False)))
            self.sensor_slices_pct_spin.setValue(float(data.get("out_sensor_slices_pct", 100.0)))
            self.sensor_geo_slices_step_spin.setValue(float(data.get("out_sensor_geo_slices_step", 5.0)))
            self.sensor_geo_slices_cell_spin.setValue(float(data.get("out_sensor_geo_slices_cell", 2.0)))
            _set_combo(self.sensor_geo_slices_fill_combo, "out_sensor_geo_slices_fill", "IDW fill")
            self.qgis_project_name_edit.setText(data.get("qgis_project_name", "EPR Survey"))

            # Task stack
            ts_data = data.get("task_stack")
            if ts_data:
                self._task_stack = TaskStack.from_dict(ts_data)
                if self._stack_panel is not None:
                    self._stack_panel._stack = self._task_stack
                    self._stack_panel.refresh()

            # Photogrammetry settings
            photo_engine = data.get("photo_engine", "metashape")
            self.photo_engine_meta_radio.setChecked(photo_engine == "metashape")
            self.photo_engine_colmap_radio.setChecked(photo_engine == "colmap")
            _set_combo(self.photo_quality_combo, "photo_quality", "Normal")
            self.photo_dense_check.setChecked(bool(data.get("photo_build_dense", True)))
            self.photo_mesh_check.setChecked(bool(data.get("photo_build_mesh", True)))
            self.photo_texture_check.setChecked(bool(data.get("photo_build_texture", False)))
            self.photo_nav_seed_check.setChecked(bool(data.get("photo_nav_seed", True)))
            last_run = data.get("photo_last_run_dir", "")
            if last_run:
                self._last_photo_run_dir = last_run
                self.photo_open_gui_btn.setEnabled(True)

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
        """Open a workspace folder and restore all form fields.

        The user selects the workspace directory.  The app looks for
        workspace.json inside it.  If workspace.json is absent (e.g. an older
        workspace that used a custom filename), a file picker opens so the user
        can locate the JSON manually.
        """
        ws_dir = QFileDialog.getExistingDirectory(self, "Open Workspace Folder")
        if not ws_dir:
            return
        json_path = Path(ws_dir) / "workspace.json"
        if not json_path.exists():
            # Backward compat: let user pick the JSON file directly.
            json_str, _ = QFileDialog.getOpenFileName(
                self, "Select workspace JSON inside folder",
                ws_dir, "JSON Files (*.json);;All Files (*)"
            )
            if not json_str:
                return
            json_path = Path(json_str)
        try:
            data = ConfigService.load_workspace(str(json_path))
            self._apply_workspace_data(data)
            self.log_text.append(f"Workspace loaded from {json_path.parent}")
            self.workspace_saved = True
            self.workspace_path = str(json_path.parent)
            self._refresh_workspace_path_label()
            self._refresh_output_source_combo()
        except Exception as exc:
            import traceback
            full = traceback.format_exc()
            self.log_text.append(f"Workspace load failed:\n{full}")
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
            job_history=self.job_history,
            threshold_history=self.threshold_history,
            frame_rate=float(self.frame_rate_spin.value()),
            generate_sensor_tiffs=self.generate_rasters_check.isChecked(),
            annotate_frames=self.annotate_frames_check.isChecked(),
            frame_quality=self.frame_quality_combo.currentText(),
            depth_source=self.depth_source,
            speed_source=self.speed_source,
            altitude_threshold=float(self.altitude_threshold or 0.0),
            depth_threshold=float(self.depth_threshold or 0.0),
            speed_threshold=float(self.speed_threshold or 0.0),
            min_segment_frames=int(self.min_segment_frames),
            sampling_mode="dynamic" if self.sampling_mode_combo.currentText() == "Dynamic spacing" else "fixed",
            dynamic_target_spacing_m=float(self.dynamic_spacing_spin.value()),
            dynamic_min_frequency_hz=float(self.dynamic_min_freq_spin.value()),
            clahe_clip_limit=float(self.clahe_clip_limit_spin.value()),
            clahe_tile_grid_size=int(self.clahe_tile_size_spin.value()),
            annotation_config=self.annotation_config,
            workspace_path=self.workspace_path,
            # Outputs tab
            out_nav_2d_cell_size=float(self.nav_2d_cell_spin.value()),
            out_nav_2d_crs="wgs84" if self.nav_crs_wgs84_radio.isChecked() else "utm",
            out_nav_3d_cell_size=float(self.nav_3d_cell_spin.value()),
            out_nav_slices_step=float(self.nav_slices_step_spin.value()),
            out_nav_slices_ppc=int(self.nav_slices_ppc_spin.value()),
            out_sensor_2d_cell_size=float(self.sensor_2d_cell_spin.value()),
            out_sensor_2d_crs="wgs84" if self.sensor_crs_wgs84_radio.isChecked() else "utm",
            out_sensor_2d_fill=self.sensor_2d_fill_combo.currentText(),
            out_sensor_3d_cell_size=float(self.sensor_3d_cell_spin.value()),
            out_sensor_3d_agg=self.sensor_3d_agg_combo.currentText(),
            out_sensor_3d_fill=self.sensor_3d_fill_combo.currentText(),
            out_sensor_3d_zero_mask=float(self.sensor_3d_zero_spin.value()),
            out_sensor_slices_step=float(self.sensor_slices_step_spin.value()),
            out_sensor_slices_ppc=int(self.sensor_slices_ppc_spin.value()),
            out_sensor_slices_color=self.sensor_slices_color_combo.currentText(),
            out_sensor_slices_log=self.sensor_slices_log_check.isChecked(),
            out_sensor_slices_local_norm=self.sensor_slices_local_norm_check.isChecked(),
            out_sensor_slices_pct=float(self.sensor_slices_pct_spin.value()),
            photo_engine="metashape" if self.photo_engine_meta_radio.isChecked() else "colmap",
            photo_quality=self.photo_quality_combo.currentText(),
            photo_build_dense=self.photo_dense_check.isChecked(),
            photo_build_mesh=self.photo_mesh_check.isChecked(),
            photo_build_texture=self.photo_texture_check.isChecked(),
            photo_nav_seed=self.photo_nav_seed_check.isChecked(),
            photo_last_run_dir=self._last_photo_run_dir,
            out_sensor_geo_slices_step=float(self.sensor_geo_slices_step_spin.value()),
            out_sensor_geo_slices_cell=float(self.sensor_geo_slices_cell_spin.value()),
            out_sensor_geo_slices_fill=self.sensor_geo_slices_fill_combo.currentText(),
            qgis_project_name=self.qgis_project_name_edit.text().strip() or "EPR Survey",
            task_stack=self._task_stack.to_dict(),
        )

    def _copy_inputs_to_workspace(self) -> None:
        """Copy all input CSV files into inputs/nav/ and inputs/sensor/ inside the workspace.

        Updates csv_path on every NavigationConfig and SensorFileConfig object
        in-place so subsequent saves store workspace-relative paths.  Files that
        are already inside the workspace directory are left untouched.  Multiple
        sources that share the same CSV file are deduplicated — the file is only
        copied once.
        """
        if not self.workspace_path:
            return
        import shutil
        ws = Path(self.workspace_path)
        _copied: dict[Path, Path] = {}

        def _copy_file(src: Path, subdir: str) -> Path:
            src = src.resolve()
            if src in _copied:
                return _copied[src]
            try:
                src.relative_to(ws)
                _copied[src] = src
                return src
            except ValueError:
                pass
            dest_dir = ws / "inputs" / subdir
            dest_dir.mkdir(parents=True, exist_ok=True)
            dest = dest_dir / src.name
            stem, suffix = src.stem, src.suffix
            n = 1
            while dest.exists():
                dest = dest_dir / f"{stem}_{n}{suffix}"
                n += 1
            shutil.copy2(str(src), str(dest))
            _copied[src] = dest
            return dest

        if self.navigation_file:
            for tvs in self.navigation_file._all_sources:
                p = Path(tvs.csv_path)
                if p.exists():
                    tvs.csv_path = _copy_file(p, "nav")

        for sf in self.sensor_files:
            p = Path(sf.csv_path)
            if p.exists():
                sf.csv_path = _copy_file(p, "sensor")

        for src_obj in filter(None, [self.depth_source, self.speed_source]):
            p = Path(src_obj.csv_path)
            if p.exists():
                src_obj.csv_path = _copy_file(p, "sensor")

    # Tracks whether we've already shown the auto-save failure warning this session
    # so we don't spam the user with repeated dialogs on every failed save.
    _auto_save_warned: bool = False

    def _workspace_json_path(self) -> "Path":
        """Return the workspace.json path inside the current workspace directory."""
        return Path(self.workspace_path) / "workspace.json"

    def _auto_save_workspace(self) -> None:
        """Save the workspace to workspace_path (if set) or the last-session path.

        Called after each segment completes and after job completion.  On failure,
        logs the error and shows a one-time warning dialog so the user knows their
        progress is not being persisted.
        """
        save_path = str(self._workspace_json_path()) if self.workspace_path else str(self._last_session_path())
        try:
            ConfigService.save_workspace(path=save_path, **self._workspace_save_kwargs())
            if self.workspace_path:
                self.workspace_saved = True
            self._auto_save_warned = False  # reset on next success
        except Exception as exc:
            self.log_text.append(f"Auto-save workspace failed: {exc}")
            if not self._auto_save_warned:
                self._auto_save_warned = True
                QMessageBox.warning(
                    self, "Workspace auto-save failed",
                    f"The workspace could not be saved automatically:\n{exc}\n\n"
                    "Your session progress is NOT being saved.  Use "
                    "File → Save Workspace to save manually, or check that the "
                    "workspace file location is accessible."
                )

    def _append_log(self, message: str) -> None:
        """Append one line to the Processing Log text area (connected to PipelineWorker.log)."""
        self.log_text.append(message)

    def _refresh_all_views(self) -> None:
        """Rebuild every dependent UI widget from current session state.

        Called after any state change that could affect multiple widgets.
        The order matches the visual top-to-bottom layout so the most visible
        elements (sensor list, nav summary) update first.
        """
        self._workspace_panel.set_sources(
            video_dir=self.video_directory,
            videos=self.videos,
            nav_file=self.navigation_file,
            sensor_files=self.sensor_files,
            depth_source=self.depth_source,
            speed_source=self.speed_source,
        )
        self._refresh_sensor_list()
        self._refresh_navigation_summary()
        self._refresh_workspace_path_label()
        self._refresh_job_builder_ui()
        self._refresh_job_history_dropdown()
        self._refresh_threshold_history_list()
        self._refresh_manual_staged_list()
        self._refresh_timeline()
        self._refresh_sampling_mode_ui()
        self._refresh_extraction_section_state()
        self._update_center_panel()
        self._refresh_summary()
        self._refresh_warnings()
        self._refresh_history_overlay()
        self._refresh_output_source_combo()
        self._refresh_outputs_status()

    # -----------------------------------------------------------------------
    # Outputs tab helpers
    # -----------------------------------------------------------------------

    def _interp_full_path(self) -> str:
        """Return the expected path of interp_full.csv in the current workspace."""
        if not self.workspace_path:
            return ""
        return str(Path(self.workspace_path) / "interp_full.csv")

    def _outputs_root(self) -> str:
        """Return the outputs directory inside the current workspace."""
        return str(Path(self.workspace_path) / "outputs") if self.workspace_path else ""

    def _refresh_output_source_combo(self) -> None:
        """Repopulate the output source combo.

        Entries:
          1. Full dataset (interp_full.csv)
          2. Per-segment entries for jobs that have interp.csv on disk
          3. Job-level entries for configured jobs with no disk data yet —
             these filter interp_full.csv to the job's intervals on demand.
        """
        if not hasattr(self, "output_source_combo"):
            return
        combo = self.output_source_combo
        prev_data = combo.currentData()
        combo.blockSignals(True)
        combo.clear()
        combo.addItem("Full dataset (interp_full.csv)", userData=None)

        job_ids_with_disk_data: set[int] = set()

        if self.workspace_path:
            ws = Path(self.workspace_path)

            # Per-segment entries: jobs that already have interp.csv on disk
            for interp_csv in sorted(ws.glob("job_*/segment_*/interp.csv")):
                job_dir_name = interp_csv.parent.parent.name   # e.g. job_001_MyJob
                seg_dir      = interp_csv.parent.name          # e.g. segment_001_20260118T…
                seg_parts    = seg_dir.split("_", 2)
                seg_num      = seg_parts[1] if len(seg_parts) > 1 else seg_dir
                seg_times    = seg_parts[2].replace("_", " → ") if len(seg_parts) > 2 else ""
                label = (f"{job_dir_name}  /  seg {seg_num}  ({seg_times})"
                         if seg_times else f"{job_dir_name}  /  {seg_dir}")
                combo.addItem(label, userData=str(interp_csv))
                try:
                    job_ids_with_disk_data.add(int(job_dir_name.split("_")[1]))
                except (IndexError, ValueError):
                    pass

        # Job-level entries: configured jobs that have no segment data on disk yet
        all_jobs = []
        if self.pending_job.intervals:
            all_jobs.append(self.pending_job)
        all_jobs.extend(j for j in self.job_history if j.intervals)
        for job in all_jobs:
            if job.job_id in job_ids_with_disk_data:
                continue  # already covered by per-segment entries above
            n     = len(job.intervals)
            name  = job.name or f"Job #{job.job_id}"
            label = f"job_{job.job_id:03d}: {name}  ({n} interval{'s' if n != 1 else ''})"
            combo.addItem(label, userData={"type": "job_filter", "job_id": job.job_id})

        # Restore previous selection when possible
        for i in range(combo.count()):
            if combo.itemData(i) == prev_data:
                combo.setCurrentIndex(i)
                break
        combo.blockSignals(False)
        self._refresh_outputs_status()

    def _output_source_paths(self) -> tuple[str, str]:
        """Return (interp_path, output_dir) for the currently selected data source.

        Full dataset  → (interp_full.csv, <workspace>/outputs/).
        Segment entry → (segment/interp.csv, <segment>/outputs/).
        Job entry (no disk data yet) → filtered interp_full.csv for that job's
            intervals, written to <workspace>/job_NNN/filtered_interp.csv.
        """
        if not hasattr(self, "output_source_combo"):
            return self._interp_full_path(), self._outputs_root()
        data = self.output_source_combo.currentData()
        if data is None:
            return self._interp_full_path(), self._outputs_root()
        if isinstance(data, str):
            return data, str(Path(data).parent / "outputs")
        if isinstance(data, dict) and data.get("type") == "job_filter":
            job_id = data["job_id"]
            job = next(
                (j for j in [self.pending_job] + list(self.job_history) if j.job_id == job_id),
                None,
            )
            if job is None or not job.intervals:
                return self._interp_full_path(), self._outputs_root()
            output_dir  = str(Path(self.workspace_path) / self._job_output_dirname(job) / "outputs")
            interp_path = self._get_filtered_interp_for_job(job)
            return interp_path, output_dir
        return self._interp_full_path(), self._outputs_root()

    def _get_filtered_interp_for_job(self, job) -> str:
        """Return path to interp_full.csv filtered to job's time intervals.

        Writes <workspace>/job_NNN/filtered_interp.csv each call (fast — just
        DataFrame filtering).  Falls back to interp_full.csv on any error.
        """
        import calendar
        import pandas as pd

        interp_full = self._interp_full_path()
        if not interp_full or not Path(interp_full).exists():
            return interp_full
        if not job.intervals:
            return interp_full

        job_dir = Path(self.workspace_path) / self._job_output_dirname(job)
        job_dir.mkdir(parents=True, exist_ok=True)
        out_path = job_dir / "filtered_interp.csv"

        # Skip re-filtering if filtered_interp.csv is already up to date
        if (out_path.exists()
                and out_path.stat().st_mtime >= Path(interp_full).stat().st_mtime):
            return str(out_path)

        try:
            df = pd.read_csv(interp_full)
            if "unix_time" in df.columns:
                mask = pd.Series(False, index=df.index)
                for interval in job.intervals:
                    t0 = float(calendar.timegm(interval.start_time.timetuple()))
                    t1 = float(calendar.timegm(interval.end_time.timetuple()))
                    mask |= (df["unix_time"] >= t0) & (df["unix_time"] < t1)
                df = df[mask]
            df.to_csv(str(out_path), index=False)
            name = job.name or f"Job #{job.job_id}"
            self.log_text.append(f"Filtered interp: {len(df)} rows for '{name}'")
        except Exception as exc:
            self.log_text.append(f"Warning: could not filter interp for job: {exc}")
            return interp_full

        return str(out_path)

    def _refresh_outputs_status(self) -> None:
        """Show/hide the prerequisite warning labels on the Nav and Sensors sub-tabs."""
        if not hasattr(self, "nav_outputs_warning"):
            return
        csv_path, _ = self._output_source_paths()
        missing = not csv_path or not Path(csv_path).exists()
        source_label = Path(csv_path).name if csv_path else "interp CSV"
        msg = (
            f"{source_label} not found.  "
            + ("Run 'Build interp_full.csv' on the Inputs tab first."
               if not csv_path or Path(csv_path).name == "interp_full.csv"
               else "Run the pipeline for this interval first.")
        )
        self.nav_outputs_warning.setText(msg)
        self.nav_outputs_warning.setVisible(missing)
        self.nav_2d_generate_btn.setEnabled(not missing)
        self.nav_3d_generate_btn.setEnabled(not missing)
        has_nav_target = bool(self._selected_nav_run)
        self.nav_slices_generate_btn.setEnabled(has_nav_target)
        self.nav_slices_generate_btn.setToolTip(
            "" if has_nav_target else
            "Select a nav_trackline run in the Workspace panel\n"
            "(or right-click a run and choose 'Make Slices')."
        )

        self.sensor_outputs_warning.setText(msg)
        self.sensor_outputs_warning.setVisible(missing)
        self.sensor_2d_generate_btn.setEnabled(not missing)
        self.sensor_3d_generate_btn.setEnabled(not missing)
        has_sensor_target = bool(self._selected_sensor_run)
        self.sensor_slices_generate_btn.setEnabled(has_sensor_target)
        self.sensor_slices_generate_btn.setToolTip(
            "" if has_sensor_target else
            "Select a sensor_3d run in the Workspace panel\n"
            "(or right-click a run and choose 'Make Slices')."
        )

        if not missing:
            self._refresh_outputs_channels()

    def _refresh_outputs_channels(self) -> None:
        """Reload sensor channel list from the currently selected interp CSV."""
        if not hasattr(self, "sensor_channel_list"):
            return
        csv_path, _ = self._output_source_paths()
        if not csv_path or not Path(csv_path).exists():
            self.sensor_channel_list.clear()
            return
        try:
            from output_service import sensor_channels_from_csv
            channels = sensor_channels_from_csv(csv_path)
            # Preserve existing check state by name
            was_checked = set()
            for i in range(self.sensor_channel_list.count()):
                item = self.sensor_channel_list.item(i)
                if item.checkState() == Qt.Checked:
                    was_checked.add(item.text())
            self.sensor_channel_list.clear()
            for ch in channels:
                item = QListWidgetItem(ch)
                item.setCheckState(Qt.Checked if (not was_checked or ch in was_checked) else Qt.Unchecked)
                self.sensor_channel_list.addItem(item)
        except Exception as exc:
            self.log_text.append(f"Channel refresh failed: {exc}")

    def _sensor_select_all_channels(self) -> None:
        for i in range(self.sensor_channel_list.count()):
            self.sensor_channel_list.item(i).setCheckState(Qt.Checked)

    def _sensor_select_no_channels(self) -> None:
        for i in range(self.sensor_channel_list.count()):
            self.sensor_channel_list.item(i).setCheckState(Qt.Unchecked)

    def _selected_sensor_channels(self) -> list[str]:
        """Return the list of sensor channels with checkboxes checked."""
        result = []
        for i in range(self.sensor_channel_list.count()):
            item = self.sensor_channel_list.item(i)
            if item.checkState() == Qt.Checked:
                result.append(item.text())
        return result

    # -----------------------------------------------------------------------
    # Output generation handlers
    # -----------------------------------------------------------------------

    def _run_output_task(self, task: str, **kwargs) -> None:
        """Start an OutputWorker for the given task in a background QThread."""
        if not self.workspace_path:
            QMessageBox.warning(self, "No workspace", "Save the workspace first.")
            return
        if self._output_worker_is_running():
            QMessageBox.warning(self, "Busy", "An output task is already running.")
            return

        kwargs.setdefault("interp_path", self._interp_full_path())
        kwargs.setdefault("output_dir",  self._outputs_root())

        self.log_text.clear()
        self._status_label.setText("Generating output…")
        self._progress_bar.setValue(0)

        thread = QThread(self)
        worker = OutputWorker(task, kwargs)
        self.output_worker_thread = thread
        self.output_worker        = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.error.connect(self._on_output_error)
        worker.finished.connect(self._on_output_finished)
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.finished.connect(self._clear_output_worker)
        thread.start()

    def _output_worker_is_running(self) -> bool:
        """Return True only if the output worker thread is actively running.

        Guards against RuntimeError when the C++ QThread was already deleted
        via deleteLater() but the Python wrapper still exists.
        """
        try:
            return bool(self.output_worker_thread and self.output_worker_thread.isRunning())
        except RuntimeError:
            self.output_worker_thread = None
            self.output_worker = None
            return False

    def _clear_output_worker(self) -> None:
        """Clear worker references after a task thread finishes."""
        self.output_worker_thread = None
        self.output_worker = None

    def _on_output_finished(self, paths: list[str]) -> None:
        self._status_label.setText("Output complete.")
        self._progress_bar.setValue(100)
        msg = "Output(s) written to:\n" + "\n".join(paths)
        self.log_text.append(msg)
        self._workspace_panel.refresh()
        # Defer dialog: showing QMessageBox directly here spins a nested event
        # loop that can process thread deleteLater() while Qt is still
        # dispatching this slot, causing a crash on cleanup.
        QTimer.singleShot(0, lambda: QMessageBox.information(self, "Output complete", msg))

    def _on_output_error(self, message: str) -> None:
        self._status_label.setText("Output failed.")
        self._progress_bar.setValue(0)
        self.log_text.append(f"Output failed: {message}")
        QTimer.singleShot(0, lambda: QMessageBox.critical(self, "Output failed", message))

    def _run_sensor_queue(
        self,
        task_name: str,
        queue: list,
        params: dict,
        results: list,
        done_title: str,
        done_status: str,
        done_msg_fn,
        per_channel_log_fn,
    ) -> None:
        if not queue:
            msg = done_msg_fn(results)
            self.log_text.append(msg)
            self._status_label.setText(done_status)
            self._progress_bar.setValue(100)
            self._workspace_panel.refresh()
            # Defer dialog to avoid nested event loop during thread cleanup.
            QTimer.singleShot(0, lambda: QMessageBox.information(
                self, done_title, msg + "\n" + "\n".join(results)
            ))
            return
        channel = queue.pop(0)
        self.log_text.append(per_channel_log_fn(channel))
        thread = QThread(self)
        worker = OutputWorker(task_name, dict(channel=channel, **params))
        self.output_worker_thread = thread
        self.output_worker        = worker

        def _cleanup_this_thread(_thread=thread, _worker=worker):
            # Only wipe self refs if they still point at THIS thread/worker;
            # the queue may have already created the next one.
            if self.output_worker_thread is _thread:
                self.output_worker_thread = None
            if self.output_worker is _worker:
                self.output_worker = None

        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.log.connect(self._append_log)
        worker.error.connect(self._on_output_error)
        worker.finished.connect(
            lambda paths: (
                results.extend(paths),
                self._run_sensor_queue(
                    task_name, queue, params, results,
                    done_title, done_status, done_msg_fn, per_channel_log_fn,
                ),
            )
        )
        worker.finished.connect(thread.quit)
        worker.error.connect(thread.quit)
        thread.finished.connect(_cleanup_this_thread)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(thread.deleteLater)
        thread.start()

    def _generate_nav_2d(self) -> None:
        interp_path, output_dir = self._output_source_paths()
        crs = "wgs84" if self.nav_crs_wgs84_radio.isChecked() else "utm"
        self._run_output_task(
            "generate_nav_2d_geotiff",
            interp_path=interp_path,
            output_dir=output_dir,
            cell_size_m=float(self.nav_2d_cell_spin.value()),
            crs_mode=crs,
        )

    def _generate_nav_3d(self) -> None:
        interp_path, output_dir = self._output_source_paths()
        self._run_output_task(
            "generate_nav_3d_ply",
            interp_path=interp_path,
            output_dir=output_dir,
            cell_size=float(self.nav_3d_cell_spin.value()),
        )

    def _generate_sensor_2d(self) -> None:
        channels = self._selected_sensor_channels()
        if not channels:
            QMessageBox.warning(self, "No channels selected",
                                "Select at least one sensor channel.")
            return
        fill_text = self.sensor_2d_fill_combo.currentText()
        fill_map  = {
            "Trackline only (no fill)": "none",
            "IDW fill":                 "idw",
            "Kriging fill":             "kriging",
            "RBF fill":                 "rbf",
        }
        fill = fill_map.get(fill_text, "none")
        crs  = "wgs84" if self.sensor_crs_wgs84_radio.isChecked() else "utm"

        if self._output_worker_is_running():
            QMessageBox.warning(self, "Busy", "An output task is already running.")
            return
        if not self.workspace_path:
            QMessageBox.warning(self, "No workspace", "Save the workspace first.")
            return

        interp_path, output_dir = self._output_source_paths()
        self.log_text.clear()
        self._status_label.setText(f"Generating 2D GeoTIFFs for {len(channels)} channel(s)…")
        self._progress_bar.setValue(0)

        self._run_sensor_queue(
            "generate_sensor_2d_geotiff",
            list(channels),
            dict(
                cell_size_m=float(self.sensor_2d_cell_spin.value()),
                crs_mode=crs,
                fill_method=fill,
                output_dir=output_dir,
                interp_path=interp_path,
            ),
            [],
            "2D Output complete",
            "2D output complete.",
            lambda r: f"{len(r)} GeoTIFF(s) written.",
            lambda ch: f"Generating 2D GeoTIFF: {ch} …",
        )

    def _generate_sensor_3d(self) -> None:
        channels = self._selected_sensor_channels()
        if not channels:
            QMessageBox.warning(self, "No channels selected",
                                "Select at least one sensor channel.")
            return
        fill_text = self.sensor_3d_fill_combo.currentText()
        fill_map  = {"IDW fill": "idw", "Kriging fill": "kriging", "RBF fill": "rbf", "No fill": "none"}
        fill = fill_map.get(fill_text, "idw")

        if self._output_worker_is_running():
            QMessageBox.warning(self, "Busy", "An output task is already running.")
            return
        if not self.workspace_path:
            QMessageBox.warning(self, "No workspace", "Save the workspace first.")
            return

        interp_path, output_dir = self._output_source_paths()
        self.log_text.clear()
        self._status_label.setText(f"Generating 3D PLYs for {len(channels)} channel(s)…")
        self._progress_bar.setValue(0)

        self._run_sensor_queue(
            "generate_sensor_3d_ply",
            list(channels),
            dict(
                cell_size=float(self.sensor_3d_cell_spin.value()),
                aggregation=self.sensor_3d_agg_combo.currentText(),
                fill_method=fill,
                zero_mask_pct=float(self.sensor_3d_zero_spin.value()),
                output_dir=output_dir,
                interp_path=interp_path,
            ),
            [],
            "3D Output complete",
            "3D output complete.",
            lambda r: f"{len(r)} PLY file(s) written.",
            lambda ch: f"Generating 3D PLY: {ch} …",
        )

    def _on_3d_run_selected(self, run_dir: str) -> None:
        """Called when the user clicks a 3D model run in the workspace panel.

        Routes to _selected_nav_run or _selected_sensor_run based on the path
        structure, so clicking a sensor_3d run never overwrites the nav target.
        """
        p = Path(run_dir)
        parent_name = p.parent.name        # "nav_trackline" | channel name
        grandparent_name = p.parent.parent.name  # "outputs" | "sensor_3d"

        is_nav    = (parent_name == "nav_trackline")
        is_sensor = (grandparent_name == "sensor_3d")

        if not is_nav and not is_sensor:
            # Unknown structure — don't silently route to the wrong target
            return

        label = f"{p.name}  ({parent_name})"
        green_style  = "color: #4caf50; font-style: normal; font-size: 10px;"
        inh_style    = "color: #aaa; font-size: 10px; font-style: italic;"

        # Read inherited settings from run.meta.json for display
        inherited_text = ""
        try:
            import json as _json
            meta_path = p / "run.meta.json"
            if meta_path.exists():
                meta = _json.loads(meta_path.read_text())
                s = meta.get("settings", {})
                parts = []
                if "cell_size_m" in s:
                    parts.append(f"cell {s['cell_size_m']} m")
                if "aggregation" in s:
                    parts.append(f"agg: {s['aggregation']}")
                if "fill_method" in s:
                    parts.append(f"fill: {s['fill_method']}")
                inherited_text = "  ·  ".join(parts)
        except Exception:
            pass

        if is_nav:
            self._selected_nav_run = run_dir
            if hasattr(self, "nav_slices_target_label"):
                self.nav_slices_target_label.setText(label)
                self.nav_slices_target_label.setStyleSheet(green_style)
            if hasattr(self, "nav_slices_inherited_label"):
                self.nav_slices_inherited_label.setText(inherited_text or "—")
                self.nav_slices_inherited_label.setStyleSheet(inh_style)
                self.nav_slices_inherited_label.setVisible(True)

        if is_sensor:
            self._selected_sensor_run = run_dir
            if hasattr(self, "sensor_slices_target_label"):
                self.sensor_slices_target_label.setText(label)
                self.sensor_slices_target_label.setStyleSheet(green_style)
            if hasattr(self, "sensor_slices_inherited_label"):
                self.sensor_slices_inherited_label.setText(inherited_text or "—")
                self.sensor_slices_inherited_label.setStyleSheet(inh_style)
                self.sensor_slices_inherited_label.setVisible(True)

        self._refresh_outputs_status()

    def _on_make_slices_requested(self, run_dir: str) -> None:
        """Called via workspace panel context menu: set target and navigate to the slices panel."""
        self._on_3d_run_selected(run_dir)

        p = Path(run_dir)
        is_nav    = (p.parent.name == "nav_trackline")
        is_sensor = (p.parent.parent.name == "sensor_3d")

        # Switch to Outputs tab
        if hasattr(self, "controls_tabs"):
            for i in range(self.controls_tabs.count()):
                if self.controls_tabs.tabText(i) == "Outputs":
                    self.controls_tabs.setCurrentIndex(i)
                    break

        # Switch to the correct sub-tab
        if hasattr(self, "outputs_subtabs"):
            target_sub = "Navigation" if is_nav else ("Sensors" if is_sensor else None)
            if target_sub:
                for i in range(self.outputs_subtabs.count()):
                    if self.outputs_subtabs.tabText(i) == target_sub:
                        self.outputs_subtabs.setCurrentIndex(i)
                        break

    def _generate_nav_slices(self) -> None:
        if not self._selected_nav_run:
            QMessageBox.warning(
                self, "No target selected",
                "Click a nav_trackline run in the Workspace panel to select it,\n"
                "or right-click and choose 'Make Slices from this run'."
            )
            return
        self._run_output_task(
            "generate_nav_slices_from_run",
            run_dir=self._selected_nav_run,
            altitude_step=float(self.nav_slices_step_spin.value()),
            pixels_per_cell=int(self.nav_slices_ppc_spin.value()),
        )

    def _generate_sensor_slices(self) -> None:
        if not self._selected_sensor_run:
            QMessageBox.warning(
                self, "No target selected",
                "Click a sensor_3d run in the Workspace panel to select it,\n"
                "or right-click and choose 'Make Slices from this run'."
            )
            return
        color_text = self.sensor_slices_color_combo.currentText()
        color_mode = "grayscale" if "grayscale" in color_text else "rgb"
        self._run_output_task(
            "generate_sensor_slices_from_run",
            run_dir=self._selected_sensor_run,
            altitude_step=float(self.sensor_slices_step_spin.value()),
            pixels_per_cell=int(self.sensor_slices_ppc_spin.value()),
            color_mode=color_mode,
            log_scale=self.sensor_slices_log_check.isChecked(),
            percentile_cap=float(self.sensor_slices_pct_spin.value()),
            local_norm=self.sensor_slices_local_norm_check.isChecked(),
        )

    def _generate_sensor_geo_slices(self) -> None:
        """Generate depth-slice GeoTIFFs (Float32, UTM) for selected sensor channels."""
        fill_text   = self.sensor_geo_slices_fill_combo.currentText()
        fill_method = (
            "idw"  if "IDW"      in fill_text else
            "rbf"  if "RBF"      in fill_text else
            "none"
        )
        channels = self._selected_sensor_channels()
        if not channels:
            QMessageBox.warning(self, "No channels", "Select at least one sensor channel.")
            return
        self._run_sensor_queue(
            "generate_depth_slice_geotiffs",
            channels,
            altitude_step=float(self.sensor_geo_slices_step_spin.value()),
            cell_size_m=float(self.sensor_geo_slices_cell_spin.value()),
            fill_method=fill_method,
        )

    def _generate_qgis_project(self) -> None:
        """Generate a QGIS .qgs project from all GeoTIFFs in the outputs directory."""
        self._run_output_task(
            "generate_qgis_project",
            project_name=self.qgis_project_name_edit.text().strip() or "EPR Survey",
        )

    def _open_ply_in_viewer(self, ply_path: str) -> None:
        """Open a PLY file in the singleton 3D viewer window."""
        from viewer_widget import get_viewer, viewer_available
        if not viewer_available():
            QMessageBox.information(
                self, "Viewer not available",
                "PyVista is not installed.\n\nInstall with:\n  pip install pyvista pyvistaqt"
            )
            return
        viewer = get_viewer(parent=self)
        viewer.load_file(ply_path)
        viewer.show()
        viewer.raise_()

    def _open_in_metashape_gui(self, psx_path: str) -> None:
        """Launch a Metashape .psx project in the native Metashape GUI."""
        import photogrammetry_service as ps
        try:
            ps.launch_in_metashape(psx_path)
        except Exception as exc:
            QMessageBox.warning(
                self, "Could not open Metashape",
                f"Failed to launch Metashape for:\n{psx_path}\n\n{exc}\n\n"
                "Ensure Metashape Professional is installed and on PATH."
            )

    def _refresh_sample_images_fields(self) -> None:
        """Enable or disable the sampling configuration fields based on the checkbox state.

        When "Sample images" is unchecked the mode/rate/spacing/quality fields are
        irrelevant and are grayed out.  Re-checking restores them, then
        _refresh_sampling_mode_ui trims to the correct subset for the current mode.
        """
        if not hasattr(self, "sampling_mode_combo"):
            return
        enabled = self.sample_images_check.isChecked()
        self.sampling_mode_combo.setEnabled(enabled)
        self.frame_quality_combo.setEnabled(enabled)
        if enabled:
            self._refresh_sampling_mode_ui()
        else:
            self.frame_rate_spin.setEnabled(False)
            self.dynamic_spacing_spin.setEnabled(False)
            self.dynamic_min_freq_spin.setEnabled(False)

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
        """No-op: threshold spinboxes removed from UI."""

    def _update_center_panel(self, index: int | None = None) -> None:
        """Switch the centre QStackedWidget to match the currently active left tab.

        Outputs / Video sub-tab → graphs panel (index 1, triggers graph refresh).
        Jobs tab                → map panel    (index 2, triggers map refresh).
        Any other tab           → timeline     (index 0).
        """
        if index is None:
            index = self.controls_tabs.currentIndex()
        tab_text = self.controls_tabs.tabText(index)
        if tab_text == "Outputs":
            # Show graphs when the Video sub-tab is active; timeline otherwise.
            if hasattr(self, "outputs_subtabs"):
                sub_text = self.outputs_subtabs.tabText(self.outputs_subtabs.currentIndex())
                if sub_text == "Video":
                    self.center_stack.setCurrentIndex(1)
                    self._refresh_postprocessing_graphs()
                else:
                    self.center_stack.setCurrentIndex(0)
                    self._refresh_outputs_status()
            else:
                self.center_stack.setCurrentIndex(0)
        elif tab_text == "Jobs":
            self.center_stack.setCurrentIndex(2)
            self._refresh_map()
        elif tab_text == "Manual Intervals":
            self.center_stack.setCurrentIndex(3)
            self._refresh_manual_map()
        elif tab_text == "Threshold Intervals":
            self.center_stack.setCurrentIndex(4)   # sensor graphs panel
            self._refresh_threshold_graphs()
        elif tab_text == "Inputs":
            self.center_stack.setCurrentIndex(0)   # coverage timeline
        else:
            self.center_stack.setCurrentIndex(5)   # blank

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
            source_line("Lat",     self.navigation_file.latitude_source),
            source_line("Lon",     self.navigation_file.longitude_source),
            source_line("Alt",     self.navigation_file.altitude_source),
            source_line("Depth",   self.navigation_file.depth_source) +
            ("  [negated]" if self.navigation_file.negate_depth else ""),
            source_line("Heading", self.navigation_file.heading_source),
            source_line("Pitch",   self.navigation_file.pitch_source),
            source_line("Roll",    self.navigation_file.roll_source),
        ]
        # Omit "<not set>" lines for optional sources that aren't configured.
        lines = [l for l in lines if "<not set>" not in l or l.startswith("Lat") or l.startswith("Lon")]
        self.navigation_summary.setPlainText("\n".join(lines))

    def _refresh_workspace_path_label(self) -> None:
        """Update the workspace path label and workspace file viewer."""
        if not hasattr(self, "workspace_path_label"):
            return
        if self.workspace_path:
            self.workspace_path_label.setText(f"Workspace: {self.workspace_path}")
        else:
            self.workspace_path_label.setText("Workspace: not saved")
        if hasattr(self, "_workspace_panel"):
            self._workspace_panel.set_root(self.workspace_path or None)

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
            f"Output: <workspace dir>/{self._job_output_dirname()}/" if self.workspace_path else "Output: <save workspace first>",
            f"Sampling: {self.sampling_mode_combo.currentText()}",
            f"Frame rate: {self.frame_rate_spin.value():.2f} Hz" if self.sampling_mode_combo.currentText() == "Fixed rate" else f"Target spacing: {self.dynamic_spacing_spin.value():.2f} m  |  f_min: {self.dynamic_min_freq_spin.value():.3f} Hz",
            f"Quality: {self.frame_quality_combo.currentText()}",
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

        alt_thresh   = float(self.altitude_threshold or 0.0)
        depth_thresh = float(self.depth_threshold   or 0.0)
        speed_thresh = float(self.speed_threshold   or 0.0)

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
        """No-op: sensor threshold UI removed from Processing tab.

        Per-channel thresholds are now defined via the Threshold Intervals tab.
        This stub is kept so callers don't need to change.
        """
        if not hasattr(self, "sensor_thresholds_layout"):
            return  # UI removed — nothing to rebuild
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
        """Returns empty dict — sensor threshold UI removed from Processing tab."""
        if not hasattr(self, "sensor_thresholds_layout"):
            return {}

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
        root = Path(self.workspace_path)
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

        if self.controls_tabs.tabText(self.controls_tabs.currentIndex()) == "Visualization":
            self._refresh_map()

    # ------------------------------------------------------------------
    # Sensor Tracklines handlers
    # ------------------------------------------------------------------

    def _load_sensor_rasters(self) -> None:
        """Load sensor tracklines directly from the configured nav + sensor sources."""
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
        combo = self.raster_channel_combo
        if combo.currentText() == "None" and combo.count() > 1:
            combo.blockSignals(True)
            combo.setCurrentIndex(1)
            combo.blockSignals(False)
            self._set_raster_controls_enabled(True)
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
        """Export the full sensor raster DataFrame as a CSV file."""
        if self._raster_df is None or len(self._raster_df) == 0:
            QMessageBox.information(self, "Nothing to export", "Load sensor tracklines first.")
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

    def _raster_export_qgis_geojson(self) -> None:
        """Export the Sensor Tracklines trackline as a QGIS-ready GeoJSON file.

        Uses _raster_df (sensor-native timestamps, GPS-geolocated) to build one
        LineString Feature per consecutive GPS point pair.  Load in QGIS and
        style with Graduated symbology on 'sensor_value'.
        """
        if self._raster_df is None or len(self._raster_df) == 0:
            QMessageBox.information(self, "Nothing to export",
                                    "Load sensor tracklines first.")
            return

        channel = self.raster_channel_combo.currentText()
        if channel == "None":
            QMessageBox.information(self, "No channel selected",
                                    "Select a channel in the Sensor Tracklines panel first.")
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
            QMessageBox.information(self, "Nothing to export", "Load sensor tracklines first.")
            return
        channel = self.raster_channel_combo.currentText()
        if channel == "None":
            QMessageBox.information(self, "No channel selected",
                                    "Select a sensor channel in the Sensor Tracklines panel first.")
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
                                    "Load sensor tracklines first.")
            return

        channel = self.raster_channel_combo.currentText()
        if channel == "None":
            QMessageBox.information(self, "No channel selected",
                                    "Select a channel in the Sensor Tracklines panel first.")
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
        except ImportError:
            QMessageBox.critical(self, "Missing dependency",
                                 "matplotlib is required for map export.\n"
                                 "Install it with: pip install matplotlib")
            return

        try:
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
            fig, ax = plt.subplots(figsize=(11, 8.5))  # US letter landscape

            # ---- GPS track (plain red — sensor coloring is in Export Raster Map) ----
            if len(nav_lons) >= 2:
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

            # ---- Legend ----
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

    # -----------------------------------------------------------------------
    # Job Builder handlers (Jobs tab)
    # -----------------------------------------------------------------------

    def _refresh_job_builder_ui(self) -> None:
        """Rebuild the job interval list, name field, and Execute button state."""
        if hasattr(self, "job_name_edit"):
            self.job_name_edit.blockSignals(True)
            if not self.job_name_edit.text() and self.pending_job.name:
                self.job_name_edit.setText(self.pending_job.name)
            self.job_name_edit.blockSignals(False)
        if hasattr(self, "job_interval_list"):
            self.job_interval_list.clear()
            for interval in self.pending_job.intervals:
                src_tag = f"[{'M' if interval.source == 'manual' else 'T'}]"
                label = (
                    f"{src_tag}  "
                    f"{interval.start_time.strftime('%Y-%m-%d %H:%M:%S')}  →  "
                    f"{interval.end_time.strftime('%H:%M:%S')}"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, id(interval))
                self.job_interval_list.addItem(item)
        # Gray out the Execute button when there's nothing to run.
        if hasattr(self, "execute_job_btn"):
            has_intervals = bool(self.pending_job.intervals)
            self.execute_job_btn.setEnabled(has_intervals)
            if not has_intervals:
                self.execute_job_btn.setToolTip(
                    "Add at least one interval to the job before executing.\n"
                    "Use the Manual Intervals or Threshold Intervals tab."
                )
            else:
                self.execute_job_btn.setToolTip(
                    "Run the pipeline for the current job."
                )

    def _refresh_job_history_dropdown(self) -> None:
        """Rebuild the Jobs-tab interval history lists and job history list."""
        # Interval histories
        for lst, source in (
            (getattr(self, "jobs_manual_history_list", None), "manual"),
            (getattr(self, "jobs_threshold_history_list", None), "threshold"),
        ):
            if lst is None:
                continue
            lst.clear()
            recs = [r for r in self.segment_history if r.interval.source == source]
            for rec in reversed(recs):
                label = (
                    f"{rec.interval.start_time.strftime('%Y-%m-%d %H:%M:%S')} → "
                    f"{rec.interval.end_time.strftime('%H:%M:%S')}  "
                    f"[{rec.job_name or f'Job #{rec.job_id}'}]  [{rec.status}]"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, rec)
                lst.addItem(item)
        # Manual interval tab history list
        if hasattr(self, "manual_history_list"):
            self.manual_history_list.clear()
            for rec in reversed(self.segment_history):
                label = (
                    f"[{'M' if rec.interval.source == 'manual' else 'T'}]  "
                    f"{rec.interval.start_time.strftime('%H:%M:%S')} → "
                    f"{rec.interval.end_time.strftime('%H:%M:%S')}  "
                    f"[{rec.job_name or f'Job #{rec.job_id}'}]"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, rec)
                self.manual_history_list.addItem(item)
        # Job history list
        if hasattr(self, "job_history_list"):
            self.job_history_list.clear()
            for job in reversed(self.job_history):
                label = (
                    f"[{job.status}]  {job.name or f'Job #{job.job_id}'}  "
                    f"— {len(job.intervals)} interval(s)"
                )
                item = QListWidgetItem(label)
                item.setData(Qt.UserRole, job)
                self.job_history_list.addItem(item)

    def _job_new(self) -> None:
        """Start a fresh job with the next serial ID."""
        self.next_job_id += 1
        self.pending_job = Job(job_id=self.next_job_id)
        self._loaded_from_completed_job = False
        if hasattr(self, "job_name_edit"):
            self.job_name_edit.clear()
        self._refresh_extraction_section_state()
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()
        self._refresh_summary()
        self._auto_save_workspace()

    def _job_clear(self) -> None:
        """Remove all intervals from the current job (keep name and ID)."""
        self.pending_job.intervals.clear()
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()
        self._refresh_summary()
        self._auto_save_workspace()

    def _job_save(self) -> None:
        """Save the current job to job history without executing.

        If the user hasn't typed a name yet, a default name is assigned
        automatically.  A name is only *required* at execution time.
        """
        name = self.job_name_edit.text().strip() if hasattr(self, "job_name_edit") else ""
        if not name:
            name = f"Job #{self.pending_job.job_id}"
            if hasattr(self, "job_name_edit"):
                self.job_name_edit.setText(name)
        self.pending_job.name = name
        self.pending_job.status = "saved"
        self.pending_job.settings_snapshot = self._collect_settings_snapshot()
        # Avoid duplicates — update if same job_id already in history
        existing = next((i for i, j in enumerate(self.job_history)
                         if j.job_id == self.pending_job.job_id), None)
        saved = copy.deepcopy(self.pending_job)
        if existing is not None:
            self.job_history[existing] = saved
        else:
            self.job_history.append(saved)
        self._refresh_job_history_dropdown()
        self._auto_save_workspace()
        self.log_text.append(f"Job '{name}' saved to history.")

    def _refresh_extraction_section_state(self) -> None:
        """Update the sampling status label and checkbox default for the current job.

        When a completed job is loaded the status label turns green, the
        "Sample images" checkbox defaults to unchecked (graying out the sampling
        fields via _refresh_sample_images_fields), and the "Create New Sampling
        Job…" button is shown.  For a fresh pending job the label shows "not yet
        extracted", the checkbox defaults to checked, and the button is hidden.
        """
        if not hasattr(self, "frame_extract_group"):
            return
        locked = self._loaded_from_completed_job
        self.new_sampling_job_button.setVisible(locked)

        if locked:
            name = self.pending_job.name or f"Job #{self.pending_job.job_id}"
            self.sampling_status_label.setText(
                f'Frames already extracted for "{name}". '
                "Uncheck to run post-processing only, or use "
                '"Create New Sampling Job…" to re-sample into a new job folder.'
            )
            self.sampling_status_label.setStyleSheet(
                "font-size: 11px; color: #155724; background: #d4edda; "
                "border: 1px solid #c3e6cb; border-radius: 3px; padding: 4px 6px;"
            )
            self.sample_images_check.setChecked(False)
        else:
            self.sampling_status_label.setText("No frames extracted yet for this job.")
            self.sampling_status_label.setStyleSheet(
                "font-size: 11px; color: #6c757d; padding: 2px 0px;"
            )
            self.sample_images_check.setChecked(True)

    def _on_new_sampling_job_clicked(self) -> None:
        """Prompt for a new job name and create a versioned job for re-sampling.

        No new workspace is created — the new job writes to a new subdirectory
        (job_NNN/) inside the existing workspace directory.
        """
        default_name = self._versioned_job_name(self.pending_job.name)
        name, ok = QInputDialog.getText(
            self,
            "New Sampling Job",
            "Enter a name for the new job:\n"
            "(Frames will be extracted into a new subfolder of this workspace.)",
            text=default_name,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        self.next_job_id += 1
        new_job = Job(
            job_id=self.next_job_id,
            name=name,
            intervals=list(self.pending_job.intervals),
        )
        self.pending_job = new_job
        self._loaded_from_completed_job = False
        if hasattr(self, "job_name_edit"):
            self.job_name_edit.setText(name)
        # Re-check extraction by default for the new job
        self.sample_images_check.setChecked(True)
        self.postprocess_extract_check.setChecked(True)
        self._refresh_extraction_section_state()
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()
        self.log_text.append(
            f"New sampling job '{name}' created (Job #{new_job.job_id}).  "
            "Frame extraction is now enabled."
        )

    def _job_output_dirname(self, job: "Job | None" = None) -> str:
        """Return the output subdirectory name for a job.

        Format: ``job_001_name`` when the job has a name, ``job_001`` otherwise.
        Non-alphanumeric characters (except hyphens) are replaced with underscores
        so the result is always a valid directory name on all platforms.
        """
        import re
        j = job or self.pending_job
        base = f"job_{j.job_id:03d}"
        if j.name:
            safe = re.sub(r"[^\w\-]", "_", j.name).strip("_")
            return f"{base}_{safe}" if safe else base
        return base

    def _versioned_job_name(self, name: str) -> str:
        """Return name_v2, name_v3, … avoiding any name already in job_history."""
        import re
        m = re.match(r'^(.*?)_v(\d+)$', name)
        base = m.group(1) if m else name
        ver  = int(m.group(2)) if m else 1
        existing = {j.name for j in self.job_history}
        while True:
            ver += 1
            candidate = f"{base}_v{ver}"
            if candidate not in existing:
                return candidate

    def _collect_settings_snapshot(self) -> dict:
        """Capture all pipeline settings for recording in job/segment history."""
        return {
            # Frame extraction
            "frame_rate":              float(self.frame_rate_spin.value()),
            "sampling_mode":           self.sampling_mode_combo.currentText(),
            "dynamic_spacing_m":       float(self.dynamic_spacing_spin.value()),
            "dynamic_min_freq_hz":     float(self.dynamic_min_freq_spin.value()),
            "frame_quality":           self.frame_quality_combo.currentText(),
            "sample_images":           self.sample_images_check.isChecked(),
            "generate_rasters":        self.generate_rasters_check.isChecked(),
            "annotate":                self.annotate_frames_check.isChecked(),
            # Post-processing steps
            "pp_extract":              self.postprocess_extract_check.isChecked(),
            "pp_rasters":              self.postprocess_generate_rasters_check.isChecked(),
            "pp_annotate":             self.postprocess_annotate_check.isChecked(),
            "pp_clahe":                self.postprocess_clahe_check.isChecked(),
            "pp_update_master":        self.postprocess_update_master_check.isChecked(),
            "pp_geo_txt":              self.postprocess_geo_txt_check.isChecked(),
            # CLAHE params
            "clahe_clip_limit":        float(self.clahe_clip_limit_spin.value()),
            "clahe_tile_grid_size":    int(self.clahe_tile_size_spin.value()),
            # Thresholds (state only — no UI widgets)
            "altitude_threshold":      float(self.altitude_threshold or 0.0),
            "depth_threshold":         float(self.depth_threshold or 0.0),
            "speed_threshold":         float(self.speed_threshold or 0.0),
            "min_segment_frames":      int(self.min_segment_frames),
        }

    def _apply_settings_snapshot(self, snap: dict) -> None:
        """Restore pipeline UI widgets from a saved settings snapshot."""
        if not snap:
            return
        def _set(spin, key, cast=float):
            if key in snap:
                spin.blockSignals(True)
                spin.setValue(cast(snap[key]))
                spin.blockSignals(False)
        def _set_check(check, key):
            if key in snap:
                check.setChecked(bool(snap[key]))
        def _set_combo(combo, key):
            if key in snap:
                idx = combo.findText(str(snap[key]))
                if idx >= 0:
                    combo.setCurrentIndex(idx)

        _set(self.frame_rate_spin,          "frame_rate")
        _set_combo(self.sampling_mode_combo, "sampling_mode")
        _set(self.dynamic_spacing_spin,     "dynamic_spacing_m")
        _set(self.dynamic_min_freq_spin,    "dynamic_min_freq_hz")
        _set_combo(self.frame_quality_combo,"frame_quality")
        _set_check(self.sample_images_check,        "sample_images")
        _set_check(self.generate_rasters_check,     "generate_rasters")
        _set_check(self.annotate_frames_check,      "annotate")
        _set_check(self.postprocess_extract_check,  "pp_extract")
        _set_check(self.postprocess_generate_rasters_check, "pp_rasters")
        _set_check(self.postprocess_annotate_check, "pp_annotate")
        _set_check(self.postprocess_clahe_check,    "pp_clahe")
        _set_check(self.postprocess_update_master_check, "pp_update_master")
        _set_check(self.postprocess_geo_txt_check,  "pp_geo_txt")
        _set(self.clahe_clip_limit_spin,    "clahe_clip_limit")
        _set(self.clahe_tile_size_spin,     "clahe_tile_grid_size", int)
        # Thresholds restored to state variables (no UI widgets)
        if "altitude_threshold"  in snap: self.altitude_threshold  = float(snap["altitude_threshold"])
        if "depth_threshold"     in snap: self.depth_threshold     = float(snap["depth_threshold"])
        if "speed_threshold"     in snap: self.speed_threshold     = float(snap["speed_threshold"])
        if "min_segment_frames"  in snap: self.min_segment_frames  = int(snap["min_segment_frames"])

    def _job_load_from_history(self) -> None:
        """Load the selected job from job history into the current job."""
        selected = self.job_history_list.selectedItems()
        if not selected:
            return
        job: Job = selected[0].data(Qt.UserRole)
        was_completed = (job.status == "completed")
        self.pending_job = copy.deepcopy(job)
        self.pending_job.status = "pending"
        # Track whether we loaded a completed job so _run_pipeline can behave correctly.
        self._loaded_from_completed_job = was_completed
        if hasattr(self, "job_name_edit"):
            self.job_name_edit.setText(self.pending_job.name)
        # Restore the settings that were active when this job was saved/executed.
        self._apply_settings_snapshot(self.pending_job.settings_snapshot)
        # If the job has already been executed, uncheck frame extraction by default:
        # frames already exist, so only post-processing steps should run.  The user
        # can re-check extraction to create a versioned new job.
        if was_completed:
            self.log_text.append(
                f"Loaded completed job '{job.name}' — settings restored. "
                "Frame extraction is locked (frames exist). "
                "Click 'Create New Sampling Job…' on the Processing tab to re-sample."
            )
        else:
            self.log_text.append(
                f"Loaded job '{self.pending_job.name}' — settings restored from snapshot."
            )
        self._refresh_extraction_section_state()
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()

    def _job_history_context_menu(self, pos) -> None:
        """Right-click on job history list → delete option."""
        from PySide6.QtWidgets import QMenu
        item = self.job_history_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        delete_act = menu.addAction("Delete job from history")
        if menu.exec(self.job_history_list.mapToGlobal(pos)) == delete_act:
            job: Job = item.data(Qt.UserRole)
            self.job_history = [j for j in self.job_history if j.job_id != job.job_id]
            self._refresh_job_history_dropdown()
            self._auto_save_workspace()

    def _jobs_history_selection_changed(self) -> None:
        """Highlight selected history intervals on the Jobs-tab map in their source color."""
        self._refresh_history_overlay()

    def _jobs_add_manual_history_to_job(self) -> None:
        for item in self.jobs_manual_history_list.selectedItems():
            rec: SegmentRecord = item.data(Qt.UserRole)
            self.pending_job.intervals.append(
                SelectedTimeRange(
                    start_time=rec.interval.start_time,
                    end_time=rec.interval.end_time,
                    source="manual",
                )
            )
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()

    def _jobs_add_threshold_history_to_job(self) -> None:
        for item in self.jobs_threshold_history_list.selectedItems():
            rec: SegmentRecord = item.data(Qt.UserRole)
            self.pending_job.intervals.append(
                SelectedTimeRange(
                    start_time=rec.interval.start_time,
                    end_time=rec.interval.end_time,
                    source="threshold",
                    threshold_desc=rec.interval.threshold_desc,
                )
            )
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()

    def _interval_list_context_menu(self, list_widget, pos, source: str) -> None:
        """Generic right-click context menu for any interval list widget."""
        from PySide6.QtWidgets import QMenu
        item = list_widget.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)

        if source == "pending_job":
            del_act = menu.addAction("Remove from job")
            act = menu.exec(list_widget.mapToGlobal(pos))
            if act == del_act:
                row = list_widget.row(item)
                if 0 <= row < len(self.pending_job.intervals):
                    del self.pending_job.intervals[row]
                    self._refresh_job_builder_ui()
                    self._refresh_history_overlay()

        elif source == "manual_staged":
            del_act = menu.addAction("Remove from staged list")
            act = menu.exec(list_widget.mapToGlobal(pos))
            if act == del_act:
                row = list_widget.row(item)
                if 0 <= row < len(self._manual_staged):
                    del self._manual_staged[row]
                    self._refresh_manual_staged_list()

        elif source in ("manual_history", "threshold_history"):
            # For segment records in history lists
            open_act  = menu.addAction("Open output folder")
            del_act   = menu.addAction("Delete from interval history")
            act = menu.exec(list_widget.mapToGlobal(pos))
            rec: SegmentRecord | None = item.data(Qt.UserRole)
            if rec is None:
                return
            if act == open_act:
                import subprocess
                folder = rec.output_path
                if Path(folder).exists():
                    subprocess.Popen(["xdg-open", folder])
                else:
                    QMessageBox.warning(self, "Folder not found",
                                        f"The folder no longer exists:\n{folder}")
            elif act == del_act:
                self.segment_history = [
                    r for r in self.segment_history
                    if not (r.job_id == rec.job_id and
                            r.interval.start_time == rec.interval.start_time)
                ]
                self._refresh_job_history_dropdown()
                self._refresh_history_overlay()
                self._auto_save_workspace()

        elif source == "threshold_results":
            del_act = menu.addAction("Remove from results")
            act = menu.exec(list_widget.mapToGlobal(pos))
            if act == del_act:
                row = list_widget.row(item)
                if 0 <= row < len(self._threshold_intervals):
                    del self._threshold_intervals[row]
                    self.threshold_results_list.takeItem(row)
                    self.threshold_results_label.setText(
                        f"{len(self._threshold_intervals)} intervals found."
                    )

    # -----------------------------------------------------------------------
    # Threshold Interval tab handlers
    # -----------------------------------------------------------------------

    def _threshold_add_constraint(self) -> None:
        """Add a new threshold constraint row to the constraint builder."""
        row_widget = QWidget()
        row_layout = QHBoxLayout(row_widget)
        row_layout.setContentsMargins(0, 0, 0, 0)

        # Enable checkbox
        enabled_check = QCheckBox()
        enabled_check.setChecked(True)
        row_layout.addWidget(enabled_check)

        # Channel selector
        channel_combo = QComboBox()
        channel_combo.setMinimumWidth(120)
        # Populate from nav + sensor channels
        for key in ("alt", "depth"):
            channel_combo.addItem(key)
        for sf in self.sensor_files:
            for ch in sf.channels:
                channel_combo.addItem(ch.display_name or ch.source_column)
        row_layout.addWidget(channel_combo, stretch=1)

        row_layout.addWidget(QLabel("Min:"))
        min_edit = QLineEdit()
        min_edit.setPlaceholderText("none")
        min_edit.setMaximumWidth(70)
        min_edit.returnPressed.connect(self._threshold_calculate)
        row_layout.addWidget(min_edit)

        row_layout.addWidget(QLabel("Max:"))
        max_edit = QLineEdit()
        max_edit.setPlaceholderText("none")
        max_edit.setMaximumWidth(70)
        max_edit.returnPressed.connect(self._threshold_calculate)
        row_layout.addWidget(max_edit)

        # Remove button
        remove_btn = QPushButton("✕")
        remove_btn.setMaximumWidth(26)
        remove_btn.setStyleSheet("color: #c0392b; font-weight: bold;")
        row_data = {"widget": row_widget, "channel_combo": channel_combo,
                    "min_edit": min_edit, "max_edit": max_edit,
                    "enabled_check": enabled_check}
        remove_btn.clicked.connect(lambda: self._threshold_remove_constraint(row_data))
        row_layout.addWidget(remove_btn)

        self._constraint_rows.append(row_data)
        # Insert before the "Add constraint" button (last item in layout)
        idx = self._constraint_group_layout.count() - 1
        self._constraint_group_layout.insertWidget(idx, row_widget)

    def _threshold_remove_constraint(self, row_data: dict) -> None:
        if len(self._constraint_rows) <= 1:
            return  # keep at least one row
        self._constraint_rows.remove(row_data)
        row_data["widget"].deleteLater()

    def _threshold_calculate(self) -> None:
        """Run threshold analysis and populate the results list."""
        constraints = []
        for row in self._constraint_rows:
            if not row["enabled_check"].isChecked():
                continue
            channel = row["channel_combo"].currentText().strip()
            if not channel:
                continue
            min_text = row["min_edit"].text().strip()
            max_text = row["max_edit"].text().strip()
            try:
                min_val = float(min_text) if min_text else None
            except ValueError:
                min_val = None
            try:
                max_val = float(max_text) if max_text else None
            except ValueError:
                max_val = None
            constraints.append({"channel": channel, "min_val": min_val, "max_val": max_val})

        if not constraints:
            QMessageBox.information(self, "No constraints", "Enter at least one threshold constraint.")
            return

        min_dur = self.threshold_min_dur_spin.value()
        try:
            from pipeline_service import PipelineService
            intervals = PipelineService.find_threshold_intervals(
                navigation_file=self.navigation_file,
                sensor_files=self.sensor_files,
                constraints=constraints,
                min_duration_s=min_dur,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Threshold analysis failed", str(exc))
            return

        self._threshold_intervals = intervals
        self.threshold_results_list.clear()
        for iv in intervals:
            dur = (iv.end_time - iv.start_time).total_seconds()
            label = (
                f"{iv.start_time.strftime('%H:%M:%S')} → "
                f"{iv.end_time.strftime('%H:%M:%S')}  "
                f"({dur:.1f} s)  {iv.threshold_desc}"
            )
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, iv)
            self.threshold_results_list.addItem(item)
        self.threshold_results_label.setText(f"{len(intervals)} intervals found.")

    def _threshold_add_to_job(self) -> None:
        """Add selected (or all) threshold results to the pending job."""
        selected = self.threshold_results_list.selectedItems()
        source_items = selected if selected else [
            self.threshold_results_list.item(i)
            for i in range(self.threshold_results_list.count())
        ]
        if not source_items:
            QMessageBox.information(self, "Nothing to add",
                                    "Calculate intervals first, then click Add.")
            return
        for item in source_items:
            iv: SelectedTimeRange = item.data(Qt.UserRole)
            self.pending_job.intervals.append(iv)
        count = len(source_items)
        total = len(self.pending_job.intervals)
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()
        msg = f"Added {count} threshold interval(s) → Job #{self.pending_job.job_id} now has {total} interval(s).  Switch to the Jobs tab to review."
        self.log_text.append(msg)
        self.threshold_results_label.setText(
            f"✓ {count} interval(s) added to Job #{self.pending_job.job_id}  ({total} total).  "
            f"View them on the Jobs tab."
        )
        self.threshold_results_label.setStyleSheet(
            "font-size: 11px; color: #27ae60; font-weight: bold;"
        )
        self._auto_save_workspace()

    def _threshold_load_saved(self) -> None:
        """Load a saved threshold config and restore its constraints."""
        selected = self.threshold_history_list.selectedItems()
        if not selected:
            return
        cfg: ThresholdConfig = selected[0].data(Qt.UserRole)
        # Clear existing constraint rows
        while len(self._constraint_rows) > 0:
            row = self._constraint_rows[-1]
            row["widget"].deleteLater()
            self._constraint_rows.pop()
        # Re-create rows from saved config
        for c in cfg.constraints:
            self._threshold_add_constraint()
            row = self._constraint_rows[-1]
            idx = row["channel_combo"].findText(c.channel)
            if idx >= 0:
                row["channel_combo"].setCurrentIndex(idx)
            row["min_edit"].setText("" if c.min_val is None else str(c.min_val))
            row["max_edit"].setText("" if c.max_val is None else str(c.max_val))

    def _threshold_history_context_menu(self, pos) -> None:
        """Right-click on threshold history list → delete option."""
        from PySide6.QtWidgets import QMenu
        item = self.threshold_history_list.itemAt(pos)
        if item is None:
            return
        menu = QMenu(self)
        del_act = menu.addAction("Delete from history")
        if menu.exec(self.threshold_history_list.mapToGlobal(pos)) == del_act:
            cfg: ThresholdConfig = item.data(Qt.UserRole)
            self.threshold_history = [
                t for t in self.threshold_history if t is not cfg
            ]
            self._refresh_threshold_history_list()

    def _refresh_threshold_history_list(self) -> None:
        """Rebuild the threshold history list widget."""
        if not hasattr(self, "threshold_history_list"):
            return
        self.threshold_history_list.clear()
        for cfg in reversed(self.threshold_history):
            item = QListWidgetItem(cfg.label())
            item.setData(Qt.UserRole, cfg)
            self.threshold_history_list.addItem(item)

    def _refresh_threshold_graphs(self) -> None:
        """Build time-series sensor graphs on the Threshold Intervals tab."""
        if not hasattr(self, "_threshold_graph_layout"):
            return
        # Remove all items except the placeholder (index 0) and trailing stretch
        while self._threshold_graph_layout.count() > 2:
            item = self._threshold_graph_layout.takeAt(1)
            if item and item.widget():
                item.widget().deleteLater()

        sources: list[tuple[str, object]] = []
        if self.navigation_file:
            for label, src in (
                ("Altitude (m)", self.navigation_file.altitude_source),
                ("Depth (m)",    self.navigation_file.depth_source),
            ):
                if src is not None:
                    sources.append((label, src))
        for sf in self.sensor_files:
            for ch in sf.channels:
                sources.append((ch.display_name or ch.source_column, (sf, ch)))

        if not sources:
            self.threshold_graph_placeholder.show()
            return
        self.threshold_graph_placeholder.hide()

        for label, src in sources[:6]:  # cap at 6 graphs to avoid clutter
            try:
                if isinstance(src, tuple):
                    sf, ch = src
                    df = SensorService.load_sensor_dataframe(sf)
                    times = df["unix_time"].to_numpy(dtype=float)
                    vals  = df[ch.source_column].to_numpy(dtype=float)
                else:
                    df    = SensorService.load_time_value_dataframe(src)
                    times = df["unix_time"].to_numpy(dtype=float)
                    vals  = df["value"].to_numpy(dtype=float)
                plot = self._create_timeseries_plot(label)
                plot.setMaximumHeight(180)
                plot.plot(times, vals, pen=pg.mkPen(color="#2196f3", width=1))
                self._threshold_graph_layout.insertWidget(
                    self._threshold_graph_layout.count() - 1, plot
                )
            except Exception:
                pass

    # -----------------------------------------------------------------------
    # Manual Interval tab handlers
    # -----------------------------------------------------------------------

    def _manual_toggle_pick_mode(self, enabled: bool) -> None:
        """Toggle two-click pick mode on the manual map widget."""
        self.manual_map_widget.set_pick_mode(enabled)
        self.manual_pick_button.setText("Pick Interval  ✓" if enabled else "Pick Interval")
        if enabled:
            self.manual_pick_status.setText("Click first point on the track…")
            self.manual_pick_status.setStyleSheet("font-size: 10px; color: #e67e22; font-weight: bold;")
        else:
            self.manual_pick_status.setText("Click to activate, then click two points on the track.")
            self.manual_pick_status.setStyleSheet("font-size: 10px; color: gray;")

    def _on_manual_pick_point_placed(self, unix_t: float) -> None:
        """Update status label when the first pick point is placed."""
        dt_str = datetime.utcfromtimestamp(unix_t).strftime("%H:%M:%S")
        self.manual_pick_status.setText(
            f"Start: {dt_str} — now click the end point."
        )
        self.manual_pick_status.setStyleSheet("font-size: 10px; color: #e67e22; font-weight: bold;")

    def _on_manual_interval_picked(self, t_start: float, t_end: float) -> None:
        """Handle a completed two-click interval pick on the manual map."""
        self.manual_pick_button.setChecked(False)
        start_dt = datetime.utcfromtimestamp(t_start)
        end_dt   = datetime.utcfromtimestamp(t_end)
        interval = SelectedTimeRange(start_time=start_dt, end_time=end_dt, source="manual")
        self._manual_staged.append(interval)
        self._refresh_manual_staged_list()
        self.manual_pick_status.setText(
            f"Added: {start_dt.strftime('%H:%M:%S')} → {end_dt.strftime('%H:%M:%S')}"
        )
        self.manual_pick_status.setStyleSheet("font-size: 10px; color: #27ae60; font-weight: bold;")

    def _refresh_manual_staged_list(self) -> None:
        """Rebuild the staged interval list on the Manual Interval tab."""
        if not hasattr(self, "manual_staged_list"):
            return
        self.manual_staged_list.clear()
        for iv in self._manual_staged:
            dur = (iv.end_time - iv.start_time).total_seconds()
            label = (
                f"{iv.start_time.strftime('%Y-%m-%d %H:%M:%S')} → "
                f"{iv.end_time.strftime('%H:%M:%S')}  ({dur:.0f} s)"
            )
            self.manual_staged_list.addItem(label)

    def _manual_clear_staged(self) -> None:
        self._manual_staged.clear()
        self._refresh_manual_staged_list()

    def _manual_add_from_history(self) -> None:
        """Stage selected recorded intervals from the manual history list."""
        for item in self.manual_history_list.selectedItems():
            rec: SegmentRecord = item.data(Qt.UserRole)
            iv = SelectedTimeRange(
                start_time=rec.interval.start_time,
                end_time=rec.interval.end_time,
                source=rec.interval.source,
                threshold_desc=rec.interval.threshold_desc,
            )
            self._manual_staged.append(iv)
        self._refresh_manual_staged_list()

    def _manual_history_selection_changed(self) -> None:
        """Highlight selected history intervals on the manual map."""
        if not hasattr(self, "manual_map_widget"):
            return
        ranges = []
        for item in self.manual_history_list.selectedItems():
            rec: SegmentRecord = item.data(Qt.UserRole)
            t0 = float(calendar.timegm(rec.interval.start_time.timetuple()))
            t1 = float(calendar.timegm(rec.interval.end_time.timetuple()))
            color = "#2196f3" if rec.interval.source == "manual" else "#4caf50"
            tooltip = (
                f"{'Manual' if rec.interval.source == 'manual' else 'Threshold'}  ·  "
                f"{rec.interval.start_time.strftime('%H:%M:%S')} → "
                f"{rec.interval.end_time.strftime('%H:%M:%S')}"
            )
            ranges.append((t0, t1, color, tooltip))
        self.manual_map_widget.set_history_ranges(ranges)

    def _manual_add_to_job(self) -> None:
        """Move all staged intervals to the pending job."""
        if not self._manual_staged:
            QMessageBox.information(self, "Nothing staged",
                                    "Pick intervals on the map or load from history first.")
            return
        for iv in self._manual_staged:
            self.pending_job.intervals.append(iv)
        count = len(self._manual_staged)
        total = len(self.pending_job.intervals)
        self._manual_staged.clear()
        self._refresh_manual_staged_list()
        self._refresh_job_builder_ui()
        self._refresh_history_overlay()
        msg = f"Added {count} interval(s) → Job #{self.pending_job.job_id} now has {total} interval(s).  Switch to the Jobs tab to review."
        self.log_text.append(msg)
        # Show confirmation on the current tab via the pick status label
        if hasattr(self, "manual_pick_status"):
            self.manual_pick_status.setText(
                f"✓ {count} interval(s) added to Job #{self.pending_job.job_id}  "
                f"({total} total).  View them on the Jobs tab."
            )
            self.manual_pick_status.setStyleSheet(
                "font-size: 10px; color: #27ae60; font-weight: bold;"
            )
        self._auto_save_workspace()

    def _on_history_mode_toggled(self, enabled: bool) -> None:
        """Switch the map trackline between normal mode and history overlay mode."""
        self.viz_history_mode_button.setText(
            "History Mode  ✓" if enabled else "History Overlay"
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
        """Push interval ranges to the Jobs-tab map with color coding:

          Yellow  — intervals in the current pending job (not yet sampled)
          Blue    — manually-picked sampled intervals (selected in history list)
          Green   — threshold-picked sampled intervals (selected in history list)
          History mode off → only yellow (pending job) is shown.
        """
        if not hasattr(self, "map_widget"):
            return

        ranges: list[tuple[float, float, str, str]] = []

        # Yellow: pending job intervals
        for iv in self.pending_job.intervals:
            t0 = float(calendar.timegm(iv.start_time.timetuple()))
            t1 = float(calendar.timegm(iv.end_time.timetuple()))
            tooltip = (
                f"Pending job  ·  "
                f"{iv.start_time.strftime('%H:%M:%S')} → {iv.end_time.strftime('%H:%M:%S')}"
            )
            ranges.append((t0, t1, "#ffd600", tooltip))   # yellow

        # Blue/green: selected history items (only when history mode is on)
        history_on = getattr(self, "viz_history_mode_button", None) and \
                     self.viz_history_mode_button.isChecked()
        if history_on:
            # Collect checked manual intervals
            for lst, color, source in (
                (getattr(self, "jobs_manual_history_list", None),    "#2196f3", "manual"),
                (getattr(self, "jobs_threshold_history_list", None), "#4caf50", "threshold"),
            ):
                if lst is None:
                    continue
                for item in lst.selectedItems():
                    rec: SegmentRecord = item.data(Qt.UserRole)
                    if rec is None:
                        continue
                    t0 = float(calendar.timegm(rec.interval.start_time.timetuple()))
                    t1 = float(calendar.timegm(rec.interval.end_time.timetuple()))
                    tooltip = (
                        f"{'Manual' if source == 'manual' else 'Threshold'}  ·  "
                        f"{rec.interval.start_time.strftime('%H:%M:%S')} → "
                        f"{rec.interval.end_time.strftime('%H:%M:%S')}  ·  "
                        f"{rec.job_name or f'Job #{rec.job_id}'}  ·  {rec.status}"
                    )
                    ranges.append((t0, t1, color, tooltip))

        self.map_widget.set_history_ranges(ranges)

    def _refresh_manual_map(self) -> None:
        """Push the nav trackline to the Manual Interval tab map widget.

        Feeds altitude (for the hover tooltip) and video coverage windows so the
        mouse-over box shows video timestamp + altitude, and colours the
        no-video portions of the trackline blue (video-covered portions red).
        """
        if not hasattr(self, "manual_map_widget") or self.navigation_file is None:
            return
        try:
            lat_src = self.navigation_file.latitude_source
            lon_src = self.navigation_file.longitude_source
            lat_df  = SensorService.load_time_value_dataframe(lat_src).sort_values("unix_time")
            lon_df  = SensorService.load_time_value_dataframe(lon_src).sort_values("unix_time")
            times   = lat_df["unix_time"].to_numpy(dtype=float)
            lats    = lat_df["value"].to_numpy(dtype=float)
            lons    = SensorService.interpolate_series(lat_df["unix_time"], lon_df["unix_time"], lon_df["value"])

            # Altitude interpolated onto the latitude time axis (for the tooltip).
            alt_values = None
            alt_src = self.navigation_file.altitude_source
            if alt_src is not None:
                try:
                    alt_df = SensorService.load_time_value_dataframe(alt_src).sort_values("unix_time")
                    alt_values = SensorService.interpolate_series(
                        lat_df["unix_time"], alt_df["unix_time"], alt_df["value"]
                    )
                except Exception:
                    alt_values = None

            mask = np.isfinite(lats) & np.isfinite(lons)
            self.manual_map_widget.set_full_trackline(
                lons[mask], lats[mask], times[mask],
                alt_values=alt_values[mask] if alt_values is not None else None,
                sensor_values=None, sensor_coloring_enabled=False,
            )
            # Video coverage windows → enables tooltip video time + no-video
            # blue colouring of the trackline.
            self.manual_map_widget.set_videos([
                (float(calendar.timegm(v.start_time.timetuple())),
                 float(calendar.timegm(v.end_time.timetuple())),
                 v.filename)
                for v in self.videos
            ])
            self.manual_map_widget.set_video_coverage_coloring(True)
        except Exception:
            pass

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

            # Downsample for display only — exports use _raster_df at full resolution.
            r_lons, r_lats, r_times, r_alt, sensor_v = self._downsample_track(
                r_lons, r_lats, r_times, r_alt, sensor_v,
                max_pts=self._MAX_DISPLAY_PTS,
            )

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
            # No raster active — draw the plain red nav trackline only.
            # Coloring is exclusively driven by Sensor Tracklines (raw instrument data).
            self.map_widget.set_nav_background_track(np.array([]), np.array([]))
            self.viz_alt_key_widget.setVisible(False)
            self.map_widget.set_full_trackline(
                nav_lons, nav_lats, nav_unix_times,
                alt_values=nav_alt_values,
                sensor_values=None,
                sensor_coloring_enabled=False,
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
