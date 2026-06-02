# navigation_import_dialog.py — Modal dialog for configuring navigation sources
#
# Responsibilities:
#   1. Present three independent "source sections" — latitude, longitude, and
#      optionally altitude/depth — each backed by a _SourceSection widget.
#   2. Each section lets the user browse to a CSV or PPI file, pick the
#      timestamp column, pick the value column, and optionally enable a
#      separate date column for instruments that log date and time separately.
#   3. Show a five-row preview table for each file so the user can verify
#      they're choosing the correct columns.
#   4. Warn the user when the selected file appears to have no header row
#      (detected by checking whether column names look like numeric data values).
#   5. On OK, validate all selections and call SensorService to build and
#      return a NavigationConfig object.
#
# Public interface:
#   dialog = NavigationImportDialog(parent)
#   if dialog.exec() == QDialog.Accepted:
#       config = dialog.get_result()   # NavigationConfig | None

from __future__ import annotations

import re         # Used in _LOOKS_LIKE_VALUE to detect headerless CSV files
from pathlib import Path  # Path objects for file paths passed to SensorService

from PySide6.QtCore import Qt  # Qt.ItemIsEditable flag for read-only table cells
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from models import NavigationConfig, TimeValueSourceConfig  # Return types
from sensor_service import SensorService  # CSV preview, column listing, config builders


# ---------------------------------------------------------------------------
# Regex for detecting headerless CSV files
# ---------------------------------------------------------------------------

# A CSV file is considered "headerless" if most of its "column names" (first
# row values) look like data rather than header strings.  This regex matches:
#   — Pure numeric values (possibly with a decimal and/or leading minus sign)
#   — Date-like values (DD/MM/YYYY or similar)
#   — Time-like values (HH:MM or HH:MM.sss)
# If the majority of column names in a file match this pattern, we warn the
# user that the file may not have a proper header row.
_LOOKS_LIKE_VALUE = re.compile(
    r"^-?\d+(\.\d+)?$"           # Numeric: -1.23, 45.678
    r"|^\d{1,2}/\d{1,2}/\d{2,4}$"  # Date-like: 12/31/2026
    r"|^\d{1,2}:\d{2}(\.\d+)?$"    # Time-like: 15:36.28
)


# ===========================================================================
# _SourceSection — reusable widget for one lat/lon/alt navigation source
# ===========================================================================

class _SourceSection(QWidget):
    """A self-contained form for configuring one coordinate timeseries source.

    Used three times inside NavigationImportDialog — once for latitude, once
    for longitude, and once for altitude/depth.  All three have the same
    layout; only the group box title and the "required" flag differ.

    UI elements:
      path_edit          — text field showing the selected file path
      browse_button      — opens a file chooser dialog
      timestamp_combo    — dropdown for the timestamp column
      separate_check     — checkbox: "Use separate date and time columns"
      date_combo         — secondary dropdown for the date column (hidden by default)
      value_combo        — dropdown for the lat/lon/alt value column
      preview_table      — read-only table showing the first 5 rows of the file
      headerless_warning — yellow warning label shown for headerless files

    After the user fills in the form, build_result() validates the inputs and
    returns a TimeValueSourceConfig (or None for the optional altitude source
    if no file was selected).
    """

    def __init__(
        self,
        title: str,
        required: bool,
        value_hints: list[str] | None = None,
        parent=None,
    ):
        """Build the form UI inside a named group box.

        Args:
            title:        Text for the QGroupBox label (e.g. "Latitude Source").
            required:     If True, build_result() raises ValueError when no file
                          is selected.  If False, it returns None instead.
            value_hints:  Ordered list of lowercase column-name candidates to try
                          when auto-selecting the value column.  Falls back to the
                          built-in lat/lon/alt/depth list when None.
        """
        super().__init__(parent)
        self.required     = required
        self._value_hints = value_hints

        layout = QVBoxLayout(self)
        group  = QGroupBox(title)
        layout.addWidget(group)
        form   = QFormLayout(group)
        self._form = form  # Keep a reference so we can toggle row visibility

        # --- File path row: text field + Browse button ---
        self.path_edit    = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_button)

        # --- Column selection combos ---
        self.timestamp_combo = QComboBox()  # Which column holds the timestamp
        self.separate_check  = QCheckBox("Use separate date and time columns")
        self.date_combo      = QComboBox()  # Secondary date column (hidden until checked)
        self.value_combo     = QComboBox()  # Which column holds lat/lon/alt

        # --- Preview table: read-only, auto-resizing columns ---
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setMinimumHeight(140)

        # --- Warning label shown when the file appears headerless ---
        self.headerless_warning = QLabel(
            "Warning: this file appears to have no column headers (first row looks like data). "
            "Column names shown are first-row values. For best results, convert .ppi files using "
            "'Convert PPI → CSV' first, or use a CSV file that has a proper header row."
        )
        self.headerless_warning.setWordWrap(True)
        self.headerless_warning.setStyleSheet(
            "color: #8B4513; background: #fff3cd; border: 1px solid #ffc107; "
            "border-radius: 3px; padding: 4px; font-size: 11px;"
        )
        self.headerless_warning.hide()  # Hidden until a suspicious file is loaded

        # Lay out the form rows; the date_combo row is hidden by default.
        form.addRow("Source file:",      path_row)
        form.addRow("Timestamp column:", self.timestamp_combo)
        form.addRow("",                  self.separate_check)
        form.addRow("Date column:",      self.date_combo)
        form.setRowVisible(self.date_combo, False)
        form.addRow("Value column:",     self.value_combo)
        layout.addWidget(self.headerless_warning)
        layout.addWidget(self.preview_table)

        # Wire up interaction signals.
        self.browse_button.clicked.connect(self._browse_file)
        self.separate_check.toggled.connect(self._on_separate_toggled)

    def _on_separate_toggled(self, checked: bool) -> None:
        """Show or hide the date_combo row depending on the checkbox state."""
        self._form.setRowVisible(self.date_combo, checked)

    def _browse_file(self) -> None:
        """Open a file dialog and populate the preview table and combo boxes.

        On success:
          — Sets path_edit to the chosen path.
          — Reads the first 5 rows into preview_table.
          — Populates all three column combo boxes with the file's headers.
          — Shows the headerless warning if the file looks like it has no header row.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Source File",
            filter="CSV/PPI Files (*.csv *.ppi);;CSV Files (*.csv);;PPI Files (*.ppi);;All Files (*)"
        )
        if not path:
            return

        self.path_edit.setText(path)
        try:
            preview_df = SensorService.read_preview(path, nrows=5)
            columns    = SensorService.read_columns(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read file", str(exc))
            return

        self._populate_preview_table(preview_df)
        self._populate_column_combos(columns)

        # Count how many column names look like numeric/date/time data values.
        # If the majority do, the file probably has no header row.
        suspicious = sum(1 for c in columns if _LOOKS_LIKE_VALUE.match(str(c)))
        self.headerless_warning.setVisible(suspicious >= max(2, len(columns) // 2))

    def _populate_preview_table(self, df) -> None:
        """Fill the preview table with the first N rows of the loaded file.

        All cells are read-only; the table auto-resizes its columns to content.

        Args:
            df: A pandas DataFrame containing the preview rows to display.
        """
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setRowCount(len(df.index))
        self.preview_table.setHorizontalHeaderLabels([str(col) for col in df.columns])

        for row_idx in range(len(df.index)):
            for col_idx, _ in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                item  = QTableWidgetItem("" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.preview_table.setItem(row_idx, col_idx, item)

        self.preview_table.resizeColumnsToContents()

    def _populate_column_combos(self, columns: list[str]) -> None:
        """Populate all three column combo boxes and apply smart defaults.

        Auto-selects likely column names by checking against a priority list
        of common header names (case-insensitive).  This avoids the user having
        to manually find the timestamp column in files with many columns.

        Heuristic order for timestamp: timestamp, time, datetime, unix_time, …
        Heuristic order for date:      date, utc_date, date_utc, obs_date, …
        Heuristic order for value:     lat, latitude, lon, longitude, alt, …

        Args:
            columns: List of column header strings from the CSV file.
        """
        self.timestamp_combo.clear()
        self.date_combo.clear()
        self.value_combo.clear()
        self.timestamp_combo.addItems(columns)
        self.date_combo.addItems(columns)
        self.value_combo.addItems(columns)

        # Build a lowercase→original lookup for case-insensitive matching.
        lowered = {column.lower(): column for column in columns}

        # Auto-select the most likely timestamp column.
        for candidate in ["timestamp", "time", "datetime", "unix_time", "unix_timestamp", "epoch"]:
            if candidate in lowered:
                self.timestamp_combo.setCurrentText(lowered[candidate])
                break

        # Auto-select the most likely date column (used with separate date/time files).
        for candidate in ["date", "utc_date", "date_utc", "obs_date", "utcdate"]:
            if candidate in lowered:
                self.date_combo.setCurrentText(lowered[candidate])
                break

        # Auto-select the most likely value column.  Use caller-supplied hints first;
        # fall back to the built-in lat/lon/alt/depth list when none are provided.
        default_hints = [
            "lat", "latitude", "lat_decimaldegrees",
            "lon", "longitude", "long", "lon_decimaldegrees",
            "alt", "altitude", "depth", "alt_meters", "value",
        ]
        for candidate in (self._value_hints or default_hints):
            if candidate in lowered:
                self.value_combo.setCurrentText(lowered[candidate])
                break

    def build_result(self) -> TimeValueSourceConfig | None:
        """Validate the form and return a TimeValueSourceConfig.

        Reads all current widget values, calls SensorService to load the file
        and compute its time range, and wraps everything in a
        TimeValueSourceConfig.

        Returns:
            A fully populated TimeValueSourceConfig on success, or None if
            no file was selected and this source is optional (required=False).

        Raises:
            ValueError: If the source is required but no file was selected,
                        or if a required column selection is empty.
        """
        path = self.path_edit.text().strip()

        if not path:
            if self.required:
                raise ValueError("A required navigation source is missing a CSV file.")
            return None  # Optional source with no file selected — silently skip

        timestamp_column = self.timestamp_combo.currentText().strip()
        value_column     = self.value_combo.currentText().strip()
        date_column      = (
            self.date_combo.currentText().strip()
            if self.separate_check.isChecked()
            else None
        )

        if not timestamp_column or not value_column:
            raise ValueError("Navigation source is missing timestamp/value column selections.")

        return SensorService.build_time_value_source_config(
            Path(path), timestamp_column, value_column, date_column
        )


# ===========================================================================
# NavigationImportDialog — the top-level modal dialog
# ===========================================================================

class NavigationImportDialog(QDialog):
    """Modal dialog for configuring all three navigation coordinate sources.

    Presents three _SourceSection widgets (latitude, longitude, altitude)
    inside a scrollable area so the dialog works on smaller screens.

    Workflow:
      1. User browses to a file for each coordinate source.
      2. User selects timestamp and value columns.
      3. User clicks OK → _on_accept() validates and builds the config.
      4. Caller retrieves the result via get_result().

    Returns a NavigationConfig that wraps three TimeValueSourceConfig objects
    (or two for lat/lon only, if altitude was not configured).
    """

    def __init__(self, parent=None):
        """Initialise the dialog and build all UI widgets."""
        super().__init__(parent)
        self.setWindowTitle("Configure Navigation Sources")
        self.resize(900, 750)

        # Stores the validated result; remains None if the dialog is cancelled
        # or if validation fails.
        self._result_config: NavigationConfig | None = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the dialog layout: intro label, scrollable source sections, buttons."""
        layout = QVBoxLayout(self)

        # Wrap the source sections in a QScrollArea so the dialog is usable
        # on small-screen laptops even with six full source sections visible.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        content       = QWidget()
        scroll_layout = QVBoxLayout(content)
        scroll.setWidget(content)

        # Introductory explanation shown at the top of the dialog.
        intro = QLabel(
            "Configure navigation sources as independent time/value CSV columns. "
            "Latitude and longitude are required. All other channels are optional. "
            "Each source may come from a different file."
        )
        intro.setWordWrap(True)
        scroll_layout.addWidget(intro)

        # --- Required position sources ---
        self.lat_section = _SourceSection(
            "Latitude Source", required=True,
            value_hints=["lat", "latitude", "lat_deg", "lat_decimaldegrees"],
            parent=self,
        )
        self.lon_section = _SourceSection(
            "Longitude Source", required=True,
            value_hints=["lon", "longitude", "long", "lon_deg", "lon_decimaldegrees"],
            parent=self,
        )
        scroll_layout.addWidget(self.lat_section)
        scroll_layout.addWidget(self.lon_section)

        # --- Optional navigation channels ---
        sep = QLabel("Optional Navigation Channels")
        sep.setStyleSheet(
            "font-weight: bold; color: #555; border-top: 1px solid #ccc; "
            "margin-top: 8px; padding-top: 6px;"
        )
        scroll_layout.addWidget(sep)

        self.alt_section = _SourceSection(
            "Altitude Source (above substrate, acoustic altimeter)",
            required=False,
            value_hints=["alt", "altitude", "alt_m", "alt_meters", "height", "height_m"],
            parent=self,
        )
        self.depth_section = _SourceSection(
            "Depth Source (below water surface, pressure sensor)",
            required=False,
            value_hints=["depth", "depth_m", "depth_msw", "depth_meters", "pressure", "dep"],
            parent=self,
        )
        self.pitch_section = _SourceSection(
            "Pitch Source (degrees, positive = nose up)",
            required=False,
            value_hints=["pitch", "pitch_deg", "pitch_degrees", "pitch_rad", "ptch"],
            parent=self,
        )
        self.roll_section = _SourceSection(
            "Roll Source (degrees, positive = starboard down)",
            required=False,
            value_hints=["roll", "roll_deg", "roll_degrees", "roll_rad"],
            parent=self,
        )
        scroll_layout.addWidget(self.alt_section)
        scroll_layout.addWidget(self.depth_section)
        scroll_layout.addWidget(self.pitch_section)
        scroll_layout.addWidget(self.roll_section)
        scroll_layout.addStretch()

        # Standard OK / Cancel button bar at the bottom.
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        """Validate all source sections and build the NavigationConfig.

        Called when the user clicks OK.  On validation failure, shows an error
        message box and stays open (does NOT call self.accept()).  On success,
        stores the result in _result_config and accepts the dialog.
        """
        try:
            lat_source   = self.lat_section.build_result()
            lon_source   = self.lon_section.build_result()
            alt_source   = self.alt_section.build_result()
            depth_source = self.depth_section.build_result()
            pitch_source = self.pitch_section.build_result()
            roll_source  = self.roll_section.build_result()

            if lat_source is None or lon_source is None:
                raise ValueError("Latitude and longitude sources are required.")

            self._result_config = SensorService.build_navigation_config(
                lat_source, lon_source,
                altitude_source=alt_source,
                depth_source=depth_source,
                pitch_source=pitch_source,
                roll_source=roll_source,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to configure navigation", str(exc))
            return  # Stay open so the user can correct the problem

        self.accept()  # Close the dialog with Accepted result

    def get_result(self) -> NavigationConfig | None:
        """Return the validated NavigationConfig produced by the last OK click.

        Returns:
            The NavigationConfig, or None if the dialog was cancelled or
            validation failed on the last OK attempt.
        """
        return self._result_config
