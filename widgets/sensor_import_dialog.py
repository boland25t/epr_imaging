# sensor_import_dialog.py — Modal dialog for adding one sensor channel from a CSV
#
# Responsibilities:
#   1. Let the user browse to a CSV or PPI file and inspect its first 5 rows.
#   2. Let the user pick the timestamp column and a single value column to extract.
#   3. Show a live timestamp parse preview so the user can verify the timestamp
#      column is being interpreted correctly before committing.
#   4. Let the user supply a human-readable display name and optional units for
#      the channel (used as the column header in master.csv and raster filenames).
#   5. On OK, validate the selections and return a SensorFileConfig object that
#      the pipeline can use to load and interpolate this sensor channel.
#
# Note: this dialog adds exactly one sensor channel per invocation.  To add
# multiple channels from the same file, the user opens the dialog multiple times.
#
# Public interface:
#   dialog = SensorImportDialog(parent)
#   if dialog.exec() == QDialog.Accepted:
#       config = dialog.get_result()   # SensorFileConfig | None

from __future__ import annotations

from pathlib import Path  # Path objects for the csv_path in SensorFileConfig

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
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

import pandas as pd  # pd.to_datetime() used in the live timestamp parse preview

from models import SensorChannel, SensorFileConfig  # Return types
from sensor_service import SensorService             # CSV read helpers and config builder


class SensorImportDialog(QDialog):
    """Modal dialog for importing one sensor data channel from a CSV or PPI file.

    Workflow:
      1. User browses to a CSV/PPI file.  The first 5 rows are shown in a preview
         table, and all column headers are loaded into the combo boxes.
      2. User selects the timestamp column.  A live parse preview immediately
         shows the interpreted UTC start and end times so timestamp mismatches
         are caught before running the pipeline.
      3. User selects the value column (e.g. "Temperature_C") and optionally
         provides a channel display name and units.
      4. User clicks OK → validation runs → SensorFileConfig is built and stored.
      5. Caller retrieves the result via get_result().

    The dialog adds exactly one SensorChannel per invocation.  Channels from
    the same file can only be added by opening the dialog again.
    """

    def __init__(self, parent=None):
        """Initialise dialog state and build the UI."""
        super().__init__(parent)
        self.setWindowTitle("Add Sensor Channel")
        self.resize(900, 600)

        # Validated result; None until the user clicks OK successfully.
        self._result_config: SensorFileConfig | None = None

        # A wider preview DataFrame (up to 20 rows) used for the live timestamp
        # parse preview so the displayed range covers more of the data.
        self._full_preview_df = None

        self._build_ui()

    def _build_ui(self) -> None:
        """Construct the full dialog layout with form fields, preview table, and buttons."""
        layout = QVBoxLayout(self)

        # Brief instruction label at the top.
        intro = QLabel(
            "Select a CSV or PPI file, choose its timestamp column and one sensor value column, "
            "then give that channel a display name."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        # --- Sensor source group box ---
        source_group = QGroupBox("Sensor Source")
        source_form  = QFormLayout(source_group)

        # File path row: text field + Browse button.
        self.path_edit     = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        path_row        = QWidget()
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.addWidget(self.path_edit)
        path_row_layout.addWidget(self.browse_button)

        # Column selection combos.
        self.timestamp_combo = QComboBox()  # Which column is the timestamp
        self.separate_check  = QCheckBox("Use separate date and time columns")
        self.date_combo      = QComboBox()  # Optional separate date column
        self.value_combo     = QComboBox()  # Which column is the sensor value

        # Display name and units fields — used for master.csv column headers
        # and GeoTIFF filenames.
        self.display_name_edit = QLineEdit()
        self.units_edit        = QLineEdit()

        # Live parse preview label — shows the UTC time range parsed from the
        # current timestamp column selection, coloured green if plausible (year ≥ 2000)
        # or red if the parse result looks wrong (too old or failed).
        self.timestamp_parsed_label = QLabel("—")
        self.timestamp_parsed_label.setStyleSheet("color: gray; font-size: 10px;")

        # Lay out form rows.  The date_combo row is hidden until separate_check is ticked.
        source_form.addRow("Source file:",      path_row)
        source_form.addRow("Timestamp column:", self.timestamp_combo)
        source_form.addRow("Parsed as:",        self.timestamp_parsed_label)
        source_form.addRow("",                  self.separate_check)
        source_form.addRow("Date column:",      self.date_combo)
        source_form.setRowVisible(self.date_combo, False)
        source_form.addRow("Value column:",     self.value_combo)
        source_form.addRow("Channel name:",     self.display_name_edit)
        source_form.addRow("Units:",            self.units_edit)
        self._source_form = source_form  # Keep reference for row visibility toggling

        layout.addWidget(source_group)

        # --- Preview table group box ---
        preview_group  = QGroupBox("Preview (first 5 rows)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group, stretch=1)

        # Standard OK / Cancel button bar at the bottom.
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        # --- Signal connections ---
        self.browse_button.clicked.connect(self._browse_file)

        # Refresh the live timestamp parse preview whenever the user changes
        # the timestamp combo or the separate date combo.
        self.timestamp_combo.currentTextChanged.connect(self._update_timestamp_preview)
        self.date_combo.currentTextChanged.connect(
            lambda _: self._update_timestamp_preview(self.timestamp_combo.currentText())
        )
        self.separate_check.toggled.connect(self._on_separate_toggled)

        # Auto-fill the display name field when the user picks a value column,
        # but only if they haven't already typed something.
        self.value_combo.currentTextChanged.connect(self._sync_default_name)

    def _browse_file(self) -> None:
        """Open a file chooser dialog and load the selected CSV/PPI file.

        On success:
          — Updates path_edit with the selected path.
          — Reads the first 5 rows into preview_table.
          — Reads up to 20 rows into _full_preview_df for timestamp preview.
          — Populates all column combo boxes.
        """
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Sensor File",
            filter="CSV/PPI Files (*.csv *.ppi);;CSV Files (*.csv);;PPI Files (*.ppi);;All Files (*)"
        )
        if not path:
            return

        self.path_edit.setText(path)
        try:
            preview_df            = SensorService.read_preview(path, nrows=5)
            columns               = SensorService.read_columns(path)
            self._full_preview_df = SensorService.read_preview(path, nrows=20)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read file", str(exc))
            return

        self._populate_preview_table(preview_df)
        self._populate_column_combos(columns)

    def _on_separate_toggled(self, checked: bool) -> None:
        """Show or hide the date_combo row and refresh the timestamp parse preview."""
        self._source_form.setRowVisible(self.date_combo, checked)
        self._update_timestamp_preview(self.timestamp_combo.currentText())

    def _update_timestamp_preview(self, column: str = "") -> None:
        """Refresh the live timestamp parse preview label.

        Runs SensorService.normalize_timestamps() on the current timestamp
        (and optionally date) column from the preview DataFrame to show the
        user the earliest and latest UTC times that would be parsed from the file.

        The label is coloured:
          — Green: timestamps parsed successfully and the year is ≥ 2000
                   (plausible for modern survey data).
          — Red:   parse failed or resulted in a suspiciously old year.
          — Grey:  no file loaded yet.

        Args:
            column: The timestamp column name to preview.  Defaults to the
                    current combo box text if empty.
        """
        column = column or self.timestamp_combo.currentText()
        df     = getattr(self, "_full_preview_df", None)

        if df is None or not column or column not in df.columns:
            self.timestamp_parsed_label.setText("—")
            return

        try:
            if self.separate_check.isChecked():
                # Combine date and time columns into a single string for parsing.
                date_col = self.date_combo.currentText()
                if date_col and date_col in df.columns:
                    combined = (
                        df[date_col].astype(str).str.strip()
                        + " "
                        + df[column].astype(str).str.strip()
                    )
                else:
                    combined = df[column]
            else:
                combined = df[column]

            # normalize_timestamps() handles numeric Unix, ISO strings, and
            # decimal-minute formats; returns a float Series of Unix seconds.
            normalized = SensorService.normalize_timestamps(combined)
            valid      = normalized.dropna()

            if valid.empty:
                self.timestamp_parsed_label.setText("No valid timestamps found")
                self.timestamp_parsed_label.setStyleSheet("color: red; font-size: 10px;")
                return

            # Convert Unix seconds back to UTC datetime for display.
            t_min      = pd.to_datetime(valid.min(), unit="s")
            t_max      = pd.to_datetime(valid.max(), unit="s")
            raw_sample = df[column].iloc[0]  # Show the raw first value for context

            text = (
                f"{t_min.strftime('%Y-%m-%d %H:%M:%S')}  →  "
                f"{t_max.strftime('%Y-%m-%d %H:%M:%S')}  (raw sample: {raw_sample})"
            )
            ok = t_min.year >= 2000  # Year < 2000 strongly suggests a bad parse
            self.timestamp_parsed_label.setText(text)
            self.timestamp_parsed_label.setStyleSheet(
                f"color: {'#2ca02c' if ok else 'red'}; font-size: 10px;"
            )

        except Exception as exc:
            self.timestamp_parsed_label.setText(f"Parse error: {exc}")
            self.timestamp_parsed_label.setStyleSheet("color: red; font-size: 10px;")

    def _populate_preview_table(self, df) -> None:
        """Fill the preview table with the first N rows of the loaded file.

        All cells are read-only.  Column widths are auto-sized to their content.

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
        """Populate all column combo boxes and apply smart default selections.

        Uses case-insensitive keyword matching to auto-select the most likely
        timestamp, date, and value columns so the user rarely needs to change
        the defaults.

        For the value combo, the first non-time column is chosen as the default
        since it's likely to be the sensor reading the user cares about.

        Args:
            columns: List of column header strings from the loaded file.
        """
        self.timestamp_combo.clear()
        self.date_combo.clear()
        self.value_combo.clear()
        self.timestamp_combo.addItems(columns)
        self.date_combo.addItems(columns)
        self.value_combo.addItems(columns)

        lowered = {column.lower(): column for column in columns}

        # Auto-select the most likely timestamp column.
        for candidate in ["timestamp", "time", "datetime", "unix_time", "unix_timestamp", "epoch"]:
            if candidate in lowered:
                self.timestamp_combo.setCurrentText(lowered[candidate])
                break

        # Auto-select the most likely date column (for separate date/time files).
        for candidate in ["date", "utc_date", "date_utc", "obs_date", "utcdate"]:
            if candidate in lowered:
                self.date_combo.setCurrentText(lowered[candidate])
                break

        # For the value column, skip all known time/date column names and pick
        # the first remaining column — this is likely the sensor reading.
        time_names         = {"timestamp", "time", "datetime", "unix_time", "unix_timestamp", "epoch"}
        non_time_candidates = [c for c in columns if c.lower() not in time_names]
        if non_time_candidates:
            self.value_combo.setCurrentText(non_time_candidates[0])
            # Pre-fill the display name with the column header as a sensible default.
            self.display_name_edit.setText(non_time_candidates[0])

    def _sync_default_name(self, value_column: str) -> None:
        """Auto-fill the display name field when a value column is selected.

        Only fills in the display name if the user hasn't already typed anything,
        so it doesn't overwrite a carefully chosen custom name.

        Args:
            value_column: The newly selected value column name.
        """
        if not self.display_name_edit.text().strip():
            self.display_name_edit.setText(value_column)

    def _on_accept(self) -> None:
        """Validate user selections and build the SensorFileConfig.

        On validation failure, shows a warning dialog and stays open.
        On success, stores the config in _result_config and accepts the dialog.
        """
        path_text = self.path_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Missing file", "Please select a sensor file.")
            return

        timestamp_column = self.timestamp_combo.currentText().strip()
        value_column     = self.value_combo.currentText().strip()

        # Fall back to the column name if the user left the display name blank.
        display_name = self.display_name_edit.text().strip() or value_column
        units        = self.units_edit.text().strip()

        if not timestamp_column:
            QMessageBox.warning(self, "Missing timestamp", "Please select a timestamp column.")
            return
        if not value_column:
            QMessageBox.warning(self, "Missing value column", "Please select a value column.")
            return

        # Only supply date_column when the separate-date checkbox is checked.
        date_column = (
            self.date_combo.currentText().strip()
            if self.separate_check.isChecked()
            else None
        )

        # Build the SensorChannel descriptor.  use_header_name=False because the
        # user explicitly provided a display name (even if it happens to match the
        # column header).
        channel = SensorChannel(
            source_column=value_column,
            display_name=display_name,
            units=units,
            use_header_name=False,
        )

        try:
            # SensorService.build_config() opens the file, reads the time range,
            # and validates that the requested columns exist.
            self._result_config = SensorService.build_config(
                csv_path=Path(path_text),
                timestamp_column=timestamp_column,
                channels=[channel],
                date_column=date_column,
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to configure sensor", str(exc))
            return

        self.accept()  # Close with Accepted result

    def get_result(self) -> SensorFileConfig | None:
        """Return the validated SensorFileConfig produced by the last OK click.

        Returns:
            The SensorFileConfig on success, or None if the dialog was cancelled
            or validation failed.
        """
        return self._result_config
