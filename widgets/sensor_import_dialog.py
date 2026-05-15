from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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

import pandas as pd

from models import SensorChannel, SensorFileConfig
from sensor_service import SensorService


class SensorImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Sensor Channel")
        self.resize(900, 600)
        self._result_config: SensorFileConfig | None = None
        self._full_preview_df = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Select one CSV file, choose its timestamp column and one sensor value column, then give that channel a display name."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)

        source_group = QGroupBox("Sensor Source")
        source_form = QFormLayout(source_group)
        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        path_row = QWidget()
        path_row_layout = QHBoxLayout(path_row)
        path_row_layout.setContentsMargins(0, 0, 0, 0)
        path_row_layout.addWidget(self.path_edit)
        path_row_layout.addWidget(self.browse_button)

        self.timestamp_combo = QComboBox()
        self.value_combo = QComboBox()
        self.display_name_edit = QLineEdit()
        self.units_edit = QLineEdit()

        source_form.addRow("CSV file:", path_row)
        source_form.addRow("Timestamp column:", self.timestamp_combo)
        self.timestamp_parsed_label = QLabel("—")
        self.timestamp_parsed_label.setStyleSheet("color: gray; font-size: 10px;")
        source_form.addRow("Parsed as:", self.timestamp_parsed_label)
        source_form.addRow("Value column:", self.value_combo)
        source_form.addRow("Channel name:", self.display_name_edit)
        source_form.addRow("Units:", self.units_edit)
        layout.addWidget(source_group)

        preview_group = QGroupBox("Preview (first 5 rows)")
        preview_layout = QVBoxLayout(preview_group)
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)
        preview_layout.addWidget(self.preview_table)
        layout.addWidget(preview_group, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self.browse_button.clicked.connect(self._browse_file)
        self.timestamp_combo.currentTextChanged.connect(self._update_timestamp_preview)
        self.value_combo.currentTextChanged.connect(self._sync_default_name)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select Sensor CSV", filter="CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        self.path_edit.setText(path)
        try:
            preview_df = SensorService.read_preview(path, nrows=5)
            columns = SensorService.read_columns(path)
            self._full_preview_df = SensorService.read_preview(path, nrows=20)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read CSV", str(exc))
            return
        self._populate_preview_table(preview_df)
        self._populate_column_combos(columns)

    def _update_timestamp_preview(self, column: str) -> None:
        if not hasattr(self, "_full_preview_df") or not column or column not in self._full_preview_df.columns:
            self.timestamp_parsed_label.setText("—")
            return
        try:
            normalized = SensorService.normalize_timestamps(self._full_preview_df[column])
            valid = normalized.dropna()
            if valid.empty:
                self.timestamp_parsed_label.setText("No valid timestamps found")
                self.timestamp_parsed_label.setStyleSheet("color: red; font-size: 10px;")
                return
            t_min = pd.to_datetime(valid.min(), unit="s")
            t_max = pd.to_datetime(valid.max(), unit="s")
            raw_sample = self._full_preview_df[column].iloc[0]
            text = f"{t_min.strftime('%Y-%m-%d %H:%M:%S')}  →  {t_max.strftime('%Y-%m-%d %H:%M:%S')}  (raw sample: {raw_sample})"
            ok = t_min.year >= 2000
            self.timestamp_parsed_label.setText(text)
            self.timestamp_parsed_label.setStyleSheet(f"color: {'#2ca02c' if ok else 'red'}; font-size: 10px;")
        except Exception as exc:
            self.timestamp_parsed_label.setText(f"Parse error: {exc}")
            self.timestamp_parsed_label.setStyleSheet("color: red; font-size: 10px;")

    def _populate_preview_table(self, df) -> None:
        self.preview_table.clear()
        self.preview_table.setColumnCount(len(df.columns))
        self.preview_table.setRowCount(len(df.index))
        self.preview_table.setHorizontalHeaderLabels([str(col) for col in df.columns])
        for row_idx in range(len(df.index)):
            for col_idx, _ in enumerate(df.columns):
                value = df.iloc[row_idx, col_idx]
                item = QTableWidgetItem("" if value is None else str(value))
                item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                self.preview_table.setItem(row_idx, col_idx, item)
        self.preview_table.resizeColumnsToContents()

    def _populate_column_combos(self, columns: list[str]) -> None:
        self.timestamp_combo.clear()
        self.value_combo.clear()
        self.timestamp_combo.addItems(columns)
        self.value_combo.addItems(columns)

        lowered = {column.lower(): column for column in columns}
        for candidate in ["timestamp", "time", "datetime", "unix_time", "unix_timestamp", "epoch"]:
            if candidate in lowered:
                self.timestamp_combo.setCurrentText(lowered[candidate])
                break

        non_time_candidates = [
            c for c in columns if c.lower() not in {"timestamp", "time", "datetime", "unix_time", "unix_timestamp", "epoch"}
        ]
        if non_time_candidates:
            self.value_combo.setCurrentText(non_time_candidates[0])
            self.display_name_edit.setText(non_time_candidates[0])

    def _sync_default_name(self, value_column: str) -> None:
        if not self.display_name_edit.text().strip():
            self.display_name_edit.setText(value_column)

    def _on_accept(self) -> None:
        path_text = self.path_edit.text().strip()
        if not path_text:
            QMessageBox.warning(self, "Missing file", "Please select a sensor CSV file.")
            return

        timestamp_column = self.timestamp_combo.currentText().strip()
        value_column = self.value_combo.currentText().strip()
        display_name = self.display_name_edit.text().strip() or value_column
        units = self.units_edit.text().strip()

        if not timestamp_column:
            QMessageBox.warning(self, "Missing timestamp", "Please select a timestamp column.")
            return
        if not value_column:
            QMessageBox.warning(self, "Missing value column", "Please select a value column.")
            return

        channel = SensorChannel(
            source_column=value_column,
            display_name=display_name,
            units=units,
            use_header_name=False,
        )
        try:
            self._result_config = SensorService.build_config(
                csv_path=Path(path_text),
                timestamp_column=timestamp_column,
                channels=[channel],
            )
        except Exception as exc:
            QMessageBox.critical(self, "Failed to configure sensor", str(exc))
            return
        self.accept()

    def get_result(self) -> SensorFileConfig | None:
        return self._result_config
