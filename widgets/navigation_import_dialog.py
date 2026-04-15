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
    QScrollArea,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from models import NavigationConfig, TimeValueSourceConfig
from sensor_service import SensorService


class _SourceSection(QWidget):
    def __init__(self, title: str, required: bool, parent=None):
        super().__init__(parent)
        self.required = required
        layout = QVBoxLayout(self)
        group = QGroupBox(title)
        layout.addWidget(group)
        form = QFormLayout(group)

        self.path_edit = QLineEdit()
        self.browse_button = QPushButton("Browse...")
        path_row = QWidget()
        path_layout = QHBoxLayout(path_row)
        path_layout.setContentsMargins(0, 0, 0, 0)
        path_layout.addWidget(self.path_edit)
        path_layout.addWidget(self.browse_button)

        self.timestamp_combo = QComboBox()
        self.value_combo = QComboBox()
        self.preview_table = QTableWidget()
        self.preview_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.preview_table.verticalHeader().setVisible(False)
        self.preview_table.setMinimumHeight(140)

        form.addRow("CSV file:", path_row)
        form.addRow("Timestamp column:", self.timestamp_combo)
        form.addRow("Value column:", self.value_combo)
        layout.addWidget(self.preview_table)
        self.browse_button.clicked.connect(self._browse_file)

    def _browse_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select CSV", filter="CSV Files (*.csv);;All Files (*)")
        if not path:
            return
        self.path_edit.setText(path)
        try:
            preview_df = SensorService.read_preview(path, nrows=5)
            columns = SensorService.read_columns(path)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to read CSV", str(exc))
            return
        self._populate_preview_table(preview_df)
        self._populate_column_combos(columns)

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
        for candidate in [
            "lat", "latitude", "lat_decimaldegrees", "lon", "longitude", "long", "lon_decimaldegrees",
            "alt", "altitude", "depth", "alt_meters", "value"
        ]:
            if candidate in lowered:
                self.value_combo.setCurrentText(lowered[candidate])
                break

    def build_result(self) -> TimeValueSourceConfig | None:
        path = self.path_edit.text().strip()
        if not path:
            if self.required:
                raise ValueError("A required navigation source is missing a CSV file.")
            return None
        timestamp_column = self.timestamp_combo.currentText().strip()
        value_column = self.value_combo.currentText().strip()
        if not timestamp_column or not value_column:
            raise ValueError("Navigation source is missing timestamp/value column selections.")
        return SensorService.build_time_value_source_config(Path(path), timestamp_column, value_column)


class NavigationImportDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Configure Navigation Sources")
        self.resize(900, 650)
        self._result_config: NavigationConfig | None = None
        self._build_ui()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        layout.addWidget(scroll)

        content = QWidget()
        scroll_layout = QVBoxLayout(content)
        scroll.setWidget(content)

        intro = QLabel(
            "Configure latitude, longitude, and altitude/depth as independent time/value sources. "
            "Each may come from a different CSV file. Each source below shows a preview of the first 5 rows."
        )
        intro.setWordWrap(True)
        scroll_layout.addWidget(intro)

        self.lat_section = _SourceSection("Latitude Source", required=True, parent=self)
        self.lon_section = _SourceSection("Longitude Source", required=True, parent=self)
        self.alt_section = _SourceSection("Altitude / Depth Source", required=False, parent=self)
        scroll_layout.addWidget(self.lat_section)
        scroll_layout.addWidget(self.lon_section)
        scroll_layout.addWidget(self.alt_section)
        scroll_layout.addStretch()

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_accept(self) -> None:
        try:
            lat_source = self.lat_section.build_result()
            lon_source = self.lon_section.build_result()
            alt_source = self.alt_section.build_result()
            if lat_source is None or lon_source is None:
                raise ValueError("Latitude and longitude sources are required.")
            self._result_config = SensorService.build_navigation_config(lat_source, lon_source, alt_source)
        except Exception as exc:
            QMessageBox.critical(self, "Failed to configure navigation", str(exc))
            return
        self.accept()

    def get_result(self) -> NavigationConfig | None:
        return self._result_config
