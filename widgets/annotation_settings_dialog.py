# annotation_settings_dialog.py — Dialog for configuring frame annotation content and style
#
# Allows the user to choose:
#   • Which fields to annotate (filename, nav channels, sensor channels)
#   • Where the text block is placed (corner)
#   • Font size, text color, and optional background rectangle

from __future__ import annotations

from typing import TYPE_CHECKING

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QCheckBox,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from models import AnnotationConfig

if TYPE_CHECKING:
    from models import NavigationConfig, SensorFileConfig


# Human-readable labels for the built-in field identifiers
_BUILTIN_LABELS: dict[str, str] = {
    "filename":  "Frame filename",
    "timestamp": "Timestamp (UTC)",
    "lat":       "Latitude",
    "lon":       "Longitude",
    "alt":       "Altitude (above substrate)",
    "depth":     "Depth (below surface)",
    "heading":   "Heading (degrees true)",
    "pitch":     "Pitch",
    "roll":      "Roll",
}

_POSITIONS = [
    ("top_left",     "Top Left"),
    ("top_right",    "Top Right"),
    ("bottom_left",  "Bottom Left"),
    ("bottom_right", "Bottom Right"),
]


class AnnotationSettingsDialog(QDialog):
    """Dialog for configuring frame annotation content and appearance.

    Shows all available annotation fields based on the current navigation
    and sensor configuration.  Built-in fields (filename, timestamp, nav
    channels) are always listed; sensor channel fields are added from the
    configured sensor files.  The user can check any combination.

    Usage:
        dlg = AnnotationSettingsDialog(
            parent,
            current_config=self.annotation_config,
            navigation_file=self.navigation_file,
            sensor_files=self.sensor_files,
        )
        if dlg.exec() == QDialog.Accepted:
            self.annotation_config = dlg.get_result()
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        current_config: AnnotationConfig | None = None,
        navigation_file: "NavigationConfig | None" = None,
        sensor_files: "list[SensorFileConfig] | None" = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Annotation Settings")
        self.setMinimumWidth(480)

        self._config = current_config or AnnotationConfig()
        self._color = QColor(*self._config.color_rgb)
        self._build_available_items(navigation_file, sensor_files or [])
        self._build_ui()

    # -----------------------------------------------------------------------
    # Build the ordered list of (identifier, label) pairs available to check
    # -----------------------------------------------------------------------

    def _build_available_items(
        self,
        nav: "NavigationConfig | None",
        sensor_files: "list[SensorFileConfig]",
    ) -> None:
        """Produce self._available: list of (identifier, display_label) pairs."""
        self._available: list[tuple[str, str]] = [
            ("filename",  _BUILTIN_LABELS["filename"]),
            ("timestamp", _BUILTIN_LABELS["timestamp"]),
            ("lat",       _BUILTIN_LABELS["lat"]),
            ("lon",       _BUILTIN_LABELS["lon"]),
            ("alt",       _BUILTIN_LABELS["alt"]),
        ]
        if nav is not None:
            if nav.depth_source is not None:
                self._available.append(("depth", _BUILTIN_LABELS["depth"]))
            if nav.pitch_source is not None:
                self._available.append(("pitch", _BUILTIN_LABELS["pitch"]))
            if nav.roll_source is not None:
                self._available.append(("roll", _BUILTIN_LABELS["roll"]))

        seen: set[str] = {ident for ident, _ in self._available}
        for sf in sensor_files:
            for ch in sf.channels:
                name = ch.display_name or ch.source_column
                if name not in seen:
                    label = f"{name} ({ch.units})" if ch.units else name
                    self._available.append((name, label))
                    seen.add(name)

    # -----------------------------------------------------------------------
    # UI construction
    # -----------------------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # --- Content panel ---
        content_group = QGroupBox("Annotation Content  (check to include, drag to reorder)")
        content_layout = QVBoxLayout(content_group)

        self._item_list = QListWidget()
        self._item_list.setDragDropMode(QListWidget.InternalMove)
        self._item_list.setDefaultDropAction(Qt.MoveAction)
        self._item_list.setToolTip(
            "Check fields to include in the annotation.  Drag rows to change order."
        )

        enabled_set = set(self._config.enabled_items)
        # Add items in the order they appear in enabled_items first (preserves user ordering),
        # then any remaining available items unchecked.
        ordered: list[tuple[str, str]] = []
        ident_to_label = dict(self._available)
        for ident in self._config.enabled_items:
            label = ident_to_label.get(ident, ident)
            ordered.append((ident, label))
        for ident, label in self._available:
            if ident not in enabled_set:
                ordered.append((ident, label))

        for ident, label in ordered:
            item = QListWidgetItem(label)
            item.setData(Qt.UserRole, ident)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if ident in enabled_set else Qt.Unchecked)
            self._item_list.addItem(item)

        self._item_list.setMinimumHeight(180)
        content_layout.addWidget(self._item_list)
        layout.addWidget(content_group)

        # --- Appearance panel ---
        appear_group = QGroupBox("Appearance")
        appear_form = QFormLayout(appear_group)
        appear_form.setLabelAlignment(Qt.AlignRight)

        # Position
        self._position_combo = QComboBox()
        for value, label in _POSITIONS:
            self._position_combo.addItem(label, userData=value)
        current_pos = next(
            (i for i, (v, _) in enumerate(_POSITIONS) if v == self._config.position), 0
        )
        self._position_combo.setCurrentIndex(current_pos)
        appear_form.addRow("Position:", self._position_combo)

        # Font size
        self._font_spin = QDoubleSpinBox()
        self._font_spin.setRange(0.3, 4.0)
        self._font_spin.setSingleStep(0.1)
        self._font_spin.setDecimals(2)
        self._font_spin.setValue(self._config.font_scale)
        self._font_spin.setToolTip("Font scale factor (0.3 = very small, 1.0 = medium, 2.0 = large).")
        appear_form.addRow("Font size:", self._font_spin)

        # Text color
        color_row = QWidget()
        color_layout = QHBoxLayout(color_row)
        color_layout.setContentsMargins(0, 0, 0, 0)
        self._color_btn = QPushButton()
        self._color_btn.setFixedWidth(60)
        self._color_btn.setFixedHeight(24)
        self._update_color_button()
        self._color_btn.clicked.connect(self._pick_color)
        self._color_label = QLabel(self._hex_color())
        self._color_label.setStyleSheet("font-family: monospace; font-size: 11px;")
        color_layout.addWidget(self._color_btn)
        color_layout.addWidget(self._color_label)
        color_layout.addStretch()
        appear_form.addRow("Text color:", color_row)

        # Background toggle
        self._bg_check = QCheckBox("Draw dark background box behind text")
        self._bg_check.setChecked(self._config.use_background)
        self._bg_check.toggled.connect(self._on_bg_toggled)
        appear_form.addRow("", self._bg_check)

        # Background opacity
        bg_opacity_row = QWidget()
        bg_opacity_layout = QHBoxLayout(bg_opacity_row)
        bg_opacity_layout.setContentsMargins(0, 0, 0, 0)
        self._bg_slider = QSlider(Qt.Horizontal)
        self._bg_slider.setRange(0, 100)
        self._bg_slider.setValue(int(self._config.bg_opacity * 100))
        self._bg_slider.setFixedWidth(120)
        self._bg_pct_label = QLabel(f"{int(self._config.bg_opacity * 100)} %")
        self._bg_pct_label.setFixedWidth(36)
        self._bg_slider.valueChanged.connect(
            lambda v: self._bg_pct_label.setText(f"{v} %")
        )
        bg_opacity_layout.addWidget(self._bg_slider)
        bg_opacity_layout.addWidget(self._bg_pct_label)
        bg_opacity_layout.addStretch()
        appear_form.addRow("Background opacity:", bg_opacity_row)

        self._on_bg_toggled(self._config.use_background)
        layout.addWidget(appear_group)

        # --- Hint ---
        hint = QLabel(
            "Tip: a dark background box (55–70 % opacity) is the most reliable way "
            "to keep text readable over any frame content."
        )
        hint.setWordWrap(True)
        hint.setStyleSheet("font-size: 10px; color: #555; font-style: italic;")
        layout.addWidget(hint)

        # --- Buttons ---
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    def _on_bg_toggled(self, checked: bool) -> None:
        self._bg_slider.setEnabled(checked)
        self._bg_pct_label.setEnabled(checked)

    def _hex_color(self) -> str:
        return self._color.name().upper()

    def _update_color_button(self) -> None:
        r, g, b = self._color.red(), self._color.green(), self._color.blue()
        luma = 0.299 * r + 0.587 * g + 0.114 * b
        text_color = "black" if luma > 140 else "white"
        self._color_btn.setStyleSheet(
            f"background-color: {self._color.name()}; color: {text_color}; "
            f"border: 1px solid #888; border-radius: 3px;"
        )
        self._color_btn.setText(self._hex_color())

    def _pick_color(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Choose Text Color")
        if chosen.isValid():
            self._color = chosen
            self._update_color_button()
            self._color_label.setText(self._hex_color())

    # -----------------------------------------------------------------------
    # Result
    # -----------------------------------------------------------------------

    def get_result(self) -> AnnotationConfig:
        """Return the configured AnnotationConfig after the dialog is accepted."""
        enabled: list[str] = []
        for i in range(self._item_list.count()):
            item = self._item_list.item(i)
            if item.checkState() == Qt.Checked:
                enabled.append(item.data(Qt.UserRole))

        position = self._position_combo.currentData()

        return AnnotationConfig(
            enabled_items=enabled,
            position=position,
            font_scale=self._font_spin.value(),
            color_rgb=(self._color.red(), self._color.green(), self._color.blue()),
            use_background=self._bg_check.isChecked(),
            bg_opacity=self._bg_slider.value() / 100.0,
        )
