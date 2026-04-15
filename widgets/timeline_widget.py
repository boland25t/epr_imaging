from __future__ import annotations

from datetime import datetime

import pyqtgraph as pg
from pyqtgraph import PlotWidget
from PySide6.QtCore import Qt

from models import SensorFileConfig, VideoRecord


class TimelineWidget(PlotWidget):
    """Layered timeline showing video, navigation, and sensor coverage."""

    def __init__(self, parent=None):
        axis = pg.DateAxisItem(orientation="bottom")
        super().__init__(parent=parent, axisItems={"bottom": axis})
        self.setBackground("#1e1e1e")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "Time")
        self.getPlotItem().hideButtons()
        self.getPlotItem().setMenuEnabled(False)
        self.getPlotItem().getAxis("left").setStyle(showValues=False)
        self._text_items: list[pg.TextItem] = []

    def clear_timeline(self) -> None:
        self.clear()
        self._text_items.clear()

    def set_data(self, videos: list[VideoRecord], sensors: list[SensorFileConfig]) -> None:
        self.clear_timeline()

        rows: list[tuple[str, float]] = []
        current_y = 0.0

        # Video row
        rows.append(("Videos", current_y))
        for video in videos:
            self._draw_segment(video.start_time, video.end_time, current_y, "#4aa3ff", width=8)
        current_y += 1.0

        # Sensor / nav rows
        for sensor in sensors:
            if sensor.channels:
                for channel in sensor.channels:
                    label = channel.display_name or channel.source_column
                    rows.append((label, current_y))
                    if sensor.start_time and sensor.end_time:
                        self._draw_segment(sensor.start_time, sensor.end_time, current_y, "#61d095", width=6)
                    current_y += 1.0
            else:
                rows.append((f"NAV: {sensor.csv_path.name}", current_y))
                if sensor.start_time and sensor.end_time:
                    self._draw_segment(sensor.start_time, sensor.end_time, current_y, "#f8c555", width=6)
                current_y += 1.0

        # Compute data extent for label placement and X range
        starts = [v.start_time.timestamp() for v in videos]
        starts += [s.start_time.timestamp() for s in sensors if s.start_time]
        ends = [v.end_time.timestamp() for v in videos]
        ends += [s.end_time.timestamp() for s in sensors if s.end_time]

        if not starts:
            return

        x_min = min(starts)
        x_max = max(ends) if ends else x_min + 3600
        data_span = max(x_max - x_min, 1.0)
        label_offset = data_span * 0.02  # 2% of the data range

        for label, y in rows:
            text = pg.TextItem(text=label, color="w", anchor=(1, 0.5))
            self.addItem(text)
            self._text_items.append(text)
            text.setPos(x_min - label_offset, y)

        self.setYRange(-0.5, max(current_y - 0.5, 0.5), padding=0.15)
        # Set X range to include labels with a small right-side margin
        self.setXRange(x_min - label_offset * 6, x_max, padding=0.03)

    def _draw_segment(self, start: datetime, end: datetime, y: float, color: str, width: int = 6) -> None:
        curve = pg.PlotDataItem(
            [start.timestamp(), end.timestamp()],
            [y, y],
            pen=pg.mkPen(color=color, width=width, cap=Qt.RoundCap),
        )
        self.addItem(curve)
