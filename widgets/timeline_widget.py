# timeline_widget.py — Horizontal timeline showing data coverage for all sources
#
# Responsibilities:
#   1. Display a read-only horizontal bar chart where each row represents one
#      data source (videos, navigation, sensor channels).
#   2. Each bar spans the wall-clock coverage of that source, coloured by type:
#       — Videos:  blue   (#4aa3ff)
#       — Sensors: green  (#61d095)
#       — Nav:     yellow (#f8c555)
#   3. Row labels are placed to the left of the earliest bar so they don't
#      overlap the data.
#   4. The X axis uses pyqtgraph's DateAxisItem to show human-readable
#      date/time tick marks automatically.
#
# This widget is intentionally display-only.  It does not emit signals or
# respond to user interaction; it is refreshed by MainWindow whenever the
# loaded data changes.

from __future__ import annotations

from datetime import datetime  # Used in _draw_segment() for the start/end parameters

import pyqtgraph as pg          # Pyqtgraph: PlotWidget, PlotDataItem, TextItem, DateAxisItem
from pyqtgraph import PlotWidget # PlotWidget is the base class we extend
from PySide6.QtCore import Qt    # Qt.RoundCap — rounds the line endpoints for a polished look

from models import SensorFileConfig, VideoRecord  # Data models describing each source's time range


class TimelineWidget(PlotWidget):
    """Layered horizontal timeline showing video, navigation, and sensor coverage.

    Inherits from pyqtgraph's PlotWidget so it slots directly into a Qt layout
    without any extra wrapper.  Each call to set_data() rebuilds the display
    from scratch: it clears all items, draws the coverage bars, and places
    row labels.

    Visual design:
      - Dark background (#1e1e1e) to contrast with the coloured bars.
      - Subtle grid lines (alpha=0.25) to aid time-reading without clutter.
      - Y axis tick marks hidden (showValues=False) because rows are labelled
        by TextItem rather than numeric tick values.

    Usage:
        timeline = TimelineWidget()
        timeline.set_data(videos, sensors)   # call after loading any source
    """

    def __init__(self, parent=None):
        """Initialise the plot axes, visual style, and label tracking list.

        A DateAxisItem is used for the bottom axis so pyqtgraph formats tick
        labels as calendar dates/times rather than raw Unix float values.
        """
        # Attach a date-aware bottom axis before calling super().__init__()
        # so it replaces the default numeric axis from the very start.
        axis = pg.DateAxisItem(orientation="bottom")
        super().__init__(parent=parent, axisItems={"bottom": axis})

        # Visual style: dark background, light grid, no navigation buttons.
        self.setBackground("#1e1e1e")
        self.showGrid(x=True, y=True, alpha=0.25)
        self.setLabel("bottom", "Time")

        # Hide the interactive toolbar that pyqtgraph shows in the plot corner.
        self.getPlotItem().hideButtons()

        # Disable the right-click context menu; this widget is display-only.
        self.getPlotItem().setMenuEnabled(False)

        # Hide Y-axis numeric tick values; row identity is shown via TextItem labels.
        self.getPlotItem().getAxis("left").setStyle(showValues=False)

        # Keep a reference to every TextItem we create so clear_timeline() can
        # remove them.  pyqtgraph's clear() removes PlotDataItems but not TextItems
        # added with addItem(), so we must remove them manually.
        self._text_items: list[pg.TextItem] = []

    def clear_timeline(self) -> None:
        """Remove all plotted items and label text items from the widget.

        Called at the start of each set_data() call to ensure a clean slate
        before drawing new content.  Two-step process: pyqtgraph's built-in
        clear() handles all curve items; the TextItem list is purged separately.
        """
        self.clear()
        self._text_items.clear()

    def set_data(self, videos: list[VideoRecord], sensors: list[SensorFileConfig]) -> None:
        """Rebuild the entire timeline display from the provided data sources.

        Iterates over videos and sensor channels in order, assigning each a
        horizontal Y position and drawing its coverage bar.  After drawing,
        labels are placed just to the left of the earliest bar start so they
        appear as row names rather than axis tick labels.

        Args:
            videos:  List of VideoRecord objects, each with start_time and end_time.
            sensors: List of SensorFileConfig objects; each may have multiple
                     named channels or be a nav-only source (no channels).
        """
        self.clear_timeline()

        # rows accumulates (label_string, y_position) for TextItem placement.
        rows: list[tuple[str, float]] = []
        current_y = 0.0

        # --- Videos row: one blue bar per video file ---
        rows.append(("Videos", current_y))
        for video in videos:
            self._draw_segment(video.start_time, video.end_time, current_y, "#4aa3ff", width=8)
        current_y += 1.0

        # --- Sensor / nav rows: one row per channel or per nav-only source ---
        for sensor in sensors:
            if sensor.channels:
                # Sensor file with named data channels (temperature, salinity, etc.).
                # Each channel gets its own row so they can have different time ranges
                # in theory (in practice they share the same file, so same range).
                for channel in sensor.channels:
                    label = channel.display_name or channel.source_column
                    rows.append((label, current_y))
                    if sensor.start_time and sensor.end_time:
                        self._draw_segment(sensor.start_time, sensor.end_time, current_y, "#61d095", width=6)
                    current_y += 1.0
            else:
                # Navigation-only source (lat/lon/alt CSV without sensor channels).
                rows.append((f"NAV: {sensor.csv_path.name}", current_y))
                if sensor.start_time and sensor.end_time:
                    self._draw_segment(sensor.start_time, sensor.end_time, current_y, "#f8c555", width=6)
                current_y += 1.0

        # --- Compute the data extent for label placement and X-range setting ---
        # Collect all start/end timestamps from every source.
        starts  = [v.start_time.timestamp() for v in videos]
        starts += [s.start_time.timestamp() for s in sensors if s.start_time]
        ends    = [v.end_time.timestamp() for v in videos]
        ends   += [s.end_time.timestamp() for s in sensors if s.end_time]

        if not starts:
            return

        x_min      = min(starts)
        x_max      = max(ends) if ends else x_min + 3600
        data_span  = max(x_max - x_min, 1.0)

        # Place labels 2% of the data span to the left of the first bar so
        # they don't overlap the bars even at different zoom levels.
        label_offset = data_span * 0.02

        for label, y in rows:
            text = pg.TextItem(text=label, color="w", anchor=(1, 0.5))
            self.addItem(text)
            self._text_items.append(text)
            text.setPos(x_min - label_offset, y)

        # Set Y range to show all rows with a small padding above and below.
        self.setYRange(-0.5, max(current_y - 0.5, 0.5), padding=0.15)

        # Set X range wide enough to include label text with a small right margin.
        self.setXRange(x_min - label_offset * 6, x_max, padding=0.03)

    def _draw_segment(self, start: datetime, end: datetime, y: float, color: str, width: int = 6) -> None:
        """Draw one horizontal coverage bar for a single data source.

        Creates a two-point horizontal line from start.timestamp() to
        end.timestamp() at height y.  RoundCap is used on the line ends
        for a polished pill-like appearance.

        Args:
            start: Wall-clock start of the coverage window.
            end:   Wall-clock end of the coverage window.
            y:     Vertical position (row index) in plot coordinates.
            color: Hex colour string for the bar (e.g. "#4aa3ff").
            width: Pen width in pixels.
        """
        curve = pg.PlotDataItem(
            [start.timestamp(), end.timestamp()],
            [y, y],
            pen=pg.mkPen(color=color, width=width, cap=Qt.RoundCap),
        )
        self.addItem(curve)
