# map_widget.py — Interactive scatter map for frame locations and GPS tracklines
#
# Responsibilities:
#   1. Display the full GPS navigation path (red line) for the dive/survey.
#   2. Overlay extracted-frame locations as coloured scatter dots.
#   3. Support interactive box selection (drag a rectangle to select dots).
#   4. Support two-click track-interval picking: the user clicks two points on
#      the GPS trackline to define a SelectedTimeRange for the pipeline.
#   5. Show a hover tooltip with GPS time and corresponding video offset when
#      the cursor is near the trackline, or a frame label when near a dot.
#   6. Support equirectangular projection (metres from centroid) in addition
#      to raw lat/lon display.
#
# Coordinate conventions:
#   - Raw storage: WGS 84 decimal degrees (lat/lon).
#   - "metres" display mode: equirectangular projection centred at the mean
#     lat/lon of all data.  X = (lon − ref_lon) × cos(ref_lat) × 111_319.5 m/°
#     Y = (lat − ref_lat) × 111_319.5 m/°.
#
# Signal protocol:
#   selection_changed(list[int])      — emitted when the box selection changes
#   segment_created(float, float)     — (start_unix, end_unix) from two-click pick
#   pick_point_placed(float)          — unix_time when the first click is placed

from __future__ import annotations

import math                    # math.cos/radians used in equirectangular projection
from datetime import datetime  # datetime.utcfromtimestamp() for tooltip GPS time display
from typing import Optional    # Optional[X] used for nullable type hints

import numpy as np             # Array math for distance calculations in hover/click handlers
import pyqtgraph as pg         # Pyqtgraph: PlotWidget, ScatterPlotItem, TextItem, etc.
from PySide6.QtCore import Qt, Signal, QRectF, QPointF  # Qt signals and geometry types
from PySide6.QtWidgets import QWidget, QVBoxLayout      # Base widget and layout


# ---------------------------------------------------------------------------
# Colour constants
# ---------------------------------------------------------------------------

# Colour palette for segment assignments (applied to frame scatter dots when
# the user has assigned frames to named segments via the Selection panel).
# Cycles when there are more segments than colours.
SEGMENT_COLORS = [
    "#2196f3", "#4caf50", "#ff9800", "#9c27b0",
    "#00bcd4", "#8b4513", "#f06292", "#1a7030", "#ff6b00", "#6a0dad",
]

# Default appearance for unselected and selected frame dots.
_UNSELECTED_FILL = "#1565c0"   # Dark blue — default dot fill
_UNSELECTED_PEN  = "#000000"   # Black outline for unselected dots
_SELECTED_FILL   = "#ffcc00"   # Yellow fill for selected/highlighted dots
_SELECTED_PEN    = "#995500"   # Dark gold outline for selected dots
_TRACKLINE       = "#e63946"   # Red — GPS navigation trackline (video-covered)
_NO_VIDEO_TRACKLINE = "#1565c0"  # Blue — trackline where no video covers the time

# Number of discrete colour buckets used when drawing a sensor-coloured trackline.
# Fewer buckets = fewer PlotCurveItem objects per redraw = faster rendering.
# Paired with _MAX_DISPLAY_PTS in MainWindow; reduce here if 5 000 pts still lags.
_COLOR_LEVELS: int = 64


# ---------------------------------------------------------------------------
# Haversine distance helper
# ---------------------------------------------------------------------------

def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return the great-circle distance in kilometres between two WGS 84 points.

    Uses the haversine formula, which is accurate for small-to-medium distances
    and doesn't require a full geodesic solver.  The result is clipped to
    prevent a domain error in asin() due to floating-point rounding near
    antipodal points (a → 1.0 can happen).

    Args:
        lat1, lon1: Decimal-degree coordinates of the first point.
        lat2, lon2: Decimal-degree coordinates of the second point.

    Returns:
        Distance in kilometres (always ≥ 0).
    """
    R = 6371.0  # Earth's mean radius in km
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi    = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    # min(1.0, a) clamps floating-point overflow before the sqrt/asin call.
    return 2 * R * math.asin(math.sqrt(min(1.0, a)))


# ---------------------------------------------------------------------------
# _SelectableViewBox — extends pyqtgraph ViewBox with rubber-band selection
# ---------------------------------------------------------------------------

class _SelectableViewBox(pg.ViewBox):
    """A pyqtgraph ViewBox that adds rubber-band (drag) selection capability.

    When select_mode is active:
      - Left-button drag: emits selection_rect_changed(QRectF) continuously
        so the MapWidget can draw a live preview rectangle.
      - Left-button release: emits selection_committed(QRectF) with the final
        rectangle so MapWidget can compute which dots fall inside it.
      - Left-button click (no drag): emits selection_committed(None) to clear
        the selection.
      - Normal pan/zoom mouse events are suppressed during select_mode
        (setMouseEnabled(False)) to avoid accidental view changes while
        drawing a selection rectangle.

    When select_mode is False, all events are delegated to the normal
    ViewBox handler (pan, zoom, etc.).
    """

    # Emitted continuously while the user is dragging a selection rectangle.
    # Payload is the current QRectF in view (data) coordinates.
    selection_rect_changed = Signal(object)

    # Emitted when the user releases the mouse button.
    # Payload is the final QRectF, or None if the user just clicked without dragging.
    selection_committed = Signal(object)

    def __init__(self, *args, **kwargs):
        """Initialise the viewbox with selection mode disabled."""
        super().__init__(*args, **kwargs)
        self._select_mode = False  # Selection mode off by default

    def set_select_mode(self, enabled: bool) -> None:
        """Enable or disable rubber-band selection mode.

        Disabling normal pan/zoom while in selection mode prevents the view
        from scrolling when the user starts a drag near the edge.
        """
        self._select_mode = enabled
        self.setMouseEnabled(x=not enabled, y=not enabled)

    def mouseDragEvent(self, ev, axis=None) -> None:
        """Handle mouse drag: emit rectangle signals when in select mode."""
        if self._select_mode and ev.button() == Qt.LeftButton:
            ev.accept()

            # Convert scene coordinates to view (data) coordinates so the
            # rectangle is in the same space as the scatter dot positions.
            start = self.mapSceneToView(ev.buttonDownScenePos())
            cur   = self.mapSceneToView(ev.scenePos())

            # .normalized() ensures the QRectF has positive width and height
            # regardless of which direction the user dragged.
            rect = QRectF(
                QPointF(start.x(), start.y()),
                QPointF(cur.x(),   cur.y()),
            ).normalized()

            if ev.isFinish():
                self.selection_committed.emit(rect)
            else:
                self.selection_rect_changed.emit(rect)
        else:
            super().mouseDragEvent(ev, axis)

    def mouseClickEvent(self, ev) -> None:
        """Handle single click: in select mode a click clears the selection."""
        if self._select_mode and ev.button() == Qt.LeftButton:
            self.selection_committed.emit(None)
            ev.accept()
        else:
            super().mouseClickEvent(ev)


# ---------------------------------------------------------------------------
# MapWidget — the main public widget
# ---------------------------------------------------------------------------

class MapWidget(QWidget):
    """QGIS-style scatter map for extracted frame locations and GPS tracks.

    The widget wraps a pyqtgraph PlotWidget inside a plain QWidget so
    it can be embedded in any Qt layout.  It manages five overlapping
    visual layers, all living in the same PlotWidget:

      1. Trackline   — red curve showing the full GPS navigation path.
      2. T1 marker   — orange star at the first click of a two-click pick.
      3. Frame dots  — blue (or coloured) scatter dots for extracted frames.
      4. Drag rect   — dashed red rectangle preview during box selection.
      5. Tooltip     — yellow floating text box near the cursor.

    Public methods (called by MainWindow):
      set_full_trackline()  — provide the GPS track with timestamps
      set_videos()          — provide video coverage for tooltip hover
      set_pick_mode()       — enter/exit two-click interval picking
      load_data()           — load frame positions for scatter plot
      set_segments()        — colour-code dots by segment assignment
      set_display_mode()    — switch between lat/lon and metres display
      set_select_mode()     — enable/disable rubber-band box selection
      set_selection()       — programmatically highlight a set of dots
      clear_selection()     — deselect all dots
      fit_view()            — auto-range to show all data

    Emitted signals (received by MainWindow):
      selection_changed(list[int])      — indices of selected frame rows
      segment_created(float, float)     — (start_unix, end_unix) from two-click pick
      pick_point_placed(float)          — unix_time of the first pick click
    """

    # Emitted when the box selection changes (rubber-band drag or explicit set).
    # Payload: list of integer row indices that fall inside the selected rectangle.
    selection_changed = Signal(list)

    # Emitted after the second click of a two-click trackline interval pick.
    # Payload: (start_unix, end_unix) as floats, always start ≤ end.
    segment_created = Signal(float, float)

    # Emitted after the first click of a two-click pick, so MainWindow can
    # update its status label ("Click the end point now").
    pick_point_placed = Signal(float)

    def __init__(self, parent: Optional[QWidget] = None):
        """Build the widget: create the PlotWidget, configure visual style,
        initialise all state variables, and wire up mouse event signals.
        """
        super().__init__(parent)

        # A zero-margin VBoxLayout so the PlotWidget fills the entire widget area.
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Use our custom ViewBox so it can emit rubber-band selection signals.
        self._vb   = _SelectableViewBox()
        self._plot = pg.PlotWidget(viewBox=self._vb)

        # White background and dark-ink axes for a QGIS-like map appearance.
        self._plot.setBackground("w")
        self._plot.showGrid(x=True, y=True, alpha=0.4)
        self._plot.getPlotItem().setMenuEnabled(False)
        self._plot.getPlotItem().getAxis("bottom").setTextPen("k")
        self._plot.getPlotItem().getAxis("left").setTextPen("k")
        self._plot.getPlotItem().getAxis("bottom").setPen(pg.mkPen("k"))
        self._plot.getPlotItem().getAxis("left").setPen(pg.mkPen("k"))
        layout.addWidget(self._plot)

        # -----------------------------------------------------------------------
        # State: frame scatter data (raw geographic coordinates)
        # -----------------------------------------------------------------------

        # WGS 84 longitude and latitude arrays for each extracted frame dot.
        self._lons:   np.ndarray = np.array([])
        self._lats:   np.ndarray = np.array([])

        # One hover-label string per frame dot (e.g. "frame_00001.jpg | lat lon | …").
        self._labels: list[str]  = []

        # -----------------------------------------------------------------------
        # State: full GPS navigation trackline
        # -----------------------------------------------------------------------

        # Dense GPS track arrays (may be much longer than the frame arrays).
        self._nav_lons:       np.ndarray = np.array([])
        self._nav_lats:       np.ndarray = np.array([])

        # Unix timestamp for each nav point; used to display GPS time in the
        # hover tooltip and to map click positions to unix times for two-click picking.
        # Must be the same length as _nav_lons/_nav_lats.
        self._nav_unix_times: np.ndarray = np.array([])

        # Projected (display) coordinates for the nav track; recomputed in _redraw()
        # and cached here so _on_mouse_moved() can query them without re-projecting.
        self._nav_px: np.ndarray = np.array([])
        self._nav_py: np.ndarray = np.array([])

        # -----------------------------------------------------------------------
        # State: video coverage for tooltip
        # -----------------------------------------------------------------------

        # List of (start_unix, end_unix, filename) tuples — one per video.
        # Used by _on_mouse_moved() to determine which video (if any) covers the
        # hovered GPS timestamp and compute the corresponding video-relative offset.
        self._videos: list[tuple[float, float, str]] = []

        # When True (and _videos is set), the plain trackline is drawn split by
        # video coverage: red where a video covers the timestamp, blue where not.
        # Only applies when sensor colouring is OFF.
        self._color_by_video_coverage: bool = False

        # -----------------------------------------------------------------------
        # State: two-click pick mode
        # -----------------------------------------------------------------------

        # When True, left-clicks on the trackline are interpreted as interval
        # endpoints rather than normal selection events.
        self._pick_mode: bool = False

        # Unix timestamp of the first click in pick mode.  None until the first
        # click is placed; reset to None after the second click completes the pick.
        self._pick_t1: Optional[float] = None

        # -----------------------------------------------------------------------
        # State: projected display coordinates for frames
        # -----------------------------------------------------------------------

        # Cached projected frame positions (lat/lon or metres) updated in _redraw().
        self._px: np.ndarray = np.array([])
        self._py: np.ndarray = np.array([])

        # -----------------------------------------------------------------------
        # State: equirectangular projection reference point
        # -----------------------------------------------------------------------

        # The centroid of all data (nav + frames) in decimal degrees.
        # Used as the origin of the equirectangular "metres" projection.
        self._ref_lat: float = 0.0
        self._ref_lon: float = 0.0

        # Current display mode: "latlon" keeps raw decimal degrees on both axes;
        # "meters" converts to approximate local metre offsets from the centroid.
        self._display_mode: str = "latlon"

        # -----------------------------------------------------------------------
        # State: selection and segment coloring
        # -----------------------------------------------------------------------

        # List of integer frame indices that are currently selected (yellow dots).
        self._selection: list[int] = []

        # Maps frame_index → segment_index for colour assignment.
        self._segment_assignments: dict[int, int] = {}

        # Ordered list of colour strings for each segment (indexed by segment_index).
        self._segment_colors: list[str] = []

        # -----------------------------------------------------------------------
        # State: trackline sensor colour mapping
        # -----------------------------------------------------------------------

        # Altitude array aligned with _nav_lons/_nav_lats.  Always stored when
        # available so the hover tooltip can show altitude regardless of which
        # channel is driving the colour.
        self._nav_alt_values: Optional[np.ndarray] = None

        # The values that actually DRIVE trackline colouring — may be altitude or
        # any other interpolated sensor channel.  None → solid red trackline.
        self._sensor_coloring_values: Optional[np.ndarray] = None

        # Human-readable label for the active sensor channel (e.g. "Depth (m)").
        # Shown in the hover tooltip alongside GPS time and coordinates.
        self._sensor_label: str = ""

        # Explicit colour-scale clamps supplied by the sidebar spinboxes.
        # When both are set these override the data min/max so the scale is stable
        # across zooms and reloads.
        self._sensor_min_clamp: Optional[float] = None
        self._sensor_max_clamp: Optional[float] = None

        # Whether the colour scale is logarithmic (True) or linear (False).
        self._sensor_log_scale: bool = False

        # Whether to apply sensor colouring to the trackline.  Sensor data may be
        # present even when False so tooltip values are always available.
        self._sensor_coloring_enabled: bool = False

        # Segment time-range highlights: list of (start_unix, end_unix, color_hex).
        # Drawn as thicker colored bands on top of the trackline in _redraw().
        self._trackline_segment_ranges: list[tuple[float, float, str]] = []

        # History mode overlay: list of (start_unix, end_unix, color_hex, tooltip_text).
        # Blue bands drawn over the trackline showing previously-processed regions.
        # When non-empty and the tooltip is shown near a history band, the tooltip
        # text overrides the default GPS-time tooltip.
        self._history_ranges: list[tuple[float, float, str, str]] = []

        # Secondary thin gray nav background track shown behind the primary
        # colored trackline when sensor raster mode is active.  Empty means hidden.
        self._nav_bg_lons: np.ndarray = np.array([])
        self._nav_bg_lats: np.ndarray = np.array([])

        # Width of the trackline in pixels (1–10).
        self._nav_trackline_width: float = 2.0

        # -----------------------------------------------------------------------
        # Pyqtgraph item references (kept so we can check / update them)
        # -----------------------------------------------------------------------

        # The red GPS trackline curve.
        self._trackline_item: Optional[pg.PlotCurveItem]  = None

        # The frame scatter plot.
        self._scatter:        Optional[pg.ScatterPlotItem] = None

        # The yellow hover tooltip text box.
        self._tooltip:        Optional[pg.TextItem]        = None

        # The dashed red rubber-band selection rectangle preview.
        self._drag_rect_line: Optional[pg.PlotCurveItem]  = None

        # -----------------------------------------------------------------------
        # Signal connections
        # -----------------------------------------------------------------------

        # ViewBox rubber-band selection → our handlers.
        self._vb.selection_rect_changed.connect(self._on_drag_rect)
        self._vb.selection_committed.connect(self._on_selection_committed)

        # Scene-level mouse events → hover tooltip and click-pick handlers.
        self._plot.scene().sigMouseMoved.connect(self._on_mouse_moved)
        self._plot.scene().sigMouseClicked.connect(self._on_mouse_clicked)

    # -----------------------------------------------------------------------
    # Public API
    # -----------------------------------------------------------------------

    def set_full_trackline(
        self,
        lons: np.ndarray,
        lats: np.ndarray,
        unix_times: Optional[np.ndarray] = None,
        alt_values: Optional[np.ndarray] = None,
        sensor_values: Optional[np.ndarray] = None,
        sensor_coloring_enabled: bool = False,
        sensor_log_scale: bool = False,
        sensor_label: str = "",
    ) -> None:
        """Provide the complete GPS navigation path and optional sensor colour data.

        alt_values is always stored for tooltip display.  sensor_values drives the
        trackline colour gradient when sensor_coloring_enabled is True; this may be
        altitude or any other interpolated channel aligned with lons/lats.

        Args:
            lons:                   Longitude array (decimal degrees).
            lats:                   Latitude array, same length as lons.
            unix_times:             Unix timestamp array aligned with lons/lats.
            alt_values:             Altitude array for tooltip (always stored).
            sensor_values:          Values that drive trackline colour (any channel).
            sensor_coloring_enabled: Whether to apply the colour gradient.
            sensor_log_scale:       Use logarithmic colour scale when True.
            sensor_label:           Display name shown in the hover tooltip.
        """
        self._nav_lons                = lons
        self._nav_lats                = lats
        self._nav_unix_times          = unix_times if unix_times is not None else np.array([])
        self._nav_alt_values          = alt_values
        self._sensor_coloring_values  = sensor_values
        self._sensor_coloring_enabled = sensor_coloring_enabled
        self._sensor_log_scale        = sensor_log_scale
        self._sensor_label            = sensor_label

        # Recompute the projection centroid (may have changed if new data was added).
        self._update_reference()
        self._redraw()

    def set_videos(self, videos: list[tuple[float, float, str]]) -> None:
        """Provide video coverage metadata so the hover tooltip can show video offsets.

        Args:
            videos: List of (start_unix, end_unix, filename) tuples.
                    The tooltip searches this list for a video whose range covers
                    the hovered GPS timestamp and shows the within-video offset.
        """
        self._videos = videos

    def set_pick_mode(self, enabled: bool) -> None:
        """Enable or disable two-click trackline interval picking.

        When enabled:
          — The first left-click on the trackline records T1 and emits
            pick_point_placed(unix_t).
          — The second click records T2, emits segment_created(min(T1,T2),
            max(T1,T2)), and automatically exits pick mode.

        Disabling pick mode also clears any pending T1 so the state is clean
        for the next time pick mode is enabled.

        Args:
            enabled: True to enter pick mode, False to exit.
        """
        self._pick_mode = enabled
        if not enabled:
            self._pick_t1 = None
        self._redraw()  # Redraw to show/hide the T1 star marker

    def load_data(self, lons: np.ndarray, lats: np.ndarray, labels: list[str]) -> None:
        """Set the extracted-frame scatter plot data.

        Replaces any previously loaded frame data.  Clears selection and
        segment assignments because they referred to the old data's indices.

        Args:
            lons:   Longitude array for each frame.
            lats:   Latitude array for each frame.
            labels: One hover-label string per frame (shown in the tooltip when
                    the cursor is near a frame dot).
        """
        self._lons   = lons
        self._lats   = lats
        self._labels = labels
        self._selection = []
        self._segment_assignments = {}
        self._segment_colors = []
        self._update_reference()
        self._redraw()

    def set_segments(self, segments: list[dict]) -> None:
        """Apply segment-based colour coding to the frame scatter dots.

        Each segment is a dict with keys: name (str), indices (list[int]),
        and optionally color (str).  Frame dots belonging to a segment are
        drawn in the segment's colour rather than the default blue.

        Args:
            segments: List of segment dicts.  If a segment doesn't specify a
                      colour, colours from SEGMENT_COLORS are assigned in order.
        """
        self._segment_assignments = {}
        self._segment_colors = []
        for seg_idx, seg in enumerate(segments):
            color = seg.get("color", SEGMENT_COLORS[seg_idx % len(SEGMENT_COLORS)])
            self._segment_colors.append(color)
            for idx in seg["indices"]:
                self._segment_assignments[idx] = seg_idx
        self._redraw()

    def set_video_coverage_coloring(self, enabled: bool) -> None:
        """Colour the plain trackline by video coverage (red=covered, blue=not).

        Requires set_videos() to have been called.  Has no effect while sensor
        colouring is active (that takes priority).
        """
        self._color_by_video_coverage = bool(enabled)
        self._redraw()

    def set_display_mode(self, mode: str) -> None:
        """Switch between raw lat/lon display and equirectangular metres projection.

        Args:
            mode: "latlon" for raw decimal-degree axes; "meters" to project all
                  coordinates to approximate local-East/North metre offsets.
        """
        if mode == self._display_mode:
            return
        self._display_mode = mode
        self._redraw()

    def set_select_mode(self, enabled: bool) -> None:
        """Enable or disable rubber-band box selection in the ViewBox.

        Delegates to _SelectableViewBox.set_select_mode() which also
        disables normal pan/zoom while selection is active.
        """
        self._vb.set_select_mode(enabled)

    def set_selection(self, indices: list[int]) -> None:
        """Programmatically select a set of frame dots by index.

        Useful when the user interacts with the results table and the map
        should highlight the corresponding rows.
        """
        self._selection = list(indices)
        self._redraw()

    def clear_selection(self) -> None:
        """Deselect all frame dots and notify listeners via selection_changed."""
        self._selection = []
        self._redraw()
        self.selection_changed.emit([])

    def fit_view(self) -> None:
        """Auto-range the view to show all plotted data with default padding."""
        self._plot.autoRange()

    def set_trackline_width(self, width: float) -> None:
        """Set the trackline pen width in pixels and redraw.

        Args:
            width: Pen width in pixels.  Clamped to [1, 10].
        """
        self._nav_trackline_width = max(1.0, min(10.0, float(width)))
        self._redraw()

    def set_trackline_sensor_range(self, v_min: float, v_max: float) -> None:
        """Override the automatic colour-scale range and redraw.

        After this call the colour mapping uses [v_min, v_max] instead of the
        data min/max, regardless of the actual sensor values in the track.

        Args:
            v_min: Value that maps to red (low / cool end of scale).
            v_max: Value that maps to blue (high / warm end of scale).
        """
        self._sensor_min_clamp = v_min
        self._sensor_max_clamp = v_max
        self._redraw()

    def set_trackline_log_scale(self, log_scale: bool) -> None:
        """Switch the colour scale between linear and logarithmic and redraw."""
        self._sensor_log_scale = log_scale
        self._redraw()

    def set_nav_background_track(
        self, lons: np.ndarray, lats: np.ndarray
    ) -> None:
        """Set a thin gray background nav track shown behind the primary trackline.

        Used when sensor raster mode is active so the full nav coverage remains
        visible even though the primary colored trackline runs at sensor temporal
        resolution.  Pass empty arrays to hide the background track.

        Args:
            lons: Longitude array (decimal degrees).
            lats: Latitude array, same length as lons.
        """
        self._nav_bg_lons = lons
        self._nav_bg_lats = lats
        self._redraw()

    def set_trackline_segment_ranges(
        self, ranges: list[tuple[float, float, str]]
    ) -> None:
        """Overlay colored bands on the trackline for each segment's time span.

        Each range draws a thicker line over the nav trackline points whose
        unix_times fall within [start_unix, end_unix], using the segment's color.

        Args:
            ranges: List of (start_unix, end_unix, color_hex) tuples.
        """
        self._trackline_segment_ranges = list(ranges)
        self._redraw()

    def set_history_ranges(
        self, ranges: list[tuple[float, float, str, str]]
    ) -> None:
        """Overlay history-mode bands on the trackline for previously-processed segments.

        Each band is drawn as a thick colored line over nav points whose unix_times
        fall within [start_unix, end_unix].  In history mode the hover tooltip for
        points within a band shows the band's tooltip_text instead of the normal
        GPS-time string.

        Args:
            ranges: List of (start_unix, end_unix, color_hex, tooltip_text) tuples.
                    Pass an empty list to clear history mode.
        """
        self._history_ranges = list(ranges)
        self._redraw()

    # -----------------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------------

    def _update_reference(self) -> None:
        """Compute the lat/lon centroid of all loaded data.

        The centroid is used as the origin of the equirectangular projection
        in "meters" mode.  It must be recomputed whenever new data is loaded
        to keep the projection centred on the current survey area.
        """
        all_lats, all_lons = [], []

        # Include the nav track if available.
        if len(self._nav_lats) > 0:
            all_lats.append(self._nav_lats)
            all_lons.append(self._nav_lons)

        # Include the frame positions if available.
        if len(self._lats) > 0:
            all_lats.append(self._lats)
            all_lons.append(self._lons)

        if all_lats:
            self._ref_lat = float(np.concatenate(all_lats).mean())
            self._ref_lon = float(np.concatenate(all_lons).mean())

    def _project(self, lons: np.ndarray, lats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Project geographic coordinates to display coordinates.

        In "latlon" mode, returns lons and lats unchanged.  In "meters" mode,
        applies an equirectangular (flat-Earth) projection centred at
        (ref_lon, ref_lat):

            x_m = (lon − ref_lon) × cos(ref_lat_rad) × 111_319.5
            y_m = (lat − ref_lat) × 111_319.5

        The factor 111_319.5 converts degrees to metres along a great circle
        (1° ≈ 111.32 km ≈ 111_319.5 m).  cos(ref_lat) scales the longitude
        axis to match the latitude axis at the reference latitude, accounting
        for meridian convergence.

        Args:
            lons: Longitude array (decimal degrees).
            lats: Latitude array (decimal degrees).

        Returns:
            A 2-tuple (x, y) in either decimal degrees or metres.
        """
        if self._display_mode == "latlon" or len(lons) == 0:
            return lons, lats
        cos_lat = math.cos(math.radians(self._ref_lat))
        x_m = (lons - self._ref_lon) * cos_lat * 111_319.5
        y_m = (lats - self._ref_lat) * 111_319.5
        return x_m, y_m

    def _update_axis_labels(self) -> None:
        """Update the axis labels to match the current display mode."""
        if self._display_mode == "meters":
            self._plot.setLabel("bottom", "Easting (m)",  **{"color": "k"})
            self._plot.setLabel("left",   "Northing (m)", **{"color": "k"})
        else:
            self._plot.setLabel("bottom", "Longitude", **{"color": "k"})
            self._plot.setLabel("left",   "Latitude",  **{"color": "k"})

    @staticmethod
    def _altitude_to_rgb(fraction: float) -> tuple[int, int, int]:
        """Map a normalised [0, 1] fraction to an RGB colour on a blue→red spectrum.

        Colour stops (linearly interpolated between adjacent pairs):
          0.00 → deep blue  (  0,   0, 210)
          0.25 → cyan       (  0, 210, 210)
          0.50 → green      (  0, 200,   0)
          0.75 → yellow     (220, 220,   0)
          1.00 → red        (220,   0,   0)

        Args:
            fraction: Normalised altitude value, clamped to [0, 1].

        Returns:
            An (R, G, B) integer tuple with each channel in [0, 255].
        """
        fraction = max(0.0, min(1.0, fraction))
        # Low altitude (0.0) → red; high altitude (1.0) → blue.
        stops = [
            (0.00, (220,   0,   0)),
            (0.25, (220, 220,   0)),
            (0.50, (  0, 200,   0)),
            (0.75, (  0, 210, 210)),
            (1.00, (  0,   0, 210)),
        ]
        for i in range(len(stops) - 1):
            t0, c0 = stops[i]
            t1, c1 = stops[i + 1]
            if t0 <= fraction <= t1:
                alpha = (fraction - t0) / (t1 - t0)
                r = int(c0[0] + alpha * (c1[0] - c0[0]))
                g = int(c0[1] + alpha * (c1[1] - c0[1]))
                b = int(c0[2] + alpha * (c1[2] - c0[2]))
                return (r, g, b)
        return (220, 0, 0)  # fallback: top of range

    def _draw_colored_trackline(
        self,
        tx: np.ndarray,
        ty: np.ndarray,
        sensor_v: np.ndarray,
        log_scale: bool = False,
    ) -> None:
        """Draw the trackline as coloured segments whose colour encodes a sensor value.

        Quantises the sensor values into N_LEVELS colour buckets then groups
        consecutive nav points that share the same bucket into runs.  Each run
        is drawn as a single PlotCurveItem, extended by one point on each side
        so adjacent runs join seamlessly.

        The number of PlotCurveItems created equals the number of colour-level
        transitions in the track, which for slowly-varying profiles is much
        smaller than the number of nav points.

        Args:
            tx:        Projected X coordinates for every nav point.
            ty:        Projected Y coordinates for every nav point.
            sensor_v:  Sensor value for every nav point (same length as tx/ty).
            log_scale: If True, apply logarithmic normalisation instead of linear.
        """
        N_LEVELS = _COLOR_LEVELS

        if self._sensor_min_clamp is not None and self._sensor_max_clamp is not None:
            v_min, v_max = self._sensor_min_clamp, self._sensor_max_clamp
        else:
            finite = sensor_v[np.isfinite(sensor_v)]
            v_min = float(finite.min()) if len(finite) > 0 else 0.0
            v_max = float(finite.max()) if len(finite) > 0 else 1.0

        if abs(v_max - v_min) < 1e-9:
            v_max = v_min + 1.0

        if log_scale and v_min > 0 and v_max > 0:
            pos_min = max(v_min, 1e-10)
            pos_max = max(v_max, pos_min * (1 + 1e-9))
            clipped = np.clip(sensor_v, pos_min, pos_max)
            fracs   = np.log10(clipped / pos_min) / np.log10(pos_max / pos_min)
        else:
            fracs = (sensor_v - v_min) / (v_max - v_min)

        fracs  = np.nan_to_num(np.clip(fracs, 0.0, 1.0), nan=0.0)
        levels = np.clip((fracs * (N_LEVELS - 1)).astype(int), 0, N_LEVELS - 1)

        N = len(tx)
        if N < 2:
            return

        # Walk through the levels array, collecting run boundaries.
        run_start = 0
        first_item = True
        for i in range(1, N + 1):
            if i == N or levels[i] != levels[run_start]:
                # Extend the run by one point on each side for seamless colour joins.
                s = max(0, run_start - 1)
                e = min(N, i + 1)

                level = int(levels[run_start])
                frac  = level / (N_LEVELS - 1)
                r, g, b = self._altitude_to_rgb(frac)

                item = pg.PlotCurveItem(
                    x=tx[s:e], y=ty[s:e],
                    pen=pg.mkPen((r, g, b), width=self._nav_trackline_width),
                    connect="all",
                    antialias=True,
                )
                self._plot.addItem(item)

                # Store the first segment as the canonical trackline reference.
                if first_item:
                    self._trackline_item = item
                    first_item = False

                if i < N:
                    run_start = i

    def _draw_coverage_trackline(self, tx: np.ndarray, ty: np.ndarray) -> None:
        """Draw the trackline split by video coverage.

        Each nav point is classified as covered (its unix time falls inside any
        video window) or not.  Consecutive points sharing a class are grouped
        into runs and drawn as one PlotCurveItem — red where covered, blue where
        not — extended by one point on each side for seamless joins.

        Args:
            tx: Projected X coordinates for every nav point.
            ty: Projected Y coordinates for every nav point.
        """
        times = self._nav_unix_times
        N = len(tx)
        if N < 2:
            return

        # Boolean coverage mask via the video windows.
        covered = np.zeros(N, dtype=bool)
        for v_start, v_end, _name in self._videos:
            covered |= (times >= v_start) & (times <= v_end)

        first_item = True
        run_start = 0
        for i in range(1, N + 1):
            if i == N or covered[i] != covered[run_start]:
                s = max(0, run_start - 1)
                e = min(N, i + 1)
                color = _TRACKLINE if covered[run_start] else _NO_VIDEO_TRACKLINE
                item = pg.PlotCurveItem(
                    x=tx[s:e], y=ty[s:e],
                    pen=pg.mkPen(color=color, width=self._nav_trackline_width),
                    connect="all",
                    antialias=False,
                )
                self._plot.addItem(item)
                if first_item:
                    self._trackline_item = item
                    first_item = False
                if i < N:
                    run_start = i

    # -----------------------------------------------------------------------
    # Drawing
    # -----------------------------------------------------------------------

    def _redraw(self) -> None:
        """Rebuild all plotted items from current state.

        Called after any state change that affects the visual representation
        (new data, selection change, mode switch, pick mode toggle, etc.).
        Clears the plot first, then re-adds items in the correct Z-order:
          1. Trackline (below everything else)
          2. T1 star marker (above trackline, below scatter)
          3. Frame scatter dots
          4. Drag rectangle (above dots, below tooltip)
          5. Tooltip (always on top, Z = 100)
        """
        # Clear all existing items; reset cached item references.
        self._plot.clear()
        self._tooltip        = None
        self._drag_rect_line = None
        self._trackline_item = None
        self._scatter        = None

        self._update_axis_labels()

        has_nav    = len(self._nav_lons) > 0
        has_frames = len(self._lons) > 0

        # Nothing to draw: show a placeholder text hint.
        if not has_nav and not has_frames:
            ph = pg.TextItem(
                "No data — load an interp.csv or scan the output directory.",
                color=(80, 80, 80), anchor=(0.5, 0.5),
            )
            ph.setPos(0, 0)
            self._plot.addItem(ph)
            self._px = self._py = np.array([])
            return

        # --- Layer 0: Nav background track (thin gray; shown in raster mode) ---
        if len(self._nav_bg_lons) >= 2:
            bx, by = self._project(self._nav_bg_lons, self._nav_bg_lats)
            bg_item = pg.PlotCurveItem(
                x=bx, y=by,
                pen=pg.mkPen(color=(180, 180, 180), width=1.0),
                connect="all",
                antialias=False,
            )
            bg_item.setZValue(-1)
            self._plot.addItem(bg_item)

        # --- Layer 1: Trackline ---
        if has_nav:
            # Project the full nav track and cache projected coords for hover math.
            tx, ty = self._project(self._nav_lons, self._nav_lats)
            self._nav_px, self._nav_py = tx, ty
        else:
            # Fall back to frame positions as the track if no nav was loaded.
            tx, ty = self._project(self._lons, self._lats)
            self._nav_px = self._nav_py = np.array([])

        # Draw the trackline: coloured by sensor values when enabled.
        # _nav_alt_values may still be set (for tooltip) even when not colouring.
        use_sensor_color = (
            has_nav
            and self._sensor_coloring_enabled
            and self._sensor_coloring_values is not None
            and len(self._sensor_coloring_values) == len(tx)
        )
        use_coverage_color = (
            has_nav
            and not use_sensor_color
            and self._color_by_video_coverage
            and bool(self._videos)
            and len(self._nav_unix_times) == len(tx) > 0
        )
        if use_sensor_color:
            self._draw_colored_trackline(
                tx, ty, self._sensor_coloring_values,
                log_scale=self._sensor_log_scale,
            )
        elif use_coverage_color:
            self._draw_coverage_trackline(tx, ty)
        else:
            self._trackline_item = pg.PlotCurveItem(
                x=tx, y=ty,
                pen=pg.mkPen(color=_TRACKLINE, width=self._nav_trackline_width),
                connect="all",
                antialias=False,
            )
            self._plot.addItem(self._trackline_item)

        # --- Layer 1b: Segment trackline highlights ---
        # Use searchsorted (nav times are sorted) so frame timestamps that fall
        # *between* GPS samples still produce a visible highlight.  Expand by one
        # nav point on each side for a clean join with the surrounding trackline.
        if (has_nav
                and self._trackline_segment_ranges
                and len(self._nav_unix_times) == len(self._nav_px) > 0):
            for start_u, end_u, color in self._trackline_segment_ranges:
                i0 = max(0, int(np.searchsorted(self._nav_unix_times, start_u, side="left")) - 1)
                i1 = min(len(self._nav_unix_times),
                         int(np.searchsorted(self._nav_unix_times, end_u, side="right")) + 1)
                if i1 - i0 >= 2:
                    item = pg.PlotCurveItem(
                        x=self._nav_px[i0:i1],
                        y=self._nav_py[i0:i1],
                        pen=pg.mkPen(color=color, width=self._nav_trackline_width + 4),
                        connect="all",
                        antialias=False,
                    )
                    item.setZValue(1)
                    self._plot.addItem(item)

        # --- Layer 1c: History / pending-job overlay ---
        # Colors: yellow (#ffd600) = pending job, blue (#2196f3) = manual history,
        # green (#4caf50) = threshold history.  When a blue and green range overlap,
        # a dashed blue line is drawn on top of the green to suggest the mix.
        if (has_nav
                and self._history_ranges
                and len(self._nav_unix_times) == len(self._nav_px) > 0):
            # Collect blue and green time ranges for overlap detection
            blue_ranges  = [(s, e) for s, e, c, _ in self._history_ranges if c == "#2196f3"]
            green_ranges = [(s, e) for s, e, c, _ in self._history_ranges if c == "#4caf50"]

            for start_u, end_u, color, _tooltip in self._history_ranges:
                i0 = max(0, int(np.searchsorted(self._nav_unix_times, start_u, side="left")) - 1)
                i1 = min(len(self._nav_unix_times),
                         int(np.searchsorted(self._nav_unix_times, end_u, side="right")) + 1)
                if i1 - i0 < 2:
                    continue
                w = self._nav_trackline_width + 6
                item = pg.PlotCurveItem(
                    x=self._nav_px[i0:i1],
                    y=self._nav_py[i0:i1],
                    pen=pg.mkPen(color=color, width=w),
                    connect="all",
                    antialias=False,
                )
                item.setZValue(2)
                self._plot.addItem(item)

                # If this is a green range that overlaps a blue range, draw a
                # dashed blue line on top to produce a visual blue-green mix.
                if color == "#4caf50":
                    overlaps = any(
                        bs < end_u and be > start_u
                        for bs, be in blue_ranges
                    )
                    if overlaps:
                        dash_item = pg.PlotCurveItem(
                            x=self._nav_px[i0:i1],
                            y=self._nav_py[i0:i1],
                            pen=pg.mkPen(
                                color="#2196f3", width=w,
                                style=Qt.DashLine,
                            ),
                            connect="all",
                            antialias=False,
                        )
                        dash_item.setZValue(3)
                        self._plot.addItem(dash_item)

        # --- Layer 2: T1 pick marker (orange star at first-click position) ---
        if (self._pick_mode
                and self._pick_t1 is not None
                and len(self._nav_unix_times) == len(self._nav_px) > 0):
            # Find the nav point closest in time to the stored T1 timestamp.
            idx = int(np.argmin(np.abs(self._nav_unix_times - self._pick_t1)))
            mx, my = float(self._nav_px[idx]), float(self._nav_py[idx])
            t1_marker = pg.ScatterPlotItem(
                x=[mx], y=[my], size=16,
                brush=pg.mkBrush("#ff5722"),   # Deep orange
                pen=pg.mkPen("k", width=2),
                symbol="star",
            )
            self._plot.addItem(t1_marker)

        # --- Layer 3: Frame scatter dots ---
        if has_frames:
            self._px, self._py = self._project(self._lons, self._lats)
            n = len(self._lons)

            spots = []
            for i in range(n):
                if i in self._selection:
                    # Selected dot: yellow with gold outline, slightly larger.
                    spots.append({
                        "pos":   (self._px[i], self._py[i]),
                        "size":  10,
                        "brush": pg.mkBrush(_SELECTED_FILL),
                        "pen":   pg.mkPen(_SELECTED_PEN, width=1),
                    })
                elif i in self._segment_assignments:
                    # Segment-assigned dot: use the segment's colour.
                    seg_idx = self._segment_assignments[i]
                    color   = self._segment_colors[seg_idx] if seg_idx < len(self._segment_colors) else _UNSELECTED_FILL
                    spots.append({
                        "pos":   (self._px[i], self._py[i]),
                        "size":  9,
                        "brush": pg.mkBrush(color),
                        "pen":   pg.mkPen("k", width=1),
                    })
                else:
                    # Default unselected dot: dark blue, small.
                    spots.append({
                        "pos":   (self._px[i], self._py[i]),
                        "size":  7,
                        "brush": pg.mkBrush(_UNSELECTED_FILL),
                        "pen":   pg.mkPen(_UNSELECTED_PEN, width=0.5),
                    })

            self._scatter = pg.ScatterPlotItem(spots=spots)
            self._plot.addItem(self._scatter)
        else:
            self._px = self._py = np.array([])

        # --- Layer 4: Drag rectangle (dashed; initially hidden) ---
        self._drag_rect_line = pg.PlotCurveItem(
            pen=pg.mkPen(color="#e63946", width=1, style=Qt.DashLine),
            connect="all",
        )
        self._drag_rect_line.setVisible(False)
        self._plot.addItem(self._drag_rect_line)

        # --- Layer 5: Hover tooltip (semi-transparent pale yellow; Z=100) ---
        self._tooltip = pg.TextItem(
            text="", color=(0, 0, 0),
            fill=pg.mkBrush(255, 255, 220, 220),  # Pale yellow, semi-transparent
            anchor=(0, 1),  # Anchor at bottom-left so it appears above/right of cursor
        )
        self._tooltip.setZValue(100)
        self._plot.addItem(self._tooltip)
        self._tooltip.hide()

    # -----------------------------------------------------------------------
    # Mouse event handlers
    # -----------------------------------------------------------------------

    def _on_mouse_moved(self, pos: QPointF) -> None:
        """Show or hide the hover tooltip as the cursor moves over the plot.

        Logic (preference order — closer wins):
          1. Check the nearest frame scatter dot within a 1.5% threshold.
          2. Check the nearest nav trackline point within the same threshold.
          3. If neither is close enough, hide the tooltip.
          4. If a frame dot is closer (or tied), show the frame label.
          5. If a nav point is closer, show GPS time and, if applicable, the
             corresponding video name and within-video time offset.

        The threshold is computed as 1.5% of the current view width/height,
        so it scales correctly with zoom level and doesn't feel sticky.

        Args:
            pos: Scene (pixel) coordinates of the current mouse position,
                 emitted by sigMouseMoved.
        """
        if self._tooltip is None:
            return
        if len(self._nav_px) == 0 and len(self._px) == 0:
            return

        # Ignore mouse events outside the plot area (e.g. over the axes).
        if not self._plot.sceneBoundingRect().contains(pos):
            self._tooltip.hide()
            return

        # Convert scene coordinates to view (data) coordinates.
        mp = self._vb.mapSceneToView(pos)
        mx, my = mp.x(), mp.y()

        # Compute a view-proportional threshold for snapping.
        # 1.5% of each axis range, squared for use with squared distances.
        vr        = self._vb.viewRange()
        xs        = vr[0][1] - vr[0][0]
        ys        = vr[1][1] - vr[1][0]
        threshold = (xs * 0.015) ** 2 + (ys * 0.015) ** 2

        # --- Find the nearest frame scatter dot ---
        frame_dist = float("inf")
        frame_idx  = -1
        if len(self._px) > 0:
            dists     = (self._px - mx) ** 2 + (self._py - my) ** 2
            frame_idx = int(np.argmin(dists))
            frame_dist = float(dists[frame_idx])

        # --- Find the nearest nav trackline point (only if timestamps available) ---
        nav_dist = float("inf")
        nav_idx  = -1
        has_nav_times = (
            len(self._nav_px) > 0 and
            len(self._nav_unix_times) == len(self._nav_px)
        )
        if has_nav_times:
            dists    = (self._nav_px - mx) ** 2 + (self._nav_py - my) ** 2
            nav_idx  = int(np.argmin(dists))
            nav_dist = float(dists[nav_idx])

        # If neither candidate is within the threshold, hide the tooltip.
        best_dist = min(frame_dist, nav_dist)
        if best_dist > threshold:
            self._tooltip.hide()
            return

        if frame_dist <= nav_dist and frame_idx >= 0:
            # --- Frame dot tooltip: show the pre-built label string ---
            self._tooltip.setText(self._labels[frame_idx])
            self._tooltip.setPos(self._px[frame_idx], self._py[frame_idx])
        else:
            # --- Nav trackline tooltip ---
            unix_t = float(self._nav_unix_times[nav_idx])

            # In history mode, check whether the hovered point falls inside a
            # history band and show the band's label instead of the GPS time.
            history_tooltip: str | None = None
            if self._history_ranges:
                for h_start, h_end, _color, h_text in self._history_ranges:
                    if h_start <= unix_t <= h_end:
                        history_tooltip = h_text
                        break

            if history_tooltip is not None:
                self._tooltip.setText(history_tooltip)
                self._tooltip.setPos(self._nav_px[nav_idx], self._nav_py[nav_idx])
                self._tooltip.show()
                return

            dt_str = datetime.utcfromtimestamp(unix_t).strftime("%Y-%m-%d %H:%M:%S")
            lat    = float(self._nav_lats[nav_idx])
            lon    = float(self._nav_lons[nav_idx])
            lines  = [
                f"GPS time:  {dt_str}",
                f"Lat:       {lat:.6f}°",
                f"Lon:       {lon:.6f}°",
            ]

            # Always show altitude when available.
            if (self._nav_alt_values is not None
                    and len(self._nav_alt_values) == len(self._nav_px)):
                alt = float(self._nav_alt_values[nav_idx])
                if np.isfinite(alt):
                    lines.append(f"Altitude:  {alt:.2f} m")

            # Also show active sensor channel if it differs from altitude.
            if (self._sensor_coloring_enabled
                    and self._sensor_coloring_values is not None
                    and self._sensor_label
                    and "altitude" not in self._sensor_label.lower()
                    and len(self._sensor_coloring_values) == len(self._nav_px)):
                val = float(self._sensor_coloring_values[nav_idx])
                if np.isfinite(val):
                    lines.append(f"{self._sensor_label}: {val:.4g}")

            # Search for a video whose time range covers this GPS timestamp.
            for v_start, v_end, v_name in self._videos:
                if v_start <= unix_t <= v_end:
                    off = unix_t - v_start
                    h, rem = divmod(off, 3600)
                    m, s   = divmod(rem, 60)
                    lines.append(f"Video:     {v_name}")
                    lines.append(f"Video time: {int(h):02d}:{int(m):02d}:{s:06.3f}")
                    break  # Stop after the first matching video

            self._tooltip.setText("\n".join(lines))
            self._tooltip.setPos(self._nav_px[nav_idx], self._nav_py[nav_idx])

        self._tooltip.show()

    def _on_mouse_clicked(self, ev) -> None:
        """Handle left-click for two-click trackline interval picking.

        Only active when pick mode is enabled.  Snaps the click to the
        nearest nav trackline point within a 3% view-range threshold
        (slightly more lenient than the hover threshold so clicking is
        not frustrating).

        First click: stores the timestamp as T1, redraws to show the star
        marker, emits pick_point_placed(unix_t).

        Second click: emits segment_created(min(T1,T2), max(T1,T2)),
        then exits pick mode automatically.

        Args:
            ev: pyqtgraph mouse click event object.
        """
        if not self._pick_mode:
            return
        if ev.button() != Qt.LeftButton:
            return

        pos = ev.scenePos()
        if not self._plot.sceneBoundingRect().contains(pos):
            return

        # Require that timestamps are available (can't do picks without them).
        if len(self._nav_px) == 0 or len(self._nav_unix_times) != len(self._nav_px):
            return

        # Convert the click position to view coordinates and snap to the nearest
        # nav point, using a 3% threshold (more generous than hover's 1.5%).
        mp   = self._vb.mapSceneToView(pos)
        mx, my = mp.x(), mp.y()
        vr   = self._vb.viewRange()
        xs   = vr[0][1] - vr[0][0]
        ys   = vr[1][1] - vr[1][0]
        threshold = (xs * 0.03) ** 2 + (ys * 0.03) ** 2

        dists   = (self._nav_px - mx) ** 2 + (self._nav_py - my) ** 2
        nearest = int(np.argmin(dists))

        # Reject the click if it's too far from any trackline point.
        if float(dists[nearest]) > threshold:
            return

        unix_t = float(self._nav_unix_times[nearest])
        ev.accept()  # Consume the event so the ViewBox doesn't also process it

        if self._pick_t1 is None:
            # --- First click: record T1 ---
            self._pick_t1 = unix_t
            self._redraw()                      # Draw the star marker at T1
            self.pick_point_placed.emit(unix_t)  # Notify MainWindow to update label
        else:
            # --- Second click: emit the completed interval and exit pick mode ---
            t1, t2        = self._pick_t1, unix_t
            self._pick_t1 = None
            self._pick_mode = False
            self._redraw()
            # Always emit (min, max) so the interval is start ≤ end regardless
            # of which end the user clicked first.
            self.segment_created.emit(min(t1, t2), max(t1, t2))

    def _on_drag_rect(self, rect: QRectF) -> None:
        """Update the dashed rubber-band rectangle preview during a drag selection.

        Converts the QRectF into a 5-point closed polygon path and sets it on
        the drag_rect_line curve so it appears as a dashed rectangle outline.

        Args:
            rect: Current selection rectangle in view (data) coordinates.
        """
        if self._drag_rect_line is None:
            return
        x0, y0, x1, y1 = rect.left(), rect.top(), rect.right(), rect.bottom()
        # Five points: go around the rectangle and close it back at the start.
        self._drag_rect_line.setData([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0])
        self._drag_rect_line.setVisible(True)

    def _on_selection_committed(self, rect: Optional[QRectF]) -> None:
        """Finalise the rubber-band selection when the user releases the mouse.

        If rect is None (a click with no drag), the selection is cleared.
        Otherwise, all frame dots whose projected positions fall inside the
        rectangle are added to the selection, and selection_changed is emitted.

        Args:
            rect: The final selection rectangle in view coordinates, or None
                  to clear the selection.
        """
        if self._drag_rect_line is not None:
            self._drag_rect_line.setVisible(False)

        if rect is None or (rect.width() == 0 and rect.height() == 0):
            # A click without dragging: clear the selection.
            self._selection = []
            self._redraw()
            self.selection_changed.emit([])
            return

        # Find all frame dots whose (px, py) fall within the rectangle bounds.
        x_min = min(rect.left(), rect.right())
        x_max = max(rect.left(), rect.right())
        y_min = min(rect.top(),  rect.bottom())
        y_max = max(rect.top(),  rect.bottom())
        selected = [
            i for i in range(len(self._px))
            if x_min <= self._px[i] <= x_max and y_min <= self._py[i] <= y_max
        ]

        self._selection = selected
        self._redraw()
        self.selection_changed.emit(selected)
