"""
product_tree_canvas.py — a "tech tree" visual of the product dependency graph.

A PROTOTYPE alternative to product_tree_widget's QTreeWidget: the same graph
(product_graph.PRODUCT_GRAPH) drawn War-Thunder-style as node CARDS laid out in
dependency TIERS with dependency LINES between them, on a pan/zoom QGraphicsView
canvas that fills the whole pane.

It is a DROP-IN sibling of ProductTreeWidget — same construction signature, the
same buildRequested / runRequested / exportRequested contract, and the same
Qt-free registry helpers (reused from product_tree_widget, not reimplemented):
produced_node_ids / runs_for_node / registry_scopes / scope_display_label /
node_description.  ALL graph math stays in product_graph.py — this module only
lays the graph out and renders it.

Two layers, like its sibling:

  * a Qt-FREE, tested core (compute_tiers / branch_of / tier_rows) that turns the
    DAG into (tier, row) positions — data roots in the leftmost tier, each product
    one tier past the deepest thing it `requires`, deterministic vertical order
    grouped by branch (video/photogrammetry chain vs nav/sensor chain, aggregate
    last); and

  * ProductTreeCanvas — the QWidget wrapping a QGraphicsScene/QGraphicsView, the
    node cards, the scope selector, the detail side-panel and the build button.

THREADING.  Every signal here is connected to a BOUND METHOD of a QObject, per
the app's threading rule.  refresh() is a plain public method the main window
calls from its own main-thread bound slots (workspace load / stack finished) —
no cross-thread signal is introduced.  The card items are plain QGraphicsItems
(not QObjects) and talk back to the canvas by direct, same-thread method calls.
"""

from __future__ import annotations

from pathlib import Path

import product_graph as pg
# Reuse the sibling's Qt-free registry core verbatim — do NOT reimplement it.
from product_tree_widget import (
    SURVEY_SCOPE,
    SURVEY_LABEL,
    _SURVEY_ALIASES,
    node_description,
    produced_node_ids,
    registry_scopes,
    runs_for_node,
    scope_display_label,
)


# ===========================================================================
# Qt-free layout core (unit-tested in tests/test_product_tree_canvas.py)
# ===========================================================================
# Branch buckets for deterministic vertical ordering within a tier.  The video /
# photogrammetry chain sits above the nav / sensor chain; aggregate nodes (report)
# — which pull from both chains — sit last, at the bottom.
BRANCH_VIDEO = "video"      # rank 0
BRANCH_NAV = "nav"          # rank 1
BRANCH_AGGREGATE = "aggregate"  # rank 2
_BRANCH_RANK = {BRANCH_VIDEO: 0, BRANCH_NAV: 1, BRANCH_AGGREGATE: 2}


def branch_of(node_id: str) -> str:
    """Which chain a node belongs to, for vertical grouping within a tier.

    Derived from the node's transitive data needs (product_graph.transitive_needs):
    a node that touches video is on the video/photogrammetry chain, one that
    touches only nav/sensors is on the nav/sensor chain, and a node that pulls
    from BOTH (the aggregate 'report') is an aggregate node placed last.
    """
    needs = pg.transitive_needs(node_id)
    has_video = "video" in needs
    has_navsensor = bool(needs & {"nav", "sensors"})
    if has_video and has_navsensor:
        return BRANCH_AGGREGATE
    if has_video:
        return BRANCH_VIDEO
    if has_navsensor:
        return BRANCH_NAV
    # A node with no data needs at all: treat as aggregate (nothing does today).
    return BRANCH_AGGREGATE


def compute_tiers() -> dict[str, int]:
    """node_id -> tier (dependency depth).

    Data-root nodes are tier 0; every other node is 1 + the max tier of the nodes
    it `requires`.  A pure function of the graph (no imports/registry), so the
    columns are stable and testable.
    """
    tiers: dict[str, int] = {}

    def depth(nid: str) -> int:
        if nid in tiers:
            return tiers[nid]
        node = pg.get_node(nid)
        if not node.requires:
            tiers[nid] = 0
            return 0
        d = 1 + max(depth(p) for p in node.requires)
        tiers[nid] = d
        return d

    for node in pg.all_nodes():
        depth(node.id)
    return tiers


def tier_rows() -> dict[int, list[str]]:
    """tier -> ordered list of node ids in that tier (top-to-bottom row order).

    Deterministic: within a tier, nodes are grouped by branch (video chain, then
    nav/sensor chain, then aggregate) and, within a branch, kept in graph
    declaration order.  Every node appears in exactly one tier, exactly once.
    """
    tiers = compute_tiers()
    decl_order = {n.id: i for i, n in enumerate(pg.all_nodes())}
    rows: dict[int, list[str]] = {}
    for nid, tier in tiers.items():
        rows.setdefault(tier, []).append(nid)
    for tier, ids in rows.items():
        ids.sort(key=lambda nid: (_BRANCH_RANK.get(branch_of(nid), 9),
                                  decl_order.get(nid, 0)))
    return rows


# ===========================================================================
# Qt layer
# ===========================================================================
from PySide6.QtCore import Qt, QRectF, QPointF, Signal, QUrl
from PySide6.QtGui import (
    QBrush, QColor, QDesktopServices, QFont, QPainter, QPainterPath, QPen,
)
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGraphicsItem,
    QGraphicsScene,
    QGraphicsView,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

# -- card geometry -----------------------------------------------------------
CARD_W = 190
CARD_H = 62
COL_SPACING = 270          # horizontal distance between tiers
ROW_SPACING = 92           # vertical distance between rows in a tier
CHECK_BOX = 16             # checkbox side length

# -- palette (state -> (fill, border, text)) ---------------------------------
_PALETTE = {
    "locked":    (QColor("#eceff1"), QColor("#b0bec5"), QColor("#90a4ae")),
    "available": (QColor("#ffffff"), QColor("#1565c0"), QColor("#1a237e")),
    "produced":  (QColor("#e8f5e9"), QColor("#2e7d32"), QColor("#1b5e20")),
    "root":      (QColor("#e3f2fd"), QColor("#5c6bc0"), QColor("#283593")),
    "root_off":  (QColor("#f5f5f5"), QColor("#bdbdbd"), QColor("#9e9e9e")),
}
_ACCENT_CHECKED = QColor("#1565c0")
_EDGE_COLOR = QColor("#b0b7c3")


class _NodeCard(QGraphicsItem):
    """One product/data-root node, drawn as a rounded-rect card.

    Not a QObject: it calls back into the owning ProductTreeCanvas by direct
    method call (same thread), never via a signal — so no worker-thread affinity
    question ever arises.
    """

    def __init__(self, canvas: "ProductTreeCanvas", node, state: str,
                 *, checkable: bool, checked: bool, run_count: int,
                 imported: bool) -> None:
        super().__init__()
        self._canvas = canvas
        self.node = node
        self.state = state             # locked | available | produced | root | root_off
        self.checkable = checkable
        self.checked = checked
        self.run_count = run_count
        self.imported = imported
        self.setAcceptHoverEvents(True)
        if state == "locked":
            reason = pg.unavailable_reason(node.id, canvas._imported) or "unavailable"
            self.setToolTip(reason)
        else:
            self.setToolTip(node_description(node))

    # -- geometry ------------------------------------------------------------
    def boundingRect(self) -> QRectF:
        return QRectF(0, 0, CARD_W, CARD_H)

    def _checkbox_rect(self) -> QRectF:
        return QRectF(10, (CARD_H - CHECK_BOX) / 2, CHECK_BOX, CHECK_BOX)

    # -- painting ------------------------------------------------------------
    def paint(self, painter: QPainter, option, widget=None) -> None:
        painter.setRenderHint(QPainter.Antialiasing, True)
        fill, border, text_color = _PALETTE[self.state]

        pen = QPen(border, 2.5 if self.state in ("produced",) else 1.6)
        if self.node.is_archived or self.state == "locked":
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(fill))
        painter.drawRoundedRect(QRectF(1, 1, CARD_W - 2, CARD_H - 2), 9, 9)

        # text region: leave room for a checkbox on product cards.
        text_left = 34 if self.checkable else 12
        font = QFont()
        font.setPointSize(9)
        font.setBold(self.state == "produced" or self.node.is_root)
        font.setItalic(self.node.is_archived)
        painter.setFont(font)
        painter.setPen(QPen(text_color))
        label_rect = QRectF(text_left, 6, CARD_W - text_left - 8, CARD_H - 26)
        painter.drawText(label_rect, Qt.AlignLeft | Qt.TextWordWrap, self.node.label)

        # badge line at the bottom.
        badge_font = QFont()
        badge_font.setPointSize(8)
        painter.setFont(badge_font)
        painter.setPen(QPen(text_color))
        painter.drawText(QRectF(text_left, CARD_H - 20, CARD_W - text_left - 8, 16),
                         Qt.AlignLeft | Qt.AlignVCenter, self._badge_text())

        # checkbox affordance (product cards only).
        if self.checkable:
            box = self._checkbox_rect()
            painter.setPen(QPen(border, 1.4))
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRoundedRect(box, 3, 3)
            if self.checked:
                painter.setBrush(QBrush(_ACCENT_CHECKED))
                painter.setPen(QPen(_ACCENT_CHECKED))
                inner = box.adjusted(3, 3, -3, -3)
                painter.drawRoundedRect(inner, 2, 2)

    def _badge_text(self) -> str:
        if self.node.is_root:
            return "imported ✓" if self.imported else "not imported ✗"
        if self.state == "produced":
            n = self.run_count
            return f"✓ produced" + (f" · {n} run{'s' if n != 1 else ''}"
                                         if n else "")
        if self.state == "locked":
            return "✗ locked"
        arch = " · archived" if self.node.is_archived else ""
        return "○ available" + arch

    # -- interaction ---------------------------------------------------------
    def mousePressEvent(self, event) -> None:
        if (self.checkable and event.button() == Qt.LeftButton
                and self._checkbox_rect().contains(event.pos())):
            self._canvas._toggle_check(self.node.id)
            event.accept()
            return
        # Body click: select the node (detail panel + nodeSelected signal).
        self._canvas._select_node(self.node.id)
        event.accept()


class _CanvasView(QGraphicsView):
    """QGraphicsView with wheel-zoom and left-drag panning over empty space.

    Clicking a card still reaches the card (its press is delivered normally);
    dragging the empty background pans the scene.
    """

    def __init__(self, scene: QGraphicsScene, parent=None) -> None:
        super().__init__(scene, parent)
        self.setRenderHint(QPainter.Antialiasing, True)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setBackgroundBrush(QBrush(QColor("#fafbfc")))
        self._panning = False
        self._pan_last = QPointF()
        self._zoom = 1.0

    def wheelEvent(self, event) -> None:
        step = 1.15 if event.angleDelta().y() > 0 else 1 / 1.15
        new_zoom = self._zoom * step
        if 0.25 <= new_zoom <= 3.0:
            self._zoom = new_zoom
            self.scale(step, step)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and not isinstance(
                self.itemAt(event.position().toPoint()), _NodeCard):
            self._panning = True
            self._pan_last = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            delta = event.position() - self._pan_last
            self._pan_last = event.position()
            self.horizontalScrollBar().setValue(
                self.horizontalScrollBar().value() - int(delta.x()))
            self.verticalScrollBar().setValue(
                self.verticalScrollBar().value() - int(delta.y()))
            event.accept()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == Qt.LeftButton:
            self._panning = False
            self.unsetCursor()
            event.accept()
            return
        super().mouseReleaseEvent(event)


class _NodeSettingsDialog(QDialog):
    """Per-node dialog: SET run options, VIEW previously-produced runs, EXPORT them.

    Rendered from the node's ``params`` schema (product_graph Node.params) plus the
    registry's completed runs for the scope (runs_for_node).  A QDialog is itself a
    QObject, so every button connects to a BOUND method here; per-run buttons carry
    their run dict as a Qt property so one bound slot serves them all (no lambdas).

    It does not own the signal contract — it calls back into the canvas, which
    emits runRequested / exportRequested — so the drop-in signal shape is unchanged.
    The collected form values are stashed on the canvas (``_pending_run_settings``)
    so a follow-up can pick them up without widening the two-arg signal.
    """

    _PROP_RUN = "_run_dict"

    def __init__(self, canvas: "ProductTreeCanvas", node, scope_id: str,
                 runs: list[dict], can_run: bool, parent=None) -> None:
        super().__init__(parent)
        self._canvas = canvas
        self._node = node
        self._scope_id = scope_id
        self._fields: dict[str, tuple[str, QWidget]] = {}
        self.setWindowTitle(f"{node.label} — {scope_display_label(scope_id)}")
        self.setMinimumWidth(420)

        lay = QVBoxLayout(self)

        desc = QLabel(node_description(node))
        desc.setWordWrap(True)
        desc.setStyleSheet("color:#555;")
        lay.addWidget(desc)

        # (1) run options form ------------------------------------------------
        opts = QGroupBox("Run options")
        form = QFormLayout(opts)
        if node.params:
            for name, spec in node.params.items():
                w = self._make_field(spec)
                self._fields[name] = (spec.get("type", "str"), w)
                form.addRow(spec.get("label", name), w)
        else:
            form.addRow(QLabel("<i>No configurable options for this node.</i>"))
        lay.addWidget(opts)

        # (2)+(3) produced runs: view + export --------------------------------
        prod = QGroupBox(f"Produced runs ({len(runs)})")
        players = QVBoxLayout(prod)
        if runs:
            for r in runs[:20]:
                players.addLayout(self._run_row(r))
        else:
            players.addWidget(QLabel("<i>No produced run for this scope yet.</i>"))
        lay.addWidget(prod)

        # buttons -------------------------------------------------------------
        bar = QDialogButtonBox()
        self._btn_new = bar.addButton("Start new run",
                                      QDialogButtonBox.AcceptRole)
        self._btn_new.setEnabled(can_run)
        bar.addButton(QDialogButtonBox.Close)
        self._btn_new.clicked.connect(self._on_new_run)
        bar.rejected.connect(self.reject)
        lay.addWidget(bar)

    # -- field construction --------------------------------------------------
    def _make_field(self, spec: dict) -> QWidget:
        t = spec.get("type", "str")
        default = spec.get("default")
        if t == "bool":
            w = QCheckBox()
            w.setChecked(bool(default))
            return w
        if t == "choice":
            w = QComboBox()
            for c in spec.get("choices", []):
                w.addItem(str(c))
            if default is not None:
                i = w.findText(str(default))
                if i >= 0:
                    w.setCurrentIndex(i)
            return w
        if t == "int":
            w = QSpinBox()
            w.setRange(-1_000_000, 1_000_000)
            w.setValue(int(default) if default is not None else 0)
            return w
        if t == "float":
            w = QDoubleSpinBox()
            w.setRange(-1_000_000.0, 1_000_000.0)
            w.setDecimals(3)
            w.setValue(float(default) if default is not None else 0.0)
            return w
        w = QLineEdit("" if default is None else str(default))
        return w

    def _run_row(self, run: dict) -> QHBoxLayout:
        row = QHBoxLayout()
        when = (run.get("finished_at") or "").replace("T", " ")[:19]
        lbl = QLabel(f"{run.get('run_id', '')}  ·  {when}  ·  "
                     f"{run.get('status', '')}")
        lbl.setStyleSheet("font-size:11px;")
        row.addWidget(lbl, 1)
        btn_view = QPushButton("View")
        btn_view.setProperty(self._PROP_RUN, run)
        btn_view.clicked.connect(self._on_view_run)
        row.addWidget(btn_view)
        btn_export = QPushButton("Export")
        btn_export.setProperty(self._PROP_RUN, run)
        btn_export.clicked.connect(self._on_export_run)
        row.addWidget(btn_export)
        return row

    # -- values --------------------------------------------------------------
    def collect_settings(self) -> dict:
        """The current form values as {param_name: value}."""
        out: dict = {}
        for name, (t, w) in self._fields.items():
            if isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[name] = w.value()
            elif isinstance(w, QLineEdit):
                out[name] = w.text()
        return out

    # -- actions (bound slots) ----------------------------------------------
    def _on_new_run(self) -> None:
        self._canvas._pending_run_settings[(self._node.id, self._scope_id)] = \
            self.collect_settings()
        self._canvas.runRequested.emit(self._node.id, self._scope_id)
        self.accept()

    def _on_view_run(self) -> None:
        run = self.sender().property(self._PROP_RUN)
        out = (run or {}).get("output_dir")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _on_export_run(self) -> None:
        self._canvas.exportRequested.emit(self._node.id, self._scope_id)
        self._on_view_run()


class ProductTreeCanvas(QWidget):
    """Tech-tree canvas view of PRODUCT_GRAPH — a drop-in sibling of
    ProductTreeWidget with the same construction + signal contract."""

    buildRequested = Signal(list)          # [{"scope_id", "node_ids"}]
    runRequested = Signal(str, str)        # (node_id, scope_id)
    exportRequested = Signal(str, str)     # (node_id, scope_id)
    nodeSelected = Signal(str, str)        # (node_id, scope_id)

    def __init__(self, workspace_dir: str = "", imported=None,
                 scope_id: str = SURVEY_SCOPE, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ws = str(workspace_dir or "")
        self._imported: set[str] = set(imported or ())
        self._scope_id = self._norm_scope(scope_id)
        self._reg = None
        self._available: set[str] = set()
        self._produced: set[str] = set()
        # checked product ids, kept per scope so switching scope preserves picks.
        self._checked_by_scope: dict[str, set[str]] = {}
        self._selected: str | None = None
        self._cards: dict[str, _NodeCard] = {}
        # last form values captured from a node dialog's "Start new run", keyed by
        # (node_id, scope_id) — lets a follow-up read the options without widening
        # the two-arg runRequested signal that keeps this a drop-in sibling.
        self._pending_run_settings: dict[tuple[str, str], dict] = {}
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _norm_scope(scope_id) -> str:
        return SURVEY_SCOPE if scope_id in _SURVEY_ALIASES else str(scope_id)

    def _checked(self) -> set[str]:
        return self._checked_by_scope.setdefault(self._scope_id, set())

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        # -- top bar: scope selector + build button --------------------------
        top = QHBoxLayout()
        top.addWidget(QLabel("Scope:"))
        self._scope_combo = QComboBox()
        self._scope_combo.currentIndexChanged.connect(self._on_scope_changed)
        top.addWidget(self._scope_combo)
        top.addStretch(1)
        self._hint = QLabel("Drag to pan · scroll to zoom · click a card to "
                            "inspect, its checkbox to select for a build")
        self._hint.setStyleSheet("color: #888; font-size: 10px;")
        top.addWidget(self._hint)
        top.addStretch(1)
        self._btn_build = QPushButton("▶  Build selected")
        self._btn_build.setStyleSheet("font-weight: bold; padding: 5px 10px;")
        self._btn_build.clicked.connect(self._emit_build)
        self._btn_build.setEnabled(False)
        top.addWidget(self._btn_build)
        outer.addLayout(top)

        # -- canvas + detail panel -------------------------------------------
        split = QSplitter(Qt.Horizontal)
        outer.addWidget(split, 1)

        self._scene = QGraphicsScene(self)
        self._view = _CanvasView(self._scene)
        split.addWidget(self._view)

        detail = QWidget()
        dlay = QVBoxLayout(detail)
        dlay.setContentsMargins(6, 0, 0, 0)
        self._info_title = QLabel("Select a product")
        self._info_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._info_title.setWordWrap(True)
        dlay.addWidget(self._info_title)
        self._info = QTextEdit()
        self._info.setReadOnly(True)
        dlay.addWidget(self._info, 1)
        btn_row = QHBoxLayout()
        self._btn_export = QPushButton("Export product…")
        self._btn_export.clicked.connect(self._export_current)
        self._btn_run = QPushButton("New run…")
        self._btn_run.clicked.connect(self._run_current)
        for b in (self._btn_export, self._btn_run):
            b.setEnabled(False)
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        dlay.addLayout(btn_row)
        split.addWidget(detail)
        split.setStretchFactor(0, 3)
        split.setStretchFactor(1, 1)
        split.setSizes([760, 300])

        # -- build preview ---------------------------------------------------
        self._preview = QLabel("Check products to build.")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(
            "color: #333; font-size: 11px; background: #f4f4f4; "
            "border: 1px solid #ddd; padding: 5px;")
        outer.addWidget(self._preview)

    # -------------------------------------------------------------- public API
    def set_workspace(self, workspace_dir: str, imported=None,
                      scope_id: str | None = None) -> None:
        """Point the canvas at a workspace + imported-data set and re-render."""
        self._ws = str(workspace_dir or "")
        if imported is not None:
            self._imported = set(imported)
        if scope_id is not None:
            self._scope_id = self._norm_scope(scope_id)
        self.refresh()

    def set_imported(self, imported) -> None:
        """Update which data roots are imported (gates availability); re-render."""
        self._imported = set(imported or ())
        self.refresh()

    def set_scope(self, scope_id: str) -> None:
        """Show a single scope; recompute produced badges and re-render."""
        self._scope_id = self._norm_scope(scope_id)
        self.refresh()

    def refresh(self) -> None:
        """Re-read the registry, recompute availability/produced for the current
        scope, and redraw the whole scene.

        Bound method: safe to call from the main window's main-thread handlers.
        Preserves the checked selection (intersected with still-available nodes).
        """
        self._reg = self._read_registry()
        self._available = pg.available_nodes(self._imported)
        self._produced = produced_node_ids(self._reg, self._scope_id)
        # Drop checks that are no longer available (e.g. data un-imported).
        chk = self._checked()
        chk &= (self._available - {n.id for n in pg.data_root_nodes()})
        self._sync_scope_combo()
        self._rebuild_scene()
        self._update_preview()
        self._render_info(self._selected)

    def checked_ids(self) -> list[tuple[str, str]]:
        """Currently checked (node_id, scope_id) targets, graph-declaration order."""
        order = {n.id: i for i, n in enumerate(pg.all_nodes())}
        return [(nid, self._scope_id)
                for nid in sorted(self._checked(), key=lambda n: order.get(n, 0))]

    # --------------------------------------------------------------- internals
    def _scope_list(self) -> list[tuple[str, str]]:
        scopes = [(SURVEY_SCOPE, SURVEY_LABEL)]
        for sid in registry_scopes(self._reg):
            scopes.append((sid, scope_display_label(sid)))
        return scopes

    def _sync_scope_combo(self) -> None:
        scopes = self._scope_list()
        self._scope_combo.blockSignals(True)
        self._scope_combo.clear()
        for sid, label in scopes:
            self._scope_combo.addItem(label, sid)
        idx = next((i for i, (sid, _) in enumerate(scopes)
                    if sid == self._scope_id), 0)
        self._scope_combo.setCurrentIndex(idx)
        self._scope_id = scopes[idx][0]
        self._scope_combo.blockSignals(False)

    def _read_registry(self):
        if not self._ws:
            return None
        try:
            from workspace_paths import PathResolver
            from manifest import Registry
            return Registry(PathResolver(self._ws).registry_json())
        except Exception:
            return None

    def _rebuild_scene(self) -> None:
        self._scene.clear()
        self._cards = {}
        positions = self._pixel_positions()

        # edges first (lower z), so cards paint on top.
        for node in pg.all_nodes():
            cx, cy = positions[node.id]
            for parent in node.requires:
                px, py = positions[parent]
                self._add_edge(px + CARD_W, py + CARD_H / 2, cx, cy + CARD_H / 2)

        # cards on top.
        for node in pg.all_nodes():
            x, y = positions[node.id]
            card = self._make_card(node)
            card.setPos(x, y)
            card.setZValue(1)
            self._scene.addItem(card)
            self._cards[node.id] = card

        # a little breathing room around the graph.
        rect = self._scene.itemsBoundingRect().adjusted(-60, -60, 60, 60)
        self._scene.setSceneRect(rect)

    def _pixel_positions(self) -> dict[str, tuple[float, float]]:
        """node_id -> (x, y) pixel top-left, from the Qt-free tier/row layout.

        Each tier is a column; rows within a tier are centred vertically so the
        columns line up around a shared midline (a tidy tech-tree look)."""
        rows = tier_rows()
        max_rows = max((len(ids) for ids in rows.values()), default=1)
        mid = (max_rows - 1) * ROW_SPACING / 2
        pos: dict[str, tuple[float, float]] = {}
        for tier, ids in rows.items():
            n = len(ids)
            start = mid - (n - 1) * ROW_SPACING / 2
            for i, nid in enumerate(ids):
                pos[nid] = (tier * COL_SPACING, start + i * ROW_SPACING)
        return pos

    def _add_edge(self, x1, y1, x2, y2) -> None:
        path = QPainterPath(QPointF(x1, y1))
        dx = (x2 - x1) * 0.5
        path.cubicTo(x1 + dx, y1, x2 - dx, y2, x2, y2)
        item = self._scene.addPath(path, QPen(_EDGE_COLOR, 1.6))
        item.setZValue(0)

    def _make_card(self, node) -> _NodeCard:
        if node.is_root:
            imported = node.id in self._imported
            return _NodeCard(self, node, "root" if imported else "root_off",
                             checkable=False, checked=False, run_count=0,
                             imported=imported)
        status = pg.node_status(node.id, self._produced, self._available)
        state = {"produced": "produced", "available": "available",
                 "unavailable": "locked"}[status]
        run_count = len(runs_for_node(self._reg, node.id, self._scope_id))
        return _NodeCard(
            self, node, state,
            checkable=(state != "locked"),
            checked=node.id in self._checked(),
            run_count=run_count, imported=True)

    # ------------------------------------------------------------------ events
    def _on_scope_changed(self, index: int) -> None:
        sid = self._scope_combo.itemData(index)
        if sid is not None:
            self.set_scope(sid)

    def _toggle_check(self, node_id: str) -> None:
        chk = self._checked()
        if node_id in chk:
            chk.discard(node_id)
        else:
            chk.add(node_id)
        card = self._cards.get(node_id)
        if card is not None:
            card.checked = node_id in chk
            card.update()
        self._update_preview()

    def _select_node(self, node_id: str) -> None:
        self._selected = node_id
        self._render_info(node_id)
        self.nodeSelected.emit(node_id, self._scope_id)
        self._open_node_dialog(node_id)

    def _open_node_dialog(self, node_id: str) -> None:
        """Open the per-node dialog: set run options, view + export produced runs.

        Data-root nodes are imported, not run, so they get no dialog (the side
        panel already summarises them)."""
        node = pg.get_node(node_id)
        if node.is_root:
            return
        runs = runs_for_node(self._reg, node_id, self._scope_id) if node.task_type else []
        can_run = node.task_type is not None and node_id in self._available
        dlg = _NodeSettingsDialog(self, node, self._scope_id, runs, can_run, self)
        dlg.exec()

    # ------------------------------------------------------------------ preview
    def _plan(self) -> dict | None:
        chk = [nid for nid, _ in self.checked_ids()]
        if not chk:
            return None
        ordered = pg.expand_targets(chk, satisfied=self._produced)
        return {"scope_id": self._scope_id, "node_ids": ordered}

    def _update_preview(self) -> None:
        plan = self._plan()
        self._btn_build.setEnabled(plan is not None)
        if plan is None:
            self._preview.setText("Check products to build.")
            return
        label = scope_display_label(self._scope_id)
        if plan["node_ids"]:
            run = " → ".join(pg.get_node(o).label for o in plan["node_ids"])
            self._preview.setText(f"{label}:  {run}")
        else:
            self._preview.setText(f"{label}:  already produced — nothing to run.")

    def _emit_build(self) -> None:
        plan = self._plan()
        self.buildRequested.emit([plan] if plan is not None else [])

    # --------------------------------------------------------------- info panel
    def _render_info(self, node_id) -> None:
        if not node_id:
            self._info_title.setText("Select a product")
            self._info.setHtml("")
            self._btn_export.setEnabled(False)
            self._btn_run.setEnabled(False)
            return
        node = pg.get_node(node_id)
        self._info_title.setText(
            f"{node.label}  ·  {scope_display_label(self._scope_id)}")

        html = ["<p style='color:#444'>" + node_description(node) + "</p>"]
        if node.params:
            html.append("<p style='color:#666'><b>Default run parameters</b></p>"
                        "<table cellspacing='3'>")
            for key, spec in node.params.items():
                html.append(f"<tr><td style='color:#666'>{spec.get('label', key)}"
                            f"</td><td><code>{spec.get('default')}</code></td></tr>")
            html.append("</table>")

        runs = runs_for_node(self._reg, node_id, self._scope_id) if node.task_type else []
        if node.task_type is None:
            html.append("<p style='color:#888'><i>Aggregate node — no run of its "
                        "own.</i></p>")
        elif runs:
            html.append(f"<p style='color:#2e7d32'><b>{len(runs)} produced run(s) "
                        "for this scope</b></p><table cellspacing='3'>"
                        "<tr><td style='color:#666'><b>Run</b></td>"
                        "<td style='color:#666'><b>Finished</b></td>"
                        "<td style='color:#666'><b>Status</b></td></tr>")
            for r in runs[:12]:
                when = (r.get("finished_at") or "").replace("T", " ")[:19]
                html.append(f"<tr><td><code>{r.get('run_id','')}</code></td>"
                            f"<td>{when}</td><td>{r.get('status','')}</td></tr>")
            html.append("</table><p style='color:#888'>Output: "
                        f"<code>{runs[0].get('output_dir','')}</code></p>")
        elif node.is_root:
            pass
        else:
            html.append("<p style='color:#888'><i>No produced run for this scope "
                        "yet.</i></p>")

        self._info.setHtml("".join(html))
        self._btn_run.setEnabled(node.task_type is not None
                                 and node_id in self._available)
        self._btn_export.setEnabled(bool(runs))

    # ------------------------------------------------------------------ actions
    def _export_current(self) -> None:
        nid = self._selected
        if not nid:
            return
        self.exportRequested.emit(nid, self._scope_id)
        runs = runs_for_node(self._reg, nid, self._scope_id)
        if runs:
            out = runs[0].get("output_dir")
            if out and Path(out).exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _run_current(self) -> None:
        if self._selected:
            self.runRequested.emit(self._selected, self._scope_id)
