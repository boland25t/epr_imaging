"""
workbench.py — the primary IDE-style three-pane "workbench" (design "B").

A persistent NAVIGATOR | EDITOR | INSPECTOR layout that makes the product tech
tree the editor's default surface and drives SCOPE from trackline-interval
selection (default = the whole trackline / survey scope).  It is a thin HOST that
composes existing, already-tested pieces — it invents no graph or registry logic:

  * NAVIGATOR (left)   — a SCOPE selector (first/default row "Whole trackline")
    plus a compact product-node OUTLINE mirroring the DAG, grouped by branch.
  * EDITOR (center)    — a QStackedWidget whose page 0 is the ProductTreeCanvas
    tech tree; the app can register its own timeline/trackline and map/3D panels
    as swap-in pages via addEditorPage()/showEditorPage().
  * INSPECTOR (right)  — a PERSISTENT (not modal) panel for the selected
    (node, scope): a settings form built from the node's params, a run-history
    list, a preview placeholder, and Build / New-run affordances.

ALL graph math is delegated to product_graph; ALL registry/scope helpers are
reused from product_tree_widget via product_tree_canvas (registry_scopes,
scope_display_label, produced_node_ids, runs_for_node, node_description) — never
reimplemented here.  The tech-tree canvas itself is reused verbatim, including
its per-node settings dialog's field-building logic (_NodeSettingsDialog._make_field)
for the inspector's settings form.

THREADING.  Per the app's rule, every signal is connected to a BOUND METHOD of a
QObject — never a lambda or plain function.  Per-run View/Export buttons carry
their run dict as a Qt property so one bound slot serves them all.  refresh() is a
plain main-thread public method the main window calls from its own bound slots.
"""

from __future__ import annotations

from pathlib import Path

import product_graph as pg
# Reuse the tech-tree canvas AND its Qt-free registry/scope helpers + its per-node
# settings-field builder — do NOT reimplement any of it here.
from product_tree_canvas import (
    ProductTreeCanvas,
    _NodeSettingsDialog,
    branch_of,
    BRANCH_VIDEO,
    BRANCH_NAV,
    BRANCH_AGGREGATE,
)
from product_tree_widget import (
    SURVEY_SCOPE,
    _SURVEY_ALIASES,
    node_description,
    produced_node_ids,
    registry_scopes,
    runs_for_node,
    scope_display_label,
)


# ===========================================================================
# Qt-free helpers (unit-tested in tests/test_workbench.py)
# ===========================================================================
# The default scope's navigator label.  Design "B" frames the survey scope as the
# WHOLE TRACKLINE (the whole dive), so the navigator's first/default row reads
# "Whole trackline" even though its scope_id is the canonical SURVEY_SCOPE.
WHOLE_TRACKLINE_LABEL = "Whole trackline"

# Branch bucket -> outline group heading, in display order.
_OUTLINE_GROUP_LABELS = {
    BRANCH_VIDEO: "Video / Photogrammetry",
    BRANCH_NAV: "Navigation / Sensors",
    BRANCH_AGGREGATE: "Aggregate",
}
_OUTLINE_GROUP_ORDER = (BRANCH_VIDEO, BRANCH_NAV, BRANCH_AGGREGATE)

# Products whose natural editor surface is the map / 3D panel rather than the
# tech tree.  Selecting one of these tries to show the "map" editor page.
SPATIAL_NODE_IDS: frozenset[str] = frozenset({
    "mesh", "orthomosaic", "dem", "dense",      # photogrammetry (3D / raster)
    "trackline", "sensor_raster", "netcdf",     # nav/sensor spatial products
})


def assemble_scopes(registry, injected=None) -> list[tuple[str, str]]:
    """Build the navigator's ordered [(scope_id, label)] list.

    The FIRST, DEFAULT entry is always the whole-trackline (survey) scope.  After
    it come any externally INJECTED scopes (the trackline intervals / interval-sets
    the app hands in via set_scopes), then every additional scope that appears in
    the registry (named interval scopes + photogrammetry chunk scopes), each
    de-duplicated and never colliding with the survey scope.  Pure: the survey
    scope's aliases are folded, registry scope discovery is delegated to
    product_tree_widget.registry_scopes, and labels to scope_display_label.
    """
    out: list[tuple[str, str]] = [(SURVEY_SCOPE, WHOLE_TRACKLINE_LABEL)]
    seen: set = {SURVEY_SCOPE} | set(_SURVEY_ALIASES)
    for sid, label in (injected or []):
        if sid in seen:
            continue
        seen.add(sid)
        out.append((sid, label or scope_display_label(sid)))
    for sid in registry_scopes(registry):
        if sid in seen:
            continue
        seen.add(sid)
        out.append((sid, scope_display_label(sid)))
    return out


def outline_groups() -> list[tuple[str, list[str]]]:
    """The navigator outline as ordered (group_label, [node_id, ...]) sections.

    Mirrors the DAG grouped by branch — the video/photogrammetry chain (its data
    root included), then the nav/sensor chain, then aggregate nodes — with nodes
    kept in graph declaration order (data roots first, dependencies before
    dependents) within each group.  Pure: branch membership is delegated to
    product_tree_canvas.branch_of, so the outline stays consistent with the canvas.
    """
    buckets: dict[str, list[str]] = {b: [] for b in _OUTLINE_GROUP_ORDER}
    for node in pg.all_nodes():
        buckets.setdefault(branch_of(node.id), []).append(node.id)
    return [(_OUTLINE_GROUP_LABELS[b], buckets[b])
            for b in _OUTLINE_GROUP_ORDER if buckets.get(b)]


def is_spatial_node(node_id: str) -> bool:
    """True if this product's natural editor surface is the map / 3D page."""
    return node_id in SPATIAL_NODE_IDS


def node_badge(node_id: str, produced: set[str], available: set[str],
               imported: set[str]) -> str:
    """A one-glyph outline badge for a node under the current scope.

    imported/not for data roots; produced ✓ / available ○ / locked ✗ for products.
    Pure — takes the precomputed sets the caller already has.
    """
    node = pg.get_node(node_id)
    if node.is_root:
        return "✓" if node_id in imported else "✗"
    if node_id in produced:
        return "✓"
    if node_id in available:
        return "○"
    return "✗"


# ===========================================================================
# Qt layer
# ===========================================================================
from PySide6.QtCore import Qt, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

_ROLE_SCOPE = Qt.UserRole + 1
_ROLE_NODE = Qt.UserRole + 2
_PROP_RUN = "_wb_run_dict"


class Workbench(QWidget):
    """The three-pane IDE-style workbench (design "B").

    Public API:
      __init__(workspace_dir="", imported=None, scope_id="survey", parent=None)
      set_workspace(workspace_dir, imported=None, scope_id=None)
      set_imported(imported)
      set_scope(scope_id)
      set_scopes(list[(scope_id, label)])        # inject trackline intervals
      addEditorPage(widget, key) / showEditorPage(key)
      refresh()
    Signals: buildRequested(list), runRequested(str, str),
             exportRequested(str, str), scopeChanged(str).

    HOW SCOPE + SELECTION WIRE THE PANES
    ------------------------------------
      * Selecting a SCOPE (navigator list) -> set_scope(): points the tech-tree
        canvas at the scope (canvas.set_scope + refresh), re-badges the outline,
        re-renders the inspector for the current node, emits scopeChanged, and —
        because scope is driven by trackline intervals — shows the "intervals"
        editor page if the app registered one (else stays on the tech tree).
      * Selecting a NODE (canvas card OR navigator outline row) -> _select_node():
        populates the inspector (settings form + run history) for (node, scope),
        and shows the "map" editor page for a spatial product if one is registered
        (else stays on the tech tree, the default surface).
    """

    buildRequested = Signal(list)          # [{"scope_id", "node_ids"}]
    runRequested = Signal(str, str)        # (node_id, scope_id)
    exportRequested = Signal(str, str)     # (node_id, scope_id)
    scopeChanged = Signal(str)             # scope_id

    def __init__(self, workspace_dir: str = "", imported=None,
                 scope_id: str = SURVEY_SCOPE,
                 parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ws = str(workspace_dir or "")
        self._imported: set[str] = set(imported or ())
        self._scope_id = SURVEY_SCOPE if scope_id in _SURVEY_ALIASES else str(scope_id)
        self._injected_scopes: list[tuple[str, str]] = []
        self._selected: str | None = None
        self._reg = None
        self._produced: set[str] = set()
        self._available: set[str] = set()
        # editor pages by key -> stack index; page 0 is always the tech tree.
        self._editor_pages: dict[str, int] = {}
        # inspector settings fields for the current node: name -> (type, widget).
        self._insp_fields: dict[str, tuple[str, QWidget]] = {}
        # last collected settings per (node, scope), stashed so a run can read them.
        self._node_settings: dict[tuple[str, str], dict] = {}

        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        self._split = QSplitter(Qt.Horizontal)
        self._split.setChildrenCollapsible(True)
        outer.addWidget(self._split)

        self._split.addWidget(self._build_navigator())
        self._split.addWidget(self._build_editor())
        self._split.addWidget(self._build_inspector())
        # ~20% / ~55% / ~25%.
        self._split.setStretchFactor(0, 20)
        self._split.setStretchFactor(1, 55)
        self._split.setStretchFactor(2, 25)
        self._split.setSizes([260, 720, 340])

    # -- navigator -----------------------------------------------------------
    def _build_navigator(self) -> QWidget:
        panel = QWidget()
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(6, 6, 6, 6)
        lay.setSpacing(4)

        lay.addWidget(self._section_label("SCOPE"))
        self._scope_list = QListWidget()
        self._scope_list.setMaximumHeight(150)
        self._scope_list.currentRowChanged.connect(self._on_scope_row_changed)
        lay.addWidget(self._scope_list)

        self._btn_define = QPushButton("Define intervals…")
        self._btn_define.setToolTip(
            "Open the trackline / timeline interval selector to define scopes.")
        self._btn_define.clicked.connect(self._on_define_intervals)
        lay.addWidget(self._btn_define)

        lay.addSpacing(6)
        lay.addWidget(self._section_label("PRODUCTS"))
        self._outline = QTreeWidget()
        self._outline.setHeaderHidden(True)
        self._outline.setColumnCount(1)
        self._outline.itemClicked.connect(self._on_outline_item_clicked)
        lay.addWidget(self._outline, 1)
        return panel

    # -- editor --------------------------------------------------------------
    def _build_editor(self) -> QWidget:
        self._editor_stack = QStackedWidget()
        self._canvas = ProductTreeCanvas(
            workspace_dir=self._ws, imported=self._imported,
            scope_id=self._scope_id)
        # Re-emit the canvas's contracts through the workbench so the app can wire
        # ONE set of signals; every relay is a bound method (threading rule).
        self._canvas.buildRequested.connect(self._relay_build)
        self._canvas.runRequested.connect(self._relay_run)
        self._canvas.exportRequested.connect(self._relay_export)
        self._canvas.nodeSelected.connect(self._on_canvas_node_selected)
        self.addEditorPage(self._canvas, "graph")     # page 0, the default surface
        return self._editor_stack

    # -- inspector -----------------------------------------------------------
    def _build_inspector(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(280)
        panel = QWidget()
        scroll.setWidget(panel)
        lay = QVBoxLayout(panel)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        # gentle empty state, shown when nothing is selected.
        self._insp_empty = QLabel(
            "Select a product — from the tech tree or the navigator outline — to "
            "inspect its settings, run history and previews here.")
        self._insp_empty.setWordWrap(True)
        self._insp_empty.setStyleSheet("color:#888; padding:12px;")
        lay.addWidget(self._insp_empty)

        # content, hidden until a node is selected.
        self._insp_content = QWidget()
        clay = QVBoxLayout(self._insp_content)
        clay.setContentsMargins(0, 0, 0, 0)
        clay.setSpacing(6)

        self._insp_title = QLabel()
        self._insp_title.setWordWrap(True)
        self._insp_title.setStyleSheet("font-weight:bold; font-size:13px;")
        clay.addWidget(self._insp_title)
        self._insp_desc = QLabel()
        self._insp_desc.setWordWrap(True)
        self._insp_desc.setStyleSheet("color:#555;")
        clay.addWidget(self._insp_desc)

        # (1) settings form ---------------------------------------------------
        self._insp_form_box = QGroupBox("Settings")
        self._insp_form = QFormLayout(self._insp_form_box)
        clay.addWidget(self._insp_form_box)

        # (2) run history -----------------------------------------------------
        self._insp_runs_box = QGroupBox("Run history")
        self._insp_runs = QVBoxLayout(self._insp_runs_box)
        clay.addWidget(self._insp_runs_box)

        # (3) preview placeholder --------------------------------------------
        self._insp_preview = QLabel("Preview unavailable in this skeleton.")
        self._insp_preview.setWordWrap(True)
        self._insp_preview.setMinimumHeight(70)
        self._insp_preview.setAlignment(Qt.AlignCenter)
        self._insp_preview.setStyleSheet(
            "color:#999; background:#f4f4f4; border:1px dashed #ccc; padding:8px;")
        clay.addWidget(self._insp_preview)

        # actions -------------------------------------------------------------
        btn_row = QHBoxLayout()
        self._btn_build = QPushButton("▶ Build (with deps)")
        self._btn_build.setStyleSheet("font-weight:bold;")
        self._btn_build.clicked.connect(self._on_inspector_build)
        self._btn_run = QPushButton("New run")
        self._btn_run.clicked.connect(self._on_inspector_run)
        btn_row.addWidget(self._btn_build)
        btn_row.addWidget(self._btn_run)
        clay.addLayout(btn_row)
        clay.addStretch(1)

        lay.addWidget(self._insp_content, 1)
        self._insp_content.hide()
        return scroll

    @staticmethod
    def _section_label(text: str) -> QLabel:
        lbl = QLabel(text)
        lbl.setStyleSheet("color:#607d8b; font-weight:bold; font-size:10px;")
        return lbl

    # -------------------------------------------------------------- public API
    def set_workspace(self, workspace_dir: str, imported=None,
                      scope_id: str | None = None) -> None:
        """Point the workbench (and its tech tree) at a workspace + imported set."""
        self._ws = str(workspace_dir or "")
        if imported is not None:
            self._imported = set(imported)
        if scope_id is not None:
            self._scope_id = (SURVEY_SCOPE if scope_id in _SURVEY_ALIASES
                              else str(scope_id))
        self._canvas.set_workspace(self._ws, self._imported, self._scope_id)
        self.refresh()

    def set_imported(self, imported) -> None:
        """Update which data roots are imported (gates availability); re-render."""
        self._imported = set(imported or ())
        self._canvas.set_imported(self._imported)
        self.refresh()

    def set_scope(self, scope_id: str) -> None:
        """Select a scope: drive the tech tree, outline, inspector and editor page.

        Bound-slot-safe.  Emits scopeChanged and, because scope is driven by
        trackline intervals, shows the "intervals" editor page if one is registered
        (else leaves the tech tree — the default surface — in place)."""
        sid = SURVEY_SCOPE if scope_id in _SURVEY_ALIASES else str(scope_id)
        self._scope_id = sid
        self._canvas.set_scope(sid)
        self._sync_scope_selection()
        self._refresh_registry_sets()
        self._refresh_outline_badges()
        self._render_inspector()
        self.scopeChanged.emit(sid)
        # Scope is chosen against the trackline; surface it if the app gave us one.
        self.showEditorPage("intervals")

    def set_scopes(self, scopes) -> None:
        """Inject externally-defined scopes — the trackline intervals / sets the
        app derives from selection — to sit above the registry-discovered scopes.

        Each item is (scope_id, label); the whole-trackline default is always kept
        first.  The current scope selection is preserved when still present."""
        self._injected_scopes = [(str(sid), str(label)) for sid, label in scopes]
        self._rebuild_scope_list()

    def addEditorPage(self, widget: QWidget, key: str) -> None:
        """Register a swap-in editor page under a key (e.g. "intervals", "map").

        The app passes its EXISTING timeline/trackline panel as "intervals" and its
        map/3D panel as "map"; the workbench only hosts them.  Page key "graph" is
        the tech tree, registered at construction."""
        idx = self._editor_stack.addWidget(widget)
        self._editor_pages[key] = idx

    def showEditorPage(self, key: str) -> bool:
        """Show the editor page registered under ``key``; return True if it existed.

        A no-op returning False when the key was never registered — callers fall
        back to leaving the tech tree (page "graph") in view."""
        idx = self._editor_pages.get(key)
        if idx is None:
            return False
        self._editor_stack.setCurrentIndex(idx)
        return True

    def refresh(self) -> None:
        """Re-read the registry via the canvas, rebuild scope list + outline, and
        re-render the inspector for the current selection.

        Bound method: safe to call from the main window's main-thread handlers."""
        self._canvas.refresh()
        self._refresh_registry_sets()
        self._rebuild_scope_list()
        self._rebuild_outline()
        self._render_inspector()

    # --------------------------------------------------------------- internals
    def _read_registry(self):
        if not self._ws:
            return None
        try:
            from workspace_paths import PathResolver
            from manifest import Registry
            return Registry(PathResolver(self._ws).registry_json())
        except Exception:
            return None

    def _refresh_registry_sets(self) -> None:
        self._reg = self._read_registry()
        self._available = pg.available_nodes(self._imported)
        self._produced = produced_node_ids(self._reg, self._scope_id)

    # -- scope list ----------------------------------------------------------
    def _scopes(self) -> list[tuple[str, str]]:
        return assemble_scopes(self._reg, self._injected_scopes)

    def _rebuild_scope_list(self) -> None:
        scopes = self._scopes()
        # keep current scope if it's still present, else fall back to survey.
        if self._scope_id not in {sid for sid, _ in scopes}:
            self._scope_id = SURVEY_SCOPE
        self._scope_list.blockSignals(True)
        self._scope_list.clear()
        for sid, label in scopes:
            item = QListWidgetItem(label)
            item.setData(_ROLE_SCOPE, sid)
            self._scope_list.addItem(item)
        row = next((i for i, (sid, _) in enumerate(scopes)
                    if sid == self._scope_id), 0)
        self._scope_list.setCurrentRow(row)
        self._scope_list.blockSignals(False)

    def _sync_scope_selection(self) -> None:
        """Move the list highlight to match self._scope_id without re-emitting."""
        self._scope_list.blockSignals(True)
        for i in range(self._scope_list.count()):
            if self._scope_list.item(i).data(_ROLE_SCOPE) == self._scope_id:
                self._scope_list.setCurrentRow(i)
                break
        self._scope_list.blockSignals(False)

    # -- outline -------------------------------------------------------------
    def _rebuild_outline(self) -> None:
        self._outline.clear()
        for group_label, node_ids in outline_groups():
            top = QTreeWidgetItem([group_label])
            top.setFlags(Qt.ItemIsEnabled)          # group heading, not selectable
            f = top.font(0)
            f.setBold(True)
            top.setFont(0, f)
            self._outline.addTopLevelItem(top)
            for nid in node_ids:
                child = QTreeWidgetItem([""])
                child.setData(0, _ROLE_NODE, nid)
                top.addChild(child)
            top.setExpanded(True)
        self._refresh_outline_badges()

    def _refresh_outline_badges(self) -> None:
        """Repaint each outline row's "label  ·  badge" for the current scope."""
        for i in range(self._outline.topLevelItemCount()):
            top = self._outline.topLevelItem(i)
            for j in range(top.childCount()):
                child = top.child(j)
                nid = child.data(0, _ROLE_NODE)
                if not nid:
                    continue
                node = pg.get_node(nid)
                badge = node_badge(nid, self._produced, self._available,
                                   self._imported)
                child.setText(0, f"{badge}  {node.label}")

    # ------------------------------------------------------------------ events
    def _on_scope_row_changed(self, row: int) -> None:
        if row < 0:
            return
        item = self._scope_list.item(row)
        sid = item.data(_ROLE_SCOPE) if item is not None else None
        if sid is not None and sid != self._scope_id:
            self.set_scope(sid)

    def _on_outline_item_clicked(self, item: QTreeWidgetItem, _col: int = 0) -> None:
        nid = item.data(0, _ROLE_NODE)
        if nid:
            self._select_node(nid)

    def _on_canvas_node_selected(self, node_id: str, _scope_id: str) -> None:
        # The canvas already opens its own per-node dialog; we just mirror the
        # selection into the persistent inspector and swap the editor page.
        self._select_node(node_id, from_canvas=True)

    def _on_define_intervals(self) -> None:
        """The 'Define intervals' affordance: surface the trackline/timeline page."""
        self.showEditorPage("intervals")

    def _select_node(self, node_id: str, *, from_canvas: bool = False) -> None:
        """Populate the inspector for (node_id, current scope) and pick the page.

        A spatial product shows the "map" editor page if registered; anything else
        leaves/returns to the tech tree (the default surface).  When the click came
        from the canvas the graph is already in view, so we don't fight it."""
        self._selected = node_id
        self._render_inspector()
        if is_spatial_node(node_id):
            if not self.showEditorPage("map") and not from_canvas:
                self.showEditorPage("graph")
        elif not from_canvas:
            self.showEditorPage("graph")

    # ---------------------------------------------------------- inspector body
    def _render_inspector(self) -> None:
        nid = self._selected
        if not nid:
            self._insp_content.hide()
            self._insp_empty.show()
            return
        self._insp_empty.hide()
        self._insp_content.show()

        node = pg.get_node(nid)
        self._insp_title.setText(
            f"{node.label}  ·  {scope_display_label(self._scope_id)}")
        self._insp_desc.setText(node_description(node))
        self._build_settings_form(node)
        self._build_run_history(node)

        runnable = node.task_type is not None and nid in self._available
        self._btn_build.setEnabled(nid in self._available and not node.is_root)
        self._btn_run.setEnabled(runnable)

    def _build_settings_form(self, node) -> None:
        self._clear_layout(self._insp_form)
        self._insp_fields = {}
        if node.is_root:
            self._insp_form.addRow(QLabel("<i>Imported data root — no run "
                                          "options.</i>"))
            return
        if not node.params:
            self._insp_form.addRow(QLabel("<i>No configurable options for this "
                                          "node.</i>"))
            return
        for name, spec in node.params.items():
            # Reuse the tech-tree dialog's field builder (bool→check, choice→combo,
            # int→spin, float→doublespin, else lineedit) — not reimplemented here.
            w = _NodeSettingsDialog._make_field(self, spec)
            self._insp_fields[name] = (spec.get("type", "str"), w)
            self._insp_form.addRow(spec.get("label", name), w)

    def _collect_settings(self) -> dict:
        """Current inspector form values as {param_name: value} (same reading rules
        as the tech-tree dialog's collect_settings)."""
        out: dict = {}
        for name, (_t, w) in self._insp_fields.items():
            if isinstance(w, QCheckBox):
                out[name] = w.isChecked()
            elif isinstance(w, QComboBox):
                out[name] = w.currentText()
            elif isinstance(w, (QSpinBox, QDoubleSpinBox)):
                out[name] = w.value()
            elif isinstance(w, QLineEdit):
                out[name] = w.text()
        return out

    def _build_run_history(self, node) -> None:
        self._clear_layout(self._insp_runs)
        runs = (runs_for_node(self._reg, node.id, self._scope_id)
                if node.task_type else [])
        self._insp_runs_box.setTitle(f"Run history ({len(runs)})")
        if not runs:
            msg = ("Aggregate node — no run of its own."
                   if node.task_type is None and not node.is_root
                   else "No produced run for this scope yet.")
            lbl = QLabel(f"<i>{msg}</i>")
            lbl.setStyleSheet("color:#888;")
            self._insp_runs.addWidget(lbl)
            return
        for run in runs[:20]:
            self._insp_runs.addLayout(self._run_row(run))

    def _run_row(self, run: dict) -> QHBoxLayout:
        row = QHBoxLayout()
        when = (run.get("finished_at") or "").replace("T", " ")[:19]
        lbl = QLabel(f"{run.get('run_id', '')} · {when} · {run.get('status', '')}")
        lbl.setStyleSheet("font-size:11px;")
        row.addWidget(lbl, 1)
        btn_view = QPushButton("View")
        btn_view.setProperty(_PROP_RUN, run)
        btn_view.clicked.connect(self._on_run_view)
        row.addWidget(btn_view)
        btn_export = QPushButton("Export")
        btn_export.setProperty(_PROP_RUN, run)
        btn_export.clicked.connect(self._on_run_export)
        row.addWidget(btn_export)
        return row

    @staticmethod
    def _clear_layout(layout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()
                continue
            child = item.layout()
            if child is not None:
                Workbench._clear_layout(child)
                child.deleteLater()

    # ------------------------------------------------------------------ actions
    def _stash_settings(self) -> None:
        if self._selected:
            self._node_settings[(self._selected, self._scope_id)] = \
                self._collect_settings()

    def _on_inspector_build(self) -> None:
        """Emit buildRequested for the selected node + its unproduced deps."""
        nid = self._selected
        if not nid:
            return
        self._stash_settings()
        ordered = pg.expand_targets([nid], satisfied=self._produced)
        self.buildRequested.emit(
            [{"scope_id": self._scope_id, "node_ids": ordered}])

    def _on_inspector_run(self) -> None:
        """Emit runRequested for a fresh single run of the selected node."""
        nid = self._selected
        if not nid:
            return
        self._stash_settings()
        self.runRequested.emit(nid, self._scope_id)

    def _on_run_view(self) -> None:
        run = self.sender().property(_PROP_RUN)
        out = (run or {}).get("output_dir")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _on_run_export(self) -> None:
        if self._selected:
            self.exportRequested.emit(self._selected, self._scope_id)
        self._on_run_view()

    # -- relays from the hosted canvas (bound methods; threading rule) -------
    def _relay_build(self, plans: list) -> None:
        self.buildRequested.emit(plans)

    def _relay_run(self, node_id: str, scope_id: str) -> None:
        self.runRequested.emit(node_id, scope_id)

    def _relay_export(self, node_id: str, scope_id: str) -> None:
        self.exportRequested.emit(node_id, scope_id)
