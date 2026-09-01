"""
product_tree_widget.py — the checkable Product Tree, the centrepiece of the
redesigned "pick what you want, the stack composes itself" workflow.

Two layers, deliberately split (mirrors product_browser.py):

  * A Qt-FREE, tested core — ``produced_node_ids`` / ``runs_for_node`` /
    ``node_description`` — that reads the manifest Registry and answers "which
    product nodes have a completed run for this scope?".  All the graph math
    (availability gating, target expansion, status badges) already lives in
    product_graph.py; this module only *renders* it and calls it.

  * ``ProductTreeWidget`` — a QWidget that draws PRODUCT_GRAPH as a checkable
    QTreeWidget rooted in the data roots (video / nav / sensors), greys out
    nodes whose data has not been imported, badges each node produced / available
    / unavailable from the registry, and, from the user's checked selection,
    previews and emits the ordered node list to build (``expand_targets``).

Signals (all carry plain data, so wiring is Qt-free on the receiving side):

    buildRequested(list)   ordered node ids to run (expand_targets output), after
                           pruning already-produced ancestors.
    runRequested(str)      a single node id the user asked to (re)run from its
                           info panel ("New run…").  The param dialog is a
                           follow-up; the id + node.params are enough to build it.
    exportRequested(str)   a node id whose existing product the user wants to
                           export/reveal (also revealed directly here as a
                           convenience so the widget is useful standalone).

THREADING.  Every slot connected to a signal here is a BOUND METHOD of this
QObject, per the app's threading rule.  refresh() is a plain public method the
main window calls directly from its own (main-thread) bound slots after a
workspace load or a stack run finishes — no cross-thread signal is introduced.
"""

from __future__ import annotations

from pathlib import Path

import product_graph as pg


# ===========================================================================
# Qt-free core (unit-tested in tests/test_product_tree.py)
# ===========================================================================
# The survey / full-dataset scope goes by two names across the codebase: the
# planner writes "full" (main_window._build_*_config default), while the bundle
# layout and product_browser call it "survey".  Treat them as one scope so a run
# recorded under either name lights up the tree.
_SURVEY_ALIASES = {"full", "survey", "", None}


def _scope_matches(run_scope, scope_id) -> bool:
    """True if a registry entry's scope_id belongs to the scope being viewed.

    ``scope_id is None`` means "any scope".  The survey scope is matched by
    either of its aliases ("full" / "survey"); a job scope matches exactly.
    """
    if scope_id is None:
        return True
    if scope_id in _SURVEY_ALIASES:
        return run_scope in _SURVEY_ALIASES
    return run_scope == scope_id


def produced_node_ids(registry, scope_id=None) -> set[str]:
    """Ids of product nodes that have a COMPLETED run for ``scope_id``.

    ``registry`` is a manifest.Registry (anything exposing ``.runs``).  A node
    counts as produced when the registry holds a completed run whose ``task_type``
    equals the node's ``task_type`` in this scope (channel is ignored, so any one
    channel of ``sensor_raster`` marks the node produced).  Pure and decoupled:
    the widget hands the resulting set to product_graph.node_status /
    expand_targets.

    NOTE on photogrammetry.  ``alignment``, ``mesh``, ``orthomosaic``, ``dem`` and
    ``dense`` all share the ``photogrammetry`` task type, so a single completed
    photogrammetry run marks every one of them produced.  That is intentionally
    coarse for this first version — the task-type match is the contract — and can
    be sharpened later by inspecting each run's ``target`` build flags.
    """
    produced_types: set[str] = set()
    for entry in getattr(registry, "runs", []) or []:
        if entry.get("status") != "completed":
            continue
        if not _scope_matches(entry.get("scope_id"), scope_id):
            continue
        tt = entry.get("task_type")
        if tt:
            produced_types.add(tt)
    return {
        n.id for n in pg.all_nodes()
        if n.task_type is not None and n.task_type in produced_types
    }


def runs_for_node(registry, node_id: str, scope_id=None) -> list[dict]:
    """Completed registry entries that produced ``node_id`` in ``scope_id``.

    Newest first (by finished_at, then run_id).  Empty for aggregate/root nodes
    (task_type is None) and for nodes with no completed run.  Each entry is the
    plain registry dict (run_id, finished_at, status, output_dir, …).
    """
    node = pg.get_node(node_id)
    if node.task_type is None:
        return []
    out = [
        entry for entry in (getattr(registry, "runs", []) or [])
        if entry.get("status") == "completed"
        and entry.get("task_type") == node.task_type
        and _scope_matches(entry.get("scope_id"), scope_id)
    ]
    out.sort(key=lambda r: (r.get("finished_at") or "", r.get("run_id") or ""),
             reverse=True)
    return out


def node_description(node) -> str:
    """A short, human sentence describing a product node.

    Synthesised from the graph (Node has no free-text field): what it is, what it
    depends on, and which task type produces it.  Kept Qt-free so it can be tested
    and reused by any view.
    """
    parts: list[str] = []
    if node.is_root:
        parts.append(f"Imported data root: {node.label.lower()}.")
        return " ".join(parts)
    if node.requires:
        req = ", ".join(pg.get_node(r).label for r in node.requires)
        parts.append(f"Requires: {req}.")
    needs = sorted(pg.transitive_needs(node.id))
    if needs:
        parts.append("Depends on imported " + ", ".join(needs) + ".")
    if node.task_type:
        parts.append(f"Runs as the '{node.task_type}' task.")
    else:
        parts.append("Aggregate product — selecting it pulls in its "
                     "dependencies, which do the real work.")
    if node.is_intermediate:
        parts.append("Intermediate (usually built on the way to another product,"
                     " but selectable on its own).")
    if node.is_archived:
        parts.append("Archived — retired from the default workflow.")
    return " ".join(parts)


# ===========================================================================
# Qt widget
# ===========================================================================
from PySide6.QtCore import Qt, Signal, QUrl
from PySide6.QtGui import QBrush, QColor, QDesktopServices, QFont
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSplitter,
    QTextEdit,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

# Badge glyphs for the status column.
_BADGE = {"produced": "✓ produced", "available": "○ available",
          "unavailable": "⨯ unavailable"}
_BADGE_COLOR = {"produced": QColor("#2e7d32"), "available": QColor("#1565c0"),
                "unavailable": QColor("#9e9e9e")}
_ROLE_NODE_ID = Qt.UserRole + 1


class ProductTreeWidget(QWidget):
    """Checkable dependency tree: pick products, preview + emit the build order."""

    buildRequested = Signal(list)      # ordered node ids (expand_targets output)
    runRequested   = Signal(str)       # node id to (re)run from its info panel
    exportRequested = Signal(str)      # node id whose product to export/reveal

    def __init__(self, workspace_dir: str = "", imported=None,
                 scope_id: str = "full", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ws = str(workspace_dir or "")
        self._imported: set[str] = set(imported or ())
        self._scope_id = scope_id
        self._produced: set[str] = set()
        self._available: set[str] = set()
        self._items: dict[str, QTreeWidgetItem] = {}
        self._building = False          # guard against itemChanged recursion
        self._current_node: str | None = None
        self._build_ui()
        self.refresh()

    # ------------------------------------------------------------------ UI
    def _build_ui(self) -> None:
        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 6, 6, 6)
        outer.setSpacing(4)

        header = QLabel("Products")
        f = QFont(); f.setBold(True)
        header.setFont(f)
        outer.addWidget(header)
        hint = QLabel("Check the products you want. The task stack composes "
                      "itself from the dependency tree.")
        hint.setStyleSheet("color: #888; font-size: 10px;")
        hint.setWordWrap(True)
        outer.addWidget(hint)

        split = QSplitter(Qt.Vertical)
        outer.addWidget(split, 1)

        # -- tree ------------------------------------------------------------
        self._tree = QTreeWidget()
        self._tree.setHeaderLabels(["Product", "Status"])
        self._tree.setColumnWidth(0, 240)
        self._tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self._tree.setAlternatingRowColors(True)
        self._tree.itemChanged.connect(self._on_item_changed)
        self._tree.currentItemChanged.connect(self._on_current_changed)
        split.addWidget(self._tree)

        # -- info panel ------------------------------------------------------
        info = QWidget()
        ilay = QVBoxLayout(info)
        ilay.setContentsMargins(0, 0, 0, 0)
        self._info_title = QLabel("Select a product")
        self._info_title.setStyleSheet("font-weight: bold; font-size: 13px;")
        self._info_title.setWordWrap(True)
        ilay.addWidget(self._info_title)
        self._info = QTextEdit()
        self._info.setReadOnly(True)
        ilay.addWidget(self._info, 1)
        btn_row = QHBoxLayout()
        self._btn_export = QPushButton("Export product…")
        self._btn_export.clicked.connect(self._export_current)
        self._btn_run = QPushButton("New run…")
        self._btn_run.clicked.connect(self._run_current)
        for b in (self._btn_export, self._btn_run):
            b.setEnabled(False)
            btn_row.addWidget(b)
        btn_row.addStretch(1)
        ilay.addLayout(btn_row)
        split.addWidget(info)
        split.setSizes([420, 220])

        # -- build row -------------------------------------------------------
        self._preview = QLabel("Check products to build.")
        self._preview.setWordWrap(True)
        self._preview.setStyleSheet(
            "color: #333; font-size: 11px; background: #f4f4f4; "
            "border: 1px solid #ddd; padding: 5px;")
        outer.addWidget(self._preview)

        self._btn_build = QPushButton("▶  Build selected")
        self._btn_build.setStyleSheet("font-weight: bold; padding: 6px;")
        self._btn_build.clicked.connect(self._emit_build)
        self._btn_build.setEnabled(False)
        outer.addWidget(self._btn_build)

    # -------------------------------------------------------------- public API
    def set_workspace(self, workspace_dir: str, imported=None,
                      scope_id: str | None = None) -> None:
        """Point the tree at a workspace + imported-data set and re-render."""
        self._ws = str(workspace_dir or "")
        if imported is not None:
            self._imported = set(imported)
        if scope_id is not None:
            self._scope_id = scope_id
        self.refresh()

    def set_imported(self, imported) -> None:
        """Update which data roots are imported (gates availability) and re-render."""
        self._imported = set(imported or ())
        self.refresh()

    def refresh(self) -> None:
        """Re-read the registry and rebuild badges + availability + preview.

        Cheap (a JSON read + set math), so the main window may call it freely
        after a workspace load or a stack run.  Preserves the checked selection
        by node id across the rebuild.
        """
        self._available = pg.available_nodes(self._imported)
        self._produced = self._read_produced()
        checked = self.checked_ids()
        self._rebuild_tree(preserve_checked=checked)
        self._update_preview()
        self._render_info(self._current_node)

    def checked_ids(self) -> list[str]:
        """Currently checked product node ids, in tree order."""
        out: list[str] = []
        for nid, item in self._items.items():
            if item.flags() & Qt.ItemIsUserCheckable and item.checkState(0) == Qt.Checked:
                out.append(nid)
        # Stable, dependency-respecting order comes out of expand_targets; here we
        # just need the raw set — return in declaration order for determinism.
        order = {n.id: i for i, n in enumerate(pg.all_nodes())}
        out.sort(key=lambda x: order.get(x, 0))
        return out

    # --------------------------------------------------------------- internals
    def _read_produced(self) -> set[str]:
        if not self._ws:
            return set()
        try:
            from workspace_paths import PathResolver
            from manifest import Registry
            reg = Registry(PathResolver(self._ws).registry_json())
        except Exception:
            return set()
        return produced_node_ids(reg, self._scope_id)

    def _read_registry(self):
        if not self._ws:
            return None
        try:
            from workspace_paths import PathResolver
            from manifest import Registry
            return Registry(PathResolver(self._ws).registry_json())
        except Exception:
            return None

    def _rebuild_tree(self, preserve_checked=()) -> None:
        preserve = set(preserve_checked)
        self._building = True
        try:
            self._tree.clear()
            self._items = {}
            visited: set[str] = set()
            for root in pg.data_root_nodes():
                item = self._make_item(root)
                self._tree.addTopLevelItem(item)
                self._add_children(item, root.id, visited, preserve)
                item.setExpanded(True)
        finally:
            self._building = False

    def _add_children(self, parent_item, parent_id, visited, preserve) -> None:
        for node in pg.all_nodes():
            if parent_id not in node.requires or node.id in visited:
                continue
            visited.add(node.id)
            item = self._make_item(node, preserve)
            parent_item.addChild(item)
            item.setExpanded(True)
            self._add_children(item, node.id, visited, preserve)

    def _make_item(self, node, preserve=()) -> QTreeWidgetItem:
        item = QTreeWidgetItem([node.label, ""])
        item.setData(0, _ROLE_NODE_ID, node.id)
        self._items[node.id] = item

        if node.is_root:
            # Roots are imported, never run: bold header, availability badge, no
            # checkbox.
            f = item.font(0); f.setBold(True); item.setFont(0, f)
            badge = ("✓ imported" if node.id in self._imported else "○ not imported")
            item.setText(1, badge)
            item.setForeground(1, QBrush(
                _BADGE_COLOR["produced" if node.id in self._imported else "unavailable"]))
            item.setToolTip(0, node_description(node))
            return item

        status = pg.node_status(node.id, self._produced, self._available)
        item.setText(1, _BADGE.get(status, status))
        item.setForeground(1, QBrush(_BADGE_COLOR.get(status, QColor("#000"))))

        base_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if status == "unavailable":
            # Disabled (immutable) checkbox + reason tooltip.
            item.setFlags(base_flags)
            item.setCheckState(0, Qt.Unchecked)
            reason = pg.unavailable_reason(node.id, self._imported) or "unavailable"
            item.setToolTip(0, reason)
            item.setForeground(0, QBrush(QColor("#9e9e9e")))
        else:
            item.setFlags(base_flags | Qt.ItemIsUserCheckable)
            item.setCheckState(0, Qt.Checked if node.id in preserve else Qt.Unchecked)
            item.setToolTip(0, node_description(node))
            if node.is_archived:
                item.setForeground(0, QBrush(QColor("#9e9e9e")))
                fnt = item.font(0); fnt.setItalic(True); item.setFont(0, fnt)
        return item

    # ------------------------------------------------------------------ events
    def _on_item_changed(self, item, column) -> None:
        if self._building or column != 0:
            return
        self._update_preview()

    def _on_current_changed(self, current, _previous) -> None:
        nid = current.data(0, _ROLE_NODE_ID) if current else None
        self._current_node = nid
        self._render_info(nid)

    # ------------------------------------------------------------------ preview
    def _plan(self) -> list[str]:
        return pg.expand_targets(self.checked_ids(), satisfied=self._produced)

    def _update_preview(self) -> None:
        checked = self.checked_ids()
        self._btn_build.setEnabled(bool(checked))
        if not checked:
            self._preview.setText("Check products to build.")
            return
        ordered = self._plan()
        needed: set[str] = set()
        for c in checked:
            needed |= pg.ancestors(c) | {c}
        needed -= pg.DATA_ROOTS
        skipped = sorted((needed & self._produced),
                         key=lambda x: [n.id for n in pg.all_nodes()].index(x))
        want = ", ".join(pg.get_node(c).label for c in checked)
        if ordered:
            run = " → ".join(pg.get_node(o).label for o in ordered)
            msg = f"To build {want}, we'll run:  {run}"
        else:
            msg = f"{want}: everything needed is already produced — nothing to run."
        if skipped:
            msg += ("   (skipping, already produced: "
                    + ", ".join(pg.get_node(s).label for s in skipped) + ")")
        self._preview.setText(msg)

    # --------------------------------------------------------------- info panel
    def _render_info(self, nid) -> None:
        if not nid:
            self._info_title.setText("Select a product")
            self._info.setHtml("")
            self._btn_export.setEnabled(False)
            self._btn_run.setEnabled(False)
            return
        node = pg.get_node(nid)
        self._info_title.setText(node.label)
        runs = runs_for_node(self._read_registry(), nid, self._scope_id) \
            if node.task_type else []

        html = ["<p style='color:#444'>" + node_description(node) + "</p>"]

        if node.params:
            html.append("<p style='color:#666'><b>Default run parameters</b></p>"
                        "<table cellspacing='3'>")
            for key, spec in node.params.items():
                label = spec.get("label", key)
                default = spec.get("default")
                html.append(f"<tr><td style='color:#666'>{label}</td>"
                            f"<td><code>{default}</code></td></tr>")
            html.append("</table>")

        if node.task_type is None:
            html.append("<p style='color:#888'><i>Aggregate node — no run of its "
                        "own.</i></p>")
        elif runs:
            html.append(f"<p style='color:#2e7d32'><b>{len(runs)} produced "
                        f"run(s) for this scope</b></p><table cellspacing='3'>")
            html.append("<tr><td style='color:#666'><b>Run</b></td>"
                        "<td style='color:#666'><b>Finished</b></td>"
                        "<td style='color:#666'><b>Status</b></td></tr>")
            for r in runs[:12]:
                when = (r.get("finished_at") or "").replace("T", " ")[:19]
                html.append(f"<tr><td><code>{r.get('run_id','')}</code></td>"
                            f"<td>{when}</td><td>{r.get('status','')}</td></tr>")
            html.append("</table>")
            latest = runs[0]
            html.append(f"<p style='color:#888'>Output: "
                        f"<code>{latest.get('output_dir','')}</code></p>")
        else:
            html.append("<p style='color:#888'><i>No produced run for this scope "
                        "yet.</i></p>")

        self._info.setHtml("".join(html))
        self._btn_run.setEnabled(node.task_type is not None
                                 and nid in self._available)
        self._btn_export.setEnabled(bool(runs))

    # ------------------------------------------------------------------ actions
    def _export_current(self) -> None:
        nid = self._current_node
        if not nid:
            return
        self.exportRequested.emit(nid)
        # Convenience: reveal the latest produced run's output directory so the
        # button does something useful even before the host wires the browser.
        runs = runs_for_node(self._read_registry(), nid, self._scope_id)
        if runs:
            out = runs[0].get("output_dir")
            if out and Path(out).exists():
                QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _run_current(self) -> None:
        if self._current_node:
            self.runRequested.emit(self._current_node)

    def _emit_build(self) -> None:
        ordered = self._plan()
        self.buildRequested.emit(ordered)
