"""
product_tree_widget.py — the checkable Product Tree, the centrepiece of the
redesigned "pick what you want, the stack composes itself" workflow.

Two layers, deliberately split (mirrors product_browser.py):

  * A Qt-FREE, tested core — ``produced_node_ids`` / ``runs_for_node`` /
    ``node_description`` / ``scope_display_label`` — that reads the manifest
    Registry and answers "which product nodes have a completed run for this
    scope?".  All the graph math (availability gating, target expansion, status
    badges, the chunked scope-relation) lives in product_graph.py; this module
    only *renders* it and calls it.

  * ``ProductTreeWidget`` — a QWidget that draws PRODUCT_GRAPH as a checkable
    QTreeWidget.  V2 adds a SCOPE AXIS:

        Imports (video / nav / sensors)      ← bold, no checkbox, gate availability
        Survey (whole dive)                  ← default scope branch (scope "survey")
          └ the product DAG, computed for this scope
        <job_2 / named interval / …>         ← one branch per non-survey scope
          └ the same DAG, computed for that scope

    A CHUNKED node (the photogrammetry branch) fans a scope into per-chunk runs:
    under a scope it is one checkable leaf, but when the registry holds more than
    one completed run of it for that scope, those runs are shown as expandable
    child leaves (one per chunk) — each individually inspectable + exportable.

Signals (all carry plain data, so wiring is Qt-free on the receiving side):

    buildRequested(list)   per-scope build plans:
                             [{"scope_id": str, "node_ids": [ordered ids]}, …]
                           where node_ids is expand_targets(checked-for-that-scope,
                           satisfied=produced-for-that-scope).
    runRequested(str, str)   (node_id, scope_id) to (re)run from an info panel.
    exportRequested(str, str) (node_id, scope_id) whose product to export/reveal
                           (also revealed directly here as a convenience).

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
SURVEY_SCOPE = "survey"
SURVEY_LABEL = "Survey (whole dive)"


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


def scope_display_label(scope_id) -> str:
    """Human label for a scope id ("job_2" → "Job 2", "survey" → the survey label)."""
    if scope_id in _SURVEY_ALIASES:
        return SURVEY_LABEL
    s = str(scope_id)
    if s.startswith("job_"):
        tail = s[len("job_"):]
        return f"Job {tail.lstrip('0') or tail}" if tail.isdigit() else f"Job {tail}"
    if s.startswith("job"):
        return s.replace("_", " ").title()
    return s.replace("_", " ")


def registry_scopes(registry) -> list[str]:
    """Distinct NON-survey scope ids that appear in the registry's runs.

    Preserves first-seen order; the survey scope is never included (it is always
    rendered as the default branch by the widget).
    """
    out: list[str] = []
    seen: set = set()
    for entry in getattr(registry, "runs", []) or []:
        sid = entry.get("scope_id")
        if sid in _SURVEY_ALIASES or sid in seen:
            continue
        seen.add(sid)
        out.append(sid)
    return out


def produced_node_ids(registry, scope_id=None) -> set[str]:
    """Ids of product nodes that have a COMPLETED run for ``scope_id``.

    ``registry`` is a manifest.Registry (anything exposing ``.runs``).  A node
    counts as produced when the registry holds a completed run whose ``task_type``
    equals the node's ``task_type`` in this scope (channel is ignored, so any one
    channel of ``sensor_raster`` marks the node produced).  Pure and decoupled:
    the widget hands the resulting set to product_graph.node_status /
    expand_targets, PER SCOPE.

    NOTE on photogrammetry.  ``alignment``, ``mesh``, ``orthomosaic``, ``dem`` and
    ``dense`` all share the ``photogrammetry`` task type, so a single completed
    photogrammetry run marks every one of them produced.  That is intentionally
    coarse for this version — the task-type match is the contract — and is what
    makes the per-chunk fan (multiple runs → multiple leaves) the more precise
    signal for the photogrammetry branch.
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
    plain registry dict (run_id, finished_at, status, output_dir, …).  For a
    chunked node this is exactly the per-chunk run list the tree fans out.
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
    if getattr(node, "chunked", False):
        parts.append("Chunked — produced per video chunk within a scope; each "
                     "chunk is an independently inspectable run.")
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
_ROLE_SCOPE_ID = Qt.UserRole + 2
_ROLE_KIND = Qt.UserRole + 3          # "import" | "scope" | "product" | "chunk"
_ROLE_RUN = Qt.UserRole + 4           # the run dict, for chunk leaves


class ProductTreeWidget(QWidget):
    """Scope-branched checkable dependency tree: pick products per scope, preview
    + emit the per-scope build order."""

    buildRequested = Signal(list)          # [{"scope_id", "node_ids"}, …]
    runRequested   = Signal(str, str)      # (node_id, scope_id)
    exportRequested = Signal(str, str)     # (node_id, scope_id)

    def __init__(self, workspace_dir: str = "", imported=None,
                 scope_id: str = SURVEY_SCOPE, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._ws = str(workspace_dir or "")
        self._imported: set[str] = set(imported or ())
        # scope_id kept only for backward-compat construction; the tree now shows
        # every scope as its own branch.
        self._scopes_override: list[tuple[str, str]] | None = None
        self._available: set[str] = set()
        self._produced_by_scope: dict[str, set[str]] = {}
        self._reg = None                    # registry snapshot for this refresh
        # (scope_id, node_id) -> checkable product leaf item
        self._items: dict[tuple[str, str], QTreeWidgetItem] = {}
        self._building = False              # guard against itemChanged recursion
        self._current: QTreeWidgetItem | None = None
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
        hint = QLabel("Check products under a scope (Survey, or a named interval)."
                      " The task stack composes itself from the dependency tree.")
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
        split.setSizes([440, 220])

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
        self.refresh()

    def set_imported(self, imported) -> None:
        """Update which data roots are imported (gates availability) and re-render."""
        self._imported = set(imported or ())
        self.refresh()

    def set_scopes(self, scopes) -> None:
        """Inject the non-survey scope branches to show, as (scope_id, label) pairs.

        The Survey branch is always rendered first regardless.  Passing None
        reverts to deriving the extra scopes from the registry's run scope ids.
        """
        self._scopes_override = None if scopes is None else [
            (str(sid), str(lbl)) for sid, lbl in scopes
            if sid not in _SURVEY_ALIASES]
        self.refresh()

    def refresh(self) -> None:
        """Re-read the registry and rebuild every scope branch (badges,
        availability, per-scope produced sets, chunk fans, preview).

        Bound method: safe to call from the main window's main-thread handlers.
        Preserves the checked (node, scope) selection across the rebuild.
        """
        self._reg = self._read_registry()
        self._available = pg.available_nodes(self._imported)
        checked = set(self.checked_ids())
        scopes = self._scope_list()
        self._produced_by_scope = {
            sid: produced_node_ids(self._reg, sid) for sid, _ in scopes}
        self._rebuild_tree(scopes, preserve_checked=checked)
        self._update_preview()
        self._render_info(self._current)

    def checked_ids(self) -> list[tuple[str, str]]:
        """Currently checked (node_id, scope_id) targets.

        A node checked under Survey and the same node checked under an interval
        are DISTINCT targets, so selection is keyed by (node, scope).  Ordered by
        scope (as rendered) then node declaration order for determinism.
        """
        node_order = {n.id: i for i, n in enumerate(pg.all_nodes())}
        out: list[tuple[str, str]] = []
        for (scope_id, node_id), item in self._items.items():
            if (item.flags() & Qt.ItemIsUserCheckable
                    and item.checkState(0) == Qt.Checked):
                out.append((node_id, scope_id))
        scope_rank = {sid: i for i, (sid, _) in enumerate(self._scope_list())}
        out.sort(key=lambda t: (scope_rank.get(t[1], 99), node_order.get(t[0], 0)))
        return out

    # --------------------------------------------------------------- internals
    def _scope_list(self) -> list[tuple[str, str]]:
        """[(scope_id, label)] to render: Survey first, then extra scopes."""
        scopes = [(SURVEY_SCOPE, SURVEY_LABEL)]
        if self._scopes_override is not None:
            scopes.extend(self._scopes_override)
        else:
            for sid in registry_scopes(self._reg):
                scopes.append((sid, scope_display_label(sid)))
        return scopes

    def _read_registry(self):
        if not self._ws:
            return None
        try:
            from workspace_paths import PathResolver
            from manifest import Registry
            return Registry(PathResolver(self._ws).registry_json())
        except Exception:
            return None

    def _rebuild_tree(self, scopes, preserve_checked=()) -> None:
        preserve = set(preserve_checked)
        self._building = True
        try:
            self._tree.clear()
            self._items = {}
            self._make_import_rows()
            for scope_id, label in scopes:
                self._add_scope_branch(scope_id, label, preserve)
        finally:
            self._building = False

    def _make_import_rows(self) -> None:
        for root in pg.data_root_nodes():
            imported = root.id in self._imported
            item = QTreeWidgetItem([root.label,
                                    "✓ imported" if imported else "○ not imported"])
            item.setData(0, _ROLE_KIND, "import")
            item.setData(0, _ROLE_NODE_ID, root.id)
            fnt = item.font(0); fnt.setBold(True); item.setFont(0, fnt)
            item.setForeground(1, QBrush(
                _BADGE_COLOR["produced" if imported else "unavailable"]))
            item.setToolTip(0, node_description(root))
            self._tree.addTopLevelItem(item)

    def _add_scope_branch(self, scope_id, label, preserve) -> None:
        branch = QTreeWidgetItem([label, ""])
        branch.setData(0, _ROLE_KIND, "scope")
        branch.setData(0, _ROLE_SCOPE_ID, scope_id)
        fnt = branch.font(0); fnt.setBold(True); branch.setFont(0, fnt)
        branch.setForeground(0, QBrush(QColor("#4527a0")))
        self._tree.addTopLevelItem(branch)
        branch.setExpanded(True)
        # The product DAG's entry points are the product children of the data
        # roots; render each product node once per scope branch (dedup by id).
        visited: set[str] = set()
        for root in pg.data_root_nodes():
            self._add_products(branch, root.id, scope_id, visited, preserve)

    def _add_products(self, parent_item, parent_id, scope_id, visited, preserve) -> None:
        for node in pg.all_nodes():
            if parent_id not in node.requires or node.id in visited:
                continue
            visited.add(node.id)
            item = self._make_product_item(node, scope_id, preserve)
            parent_item.addChild(item)
            item.setExpanded(True)
            self._add_products(item, node.id, scope_id, visited, preserve)

    def _make_product_item(self, node, scope_id, preserve=()) -> QTreeWidgetItem:
        produced = self._produced_by_scope.get(scope_id, set())
        item = QTreeWidgetItem([node.label, ""])
        item.setData(0, _ROLE_KIND, "product")
        item.setData(0, _ROLE_NODE_ID, node.id)
        item.setData(0, _ROLE_SCOPE_ID, scope_id)
        self._items[(scope_id, node.id)] = item

        status = pg.node_status(node.id, produced, self._available)
        item.setText(1, _BADGE.get(status, status))
        item.setForeground(1, QBrush(_BADGE_COLOR.get(status, QColor("#000"))))

        base_flags = Qt.ItemIsSelectable | Qt.ItemIsEnabled
        if status == "unavailable":
            item.setFlags(base_flags)          # immutable (disabled) checkbox
            item.setCheckState(0, Qt.Unchecked)
            reason = pg.unavailable_reason(node.id, self._imported) or "unavailable"
            item.setToolTip(0, reason)
            item.setForeground(0, QBrush(QColor("#9e9e9e")))
        else:
            item.setFlags(base_flags | Qt.ItemIsUserCheckable)
            item.setCheckState(
                0, Qt.Checked if (scope_id, node.id) in preserve else Qt.Unchecked)
            item.setToolTip(0, node_description(node))
            if node.is_archived:
                item.setForeground(0, QBrush(QColor("#9e9e9e")))
                fnt = item.font(0); fnt.setItalic(True); item.setFont(0, fnt)

        # Chunked node with >1 completed run for this scope: fan out per-chunk
        # child leaves (each = one run = one chunk), inspectable + exportable.
        if getattr(node, "chunked", False):
            runs = runs_for_node(self._reg, node.id, scope_id)
            if len(runs) > 1:
                for i, run in enumerate(runs, 1):
                    self._add_chunk_child(item, node, scope_id, run, i)
        return item

    def _add_chunk_child(self, parent_item, node, scope_id, run, index) -> None:
        when = (run.get("finished_at") or "").replace("T", " ")[:19]
        rid = run.get("run_id", "")
        leaf = QTreeWidgetItem([f"Chunk {index} · {rid}", "✓ produced"])
        leaf.setData(0, _ROLE_KIND, "chunk")
        leaf.setData(0, _ROLE_NODE_ID, node.id)
        leaf.setData(0, _ROLE_SCOPE_ID, scope_id)
        leaf.setData(0, _ROLE_RUN, run)
        leaf.setFlags(Qt.ItemIsSelectable | Qt.ItemIsEnabled)   # not a build target
        leaf.setForeground(1, QBrush(_BADGE_COLOR["produced"]))
        leaf.setToolTip(0, f"{node.label} — run {rid} ({when})\n{run.get('output_dir','')}")
        parent_item.addChild(leaf)

    # ------------------------------------------------------------------ events
    def _on_item_changed(self, item, column) -> None:
        if self._building or column != 0:
            return
        self._update_preview()

    def _on_current_changed(self, current, _previous) -> None:
        self._current = current
        self._render_info(current)

    # ------------------------------------------------------------------ preview
    def _plans(self) -> list[dict]:
        """Per-scope build plans from the current checked selection."""
        by_scope: dict[str, list[str]] = {}
        for node_id, scope_id in self.checked_ids():
            by_scope.setdefault(scope_id, []).append(node_id)
        plans: list[dict] = []
        for scope_id, _label in self._scope_list():
            nids = by_scope.get(scope_id)
            if not nids:
                continue
            produced = self._produced_by_scope.get(scope_id, set())
            ordered = pg.expand_targets(nids, satisfied=produced)
            plans.append({"scope_id": scope_id, "node_ids": ordered})
        return plans

    def _update_preview(self) -> None:
        checked = self.checked_ids()
        self._btn_build.setEnabled(bool(checked))
        if not checked:
            self._preview.setText("Check products to build.")
            return
        labels = {sid: lbl for sid, lbl in self._scope_list()}
        lines: list[str] = []
        for plan in self._plans():
            sid = plan["scope_id"]
            ordered = plan["node_ids"]
            if ordered:
                run = " → ".join(pg.get_node(o).label for o in ordered)
                lines.append(f"{labels.get(sid, sid)}:  {run}")
            else:
                lines.append(f"{labels.get(sid, sid)}:  already produced — "
                             "nothing to run.")
        self._preview.setText("\n".join(lines))

    # --------------------------------------------------------------- info panel
    def _render_info(self, item) -> None:
        nid = item.data(0, _ROLE_NODE_ID) if item else None
        kind = item.data(0, _ROLE_KIND) if item else None
        scope_id = item.data(0, _ROLE_SCOPE_ID) if item else None
        if not nid or kind in (None, "scope"):
            self._info_title.setText("Select a product")
            self._info.setHtml("" if kind != "scope"
                               else f"<p style='color:#666'>Scope: "
                                    f"<b>{scope_display_label(scope_id)}</b>. Check "
                                    "products under this branch to build them for "
                                    "this scope.</p>")
            self._btn_export.setEnabled(False)
            self._btn_run.setEnabled(False)
            return

        node = pg.get_node(nid)
        run = item.data(0, _ROLE_RUN) if kind == "chunk" else None
        self._info_title.setText(
            f"{node.label}  ·  {scope_display_label(scope_id)}"
            + (f"  ·  chunk {run.get('run_id','')}" if run else ""))

        html = ["<p style='color:#444'>" + node_description(node) + "</p>"]
        if node.params and kind != "chunk":
            html.append("<p style='color:#666'><b>Default run parameters</b></p>"
                        "<table cellspacing='3'>")
            for key, spec in node.params.items():
                html.append(f"<tr><td style='color:#666'>{spec.get('label', key)}"
                            f"</td><td><code>{spec.get('default')}</code></td></tr>")
            html.append("</table>")

        runs = runs_for_node(self._reg, nid, scope_id) if node.task_type else []
        if kind == "chunk" and run is not None:
            when = (run.get("finished_at") or "").replace("T", " ")[:19]
            html.append("<p style='color:#2e7d32'><b>Chunk run</b></p>"
                        "<table cellspacing='3'>"
                        f"<tr><td style='color:#666'>Run</td><td><code>"
                        f"{run.get('run_id','')}</code></td></tr>"
                        f"<tr><td style='color:#666'>Finished</td><td>{when}</td></tr>"
                        f"<tr><td style='color:#666'>Output</td><td><code>"
                        f"{run.get('output_dir','')}</code></td></tr></table>")
        elif node.task_type is None:
            html.append("<p style='color:#888'><i>Aggregate node — no run of its "
                        "own.</i></p>")
        elif runs:
            fan = " (fanned into per-chunk leaves)" if (
                getattr(node, "chunked", False) and len(runs) > 1) else ""
            html.append(f"<p style='color:#2e7d32'><b>{len(runs)} produced "
                        f"run(s) for this scope{fan}</b></p><table cellspacing='3'>"
                        "<tr><td style='color:#666'><b>Run</b></td>"
                        "<td style='color:#666'><b>Finished</b></td>"
                        "<td style='color:#666'><b>Status</b></td></tr>")
            for r in runs[:12]:
                when = (r.get("finished_at") or "").replace("T", " ")[:19]
                html.append(f"<tr><td><code>{r.get('run_id','')}</code></td>"
                            f"<td>{when}</td><td>{r.get('status','')}</td></tr>")
            html.append("</table><p style='color:#888'>Output: "
                        f"<code>{runs[0].get('output_dir','')}</code></p>")
        else:
            html.append("<p style='color:#888'><i>No produced run for this scope "
                        "yet.</i></p>")

        self._info.setHtml("".join(html))
        self._btn_run.setEnabled(node.task_type is not None and nid in self._available)
        self._btn_export.setEnabled(bool(run) or bool(runs))

    # ------------------------------------------------------------------ actions
    def _current_target(self) -> tuple[str | None, str | None, dict | None]:
        item = self._current
        if not item:
            return None, None, None
        nid = item.data(0, _ROLE_NODE_ID)
        scope_id = item.data(0, _ROLE_SCOPE_ID)
        run = item.data(0, _ROLE_RUN)
        return nid, scope_id, run

    def _export_current(self) -> None:
        nid, scope_id, run = self._current_target()
        if not nid:
            return
        self.exportRequested.emit(nid, scope_id or SURVEY_SCOPE)
        # Convenience: reveal the relevant run's output directory.
        out = None
        if run is not None:
            out = run.get("output_dir")
        else:
            runs = runs_for_node(self._reg, nid, scope_id)
            if runs:
                out = runs[0].get("output_dir")
        if out and Path(out).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(out)))

    def _run_current(self) -> None:
        nid, scope_id, _run = self._current_target()
        if nid:
            self.runRequested.emit(nid, scope_id or SURVEY_SCOPE)

    def _emit_build(self) -> None:
        self.buildRequested.emit(self._plans())
